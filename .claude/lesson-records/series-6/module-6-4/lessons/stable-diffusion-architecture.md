# stable-diffusion-architecture -- Lesson Planning Document

**Module:** 6.4 (Stable Diffusion)
**Position:** Lesson 1 of 3 (Lesson 15 in Series 6)
**Type:** CONSOLIDATE (zero new concepts -- assembles every component from Modules 6.1-6.3 into one pipeline)
**Previous lesson:** from-pixels-to-latents (CONSOLIDATE, Module 6.3 Lesson 5)
**Next lesson:** samplers-and-efficiency (STRETCH)

---

## Phase 1: Student State

### Relevant concepts the student has

| Concept | Depth | Source Lesson | Notes |
|---------|-------|--------------|-------|
| Latent diffusion as an architectural pattern (encode with VAE, diffuse in latent space, decode back) | DEVELOPED | from-pixels-to-latents (6.3.5) | Taught just one lesson ago. Student saw the pipeline overview, the side-by-side algorithm comparison (pixel-space vs latent-space, only 3 of 7 steps differ), the "of course" intuitive chain, and the dimension walkthrough (512x512x3 -> 64x64x4 -> diffuse -> decode). This is the conceptual foundation this lesson builds the concrete pipeline on. |
| Frozen-VAE pattern (VAE trained separately, frozen during diffusion training, modular pipeline) | INTRODUCED | from-pixels-to-latents (6.3.5) | Student knows the VAE and diffusion model are independent components. Knows gradients do not flow through the VAE during diffusion training. Has not worked with the frozen VAE in code. |
| DDPM training algorithm (sample image, random timestep, sample noise, closed-form noisy version, predict noise, MSE loss) | APPLIED | build-a-diffusion-model (6.2.5) | Student implemented the full training loop on MNIST. Transferable to latent diffusion unchanged (student already saw this transfer in 6.3.5). |
| DDPM sampling algorithm (loop from z_T to z_0, reverse step formula at each step) | APPLIED | build-a-diffusion-model (6.2.5) | Student implemented the full sampling loop. In latent diffusion, this loop runs on 64x64x4 tensors, with a final VAE decode step. |
| U-Net encoder-decoder architecture with skip connections, dual-path information flow (bottleneck = WHAT, skips = WHERE) | DEVELOPED | unet-architecture (6.3.1) | Taught via dimension walkthrough, Mermaid diagram, pseudocode forward pass. Student can trace data flow through encoder, bottleneck, decoder. Knows channel dimensions at each level. |
| Residual blocks within the U-Net (Conv -> Norm -> Activation -> Conv -> Norm -> Activation + skip) | INTRODUCED | unet-architecture (6.3.1) | Student knows the structure and purpose. Has not implemented it. |
| Sinusoidal timestep embedding (same formula as positional encoding, applied to noise level) | DEVELOPED | conditioning-the-unet (6.3.2) | Direct transfer from positional encoding (4.1.3). Student saw the formula, concrete numerical example, and MLP refinement. |
| Adaptive group normalization (timestep-dependent gamma and beta modulating feature maps at every layer) | DEVELOPED | conditioning-the-unet (6.3.2) | Core timestep conditioning mechanism. Student knows the formula: AdaGN(x, t) = gamma(t) * Normalize(x) + beta(t). Knows per-block linear projection pattern. |
| CLIP dual-encoder architecture (image encoder + text encoder -> shared embedding space via contrastive loss) | INTRODUCED | clip (6.3.3) | Student knows the two-encoder structure, that the loss creates alignment, and that the shared space enables zero-shot transfer. Has not built CLIP. |
| CLIP text encoder produces a SEQUENCE of token embeddings (not just the single summary vector) | INTRODUCED | text-conditioning-and-guidance (6.3.4) | Clarified in the recap of Lesson 13: CLIP outputs per-token embeddings that become K/V for cross-attention, not just the CLS pooled vector used for similarity. |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings, same QKV formula as self-attention) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Taught as a one-line change from self-attention. Student can explain the formula, the rectangular attention matrix (256 spatial x T text tokens), per-spatial-location attention. Has implemented it in a notebook exercise. |
| Per-spatial-location cross-attention (each spatial location attends independently to text tokens, creating spatially-varying conditioning) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Core insight: different locations get different signals from the same prompt. Student saw the concrete attention weight table for "a cat sitting in a sunset." |
| Classifier-free guidance (training with random text dropout, two forward passes at inference, amplify text direction) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Student knows the formula: epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond). Knows w=7.5 is typical. Has implemented it in a notebook exercise. |
| Guidance scale tradeoff (w=1 no amplification, w=7.5 typical, w=20 oversaturated) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Three GradientCards showing the spectrum. Student knows higher is not always better. |
| Block ordering at attention resolutions (residual block -> self-attention -> cross-attention, at 16x16 and 32x32) | INTRODUCED | text-conditioning-and-guidance (6.3.4) | Student knows the three-block pattern and that attention only runs at middle resolutions due to O(n^2) cost. |
| Computational cost reduction via latent space (512x512x3 = 786,432 vs 64x64x4 = 16,384, 48x compression) | DEVELOPED | from-pixels-to-latents (6.3.5) | Student computed the compression ratio. Connected to capstone timing experience. |
| Spatially-varying vs global conditioning (timestep via adaptive norm is global, text via cross-attention is spatially varying) | INTRODUCED | text-conditioning-and-guidance (6.3.4) | ComparisonRow contrasting the two conditioning mechanisms. Student knows they coexist. |
| CFG doubles inference compute (two forward passes per step, batchable) | INTRODUCED | text-conditioning-and-guidance (6.3.4) | Brief note: 50 steps = 100 forward passes. Student knows the cost but has not felt it. |

### Mental models and analogies already established

- **"The VAE is a translator between two languages"** -- pixel language (human) and latent language (diffusion model). Encoder translates pixel -> latent, decoder translates latent -> pixel. (from-pixels-to-latents, 6.3.5)
- **"Same orchestra, smaller hall"** -- same diffusion algorithm, same conditioning, just in a compressed latent space. (from-pixels-to-latents, 6.3.5)
- **"The bottleneck decides WHAT, the skip connections decide WHERE"** -- dual-path U-Net mental model. (unet-architecture, 6.3.1)
- **"The timestep tells the network WHEN. The text tells it WHAT. CFG turns up the volume on the WHAT."** -- complete conditioning mental model. (text-conditioning-and-guidance, 6.3.4)
- **"Same formula, different source for K and V"** -- cross-attention as a minimal delta from self-attention. (text-conditioning-and-guidance, 6.3.4)
- **"Two encoders, one shared space -- the loss creates the alignment, not the architecture"** -- CLIP mental model. (clip, 6.3.3)
- **"Contrast slider"** -- CFG guidance scale as contrast adjustment. (text-conditioning-and-guidance, 6.3.4)
- **"Same building blocks, different question"** -- recurring meta-pattern throughout Series 6.
- **"The VAE built the roads. Diffusion walks them."** -- why latent diffusion works. (from-pixels-to-latents, 6.3.5)

### What was explicitly NOT covered in prior lessons (relevant here)

- **The complete end-to-end data flow as one traced pipeline** -- The student has seen each component individually and the conceptual latent diffusion pipeline (6.3.5), but has never traced a single generation from text prompt through tokenizer, CLIP, denoising loop with CFG, to VAE decode, with real tensor shapes at every handoff point.
- **Training pipeline vs inference pipeline** -- The student has seen the training algorithm (6.3.5) and the sampling algorithm separately, but has not explicitly compared "what happens during training" vs "what happens during inference" as two distinct modes of the same system.
- **Component parameter counts and scale** -- The student has no sense of how large each component is (CLIP ~123M, U-Net ~860M, VAE ~84M). This grounds the "modular pipeline" understanding in practical reality.
- **The tokenizer step** -- Text prompt goes to a tokenizer before CLIP. The student knows about tokenization from Series 4.1 (BPE for GPT), but has not seen CLIP's tokenizer specifically.
- **Negative prompts** -- How negative prompts fit into the CFG framework (they replace the empty-string unconditional prediction). The student has the CFG formula but has not seen this application.
- **The concrete denoising loop with all conditioning happening simultaneously** -- Timestep conditioning via adaptive group norm AND cross-attention with text AND CFG, all happening at each step. The student learned each mechanism separately; this lesson shows them firing together.

### Readiness assessment

The student is maximally prepared for this lesson. Every single concept required has been taught at DEVELOPED or APPLIED depth across Modules 6.1-6.3. There are zero gaps and zero fading concerns -- the previous lesson (from-pixels-to-latents) was the immediate predecessor and it already reactivated fading VAE concepts.

This lesson's job is purely assembly: take the pieces the student knows individually and show how they fit together as one system. The cognitive load is CONSOLIDATE -- zero new algorithms, zero new math, zero new architectural concepts. The only novelty is seeing the complete data flow as one coherent trace, with real tensor shapes at every handoff.

The emotional setup is strong. The student has spent 14 lessons building each component from scratch. They understand WHY each piece exists. This lesson delivers the "I can see the whole thing now" payoff.

---

## Phase 2: Analysis

### Target concept

This lesson teaches the student to trace the complete Stable Diffusion pipeline from text prompt to generated image, with real tensor shapes at every stage, understanding how CLIP, the conditioned U-Net, and the VAE work together as one system.

### Prerequisites table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Latent diffusion pipeline (encode, diffuse in latent space, decode) | DEVELOPED | DEVELOPED | from-pixels-to-latents (6.3.5) | OK | The student saw the pipeline overview one lesson ago. This lesson adds concrete tensor shapes and traces the full data flow. |
| Frozen-VAE pattern | INTRODUCED | INTRODUCED | from-pixels-to-latents (6.3.5) | OK | Student needs to understand that the VAE is frozen and modular. Already taught at the required depth. |
| DDPM sampling algorithm | APPLIED | APPLIED | build-a-diffusion-model (6.2.5) | OK | Student implemented this. The denoising loop is the core of the inference pipeline. |
| U-Net architecture with skip connections | DEVELOPED | DEVELOPED | unet-architecture (6.3.1) | OK | Student can trace encoder-decoder data flow. This lesson places the U-Net within the larger pipeline. |
| Sinusoidal timestep embedding + adaptive group normalization | DEVELOPED | DEVELOPED | conditioning-the-unet (6.3.2) | OK | Student knows how timestep conditioning works. This lesson shows it happening inside the denoising loop. |
| CLIP text encoder (produces sequence of token embeddings) | INTRODUCED | INTRODUCED | clip (6.3.3), text-conditioning-and-guidance (6.3.4) | OK | Student knows CLIP's architecture and that it produces per-token embeddings. This lesson traces the specific tensor shapes (77x768). |
| Cross-attention (Q from spatial features, K/V from text embeddings) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Student can explain the formula and the per-spatial-location mechanism. This lesson shows where it sits in the denoising loop. |
| Classifier-free guidance (two forward passes, amplify text direction) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Student knows the formula and typical guidance scales. This lesson shows the two forward passes concretely in the loop. |
| Block ordering at attention resolutions | INTRODUCED | INTRODUCED | text-conditioning-and-guidance (6.3.4) | OK | Student knows residual -> self-attention -> cross-attention at middle resolutions. Needed for the internal U-Net data flow trace. |
| Computational cost reduction (48x compression) | DEVELOPED | DEVELOPED | from-pixels-to-latents (6.3.5) | OK | Student computed this. Grounds the tensor shape walkthrough in practical context. |
| Tokenization (text -> token IDs) | DEVELOPED | DEVELOPED | tokenization (4.1.2) | OK | Student knows BPE tokenization. CLIP's tokenizer uses the same principle (slightly different vocabulary). No gap -- just a new context for a known concept. |

### Gap resolution

No gaps. Every prerequisite is at or above the required depth. No fading concerns -- the most recent prerequisite is one lesson old, and the oldest relevant concepts (VAE) were reactivated in the previous lesson.

### Misconceptions table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Stable Diffusion is one big model trained end-to-end" | In most deep learning the student has seen (CNNs, transformers, GPT), the entire model is trained end-to-end with one loss function. The student may default to this assumption despite learning about frozen components individually. Seeing all components running together may trigger the "one big model" framing. | Show the parameter counts of each component separately (CLIP ~123M, U-Net ~860M, VAE ~84M, total ~1.07B) and note that each was trained with a different loss on different data. CLIP was trained on 400M text-image pairs with contrastive loss. The VAE was trained on image reconstruction. The U-Net was trained on latent diffusion. They were never trained together. Swapping the text encoder (e.g., using OpenCLIP instead of OpenAI CLIP) is possible precisely because they are independent. | Section 5 (after the full pipeline trace), as an explicit "One Pipeline, Three Models" breakdown. Address after the student has seen the full data flow so they have the context to appreciate the modularity. |
| "The U-Net sees the text prompt directly" | The student knows cross-attention exists but may mentally shortcut to "the U-Net processes text." When tracing the pipeline, they may not realize that the U-Net never sees raw text or even token IDs -- it only sees 768-dimensional embedding vectors produced by a completely separate model (CLIP). | Trace the data flow: prompt "a cat on a beach" -> tokenizer -> [49406, 320, 2368, ...] (integer IDs) -> CLIP text encoder -> 77x768 float tensor. The U-Net receives ONLY the 77x768 tensor. It has no access to the original text, no tokenizer, no vocabulary. If you changed the text encoder to produce the same 77x768 tensor from a different language, the U-Net would generate the same image. The U-Net is a tensor-processing machine that knows nothing about language. | Section 4 (pipeline trace), at the CLIP -> U-Net handoff. Address in-line as the data flows through. |
| "CFG happens after the denoising loop (as a post-processing step)" | The student learned CFG as a technique with a formula. They may think of it as "amplify the text signal" without realizing it happens at EVERY step of the denoising loop, requiring two full U-Net forward passes per step. The mental model might be: run the loop, then adjust. | Show the denoising loop pseudocode with CFG inside: `for t in timesteps: eps_uncond = unet(z_t, t, text_uncond); eps_cond = unet(z_t, t, text_cond); eps = eps_uncond + w * (eps_cond - eps_uncond); z_{t-1} = scheduler.step(eps, t, z_t)`. The two forward passes happen at every step. With 50 steps, that is 100 U-Net forward passes, not 50. CFG is not post-processing; it is woven into every step of the generation. | Section 6 (the denoising loop walkthrough), showing CFG as part of the per-step procedure. |
| "The VAE encoder runs during inference (generation)" | The student knows the training pipeline uses `z_0 = VAE.encode(x_0)`. They may assume inference also starts by encoding something. But during text-to-image generation, there is no input image to encode. The starting point is pure random noise in latent space: `z_T ~ N(0, I)`. The VAE encoder is not used at all during text-to-image inference -- only the decoder. | Compare training vs inference explicitly. Training: real image -> VAE encode -> z_0 -> add noise -> z_t -> U-Net predicts noise. Inference: z_T sampled from N(0,I) -> denoising loop -> z_0 -> VAE decode -> generated image. The encoder appears in training but NOT in inference. (Note: img2img DOES use the encoder, but that is Module 6.5 scope.) | Section 7 (training vs inference comparison), as an explicit callout in a WarningBlock. |
| "More parameters always means a more important component" | The U-Net has ~860M parameters while CLIP has ~123M and the VAE has ~84M. The student might conclude the U-Net is "the important part" and the others are minor. But CLIP determines what the image depicts (the semantic content), and the VAE determines the image quality (reconstruction fidelity). The U-Net does the generation, but it is useless without the other two. | Imagine removing each component: without CLIP, the U-Net generates images but you cannot control what they are (unconditional generation only). Without the VAE, you would need pixel-space diffusion (1000x slower). Without the U-Net, you have a text encoder and a decoder but nothing to generate latents. Each component is essential. The parameter count reflects the difficulty of the task each component performs (denoising at multiple scales is hard), not its importance to the pipeline. | Section 5 (component breakdown), as a TipBlock after showing parameter counts. |

### Examples planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Full pipeline trace: "a cat sitting on a beach at sunset" -> tokenize -> CLIP encode -> denoising loop with CFG -> VAE decode -> pixel image, with tensor shapes at every stage | Positive (concrete, end-to-end) | The centerpiece of the lesson. Traces one complete generation with specific tensor shapes: prompt -> 77 token IDs -> 77x768 text embeddings -> z_T (4x64x64) -> 50 steps of [two U-Net passes, CFG combine, scheduler step] -> z_0 (4x64x64) -> VAE decode -> 3x512x512 pixel image. Every handoff between components is annotated with shape and dtype. | This is the assembly lesson. The student has never seen the complete trace. The prompt includes multiple semantic elements ("cat," "beach," "sunset") so the student can connect to the per-spatial-location cross-attention they learned in 6.3.4 (different regions attend to different words). The prompt is familiar from the attention weight table example in that lesson, providing continuity. |
| Training vs inference pipeline comparison | Positive (structural) | Show that the same components are arranged differently for training vs inference. Training: real image -> VAE encode -> add noise -> U-Net predicts noise -> MSE loss. Inference: random z_T -> denoising loop -> z_0 -> VAE decode -> image. The U-Net is the same, the VAE is the same, but the data flow direction and the role of each component differ. | Addresses misconception #4 (VAE encoder during inference) and deepens understanding of the system. The student implemented the training loop (6.2.5) and the sampling loop (6.2.5) separately. This comparison shows them as two modes of the same system, which is a new perspective. |
| One step of the denoising loop, expanded: z_t -> duplicate for CFG -> two U-Net forward passes (one with text, one without) -> CFG combine -> scheduler step -> z_{t-1}, with tensor shapes at each sub-step | Positive (detailed) | Zooms into what happens at ONE step of the loop. Shows CFG as part of the step, not as a post-processing step. Shows the two U-Net passes, the tensor shapes flowing through cross-attention and adaptive norm, and the scheduler combining everything into z_{t-1}. | Addresses misconception #3 (CFG as post-processing). The student knows each mechanism individually but has not seen them firing simultaneously within one denoising step. This example makes the per-step procedure concrete and shows the computational cost (two full U-Net passes). |
| Negative: removing the text encoder (unconditional generation) | Negative | What happens if you run the pipeline without CLIP? The U-Net still denoises, but every denoising step uses epsilon_uncond only (no text signal to amplify with CFG). The result: coherent images, but you cannot control what they depict. This is pixel-space unconditional diffusion running in latent space -- exactly what the student built in Module 6.2, just faster. | Demonstrates that CLIP is what gives Stable Diffusion its controllability. Without it, you have latent diffusion (fast generation) but not Stable Diffusion (text-guided generation). This grounds the modularity claim: each component contributes something specific and removable. |
| Negative: feeding raw text string to the U-Net instead of CLIP embeddings | Negative | What if you tried to give the U-Net the text "a cat" directly? The U-Net expects a 77x768 float tensor as the cross-attention K/V input. A text string is not a tensor. Even token IDs (integers) would not work -- the U-Net needs dense 768-dimensional embedding vectors that encode visual-semantic meaning. CLIP is the translator from human language to the geometric space the U-Net can attend to. | Reinforces misconception #2 (U-Net sees text directly). Makes the role of CLIP concrete: it is not optional preprocessing, it is the essential translation from language to the geometric representation space that cross-attention operates in. Connects to the "two languages" analogy from 6.3.3 (CLIP creates the shared space) and the "translator" analogy from 6.3.5 (the VAE translates between pixel and latent languages). |

---

## Phase 3: Design

### Narrative arc

The student has spent 14 lessons across three modules learning every component of Stable Diffusion from scratch. They built a VAE and explored its latent space. They derived the diffusion algorithm and implemented a pixel-space model. They studied the U-Net architecture, timestep conditioning, CLIP, cross-attention, classifier-free guidance, and latent diffusion. Each lesson focused on one piece at a time, building deep understanding of WHY each component exists and HOW it works. But the student has never seen all the pieces running together. It is like learning every instrument in an orchestra individually -- the violin, the oboe, the timpani -- and then finally hearing the symphony. This lesson is the symphony. The student traces one complete generation from a text prompt ("a cat sitting on a beach at sunset") through every component, watching real tensor shapes transform at each stage: text -> 77 token IDs -> 77x768 embeddings -> a denoising loop where the U-Net makes 100 forward passes (50 steps x 2 for CFG), with cross-attention pulling from text and adaptive group norm responding to the timestep, all operating on a tiny 4x64x64 latent tensor -> VAE decode -> a 3x512x512 pixel image. Nothing in this pipeline is new. Every piece is something the student built or deeply studied. The revelation is not any individual component but the elegance of their integration: three independently trained models, connected by tensor handoffs, producing images from text.

### Modalities planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual (diagram) | Full pipeline diagram as a horizontal data flow: [Text Prompt] -> [Tokenizer] -> [CLIP Text Encoder] -> [Denoising Loop: z_T -> {U-Net x2 for CFG, with timestep embedding + cross-attention annotated} -> ... -> z_0] -> [VAE Decoder] -> [Image]. Tensor shapes annotated at every arrow. Color-coded by component (CLIP = one color, U-Net/loop = another, VAE = another). This is the centerpiece visual of the lesson. | Assembly lessons NEED a visual that shows the whole system at a glance. The student has seen each component's internal diagram (U-Net Mermaid, cross-attention SVG, pipeline comparison). Now they need the system-level view where each component is a box and the arrows between boxes carry specific tensor shapes. This is the "zoom out" visual after 14 lessons of "zoom in." |
| Symbolic (code/pseudocode) | Complete inference pseudocode, ~20 lines: tokenize prompt, encode with CLIP, sample z_T, loop over timesteps (embed t, U-Net forward x2, CFG combine, scheduler step), decode with VAE. Each line annotated with tensor shape in comments. This is executable-level pseudocode, not conceptual. | The symbolic representation makes the assembly concrete and precise. The student has seen pseudocode for individual components (U-Net forward pass in 6.3.1, training algorithm in 6.3.5). Seeing the complete inference procedure as pseudocode lets them verify their understanding: "yes, I know what every line does." The shape annotations verify dimensional consistency across components. |
| Concrete example | Traced generation of "a cat sitting on a beach at sunset" with specific tensor shapes at every stage. Including: token IDs from the tokenizer (showing padding to 77), the 77x768 text embedding tensor shape, the initial z_T shape (4x64x64), intermediate shapes within the U-Net (64x64x320, 32x32x640, 16x16x1280, 8x8x1280), the cross-attention matrix shape at 16x16 resolution (256x77), the CFG combination producing one noise prediction, the scheduler step producing z_{t-1}, and the final VAE decode from 4x64x64 to 3x512x512. | Real tensor shapes are the connective tissue of this lesson. When the student sees "77x768," they think "77 tokens, 768 CLIP embedding dimension." When they see "256x77," they think "16x16=256 spatial locations, each attending to 77 text tokens." Every shape MEANS something because the student built the concepts behind those dimensions. Abstract pipeline diagrams are not enough for an assembly lesson -- the student needs to see the numbers to verify that the pieces physically fit together. |
| Verbal/Analogy | (1) "Assembly line" -- each component does one job and passes the result to the next. CLIP translates text to vectors. The U-Net denoises latents using those vectors for guidance. The VAE translates the final latent back to pixels. No component knows what the others do internally. (2) Extension of the "translator" analogy from 6.3.5: CLIP is a translator from human language to geometric-meaning space. The VAE is a translator from pixel space to latent space (and back). The U-Net speaks only "latent language" and "geometric-meaning language." Three translators, one pipeline. | The assembly line analogy reinforces the modularity and the tensor-handoff model. The triple-translator extension connects to two existing analogies (CLIP's shared space from 6.3.3 and VAE as translator from 6.3.5) and unifies them: the pipeline is three translators working in sequence, each bridging a different representational gap. |
| Intuitive | The "everything you know, nothing you don't" moment. After the full pipeline trace, an explicit enumeration: "Every component in this pipeline is something you learned. The CLIP encoder is from Lesson 12. The cross-attention is from Lesson 13. The CFG formula is from Lesson 13. The adaptive group norm is from Lesson 11. The U-Net architecture is from Lesson 10. The VAE is from Lessons 2-4. The latent diffusion idea is from Lesson 14. There is nothing in this pipeline you have not already studied." | The intuitive modality here is not an analogy but a feeling: the student realizes they genuinely understand every piece of a state-of-the-art AI system. This is the emotional payoff of 14 lessons of deliberate practice. Naming each component and its source lesson makes the accumulated knowledge tangible and creates the "I actually understand this" feeling that motivates continued learning. |

### Cognitive load assessment

- **New concepts in this lesson:** Zero. Every component (CLIP, U-Net, cross-attention, CFG, adaptive group norm, VAE, latent diffusion) was taught in Modules 6.1-6.3. The only novelty is seeing them connected as one pipeline, with real tensor shapes at the handoffs. Two new FACTS are introduced:
  1. Component parameter counts (CLIP ~123M, U-Net ~860M, VAE ~84M) -- contextual information, not a concept.
  2. Training vs inference pipeline comparison -- a structural observation about concepts the student already knows.

  Neither requires new mathematical or architectural understanding.

- **Previous lesson load:** CONSOLIDATE (from-pixels-to-latents)
- **This lesson's load:** CONSOLIDATE -- appropriate. Two CONSOLIDATE lessons in sequence is unusual but justified: Module 6.3's final lesson synthesized VAE + diffusion conceptually, and this lesson synthesizes the full pipeline concretely. The previous lesson answered "what is latent diffusion?"; this lesson answers "what does the full pipeline look like in practice?" They are complementary, not redundant. The cognitive work is connecting and tracing, not learning.

### Connections to prior concepts

| Prior Concept | How It Connects | Risk of Misleading? |
|---------------|----------------|--------------------|
| Latent diffusion pipeline (6.3.5) | The foundation. This lesson takes the conceptual pipeline from 6.3.5 and makes it concrete with real tensor shapes and complete pseudocode. The student already knows the structure; this lesson fills in the dimensions. | No risk. Direct extension of the immediate prior lesson. |
| Cross-attention data flow (6.3.4) | The student knows Q comes from spatial features, K/V from text. This lesson shows the specific shapes: at 16x16 resolution, Q is 256xd_k, K is 77xd_k, attention matrix is 256x77. The "a cat sitting on a beach at sunset" prompt is a callback to the attention weight table from 6.3.4. | No risk. Same mechanism, now placed within the pipeline trace. The prompt continuity helps the student recognize the connection. |
| CFG formula (6.3.4) | The student knows epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond). This lesson shows the formula executing INSIDE the denoising loop, at every step, requiring two U-Net passes. | Low risk. The student may have conceptualized CFG abstractly. Seeing it in the loop makes it concrete. |
| Tokenization (4.1.2) | Text prompt goes through a tokenizer before CLIP. The student learned BPE tokenization for GPT. CLIP uses a similar tokenizer with a different vocabulary and adds special tokens (SOT, EOT, padding to 77). | Low risk. The tokenizer is a familiar concept in a new context. The padding to 77 is a new detail but not a new concept. |
| U-Net internal dimensions (6.3.1) | The student traced 64x64x64 -> 32x32x128 -> 16x16x256 -> 8x8x512 in a toy U-Net. Stable Diffusion's U-Net has different channel counts (320 -> 640 -> 1280 -> 1280) but the same spatial downsampling pattern. | Low risk. The pattern is the same; the numbers are bigger. Explicitly note that the toy dimensions from 6.3.1 were illustrative and SD's are larger but follow the same pattern. |
| "Same building blocks, different question" (recurring) | This lesson extends the pattern one final time: the same building blocks (U-Net, attention, VAE, noise schedule), now answering "can you generate images from text descriptions?" | No risk. Natural extension of a familiar meta-pattern. |

### Scope boundaries

**This lesson IS about:**
- The complete Stable Diffusion inference pipeline, traced from text prompt to pixel image with real tensor shapes
- The denoising loop with all conditioning mechanisms firing simultaneously (timestep via adaptive group norm + text via cross-attention + CFG combining two forward passes)
- Training vs inference comparison (when the VAE encoder is used vs when it is not)
- Component parameter counts and scale (grounding the modularity in practical reality)
- The modularity principle: three independently trained models connected by tensor handoffs
- Negative prompts as a direct application of the CFG formula (replacing the empty unconditional text)

**This lesson is NOT about:**
- Implementing the pipeline from scratch (the notebook uses pre-trained components from diffusers)
- Samplers beyond DDPM (DDIM, Euler, DPM-Solver are Module 6.4 Lesson 2)
- Any new mathematical formulas or derivations (this is CONSOLIDATE)
- SD v1 vs v2 vs XL architectural differences
- LoRA, fine-tuning, or customization (Module 6.5)
- Img2img or inpainting (Module 6.5)
- Training Stable Diffusion from scratch (out of scope for the course)
- Prompt engineering techniques
- The internal architecture of CLIP's text encoder (transformer -- the student knows this from 6.3.3)
- The internal architecture of SD's VAE (KL-regularized autoencoder with ResNet blocks -- details not needed)

**Depth targets:**
- Full SD pipeline data flow (text -> CLIP -> denoising loop -> VAE decode): DEVELOPED (student can trace every tensor shape, explain every handoff, identify each component's role)
- Training vs inference pipeline: INTRODUCED (student knows the difference, can explain when the encoder is used)
- Component modularity (independently trained, swappable): INTRODUCED (student knows the principle and the parameter counts)
- Negative prompts as CFG application: INTRODUCED (student knows the mechanism, has not practiced it)

### Lesson outline

**1. Context + Constraints**
- This is the first lesson of Module 6.4. The student has learned every component of Stable Diffusion individually across Modules 6.1-6.3. This lesson assembles them into one pipeline.
- This is a CONSOLIDATE lesson: no new concepts, no new math. Every component in the pipeline is something the student already knows. The only novelty is seeing them work together.
- Scope: full pipeline data flow with real tensor shapes. NOT samplers (Lesson 16), NOT implementation from scratch, NOT fine-tuning (Module 6.5).
- By the end: the student can trace a complete generation from text prompt to pixel image, naming every component and the tensor shape at every handoff.

**2. Hook: "You know every instrument. Time to hear the symphony."**
- Type: Before/after, accumulated knowledge payoff.
- Brief enumeration: "Over the last 14 lessons, you learned: how to compress images (VAE), how to denoise (diffusion), the architecture that denoises (U-Net), how it knows the noise level (timestep conditioning), how to connect language to vision (CLIP), how to inject text (cross-attention), how to make text matter (CFG), and how to make it fast (latent diffusion)."
- "You have never seen all of them working together. This lesson assembles the full pipeline and traces one generation from start to finish."
- The prompt for the traced example: "a cat sitting on a beach at sunset" -- deliberately chosen as a callback to the cross-attention lesson (6.3.4) where the student saw per-spatial-location attention weights for this prompt.

**3. Explain: The Full Pipeline Trace (DEVELOPED)**

**3a. Text to embeddings: the CLIP stage**
- Input: text prompt "a cat sitting on a beach at sunset"
- Tokenizer: splits text into subword tokens, adds SOT/EOT special tokens, pads to 77 tokens. Output: tensor of 77 integer token IDs. Brief callback to BPE tokenization from 4.1.2.
- CLIP text encoder: the transformer processes the 77 token IDs and outputs 77x768-dimensional embedding vectors. These are NOT the simple one-hot or lookup embeddings from 4.1.1 -- they are contextual embeddings where each token's representation includes context from all other tokens (via self-attention in CLIP's transformer).
- Shape checkpoint: prompt -> [77] int token IDs -> [77, 768] float embeddings. This 77x768 tensor is the ONLY thing the rest of the pipeline sees. The text is gone.
- WarningBlock addressing misconception #2: "The U-Net will never see the text 'a cat sitting on a beach at sunset.' It will only see a 77x768 tensor of floating-point numbers. CLIP is the translator from human language to the geometric representation space the U-Net operates in."

**3b. The starting point: random noise in latent space**
- For text-to-image generation: z_T ~ N(0, I) with shape [4, 64, 64]. This is the starting point. There is no input image during inference.
- The 4 channels correspond to the VAE's latent dimensions. The 64x64 spatial dimensions correspond to 512x512 pixels (8x downsampling factor in each spatial direction).
- The seed (random number generator state) determines z_T and therefore the generated image. Same prompt + same seed = same image. Different seed = different image from the same "concept."

**3c. The denoising loop: where everything happens**
- High-level structure: loop over timesteps [T, T-1, ..., 1], at each step: (1) embed timestep, (2) two U-Net forward passes (unconditional + conditional), (3) CFG combine, (4) scheduler step to get z_{t-1}.
- One step expanded:
  1. **Timestep embedding:** t -> sinusoidal encoding -> MLP -> t_emb (512-dim vector). This is the "conductor's score" from 6.3.2.
  2. **Unconditional U-Net pass:** unet(z_t, t_emb, text_uncond) -> epsilon_uncond. The text_uncond is CLIP encoding of an empty string (or negative prompt). Inside the U-Net: adaptive group norm uses t_emb at every residual block, cross-attention uses text_uncond at 16x16 and 32x32 resolutions.
  3. **Conditional U-Net pass:** unet(z_t, t_emb, text_cond) -> epsilon_cond. Same U-Net, same weights, same z_t, same t_emb. The only difference: cross-attention uses the real text embeddings. The "contrast slider" from 6.3.4.
  4. **CFG combine:** epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond). Typical w = 7.5.
  5. **Scheduler step:** uses epsilon_cfg and the noise schedule to compute z_{t-1}. This is the reverse step formula from Module 6.2.
- After all steps: z_0 has shape [4, 64, 64]. It is the denoised latent representation.
- Concrete shapes inside the U-Net at one step: z_t [4, 64, 64] -> encoder path (64x64x320, 32x32x640, 16x16x1280, 8x8x1280) -> bottleneck (8x8x1280) -> decoder path back up with skip connections. Cross-attention at 16x16 (256 spatial locations x 77 text tokens) and 32x32 (1024 x 77). Self-attention at same resolutions.

**3d. Latent to pixels: the VAE decode step**
- z_0 [4, 64, 64] -> VAE decoder -> image [3, 512, 512].
- This is the "translator from latent language to pixel language" from 6.3.5.
- The decoder runs ONCE, after the entire denoising loop. It is fast compared to the 50-step (100 forward pass) denoising loop.

**3e. Complete pipeline diagram**
- Visual: full horizontal data flow with tensor shapes at every arrow.
- Color-coded by component origin (CLIP, U-Net, VAE).
- The denoising loop section is visually prominent (it is the computational bottleneck -- 100 U-Net forward passes).

**4. Check (predict-and-verify)**
- "How many U-Net forward passes happen in a 50-step generation with CFG?" (Answer: 100 -- two per step, one unconditional + one conditional.)
- "At what point in the pipeline does text actually influence the image?" (Answer: Inside every denoising step, via cross-attention. CLIP encodes the text ONCE at the start; cross-attention applies the text embeddings at EVERY step. The text influence accumulates across all 50 steps, not just once.)
- "If you changed the seed but kept the same prompt and settings, what would change and what would stay the same in the pipeline?" (Answer: z_T would be different random noise. Everything else -- CLIP encoding, guidance scale, number of steps -- stays the same. The result is a different image of the same concept.)

**5. Elaborate: One Pipeline, Three Models**

**5a. Component breakdown with parameter counts:**
- CLIP text encoder: ~123M parameters. Trained by OpenAI on 400M text-image pairs with contrastive loss. Frozen in SD.
- VAE: ~84M parameters. Trained on image reconstruction with perceptual + adversarial loss. Frozen in SD.
- U-Net: ~860M parameters. Trained on the diffusion objective (predict noise in latent space). The only component trained for the diffusion task.
- Total: ~1.07B parameters. But they were never trained together.
- TipBlock addressing misconception #5: "The U-Net has ~860M parameters but it is useless without CLIP (what to generate) and the VAE (how to translate to pixels). Each component is essential. Parameter count reflects task difficulty, not importance."

**5b. Modularity in practice:**
- You can swap the text encoder (OpenAI CLIP vs OpenCLIP vs CLIP trained on different data).
- You can swap the VAE (higher-quality VAE = sharper decoding).
- You can swap the scheduler/sampler (DDPM, DDIM, Euler -- Lesson 16).
- Each swap is possible BECAUSE the components communicate through standardized tensor shapes, not shared weights.

**5c. Negative prompts:**
- "In the CFG formula, epsilon_uncond is the U-Net's prediction when conditioned on 'no text.' By default, this means an empty string encoded by CLIP."
- "A negative prompt replaces the empty string. Instead of epsilon_uncond = unet(z_t, t, CLIP('')), you use epsilon_uncond = unet(z_t, t, CLIP('blurry, low quality'))."
- "The CFG formula is unchanged: epsilon_cfg = epsilon_neg + w * (epsilon_cond - epsilon_neg). The result: the model steers TOWARD the positive prompt and AWAY FROM the negative prompt."
- "Negative prompts are not a new mechanism. They are a direct application of the CFG formula you already know."

**6. Explain: Training vs Inference (INTRODUCED)**

**6a. ComparisonRow: training pipeline vs inference pipeline**
- Training: dataset image -> VAE encode -> z_0 -> sample t, epsilon -> z_t = forward process -> U-Net predicts noise -> MSE loss -> update U-Net weights.
- Inference: sample z_T ~ N(0,I) -> denoising loop (50 steps with CFG) -> z_0 -> VAE decode -> generated image.
- Key differences highlighted:
  1. Training uses the VAE encoder. Inference does not (for text-to-image).
  2. Training processes one random timestep per image. Inference loops through all timesteps.
  3. Training updates U-Net weights. Inference does not update anything.
  4. Training needs real images. Inference starts from noise.
- WarningBlock addressing misconception #4: "During text-to-image generation, the VAE encoder is never used. There is no input image to encode. The starting point is random noise in latent space. Only the VAE decoder runs -- once, at the very end."

**7. Check (transfer)**
- "Your colleague wants to add a new conditioning signal to Stable Diffusion -- for example, a depth map that controls the 3D structure of the generated image. Based on your understanding of the pipeline, where would this signal enter? What form would it need to take?" (Answer: It would need to enter the U-Net, probably via cross-attention or by concatenation with the latent input. It would need to be encoded into a tensor representation the U-Net can process -- similar to how text becomes a 77x768 tensor via CLIP. The rest of the pipeline (VAE, scheduler) would be unchanged. This is essentially what ControlNet does -- Module 7.)
- "If you doubled the VAE's compression factor (32x32x4 instead of 64x64x4 latents), what would change in the pipeline and what would stay the same?" (Answer: The U-Net would process 32x32x4 tensors -- 4x fewer spatial positions, much faster. Cross-attention at the lowest resolution would have fewer spatial tokens. The denoising algorithm, CFG, timestep conditioning would all stay the same. The risk: the VAE might lose too much detail at higher compression, producing lower-quality decoded images. The tradeoff is compression vs reconstruction quality.)

**8. Practice -- Notebook exercises (Colab)**
- **Exercise design rationale:** This is a CONSOLIDATE lesson. The exercises should verify the student can trace the pipeline and identify components, not implement anything from scratch. The exercises use pre-trained diffusers components and focus on inspecting tensor shapes, understanding component roles, and verifying that the student can predict what each stage produces.
- **Exercise sequence (mostly independent):**
  1. **(Guided) Load and inspect components:** Load the three SD components separately (CLIP text encoder, VAE, U-Net) from diffusers. Print each component's parameter count. Verify the shapes match the lesson's predictions. Predict-before-run: "How many parameters does the U-Net have?"
  2. **(Guided) Trace the CLIP stage:** Tokenize a prompt. Inspect the token IDs (including padding to 77). Run the CLIP text encoder. Inspect the output shape (batch, 77, 768). Try a different prompt and verify the output shape stays the same (77x768 regardless of prompt length). Why?
  3. **(Supported) Trace one denoising step:** Sample z_T. Embed a timestep. Run one U-Net forward pass with the text embeddings. Inspect the output shape (should match z_T). Run a second pass with empty-string embeddings. Apply the CFG formula manually. Apply one scheduler step. Compare z_T and z_{T-1} -- the latent should be slightly less noisy.
  4. **(Independent) Full pipeline trace:** Execute the complete pipeline manually: tokenize, CLIP encode, sample z_T, run the full denoising loop with CFG, VAE decode, display the image. Verify every tensor shape matches the lesson's predictions. Reflection: "Which component took the longest to run? Why?"
- **Solutions should emphasize:** Tensor shape verification at every stage, the modularity of the components (each loaded separately, communicating via tensors), the 100 U-Net forward passes in a 50-step generation, and the dominance of U-Net compute time.

**9. Summarize**
- "Stable Diffusion is three independently trained models connected by tensor handoffs:"
  1. **CLIP** translates text to a 77x768 embedding tensor.
  2. **U-Net** denoises a 4x64x64 latent tensor over 50 steps, using cross-attention (text), adaptive group norm (timestep), and CFG (two passes per step).
  3. **VAE decoder** translates the final 4x64x64 latent to a 3x512x512 pixel image.
- "Nothing in this pipeline is new to you. Every piece is something you built or deeply studied."
- Mental model echo: "CLIP translates language. The U-Net generates in latent space. The VAE translates back to pixels. Three translators, one pipeline."

**10. Next step**
- "The pipeline works, but DDPM's 1000 steps (or even 50) is still slow. Lesson 16 explains why advanced samplers let you generate in 20 steps without retraining the model."
- Preview: "The U-Net predicts noise at each step. The SAMPLER decides how to USE that prediction to take a step. Swapping the sampler is like changing the route, not the vehicle."

---

## Widget Assessment

**Widget needed:** No dedicated interactive widget.

**Rationale:** This is a CONSOLIDATE assembly lesson. The core insight is structural (how components connect) and dimensional (tensor shapes at handoffs), which is best conveyed by pipeline diagrams and annotated pseudocode. An interactive widget would add complexity without pedagogical value -- the student does not need to manipulate anything, they need to SEE the complete system.

The lesson uses:
- Full pipeline diagram (visual): horizontal data flow with tensor shapes and color-coded components
- Annotated pseudocode: complete inference procedure with shape comments at every line
- ComparisonRow: training pipeline vs inference pipeline
- WarningBlocks: U-Net does not see text, VAE encoder not used during inference
- TipBlock: parameter counts and component importance
- ConceptBlock: negative prompts as CFG application
- Callbacks to mental models from 6.3.1 (WHAT/WHERE), 6.3.2 (WHEN), 6.3.4 (cross-attention, CFG contrast slider), 6.3.5 (translator, same orchestra)
- Colab notebook with 4 exercises (Guided -> Guided -> Supported -> Independent)

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (zero gaps)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps to resolve)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (accumulated knowledge payoff, "hear the symphony")
- [x] At least 3 modalities planned for the core concept, each with rationale (visual, symbolic, concrete, verbal/analogy, intuitive -- 5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 0 new concepts (well within limit of 3)
- [x] Every new concept connected to at least one existing concept (pipeline connects to latent diffusion from 6.3.5; every component connects to its source lesson)
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings. One improvement finding (spaced em dashes throughout rendered text) needs a fix pass. Three polish items are minor and can be addressed opportunistically.

### Findings

#### [IMPROVEMENT] -- Spaced em dashes in rendered text throughout the lesson

**Location:** Multiple locations -- subtitle props (lines 249, 298, 520, 796), SummaryBlock description strings (lines 1100, 1106, 1112), ComparisonRow item strings (lines 741, 756)
**Issue:** The Writing Style Rule requires em dashes with no spaces: `word—word` not `word — word`. At least 9 instances of spaced em dashes appear in student-visible text: subtitle props rendered by SectionHeader, SummaryBlock description strings, and ComparisonRow item strings. These all use ` — ` (space-dash-space) instead of `—` (no spaces). Python comments inside the CodeBlock (lines 525, 527, 530, 534) are arguable since they are code comments, but the subtitle, SummaryBlock, and ComparisonRow instances are clearly rendered prose.
**Student impact:** No comprehension impact -- this is a style consistency issue. The spaced em dashes work fine visually but violate the established convention across all other lessons.
**Suggested fix:** Replace all ` — ` with `—` (or `&mdash;` where in JSX prose) in the subtitle props, SummaryBlock descriptions, and ComparisonRow items. The Python code comments can stay as-is since they follow Python comment conventions rather than prose conventions.

#### [POLISH] -- Notebook Exercise 4 tip reveals VAE scaling factor implementation detail

**Location:** Notebook, Exercise 4 markdown cell (cell-19)
**Issue:** The markdown tip `vae.decode(z_0 / vae.config.scaling_factor).sample` provides an implementation-specific detail (the scaling factor division) that the student would need to discover or look up in an Independent exercise. This slightly undermines the Independent label since it reveals a non-obvious API detail.
**Student impact:** Minimal. The scaling factor is a diffusers implementation detail, not a conceptual exercise. Withholding it would cause frustration without teaching anything. The Independent label is about tracing the conceptual pipeline, not debugging API quirks.
**Suggested fix:** Acceptable as-is. If tightening, move the scaling factor hint into the solution `<details>` block instead of the markdown tips, and replace the tip with a more general note: "Check the VAE decode API -- you may need a scaling adjustment." However, this is borderline pedantic.

#### [POLISH] -- ComparisonRow for "Without CLIP" / "Without the VAE" uses rose for both sides

**Location:** Section after "One Pipeline, Three Models" (lines 736-759)
**Issue:** Both sides of the ComparisonRow use `color: 'rose'`. ComparisonRow is typically used to contrast two things with different colors (e.g., amber vs blue for training vs inference, which this lesson does correctly later). Using the same color for both sides is technically valid since these are both "negative" scenarios (removing a component), but it slightly undermines the visual contrast that ComparisonRow is designed to provide.
**Student impact:** Minimal. The student can still read and understand both cards. The rose color on both sides communicates "these are both bad outcomes" which is the correct message.
**Suggested fix:** Acceptable as-is. The same-color choice is a deliberate signal that both scenarios are degraded versions. If changing, use rose/amber or rose/orange to add slight visual differentiation while maintaining the "negative outcome" feel.

#### [POLISH] -- Mermaid diagram styling may not render optimally with all themes

**Location:** Pipeline diagram (lines 473-496)
**Issue:** The Mermaid diagram uses hardcoded fill colors (`fill:#1e293b`, `stroke:#6366f1`, etc.) that are designed for dark mode. If the app ever adds a light mode, these colors would have low contrast. The diagram description text mentions "color-coded by component" but the subgraph labels already differentiate components, so the fill colors are supplementary.
**Student impact:** None currently (app appears to be dark-mode only). Future-proofing concern.
**Suggested fix:** No action needed now. Note for future theme work.

### Review Notes

**What works well:**
- The lesson delivers exactly what a CONSOLIDATE lesson should: zero new concepts, pure assembly, with a strong emotional payoff. The "symphony" analogy and the "everything you know" enumeration at the end are effective.
- Pipeline trace is complete and accurate. All tensor shapes match Stable Diffusion v1.5 specifications (77 tokens, 768 CLIP dimensions, 4x64x64 latents, 320/640/1280/1280 U-Net channels, 3x512x512 output).
- All 5 planned misconceptions are addressed with concrete negative examples, placed at the right locations in the lesson.
- The notebook matches the planning document's 4 exercises with correct scaffolding progression (Guided -> Guided -> Supported -> Independent).
- Row layout is used correctly throughout -- every content section is wrapped in `<Row><Row.Content>...</Row.Content></Row>`, with asides where appropriate.
- Block components (ConceptBlock, WarningBlock, TipBlock, InsightBlock, ComparisonRow, GradientCard, SummaryBlock) are all used appropriately -- WarningBlocks for misconceptions, ConceptBlocks for shape checkpoints, TipBlocks for supplementary information, InsightBlocks for "aha" moments.
- The pseudocode is a strong pedagogical choice -- annotating every line with its source lesson makes the "you know all of this" claim concrete and verifiable.
- Callbacks to prior lessons are specific and accurate (correct lesson names, correct concepts, correct analogies).

**Patterns observed:**
- The spaced em dash issue is the only systematic style violation. All other writing conventions are followed correctly.
- The lesson is well-calibrated for CONSOLIDATE cognitive load -- at no point does the student encounter a concept they haven't seen before. Even "negative prompts" is framed as a direct application of the known CFG formula, not a new concept.

---

## Review -- 2026-02-14 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

The iteration 1 improvement finding (spaced em dashes in 9 student-visible locations) has been fully resolved. All subtitle props, SummaryBlock descriptions, and ComparisonRow items now use unspaced em dashes (`—` or `&mdash;`), matching the Writing Style Rule. No new issues were introduced by the fix. The three iteration 1 polish items (notebook VAE scaling factor tip, same-color ComparisonRow, Mermaid dark-mode colors) remain as-is per the original recommendation and do not require action.

### Findings

None. All iteration 1 findings are resolved or accepted as-is.

### Review Notes

**Verification of iteration 1 fix:**
- Subtitle props (lines 249, 298, 520, 796): all now use `—` with no spaces. Confirmed.
- SummaryBlock descriptions (lines 1100, 1106, 1112): all now use `—` with no spaces. Confirmed.
- ComparisonRow items (lines 741, 756): all now use `—` with no spaces. Confirmed.
- Python code comments inside CodeBlock (lines 525, 527, 530, 534): kept as spaced em dashes per Python comment conventions. Correct.
- JSX comments (lines 104, 202): spaced em dashes in non-rendered comments. Acceptable.
- Mermaid diagram label (line 480): spaced em dash in Mermaid syntax where `&mdash;` would not render. Acceptable.

**Regression check:** The fix was a targeted string replacement in 9 locations. No new JSX syntax issues, no broken rendering, no awkward phrasing introduced. All fixes read naturally in context.

**Iteration 1 polish items (retained as-is):**
1. Notebook Exercise 4 VAE scaling factor tip -- acceptable; withholding would cause frustration over an API detail.
2. ComparisonRow same-color rose -- acceptable; deliberate signal that both scenarios are degraded.
3. Mermaid hardcoded dark-mode colors -- acceptable; app is dark-mode only, future-proofing concern.

Lesson is ready to proceed to Phase 5 (Record).
