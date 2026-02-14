# Module 6.5: Customization -- Record

**Goal:** The student can customize Stable Diffusion through LoRA fine-tuning for style and subject adaptation, img2img/inpainting for inference-time control, and textual inversion for concept injection--understanding the mechanisms behind each technique and when to apply them.
**Status:** Complete (3 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| LoRA placement in diffusion U-Net (cross-attention projections W_Q, W_K, W_V, W_out as primary targets because cross-attention is where text meaning meets spatial features) | DEVELOPED | lora-finetuning | Core new concept #1. Motivated by reasoning: style/subject = how text maps to visual features, cross-attention = where text meets image. Negative example: conv-only LoRA produces minimal style effect because conv layers do not interact with text conditioning. "Steering wheel in the trunk" analogy. |
| Diffusion LoRA training loop (DDPM training with frozen base weights; encode image with frozen VAE, sample random timestep, add noise, U-Net predicts noise, MSE loss, backprop into LoRA params only) | DEVELOPED | lora-finetuning | Core new concept #2. Taught via side-by-side ComparisonRow with LLM LoRA training loop showing "80% identical." Concrete worked example tracing one training step for watercolor style LoRA with tensor shapes at every stage. |
| Style vs subject LoRA (same mechanism, different data strategies: style uses 50-200 diverse images, subject uses 5-20 images with rare-token identifier like "sks") | INTRODUCED | lora-finetuning | ComparisonRow side-by-side. Rare-token identifier explained as avoiding pollution of common words. |
| Diffusion LoRA rank and alpha (typically r=4-8, lower than LLM LoRA r=8-64, because style adaptation is a smaller delta; alpha typically 1.0 or equal to rank) | INTRODUCED | lora-finetuning | Three GradientCards for r=1-2 (underfitting), r=4-8 (sweet spot), r=16-64 (overfitting risk). Lower rank as implicit regularization for small diffusion datasets. |
| Multiple LoRA composition (summing bypasses: W_combined = W + BA_style * alpha_1/r_1 + BA_subject * alpha_2/r_2; small swappable files, 2-50 MB each) | INTRODUCED | lora-finetuning | Extends the SD modularity concept from 6.4.1. WarningBlock on composition interference when two LoRAs push projections in conflicting directions. |
| LoRA overfitting in diffusion (generated images become copies of training images rather than novel compositions; mitigated by fewer steps, lower rank, more diverse images, lower learning rate) | INTRODUCED | lora-finetuning | Framed as the most common failure mode. Connects to "finetuning is a refinement, not a revolution" mental model from 4.4.4. |
| LoRA effect across timesteps (cross-attention has different influence at different noise levels; style LoRAs have strongest visible effect at low noise where spatial features are well-formed) | INTRODUCED | lora-finetuning | Connected to coarse-to-fine denoising mental model from 6.3.1. LoRA modifies same weights at every timestep but the effect is mediated by the quality of spatial features. |
| Style leakage (LoRA modifies cross-attention projections used for ALL prompts, not gated by specific words; style applies even to prompts that do not mention the trained style) | INTRODUCED | lora-finetuning | Taught via predict-and-verify question. The style is baked into the projections themselves. |
| "Same detour, different highway" mental model (LoRA bypass mechanism identical to Module 4.4; the highway is the diffusion U-Net cross-attention instead of LLM self-attention; the traffic is spatial features and text embeddings instead of token sequences) | DEVELOPED | lora-finetuning | Primary mental model for the lesson. Extends the "highway and detour" analogy from 4.4.4 to the diffusion context. |
| Img2img as partial denoising (start denoising loop from a noised real image instead of pure noise; VAE encode input, noise to t_start with forward process formula, denoise from t_start to t=0) | DEVELOPED | img2img-and-inpainting | Core new concept #1. Taught as a reconfiguration of the existing pipeline--one change to the starting point. Pipeline comparison diagram (txt2img vs img2img) highlights the single difference. Worked example traces full pipeline with tensor shapes. |
| Strength parameter as starting timestep mapping (strength = fraction of denoising steps that run; maps onto alpha-bar curve nonlinearly; low strength = detail refinement only, high strength = structural reimagination) | DEVELOPED | img2img-and-inpainting | Directly connected to alpha-bar curve from 6.2.2 and coarse-to-fine denoising from 6.2.4/6.3.1. Three GradientCards for low/medium/high strength tiers. Boundary cases: strength=1.0 collapses to standard txt2img, strength=0.0 returns input unchanged. |
| Inpainting as per-step spatial masking (binary mask applied at each denoising step: masked regions use model prediction, unmasked regions use re-noised original; formula z_t_combined = m * z_t_denoised + (1-m) * forward(z_0_original, t)) | DEVELOPED | img2img-and-inpainting | Core new concept #2. Taught as one-line addition to the denoising loop. Concrete tree-to-fountain scenario before formula (concrete-before-abstract). Re-noising explained via noise-level consistency requirement. |
| Seamless inpainting boundary blending (U-Net sees full latent at every step; global receptive field at 8x8 bottleneck means predictions for masked region account for unmasked context; fundamentally different from cut-and-paste) | DEVELOPED | img2img-and-inpainting | Connected to U-Net receptive field from 6.3.1. Contrasted with pixel-space cut-and-paste to address seam misconception. |
| VAE encoder used during img2img inference (VAE encoder NOT used in txt2img but IS needed for img2img to convert input image to latent space; resolves the "encoder is not used during inference" statement from 6.4.1) | DEVELOPED | img2img-and-inpainting | Callback to Misconception #4 from 6.4.1. Aside explicitly addresses the transition: "We said the encoder is not used during text-to-image. That was correct. Img2img is image-to-image, and it needs the encoder." |
| Forward process formula in third and fourth contexts (inference-time img2img noising and inpainting per-step re-noising, after training and capstone contexts) | DEVELOPED | img2img-and-inpainting | Reinforces x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon from 6.2.2. Three new applications: img2img starting point noising, inpainting per-step re-noising of unmasked regions, LoRA training noising. |
| Customization spectrum (LoRA changes weights via training, img2img/inpainting changes inference process without training, textual inversion changes embeddings via training--three knobs on three different pipeline parts) | DEVELOPED | img2img-and-inpainting, textual-inversion | INTRODUCED in img2img-and-inpainting via three GradientCards. Elevated to DEVELOPED in textual-inversion where all three knobs have been taught and compared with use-case recommendations. |
| Pseudo-token creation and embedding optimization (add a new token to CLIP's vocabulary, extend embedding table by one row [49408,768]->[49409,768], initialize from related word or random, optimize only that 768-float vector via standard DDPM training loop while freezing all model weights) | DEVELOPED | textual-inversion | Core new concept #1. Taught via derivation challenge (student derives the technique from three premises before being told its name), then concrete pseudocode, Mermaid gradient flow diagram, and ComparisonRow with LoRA training loop. "Pseudocode vs Reality" GradientCard addresses the PyTorch subtlety of enabling grad on full embedding table and restoring non-pseudo-token rows. |
| Textual inversion expressiveness boundaries vs LoRA (one 768-dim vector captures object identity and simple visual attributes; cannot encode compositional or spatial style patterns; 768 params vs 200K-400K LoRA params; ~4 KB output vs 2-50 MB LoRA) | DEVELOPED | textual-inversion | Core new concept #2. ComparisonRow (textual inversion vs LoRA) on parameters, training steps, file size, and capabilities. Negative example: training `<ghibli-style>` for complex spatial style works poorly because one vector cannot encode compositional recipes. Three GradientCards for when to use each technique. |
| Two-stage CLIP text encoding pipeline (stage 1: token embedding lookup, stage 2: CLIP transformer contextualizes via self-attention; textual inversion optimizes at stage 1 but the U-Net sees stage 2 output) | DEVELOPED | textual-inversion | Explicitly separates the two stages for the first time. Key insight: same initial embedding produces different contextual embeddings depending on surrounding prompt. Addresses misconception that pseudo-token is processed in isolation. |
| Gradient flow through frozen layers to embedding table (gradients pass through frozen U-Net and frozen CLIP encoder without updating their weights; only the embedding row receives the update; ~123M frozen parameters as a gradient pathway) | DEVELOPED | textual-inversion | Mermaid diagram showing forward pass (solid arrows) and backward pass (dashed arrows) through frozen layers. WarningBlock: "Frozen does not mean invisible to gradients." Extends the "freeze everything except X" pattern from LoRA with X now at the very start of the pipeline. |
| Embedding initialization strategy (random vs from related word; starting near "cat" leverages "geometry encodes meaning" for faster convergence) | INTRODUCED | textual-inversion | ComparisonRow (random vs related word). Connected to embedding space clustering from 4.1.3. Not DEVELOPED because student does not experiment with both strategies. |
| Learned embedding as novel point in continuous space (not equal to any existing vocabulary token; sits between related tokens; continuous space has infinitely more positions than discrete vocabulary entries) | INTRODUCED | textual-inversion | Connects to EmbeddingSpaceExplorer experience from 4.1.3. Addresses misconception that optimization converges to an existing token. |
| Multi-token textual inversion (using 2-4 pseudo-tokens for one concept; 1536-3072 parameters; each token can specialize for texture/color/shape) | MENTIONED | textual-inversion | Brief ConceptBlock aside. Not developed further. |
| Model compatibility for textual inversion embeddings (embedding only valid in the CLIP space where it was trained; SD v1.4/v1.5 share CLIP so embeddings transfer; SD v2.0/SDXL use different encoders so embeddings do not transfer) | INTRODUCED | textual-inversion | GradientCard with language analogy. Transfer check question reinforces via SDXL cross-model scenario. |
| Zero weight change verification (any prompt without the pseudo-token produces identical output before and after training; the model is unchanged, only the vocabulary grew) | DEVELOPED | textual-inversion | GradientCard proof. Predict-and-verify question: "Has the output changed?" for a prompt without the pseudo-token. Addresses misconception #1 (textual inversion modifies model weights). |

## Per-Lesson Summaries

### lora-finetuning (Lesson 1)
**Status:** Built, reviewed (PASS on iteration 3), two polish items noted
**Cognitive load:** BUILD (2 new concepts: LoRA placement in U-Net, diffusion LoRA training loop)
**Notebook:** `notebooks/6-5-1-lora-finetuning.ipynb` (4 exercises: inspect LoRA target layers, one training step by hand, train a style LoRA with real data, LoRA composition experiment)

**Concepts taught:**
- LoRA placement in diffusion U-Net (DEVELOPED)--cross-attention projections as primary target, motivated by "cross-attention is where text meets image"
- Diffusion LoRA training loop (DEVELOPED)--DDPM training + LoRA frozen-base-weights pattern, taught via side-by-side comparison with LLM LoRA training
- Style vs subject LoRA data strategies (INTRODUCED)--50-200 images for style, 5-20 for subject with rare-token identifier
- Diffusion LoRA rank and alpha (INTRODUCED)--r=4-8 sweet spot, lower than LLM LoRA due to smaller delta
- Multiple LoRA composition (INTRODUCED)--summing bypasses, swappable small files, interference warning
- LoRA overfitting symptoms and mitigation (INTRODUCED)
- LoRA effect across timesteps (INTRODUCED)--connected to coarse-to-fine mental model
- Style leakage (INTRODUCED)--LoRA applies to all prompts, not gated by specific words

**Mental models established:**
- "Same detour, different highway"--the LoRA bypass is the same trainable shortcut from Module 4.4. The highway it attaches to is the diffusion U-Net's cross-attention projections. The traffic is spatial features and text embeddings instead of token sequences. The detour mechanism is identical.
- "Of course cross-attention"--if you want to change what "watercolor" means visually, you change the projections that translate between text meaning and spatial features. Conv layers handle edges and textures at a low level; they do not know about "watercolor."

**Analogies used:**
- "Same detour, different highway" (extending "highway and detour" from 4.4.4 to diffusion context)
- "Steering wheel in the trunk" (conv-only LoRA negative example--mechanism in the wrong location)
- "Of course cross-attention" (intuition for target layer selection)
- "Finetuning is a refinement, not a revolution" callback (why few images suffice; model already knows how to generate)

**How concepts were taught:**
- Quick recap: reactivated LoRA mechanism (highway + detour, B=0 init, merge at inference) and SD pipeline (text -> CLIP -> U-Net cross-attention -> VAE decode). Both at DEVELOPED depth but 14+ lessons ago; brief reactivation per Reinforcement Rule.
- Hook: prediction checkpoint with three questions (target layers, loss function, data requirements). Calibrates overconfidence--most students get #1 partially right, #2 wrong (guess cross-entropy), #3 wrong (guess too many images).
- Where LoRA goes: Mermaid diagram of U-Net residual block with LoRA at cross-attention projections (color-coded frozen=gray, LoRA=violet). Two GradientCards: LoRA-enabled vs frozen layers. "Of course" intuition connecting to cross-attention's role from 6.3.4. Negative example: conv-only LoRA with "steering wheel in the trunk" analogy.
- Training loop: ComparisonRow (LLM LoRA step vs diffusion LoRA step, 6 items per side). Worked example: one training step for watercolor style LoRA with tensor shapes at every stage (image [3,512,512] -> VAE encode -> z_0 [4,64,64] -> noise -> U-Net predicts noise -> MSE loss -> backprop into LoRA only). Every line mapped to its source lesson.
- Check: two predict-and-verify questions (style leakage, merge and speed).
- Style vs subject: ComparisonRow (violet/emerald). Rare-token identifier explained. TipBlock: "Why So Few Images?"
- Rank and alpha: three GradientCards (r=1-2 underfit, r=4-8 sweet spot, r=16-64 overfit). ConceptBlock on alpha scaling.
- Multiple LoRA composition: formula for combined weights. WarningBlock on interference.
- Practical nuances: overfitting symptoms and mitigation. LoRA effect across timesteps connected to coarse-to-fine from 6.3.1.
- Transfer check: edge case (color palette--boundary of "of course cross-attention" heuristic) and mechanism vs context question.
- Practice: notebook with 4 exercises (Guided: inspect layers, Guided: one training step, Supported: train style LoRA with naruto-blip-captions dataset, Independent: LoRA composition experiment).

**Misconceptions addressed:**
1. "LoRA for diffusion works identically to LoRA for LLMs--same target layers, same training loop, same data format"--side-by-side ComparisonRow showing different data, preprocessing, forward pass, and loss. The mechanism transfers; the training context does not.
2. "LoRA adapters are applied to every layer of the U-Net"--negative example: conv-only LoRA produces minimal style effect. Cross-attention is where text meets image; conv layers handle spatial processing without text interaction.
3. "You need hundreds or thousands of images to train a diffusion LoRA"--subject LoRAs train on 5-20 images. The model already knows how to generate; the LoRA encodes the delta. Overfitting is the bigger risk.
4. "Training a LoRA changes how the model denoises at all timesteps equally"--cross-attention influence varies with noise level. Style LoRAs have strongest effect at low noise where spatial features are well-formed.
5. "You can just use LLM LoRA knowledge and skip this lesson"--prediction checkpoint: loss function is MSE on noise, not cross-entropy on tokens. Data format is images+captions, not text sequences.

**What is NOT covered (deferred):**
- Reimplementing LoRA from scratch (done in 4.4.4)
- DreamBooth (different fine-tuning technique, full weight modification)
- Textual inversion (Lesson 3 of this module)
- ControlNet or structural conditioning (Series 7)
- Img2img or inpainting (Lesson 2 of this module)
- SD v1 vs v2 vs XL LoRA differences
- LoRA mathematical derivation or SVD

**Review notes:**
- Iteration 1: MAJOR REVISION--1 critical (notebook Exercise 3 used synthetic random noise as training data instead of real images), 3 improvement (VRAM management, Exercise 4 external LoRA repos, Exercise 4 solution missing reasoning), 2 polish (Mermaid emoji, spaced em dashes in code comments)
- Iteration 2: NEEDS REVISION--0 critical, 2 improvement (cell-26 loads two pipelines simultaneously risking OOM, Exercise 4 trains 200 extra steps for infrastructure not learning), 2 polish (notebook spaced em dashes, Mermaid emoji)
- Iteration 3: PASS--all fixes applied. Notebook uses naruto-blip-captions dataset, VRAM-conscious sequential pipeline loading, 100-step second LoRA training. Two polish items remaining (spaced em dashes in notebook code comments, Mermaid emoji rendering).

### img2img-and-inpainting (Lesson 2)
**Status:** Built, reviewed (PASS on iteration 3), one polish item noted (HTML comment em dashes)
**Cognitive load:** BUILD (2 new concepts: img2img as partial denoising, inpainting as per-step spatial masking)
**Notebook:** `notebooks/6-5-2-img2img-and-inpainting.ipynb` (4 exercises: img2img strength exploration, implement img2img from scratch, inpainting with mask design, creative multi-step workflow)

**Concepts taught:**
- Img2img as partial denoising (DEVELOPED)--start denoising from a noised real image; VAE encode, forward process noise to t_start, denoise from there
- Strength parameter as starting timestep mapping (DEVELOPED)--maps onto alpha-bar curve nonlinearly; low strength = detail refinement, high strength = structural reimagination
- Inpainting as per-step spatial masking (DEVELOPED)--binary mask applied at each step; masked regions denoised, unmasked regions replaced with re-noised original
- Seamless boundary blending via U-Net receptive field (DEVELOPED)--global receptive field at 8x8 bottleneck means model accounts for unmasked context
- VAE encoder used during img2img inference (DEVELOPED)--resolves "encoder not used during inference" from 6.4.1
- Forward process formula in new inference-time contexts (DEVELOPED)--reinforces formula from 6.2.2 in img2img noising and inpainting re-noising
- Customization spectrum: weights vs inference vs embeddings (INTRODUCED)--module-level framing comparing LoRA, img2img/inpainting, textual inversion

**Mental models established:**
- "Same denoising process, different starting point (img2img) or selective application (inpainting)"--the pipeline is unchanged; img2img moves the starting line, inpainting adds a spatial filter
- "Three knobs on three different parts of the pipeline"--LoRA changes weights, img2img/inpainting changes inference, textual inversion changes embeddings

**Analogies used:**
- "Starting mid-hike"--standard txt2img hikes from the summit (pure noise) to the valley (clean image); img2img starts partway down
- "Not image blending"--contrasts with pixel-space alpha compositing from 6.1.4; img2img at strength=0.5 produces a coherent image, not a ghostly double exposure
- "Not cut-and-paste"--inpainting operates iteratively in latent space with full context, unlike pixel-space pasting with no context

**How concepts were taught:**
- Quick recap: reactivated forward process closed-form formula, denoising loop, alpha-bar as signal-to-noise dial. Three pieces the student already has, framed as the building blocks for img2img.
- Hook: challenge question "What if you didn't start from pure noise?" with reveal via `<details>`. Student should be able to derive img2img from prior knowledge before being told. Landscape-to-watercolor scenario.
- Img2img mechanism: ComparisonRow (txt2img pipeline vs img2img pipeline) highlighting the single difference (starting point). VAE encoder callback in aside resolving the "encoder is not used" statement from 6.4.1. Worked example with tensor shapes: 512x512 input -> VAE encode [4,64,64] -> forward process noise to t_start -> denoise from t_start to t=0 -> VAE decode.
- Strength parameter: three-tier GradientCards (low/medium/high). Boundary cases via ComparisonRow (strength=1.0 = standard txt2img, strength=0.0 = input unchanged). Nonlinearity explained via alpha-bar curve and coarse-to-fine mental model from 6.2.4.
- Check: two predict-and-verify questions (dog-to-cat at strength=0.5 vs strength=0.1).
- Inpainting mechanism: motivation paragraph ("what if you only want to change part?"), concrete tree-to-fountain scenario before formula (concrete-before-abstract), BlockMath formula, pseudocode showing one-line addition to denoising loop. Why re-noise explained via noise-level consistency GradientCard.
- Boundary blending: connected to U-Net receptive field from 6.3.1. Contrasted with cut-and-paste (InsightBlock).
- Inpainting boundary cases: ComparisonRow (full mask = img2img, empty mask = unchanged).
- Check: two predict-and-verify questions (sky replacement boundary, full-image mask).
- Practical considerations: sketch-to-image application, mask sizing tips, prompt interaction with inpainting.
- Customization spectrum: three GradientCards comparing LoRA / img2img-inpainting / textual inversion. "Three knobs" InsightBlock.
- Transfer check: outpainting derivation (apply inpainting mechanism to image extension) and sampler orthogonality question.
- Practice: notebook with 4 exercises (Guided: strength exploration grid, Guided: implement img2img from scratch with raw components, Supported: inpainting with tight vs generous mask comparison, Independent: creative multi-step workflow combining both techniques).

**Misconceptions addressed:**
1. "Img2img modifies the original image directly (like a Photoshop filter)"--strength=1.0 boundary case: original is completely destroyed, output is standard txt2img. A filter would preserve structure.
2. "Img2img blends the original with the generated image at pixel level"--contrasted with pixel-space interpolation from 6.1.4 (exploring latent spaces). Img2img at strength=0.5 produces a coherent image, not a double exposure.
3. "Inpainting uses a separate, specialized model"--same U-Net, VAE, CLIP. The mask is the entire mechanism. Specialized inpainting models mentioned as optimizations, not requirements.
4. "Strength parameter linearly scales how much output resembles input"--nonlinear due to alpha-bar curve and coarse-to-fine denoising. Jump from 0.3 to 0.5 qualitatively different from 0.7 to 0.9.
5. "Inpainting boundaries will show visible seams"--U-Net global receptive field at 8x8 bottleneck. Model sees full latent at every step. Fundamentally different from cut-and-paste.

**What is NOT covered (deferred):**
- Training any model (img2img and inpainting are inference-time only)
- ControlNet or structural conditioning (Series 7)
- Specialized inpainting models with extra input channels (mentioned only)
- Outpainting / image extension (mentioned in transfer question, not developed)
- SDEdit or other img2img variants beyond standard noise-and-denoise
- Depth-guided or edge-guided generation (Series 7 / ControlNet)
- Video inpainting
- Textual inversion (Lesson 3 of this module)

**Review notes:**
- Iteration 1: MAJOR REVISION--1 critical (notebook cell-12 syntax error blocking Exercise 2), 3 improvement (sampler mismatch between Ex1 and Ex2, Exercise 3 scaffolding inconsistency, inpainting formula before concrete scenario), 2 polish (HTML comment em dashes, alpha-bar curve plot mapping)
- Iteration 2: NEEDS REVISION--0 critical, 1 improvement (VAE encode non-determinism in Exercise 2 comparison), 1 polish (notebook spaced em dashes)
- Iteration 3: PASS--all fixes applied. VAE encode uses .mode() for deterministic comparison. Both exercises use DDIMScheduler. Alpha-bar curve uses exact scheduler positions. One polish item remaining (HTML comment em dashes).

### textual-inversion (Lesson 3)
**Status:** Built, reviewed (PASS on iteration 2), one polish item noted (HTML comment em dashes)
**Cognitive load:** STRETCH (2 new concepts: pseudo-token embedding optimization, expressiveness boundaries vs LoRA)
**Notebook:** `notebooks/6-5-3-textual-inversion.ipynb` (4 exercises: explore CLIP embedding table, one training step by hand with gradient verification, train full textual inversion embedding, compare textual inversion vs LoRA)

**Concepts taught:**
- Pseudo-token creation and embedding optimization (DEVELOPED)--add token to vocabulary, extend embedding table by one row, freeze entire model, optimize only that row via DDPM training loop
- Textual inversion expressiveness boundaries vs LoRA (DEVELOPED)--768 params vs 200K-400K; captures object identity but not complex spatial styles
- Two-stage CLIP text encoding pipeline (DEVELOPED)--embedding lookup (stage 1, where TI intervenes) vs CLIP transformer contextualization (stage 2)
- Gradient flow through frozen layers to embedding table (DEVELOPED)--gradients pass through frozen U-Net and CLIP encoder to reach the one trainable row
- Zero weight change verification (DEVELOPED)--prompts without pseudo-token produce identical output before and after training
- Embedding initialization strategy (INTRODUCED)--random vs from related word, connected to "geometry encodes meaning"
- Learned embedding as novel point in continuous space (INTRODUCED)--not equal to any existing vocabulary token
- Model compatibility for textual inversion embeddings (INTRODUCED)--embedding only valid in the CLIP space where trained
- Multi-token textual inversion (MENTIONED)--2-4 pseudo-tokens for more capacity
- Customization spectrum elevated to DEVELOPED--all three knobs (weights, inference, embeddings) now taught and compared

**Mental models established:**
- "Inventing a new word in CLIP's language"--finding the right point in embedding space where, if a word existed there, the model would understand it as your concept. Not changing the grammar (U-Net weights) or accent (inference process); adding one word to the dictionary (embedding table).
- "Three knobs on three different parts of the pipeline" (elevated from INTRODUCED to DEVELOPED)--LoRA changes weights, img2img/inpainting changes inference, textual inversion changes embeddings. Complete customization spectrum.

**Analogies used:**
- "Inventing a new word" (core analogy: every word has a precise geometric position; your concept has no word; textual inversion coins one)
- "Not changing the machine, changing what the machine hears" (framing for the conceptual inversion)
- "Same pattern, different target" (freeze everything except X; X was LoRA matrices, now X is one embedding row)
- "Word from one language in another language's conversation" (model compatibility--embedding only means something in its training space)

**How concepts were taught:**
- Three premises recap: reactivated two-stage CLIP pipeline, "embeddings are learned parameters," and "CLIP's space is continuous and meaningful" as the three premises from which textual inversion follows logically.
- Hook: derivation challenge. Student given three premises and challenged to derive textual inversion before being told its name. `<details>` reveal: "This technique is called textual inversion. You just reinvented it."
- Pseudo-token mechanism: token added to vocabulary (ID 49409), embedding table extended by one row (768 floats). TipBlock: one row, 768 trainable floats, that is the entire "model."
- Initialization: ComparisonRow (random vs from related word). Connected to "geometry encodes meaning" and EmbeddingSpaceExplorer from 4.1.3.
- Training loop: ComparisonRow (LoRA training step vs textual inversion training step, 6 items per side). Steps 1-5 nearly identical; only step 6 differs (what receives the gradient update).
- Gradient flow: Mermaid diagram showing forward pass (solid arrows) and backward pass (dashed arrows) through frozen U-Net and CLIP encoder to the one trainable embedding row. Color-coded: violet for trainable, gray for frozen.
- Zero weight change: GradientCard verification. Any prompt without `<my-cat>` produces identical results before and after training.
- Pseudocode: 5-line setup (add token, resize embeddings, freeze everything, unfreeze one row, create optimizer) + identical DDPM training loop. "Pseudocode vs Reality" GradientCard explains the actual PyTorch pattern (enable grad on full table, restore non-pseudo-token rows after each step).
- Two-stage pipeline: walked through tokenizer -> embedding lookup -> CLIP transformer self-attention -> contextual embeddings. Key insight: same initial embedding produces different contextual embeddings depending on surrounding prompt.
- Check: two predict-and-verify questions (unchanged prompts produce identical output; novel composition works because U-Net knowledge is preserved).
- Expressiveness: ComparisonRow (textual inversion vs LoRA on params, training steps, file size, capabilities). Negative example: `<ghibli-style>` for complex spatial style fails because one vector cannot encode compositional recipes. Three GradientCards: when to use textual inversion vs LoRA vs img2img/inpainting.
- "One word ceiling": single token participates in cross-attention at token level, not architectural level. Cannot encode compositional instructions.
- Learned embedding in geometric space: connected to EmbeddingSpaceExplorer. Continuous space has infinitely more positions than discrete vocabulary entries.
- Model compatibility: GradientCard. SD v1.4/v1.5 share CLIP encoder so embeddings transfer. SD v2.0/SDXL use different encoders so embeddings do not transfer.
- Transfer check: company logo (textual inversion recommended--specific visual object) and cross-model transfer (SDXL incompatible).
- Practice: notebook with 4 exercises (Guided: explore CLIP embedding table and cosine similarities, Guided: one training step with gradient verification and restore-original-rows pattern, Supported: train full embedding on concept images for 3000 steps, Independent: compare textual inversion vs LoRA on same images).
- Module completion: ModuleCompleteBlock listing all three customization techniques. Series completion GradientCard summarizing the entire Series 6 journey.

**Misconceptions addressed:**
1. "Textual inversion modifies the model's weights"--zero weight change verification. Any prompt without the pseudo-token produces identical output. Only the embedding vocabulary grew.
2. "The pseudo-token's embedding is processed in isolation (bag of separate embeddings)"--CLIP transformer applies self-attention across all tokens. Same initial embedding produces different contextual embeddings in different prompts.
3. "Textual inversion is as expressive as LoRA"--negative example: complex spatial style (Studio Ghibli) fails because one 768-dim vector cannot encode compositional recipes. LoRA modifies thousands of projection parameters.
4. "You need to train or fine-tune the CLIP text encoder"--CLIP encoder is completely frozen. Gradients flow THROUGH it but do not update its ~123M parameters. Only the 768-float embedding vector is optimized.
5. "The pseudo-token must be a 'real' word or the closest existing word"--after training, cosine similarity with all 49,408 vocabulary entries shows the learned vector is not close to any single existing token. It is a genuinely new point in continuous space.

**What is NOT covered (deferred):**
- DreamBooth (full or partial model fine-tuning)
- Multi-token textual inversion (mentioned only)
- Hypernetworks or other embedding-space methods
- Training tricks (progressive schedules, mixing real and synthetic images)
- CLIP internal architecture details or reimplementing CLIP
- ControlNet or structural conditioning (Series 7)
- Production deployment or speed optimization

**Review notes:**
- Iteration 1: MAJOR REVISION--1 critical (notebook Exercise 2 gradient verification shows wrong result: claims only pseudo-token row has non-zero gradients, but in PyTorch all rows that participate in forward pass receive gradients via autograd), 4 improvement (lesson pseudocode shows non-functional freeze/unfreeze pattern, Exercise 3 float32 VRAM pressure, narrative drifts from object to style without acknowledgment, missing sklearn in pip install), 3 polish (spaced em dashes in TSX prose, Mermaid emoji characters, HTML comment em dashes)
- Iteration 2: PASS--critical and 4 of 5 improvement findings fixed. Exercise 2 rewritten to correctly show multiple rows get gradients and demonstrate restore-original-rows pattern. "Pseudocode vs Reality" GradientCard added. Exercise 3 reframed as deliberate pedagogical choice to experience expressiveness limitation. float32 VRAM accepted as correctness tradeoff. One polish item remaining (HTML comment em dashes, not student-visible).
