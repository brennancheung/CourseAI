# Lesson: LoRA Fine-Tuning for Diffusion Models

**Module:** 6.5 Customization
**Position:** Lesson 1 of 3
**Slug:** `lora-finetuning`

---

## Phase 1: Orient -- Student State

The student arrives with an unusually strong foundation for this lesson. They have deep knowledge of LoRA from Module 4.4 (LLM context) and comprehensive understanding of the Stable Diffusion pipeline from Module 6.4.

**Relevant concepts the student has:**

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| LoRA: Low-Rank Adaptation (bypass architecture, forward pass h = Wx + BAx*(alpha/r), B=0 init, merge at inference) | DEVELOPED | lora-and-quantization (4.4.4) | Core mechanism fully understood. Highway-and-detour mental model. Implemented LoRALinear from scratch in notebook. |
| Low-rank decomposition / matrix factorization (W ≈ BA where B is m x r, A is r x n) | INTRODUCED | lora-and-quantization (4.4.4) | Intuitive understanding via concrete 4x4 rank-1 example. "The rows are not all independent -- they lie in a small subspace." No SVD. |
| LoRA hyperparameters (rank r, alpha scaling, target modules W_Q and W_V) | DEVELOPED | lora-and-quantization (4.4.4) | r=4,8,16 common values. Alpha often 2r or fixed at 16. Standard practice: apply to W_Q and W_V. Overfitting checkpoint question understood. |
| Why LoRA works: finetuning weight changes are low-rank ("refinement, not revolution") | DEVELOPED | lora-and-quantization (4.4.4) | Hu et al. results at GPT-3 scale. Low-rank constraint as implicit regularization. ComparisonRow on when LoRA excels vs underperforms. |
| "Finetuning is a refinement, not a revolution" mental model | DEVELOPED | lora-and-quantization (4.4.4) | Established and reinforced. Core intuition for why low-rank adaptation works. |
| "Highway and detour" LoRA analogy | DEVELOPED | lora-and-quantization (4.4.4) | Highway = frozen W, detour = trainable BA. Outputs summed. Detour starts at zero. |
| PEFT library for LoRA finetuning | INTRODUCED | lora-and-quantization (4.4.4) | Used in notebook Exercise 4. Conceptual understanding from manual implementation makes library transparent. |
| Full SD pipeline data flow (text -> CLIP -> U-Net denoising loop with CFG -> VAE decode) | DEVELOPED | stable-diffusion-architecture (6.4.1) | Complete pipeline traced with tensor shapes at every handoff. "Three translators, one pipeline." |
| Component modularity (three independently trained models, swappable via tensor interfaces) | INTRODUCED | stable-diffusion-architecture (6.4.1) | CLIP ~123M, U-Net ~860M, VAE ~84M. Never trained together. Swappable. |
| Cross-attention mechanism in U-Net (Q from spatial features, K/V from text embeddings) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Taught as one-line change from self-attention. Per-spatial-location attention. Shape walkthrough. Concrete attention weight table. |
| Cross-attention projection matrices (W_Q, W_K, W_V, W_out at attention resolutions) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Three-lens callback from 4.2. W_Q applied to spatial features, W_K/W_V applied to text embeddings. |
| DDPM training objective (sample random timestep, noise the image, predict the noise, MSE loss) | DEVELOPED | learning-to-denoise (6.2.3) | Full training loop understood. "Surprisingly simple" framing. |
| Forward process closed-form (jump to any timestep without iterating) | DEVELOPED | the-forward-process (6.2.2) | x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon. Student understands and has used this. |
| Frozen-VAE pattern (VAE trained separately, frozen during diffusion training) | INTRODUCED | from-pixels-to-latents (6.3.5) | Two-stage pipeline: train VAE, freeze, train diffusion on frozen latents. |
| Diffusers API parameter comprehension (every pipe() parameter mapped to concept) | APPLIED | generate-with-stable-diffusion (6.4.3) | "The API is a dashboard. You built the machine behind it." |
| Sampler as inference-time choice (swappable, no retraining) | DEVELOPED | samplers-and-efficiency (6.4.2) | Foundational insight reinforced throughout Module 6.4. |

**Mental models and analogies already established:**
- "Highway and detour" (LoRA bypass)
- "Finetuning is a refinement, not a revolution" (why LoRA works)
- "LoRA is the surgical version of 'freeze the backbone'" (LoRA philosophy)
- "Three translators, one pipeline" (SD modularity)
- "The API is a dashboard. You built the machine behind it." (parameter comprehension)
- "Same formula, different source for K and V" (cross-attention)
- "The timestep tells the network WHEN. The text tells it WHAT. CFG turns up the volume on the WHAT." (conditioning)

**What was explicitly NOT covered that is relevant:**
- LoRA applied to diffusion models (deferred from 4.4.4 and 6.4.3)
- Which U-Net layers are targeted by diffusion LoRA (cross-attention projections)
- How the diffusion training loop works with LoRA (noise-conditioned loss with frozen base)
- Practical workflow for training a LoRA with your own images
- Multiple LoRA composition (stacking adapters)

**Readiness assessment:** Excellent. The student has LoRA at DEVELOPED depth from the LLM context and the full SD pipeline at DEVELOPED depth. This is a transfer lesson -- applying a known technique to a known system. The main new content is the specific intersection: which layers, what training data, how the training loop adapts.

---

## Phase 2: Analyze

### Target Concept

**This lesson teaches the student to understand how LoRA fine-tuning adapts the Stable Diffusion U-Net for new styles and subjects by targeting cross-attention projections, using the same noise-prediction training objective they already know but with LoRA adapters as the only trainable parameters.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| LoRA mechanism (bypass architecture, forward pass, merge) | DEVELOPED | DEVELOPED | lora-and-quantization (4.4.4) | OK | Student needs to reason about where LoRA is inserted and why. DEVELOPED required, DEVELOPED achieved. |
| Low-rank decomposition intuition | INTRODUCED | INTRODUCED | lora-and-quantization (4.4.4) | OK | Needs conceptual understanding, not formal linear algebra. |
| LoRA hyperparameters (r, alpha, target modules) | DEVELOPED | DEVELOPED | lora-and-quantization (4.4.4) | OK | Student will need to reason about different r values for diffusion vs LLM use cases. |
| Cross-attention in U-Net (Q from spatial, K/V from text, W_Q/W_K/W_V projections) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Critical for understanding WHY cross-attention projections are the primary LoRA target. |
| DDPM training objective (noise prediction, MSE loss, random timestep) | DEVELOPED | DEVELOPED | learning-to-denoise (6.2.3) | OK | The diffusion LoRA training loop is this training loop with LoRA adapters unfrozen. |
| Full SD pipeline data flow | DEVELOPED | DEVELOPED | stable-diffusion-architecture (6.4.1) | OK | Student needs to locate where in the pipeline LoRA adapters sit. |
| Component modularity (independently trained, swappable) | INTRODUCED | INTRODUCED | stable-diffusion-architecture (6.4.1) | OK | LoRA adapters are swappable additions to one component. Extends the modularity concept. |
| Frozen-VAE pattern | INTRODUCED | INTRODUCED | from-pixels-to-latents (6.3.5) | OK | LoRA extends this: not just VAE is frozen, but U-Net base weights too. Only LoRA adapters train. |
| Forward process closed-form | DEVELOPED | DEVELOPED | the-forward-process (6.2.2) | OK | Used during LoRA training to noise the target images. |
| Diffusers API | APPLIED | APPLIED | generate-with-stable-diffusion (6.4.3) | OK | Notebook will use diffusers for LoRA training and inference. |

**All prerequisites are OK. No gaps to resolve.**

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "LoRA for diffusion works identically to LoRA for LLMs -- same target layers, same training loop, same data format" | The LoRA mechanism IS identical. Easy to assume the entire process transfers unchanged. Module 4.4 taught LoRA for W_Q and W_V in transformer attention -- same names appear in U-Net cross-attention. | Training loop comparison: LLM LoRA uses next-token prediction loss on text sequences. Diffusion LoRA uses noise-prediction MSE loss on noised images at random timesteps. The loss function, data format, and what the model predicts are all different. The LoRA mechanism is the same; the training context is not. | Section 5 (Explain) -- side-by-side ComparisonRow of LLM LoRA training step vs diffusion LoRA training step |
| "LoRA adapters are applied to every layer of the U-Net" | In Module 4.4, LoRA was applied to attention projections, and diffusion U-Nets have attention at multiple resolutions. Natural to assume blanket application. | Applying LoRA to conv layers (the majority of U-Net parameters) produces minimal improvement for style/subject adaptation because conv layers handle spatial processing, not text-to-image mapping. The cross-attention projections are where text meaning meets spatial features -- that is where style and subject live. | Section 5 (Explain) -- after introducing target layers, explain why cross-attention specifically |
| "You need hundreds or thousands of images to train a diffusion LoRA" | LLM finetuning datasets are large (Alpaca 52K, LIMA 1K). Natural to assume diffusion LoRA needs similar scale. | Subject LoRAs (a specific face, pet, object) routinely train on 5-20 images. The model already knows how to generate diverse images; the LoRA only needs to encode the delta to the specific subject. Overfitting (generating only the training images) is a bigger risk than underfitting with too few images. | Section 7 (Elaborate) -- practical training considerations |
| "Training a LoRA changes how the model denoises at all timesteps equally" | LoRA modifies projection matrices, which are the same weights at every timestep. Natural to assume uniform effect. | LoRA's effect is mediated through cross-attention, which has different influence at different noise levels. At high noise (t=900), spatial features are largely noise so cross-attention influence is limited. At low noise (t=50), spatial features are well-formed and cross-attention steers fine details -- this is where style LoRAs have their strongest effect. The student already knows this from the coarse-to-fine denoising mental model (6.3.1). | Section 7 (Elaborate) -- connecting to multi-resolution mental model |
| "You can just use your LLM LoRA knowledge and skip this lesson -- it is the same thing" | The mechanism IS the same. The student may feel overconfident. | Ask the student to predict: what is the loss function for diffusion LoRA training? What is the input data format? What does the model predict? If they answer "next-token cross-entropy on text" they are wrong -- it is "noise-prediction MSE on noised images." The mechanism transfers; the training context does not. | Section 4 (Hook) -- prediction checkpoint to calibrate confidence |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Style LoRA: training on 50-100 images of a specific art style (e.g., watercolor illustration) to generate new images in that style | Positive | Shows LoRA's primary use case for diffusion: style transfer. Demonstrates that the model learns a stylistic delta, not a new generative model. | Style adaptation is the most common LoRA application. Uses enough images to show it is not memorization. The result (generating novel subjects in the learned style) demonstrates that the base model's knowledge is preserved. |
| Subject LoRA: training on 10-15 images of a specific object or person to generate them in new contexts | Positive | Shows LoRA for identity preservation -- a different use case with different data requirements. | Contrasts with style LoRA: fewer images, different prompt structure (using a rare-token identifier like "sks"), different failure modes (face distortion vs style inconsistency). Demonstrates the versatility of the same mechanism. |
| Applying LoRA to conv layers only (no cross-attention targeting) for a style task | Negative | Shows why target layer selection matters -- the mechanism alone is not enough, the WHERE is critical. | Disproves the misconception that LoRA should be applied everywhere. Conv layers handle spatial processing; cross-attention layers handle text-to-image mapping. Style lives in the text-to-image mapping. |

---

## Phase 3: Design

### Narrative Arc

You have spent three modules building every piece of Stable Diffusion from scratch, and Module 6.4 showed you the assembled system generating images from text prompts. But the model only generates what it was trained on. If you want it to produce watercolor illustrations, or images of your specific pet, or a particular artistic style, you are out of luck -- the model never saw those during training. Full finetuning would require enormous compute and risks catastrophic forgetting. But you already know a technique designed for exactly this problem. In Module 4.4, you learned that LoRA injects tiny trainable detours alongside frozen weights, adapting a model with a fraction of the parameters. The mechanism transfers directly. What changes is the context: instead of adapting a language model for classification or instruction-following, you are adapting a diffusion model for a new visual style or subject. The key question this lesson answers is: where in the U-Net pipeline should those detours go, and how does the training loop change when your "tokens" are noised images instead of text?

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual/Diagram | SVG diagram showing the SD U-Net with LoRA bypass modules inserted at cross-attention projection matrices. Highlights which layers get LoRA (cross-attention W_Q, W_K, W_V, W_out) vs which stay frozen (conv blocks, self-attention, adaptive group norm). Color-coded: frozen=gray, LoRA-enabled=highlighted. | The "where" question is inherently spatial/architectural. The student needs to SEE where in the U-Net the adapters sit. Extends the U-Net architecture diagrams from 6.3.1 with a new overlay. |
| Symbolic/Code | Side-by-side training loop pseudocode: LLM LoRA training step (left) vs diffusion LoRA training step (right). Highlights the 3-4 lines that differ (data loading, forward pass input, loss function, what the model predicts). | The training loop difference is the core new content. Code makes the comparison precise and shows the student that 80% of the loop is identical. |
| Concrete example | Worked example: "Training a watercolor style LoRA." Trace one training step end-to-end: select a training image, encode with frozen VAE, sample random timestep t=500, noise the latent, predict noise with U-Net (only LoRA params have gradients), compute MSE loss, update LoRA params. Specific tensor shapes at each stage. | A concrete end-to-end trace grounds the abstract concept. The student has seen this pattern (full training step trace) in multiple prior lessons. The specific shapes make it verifiable. |
| Verbal/Analogy | "Same detour, different highway" -- the LoRA bypass is the same small trainable detour from Module 4.4. The highway it attaches to is different (diffusion U-Net cross-attention instead of LLM self-attention). The traffic on the highway is different (spatial features and text embeddings instead of token sequences). But the detour mechanism is identical. | Extends the established "highway and detour" analogy. Emphasizes transfer: same mechanism, new context. Prevents the misconception that diffusion LoRA is fundamentally different. |
| Intuitive | "Of course cross-attention is the target": style and subject are about how text concepts map to visual features. Cross-attention is WHERE text meets image in the U-Net. If you want to change what "watercolor" means visually, you change the projections that translate between text meaning and spatial features. The conv layers handle edges and textures at a low level -- they do not know about "watercolor." | Provides the "of course" moment. Connects the target layer choice to the student's deep understanding of cross-attention's role from 6.3.4. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. Where LoRA is applied in a diffusion U-Net and why (cross-attention projections as the text-to-image interface)
  2. The diffusion LoRA training loop (noise-prediction MSE with LoRA adapters as only trainable params)

  Supporting practical concepts (not conceptually new, but new application):
  - Training data requirements for style vs subject LoRA
  - Multiple LoRA composition
  - Rare-token identifiers for subject LoRA (e.g., "sks")

- **Previous lesson's load:** Module 6.4 Lesson 3 (generate-with-stable-diffusion) was CONSOLIDATE.

- **This lesson's load:** BUILD. The LoRA mechanism is at DEVELOPED depth from Module 4.4. The diffusion pipeline is at DEVELOPED from Module 6.4. This lesson connects two well-understood systems. The genuinely new content (target layer rationale and training loop adaptation) builds on strong foundations. Two new concepts is within the 2-3 limit.

- **Appropriate given trajectory:** Yes. The previous lesson was CONSOLIDATE (zero new concepts). BUILD is a comfortable step up. The student has high confidence from the Module 6.4 capstone. Starting a new module with BUILD (not STRETCH) respects the module transition.

### Connections to Prior Concepts

- **LoRA mechanism (4.4.4):** Direct transfer. "Same detour, different highway." The bypass architecture, B=0 initialization, merge-at-inference -- all identical. The student's LoRALinear implementation from the notebook applies unchanged.
- **Cross-attention projections (6.3.4):** The target layer rationale connects directly. W_Q projects spatial features (seeking lens), W_K/W_V project text embeddings (advertising/contributing lenses). LoRA on these projections modifies how text meaning maps to visual features -- exactly what style and subject adaptation require.
- **DDPM training loop (6.2.3):** The diffusion LoRA training loop IS the DDPM training loop with two modifications: (1) encode images with frozen VAE first (latent diffusion, from 6.3.5), (2) only LoRA adapter params have gradients (from 4.4.4). The student has both pieces.
- **Component modularity (6.4.1):** LoRA adapters are another manifestation of modularity. They are small, swappable, and can be composed. Multiple LoRA adapters on the same base model extend the "components connected by tensor handoffs" mental model.
- **"Finetuning is a refinement, not a revolution" (4.4.4):** Applies directly. Style/subject LoRA is a small adjustment in a subspace of the U-Net's weight space. The base model's general knowledge is preserved.

**Potentially misleading prior analogies:**
- The "highway and detour" analogy applies well but the student might over-extend it: in the LLM context, LoRA targeted self-attention projections (Q, V). In diffusion, the primary targets are cross-attention projections. Same mechanism, different layers. The analogy transfers; the specific layer names need updating.

### Scope Boundaries

**This lesson IS about:**
- Where LoRA adapters are placed in the diffusion U-Net and why cross-attention is the primary target
- How the diffusion LoRA training loop differs from LLM LoRA training
- Practical data requirements for style LoRA vs subject LoRA
- Multiple LoRA composition (applying two LoRAs simultaneously)
- Using diffusers + PEFT for practical LoRA training

**This lesson is NOT about:**
- Reimplementing LoRA from scratch (done in 4.4.4)
- The LoRA mathematical derivation or SVD (scoped out in 4.4.4 and stays out)
- DreamBooth (a different fine-tuning technique that modifies all U-Net weights)
- ControlNet or other structural conditioning (Series 7)
- Img2img or inpainting (Lesson 19)
- Textual inversion (Lesson 20)
- Hyperparameter optimization or training recipe details beyond practical guidance
- SD v1 vs v2 vs XL LoRA differences (different cross-attention dimensions but same principle)

**Target depth:** LoRA for diffusion at DEVELOPED. The student should be able to explain where adapters go, why, trace one training step, and use diffusers to train a LoRA. Not APPLIED because the notebook is Guided/Supported, not fully independent LoRA training from a blank slate.

### Lesson Outline

1. **Context + Constraints**
   - "You know LoRA from Module 4.4 (LLMs). You know the SD pipeline from Module 6.4. This lesson connects them: how to apply LoRA to the diffusion U-Net for style and subject customization."
   - Scope: NOT reimplementing LoRA. NOT DreamBooth. NOT full finetuning. Just LoRA applied to diffusion.

2. **Recap** (brief, focused)
   - Quick LoRA mechanism recap: highway + detour, B=0 init, merge at inference. One paragraph + the highway-detour diagram callback. Not re-teaching -- reactivating. LoRA was 14 lessons ago; per the Reinforcement Rule, assume it is fading. But it was at DEVELOPED, so a brief recap suffices.
   - Quick SD pipeline recap: text -> CLIP -> U-Net (cross-attention + adaptive norm) -> VAE decode. One diagram callback. Focus on where cross-attention sits.

3. **Hook** -- Prediction checkpoint
   - "You know LoRA. You know Stable Diffusion. Before we connect them, predict: (1) Which layers of the U-Net would you target with LoRA for a style adaptation task? (2) What is the loss function for diffusion LoRA training? (3) How many training images do you need?"
   - `<details>` reveals after each prediction. This calibrates confidence: most students will get #1 partially right (attention layers), #2 wrong (will guess cross-entropy or something text-like), and #3 wrong (will guess too many).

4. **Explain** -- Where LoRA goes in the U-Net
   - SVG diagram: U-Net architecture with LoRA bypass modules at cross-attention projections. Color-coded frozen (gray) vs LoRA-enabled (highlighted).
   - "Of course" intuition: style/subject = how text maps to visual features. Cross-attention = where text meets image. LoRA on cross-attention projections = modifying the text-to-image translation. Conv layers handle low-level spatial features (edges, textures) -- they do not know about "watercolor."
   - GradientCards: which layers get LoRA (W_Q, W_K, W_V, W_out in cross-attention blocks at attention resolutions 16x16 and 32x32) vs which stay frozen (conv blocks, self-attention, adaptive group norm, VAE, CLIP).
   - Negative example: applying LoRA to conv layers only. Minimal style effect because conv layers do not interact with text conditioning. The style concept must pass through cross-attention to influence generation.

5. **Explain** -- The diffusion LoRA training loop
   - Side-by-side ComparisonRow: LLM LoRA training step (5-6 lines) vs diffusion LoRA training step (5-6 lines). Highlight differences:
     - Data: text sequence vs (image, caption) pair
     - Preprocessing: tokenize text vs encode image with frozen VAE + sample timestep + add noise
     - Forward pass: predict next token vs predict noise
     - Loss: cross-entropy on token prediction vs MSE on noise prediction
     - What is identical: freeze base weights, only LoRA params in optimizer, same bypass mechanism
   - Concrete worked example: one training step for a watercolor style LoRA. Training image (512x512 watercolor painting) -> VAE encode -> z_0 [4, 64, 64] -> sample t=500 -> forward process: z_t = sqrt(alpha_bar_500) * z_0 + sqrt(1-alpha_bar_500) * epsilon -> caption "a watercolor painting of a village" -> CLIP encode -> U-Net predicts epsilon_hat (only LoRA params have gradients) -> loss = MSE(epsilon, epsilon_hat) -> backprop into LoRA params only -> optimizer step.
   - InsightBlock: "Same DDPM training. Same LoRA adapters. The only new thing is combining them."

6. **Check** -- Predict-and-verify
   - "If you trained a style LoRA on watercolor images with captions mentioning 'watercolor,' then generate with the prompt 'a photograph of a mountain,' what would you expect?" (`<details>` reveal: the style would likely still apply because the LoRA modified how the U-Net processes ALL text-to-image mappings, not just ones containing 'watercolor.' The cross-attention projections are used for every prompt.)
   - "What happens if you merge the LoRA adapters into the base weights (W_merged = W + BA*(alpha/r)) and then remove the adapters? Does inference speed change?" (`<details>` reveal: No. Merged weights are the same shape as original weights. Zero inference overhead. Same as LLM LoRA -- this transfers directly.)

7. **Elaborate** -- Practical considerations and nuance
   - **Style vs subject LoRA:** ComparisonRow. Style: 50-200 diverse images in the target style, varied subjects. Subject: 5-20 images of the specific subject in varied poses/lighting. Rare-token identifier (e.g., "a photo of sks dog") to avoid polluting common words.
   - **Rank and alpha for diffusion:** Typically r=4-8 (lower than LLM LoRA, which often uses r=8-64). Diffusion style adaptation is a smaller delta. Alpha typically 1.0 or equal to rank.
   - **Multiple LoRA composition:** Apply a style LoRA AND a subject LoRA simultaneously. Mechanically: sum the bypass outputs. W_combined = W + BA_style*(alpha/r) + BA_subject*(alpha/r). WarningBlock: compositions can interfere, especially if both modify the same projection in conflicting directions.
   - **Overfitting symptoms:** Generated images look like copies of the training images rather than novel compositions in the learned style. The model memorizes specific images instead of learning the style delta. Mitigation: fewer training steps, lower rank, more diverse training images.
   - **LoRA's effect across timesteps:** Connect to the multi-resolution mental model from 6.3.1. Cross-attention has different influence at different noise levels. Style LoRAs have their strongest visible effect at lower noise levels (fine details, textures) where cross-attention steers the rendering of well-formed spatial features.

8. **Check** -- Transfer question
   - "Imagine you want the model to generate images with a specific color palette (only blues and greens). Would LoRA on cross-attention projections be the best approach, or would you target different layers? Why?" (`<details>` reveal: color palette is arguably a lower-level visual property. Cross-attention LoRA would work if the captions describe colors, but targeting the decoder-side conv layers might be more direct for pure color shifts. This is a genuine edge case where the "of course cross-attention" heuristic meets its boundary.)
   - "Can you train a diffusion LoRA using the LLM LoRA training loop (next-token prediction on text)? Why or why not?" (`<details>` reveal: No. The U-Net does not predict text tokens. It predicts noise vectors. The loss function, input format, and output format are all different. The LoRA mechanism transfers; the training procedure does not.)

9. **Practice** -- Notebook exercises (Colab)
   - **Exercise 1 (Guided):** Inspect LoRA target layers. Load a pretrained SD model, list all cross-attention projection layers by name, compute total LoRA params for rank-4 vs rank-16, compare to total U-Net params. Purpose: ground the architectural understanding in real model inspection.
   - **Exercise 2 (Guided):** One LoRA training step by hand. Load a pretrained SD pipeline, encode one image with VAE, sample a timestep, add noise with forward process, run U-Net forward pass with LoRA adapters, compute MSE loss, print gradients. Purpose: verify the training loop from the lesson with real tensors.
   - **Exercise 3 (Supported):** Train a style LoRA. Use diffusers + PEFT to train a LoRA on a small style dataset (~20-50 images). Train for a few hundred steps. Generate images with and without the LoRA to see the style effect. Compare different rank values. Purpose: end-to-end LoRA training workflow.
   - **Exercise 4 (Independent):** LoRA composition experiment. Load two pre-trained LoRA adapters (provided), apply them individually and together. Compare outputs. Experiment with scaling the alpha of each LoRA to control the blend. Purpose: explore composition, a practical skill not fully guided.
   - Exercises are cumulative: Ex1 builds understanding of the architecture, Ex2 traces one step, Ex3 trains end-to-end, Ex4 explores advanced usage.
   - Key reasoning to emphasize in solutions: WHY cross-attention layers (not just which), gradient flow verification (base weights should have no gradients), and the effect of rank on output quality vs overfitting.

10. **Summarize**
    - Three key takeaways: (1) LoRA for diffusion uses the same bypass mechanism as LoRA for LLMs -- the detour is the same, the highway is different. (2) Cross-attention projections are the primary target because that is where text meaning meets spatial features -- where style and subject live. (3) The training loop is DDPM training with frozen base weights and LoRA adapters as the only trainable parameters.
    - Mental model echo: "Same detour, different highway."

11. **Next step**
    - Preview img2img and inpainting: "LoRA customizes the model's weights. But you can also customize the inference process itself. What if you started from a real image instead of pure noise? What if you could selectively edit just part of an image? That is next."

---

## Widget Assessment

**Widget needed:** No custom interactive widget required.

The lesson's core visual is an SVG diagram of the U-Net with LoRA bypass modules at cross-attention projections (static, not interactive). The training loop comparison is best served by a ComparisonRow (existing component). The worked example is text + code. No concept in this lesson benefits from interactive manipulation -- the LoRA mechanism was already explored interactively in Module 4.4, and re-implementing it for diffusion would duplicate rather than extend that experience.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (2 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: MAJOR REVISION

Critical finding in the notebook requires a fix before the lesson is usable. The lesson `.tsx` itself is strong.

### Findings

### [CRITICAL] -- Notebook Exercise 3 uses meaningless synthetic training data

**Location:** Notebook cell-21 (synthetic dataset creation) and cell-23 (training loop)
**Issue:** Exercise 3 creates "training images" by generating random numpy arrays with a warm color bias. These are random noise with slightly shifted RGB channels, not images with any coherent visual style. Training a LoRA on random noise does not demonstrate style adaptation -- the LoRA would learn to reproduce noise patterns, not a "warm style." The planning document specifies "Use diffusers + PEFT to train a LoRA on a small style dataset (~20-50 images)" and the lesson component's exercise description says "Use diffusers + PEFT to train a LoRA on a small style dataset (~20-50 images) for a few hundred steps. Generate with and without the LoRA to see the effect."
**Student impact:** The student completes the training loop mechanically but never sees a meaningful style effect. The before/after comparison would show minimal or incoherent differences, undermining the lesson's core claim that LoRA can adapt style. The student may conclude that LoRA does not work well, or that the effect is always subtle -- both wrong conclusions. This defeats the purpose of the exercise, which is to ground the architectural understanding in a visible, convincing result.
**Suggested fix:** Use a real small dataset from Hugging Face (e.g., `lambdalabs/naruto-blip-captions` which has ~1000 anime-style images with captions, or a curated subset of an art dataset). Alternatively, use `datasets.load_dataset()` to pull a small style-consistent image set. The dataset should have a visually distinctive style so the before/after comparison shows a clear effect. If dataset size or download time is a concern, use a subset of 20-50 images.

### [IMPROVEMENT] -- Notebook VRAM management risks OOM on free-tier Colab

**Location:** Notebook cells 4, 9, 20 (three separate model loads)
**Issue:** The notebook loads the full SD pipeline in the shared helpers (cell-4), then loads all components individually in float32 for Exercise 2 (cell-9), and again for Exercise 3 (cell-20). This means having multiple copies of the U-Net (~860M params) in VRAM simultaneously. Even with the cleanup cells (cell-18, cell-27), peak VRAM usage during transitions could exceed 16GB (T4 free-tier Colab). The shared pipeline from cell-4 is only used for Exercise 1's `pipe.unet` parameter inspection, which does not require the full pipeline (just the U-Net).
**Student impact:** Students on free-tier Colab T4 GPUs may encounter OOM errors. This creates a frustrating debugging experience unrelated to the lesson content and could prevent them from completing Exercises 2-4.
**Suggested fix:** For Exercise 1, load only the U-Net (not the full pipeline) using `UNet2DConditionModel.from_pretrained()`. Remove the shared pipeline load entirely. Each exercise should load only what it needs and clean up before the next. Add a VRAM usage tip noting that Exercise 3 requires the most memory and suggesting runtime restart if OOM occurs.

### [IMPROVEMENT] -- Notebook Exercise 4 relies on fragile external LoRA repos

**Location:** Notebook cell-28 and cell-29 (Exercise 4 instructions and starter code)
**Issue:** The exercise asks students to search Hugging Face for community LoRA adapters compatible with SD v1.5. No specific working repos are provided (the solution uses placeholders like `YOUR_FIRST_LORA_REPO`). SD v1.5 is becoming less common as SDXL and newer models dominate, so compatible LoRAs may become harder to find. The exercise also suggests `nerijs/pixel-art-xl` which is explicitly for SDXL, not SD v1.5.
**Student impact:** The student spends time searching for compatible LoRAs instead of learning about composition. They may download incompatible LoRAs (wrong SD version) and encounter cryptic dimension mismatch errors. The exercise becomes a scavenger hunt rather than a learning experience.
**Suggested fix:** Either (a) provide two specific, well-known SD v1.5 LoRA repos that are likely to remain available (e.g., from the diffusers examples or official HuggingFace repos), or (b) train two small LoRAs earlier in the notebook (one from Exercise 3's style dataset, one from a different dataset) and compose those. Option (b) is more self-contained and eliminates the external dependency entirely.

### [IMPROVEMENT] -- Notebook Exercise 4 solution does not include reasoning before code

**Location:** Notebook cell-30 (Exercise 4 solution `<details>` block)
**Issue:** The solution jumps straight into a long code block. The planning document and the review skill both specify that solutions should "include reasoning before code." While there is a brief sentence before the code and good observations at the end, the solution does not walk through the reasoning of *why* you would structure the experiment this way (e.g., why same seed matters, why you test individual adapters before composition, why the weight grid matters).
**Student impact:** A student who is stuck gets a code dump rather than a reasoning scaffold. They can copy-paste and see results but may not understand the experimental design logic.
**Suggested fix:** Add 3-4 sentences before the code block explaining the experimental design: (1) use the same seed for all comparisons so z_T is identical, (2) test each adapter individually first to establish baselines, (3) then test composition at different weight ratios to see how the blend changes, (4) the weight grid (1.0/1.0 vs 0.5/0.5 vs asymmetric) reveals whether composition is smooth or one-LoRA-dominant.

### [POLISH] -- Mermaid diagram uses emoji for frozen/LoRA indicators

**Location:** Lesson Section 5 (Where LoRA Goes), Mermaid diagram around line 254-277
**Issue:** The Mermaid diagram uses emoji characters (snowflake for frozen, fire for LoRA-enabled) in the node labels. While semantically appropriate, emoji rendering in Mermaid diagrams can be inconsistent across browsers and operating systems, potentially showing as empty boxes or misaligned text on some platforms.
**Student impact:** Minor visual inconsistency on some platforms. The diagram would still be understandable from the color coding and text labels alone.
**Suggested fix:** Replace emoji with text labels: "W_Q + LoRA" with `classDef lora` styling is already sufficient to convey LoRA-enabled status. The frozen snowflake could be replaced with "(frozen)" text. Low priority since color coding carries the primary information.

### [POLISH] -- Spaced em dashes in CodeBlock Python comments

**Location:** Lesson lines 462, 465 (inside the CodeBlock for the worked example)
**Issue:** Two Python comments inside the CodeBlock use spaced em dashes: `# 7. Compute loss — same MSE from DDPM training` and `# 8. Backprop — gradients flow only through LoRA adapters`. The writing style rule specifies no spaces around em dashes.
**Student impact:** Negligible. These are code comments inside a syntax-highlighted code block, not lesson prose. The student reads them as code annotations.
**Suggested fix:** Replace with unspaced em dashes or with regular dashes. Low priority since they are in code comments, not lesson text.

### Review Notes

**What works well:**
- The lesson is exceptionally well-connected to prior knowledge. Every new concept is explicitly linked to something the student already knows, with specific lesson names cited. The "Same detour, different highway" framing is clean and memorable.
- The prediction checkpoint (hook) is one of the best hook designs in the course. It calibrates confidence by asking the student to predict answers they will likely get partially wrong, creating genuine curiosity.
- The ComparisonRow of LLM LoRA vs diffusion LoRA training steps is effective. The "80% identical" framing is reassuring and accurate.
- The negative example (conv-only LoRA) with the "steering wheel in the trunk" analogy is vivid and defines a clear boundary.
- The worked example with tensor shapes at every stage is concrete and verifiable.
- The lesson component follows all structural patterns correctly (Row layout, block components, section progression).

**What needs fixing:**
- The notebook is the weak point. Exercise 3's synthetic data is the critical issue -- the student needs to see a real style effect to believe the lesson's claims. Exercise 4's reliance on external repos adds fragility.
- The notebook VRAM management needs attention to avoid OOM frustrating the learning experience.

**Pattern observation:**
- The lesson `.tsx` is at a higher quality level than the notebook. The lesson was clearly designed with pedagogical care, while the notebook made pragmatic shortcuts (synthetic data, placeholder repos) that undermine the learning objectives. This suggests the notebook may have been built under time/complexity pressure and would benefit from a focused revision pass.
