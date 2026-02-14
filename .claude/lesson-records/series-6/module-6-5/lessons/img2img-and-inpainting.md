# Lesson: Img2img & Inpainting

**Module:** 6.5 Customization
**Position:** Lesson 2 of 3
**Slug:** `img2img-and-inpainting`

---

## Phase 1: Orient -- Student State

The student arrives fresh from the LoRA fine-tuning lesson, where they applied a known training technique (LoRA) to the diffusion U-Net. They have deep understanding of the full Stable Diffusion pipeline, the forward process closed-form formula, the denoising/sampling loop, and the VAE encoder-decoder. Critically, this lesson requires NO training--it is entirely about inference-time modifications to processes the student already understands well.

**Relevant concepts the student has:**

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Forward process closed-form: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon | DEVELOPED | the-forward-process (6.2.2) | Student derived this step by step, verified via 1D pixel walkthrough, and used it in the build-a-diffusion-model capstone. Also used in LoRA training loop (noising images before U-Net prediction). This is the core mechanism behind img2img. |
| DDPM sampling algorithm (full reverse loop from x_T to x_0) | APPLIED | build-a-diffusion-model (6.2.5) | Student implemented the reverse loop from scratch in PyTorch. Understands that denoising starts from pure noise z_T and iterates to z_0. |
| DDIM predict-x0-then-jump mechanism | DEVELOPED | samplers-and-efficiency (6.4.2) | Student understands DDIM's two-step: predict x_hat_0 from noise prediction, then leap to arbitrary target timestep. Understands why DDIM can skip timesteps while DDPM cannot. |
| Sampler as inference-time choice (swappable, no retraining) | DEVELOPED | samplers-and-efficiency (6.4.2) | Foundational insight: the model predicts noise, the sampler decides what to do with the prediction. Student has swapped samplers in notebook exercises. |
| Full SD pipeline data flow (text -> CLIP -> U-Net denoising loop with CFG -> VAE decode) | DEVELOPED | stable-diffusion-architecture (6.4.1) | Complete pipeline traced with tensor shapes at every handoff. Student knows exactly where the denoising loop sits and what feeds into it. |
| VAE encoder (image -> latent representation) and decoder (latent -> image) | DEVELOPED | from-pixels-to-latents (6.3.5), autoencoders (6.1.2) | Student built autoencoders from scratch in 6.1 and understands the frozen-VAE pattern in SD. Knows the VAE encoder compresses 512x512x3 to 64x64x4. Critical: the student knows the VAE encoder exists but was told it is NOT used during text-to-image inference (Misconception #4 from 6.4.1). Img2img DOES use the VAE encoder--this is a key new point. |
| Frozen-VAE pattern (VAE trained separately, frozen during diffusion) | INTRODUCED | from-pixels-to-latents (6.3.5) | VAE is a fixed translator between pixel space and latent space. The diffusion model operates entirely in latent space. |
| Coarse-to-fine denoising progression (early steps = structure, late steps = details) | DEVELOPED | sampling-and-generation (6.2.4), unet-architecture (6.3.1) | Established via DenoisingTrajectoryWidget. At t=900 model makes bold structural decisions; at t=50 it polishes textures. This is the key intuition behind the strength parameter in img2img. |
| Alpha-bar as signal-to-noise dial | DEVELOPED | the-forward-process (6.2.2) | Alpha_bar_t starts near 1 (clean) and drops to near 0 (pure noise). Student interacted with AlphaBarCurveWidget. This concept maps directly to the img2img strength parameter. |
| Noise schedule (beta_t, alpha_t, alpha_bar_t) | DEVELOPED | the-forward-process (6.2.2) | Student understands the noise schedule as a design choice and has computed alpha_bar values. |
| Diffusers API parameter comprehension | APPLIED | generate-with-stable-diffusion (6.4.3) | Student used every pipe() parameter with understanding. "The API is a dashboard. You built the machine behind it." |
| Latent representation is not a small image (64x64x4 is abstract learned representation) | INTRODUCED | from-pixels-to-latents (6.3.5) | Student knows the latent tensor has no direct visual interpretation. The decoder is essential. |
| Component modularity (independently trained models, swappable via tensor interfaces) | INTRODUCED | stable-diffusion-architecture (6.4.1) | Three translators, one pipeline. |
| LoRA placement in diffusion U-Net (cross-attention projections) | DEVELOPED | lora-finetuning (6.5.1) | Just taught. LoRA modifies the model's weights. Img2img and inpainting do NOT modify weights--purely inference-time. This contrast is pedagogically valuable. |
| Classifier-free guidance (CFG) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Two forward passes per step, amplifying text direction. The student understands CFG operates at every denoising step, which is relevant because img2img and inpainting use CFG exactly as before. |

**Mental models and analogies already established:**
- "Alpha-bar is the signal-to-noise dial" (forward process)
- "The model defines where to go. The sampler defines how to get there." (samplers)
- "Predict and leap" (DDIM mechanism)
- "Early steps create structure, late steps refine details" (coarse-to-fine)
- "The VAE is a translator between pixel language and latent language" (latent diffusion)
- "The API is a dashboard. You built the machine behind it." (diffusers comprehension)
- "Same detour, different highway" (LoRA--just taught, contrasts with this lesson)
- "Three translators, one pipeline" (SD modularity)

**What was explicitly NOT covered that is relevant:**
- Img2img (deferred from 6.4.1 and 6.5.1, noted as "Module 6.5 Lesson 2")
- Inpainting (deferred alongside img2img)
- VAE encoder use during inference (explicitly told it is NOT used in text-to-image; img2img changes this)
- Starting the denoising process from a partially noised real image (always started from pure noise z_T)
- Spatial masking of latents during denoising
- Strength parameter as mapping to a starting timestep

**Readiness assessment:** Excellent. The student has the forward process closed-form at DEVELOPED (the exact formula used by img2img to noise the input image), the sampling algorithm at APPLIED (they built the denoising loop), coarse-to-fine intuition at DEVELOPED (the foundation for understanding the strength parameter), and the VAE encoder-decoder at DEVELOPED. Every piece needed for img2img and inpainting is already in the student's repertoire at sufficient depth. This lesson reconfigures existing knowledge--no fundamentally new algorithms.

---

## Phase 2: Analyze

### Target Concept

**This lesson teaches the student to understand how img2img and inpainting modify the standard Stable Diffusion inference process--img2img by starting the denoising loop from a noised real image instead of pure noise, and inpainting by applying a spatial mask that preserves original latents in unmasked regions at each denoising step.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Forward process closed-form (x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon) | DEVELOPED | DEVELOPED | the-forward-process (6.2.2) | OK | Img2img uses this formula to noise the input image to a specific timestep. Student must reason about the formula, not just recognize it. DEVELOPED achieved. |
| DDPM/DDIM sampling algorithm (reverse denoising loop) | DEVELOPED | APPLIED | build-a-diffusion-model (6.2.5), samplers-and-efficiency (6.4.2) | OK | Student needs to understand the loop well enough to see where img2img modifies it (starting point) and where inpainting modifies it (per-step masking). APPLIED exceeds DEVELOPED. |
| Coarse-to-fine denoising progression | DEVELOPED | DEVELOPED | sampling-and-generation (6.2.4) | OK | The strength parameter's effect on output is only intuitive if the student understands that early steps set structure and late steps refine details. Skipping early steps means keeping the original structure. |
| Alpha-bar (cumulative signal fraction) | DEVELOPED | DEVELOPED | the-forward-process (6.2.2) | OK | The strength parameter maps to a starting alpha_bar value. Student needs to reason about what alpha_bar_t=0.8 vs alpha_bar_t=0.2 means for signal preservation. |
| VAE encoder and decoder | INTRODUCED | DEVELOPED | from-pixels-to-latents (6.3.5), autoencoders (6.1.2) | OK | Img2img uses the VAE encoder to convert the input image to latent space. Student has the VAE at DEVELOPED from building it in 6.1 and understanding SD's VAE in 6.3.5. Exceeds requirement. |
| Full SD pipeline data flow | DEVELOPED | DEVELOPED | stable-diffusion-architecture (6.4.1) | OK | Student must locate exactly where img2img changes the pipeline (starting point of the denoising loop) and where inpainting adds a step (per-step mask application). |
| Diffusers API | APPLIED | APPLIED | generate-with-stable-diffusion (6.4.3) | OK | Notebook will use diffusers pipelines for img2img and inpainting. Student already comfortable with the API. |
| Latent diffusion (diffusion in latent space, not pixel space) | DEVELOPED | DEVELOPED | from-pixels-to-latents (6.3.5) | OK | Both img2img and inpainting operate in latent space. Student needs to know that the input image is encoded to latents, noised in latent space, and denoised in latent space. |
| CFG (classifier-free guidance) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | CFG applies identically in img2img and inpainting. No change needed in student's understanding. |
| Sampler as inference-time choice | DEVELOPED | DEVELOPED | samplers-and-efficiency (6.4.2) | OK | Img2img and inpainting work with any sampler. The choice of sampler is orthogonal to the img2img/inpainting modifications. |

**All prerequisites are OK. No gaps to resolve.**

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Img2img modifies the original image directly (like a Photoshop filter)" | The name "image-to-image" implies a direct transformation. Students familiar with image editing think of pixel-level operations. Prior experience with image filters in Series 3 (CNNs) reinforces the idea of operating on the input. | Img2img with strength=1.0: the original image is completely destroyed (noised to pure noise) and the output bears no structural resemblance to the input, only following the text prompt. A Photoshop filter would always preserve the input structure. At strength=1.0, img2img IS standard text-to-image generation--the input image is irrelevant. | Section 5 (Explain)--immediately after introducing the mechanism, show the strength=1.0 case as the boundary where img2img collapses to txt2img |
| "Img2img somehow blends the original image with the generated image at the pixel level" | "Image blending" is a common concept (alpha compositing, layer opacity). Students may think img2img takes the generated result and merges it with the original using some blending weight. | If blending were the mechanism, strength=0.5 would produce a ghostly double exposure (half original, half generated)--the same problem as pixel-space interpolation from exploring-latent-spaces (6.1.4). Instead, img2img at strength=0.5 produces a single coherent image that shares the original's broad structure but has been creatively reinterpreted. The mechanism is denoising, not blending. | Section 5 (Explain)--connect to the pixel-vs-latent interpolation lesson (6.1.4) where ghostly overlays were the negative example |
| "Inpainting uses a separate, specialized model" | Inpainting sounds like a specialized capability. The student may assume it requires a different architecture or training procedure (like how LoRA required a training loop). Community discussions often mention "inpainting models." | Standard inpainting uses the exact same U-Net, VAE, and CLIP from text-to-image. The mask simply controls which latent regions are denoised vs preserved at each step. No additional training, no architectural changes. (Specialized inpainting models exist as fine-tuned variants with an extra input channel for the mask, but the core mechanism works without them.) | Section 7 (Elaborate)--after explaining the masking mechanism, note the existence of specialized models but clarify they are optimizations, not requirements |
| "The strength parameter linearly scales how much the output resembles the input" | "Strength" suggests a linear control. If strength=0.5, the output should be "50% like the original." This linearity assumption is natural but wrong. | Due to the coarse-to-fine nature of denoising, the effect is highly nonlinear. Strength=0.8 allows structural changes (the model can decide "this is a cat, not a dog"). Strength=0.3 preserves all structure and only changes textures/colors. The jump from 0.3 to 0.5 is qualitatively different from the jump from 0.7 to 0.9 because different denoising phases control different aspects. | Section 5 (Explain)--after presenting the mechanism, use coarse-to-fine mental model to explain nonlinearity, with specific examples at 3-4 strength values |
| "Inpainting boundaries will show visible seams between the masked and unmasked regions" | In traditional image editing, pasting a generated region into an existing image creates visible seams (color mismatches, edge artifacts). The student may assume inpainting has the same problem. | The denoising process naturally handles boundary transitions. At each step, the model sees the full latent (both denoised masked region and original unmasked region) and its predictions account for the context. The U-Net's receptive field spans the entire image at low resolutions, so the model "knows" what surrounds the masked area. This is fundamentally different from cut-and-paste. | Section 6 (Explain inpainting)--demonstrate why seamless boundaries are a natural consequence of the denoising process and the U-Net's receptive field |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Img2img at multiple strength values (0.2, 0.5, 0.8, 1.0) on the same input image with the same prompt | Positive | Shows the full spectrum of img2img behavior--from subtle style transfer (0.2) to major reinterpretation (0.8) to complete regeneration (1.0). Makes the strength parameter tangible. | A single input processed at multiple strengths is the clearest way to demonstrate the mechanism. The 1.0 boundary case explicitly connects to standard text-to-image (no input image influence at all). The student can reason about each case using the coarse-to-fine mental model. |
| Img2img with strength=1.0 (collapses to standard text-to-image) | Negative (boundary case) | Disproves the "img2img modifies the image" misconception by showing that at maximum strength, the input image has zero influence. Defines the boundary of img2img behavior. | The strength=1.0 case is the purest negative example: if img2img were "modifying" the image, then even at max strength some input influence should remain. Instead, the output is indistinguishable from txt2img because the image is fully noised to random noise before denoising begins. |
| Inpainting: selectively replacing one object in a scene while preserving the rest | Positive | Shows the core inpainting use case. A mask covers one region (e.g., replace a vase with a plant), the prompt describes the desired content, and the result seamlessly integrates the new object with the preserved background. | Object replacement is the most common and impressive inpainting application. It demonstrates spatial selectivity (only the masked region changes), boundary blending (no seams), and text-guided generation (the prompt controls what fills the masked area). |
| Inpainting with the mask covering the entire image (collapses to standard img2img/txt2img) | Negative (boundary case) | Defines the boundary: when mask=1 everywhere, every latent is denoised and none are preserved, so inpainting degenerates to standard generation. Shows that masking is the only difference between inpainting and standard denoising. | The full-mask boundary case makes the mechanism crystalline: inpainting IS standard denoising plus a per-step mask. Remove the mask, and you get standard generation. The mask is the entire mechanism. |
| Img2img on a hand-drawn sketch with a descriptive prompt (sketch-to-image) | Positive (stretch) | Shows a practical, visually impressive application that demonstrates the power of partial denoising. A rough sketch provides structure; the model fills in realistic detail guided by the text prompt. | Sketch-to-image is the "wow" application of img2img. It demonstrates that the input does not need to be a photograph--any image providing structural guidance works. This extends the student's understanding beyond "editing photos" to "using images as compositional scaffolding." |

---

## Phase 3: Design

### Narrative Arc

The previous lesson showed how to customize what the model knows by modifying its weights with LoRA. But LoRA requires training data and a training loop. What if you want to work with a specific existing image--keep its composition but change its style, or edit just one part of it? You do not need to retrain anything. The denoising process you already understand completely contains the answer. Every time you generate an image, the model starts from pure noise and iteratively denoises to an image. But there is nothing sacred about starting from pure noise. What if you started from a noised version of a real image? The forward process formula you derived in Module 6.2 lets you add a precise amount of noise to any image. Feed that partially noised image into the denoising loop partway through, and the model completes the denoising--preserving the original's structure where the noise is light, reimagining it where the noise is heavy. That is img2img. And once you see that, inpainting is one more step: what if you only denoise part of the image and preserve the rest? A spatial mask at each denoising step controls which regions the model can change. No new math, no new training, no new architecture--just two clever reconfigurations of the inference process you already know.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual/Diagram | Side-by-side pipeline comparison diagram: standard txt2img pipeline (start from z_T ~ N(0,I)) vs img2img pipeline (start from z_{t_start} = noise(encode(image))). Highlight the one difference: the starting point of the denoising loop. A second diagram for inpainting showing the per-step mask application: at each denoising step, masked regions use the model's prediction while unmasked regions are replaced with the original latent (re-noised to the current timestep). | The "where does this change the pipeline" question is inherently visual. The student needs to see that img2img changes exactly ONE thing (the starting point) and inpainting adds exactly ONE thing (per-step masking). Pipeline diagrams make these minimal modifications visually obvious against the familiar pipeline structure. |
| Concrete example | Worked example for img2img: take a photo (512x512), encode with VAE to z_0 [4, 64, 64], set strength=0.7 (meaning start at 70% of the noise schedule, so t_start = 0.7 * T), noise with forward process formula z_{t_start} = sqrt(alpha_bar_{t_start}) * z_0 + sqrt(1-alpha_bar_{t_start}) * epsilon, then denoise from t_start to t=0 using the standard sampler. Specific alpha_bar values and step counts at each strength value. | A concrete end-to-end trace grounds the abstract mechanism. The student has seen this pattern (full step trace with tensor shapes) in every prior lesson. Using specific alpha_bar values makes it verifiable and connects to the forward process lesson where they computed these values. |
| Verbal/Analogy | "Starting mid-hike" analogy: standard txt2img is hiking from the summit (pure noise) to the valley (clean image). Img2img is starting partway down--the input image determines the trail, and the model completes the descent. Low strength means starting near the valley (minor adjustments); high strength means starting near the summit (major creative freedom). Inpainting is hiking with some valleys already filled in--you only hike the unfilled sections. | Extends the existing hiking analogy from sampling-and-generation (6.2.4) where the model was described as a hiker with a compass. Now the starting point changes. The student already has this spatial metaphor for the denoising trajectory. |
| Symbolic/Code | Pseudocode showing the img2img modification as a 3-line change to the standard inference loop: (1) z_0 = vae.encode(image), (2) z_{t_start} = forward_process(z_0, t_start), (3) change loop start from T to t_start. For inpainting: add one line inside the loop (z_t = mask * z_t_denoised + (1-mask) * forward_process(z_0_original, t)). | Code makes the minimal nature of the change unmistakable. The student should see that img2img is literally 3 lines different from txt2img, and inpainting adds 1 line to img2img. This prevents the misconception that these are fundamentally different algorithms. |
| Intuitive/Geometric | The strength parameter mapped onto the alpha-bar curve the student already explored with the AlphaBarCurveWidget. Strength=0.3 means starting at a point where alpha_bar is still high (mostly signal, little noise)--the original image's structure is strongly preserved. Strength=0.9 means starting where alpha_bar is low (mostly noise)--the original is almost completely destroyed. The nonlinear alpha-bar curve explains why strength does not behave linearly. | The alpha-bar curve is one of the student's most interactive prior experiences (dragging along the curve, seeing images at different noise levels). Mapping the strength parameter onto this curve connects the new concept to a vivid, practiced prior experience. The nonlinearity of alpha_bar directly explains the nonlinearity of the strength parameter. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. Img2img as partial denoising (start the denoising loop from a noised real image instead of pure noise; strength parameter as mapping to a starting timestep)
  2. Inpainting as per-step spatial masking (apply a binary mask at each denoising step to control which regions are denoised vs preserved)

  Supporting concepts (not conceptually new, but new applications):
  - VAE encoder used during inference (previously told it was NOT used in txt2img; img2img activates it)
  - The strength parameter's nonlinear behavior (emerges from the alpha-bar curve they already know)
  - Boundary blending in inpainting (emerges from U-Net receptive field they already know)

- **Previous lesson's load:** LoRA fine-tuning was BUILD (2 new concepts: LoRA placement, diffusion LoRA training loop).

- **This lesson's load:** BUILD. Both new concepts are reconfigurations of familiar mechanisms (the forward process formula and the denoising loop), not fundamentally new algorithms. The student has all the pieces; this lesson shows them how to rearrange those pieces. Two new concepts is within the 2-3 limit.

- **Appropriate given trajectory:** Yes. The module plan specifies BUILD -> BUILD -> STRETCH. Two consecutive BUILD lessons is appropriate here because they are genuinely different types: Lesson 1 (LoRA) applied a training technique to diffusion; Lesson 2 (img2img/inpainting) reconfigures the inference process. The STRETCH lesson on textual inversion follows.

### Connections to Prior Concepts

- **Forward process closed-form (6.2.2):** Direct application. The formula x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon is exactly what img2img uses to noise the input image to the starting timestep. The student derived this formula, verified it, and used it in the capstone and LoRA training. Now it appears in a third context: inference-time image editing.
- **Alpha-bar curve (6.2.2):** The strength parameter maps directly onto the alpha_bar curve the student explored interactively. Strength=0.3 corresponds to a low noise level (high alpha_bar); strength=0.9 corresponds to a high noise level (low alpha_bar). The nonlinearity of the strength parameter's effect is the nonlinearity of the alpha-bar curve.
- **Coarse-to-fine denoising (6.2.4, 6.3.1):** Explains why the strength parameter has qualitatively different effects at different values. Low strength: only the detail-refinement steps run, so only textures/colors change. High strength: the structure-creation steps run, so the model can reimagine the entire composition.
- **VAE encoder/decoder (6.1.2, 6.3.5):** Img2img reactivates the VAE encoder, which the student built in 6.1 but was told is NOT used during txt2img inference (Misconception #4 in 6.4.1). This is a satisfying "now it IS used" callback.
- **Pixel-space vs latent-space interpolation (6.1.4):** The "blending" misconception about img2img directly parallels the pixel-vs-latent interpolation lesson. Pixel blending produces ghostly overlays; latent-space denoising produces coherent images. Same lesson, new context.
- **Sampling algorithm (6.2.4, 6.2.5, 6.4.2):** The denoising loop is unchanged. The only modification is the starting point (img2img) or a per-step mask (inpainting). The sampler (DDIM, DPM-Solver, etc.) works identically.
- **U-Net receptive field (6.3.1):** At low resolutions (8x8 bottleneck), each latent pixel has global receptive field. This explains why inpainting boundaries blend seamlessly--the U-Net "sees" the entire image, including unmasked context, when predicting noise for the masked region.
- **LoRA (6.5.1):** Contrast. LoRA modifies the model's weights (training required). Img2img/inpainting modify the inference process (no training). Together with textual inversion (next lesson), these form three fundamentally different customization strategies: change the weights, change the inference, change the embeddings.

**Potentially misleading prior analogies:**
- "The VAE encoder is NOT used during inference" (Misconception #4, 6.4.1): This was correct for txt2img but wrong for img2img. The lesson must explicitly address this transition: "We told you the encoder is not used during text-to-image. That was correct. Img2img is not text-to-image--it is image-to-image, and it needs the encoder to convert the input to latent space."

### Scope Boundaries

**This lesson IS about:**
- Img2img: starting the denoising loop from a noised real image instead of pure noise
- The strength parameter: how it maps to a starting timestep and why its effect is nonlinear
- Inpainting: per-step spatial masking to selectively edit regions
- Why inpainting boundaries blend seamlessly (U-Net receptive field, denoising process)
- Practical use with diffusers (StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline)
- The connection between img2img/inpainting and the forward process / sampling formulas

**This lesson is NOT about:**
- Training any model (img2img and inpainting are inference-time only)
- LoRA fine-tuning (Lesson 1 of this module)
- Textual inversion (Lesson 3 of this module)
- ControlNet or structural conditioning (Series 7)
- Specialized inpainting models with extra input channels (mentioned for context, not developed)
- Outpainting / image extension (mentioned as a natural extension, not developed)
- SDEdit or other img2img variants beyond the standard noise-and-denoise approach
- Depth-guided or edge-guided generation (Series 7 / ControlNet territory)
- Img2img with specific negative prompt strategies or prompt engineering
- Video inpainting

**Target depth:** Img2img at DEVELOPED. Inpainting at DEVELOPED. The student should be able to explain the mechanism of each, trace the modification to the standard pipeline, predict the effect of the strength parameter using the alpha-bar curve, and use diffusers to run both. Not APPLIED because the notebook exercises are Guided/Supported, not fully independent pipeline design from scratch.

### Lesson Outline

1. **Context + Constraints**
   - "You know how to customize the model's weights (LoRA). This lesson customizes the inference process instead. No training required--everything happens at generation time."
   - Scope: NOT ControlNet. NOT training. NOT specialized inpainting models. Just two modifications to the denoising process you already know.

2. **Recap** (brief, focused)
   - Quick reactivation of the forward process closed-form formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon. One paragraph + formula callback. Not re-teaching--reactivating. The student derived this in 6.2.2 and used it in the 6.2.5 capstone and 6.5.1 LoRA training, but the specific "noise a real image to an arbitrary timestep" framing has not been used since 6.2.
   - Quick reactivation of the denoising loop: start from z_T, iterate to z_0. One sentence referencing the sampling algorithm from 6.2.4. The student implemented this.
   - Quick reactivation of the alpha-bar curve meaning: alpha_bar near 1 = mostly signal; near 0 = mostly noise. One sentence.

3. **Hook** -- "What if you didn't start from pure noise?"
   - Type: Challenge/puzzle. Present the student with a scenario: "You have a photo of a landscape. You want to turn it into a watercolor painting of the same scene--same composition, same mountains, same sky, but in watercolor style. You could train a LoRA on watercolor images. Or... you could do it right now, with zero training, using only the concepts you already have."
   - Pause for thinking. Then the reveal: "You know how to add a precise amount of noise to any image (the forward process formula). You know the model can denoise from any starting noise level. What if you noise your landscape photo partway, then let the model denoise with the prompt 'a watercolor painting of mountains'?"
   - This is a predict-and-reveal hook. The student should be able to reason their way to the img2img mechanism because they have all the pieces.

4. **Explain** -- Img2img mechanism
   - The one change: instead of z_T ~ N(0, I), use z_{t_start} = forward_process(vae.encode(image), t_start).
   - Pipeline comparison diagram: txt2img pipeline vs img2img pipeline. Highlight the one difference.
   - VAE encoder callback: "Remember we said the VAE encoder is not used during text-to-image inference? That was correct. Img2img IS different--it is image-to-image, and the encoder converts your input to latent space."
   - The strength parameter: strength determines t_start. Strength = fraction of the denoising process that runs. strength=0.8 means start at step 0.2*T (80% of the process remaining), so 80% of the noise schedule is traversed. Map this onto the alpha-bar curve.
   - Concrete worked example: input photo 512x512 -> VAE encode -> z_0 [4, 64, 64] -> strength=0.7 with 50 DDIM steps means t_start at step 15 (35 steps remaining) -> noise z_0 with forward process to alpha_bar_{t_start} -> denoise from step 15 to step 0 using DDIM -> VAE decode -> output 512x512.
   - Pseudocode showing the 3-line change from txt2img.
   - Boundary case: strength=1.0 means t_start corresponds to maximum noise--the original image is fully destroyed, and the output is standard txt2img. strength=0.0 means no denoising steps run--the output IS the input image unchanged.
   - Why the effect is nonlinear: map strength values onto the alpha-bar curve. Strength=0.2 starts where alpha_bar is high (mostly signal), so only fine details change. Strength=0.8 starts where alpha_bar is low (mostly noise), so the model reimagines the composition. The coarse-to-fine mental model from 6.2.4 explains this directly.

5. **Check** -- Predict-and-verify for img2img
   - "You apply img2img with strength=0.5 to a photo of a dog, with the prompt 'a cat sitting on grass.' What would you expect?" (`<details>` reveal: the broad composition is preserved--similar pose, similar background layout--but the subject is reinterpreted. At strength=0.5, the structural steps have enough latitude to change the animal, but the overall scene composition (grass, sky, approximate object placement) is preserved from the original.)
   - "You apply img2img with strength=0.1 to the same dog photo, same prompt. What changes?" (`<details>` reveal: almost nothing. Only the finest details (color tints, texture grain) change. The structure, shape, and identity of the dog are fully preserved because the model only runs the detail-refinement steps. The prompt has minimal influence because the coarse structure is locked in.)

6. **Explain** -- Inpainting mechanism
   - Motivate: "Img2img changes the entire image. What if you only want to change part of it?"
   - The mechanism: add one operation inside the denoising loop. At each step, after the model predicts the denoised latent, replace the unmasked regions with the original image's latent (re-noised to the current timestep).
   - Formula: z_t_combined = mask * z_t_denoised + (1 - mask) * forward_process(z_0_original, t)
   - Why re-noise the original for unmasked regions: the denoising loop expects latents at the CURRENT noise level. Injecting clean (z_0) latents into a partially denoised (z_t) tensor would create an inconsistency the model cannot handle. Re-noising to the current t keeps everything consistent.
   - Why boundaries blend naturally: the U-Net sees the FULL latent at each step, including both masked and unmasked regions. At the 8x8 bottleneck resolution, each position has global receptive field (callback to 6.3.1). The model's predictions for the masked region account for the unmasked context. This is fundamentally different from cut-and-paste.
   - Pseudocode showing the 1-line addition to the denoising loop.
   - Diagram: per-step masking illustration showing the mask applied at each timestep, with re-noised original latents replacing unmasked regions.
   - Boundary case: mask covering the entire image collapses to standard img2img/txt2img. Mask covering nothing preserves the original image entirely.

7. **Check** -- Predict-and-verify for inpainting
   - "You inpaint the sky of a landscape photo with the prompt 'dramatic thunderstorm clouds.' The mask covers only the sky. What happens at the boundary between sky and mountains?" (`<details>` reveal: the boundary blends naturally. The U-Net sees the mountain latents in the unmasked region and the denoised sky in the masked region at every step. Its predictions for the sky region near the boundary account for the mountains, producing coherent lighting, color transitions, and overlap.)
   - "What happens if you inpaint with a mask covering the entire image?" (`<details>` reveal: every region is denoised, none are preserved. This is identical to standard img2img. The mask is the entire difference between inpainting and img2img.)

8. **Elaborate** -- Practical considerations and nuance
   - **Strength selection guide:** Three tiers:
     - Low strength (0.1-0.3): Color grading, style tinting, minor texture changes. Original structure fully preserved. Good for "same image, slightly different feel."
     - Medium strength (0.4-0.6): Style transfer, subject reinterpretation, moderate creative changes. Broad composition preserved but details change significantly. The "sweet spot" for most editing tasks.
     - High strength (0.7-0.9): Major creative reinterpretation. Only the vaguest spatial hints from the original survive. Good for using sketches as composition guides.
   - **Sketch-to-image as a practical application:** A rough hand-drawn sketch provides structural guidance at high strength; the model fills in realistic detail. The sketch does not need to be good--just enough to anchor the composition.
   - **Inpainting mask considerations:** Mask shape affects results. Too-tight masks leave insufficient context for the model. Slightly oversized masks produce better results because the model has room to integrate the new content with its surroundings. Feathered (soft) mask edges can help gradual transitions.
   - **Inpainting with text prompts:** The prompt guides WHAT fills the masked region. Without a specific prompt, the model fills with contextually plausible content. With a prompt, you direct it ("replace this tree with a fountain").
   - **Specialized inpainting models:** Briefly note that dedicated inpainting models exist (e.g., SD inpainting models fine-tuned with an extra mask input channel to the U-Net). These produce better results at mask boundaries but use the same underlying mechanism. They are an optimization, not a fundamentally different approach.
   - **Connection to the customization spectrum:** LoRA changes the weights (requires training). Img2img/inpainting change the inference process (no training). Textual inversion (next lesson) changes the embeddings (requires training but no weight changes). Three different knobs on three different parts of the pipeline.

9. **Check** -- Transfer question
   - "A colleague says they want to 'outpaint'--extend an image beyond its borders, adding new content that seamlessly continues the existing scene. You have not seen outpainting before. Using what you learned about inpainting, can you describe how outpainting might work?" (`<details>` reveal: pad the image with blank/noise pixels, create a mask that covers the new regions (and slightly overlaps the original edges for blending), then run inpainting. The model denoises the masked extension regions using the unmasked original as context. It is inpainting with the mask on the outside instead of the inside.)
   - "Your friend asks: 'Does img2img work better with DDIM or DPM-Solver?' How would you reason about this?" (`<details>` reveal: the sampler is orthogonal to the img2img mechanism. Img2img changes the starting point; the sampler determines how to get from there to z_0. Any sampler works. The same practical guidance from 6.4.2 applies: DPM-Solver++ at 20-30 steps for efficiency, DDIM for reproducibility. The sampler does not "know" whether it started from pure noise or a noised image.)

10. **Practice** -- Notebook exercises (Colab)
    - **Exercise 1 (Guided):** Img2img strength exploration. Load a pretrained SD pipeline (StableDiffusionImg2ImgPipeline). Take a single input image. Run img2img at strengths 0.1, 0.3, 0.5, 0.7, 0.9 with the same prompt and seed. Display results in a grid. Purpose: make the strength parameter tangible by seeing the full spectrum. Predict-before-run: "Before running, predict which strength will preserve the most original structure."
    - **Exercise 2 (Guided):** Trace the img2img mechanism by hand. Encode an image with VAE, manually add noise using the forward process formula at a specific timestep (corresponding to strength=0.7), then run the standard denoising loop from that timestep. Compare the output to the img2img pipeline's output (should match). Purpose: verify that img2img IS the forward-process-then-denoise mechanism from the lesson--not a black box.
    - **Exercise 3 (Supported):** Inpainting with mask creation. Load StableDiffusionInpaintPipeline. Create a binary mask (using PIL or numpy) for a specific region of an image. Run inpainting with a descriptive prompt. Experiment with mask size (tight vs generous). Purpose: hands-on inpainting with attention to mask design and boundary quality.
    - **Exercise 4 (Independent):** Creative application. Given a hand-drawn sketch (or student draws their own simple sketch), use img2img at high strength to turn it into a realistic image. Then use inpainting to modify a specific element of the result. Purpose: combine both techniques in a practical workflow, demonstrating the creative pipeline of sketch -> img2img -> selective inpainting refinement.
    - Exercises are cumulative: Ex1 builds strength intuition, Ex2 grounds the mechanism in code, Ex3 introduces inpainting, Ex4 combines both in a creative workflow.
    - Key reasoning to emphasize in solutions: WHY specific strength values produce specific effects (coarse-to-fine), WHY the VAE encoder is needed (input must be in latent space), WHY unmasked regions are re-noised at each inpainting step (consistency with current noise level).

11. **Summarize**
    - Three key takeaways: (1) Img2img uses the forward process formula to noise a real image to a specific timestep, then denoises from there instead of from pure noise. The strength parameter determines the starting point on the alpha-bar curve. (2) Inpainting adds a per-step spatial mask to the denoising loop, preserving original latents in unmasked regions and denoising only the masked area. Boundaries blend naturally because the U-Net sees the full image at every step. (3) Neither technique requires training--they are purely inference-time modifications to the denoising process you already know.
    - Mental model echo: "Same denoising process, different starting point (img2img) or selective application (inpainting)."

12. **Next step**
    - Preview textual inversion: "LoRA customized the model's weights. Img2img and inpainting customized the inference process. What if you could teach the model a new concept--your pet, your art style--without changing any weights at all? What if you could optimize a single embedding vector in CLIP's space to represent something the model has never seen? That is textual inversion, and it is the most surprising customization technique of the three."

---

## Widget Assessment

**Widget needed:** No custom interactive widget required.

The lesson's core visual needs are:
1. Pipeline comparison diagrams (txt2img vs img2img vs inpainting)--static SVG or Mermaid diagrams, not interactive
2. Strength spectrum illustration--a static or lightly interactive row of images at different strength values, achievable with GradientCards or a simple image grid
3. Inpainting per-step masking diagram--static SVG showing the mask application at each timestep

None of these require a custom React widget. The alpha-bar curve connection could reference the student's prior experience with the AlphaBarCurveWidget from 6.2.2 without rebuilding it. The notebook exercises provide the hands-on interactive experience (running real img2img and inpainting with diffusers). The concepts in this lesson are best served by clear diagrams and worked examples rather than interactive manipulation.

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
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 2 negative/boundary)
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

One critical finding in the notebook (syntax error that will crash cell-12 and block the student from completing Exercise 2) plus three improvement findings that would weaken the lesson's effectiveness.

### Findings

#### [CRITICAL] -- Notebook cell-12 has a Python syntax error that will crash

**Location:** Notebook `6-5-2-img2img-and-inpainting.ipynb`, cell-12 (Exercise 2 Step 1: VAE encode)
**Issue:** The line `print(f'This is the same encoding from "From Pixels to Latents."\')` has a backslash before the closing single quote, which escapes it and leaves the f-string unclosed. Python will raise a `SyntaxError`. The entire cell will fail to execute, blocking the student from completing Exercise 2 (the most pedagogically important exercise, since it implements img2img from scratch).
**Student impact:** The student runs cell-12 and gets an immediate `SyntaxError`. They cannot proceed with Exercise 2 without debugging the notebook itself--a frustrating, momentum-killing experience that has nothing to do with the lesson content.
**Suggested fix:** Change the line to: `print(f'This is the same encoding from "From Pixels to Latents."')` (remove the trailing backslash before the closing quote). Alternatively, use a regular `print()` call instead of an f-string since there are no interpolated variables on that line.

#### [IMPROVEMENT] -- Exercise 2 claims output will match Exercise 1 but uses a different sampler

**Location:** Notebook cell-10 (Exercise 2 introduction) and cell-11 (component loading)
**Issue:** The Exercise 2 intro says "compare to the pipeline's output to verify they match" and asks the predict-before-run question "Will the manually constructed output match the pipeline output from Exercise 1? (Same seed, same strength, same prompt.)" However, Exercise 1 uses `DPMSolverMultistepScheduler` while Exercise 2 uses `DDIMScheduler`. Different samplers produce different outputs even with the same seed, strength, and prompt. The student will see two different images and be confused about whether they did something wrong.
**Student impact:** The student follows the instructions, compares the two outputs, sees they do NOT match, and questions whether their manual implementation is correct. This undermines the core pedagogical point ("img2img IS the forward-process-then-denoise mechanism") by introducing doubt.
**Suggested fix:** Either (a) use the same sampler in both exercises (switch Exercise 2 to DPM-Solver to match Exercise 1), or (b) remove the comparison claim and the misleading prediction question. Option (a) is better pedagogically because the matching outputs would powerfully confirm the lesson's claim that img2img is just the standard pipeline with a different starting point. Option (b) is simpler but loses that confirmation moment.

#### [IMPROVEMENT] -- Exercise 3 scaffolding inconsistency: tight mask is pre-filled but labeled as TODO

**Location:** Notebook cell-21 (Exercise 3: mask creation)
**Issue:** The tight mask is already fully implemented (`tight_draw.rectangle([0, 0, 512, 200], fill=255)`) with the comment "Replace with your coordinates," but it is presented as a TODO. Meanwhile, the generous mask is set to `None` with "Replace this line." This creates a confusing mixed signal: the student reads "TODO" and thinks they should figure out the tight mask coordinates, but the code already does it. The actual student task is only the generous mask, but the framing suggests both are TODOs.
**Student impact:** The student either (a) wastes time re-deriving tight mask coordinates that are already correct, or (b) is confused about what they actually need to do. Neither outcome serves the learning goal.
**Suggested fix:** Make the scaffolding levels explicit. Remove "TODO" from the tight mask section and add a brief comment like `# Pre-filled: the tight mask covers the sky region (top 200 pixels)`. Keep the generous mask as the actual TODO. This makes the scaffolding progression clear: "Here is the tight mask. Your task: create the generous mask with 40px extra padding."

#### [IMPROVEMENT] -- Inpainting mechanism explanation jumps to the formula before grounding with a concrete scenario

**Location:** Lesson TSX, Section 7 ("Inpainting: Selective Editing"), lines 566-586
**Issue:** The section opens with a good motivation paragraph ("Img2img changes the entire image. But what if you only want to replace one object?"). But the very next paragraph jumps directly to "The mechanism is one line added to the denoising loop" followed by the BlockMath formula. The formula `z_t^{combined} = m * z_t^{denoised} + (1-m) * forward(z_0^{original}, t)` arrives before the student has a concrete picture of what is happening. The ordering rule "concrete before abstract" is violated here--the formula is the abstract representation, and it appears before any concrete walkthrough.
**Student impact:** The student reads the formula and has to reverse-engineer what it means. The symbols m, z_t^{denoised}, and forward(z_0^{original}, t) are each doing something specific, but the student encounters them all at once in a single equation. The "why re-noise the original" GradientCard helps, but it comes AFTER the formula rather than before, so the student has already been confused by the time the explanation arrives.
**Suggested fix:** Before the formula, add a brief concrete scenario: "Imagine a landscape with a tree you want to replace with a fountain. At each denoising step, the model produces a prediction for the full image. You keep its prediction for the tree region (where the fountain will appear) and throw away its prediction for the rest of the image, replacing it with the original landscape re-noised to the current timestep. This way, the model can only change the tree region." Then present the formula as the symbolic representation of what was just described in words. This follows the same concrete-before-abstract pattern used effectively in the img2img section.

#### [POLISH] -- Spaced em dashes in HTML comments

**Location:** Lesson TSX, HTML comment blocks (lines 149, 205, 506, 558, 754, 805, 907, 963)
**Issue:** The HTML comment section dividers use spaced em dashes: `Section 4: Hook — "What if..."`. While these are invisible to the student (they are code comments), the project convention specifies no-space em dashes throughout.
**Student impact:** None (comments are not rendered). This is purely a code consistency issue.
**Suggested fix:** Change to `Section 4: Hook--"What if..."` or use a plain ASCII dash in comments. Low priority since these are not visible.

#### [POLISH] -- Alpha-bar curve plot in notebook cell-7 maps strength to total_timesteps (1000) rather than inference timesteps (30)

**Location:** Notebook cell-7 (alpha-bar curve visualization in Exercise 1)
**Issue:** The code computes `t_start_idx = int(strength * total_timesteps)` where `total_timesteps = len(alphas_cumprod) = 1000`. This maps strength linearly onto the full 1000-timestep schedule. However, in actual inference with 30 steps, the scheduler selects a subset of 30 timesteps from the 1000. The mapping from strength to the actual starting timestep in the 30-step schedule is slightly different from the linear mapping onto the 1000-step schedule. The plotted positions are approximately correct but not exact.
**Student impact:** Minimal--the visualization conveys the right qualitative message (nonlinear curve, different alpha-bar values at different strengths). The student would not notice the discrepancy. But a student who carefully computes the actual t_start from Exercise 2 (where the real 30-step schedule is used) might notice the alpha-bar value does not exactly match the plotted value.
**Suggested fix:** Use the scheduler's actual timestep mapping: `scheduler.set_timesteps(30); actual_timesteps = scheduler.timesteps; t_start_idx = actual_timesteps[int((1-strength) * 30)]`. This would produce exact positions. Low priority since the qualitative message is correct.

### Review Notes

**What works well:**
- The lesson structure closely follows the plan. The narrative arc of "you already have all the pieces" is compelling and consistently reinforced throughout.
- The hook section (Section 4) is excellent--it genuinely challenges the student to derive img2img from prior knowledge before revealing the answer. The `<details>` reveal is well-placed.
- The boundary cases for both img2img (strength=1.0) and inpainting (full-image mask) are clearly presented and effectively address the planned misconceptions.
- The connection between strength and the alpha-bar curve is well-developed with the three-tier GradientCards and the coarse-to-fine callback.
- The VAE encoder callback (aside in Section 5) elegantly resolves the prior "encoder is not used during inference" statement.
- The notebook exercises have good scaffolding progression (Guided -> Guided -> Supported -> Independent) and good VRAM management (one pipeline at a time with cleanup).
- The customization spectrum comparison (LoRA / img2img-inpainting / textual inversion) provides excellent module-level context.

**Patterns to watch:**
- The lesson is text-heavy with no custom interactive widgets, which is appropriate per the Widget Assessment. But the notebook is doing the heavy lifting for interactivity--the critical syntax error in cell-12 therefore has outsized impact.
- The sampler mismatch between Exercises 1 and 2 is an implementation oversight that directly undermines the "verify they match" pedagogical moment. This is the kind of thing that erodes student trust in the material.

**Priority for fixes:**
1. Fix the syntax error in cell-12 (Critical--blocks Exercise 2)
2. Fix the sampler mismatch between Ex1 and Ex2 (Improvement--misleading comparison)
3. Add concrete scenario before the inpainting formula (Improvement--ordering rule violation)
4. Clarify Exercise 3 TODO scaffolding (Improvement--confusing mixed signal)

---

## Review -- 2026-02-14 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

All Iteration 1 findings have been addressed. The critical syntax error is fixed, both exercises use DDIM, the inpainting section now has a concrete scenario before the formula, and Exercise 3 scaffolding is clear. One new improvement-level finding (VAE encode determinism in the Exercise 1 vs Exercise 2 comparison) and one remaining polish-level finding (spaced em dashes in notebook markdown cells).

### Iteration 1 Fix Verification

| Iteration 1 Finding | Status | Verification |
|---------------------|--------|--------------|
| [CRITICAL] cell-12 syntax error | FIXED | Print statements in cell-12 now use proper string quoting. No backslash-before-closing-quote. The problematic f-strings were replaced with regular print calls where no interpolation was needed. Cell will execute cleanly. |
| [IMPROVEMENT] Sampler mismatch between Ex1 and Ex2 | FIXED | Cell-6 now imports and uses `DDIMScheduler` (`from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler`; `img2img_pipe.scheduler = DDIMScheduler.from_config(...)`). Cell-11 also uses `DDIMScheduler`. Both exercises use the same sampler. Cell-6 comment explains the choice: "its deterministic stepping makes comparison between the pipeline (Exercise 1) and our manual implementation (Exercise 2) straightforward." |
| [IMPROVEMENT] Exercise 3 scaffolding inconsistency | FIXED | Cell-21 tight mask section now reads `# Pre-filled: the tight mask covers the sky region (top 200 pixels). / # This is intentionally snug against the sky-mountain boundary.` No TODO label on the pre-filled code. The TODO is only on the generous mask. Clear scaffolding progression. |
| [IMPROVEMENT] Inpainting formula before concrete scenario | FIXED | Section 7 (lines 573-580) now has a concrete walkthrough before the formula: "Imagine a landscape with a tree you want to replace with a fountain. At each denoising step, the model produces a prediction for the full image. You keep its prediction for the tree region..." The BlockMath formula follows as the symbolic version. Concrete-before-abstract ordering restored. |
| [POLISH] Spaced em dashes in HTML comments | NOT FIXED (intentional) | HTML comments still use the Unicode em dash character. These are invisible to the student. Low priority, noted but acceptable. |
| [POLISH] Alpha-bar curve plot mapping | FIXED | Cell-7 now uses the scheduler's actual timestep mapping: `img2img_pipe.scheduler.set_timesteps(num_inference_steps)` then `actual_timesteps = img2img_pipe.scheduler.timesteps.cpu().numpy()`. Each strength is mapped to its actual start step index in the 30-step DDIM schedule, producing exact alpha-bar positions. The plot title and print statements correctly reference the 30-step DDIM schedule. |

### Findings

#### [IMPROVEMENT] -- VAE encode non-determinism may cause Exercise 2 output to not exactly match Exercise 1

**Location:** Notebook cell-12 (Exercise 2 Step 1: VAE encode)
**Issue:** Cell-12 calls `latent_dist.latent_dist.sample()` without passing a `generator` argument. The VAE encoder's `sample()` method draws from the latent distribution (mean + variance), introducing stochastic variation. The `StableDiffusionImg2ImgPipeline` in Exercise 1 internally passes the user's generator to the VAE encode step (`latent_dist.sample(generator=generator)`). Because the manual implementation does not pass a generator, the VAE encoding produces slightly different latents, and the generator state diverges before noise sampling. The Exercise 2 intro (cell-10) claims: "the manual output should match Exercise 1's pipeline output at strength=0.7." The outputs will be visually very similar but not pixel-identical.
**Student impact:** The student compares the manual result to the Exercise 1 grid (strength=0.7) and notices they are very close but not exactly the same. This is a minor discrepancy that could cause brief confusion ("Did I do something wrong?"), but the overall pedagogical point still lands because the images will be clearly similar in composition and style. The impact is significantly less than the prior sampler mismatch (which produced obviously different images), but it slightly undermines the "should match" claim.
**Suggested fix:** Either (a) pass the generator to the VAE sample call: `z_0 = latent_dist.latent_dist.sample(generator=generator) * vae.config.scaling_factor` (requires creating the generator before VAE encoding and managing the state carefully to match the pipeline's internal ordering), or (b) use `latent_dist.latent_dist.mode()` (which returns the mean, deterministic, no generator needed) and note in a comment that this is a simplification that avoids the stochastic sampling. Option (b) is simpler and introduces the idea that the VAE's distribution is tight enough that the mean is a good approximation. Or (c) soften the "should match" claim to "should produce a very similar result" and add a brief note explaining that minor differences come from VAE encoder sampling randomness.

#### [POLISH] -- Spaced em dashes in notebook markdown cells

**Location:** Notebook cells: cell-23 solution ("TODO 1 — Create", "TODO 2 — Run") and cell-27 Exercise 4 instructions ("— for the initial transformation", "— for the selective edit", "— for mask creation")
**Issue:** Five instances of spaced em dashes in student-visible notebook markdown. The project convention is no-space em dashes (`word—word`).
**Student impact:** None pedagogically. This is a style consistency issue. The student reads the content without noticing the spacing.
**Suggested fix:** Change to no-space em dashes: "TODO 1—Create", "TODO 2—Run", "`StableDiffusionImg2ImgPipeline`—for the initial transformation", etc.

### Review Notes

**What works well:**
- All Iteration 1 fixes were applied correctly and effectively. The sampler alignment between Exercises 1 and 2 is the most impactful fix--the notebook now tells a coherent story where the manual implementation confirms the pipeline's output.
- The concrete scenario before the inpainting formula (the tree-to-fountain walkthrough) reads naturally and follows the same concrete-before-abstract pattern used in the img2img section. The flow from motivation paragraph -> concrete scenario -> formula -> explanation is smooth.
- The Exercise 3 scaffolding is now unambiguous: pre-filled tight mask with clear labeling, student task is only the generous mask. The TODO/pre-filled boundary is clean.
- The alpha-bar curve visualization in cell-7 now uses exact scheduler positions, which will produce correct values that match what the student computes in Exercise 2. This eliminates a potential source of confusion for detail-oriented students.
- The lesson TSX compiles cleanly (typecheck and lint both pass).
- The lesson structure, narrative arc, pedagogical principles, modality coverage, example quality, and notebook scaffolding progression are all strong. The core teaching approach--"you already have all the pieces, here is how they recombine"--is consistently reinforced and genuinely effective.

**Overall assessment:**
The lesson and notebook are in good shape. The one improvement finding (VAE encode determinism) is a subtle implementation detail that slightly overpromises the "should match" comparison but does not meaningfully impair learning. A quick fix (option b or c from the suggested fix) would resolve it cleanly. The polish finding is minor. After addressing the improvement finding, this lesson should pass.

---

## Review -- 2026-02-14 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All Iteration 2 findings have been addressed. The lesson and notebook are pedagogically sound, well-structured, and ready to ship.

### Iteration 2 Fix Verification

| Iteration 2 Finding | Status | Verification |
|---------------------|--------|--------------|
| [IMPROVEMENT] VAE encode non-determinism causing Exercise 2 output to not match Exercise 1 | FIXED | Cell-12 now uses `.mode()` (the distribution mean) instead of `.sample()`. The comment explains: "We use .mode() (the distribution mean) rather than .sample() for deterministic encoding. The VAE's learned variance is small enough that the mean is an excellent approximation, and this avoids generator-state differences when comparing to the pipeline output in Exercise 1." This is option (b) from the suggested fix, cleanly implemented with a pedagogically valuable note about why the mean is a good approximation. |
| [POLISH] Spaced em dashes in notebook markdown cells | FIXED | Cell-23 solution text now uses no-space double hyphens: "TODO 1--Create the generous mask:", "TODO 2--Run inpainting:". Cell-27 Exercise 4 instructions also use no-space double hyphens: "`StableDiffusionImg2ImgPipeline`--for the initial transformation", "`StableDiffusionInpaintPipeline`--for the selective edit", etc. Consistent with project convention. |

### Findings

#### [POLISH] -- HTML comment section dividers still use Unicode em dash character

**Location:** Lesson TSX, HTML comment blocks (lines 149, 205, 506, 558, etc.)
**Issue:** The HTML comment section dividers use the Unicode em dash character (e.g., `Section 4: Hook—"What if you didn't start from pure noise?"`). The project convention specifies no-space em dashes, which these satisfy (no spaces around the em dash), but previous review iterations noted the inconsistency of using the Unicode character in code comments versus double hyphens used elsewhere in the codebase.
**Student impact:** None. Comments are not rendered. This is purely a code consistency issue.
**Suggested fix:** Low priority. Could change to double hyphens in comments for consistency, but this is cosmetic and invisible to the student.

### Review Notes

**Full lesson re-evaluation (final iteration thoroughness):**

The lesson was re-read from scratch as a student with the mental state documented in the module record. No confusion points, cognitive leaps, or disengagement triggers were found. The lesson is effective for the target student profile.

**Pedagogical principles check (all pass):**
- Motivation Rule: Problem before solution in both img2img (hook) and inpainting (motivation paragraph).
- Modality Rule: 5 modalities present (visual/diagram via ComparisonRow, concrete example via worked trace, verbal via throughout, symbolic/code via pseudocode, intuitive/geometric via alpha-bar mapping).
- Example Rules: 3 positive + 2 negative examples. Minimum exceeded.
- Misconception Rule: 5 misconceptions addressed with concrete negative examples at appropriate locations.
- Ordering Rules: Concrete before abstract throughout (tree-to-fountain before formula, worked example before generalization).
- Load Rule: 2 genuinely new concepts. Within limit.
- Connection Rule: Every new concept explicitly linked to existing knowledge (forward process, alpha-bar, coarse-to-fine, VAE encoder, U-Net receptive field).
- Reinforcement Rule: Forward process formula used in three new contexts. Alpha-bar curve referenced. No fading concepts.
- Interaction Design Rule: `<details>` summaries have `cursor-pointer`. No other interactive elements requiring cursor styling.
- Writing Style Rule: All em dashes in student-visible lesson prose are no-space format. Clean.

**Notebook evaluation (all pass):**
- 4 exercises matching the plan (Guided, Guided, Supported, Independent).
- Scaffolding progression correct: first exercise is Guided, only one Independent at the end.
- Guided exercises use predict-before-run pattern.
- Supported/Independent exercises have `<details>` solution blocks with reasoning before code, common mistakes, and design rationale.
- Self-contained setup with `!pip install`, all imports in cell-2, random seeds set, no local file references.
- VRAM management is explicit: one pipeline at a time, `cleanup()` between exercises.
- Terminology aligned with lesson TSX throughout.
- No new concepts introduced in the notebook.

**What works especially well:**
- The "you already have all the pieces" narrative is the strongest aspect of this lesson. Every section reinforces that img2img and inpainting are reconfigurations of known mechanisms, not new algorithms. The hook makes the student derive img2img before being told, which is powerful.
- The VAE encoder callback in the aside ("We said the encoder is not used during text-to-image. That was correct.") is elegant and resolves a prior statement without making it feel like a correction.
- The customization spectrum (LoRA / img2img-inpainting / textual inversion) provides excellent module-level framing and foreshadows Lesson 3.
- The boundary cases for both techniques (strength=1.0, full-image mask) are the cleanest negative examples in this lesson--they reduce each mechanism to a degenerate case that the student already understands.
- Exercise 2 (manual implementation) is the pedagogical crown jewel of the notebook. Implementing img2img from scratch with raw components, then comparing to the pipeline output, makes the "it is just the forward process + denoising loop" claim verifiable rather than claimed.
- The `.mode()` fix in cell-12 is well-documented and introduces a useful concept (VAE distribution is tight, mean is a good approximation) without derailing the exercise.

**Conclusion:**
This lesson is ready to ship. The one remaining polish item (HTML comment em dashes) is invisible to the student and does not affect the learning experience. The lesson effectively teaches both img2img and inpainting as inference-time reconfigurations, grounds them in prior knowledge, and provides a strong notebook with progressive scaffolding.
