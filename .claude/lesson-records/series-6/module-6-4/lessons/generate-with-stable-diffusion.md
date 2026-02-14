# generate-with-stable-diffusion -- Lesson Planning Document

**Module:** 6.4 (Stable Diffusion)
**Position:** Lesson 3 of 3 (Lesson 17 in Series 6)
**Type:** CONSOLIDATE (zero new concepts -- the payoff moment where the student uses the real library and understands every parameter)
**Previous lesson:** samplers-and-efficiency (STRETCH)
**Next lesson:** lora-finetuning (Module 6.5, Lesson 1)

---

## Phase 1: Student State

### Relevant concepts the student has

| Concept | Depth | Source Lesson | Notes |
|---------|-------|--------------|-------|
| Full SD pipeline data flow (text prompt -> tokenizer -> CLIP text encoder -> denoising loop with CFG -> VAE decode -> pixel image, with tensor shapes at every handoff) | DEVELOPED | stable-diffusion-architecture (6.4.1) | Student traced the complete pipeline with "a cat sitting on a beach at sunset." Knows every tensor shape: prompt -> [77] token IDs -> [77, 768] embeddings -> z_T [4, 64, 64] -> 50 steps (100 U-Net passes with CFG) -> z_0 [4, 64, 64] -> VAE decode -> [3, 512, 512]. This is the mental model the student will now map to API parameters. |
| Classifier-free guidance (training with random text dropout, two forward passes at inference, amplify text direction) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Student knows the formula: epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond). Knows w=7.5 is typical. Has implemented it in a notebook exercise. This maps directly to `guidance_scale`. |
| Guidance scale tradeoff (w=1 no amplification, w=7.5 typical, w=20 oversaturated) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Three GradientCards showing the spectrum. Student knows higher is not always better. They can predict the effect of changing `guidance_scale` before running the code. |
| DDIM predict-x_0-then-jump mechanism (predict clean image, leap to target timestep via closed-form formula) | DEVELOPED | samplers-and-efficiency (6.4.2) | Student can explain the two-step process, trace a step with the formula, and explain why it enables step-skipping. Two concrete examples traced (t=1000->t=800, t=200->t=0). Maps to `scheduler` choice. |
| Sampler comparison and practical selection guidance (DPM-Solver++ at 20-30 steps as default, DDIM at 50 for reproducibility, Euler at 30-50 for debugging, DDPM at 1000 for max quality) | INTRODUCED | samplers-and-efficiency (6.4.2) | Four GradientCards with recommended step counts and use cases. Student knows which sampler to pick and why. Maps to both `scheduler` and `num_inference_steps`. |
| Sampler as inference-time choice (swapping samplers requires zero retraining; model predicts noise, sampler decides what to do with prediction) | DEVELOPED | samplers-and-efficiency (6.4.2) | Foundational insight from Lesson 16. Student knows a one-liner in diffusers swaps schedulers. |
| DDIM deterministic generation (sigma=0 gives same seed = same image regardless of step count) | INTRODUCED | samplers-and-efficiency (6.4.2) | Connected to temperature analogy. Student knows deterministic sampling enables reproducibility and A/B testing. Maps to `generator` (seed). |
| Component modularity (three independently trained models connected by tensor handoffs, not shared weights) | INTRODUCED | stable-diffusion-architecture (6.4.1) | Three GradientCards with parameter counts (CLIP ~123M, U-Net ~860M, VAE ~84M). Student knows swappability. Maps to model ID / checkpoint selection. |
| Negative prompts as CFG application (replacing empty-string unconditional embedding with CLIP encoding of undesired attributes) | INTRODUCED | stable-diffusion-architecture (6.4.1) | epsilon_cfg = epsilon_neg + w * (epsilon_cond - epsilon_neg). Student knows this is not a new mechanism. Maps to `negative_prompt`. |
| CLIP text encoder produces sequence of token embeddings ([77, 768] regardless of prompt length) | INTRODUCED | stable-diffusion-architecture (6.4.1), text-conditioning-and-guidance (6.3.4) | Student knows the tokenizer pads to 77 tokens with SOT/EOT. Maps to `prompt` (and understanding its 77-token limit). |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings, per-spatial-location conditioning) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Student knows each spatial location attends to different text tokens. Explains why prompt wording matters: cross-attention can only attend to tokens that are present. |
| Latent diffusion as architectural pattern (encode with VAE, diffuse in latent space, decode back) | DEVELOPED | from-pixels-to-latents (6.3.5) | Student understands the 48x compression (512x512x3 -> 64x64x4). Maps to `height`/`width` (must be multiples of 8, because 8x spatial downsampling). |
| Computational cost reduction via latent space (512x512x3 = 786,432 vs 64x64x4 = 16,384) | DEVELOPED | from-pixels-to-latents (6.3.5) | Student computed the compression ratio. Explains why larger images cost quadratically more (latent spatial area scales with pixel area). |
| Seed determines z_T and therefore the generated image (same prompt + same seed = same image) | INTRODUCED | stable-diffusion-architecture (6.4.1) | Brief explanation. Student knows the concept but has not experimented with seeds systematically. Maps to `generator` (torch.Generator with manual_seed). |
| ODE perspective on diffusion (model predictions define a trajectory; samplers are ODE solvers) | INTRODUCED | samplers-and-efficiency (6.4.2) | Student understands the trajectory interpretation. Explains why `num_inference_steps` interacts with scheduler choice: different solvers traverse the same trajectory differently. |
| Coarse-to-fine denoising progression (early steps create structure, late steps refine details) | DEVELOPED | sampling-and-generation (6.2.4) | Extended via DenoisingTrajectoryWidget. Explains why low step counts lose detail (coarse structure only) and why a seed "looks" a certain way (the structural decisions happen early). |

### Mental models and analogies already established

- **"The model defines where to go. The sampler defines how to get there."** -- Core sampler mental model from samplers-and-efficiency (6.4.2). The model predicts noise; the sampler decides how to use that prediction.
- **"Three translators, one pipeline."** -- CLIP translates language to geometric-meaning space, the U-Net generates in latent space, the VAE translates latent to pixel language. From stable-diffusion-architecture (6.4.1).
- **"Assembly line"** -- Each component does one job and passes the result via tensor handoffs. From stable-diffusion-architecture (6.4.1).
- **"Contrast slider"** -- CFG guidance scale as contrast adjustment. From text-conditioning-and-guidance (6.3.4).
- **"Same vehicle, different route"** -- DDPM takes every back road, DDIM takes the highway, DPM-Solver uses GPS. From samplers-and-efficiency (6.4.2).
- **"Predict and leap"** -- DDIM's mechanism: predict destination, jump to target timestep. From samplers-and-efficiency (6.4.2).
- **"The timestep tells the network WHEN. The text tells it WHAT. CFG turns up the volume on the WHAT."** -- Complete conditioning mental model from text-conditioning-and-guidance (6.3.4).
- **"Same building blocks, different question"** -- Recurring meta-pattern throughout Series 6.
- **"The VAE is a translator between two languages"** -- Pixel language and latent language. From from-pixels-to-latents (6.3.5).

### What was explicitly NOT covered in prior lessons (relevant here)

- **Using the diffusers high-level pipeline API (`StableDiffusionPipeline`)** -- The student has used individual diffusers components (schedulers, CLIP, VAE, U-Net separately) in notebooks for Lessons 15 and 16, but has not used the high-level `pipe()` call that wraps everything into a single function. The mapping from API parameters to internal pipeline behavior is the core content of this lesson.
- **Systematic parameter exploration with real generated images** -- The student has generated images in Lesson 16's notebook but focused on sampler comparison, not on exploring the full parameter space (varying prompts, guidance scales, seeds, image sizes, negative prompts together).
- **The 77-token limit as a practical constraint** -- The student knows CLIP tokenizes to 77 tokens, but has not confronted what happens when a prompt is too long (truncation) and how to work within this limit.
- **Height/width as multiples of 8** -- The student knows the 8x spatial downsampling factor but has not connected this to the practical constraint that image dimensions must be multiples of 8.
- **Practical prompt structure (what makes prompts effective)** -- The student knows cross-attention attends to tokens, but has not explored how prompt structure (ordering, emphasis, specificity) affects generation in practice.

### Readiness assessment

The student is maximally prepared. Every concept that maps to an API parameter has been taught at DEVELOPED or INTRODUCED depth across Modules 6.1-6.4. There are zero gaps and zero fading concerns -- the two preceding lessons in this module are the immediate predecessors.

This is the easiest lesson in the module cognitively but the most satisfying emotionally. The student has spent 16 lessons building every piece from scratch. Now they use the real tool and understand everything it does. The cognitive work is connecting (mapping API parameters to concepts), not learning (no new concepts). The emotional work is experiencing the payoff: "I know what all of this means."

The previous lesson was STRETCH (three genuinely new concepts). This CONSOLIDATE lesson provides cognitive relief while delivering the module's promised capstone experience.

---

## Phase 2: Analysis

### Target concept

This lesson teaches the student to use the diffusers library to generate images with Stable Diffusion, understanding what every parameter controls because each one maps to a concept they built from scratch across Modules 6.1-6.4.

### Prerequisites table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Full SD pipeline data flow | DEVELOPED | DEVELOPED | stable-diffusion-architecture (6.4.1) | OK | Student must know the pipeline to understand what the high-level API wraps. Has the complete trace with tensor shapes. |
| Classifier-free guidance formula and tradeoffs | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Must understand CFG to predict the effect of `guidance_scale`. Student has the formula and implemented it. |
| Sampler comparison and practical selection | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) | OK | Must know which sampler to choose and why. Student has the four GradientCards with recommendations. |
| Sampler as inference-time choice (no retraining) | INTRODUCED | DEVELOPED | samplers-and-efficiency (6.4.2) | OK | Must understand that swapping schedulers is a one-liner. Student has this at DEVELOPED. |
| DDIM deterministic generation (seed reproducibility) | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) | OK | Must understand why same seed + deterministic sampler = same image. |
| Negative prompts as CFG application | INTRODUCED | INTRODUCED | stable-diffusion-architecture (6.4.1) | OK | Must understand the mechanism to predict negative prompt effects. Student knows the formula substitution. |
| CLIP tokenizer padding to 77 tokens | INTRODUCED | INTRODUCED | stable-diffusion-architecture (6.4.1) | OK | Must understand the 77-token context length to understand prompt length constraints. |
| Latent diffusion compression (8x spatial downsampling) | INTRODUCED | DEVELOPED | from-pixels-to-latents (6.3.5) | OK | Must understand why height/width must be multiples of 8. Student has the dimension walkthrough. |
| Seed determines z_T and therefore the image | INTRODUCED | INTRODUCED | stable-diffusion-architecture (6.4.1) | OK | Must understand the seed concept to use `generator` meaningfully. |
| Compute cost breakdown (denoising loop dominates) | INTRODUCED | INTRODUCED | stable-diffusion-architecture (6.4.1) | OK | Must understand where time goes to reason about speed vs quality tradeoffs. |

### Gap resolution

No gaps. Every prerequisite is at or above the required depth. This is a CONSOLIDATE lesson that connects existing knowledge to API parameters -- no new concepts needed.

### Misconceptions table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The diffusers library is a black box I'm just calling" | Every other library the student has used (PyTorch, torchvision) involved calling functions without necessarily understanding the internals. The pattern of `pipe(prompt)` looks like a black box call. The student's instinct from software engineering is "use the API, don't worry about internals." | Map every parameter to its source lesson. `guidance_scale=7.5` is not a magic number -- it is the w in epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond) from Lesson 13. `num_inference_steps=20` is not a quality dial -- it is the number of DPM-Solver steps along the ODE trajectory from Lesson 16. The student can predict what each parameter change will do BEFORE running the code, because they built the concepts. | Section 4 (core content), as the organizing principle of the entire lesson. Each parameter is introduced with "you already know this as..." |
| "Higher guidance_scale always means better images" | The name "guidance_scale" implies "more guidance = better." The student may remember w=7.5 as the default without remembering the oversaturation at w=20. When exploring, they may try w=15 or w=20 expecting improvement. | Generate the same prompt at w=3, w=7.5, w=15, w=25. At w=3, the image is vague and unfaithful to the prompt. At w=7.5, it is balanced. At w=15, colors are oversaturated. At w=25, the image is distorted with extreme contrast and unnatural features. The sweet spot is a tradeoff, not an escalator. | Section 4 (guidance_scale parameter exploration), with a concrete comparison grid. Callback to "contrast slider" analogy from 6.3.4. |
| "More inference steps always means better quality" | This is the natural assumption: more compute = better results. The student learned in Lesson 16 that advanced samplers have a sweet spot, but the instinct is strong. With DPM-Solver, going from 20 to 100 steps wastes compute with negligible quality improvement. | Generate the same prompt/seed with DPM-Solver++ at 5, 10, 20, 50, 100 steps. At 5, quality is poor. At 10, it is recognizable. At 20, it is good. At 50, it is indistinguishable from 20. At 100, there is no visible improvement but generation takes 5x longer. The quality curve plateaus. (Originally planned up to 200 steps, but 100 already proves the plateau; 200 adds significant generation time for the same pedagogical point.) | Section 4 (num_inference_steps exploration). Callback to "More steps does NOT always help" WarningBlock from Lesson 16. |
| "Negative prompts work by subtracting concepts from the image" | The word "negative" suggests subtraction or removal. The student might think the pipeline generates an image and then removes the negative prompt concepts. The actual mechanism is directional: CFG steers TOWARD the positive prompt and AWAY FROM the negative prompt at every denoising step. Nothing is generated and then removed. | Show the CFG formula with negative prompt: epsilon_cfg = epsilon_neg + w * (epsilon_cond - epsilon_neg). The negative prompt replaces the unconditional prediction, changing the DIRECTION of guidance at every step. It is a steering mechanism, not an eraser. Generate with negative_prompt="blurry, low quality" and observe that the image is not "the same image minus blurriness" but a fundamentally different generation steered away from blurry outputs from the first step onward. | Section 4 (negative_prompt exploration), with the formula callback. |
| "The prompt is just keywords the model matches" | Software engineers are used to keyword search. The student might treat the prompt like a search query: "cat, beach, sunset" as equivalent to "a cat sitting on a beach at sunset." But cross-attention operates on token embeddings where word order, articles, and phrasing all matter because CLIP produces contextual embeddings (self-attention within the text encoder makes each token's representation context-dependent). | Generate "a cat sitting on a beach" vs "cat beach sitting a on" vs "a beach sitting on a cat." The first produces the expected image. The second is degraded because the tokenizer and CLIP's self-attention produce different embeddings for scrambled tokens. The third may produce something unexpected because the syntactic structure guides cross-attention differently. Prompts are sentences, not keyword bags. | Section 4 (prompt exploration), with callback to CLIP's contextual embeddings and cross-attention per-spatial-location mechanism from 6.3.4. |

### Examples planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Parameter mapping table: every `pipe()` parameter linked to its source concept and lesson | Positive (anchor) | The centerpiece of the lesson. A comprehensive table showing: `prompt` -> CLIP encoding -> cross-attention (6.3.3-6.3.4), `negative_prompt` -> unconditional CFG direction (6.4.1), `guidance_scale` -> CFG w parameter (6.3.4), `num_inference_steps` -> sampler step count (6.4.2), `scheduler` -> sampler choice (6.4.2), `height`/`width` -> latent dimensions via 8x downsampling (6.3.5), `generator` -> seed -> initial z_T (6.4.1). Each row connects API to concept to source. | This table IS the lesson. It makes the "you know what everything means" claim concrete and verifiable. Each parameter becomes a callback to a specific lesson the student completed. The table format enables the student to use it as a reference after the lesson. |
| Guidance scale comparison grid: same prompt/seed at w=3, 7.5, 15, 25 | Positive (concrete) | Shows the CFG tradeoff with real images. The student predicts what will happen (from their knowledge of the CFG formula), then verifies by generating. At w=3, the unconditional prediction dominates (vague). At w=7.5, balanced. At w=15+, the extrapolation overshoots (oversaturated, distorted). | This is the most impactful parameter for the student to understand viscerally. They learned the formula; now they SEE the formula's effect on real images. The prediction-before-verification pattern reinforces that the student genuinely understands the parameter, not just memorized a default value. |
| Step count comparison grid: same prompt/seed/sampler at 5, 10, 20, 50, 100 steps | Positive (concrete) | Shows the quality plateau with real images. Reinforces the sampler lesson: DPM-Solver at 20 steps is the sweet spot, more steps waste compute. The student sees that the quality curve flattens, connecting to the ODE trajectory concept (once you are close to the endpoint, additional steps cannot improve quality). | Completes the mental model from Lesson 16. The student saw sampler comparisons in the notebook but with a focus on comparing samplers. This comparison focuses on the practical question "how many steps should I actually use?" with a real quality curve. |
| Negative prompt A/B: same prompt with and without `negative_prompt="blurry, low quality, deformed"` | Positive (concrete) | Shows negative prompts as directional steering, not subtraction. The student can see that the negative prompt does not "remove blur from an otherwise blurry image" -- it steers the entire generation in a different direction from step 1. | Connects the INTRODUCED concept from Lesson 15 to hands-on experience. The student knew the formula; now they see it work on real images and can reason about what negative prompts to write based on the CFG formula. |
| Scrambled prompt: "a cat sitting on a beach at sunset" vs "cat beach sunset sitting a on" | Negative | Disproves misconception #5 (prompts are keywords). Shows that CLIP produces different embeddings for scrambled text because its transformer's self-attention makes embeddings context-dependent. Cross-attention then attends to different representations. Prompts are sentences, not keyword bags. | This is a practical lesson the student needs for real usage. Understanding WHY prompts are sentences (not keywords) connects back to how CLIP and cross-attention work, reinforcing the "I understand the system" theme. The scrambled prompt produces noticeably different/worse results, making the distinction visceral. |

---

## Phase 3: Design

### Narrative arc

For 16 lessons, the student has been building. They learned to compress images (VAE), to destroy and reconstruct (diffusion), to condition on time and text (U-Net, CLIP, cross-attention, CFG), to work in latent space, to assemble the full pipeline, and to choose a sampler. Each lesson added one piece at a time, with the promise that it would all come together. This is the lesson where it comes together. The student opens a Colab notebook, loads Stable Diffusion with three lines of code, and types a prompt. An image appears. But unlike the thousands of people who use Stable Diffusion as a black box, this student knows what happened inside. `guidance_scale=7.5` is not a default they copied from a tutorial -- it is the w in the CFG formula they implemented in Lesson 13. `num_inference_steps=20` is not a speed setting -- it is 20 DPM-Solver steps along the ODE trajectory they learned about in Lesson 16. `negative_prompt` is not a magic feature -- it is the CFG formula with a different unconditional input, exactly as they saw in Lesson 15. Every parameter maps to a concept they built from scratch. The lesson is not a diffusers tutorial. It is a victory lap. The student proves to themselves that they understand every parameter by predicting what each change will do before running the code, then verifying their predictions with real generated images.

### Modalities planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Symbolic (table/reference) | The parameter mapping table: each row links an API parameter to its internal behavior, the concept it controls, and the source lesson where the student learned it. This is a structured reference that the student can revisit after the lesson. Columns: Parameter, What It Controls, Internal Behavior, Source Concept, Source Lesson(s). | This lesson's core insight is a mapping, not a mechanism. A table is the right representation because the student is connecting two columns (API surface and conceptual understanding) that they already know individually. The table makes the connections explicit and scannable. |
| Concrete example (generated images) | Multiple comparison grids: guidance scale sweep, step count sweep, negative prompt A/B, prompt wording comparison, seed variation, resolution comparison. Each grid shows real generated images with one parameter varied and all others held constant. | This is the lesson where abstract concepts become real images. The student has been working with formulas and diagrams for 16 lessons. Seeing that `guidance_scale=25` produces exactly the oversaturated distortion they predicted from the CFG formula is the most powerful validation of their understanding. Concrete images are the primary modality. |
| Verbal/Analogy | "Driving a car you built" -- the student is not following a tutorial for a car someone else designed. They built the engine (diffusion), the transmission (sampler), the steering (CFG), the GPS (CLIP), and the dashboard (API). Now they drive it. Every gauge on the dashboard corresponds to a part they assembled. | The analogy captures the emotional core of this lesson. Most diffusers tutorials treat the API as a black box. This student built the box. The "car you built" analogy reframes the API experience from consumption to mastery. |
| Intuitive | The "I know what this does" moment, repeated for each parameter. The lesson structures each parameter as: (1) name the parameter, (2) student predicts what it controls based on their prior knowledge, (3) student predicts what changing it will do, (4) run the code and verify. The cumulative effect of getting predictions right, parameter after parameter, builds the "I actually understand this system" feeling. | The predict-then-verify pattern is the most powerful modality for a CONSOLIDATE lesson. It does not teach new concepts -- it validates existing ones. Each correct prediction is a micro-payoff that accumulates into the macro-payoff: genuine comprehension of a state-of-the-art AI system. |

### Cognitive load assessment

- **New concepts in this lesson:** Zero. Every parameter maps to a concept already taught at DEVELOPED or INTRODUCED depth. The only new information is API syntax (function names, parameter names), which is factual, not conceptual.

- **Previous lesson load:** STRETCH (samplers-and-efficiency, three genuinely new concepts)
- **This lesson's load:** CONSOLIDATE -- appropriate. After STRETCH, the student needs cognitive relief. This lesson provides it through hands-on generation with familiar concepts, not new learning. The work is connecting and verifying, not understanding. The module plan explicitly describes this as "a victory lap where the student proves to themselves that they understand the system."

### Connections to prior concepts

| Prior Concept | How It Connects | Risk of Misleading? |
|---------------|----------------|--------------------|
| CFG formula and guidance_scale (6.3.4) | `guidance_scale` IS the w parameter. The student has the formula, has implemented it, and has seen the spectrum (w=1 to w=20). Now they set it in the API and see its effect on real images. | No risk. Direct 1:1 mapping. |
| Sampler comparison (6.4.2) | `scheduler` IS the sampler choice. The student has the four GradientCards (DPM-Solver++, DDIM, Euler, DDPM) with recommended step counts. Now they swap schedulers in diffusers and see the quality/speed tradeoff. | No risk. The previous lesson explicitly showed the one-liner swap. |
| Seed and z_T (6.4.1) | `generator` with manual_seed determines z_T. The student knows same prompt + same seed = same image. Now they experiment with seed variation and see that seeds control "the identity" while prompts control "the concept." | No risk. Direct mapping, already understood conceptually. |
| Negative prompts as CFG (6.4.1) | `negative_prompt` replaces the empty-string unconditional in CFG. The student knows the formula substitution. Now they see it work in practice. | Low risk. The student might over-attribute power to negative prompts (they steer but do not guarantee avoidance). Address with a brief note that negative prompts are probabilistic steering, not absolute filtering. |
| CLIP 77-token limit (6.4.1) | `prompt` is tokenized to 77 tokens. Long prompts are truncated. The student can reason about why: CLIP's fixed context window from the tokenizer padding. | No risk. New practical detail, familiar underlying concept. |
| 8x spatial downsampling (6.3.5) | `height` and `width` must be multiples of 8 because the VAE's encoder downsamples 8x in each spatial dimension. 512/8 = 64. 768/8 = 96. The student can compute the latent dimensions from the pixel dimensions. | No risk. Direct application of known compression ratio. |
| "Contrast slider" analogy (6.3.4) | Extends to `guidance_scale` in the API. "You are adjusting the contrast slider." The student can predict the visual effect. | No risk. Natural extension. |
| "Same vehicle, different route" (6.4.2) | Extends to scheduler swapping. Same model, different sampler = same vehicle, different route. | No risk. Natural extension. |

### Scope boundaries

**This lesson IS about:**
- Using the diffusers `StableDiffusionPipeline` to generate images
- Mapping every API parameter to its underlying concept and source lesson
- Exploring parameter effects through predict-then-verify experiments (guidance_scale, num_inference_steps, scheduler, negative_prompt, seed, height/width, prompt wording)
- Building practical intuition for parameter selection through hands-on experimentation
- The emotional payoff of understanding every parameter in a state-of-the-art AI system

**This lesson is NOT about:**
- New mathematical formulas or derivations (this is CONSOLIDATE)
- The internal implementation of the diffusers library (how `pipe()` is coded)
- Prompt engineering as a deep topic (the student explores prompt effects but this is not a prompt engineering course)
- Fine-tuning, LoRA, or customization (Module 6.5)
- Img2img or inpainting (Module 6.5)
- SD v1 vs v2 vs XL differences
- Advanced pipeline configurations (custom pipelines, callback functions, attention processors)
- ControlNet or other conditioning beyond text (Series 7)
- Optimizing generation speed (model compilation, attention slicing, VAE tiling)

**Depth targets:**
- Diffusers API parameter comprehension: APPLIED (student uses every parameter with understanding, can predict effects, can troubleshoot unexpected results by reasoning about the underlying concepts)
- Parameter-to-concept mapping: DEVELOPED (student can explain what each parameter controls and why, citing the underlying mechanism)
- Practical parameter selection: INTRODUCED (student has rules of thumb for good defaults and knows how to experiment systematically)

### Lesson outline

**1. Context + Constraints**
- This is the final lesson in Module 6.4 and the capstone of the Stable Diffusion pipeline arc (Modules 6.1-6.4). The student has built every concept from scratch. This lesson is where they drive the real machine.
- This is a CONSOLIDATE lesson: zero new concepts. Every parameter they will set maps to something they already know.
- Scope: using diffusers to generate images with full parameter comprehension. NOT fine-tuning, NOT advanced pipelines, NOT prompt engineering in depth.
- By the end: the student can load Stable Diffusion, generate images, and explain what every parameter does because they built each underlying concept.

**2. Hook: "You built this."**
- Type: Accumulated knowledge payoff / challenge preview.
- Show the student a simple diffusers code snippet:
  ```python
  pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
  image = pipe(
      prompt="a cat sitting on a beach at sunset",
      negative_prompt="blurry, low quality",
      guidance_scale=7.5,
      num_inference_steps=20,
      generator=torch.Generator().manual_seed(42),
      height=512,
      width=512,
  ).images[0]
  ```
- "Every parameter in this call maps to a concept you built from scratch. `guidance_scale` is the w in a formula you implemented. `num_inference_steps` is the step count along a trajectory you learned to traverse. `negative_prompt` is a substitution in the CFG formula you already know. This is not a tutorial. This is you driving a machine you built."
- Challenge: "For each parameter, predict what it controls and what happens when you change it. Then run the code and verify."

**3. Explain: The Parameter Map (DEVELOPED)**

**3a. The parameter mapping table**
- Centerpiece of the lesson. Each parameter linked to its internal behavior and source:

| Parameter | What It Controls | Internal Behavior | Source Concept | Source Lesson(s) |
|-----------|-----------------|-------------------|---------------|-----------------|
| `prompt` | What the image depicts | Tokenized (77 tokens max, padded), encoded by CLIP text encoder to [1, 77, 768] embeddings, injected via cross-attention K/V at every denoising step | CLIP encoding, cross-attention, per-spatial-location conditioning | clip (6.3.3), text-conditioning-and-guidance (6.3.4) |
| `negative_prompt` | What to steer away from | Replaces empty-string unconditional embedding in CFG: epsilon_cfg = epsilon_neg + w * (epsilon_cond - epsilon_neg) | CFG formula with negative prompt | text-conditioning-and-guidance (6.3.4), stable-diffusion-architecture (6.4.1) |
| `guidance_scale` | How strongly to follow the prompt | The w in CFG: larger w amplifies the difference between conditional and unconditional predictions | Classifier-free guidance, guidance scale tradeoff | text-conditioning-and-guidance (6.3.4) |
| `num_inference_steps` | How many denoising steps | Number of scheduler steps from z_T to z_0. Each step is 1-3 U-Net evaluations depending on solver order. With CFG, double the evaluations. | Sampler step count, ODE trajectory | samplers-and-efficiency (6.4.2) |
| `scheduler` | Which sampler algorithm | The ODE solver strategy: DPM-Solver++ (default, 20-30 steps), DDIM (deterministic, 50 steps), Euler, DDPM (1000 steps) | Sampler comparison and selection | samplers-and-efficiency (6.4.2) |
| `generator` | Random seed for z_T | torch.Generator().manual_seed(N) determines the initial random latent z_T [4, 64, 64]. Same seed + same everything else = same image | Seed determines z_T | stable-diffusion-architecture (6.4.1) |
| `height` / `width` | Image dimensions in pixels | Must be multiples of 8 (VAE 8x downsampling). 512x512 -> 64x64x4 latent. 768x512 -> 96x64x4 latent. Larger = more computation (quadratic in spatial area) | Latent diffusion compression, VAE spatial downsampling | from-pixels-to-latents (6.3.5) |

- InsightBlock: "Every row in this table connects to a specific lesson. You built each of these concepts. The API is not a black box -- it is a dashboard for a machine you understand."

**3b. Parameter-by-parameter exploration (predict-then-verify structure)**

For each parameter, the lesson follows the same structure:
1. Name the parameter and its mapping
2. Student predicts the effect of changing it (based on prior knowledge)
3. Show generated images confirming the prediction
4. Note any practical subtleties

**`guidance_scale` exploration:**
- Callback to "contrast slider" from 6.3.4.
- Prediction prompt: "At w=3, the unconditional prediction has more weight. What does this mean for the image?" (Answer: less faithful to the prompt, more 'generic.')
- Comparison grid: same prompt/seed at w=1, 3, 7.5, 15, 25.
- Practical note: most models are trained with an implicit assumption of w in the 5-15 range. Going outside this range pushes the model into territory it was not optimized for.
- ConceptBlock: "guidance_scale is not a quality dial. It is a tradeoff between prompt fidelity and image naturalness."

**`num_inference_steps` exploration:**
- Callback to sampler sweet spots from 6.4.2.
- Prediction prompt: "With DPM-Solver++, what quality difference do you expect between 20 and 100 steps?" (Answer: minimal, because DPM-Solver++ plateaus around 20-30.)
- Comparison grid: same prompt/seed at 5, 10, 20, 50, 100 steps (DPM-Solver++).
- Practical note: the default of 50 in many tutorials is a conservative holdover from DDIM. With DPM-Solver++, 20-25 steps is the sweet spot.
- WarningBlock: "More steps is not more quality. It is more compute for diminishing returns."

**`scheduler` exploration:**
- Callback to sampler comparison from 6.4.2.
- Show how to swap schedulers in diffusers (one-liner: `pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)`).
- Side-by-side: DPM-Solver++ at 20 steps vs DDIM at 50 steps vs Euler at 30 steps. Same model, same prompt, same seed. Comparable quality, different generation times.
- Practical note: DPM-Solver++ is the default for speed. DDIM is the default for reproducibility and interpolation experiments.

**`negative_prompt` exploration:**
- Callback to CFG formula substitution from 6.4.1.
- Prediction prompt: "negative_prompt replaces the empty string in CFG. What direction does this steer the generation?" (Answer: away from the negative prompt's semantic meaning, at every step.)
- A/B comparison: same prompt/seed with and without `negative_prompt="blurry, low quality, deformed, watermark"`.
- Practical note: negative prompts are most effective for steering away from common failure modes (blurriness, deformation) rather than specific content ("no dogs"). The steering is probabilistic, not absolute.
- TipBlock: "Negative prompts are not an eraser. They are a compass pointing away from undesirable directions in the generation trajectory."

**`generator` (seed) exploration:**
- Callback to z_T determinism from 6.4.1.
- Show multiple seeds with the same prompt: each produces a structurally different image of the same concept.
- Show the same seed with different prompts: structural similarity may persist (the same "noise shape" interpreted differently).
- Practical note: seeds control the "identity" of the image; prompts control the "concept." Finding a good seed for a particular composition, then refining the prompt, is a practical workflow.

**`height` / `width` exploration:**
- Callback to 8x downsampling from 6.3.5.
- Prediction prompt: "What latent tensor shape corresponds to a 768x512 image?" (Answer: 96x64x4. 768/8=96, 512/8=64.)
- Practical note: SD v1.5 was trained on 512x512. Non-square or larger resolutions may produce artifacts (repeated subjects, poor composition) because the model was not trained on those aspect ratios. Briefly note that SD XL addressed this with multi-resolution training (Series 7 scope).
- WarningBlock: "Non-square images may produce unexpected compositions. The model was trained on 512x512."

**`prompt` exploration:**
- Callback to CLIP contextual embeddings and cross-attention from 6.3.3-6.3.4.
- Scrambled prompt experiment: "a cat sitting on a beach at sunset" vs "cat beach sunset sitting a on." Different results because CLIP's transformer produces context-dependent embeddings.
- 77-token limit: demonstrate what happens with a very long prompt (truncation). The student can compute: 77 tokens is roughly 50-60 words.
- Practical note: front-load the most important concepts in the prompt because of the 77-token limit and because early tokens tend to have stronger influence through cross-attention.

**4. Check (predict-and-verify)**
- "You want to generate a detailed portrait but the result looks oversaturated with unnatural colors. Which parameter is most likely too high, and what value would you try?" (Answer: `guidance_scale` is probably too high. Try reducing from the current value toward 7-8. The oversaturation comes from CFG extrapolating too far beyond the conditional prediction.)
- "Your colleague says they get better results with 200 steps than 20 steps using DPM-Solver++. What would you investigate?" (Answer: compare the actual images at 20 and 200 steps side by side. With DPM-Solver++, the quality difference should be negligible. If they see a real difference, check whether the scheduler is actually DPM-Solver++ and not DDPM, or whether the seed is being reset differently between runs.)
- "You want the exact same image every time you run the code. What two things must you set?" (Answer: a fixed seed via `generator=torch.Generator().manual_seed(N)`, AND a deterministic scheduler like DDIM with sigma=0 or DPM-Solver++. DDPM with the same seed still introduces per-step stochastic noise that changes with generator state.)

**5. Elaborate: Practical Patterns**

**5a. The experimental workflow**
- Fix a seed and prompt. Vary one parameter at a time. This is the same controlled experiment methodology from science: change one variable, observe the effect.
- Start with defaults: DPM-Solver++ at 20 steps, guidance_scale=7.5, 512x512. Then refine: adjust guidance_scale for the right fidelity/quality tradeoff, add a negative prompt if common artifacts appear, try different seeds for different compositions.
- TipBlock: "The best workflow is not random exploration. It is systematic: fix everything except one parameter, vary it, understand its effect, then move to the next."

**5b. Common failure modes and their explanations**
- Oversaturated/distorted: guidance_scale too high (CFG extrapolating too far).
- Blurry/vague: guidance_scale too low (unconditional prediction dominating) OR too few steps (trajectory not fully traversed).
- Repeated subjects in non-square images: model trained on 512x512, larger resolutions trigger repeated patterns.
- Prompt not followed: prompt too long (truncated past 77 tokens) OR guidance_scale too low OR conflicting concepts in the prompt.
- Each failure maps to a concept the student knows. The student can diagnose these without memorizing a troubleshooting guide -- they reason from the underlying mechanisms.

**5c. What the API hides (brief)**
- The `pipe()` call wraps: tokenization, CLIP encoding, z_T sampling, the full denoising loop with CFG, VAE decoding. All in one function.
- The student has done each of these steps manually in Lesson 15's notebook. The API is convenience, not mystery.
- InsightBlock: "The gap between 'using an API' and 'understanding an API' is the gap between following a recipe and knowing why each ingredient works. You have the ingredients."

**6. Check (transfer)**
- "Your friend is generating images with a model they downloaded, and every image looks identical even though they change the prompt. They are not setting a seed. What could explain this?" (Answer: the scheduler might be deterministic AND the default seed/generator state might be the same every time. Also check: is the prompt actually being encoded differently? If CLIP encoding fails silently and always returns the same tensor, changing the prompt would have no effect. The modular pipeline means each component can fail independently.)
- "You want to generate a 1024x1024 image with SD v1.5. How much more compute will the denoising loop take compared to 512x512?" (Answer: the latent tensor is 128x128x4 instead of 64x64x4. That is 4x the spatial area, so each U-Net pass processes 4x the data. With the same number of steps, the denoising loop takes roughly 4x longer. The CLIP encoding and VAE decode also take longer but the loop dominates.)

**7. Practice -- Notebook exercises (Colab)**
- **Exercise design rationale:** This is a CONSOLIDATE lesson where the notebook IS the primary vehicle. The lesson component provides the conceptual mapping; the notebook provides the hands-on verification. Exercises should be exploratory and rewarding, not challenging. The student earns the satisfaction of generating images with understanding.
- **Exercise sequence (mostly independent after Exercise 1):**
  1. **(Guided) Load and generate your first image:** Load `StableDiffusionPipeline`, generate one image with default parameters. Display it. Then modify the prompt and generate again. The student's first experience of "I typed words and got an image, and I know what happened inside."
  2. **(Guided) Guidance scale sweep:** Fix seed and prompt. Generate at guidance_scale = 1, 3, 7.5, 15, 25. Display as a grid. Predict-before-run: "Which value will produce the most prompt-faithful image? Which will be oversaturated?" Verify against the CFG formula.
  3. **(Supported) Step count and scheduler comparison:** Fix seed and prompt. Generate at num_inference_steps = 5, 10, 20, 50, 100 with DPM-Solver++. Then swap to DDIM at 50 steps and Euler at 30 steps. Display all results in a grid with generation times. The student identifies the quality plateau and compares samplers in practice.
  4. **(Independent) Systematic exploration:** The student designs their own experiment: pick a parameter, form a hypothesis about its effect based on what they learned, design a controlled comparison (fix everything else, vary the target parameter), generate the images, and write a one-sentence conclusion. Suggested parameters: negative prompts, seeds, non-square resolutions, prompt wording. The student chooses one and executes.
- **Solutions should emphasize:** Parameter-to-concept connections (each observation should be explained via the underlying mechanism), the predict-then-verify pattern (did your prediction match?), and the practical workflow (systematic experimentation, not random exploration).

**8. Summarize**
- "Every parameter in the diffusers API maps to a concept you built:"
  - `prompt` -> CLIP encoding -> cross-attention (Lessons 12-13)
  - `guidance_scale` -> CFG w parameter (Lesson 13)
  - `num_inference_steps` -> sampler step count (Lesson 16)
  - `scheduler` -> sampler choice (Lesson 16)
  - `negative_prompt` -> CFG unconditional substitution (Lesson 15)
  - `generator` -> seed -> z_T (Lesson 15)
  - `height`/`width` -> latent dimensions via VAE (Lesson 14)
- "You did not follow a tutorial. You understood the system."
- Mental model echo: "The API is a dashboard. You built the machine behind it."

**9. Next step**
- ModuleCompleteBlock for Module 6.4: "You can now trace the complete Stable Diffusion pipeline from text to image, choose the right sampler for your use case, and generate images with full parameter comprehension."
- "Module 6.5 teaches you to customize this machine. Lesson 18 introduces LoRA: inject small trainable matrices into the frozen U-Net to teach it new styles or subjects, without retraining the 860M-parameter model from scratch."
- Preview connection: "Remember how the U-Net, CLIP, and VAE are independently trained and swappable? LoRA takes this modularity further: you can add and remove fine-tuning layers like plugins."

---

## Widget Assessment

**Widget needed:** No.

**Rationale:** This is a hands-on CONSOLIDATE lesson where the Colab notebook IS the primary interactive experience. The student generates real images with Stable Diffusion and explores parameters. No in-lesson widget can match the pedagogical power of generating actual images and seeing concept-to-parameter connections verified in real time.

The lesson uses:
- Parameter mapping table (symbolic/reference): the centerpiece connecting API to concepts
- Multiple comparison grids (concrete): guidance scale sweep, step count sweep, negative prompt A/B, prompt experiments, described in the lesson text and reproduced in the notebook
- InsightBlocks: "you built this machine," parameter-to-concept connections
- ConceptBlocks: guidance_scale as tradeoff (not quality dial), negative prompts as steering (not erasure)
- WarningBlocks: more steps is not more quality, non-square resolutions, 77-token limit
- TipBlocks: systematic experimental workflow, practical defaults
- Callbacks to mental models from all prior modules in Series 6
- Colab notebook with 4 exercises (Guided -> Guided -> Supported -> Independent)

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (zero gaps)
- [x] No multi-concept jumps in widgets/exercises (notebook exercises use pre-trained diffusers pipeline, conceptual connections only)
- [x] All gaps have explicit resolution plans (no gaps to resolve)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (16-lesson accumulated knowledge payoff, "driving a machine you built")
- [x] At least 3 modalities planned for the core concept, each with rationale (symbolic/table, concrete/images, verbal/analogy, intuitive/predict-verify -- 4 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (4 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 0 new concepts (well within limit of 3)
- [x] Every new concept connected to at least one existing concept (every parameter maps to a prior lesson)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings. Three improvement findings that would meaningfully strengthen the lesson, and three polish items.

### Findings

#### [IMPROVEMENT] — Predict-and-verify questions answer themselves before the student can think

**Location:** Sections 5a (guidance_scale), 5d (negative_prompt), 5f (height/width)
**Issue:** Several "Predict before you run" prompts immediately provide the answer in the same paragraph or the very next sentence. For example, in the guidance_scale section (line 258): "At w=3, the unconditional prediction has more weight. What does this mean for the image?" is followed immediately by the answer in the GradientCards below. More egregiously, in the negative_prompt section (line 420-425): "What direction does this steer the generation? The answer: away from the negative prompt's semantic meaning, at every denoising step." The answer is literally in the same paragraph as the question. Similarly, the height/width section (line 511-515) asks "What latent tensor shape corresponds to a 768x512 image?" and immediately answers "96 x 64 x 4 because 768/8=96 and 512/8=64."
**Student impact:** The predict-then-verify pattern is the most important pedagogical tool in this CONSOLIDATE lesson. When the answer is given immediately after the question, the student reads the answer passively instead of engaging in the prediction. The "I got it right" micro-payoff is lost. The student never actually tests their understanding against the question.
**Suggested fix:** For each predict-before-you-run prompt, separate the question from the answer with a visual break (e.g., a `<details>` reveal pattern, or defer the answer to the GradientCards/comparison grids below). The guidance_scale section is least problematic because the GradientCards serve as the reveal. The negative_prompt and height/width sections need the answer moved behind a reveal or at least placed after the student has had a moment to think. The notebook handles this well (the code cell is the "reveal"); the lesson text should mirror that pattern.

#### [IMPROVEMENT] — Missing negative example for "prompt is just keywords" misconception

**Location:** Section 5g (prompt), planned misconception #5
**Issue:** The planning document specifies a negative example: "Generate 'a cat sitting on a beach at sunset' vs 'cat beach sunset sitting a on' vs 'a beach sitting on a cat.'" The built lesson includes a ComparisonRow contrasting "Structured Prompt" vs "Scrambled Prompt" (lines 561-583), but this is a description of what would happen, not a concrete generated-image comparison or a worked example. The lesson says "CLIP produces DIFFERENT embeddings" and "Degraded or unexpected generation" without showing or demonstrating the degradation. The third variant ("a beach sitting on a cat") is omitted entirely.
**Student impact:** The misconception that prompts are keyword bags is one of the most practically important for real SD usage. Without a concrete example showing the actual difference (or at least specific details about how the images differ), the student is told the concept rather than shown it. The ComparisonRow reads more like a claim than evidence. The notebook does include a prompt wording experiment in the Exercise 4 examples, but the lesson itself lacks the concrete negative example.
**Suggested fix:** Either (a) add a brief concrete description of what the scrambled prompt would actually produce (e.g., "The scrambled version may produce a beach scene without a clear cat, or a cat that is not sitting, because the cross-attention patterns are disrupted by the different token embeddings") or (b) add the third variant ("a beach sitting on a cat") as a second negative example showing that word order changes meaning, not just quality. The ComparisonRow format is fine, but the right column needs more specificity about the failure mode.

#### [IMPROVEMENT] — Notebook Exercise 3 solution block omits common mistakes and alternative approaches

**Location:** Notebook cell-15 (Exercise 3 solution)
**Issue:** The review skill requires that "solutions include reasoning before code" and "mention common mistakes or alternative approaches." The Exercise 3 solution does include reasoning ("The key insight: all three schedulers use the same trained U-Net weights...") and explains what to observe, but the "Common mistake" section only mentions one issue (forgetting to re-seed the generator). For a Supported exercise, the solution should be more thorough. Missing: (a) what happens if you accidentally use `device="cpu"` for the generator on a GPU pipeline (produces an error or incorrect results), (b) the observation that different schedulers may reorder the results because of different internal noise handling even with the same seed, (c) alternative approach of using a loop over a list of tuples instead of separate code blocks.
**Student impact:** The student doing a Supported exercise is expected to write most of the code themselves. When they hit common errors (especially the device mismatch issue, which is a frequent diffusers stumbling block), a thorough solution block saves frustration and teaches debugging patterns.
**Suggested fix:** Add 1-2 more common mistakes to the solution block, particularly the generator device mismatch. A brief note about alternative loop structures would also help.

#### [POLISH] — Aside references "Sampling and Generation" lesson but aside title says "Seed and Structure"

**Location:** Section 5e aside, ConceptBlock (lines 487-494)
**Issue:** The ConceptBlock says "Remember from **Sampling and Generation**: early denoising steps create structure, late steps refine details." The lesson it references is correct (sampling-and-generation, 6.2.4), but the concept was more precisely DEVELOPED via the DenoisingTrajectoryWidget in that lesson. The aside could be slightly more precise by saying "Remember from **Sampling and Generation** (and the DenoisingTrajectoryWidget)" to trigger a more specific memory.
**Student impact:** Minor. The student will recognize the reference. This is about precision of callbacks, not correctness.
**Suggested fix:** No change needed unless other edits are being made nearby.

#### [POLISH] — Notebook step count sweep goes to 100 but planning doc specified up to 200

**Location:** Notebook cell-11 (step count sweep)
**Issue:** The planning document specifies "Generate the same prompt/seed with DPM-Solver++ at 5, 10, 20, 50, 100, 200 steps." The notebook only goes to 100. The lesson text (GradientCards) also only goes to 100. The omission of 200 is reasonable (100 already proves the plateau, and 200 would add significant generation time for the same pedagogical point), but it is an undocumented deviation from the plan.
**Student impact:** Negligible. 100 steps is sufficient to demonstrate the plateau.
**Suggested fix:** Update the planning document's misconception table to reflect the actual range used (5, 10, 20, 50, 100) or add a brief note explaining the deviation.

#### [POLISH] — NextStepBlock description uses raw HTML entity `&mdash;`

**Location:** Line 1001, NextStepBlock description prop
**Issue:** The `description` prop value contains `&mdash;` which is an HTML entity inside a JSX string attribute. Depending on how the `NextStepBlock` component renders this string, it may display as the literal text `&mdash;` rather than an em dash. In JSX, string props typically need the actual Unicode character or the JSX expression `{'\u2014'}` rather than HTML entities.
**Student impact:** The student would see `&mdash;` as literal text in the description instead of an em dash.
**Suggested fix:** Replace `&mdash;` in the description prop with the Unicode em dash character directly: `description="Inject small trainable matrices into the frozen U-Net to teach it new styles or subjects\u2014without retraining from scratch."` or use a JSX expression if the component accepts ReactNode.

### Review Notes

**What works well:**

The lesson achieves its stated goal as a CONSOLIDATE victory lap. The parameter mapping table (Section 4) is the strongest element -- it concretely connects every API parameter to its source concept and lesson, making the "you built this" claim verifiable. The tone throughout is motivating without being condescending. The "not a tutorial, not a black box" framing is effective and consistent.

The notebook is strong. The progression from Guided (Exercises 1-2) to Supported (Exercise 3) to Independent (Exercise 4) is well-scaffolded. Exercise 4's two detailed example experiments (negative prompt steering, prompt wording) provide excellent models for the student's own experiment without being prescriptive. The predict-before-run prompts in the notebook are better separated from their answers than in the lesson text (the code cells serve as the natural "reveal").

The em dash usage is correct throughout (unspaced `&mdash;`). All interactive elements (summary/details) have `cursor-pointer`. The Row layout is used consistently. The lesson stays within its scope boundaries.

**Pattern to watch:**

The main weakness is that the lesson text provides answers too quickly after posing predict-before-you-run questions. This is a structural issue: the lesson is showing comparison grids and GradientCards immediately after posing the question, giving the student no chance to actually predict. The notebook handles this better because the student must run code to see results. Consider whether some of the lesson text's "predict" prompts should use the `<details>` reveal pattern used in the Check sections (which work well).

The NextStepBlock `&mdash;` issue in the description prop should be checked -- it may be rendering as literal text depending on how the component handles string props vs JSX children.

---

## Review — 2026-02-14 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All three improvement findings from iteration 1 were addressed correctly. No new improvement or critical issues were introduced by the fixes. Two polish items noted (one carried from iteration 1, one newly observed).

### Verification of Iteration 1 Fixes

**[FIX VERIFIED] Predict-and-verify questions now use `<details>` reveals.**
The negative_prompt section (lines 426-439) and height/width section (lines 529-539) both use `<details>` elements with `<summary className="font-medium cursor-pointer text-primary text-sm">Reveal</summary>` to separate the prediction question from the answer. The student must click to see the answer. The guidance_scale section retains GradientCards as the implicit reveal, which is acceptable as noted in the original finding. The Predict and Verify section (Section 6) and Transfer Questions (Section 10) also use `<details>` reveals consistently. All `<summary>` elements have `cursor-pointer`. Fix is clean, no issues introduced.

**[FIX VERIFIED] Three-column prompt comparison is accurate and specific.**
The three GradientCards (lines 583-614) now include: (1) Structured Prompt with an explanation of CLIP's self-attention producing contextual embeddings and cross-attention mapping spatial positions, (2) Scrambled Prompt with specific failure modes ("a beach scene without a clear cat, or a cat that is not sitting--because cross-attention patterns are disrupted"), (3) Reversed Meaning showing semantic inversion ("CLIP's transformer encodes 'beach' as the subject and 'cat' as the surface"). This addresses both sub-findings from iteration 1: the scrambled prompt card is now specific about degradation, and the third variant ("a beach sitting on a cat") is present as a separate card showing that word order changes meaning, not just quality.

**[FIX VERIFIED] Notebook Exercise 3 common mistakes section is adequate.**
Cell-15's solution block now includes two common mistakes: (1) forgetting to re-seed the generator (with explanation that generator state is consumed during sampling), and (2) generator device mismatch (with specific guidance to match device to pipeline). Both include concrete error descriptions and fixes. This is adequate for a Supported exercise.

**[FIX VERIFIED] NextStepBlock em dash renders correctly.**
Line 1032 now uses a Unicode em dash character directly in the string: `description="Inject small trainable matrices into the frozen U-Net to teach it new styles or subjects\u2014without retraining from scratch."` (rendered as the actual em dash character in the source file). The `NextStepBlock` component renders `description` as a string prop via `{description}`, so the Unicode character renders correctly as an em dash.

### Findings

#### [POLISH] — Notebook markdown cells use spaced em dashes

**Location:** Notebook cells 9, 16, 18 (multiple instances in Exercise 2 intro, Exercise 4 description, Exercise 4 example experiments)
**Issue:** The notebook's markdown cells use spaced em dashes in several places, e.g., "**Part A — Guidance Scale Sweep**", "`negative_prompt` — How does adding...", "**Form a hypothesis** — predict what will happen". The Writing Style Rule requires em dashes with no spaces: `word—word` not `word — word`. These are visible to the student in the rendered Colab notebook.
**Student impact:** Negligible. The content reads fine. This is a formatting consistency issue between the lesson component (which uses unspaced em dashes correctly throughout) and the notebook.
**Suggested fix:** Replace spaced ` — ` with unspaced `—` in notebook markdown cells for consistency with the lesson component.

#### [POLISH] — Aside references "Sampling and Generation" without widget callback specificity

**Location:** Section 5e aside, ConceptBlock (line 501)
**Issue:** Carried from iteration 1 (polish finding #4). The ConceptBlock says "Remember from **Sampling and Generation**: early denoising steps create structure, late steps refine details." Could be slightly more precise by referencing the DenoisingTrajectoryWidget, which is where this concept was most vividly experienced.
**Student impact:** Negligible. The student will recognize the reference. The lesson name is correct.
**Suggested fix:** No change needed. Carried forward for completeness.

### Review Notes

**What works well:**

All three iteration 1 improvement findings were addressed cleanly. The `<details>` reveal pattern is now used consistently in the negative_prompt section, height/width section, Predict and Verify section, and Transfer Questions section. The pattern is well-implemented with appropriate styling (`cursor-pointer`, `text-primary`, proper spacing). The three-column prompt comparison is now one of the stronger sections of the lesson, with each card providing specific failure mode descriptions grounded in the CLIP/cross-attention mechanism. The notebook Exercise 3 solution is now thorough enough for a Supported exercise.

The fixes did not introduce any new issues. The lesson maintains its strong CONSOLIDATE character throughout. The parameter mapping table remains the centerpiece. The predict-then-verify pattern now works properly across the lesson (questions separated from answers by reveals, not immediately answered in the next sentence). The notebook is well-scaffolded with appropriate progression.

**Verdict rationale:**

Zero critical or improvement findings. The two polish items are minor formatting consistency issues that do not affect the student's learning experience. The lesson achieves its stated goal as a victory lap: the student can use every diffusers API parameter with understanding because each maps to a concept they built from scratch. The lesson is ready to ship.
