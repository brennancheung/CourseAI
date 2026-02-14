# Module 6.4: Stable Diffusion -- Record

**Goal:** The student can trace the complete Stable Diffusion pipeline from text prompt to generated image, understanding how CLIP, the conditioned U-Net, and the VAE work together as one modular system, and can use advanced samplers and the diffusers library to generate images with full understanding of every parameter.
**Status:** Complete (3 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Full SD pipeline data flow (text prompt -> tokenizer -> CLIP text encoder -> denoising loop with CFG -> VAE decode -> pixel image, with tensor shapes at every handoff) | DEVELOPED | stable-diffusion-architecture | Centerpiece of the lesson. Traced with "a cat sitting on a beach at sunset" (callback to cross-attention lesson 6.3.4). Tensor shapes at every stage: prompt -> [77] int token IDs -> [77, 768] float embeddings -> z_T [4, 64, 64] -> 50 steps (100 U-Net passes with CFG) -> z_0 [4, 64, 64] -> VAE decode -> [3, 512, 512] pixel image. Full horizontal Mermaid pipeline diagram color-coded by component. Complete inference pseudocode (~20 lines) with shape annotations and source lesson labels per line. |
| Component modularity (three independently trained models connected by tensor handoffs, not shared weights or end-to-end gradients) | INTRODUCED | stable-diffusion-architecture | Three GradientCards showing parameter counts: CLIP ~123M (contrastive loss, 400M pairs, frozen), U-Net ~860M (diffusion MSE loss, the only diffusion-trained component), VAE ~84M (perceptual + adversarial loss, frozen). Total ~1.07B but never trained together. Swappability demonstrated: can swap text encoder, VAE, or scheduler independently. WarningBlock: "Not End-to-End." |
| Training vs inference pipeline (same components, different data flow direction and roles) | INTRODUCED | stable-diffusion-architecture | ComparisonRow (amber=training, blue=inference) with 7 items per side. Four key differences highlighted: (1) training uses VAE encoder, inference does not, (2) training processes one random timestep, inference loops all, (3) training updates U-Net weights, inference does not, (4) training needs real images, inference starts from noise. WarningBlock addressing misconception that VAE encoder runs during text-to-image inference. |
| Negative prompts as CFG application (replacing empty-string unconditional embedding with CLIP encoding of undesired attributes) | INTRODUCED | stable-diffusion-architecture | Framed as "not a new mechanism" but a direct application of the known CFG formula. epsilon_cfg = epsilon_neg + w * (epsilon_cond - epsilon_neg). Steers toward positive prompt and away from negative prompt. InsightBlock: "Same Formula, Different Input." |
| CLIP tokenizer padding to 77 tokens (fixed context length with SOT/EOT special tokens, always produces [77, 768] regardless of prompt length) | INTRODUCED | stable-diffusion-architecture | Brief treatment as part of the pipeline trace. Connected to BPE tokenization from 4.1.2. Shape invariance verified in notebook (short and long prompts both produce [1, 77, 768]). The fixed shape is what makes the CLIP-to-U-Net interface standardized. |
| Seed determines z_T and therefore the generated image (same prompt + same seed = same image) | INTRODUCED | stable-diffusion-architecture | Brief explanation in the "Starting Point" section. Not practiced beyond mention. |
| U-Net internal tensor shapes in SD v1.5 (320 -> 640 -> 1280 -> 1280 channel progression, cross-attention at 16x16 producing 256x77 and at 32x32 producing 1024x77 attention matrices) | INTRODUCED | stable-diffusion-architecture | Connected to toy U-Net dimensions from 6.3.1 (64 -> 128 -> 256 -> 512). ConceptBlock: "Bigger Numbers, Same Pattern." Specific SD v1.5 shapes given but not practiced. |
| Compute cost breakdown (denoising loop dominates: 100 U-Net passes vs 1 CLIP encoding vs 1 VAE decode) | INTRODUCED | stable-diffusion-architecture | TipBlock in the VAE decode section. Verified empirically in notebook Exercise 4 with timing measurements. |
| DDIM predict-x₀-then-jump mechanism (predict clean image from noise prediction via rearranged closed-form formula, then leap to arbitrary target timestep using alpha-bar values) | DEVELOPED | samplers-and-efficiency | Centerpiece of the lesson. Two-step mechanism: (1) predict x̂₀ = (x_t - sqrt(1-ᾱ_t) * ε_θ) / sqrt(ᾱ_t), (2) leap to x_{t_next} using ᾱ_{t_next}. Two concrete examples traced: t=1000→t=800 (high noise, crude prediction) and t=200→t=0 (low noise, accurate prediction). DDIM two-hop SVG diagram. Side-by-side ComparisonRow (DDPM vs DDIM step formula). PhaseCards for the two sub-steps. Negative example: naive DDPM step-skipping produces garbage because coefficients are calibrated for adjacent timesteps only. |
| ODE perspective on diffusion (model's noise predictions at every (x_t, t) define a smooth vector field / trajectory from noise to data; samplers are different ODE solvers following the same trajectory) | INTRODUCED | samplers-and-efficiency | Bridged via gradient descent: "You have been using Euler's method since Series 1." Gradient descent update θ_new = θ_old - lr * ∇L has the same structure as Euler's method x_{t+h} = x_t + h * f(x_t, t). The model's predictions define a vector field; following the ODE from t=T to t=0 traces the generation trajectory. "DDIM IS approximately Euler's method on the diffusion ODE." Probability flow ODE mentioned for paper-reading vocabulary. |
| Euler method as ODE solver (compute direction at current point, take step of size h, repeat -- structurally identical to gradient descent) | INTRODUCED | samplers-and-efficiency | Bridged from gradient descent (APPLIED from Series 1). ConceptBlock: "An ODE says 'at every point, the direction to move is...' A solver says 'OK, I will take a step of size h in that direction, then check again.'" First-order: one model evaluation per step, assumes straight-line trajectory between steps. |
| DPM-Solver / higher-order ODE solvers (evaluate model at multiple nearby timesteps to estimate trajectory curvature, enabling larger accurate steps with fewer total evaluations) | INTRODUCED | samplers-and-efficiency | Three tiers: DPM-Solver-1 (first-order, essentially Euler), DPM-Solver-2 (second-order, two evaluations per step, accounts for curvature), DPM-Solver++ (third-order, adaptive, current standard at 15-20 steps). Driving analogy: higher-order methods "read the road ahead." TipBlock on net evaluation counts: DPM-Solver-2 at 20 steps ≈ 40 evaluations vs DDIM at 50 steps = 50 evaluations. |
| Sampler comparison and practical selection guidance (DPM-Solver++ at 20-30 steps as default, DDIM at 50 for reproducibility, Euler at 30-50 for debugging, DDPM at 1000 for max quality) | INTRODUCED | samplers-and-efficiency | Four GradientCards with recommended step counts and use cases. WarningBlock: "More steps does NOT always help" -- advanced samplers have a sweet spot, diminishing returns beyond it. ComparisonRow: deterministic vs stochastic samplers (DDIM σ=0 / Euler vs DDPM / DDIM σ>0), connected to temperature analogy from 6.2.4. |
| DDIM deterministic generation (σ=0 gives same seed = same image regardless of step count; σ parameter interpolates between DDIM and DDPM behavior) | INTRODUCED | samplers-and-efficiency | Connected to temperature analogy from sampling-and-generation (6.2.4): σ is the temperature dial. Practical benefits: reproducibility, interpolation in z_T space, A/B testing, editing workflows. |
| Sampler as inference-time choice (swapping samplers requires zero retraining; model predicts noise, sampler decides what to do with the prediction) | DEVELOPED | samplers-and-efficiency | Foundational insight established in hook section and reinforced throughout. WarningBlock: "No Retraining" with diffusers one-liner code. Three samplers demonstrated on same model/prompt/seed. GradientCards showing DDPM 1000 steps vs DDIM 50 steps vs DPM-Solver 20 steps. |
| Why DDPM requires many steps (Markov chain constraint: reverse step formula uses α_t and β_t calibrated for adjacent timesteps only; coefficients break with large step gaps) | DEVELOPED | samplers-and-efficiency | ConceptBlock: "Adjacent-Step Assumption." Negative example: naive step-skipping at every 50th timestep produces garbage because β_t does not scale linearly across jumps. Gear ratio analogy: city driving gear on the highway breaks the engine. |
| DDPM vs DDIM (elevated from MENTIONED to DEVELOPED) | DEVELOPED | samplers-and-efficiency | Previously MENTIONED in sampling-and-generation (6.2.4) as a ComparisonRow name-drop. Now DEVELOPED: student understands the mechanism (predict-x₀-then-jump vs adjacent-step), the formula difference (ᾱ vs α), the determinism difference (σ=0 vs stochastic noise), and when to use each. |
| Diffusers API parameter comprehension (every pipe() parameter mapped to its underlying concept and source lesson) | APPLIED | generate-with-stable-diffusion | Seven parameters mapped: prompt -> CLIP encoding -> cross-attention, negative_prompt -> CFG unconditional substitution, guidance_scale -> CFG w parameter, num_inference_steps -> sampler step count, scheduler -> sampler choice, generator -> seed -> z_T, height/width -> latent dimensions via VAE 8x downsampling. Student uses every parameter with understanding, predicts effects, and troubleshoots by reasoning from mechanisms. Parameter mapping table as centerpiece. |
| Parameter-to-concept mapping (explaining what each diffusers parameter controls and why, citing the underlying mechanism) | DEVELOPED | generate-with-stable-diffusion | Each parameter explored individually with predict-then-verify structure: guidance_scale as CFG w (contrast slider callback), num_inference_steps as sampler sweet spot (DPM-Solver++ plateau at 20-30), scheduler as inference-time sampler swap (one-liner code), negative_prompt as CFG formula substitution (compass not eraser), generator/seed as z_T determinism, height/width as VAE 8x constraint, prompt as CLIP contextual embeddings (sentences not keywords). |
| Practical parameter selection and experimental workflow (systematic experimentation: fix everything except one parameter, vary it, understand, move to next) | INTRODUCED | generate-with-stable-diffusion | Workflow: fix seed and prompt, start with defaults (DPM-Solver++ at 20 steps, guidance_scale=7.5, 512x512), vary one parameter at a time. Common failure mode diagnosis: oversaturation = guidance_scale too high, blurry = guidance_scale too low or too few steps, repeated subjects = non-square resolution, prompt not followed = truncation or low guidance. |
| Prompt structure effects on generation (CLIP's contextual embeddings make word order matter; prompts are sentences not keyword bags) | INTRODUCED | generate-with-stable-diffusion | Three-column GradientCard comparison: structured prompt ("a cat sitting on a beach at sunset"), scrambled prompt ("cat beach sunset sitting a on"), reversed meaning ("a beach sitting on a cat"). Each produces different results because CLIP's self-attention produces context-dependent embeddings. 77-token limit means ~50-60 words, front-load important concepts. |
| Negative prompts as directional steering (not erasure; replaces unconditional embedding in CFG, steers generation away from negative meaning at every denoising step) | DEVELOPED | generate-with-stable-diffusion | Elevated from INTRODUCED in stable-diffusion-architecture. CFG formula shown with negative prompt substitution. Details reveal pattern for student prediction. "Compass, not eraser" analogy. Student understands negative prompts are probabilistic steering at every step, not post-processing removal. |

## Per-Lesson Summaries

### stable-diffusion-architecture (Lesson 1)
**Status:** Built, reviewed (PASS on iteration 2), three polish items noted
**Cognitive load:** CONSOLIDATE (zero new concepts -- assembles all components from Modules 6.1-6.3 into one pipeline)
**Notebook:** `notebooks/6-4-1-stable-diffusion-architecture.ipynb` (4 exercises: load and inspect components, trace CLIP stage, trace one denoising step, full pipeline trace)

**Concepts taught:**
- Full SD pipeline data flow with real tensor shapes at every handoff (DEVELOPED) -- the assembly of 14 lessons of component knowledge
- Component modularity with parameter counts (INTRODUCED) -- three independently trained models, swappable via standardized tensor interfaces
- Training vs inference pipeline comparison (INTRODUCED) -- same components, different data flow
- Negative prompts as CFG application (INTRODUCED) -- not a new mechanism, direct use of the known CFG formula
- CLIP tokenizer padding, seed determinism, SD v1.5 U-Net shapes, compute cost breakdown (INTRODUCED)

**Mental models established:**
- "Three translators, one pipeline" -- CLIP translates language to geometric-meaning space, the U-Net generates in latent space, the VAE translates latent language back to pixel language. Extension of the "translator" analogy from 6.3.5 and the "shared space" concept from 6.3.3.
- "You know every line" -- every line of the inference pseudocode maps to a specific prior lesson. The annotated pseudocode with source lesson labels makes accumulated knowledge tangible and verifiable.
- "Assembly line" -- each component does one job and passes the result to the next via tensor handoffs. No component knows what the others do internally.

**Analogies used:**
- "You know every instrument. Time to hear the symphony." (hook -- accumulated knowledge payoff)
- "Three translators, one pipeline" (extending translator analogy from 6.3.5 and shared space from 6.3.3)
- "Assembly line" (modularity via tensor handoffs)
- "Same building blocks, different question" (recurring meta-pattern, final application)
- "Bigger numbers, same pattern" (SD U-Net channels vs toy U-Net channels from 6.3.1)

**How concepts were taught:**
- Hook: enumeration of all 14 prior lessons' contributions, framing this lesson as the symphony after learning every instrument. Deliberate use of the prompt "a cat sitting on a beach at sunset" as callback to 6.3.4's cross-attention attention weight table.
- CLIP stage (Stage 1): tokenizer (callback to BPE from 4.1.2), fixed padding to 77 tokens, CLIP text encoder producing [77, 768] contextual embeddings. Shape checkpoint in aside. WarningBlock: "The U-Net Never Sees Text."
- Starting point (Stage 2): z_T ~ N(0, I) with shape [4, 64, 64]. No input image during inference. Seed determines output. TipBlock explaining 4 channels.
- Denoising loop (Stage 3): per-step breakdown with five sub-operations (timestep embedding, unconditional U-Net pass, conditional U-Net pass, CFG combine, scheduler step). Each sub-operation linked to its source lesson. InsightBlock: "Three Conditioning Signals" (WHEN/WHAT/volume). WarningBlock: "CFG Is Per-Step, Not Post-Processing." Internal U-Net shapes traced (320/640/1280/1280 channels). Cross-attention matrix shapes (256x77 at 16x16, 1024x77 at 32x32).
- VAE decode (Stage 4): z_0 -> decoder -> [3, 512, 512]. Runs once. "Translator from latent to pixel language" callback. TipBlock on compute cost breakdown.
- Pipeline diagram: Mermaid horizontal data flow with tensor shapes at every arrow, color-coded by component.
- Inference pseudocode: complete ~20-line procedure with shape comments. InsightBlock mapping each line to its source lesson.
- Predict-and-verify: three questions (100 U-Net passes, text influence via cross-attention at every step, seed changes z_T only).
- One Pipeline Three Models: three GradientCards (CLIP, U-Net, VAE) with parameter counts and training details. Modularity via tensor shapes. TipBlock: "Parameter Count does not equal Importance." ComparisonRow negative examples (without CLIP, without VAE). Raw text negative example.
- Negative prompts: CFG formula with negative prompt replacing empty-string unconditional. "Not a new mechanism."
- Training vs inference: ComparisonRow (amber/blue) with four key differences. WarningBlock about VAE encoder not used during text-to-image inference.
- Transfer questions: depth map conditioning (foreshadows ControlNet), doubled VAE compression tradeoff.
- "Everything You Know" enumeration: 8 cards mapping each pipeline component to its source lesson(s).

**Misconceptions addressed:**
1. "Stable Diffusion is one big model trained end-to-end" -- WarningBlock showing three separate components with different losses, different data, different training. Parameter counts (CLIP 123M + U-Net 860M + VAE 84M) and swappability as evidence.
2. "The U-Net sees the text prompt directly" -- WarningBlock: U-Net only sees [77, 768] float tensor. CLIP is the translator. If CLIP produced the same tensor from a different language, same image.
3. "CFG happens after the denoising loop (post-processing)" -- WarningBlock and per-step breakdown showing two U-Net passes at every step. 50 steps = 100 forward passes.
4. "The VAE encoder runs during text-to-image inference" -- WarningBlock and ComparisonRow: no input image to encode during generation. z_T is random noise. Only VAE decoder runs, once, at the end. (Img2img DOES use encoder, deferred to Module 6.5.)
5. "More parameters means more important" -- TipBlock: U-Net is 860M but useless without CLIP (no text control) or VAE (48x slower). Parameter count reflects task difficulty, not importance.

**What is NOT covered (deferred):**
- Implementing the pipeline from scratch (notebook uses pre-trained diffusers components)
- Samplers beyond DDPM (DDIM, Euler, DPM-Solver -- Lesson 2: samplers-and-efficiency)
- Any new mathematical formulas or derivations (this is CONSOLIDATE)
- SD v1 vs v2 vs XL architectural differences
- LoRA, fine-tuning, customization (Module 6.5)
- Img2img, inpainting (Module 6.5)
- Training Stable Diffusion from scratch
- Prompt engineering techniques

**Review notes:**
- Iteration 1: NEEDS REVISION -- 0 critical, 1 improvement (spaced em dashes in 9 student-visible locations), 3 polish (notebook VAE scaling factor tip in Independent exercise, ComparisonRow same-color rose for both negative examples, Mermaid hardcoded dark-mode colors)
- Iteration 2: PASS -- improvement finding fixed (all spaced em dashes replaced with unspaced), 3 polish items retained as-is per recommendation

### samplers-and-efficiency (Lesson 2)
**Status:** Built, reviewed (PASS on iteration 2), two polish items noted
**Cognitive load:** STRETCH (three genuinely new concepts: DDIM predict-and-leap, ODE perspective, higher-order solvers)
**Notebook:** `notebooks/6-4-2-samplers-and-efficiency.ipynb` (4 exercises: sampler comparison, DDIM determinism, step count exploration, inspect DDIM intermediates)

**Concepts taught:**
- DDIM predict-x₀-then-jump mechanism (DEVELOPED) -- two-step formula using the closed-form formula from 6.2.2 in reverse: predict x̂₀ from noise prediction, leap to arbitrary target timestep via ᾱ values
- ODE perspective on diffusion (INTRODUCED) -- model predictions define a smooth trajectory; samplers are ODE solvers following that trajectory with different step sizes
- Euler method as ODE solver (INTRODUCED) -- bridged from gradient descent (same structure: compute direction, scale by step size, update)
- DPM-Solver / higher-order solvers (INTRODUCED) -- multiple evaluations per step to estimate curvature, fewer total evaluations for same quality
- Sampler comparison and practical guidance (INTRODUCED) -- DPM-Solver++ at 20-30 steps as default, DDIM at 50 for reproducibility
- DDIM deterministic generation with σ parameter (INTRODUCED)
- Sampler as inference-time choice, no retraining needed (DEVELOPED)
- Why DDPM requires many steps / Markov chain constraint (DEVELOPED)
- DDPM vs DDIM elevated from MENTIONED to DEVELOPED

**Mental models established:**
- "The model defines where to go. The sampler defines how to get there." -- Core mental model of the lesson. The model predicts noise at every point; the sampler decides how to use that prediction to take a step.
- "Predict and leap" -- DDIM's mechanism. At each step: "Where do I think the destination is?" (predict x̂₀), "Given that destination, where should I be at the next checkpoint?" (leap via closed-form formula). A navigator constantly recalculating the route.
- "Same vehicle, different route" -- DDPM takes every back road (drunkard's walk), DDIM takes the highway, DPM-Solver uses GPS that reads the road ahead.

**Analogies used:**
- "Predict and leap" / "navigator constantly recalculating the route" (DDIM mechanism)
- "Same vehicle, different route" / "drunkard's walk vs highway vs GPS" (sampler comparison)
- "Gradient descent IS Euler's method" (ODE bridge -- most practiced algorithm reframed)
- "Driving on a curvy road" (higher-order solvers read the road ahead, Euler misses turns at high speed)
- "Car with a fixed gear ratio" (DDPM formula calibrated for city driving / tiny steps; highway / big jumps need a different gear)
- "Same landscape, different lens" (Markov chain view vs ODE view -- both describe the same process)

**How concepts were taught:**
- Hook: three GradientCards showing DDPM at 1000 steps, DDIM at 50 steps, DPM-Solver at 20 steps -- same model, same weights, comparable results. WarningBlock: "No Retraining" with diffusers one-liner.
- Why DDPM is slow: ConceptBlock on adjacent-step assumption. Negative example: naive step-skipping produces garbage because coefficients are calibrated for tiny β_t. Gear ratio analogy.
- DDIM mechanism: derived from the closed-form formula (callback to 6.2.2). PhaseCards for the two sub-steps. SVG two-hop diagram (dashed blue predict arc, solid violet leap arc). Two concrete examples traced: t=1000→t=800 (crude prediction at high noise) and t=200→t=0 (accurate prediction at low noise, shows refinement converges). ComparisonRow: DDPM step formula vs DDIM step formula. ConceptBlock: "Alpha-bar, Not Alpha."
- Deterministic generation: σ parameter interpolates DDIM/DDPM. Connected to temperature analogy from 6.2.4. Practical benefits listed.
- Predict-and-verify #1: step size calculation, why not use first-step x̂₀ prediction directly.
- ODE perspective: bridged via gradient descent → Euler's method. ConceptBlock on ODE in plain terms. Vector field / trajectory interpretation. SVG trajectory diagram (three paths: DDPM dense/jittery polyline, DDIM smooth curve, DPM-Solver minimal curve). "DDIM IS approximately Euler's method on the diffusion ODE." Probability flow ODE mentioned for vocabulary.
- Higher-order solvers: DPM-Solver-1/2/++ progression. Driving analogy for curvature estimation. TipBlock on net evaluation counts.
- Practical guidance: four GradientCards (DPM-Solver++, DDIM, Euler, DDPM) with recommended step counts. WarningBlock: "More steps does NOT always help." ComparisonRow: deterministic vs stochastic.
- Transfer questions: solver order 10 (diminishing returns), different noise schedule (DDIM still works, just recompute ᾱ).
- Notebook exercises: (1) Guided -- same model, three samplers, compare quality and timing. (2) Guided -- DDIM determinism vs DDPM stochasticity. (3) Supported -- step count exploration with DPM-Solver at 5/10/20/50/100/200 steps. (4) Independent -- VAE-decode DDIM intermediates at 10 steps, observe coarse-to-fine, compare to DDPM at 10 steps.

**Misconceptions addressed:**
1. "Better samplers require retraining the model" -- established in hook: same weights, swappable with one line in diffusers. Reinforced with each new sampler.
2. "DDIM is a different generative model from DDPM" -- framed as "same model, different sampler." Only the reverse process changes, not the trained model.
3. "Fewer steps always means worse quality" -- nonlinear relationship. DPM-Solver at 20 steps matches DDPM at 1000. Quality-vs-steps sweet spot varies by sampler.
4. "The ODE perspective is a completely new framework" -- "Same landscape, different lens." Markov chain and ODE views describe the same process, just different descriptions of the path.
5. "DDIM works by just skipping timesteps in the DDPM algorithm" -- negative example: naive step-skipping with DDPM formula produces garbage. DDIM uses a completely different formula designed for arbitrary step sizes.

**What is NOT covered (deferred):**
- Deriving DDIM or DPM-Solver from first principles (student sees formulas and mechanism, does not derive from variational bound or probability flow ODE)
- Score-based models or SDE/ODE duality in full rigor (mentioned for vocabulary only)
- Implementing samplers from scratch (notebook uses diffusers schedulers)
- Ancestral sampling, Karras samplers, UniPC (too many variants would overwhelm)
- Training-based acceleration (distillation, consistency models -- Series 7)

**Review notes:**
- Iteration 1: MAJOR REVISION -- 1 critical (notebook Exercise 2 DDPM stochasticity claim factually wrong due to generator re-seeding), 3 improvement (missing SVG visual modality, missing second concrete DDIM example t=200→t=0, Exercise 3 scaffolding inconsistent with Supported label), 2 polish (spaced em dashes in step labels, DDPM at 50 steps quality note)
- Iteration 2: PASS -- all findings fixed correctly. Two polish items noted (alpha-bar ≈ vs = notation inconsistency, Exercise 2 title vs demonstration mechanism nuance). Lesson ready to ship.

### generate-with-stable-diffusion (Lesson 3)
**Status:** Built, reviewed (PASS on iteration 2), two polish items noted
**Cognitive load:** CONSOLIDATE (zero new concepts--the payoff moment where the student uses the real library and maps every parameter to a concept they built from scratch)
**Notebook:** `notebooks/6-4-3-generate-with-stable-diffusion.ipynb` (4 exercises: load and generate, guidance scale sweep, step count and scheduler comparison, independent experiment design)

**Concepts taught:**
- Diffusers API parameter comprehension (APPLIED)--every `pipe()` parameter mapped to its underlying concept and source lesson, with predict-then-verify validation
- Parameter-to-concept mapping (DEVELOPED)--seven parameters explored individually: prompt, negative_prompt, guidance_scale, num_inference_steps, scheduler, generator, height/width
- Practical parameter selection and experimental workflow (INTRODUCED)--systematic experimentation methodology, common failure mode diagnosis from mechanisms
- Prompt structure effects on generation (INTRODUCED)--CLIP contextual embeddings mean word order matters, 77-token limit
- Negative prompts as directional steering (elevated from INTRODUCED to DEVELOPED)--"compass, not eraser" analogy, predict-then-verify with details reveal

**Mental models established:**
- "The API is a dashboard. You built the machine behind it."--core framing for the lesson. The student is not following a tutorial; they are driving a machine they understand.
- "Driving a car you built"--engine (diffusion), transmission (sampler), steering (CFG), GPS (CLIP), dashboard (API). Every gauge corresponds to a part they assembled.
- "Recipe vs ingredients"--the gap between using an API and understanding an API is the gap between following a recipe and knowing why each ingredient works.

**Analogies used:**
- "Driving a car you built" (hook--API as dashboard for a machine the student assembled)
- "The API is a dashboard" (summary--mental model echo)
- "Recipe vs ingredients" (what the API hides--understanding vs usage)
- "Compass, not eraser" (negative prompts--directional steering, not removal)
- "Contrast slider" callback (guidance_scale--from 6.3.4)
- "Same vehicle, different route" callback (scheduler swapping--from 6.4.2)

**How concepts were taught:**
- Hook: code snippet of `pipe()` call with all seven parameters. "Every parameter in this call maps to a concept you built from scratch." Challenge to predict effects before running.
- Parameter mapping table: centerpiece of the lesson. Seven rows connecting API parameter -> what it controls -> internal behavior -> source lesson. InsightBlock: "Not a Black Box."
- Parameter-by-parameter exploration: each parameter gets its own section with predict-then-verify structure. guidance_scale: five GradientCards (w=1,3,7.5,15,25) showing CFG tradeoff. num_inference_steps: five GradientCards (5,10,20,50,100 steps) showing DPM-Solver++ plateau. scheduler: ComparisonRow (DPM-Solver++ vs DDIM) with practical guidance. negative_prompt: CFG formula with substitution, `<details>` reveal for prediction. generator/seed: code showing multiple seeds with same prompt. height/width: `<details>` reveal for latent dimension calculation. prompt: three GradientCards (structured vs scrambled vs reversed meaning).
- Predict and verify: three scenario-based questions with `<details>` reveals (oversaturation diagnosis, step count misconception, reproducibility requirements).
- Practical patterns: systematic experimental workflow (fix everything, vary one parameter). Common failure modes mapped to underlying concepts (oversaturation = CFG too high, blurry = CFG too low or few steps, repeated subjects = non-square, prompt not followed = truncation/low guidance).
- What the API hides: five stages (tokenization, CLIP encoding, z_T sampling, denoising loop with CFG, VAE decoding) wrapped in one function call. Student did each manually in prior lessons.
- Transfer questions: two `<details>` reveal questions (identical images despite prompt changes, 1024x1024 compute cost).
- Module completion: ModuleCompleteBlock listing four achievements. Preview of Module 6.5 (LoRA) with connection to modularity concept.

**Misconceptions addressed:**
1. "The diffusers library is a black box I'm just calling"--parameter mapping table makes every connection explicit. Student predicts before running, proving understanding.
2. "Higher guidance_scale always means better images"--five GradientCards showing w=1 through w=25. ConceptBlock: "Not a Quality Dial." Callback to contrast slider analogy.
3. "More inference steps always means better quality"--five GradientCards showing 5 through 100 steps. WarningBlock: "More Steps Is Not More Quality." Quality curve plateaus at 20 for DPM-Solver++.
4. "Negative prompts work by subtracting concepts from the image"--CFG formula with substitution shown. TipBlock: "Compass, Not Eraser." Steering at every step, not post-processing removal.
5. "The prompt is just keywords the model matches"--three GradientCards: structured, scrambled, reversed meaning. CLIP's self-attention produces context-dependent embeddings. InsightBlock: "Why Prompts Are Sentences."

**What is NOT covered (deferred):**
- New mathematical formulas or derivations (this is CONSOLIDATE)
- Internal implementation of the diffusers library
- Prompt engineering as a deep topic
- Fine-tuning, LoRA, or customization (Module 6.5)
- Img2img, inpainting, ControlNet
- SD v1 vs v2 vs XL differences
- Advanced pipeline configurations (custom pipelines, callbacks, attention processors)
- Optimizing generation speed (model compilation, attention slicing, VAE tiling)

**Review notes:**
- Iteration 1: NEEDS REVISION--0 critical, 3 improvement (predict-and-verify questions answer themselves before student can think, missing negative example for prompt-is-keywords misconception, notebook Exercise 3 solution omits common mistakes), 3 polish (aside reference specificity, notebook step count range deviation from plan, NextStepBlock em dash entity)
- Iteration 2: PASS--all three improvement findings fixed correctly (details reveals added, three-column prompt comparison with specific failure modes, notebook common mistakes expanded). Two polish items noted (notebook spaced em dashes, aside reference specificity). Lesson ready to ship.
