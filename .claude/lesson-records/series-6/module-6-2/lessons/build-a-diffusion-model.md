# Lesson: Build a Diffusion Model

**Module:** 6.2 (Diffusion)
**Series:** 6 (Stable Diffusion)
**Position:** Lesson 5 of 5 (Lesson 9 overall in series)
**Slug:** `build-a-diffusion-model`
**Cognitive load:** CONSOLIDATE
**Previous lesson:** sampling-and-generation (BUILD)
**Next lesson:** Module 6.3 (Latent Diffusion)
**Notebook:** Full Colab notebook -- this IS the lesson

---

## Phase 1: Orient (Student State)

The student has completed all four theory lessons of Module 6.2. They understand diffusion conceptually, mathematically, and algorithmically. They have never written a single line of diffusion code. This lesson closes that gap: implement everything from scratch, train on real data, generate images, and feel the pain of pixel-space diffusion firsthand.

### Relevant Concepts with Depths

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Forward process (gradual noise destruction) | INTRODUCED | the-diffusion-idea | Knows the concept. Has interacted with widgets showing noise progression. Has NOT implemented it in code. |
| Closed-form formula q(x_t\|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon | DEVELOPED | the-forward-process | Derived step by step, verified with 1D pixel walkthrough, used in notebook exercises for learning-to-denoise. Can use the formula as a tool. Has written code to apply it to images (learning-to-denoise notebook Exercise 1). |
| Alpha-bar (cumulative signal fraction) | DEVELOPED | the-forward-process | Understands as the signal-to-noise dial. Has computed it from a beta schedule in notebook code. |
| Noise schedule (beta_t) | DEVELOPED | the-forward-process | Understands as a design choice. Knows linear and cosine schedules. Has NOT implemented a schedule from scratch. |
| Variance-preserving formulation | DEVELOPED | the-forward-process | Understands why signal is scaled down before noise added. Total variance stays at 1. |
| DDPM training objective (predict epsilon, MSE loss) | DEVELOPED | learning-to-denoise | Knows the model predicts noise. Knows MSE is the loss. Has written a skeleton training step in notebook (learning-to-denoise Exercise 3). |
| DDPM training algorithm (7-step procedure) | DEVELOPED | learning-to-denoise | Can trace all 7 steps. Has written pseudocode-level implementation in notebook. |
| DDPM reverse step formula | DEVELOPED | sampling-and-generation | x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta) + sigma_t * z. Can trace through it numerically. Has NOT implemented it. |
| DDPM sampling algorithm (full loop from x_T to x_0) | DEVELOPED | sampling-and-generation | Knows the 3-phase algorithm (sample x_T, loop from T to 1, return x_0). Knows the t=1 special case (z=0). Has NOT implemented it. |
| Stochastic noise injection during sampling | DEVELOPED | sampling-and-generation | Understands why noise is added back (exploration, diversity, preventing blurry averages). Connects to temperature in LLMs. |
| 1000-step computational cost of sampling | INTRODUCED | sampling-and-generation | Knows sampling requires T forward passes. Told it takes seconds to minutes per image. Has NOT experienced it. |
| Timestep embedding (network receives t as input) | INTRODUCED | learning-to-denoise | Knows one network handles all timesteps by receiving t. Does NOT know architecture internals of how t is embedded. |
| MSE loss | APPLIED | Series 1 through learning-to-denoise | Deeply familiar. Has used nn.MSELoss many times. |
| Training loop (forward -> loss -> backward -> update) | APPLIED | Series 1-2 | Deeply familiar. Has built many training loops. |
| Encoder-decoder architecture (hourglass shape) | DEVELOPED | autoencoders (6.1) | Built an autoencoder with encoder-decoder. Understands bottleneck. The U-Net is an encoder-decoder with skip connections. |
| Conv2d / ConvTranspose2d | DEVELOPED | Series 3 + autoencoders | Has used both in CNNs and the autoencoder. Core building blocks of the U-Net. |
| PyTorch nn.Module | APPLIED | Series 2 | Has built many custom modules. Can define __init__ and forward methods fluently. |
| DataLoader and training loop patterns | APPLIED | Series 2 | Has loaded Fashion-MNIST, CIFAR-10 style datasets. Knows the for-loop-over-batches pattern. |
| Coarse-to-fine denoising progression | DEVELOPED | sampling-and-generation | Extended from INTRODUCED in the-diffusion-idea to DEVELOPED via DenoisingTrajectoryWidget. |

### Mental Models and Analogies Already Established

- "Destruction is easy; creation from scratch is impossibly hard; undoing a small step of destruction is learnable."
- "Alpha-bar is the signal-to-noise dial. The closed-form formula lets you turn that dial to any position."
- "Same building blocks, different question" -- MSE, conv layers, backprop are shared; the question is "what noise was added?"
- Open-book exam analogy for training (noisy image = question, epsilon = answer key, MSE = grade). In sampling, there is no answer key.
- Temperature analogy for stochastic noise injection (sigma > 0 = diverse, sigma = 0 = deterministic).
- "The last step commits. Every step before it explores."
- Sculpting from marble: rough shape -> details -> polish.
- Hiking with a miscalibrated compass: noise injection prevents consistently ending up in the wrong spot.

### What Was Explicitly NOT Covered

- Any diffusion code implementation (this lesson)
- U-Net architecture details -- how the network internally processes x_t and uses t (Module 6.3)
- Classifier-free guidance or conditional generation (Module 6.3)
- DDIM or accelerated samplers (Module 6.4)
- Score matching, SDEs, continuous-time formulations (out of scope)
- Latent diffusion (Module 6.3)

### Readiness Assessment

The student is thoroughly prepared. Every component of the diffusion pipeline has been taught at DEVELOPED depth. The student has partial code experience from the learning-to-denoise notebook (creating noisy images, computing MSE loss, writing a training step skeleton). The remaining implementation work is: (1) building a simple U-Net-like architecture, (2) assembling the full training loop, (3) implementing the sampling loop, and (4) running it on real data. The student has built encoder-decoders (autoencoder) and training loops (many times) before. The only genuinely new coding task is the U-Net with timestep conditioning -- and the architecture is explicitly kept minimal (this is NOT the full U-Net from Module 6.3).

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to implement and train a working pixel-space DDPM from scratch -- forward process, training loop, and sampling loop -- generating real images from MNIST and experiencing the computational cost of 1000-step sampling firsthand.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Closed-form formula q(x_t\|x_0) | DEVELOPED (must implement in code) | DEVELOPED | the-forward-process | OK | Student derived it, verified numerically, wrote code applying it in learning-to-denoise notebook. Ready to implement as a function. |
| Alpha-bar and noise schedule (beta_t) | DEVELOPED (must compute from schedule) | DEVELOPED | the-forward-process | OK | Student understands the relationships: alpha_t = 1 - beta_t, alpha_bar = cumulative product. Has computed these in notebook code. |
| DDPM training algorithm (7 steps) | DEVELOPED (must implement full loop) | DEVELOPED | learning-to-denoise | OK | Can trace all 7 steps. Wrote a skeleton training step in learning-to-denoise Exercise 3. Ready to assemble the full loop. |
| DDPM sampling algorithm (T to 0 loop) | DEVELOPED (must implement) | DEVELOPED | sampling-and-generation | OK | Can trace the algorithm, understands the reverse step formula, knows the t=1 special case. Has NOT implemented it, but has all the knowledge needed. |
| MSE loss (nn.MSELoss) | APPLIED (use directly) | APPLIED | Series 1 + learning-to-denoise | OK | Has used many times. One line of code. |
| PyTorch training loop pattern | APPLIED (must assemble) | APPLIED | Series 2 | OK | Has built many training loops. The diffusion-specific parts (sampling t, creating x_t) are new but understood conceptually. |
| Conv2d / ConvTranspose2d | DEVELOPED (must use in architecture) | DEVELOPED | Series 3 + autoencoders | OK | Has built CNNs and an autoencoder. Knows how spatial dimensions change with stride and padding. |
| Encoder-decoder architecture | DEVELOPED (must build U-Net variant) | DEVELOPED | autoencoders (6.1) | OK | Built an autoencoder with this structure. The U-Net adds skip connections, which are new but simple (concatenation). |
| Timestep embedding (network receives t) | INTRODUCED (must implement minimal version) | INTRODUCED | learning-to-denoise | GAP (small) | Student knows the concept (one network, t as input) but has never seen how t is actually provided to the network. Needs a brief explanation of a simple embedding approach. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Timestep embedding implementation | Small (student knows WHY the network receives t, just not HOW in code) | Brief guided section in the notebook. Frame as: "The network needs to know which noise level it is working at. The simplest approach: embed the timestep t into a vector using a learned linear layer, then add that vector to the feature maps at each resolution level." Show the code pattern. This is a practical tool, not deep architecture theory -- the full timestep embedding (sinusoidal positional encoding, adaptive group norm) is deferred to Module 6.3. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The U-Net must be complex and large to produce good results" | The student has seen impressive Stable Diffusion outputs. They may assume the network architecture is the source of quality and that a simple architecture will produce nothing useful. | A minimal U-Net (3-4 layers, <1M parameters) trained on MNIST for 10-20 epochs produces recognizable (if imperfect) digits. Architecture sophistication improves quality; it does not gate whether diffusion works at all. The student sees generation happen with their toy model. | Notebook setup, when introducing the architecture. Frame the simple U-Net as "enough to prove diffusion works" and set explicit expectations: "Your generated digits will be recognizable but imperfect. The architecture improvements in Module 6.3 are what take quality from 'recognizable' to 'impressive.'" |
| "Training should converge quickly since MSE is a simple loss" | MSE loss is familiar and every prior MSE-based model (linear regression, autoencoder) converged in a few epochs. The student may expect similarly fast convergence. | Pixel-space diffusion on MNIST requires many more epochs than the autoencoder did. The model must learn to denoise at ALL noise levels simultaneously, and the random timestep sampling means it sees each noise level sparsely. Training loss decreases gradually rather than dropping sharply. The student experiences this: 5 epochs is not enough, 20+ starts showing results. | During training, when the student observes the loss curve. Note this explicitly: "This will take longer than your autoencoder. The model is learning a much harder task -- denoising at 1000 different noise levels with one network." |
| "If the loss is low, generation must be good" | Every prior model had a direct relationship between loss and output quality. Lower loss = better predictions/reconstructions. | Low training loss means the model predicts noise well on average across all timesteps. But generation quality depends on how errors ACCUMULATE across 1000 sampling steps. A model can have decent average loss but produce poor samples if its errors are correlated or if certain noise levels are undertrained. The student will see: loss looks reasonable, but early samples are still messy. More training improves both loss and sample quality, but the relationship is not as direct as in classification. | After initial training, before generating samples. Frame the gap explicitly: "Low loss does not guarantee good generations -- you will see this for yourself." |
| "Sampling should be roughly as fast as training" | Training processes one image with one timestep per iteration (fast). The student may not internalize that sampling requires 1000 sequential forward passes per image until they experience the wait. | Time one training step vs one sampling run. Training step: ~10ms. Generating one image: ~10 seconds (1000 x 10ms, sequential). Generating a grid of 64 images: minutes. The student must sit and wait. This wait IS the lesson -- it is the visceral motivation for latent diffusion. | During sampling. The notebook explicitly times both operations and displays the results. The student is asked to predict the sampling time before running it, then experiences the actual wait. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Forward process implementation: create noisy images at various timesteps and display them | Positive | Verify the student's forward process code matches what they saw in widgets. Visual confirmation that the code produces the expected noise progression. | Bridges from theory to code. The student has seen noise progression in three widgets -- now they produce the same result with their own code. The visual continuity ("this looks like what the widget showed") builds confidence. |
| Training on MNIST with loss curve | Positive | The student trains a real model and watches the loss decrease. This is the first time they see diffusion training happen in practice, not theory. | MNIST is the simplest dataset where diffusion produces recognizable results. Low resolution (28x28) keeps training time manageable. The student has used MNIST throughout the course (Series 1-3), so the dataset is deeply familiar -- only the training objective is new. |
| Generate a grid of samples after minimal training (5 epochs) | Negative | Show that insufficient training produces recognizable but poor samples -- blobby, digit-like shapes without clear class identity. | The student sees that diffusion training requires patience. This negative example motivates continued training and sets realistic expectations. Also demonstrates that the model IS learning structure (not random noise), just not enough yet. |
| Generate a grid of samples after sufficient training (20+ epochs) | Positive | The payoff: recognizable, varied digits generated from pure noise. Different samples show different digits, demonstrating diversity. | The emotional climax of the module. The student built this from scratch. The digits are imperfect but real -- generated images that never existed in the training set. Callback to exploring-latent-spaces: "You sampled novel images from a VAE. Now you are sampling from a diffusion model you built yourself." |
| Timed comparison: training step vs sampling | Positive (stretch) | Quantify the 1000-step cost. One training step takes ~10ms (one timestep). One sampling run takes ~10s (all 1000 timesteps). The student calculates: generating 64 images for a grid would take ~10 minutes. | This is the deliberate pain. The student does not just hear "sampling is slow" -- they measure it, calculate it, and feel it. The wait motivates everything in Modules 6.3 and 6.4 (latent diffusion, accelerated samplers). |

---

## Phase 3: Design

### Narrative Arc

The student has spent four lessons building up every piece of the diffusion pipeline: intuition for why it works, the math of noise destruction, the training objective, and the sampling algorithm. They can trace through every step on paper. But they have never built one. This lesson is the proof -- can you take everything you have learned and make it real? The answer is yes. The student will implement the forward process (one function), build a minimal U-Net (the simplest architecture that works), write the training loop (familiar pattern, diffusion-specific data preparation), and implement the sampling loop (the algorithm from last lesson, now in PyTorch). They will train on MNIST, generate images, and see digits emerge from pure noise. This is the emotional climax of Module 6.2 -- the moment theory becomes tangible. But there is a deliberate sting in the tail. Generating one batch of images takes an uncomfortably long time. The student will time the sampling, calculate how long a reasonable image grid would take, and feel the pain of 1000-step pixel-space diffusion. That pain is not a failure of their implementation. It is a fundamental property of pixel-space DDPM. And it is exactly what motivates latent diffusion in Module 6.3: "What if you ran diffusion in a compressed latent space instead of pixel space?"

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Symbolic/Code | The entire lesson is code: forward process function, U-Net architecture, training loop, sampling loop. Every formula from the module is translated to PyTorch. | This is a CONSOLIDATE lesson. The primary modality is implementation. Every formula the student learned becomes a line of code. The translation from math to code IS the learning activity. |
| Concrete example | Generated images at various stages: after 5 epochs (messy), after 20 epochs (recognizable), noise progression from the student's own code (matching widget visuals). Timed sampling with real numbers. | The outputs ARE the concrete examples. The student sees diffusion work (or not yet work) at different stages. The timing numbers make the computational cost concrete rather than abstract. |
| Visual | Generated image grids, loss curves, noise progression strips from the student's own forward process. Side-by-side: early training vs late training samples. | The visual modality is the output of the code. The student verifies their implementation visually at every stage: "does the noise progression look right? do the generated digits look like digits?" |
| Intuitive | "You built this. Every piece came from the last four lessons. The forward process is the formula from Lesson 2. The training loop is MSE on noise -- Lesson 3. The sampling loop is the algorithm from Lesson 4. The architecture is an encoder-decoder you already know." | The CONSOLIDATE lesson's power is the "I already knew all of this" feeling. Each implementation step should callback to the lesson where it was taught, reinforcing that nothing new is being introduced. |
| Verbal/Connection | Explicit callbacks at each implementation step: "This is the closed-form formula from The Forward Process." "This is the 7-step algorithm from Learning to Denoise." "This is the reverse step from Sampling and Generation." | Grounding every code block in the lesson where the concept was taught. The student never wonders "where did this come from?" -- every piece has a named origin. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 0 genuinely new theoretical concepts. This is a CONSOLIDATE lesson -- every concept has been taught. The only new element is the minimal U-Net architecture, which is a practical implementation detail (not a conceptual novelty) kept deliberately simple.
- **New skills:** Translating diffusion formulas to PyTorch code, assembling components into a working system, debugging implementation against expected behavior.
- **Previous lesson load:** BUILD (sampling-and-generation)
- **This lesson's load:** CONSOLIDATE -- appropriate. After BUILD -> STRETCH -> BUILD -> BUILD, the module concludes with a CONSOLIDATE capstone. Zero new theory. The cognitive demand is integration, not acquisition.

### Connections to Prior Concepts

| Existing Concept | Connection | How |
|-----------------|------------|-----|
| Closed-form formula (the-forward-process) | Implemented as `q_sample()` function | "This is the formula you derived in The Forward Process. Now it is a Python function." |
| DDPM training algorithm (learning-to-denoise) | Implemented as the training loop | "These are the 7 steps from Learning to Denoise. Steps 1-4 are data preparation. Steps 5-7 are the standard loop." |
| DDPM sampling algorithm (sampling-and-generation) | Implemented as `sample()` function | "This is the algorithm from Sampling and Generation. The loop from T to 1, the t=1 special case, all of it." |
| Reverse step formula (sampling-and-generation) | The core computation inside the sampling loop | "This is the formula you traced with numbers at t=500. Now it is one line of PyTorch." |
| MSE loss (Series 1 through learning-to-denoise) | nn.MSELoss in the training loop | "The same loss from your third lesson in the entire course." |
| Encoder-decoder / autoencoder (Module 6.1) | The U-Net architecture | "Remember the autoencoder? Encoder compresses, decoder expands. The U-Net is the same shape with one addition: skip connections that pass high-resolution features from encoder to decoder." |
| Conv2d / ConvTranspose2d (Series 3 + autoencoders) | Building blocks of the U-Net | "The same layers you used in CNNs and the autoencoder." |
| VAE generation (exploring-latent-spaces) | Comparison to diffusion generation quality and speed | "You generated images from a VAE. Now from a diffusion model. Compare the quality -- and the speed." |
| 1000-step computational cost (sampling-and-generation) | Experienced firsthand via timed sampling | "We told you sampling takes seconds to minutes. Now you will measure it yourself." |

**Potentially misleading prior analogies:**
- None significant. This is a CONSOLIDATE lesson that reinforces established mental models rather than introducing concepts that might conflict with prior analogies.

### Scope Boundaries

**This lesson IS about:**
- Implementing the forward process (noise schedule, alpha_bar computation, noisy image creation)
- Building a minimal U-Net (simple encoder-decoder with skip connections and timestep conditioning)
- Implementing the DDPM training loop (all 7 steps)
- Implementing the DDPM sampling loop (T to 0)
- Training on MNIST and generating images
- Experiencing the computational cost of pixel-space sampling
- Target depth: APPLIED for the forward process, training algorithm, and sampling algorithm. The student has built it from scratch.

**This lesson is NOT about:**
- Full U-Net architecture with attention, group norm, sinusoidal embeddings (Module 6.3)
- Conditional generation or classifier-free guidance (Module 6.3)
- Latent diffusion (Module 6.3)
- DDIM or accelerated samplers (Module 6.4)
- Training on high-resolution images (pixel-space diffusion on MNIST/CIFAR only)
- Optimizing training speed or sample quality beyond "recognizable digits"
- FID scores or formal evaluation metrics
- Any new theoretical concepts -- this is pure implementation of existing knowledge

**Architecture scope:** The U-Net is deliberately minimal: 3-4 resolution levels, basic conv blocks (Conv2d + ReLU + BatchNorm), skip connections via concatenation, timestep embedding via a simple learned linear projection added to feature maps. No attention layers, no group normalization, no sinusoidal positional encoding. The architecture is "enough to prove diffusion works," not "the architecture used in real systems." Module 6.3 will build the full architecture.

### Lesson Outline

This lesson is primarily a Colab notebook. The lesson page in the app serves as a brief framing and launch point. The notebook IS the lesson.

**Lesson Page (brief):**

1. **Context + Constraints** -- "You have learned every piece of the diffusion pipeline. This lesson proves it by building one from scratch. You will implement the forward process, build a simple denoising network, write the training loop, train on MNIST, implement sampling, and generate images. Everything in this notebook comes from the last four lessons -- no new theory, just code." Set architecture expectations: "The network is deliberately simple. Real diffusion models use a more sophisticated U-Net with attention and better timestep conditioning. You will build that in Module 6.3. Today, the simplest architecture that works is enough."

2. **Hook (Challenge Preview)** -- "By the end of this notebook, you will generate images like these:" [show a grid of generated MNIST digits -- imperfect but recognizable]. "You will also discover why pixel-space diffusion was never the final answer." Frame the lesson as both a payoff and a setup: the magic of generation AND the pain of slowness.

3. **Launch** -- Colab link to the notebook.

4. **Summarize / Next Step** -- Brief wrap-up after the notebook. "You built a working diffusion model. You experienced generation from pure noise. You also experienced the 1000-step wait. Module 6.3 answers the question this pain raises: what if you ran diffusion in a compressed latent space?"

**Notebook Structure:**

The notebook has 6 parts, progressing from setup through implementation to experience:

**Part 1: Setup and Noise Schedule (Guided)**
- Import PyTorch, torchvision, matplotlib
- Load MNIST, normalize to [-1, 1] (connect to variance-preserving: "images normalized to roughly unit variance")
- Implement the linear noise schedule: compute beta_t, alpha_t, alpha_bar_t
- Verify: plot alpha_bar curve and compare to widget from The Forward Process
- Exercise: "Does your alpha_bar curve look like the one from the widget? It should start near 1 and drop to near 0."

**Part 2: Forward Process (Guided)**
- Implement `q_sample(x_0, t, noise)`: the closed-form formula as a function
- Callback: "This is the formula from The Forward Process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon"
- Visualize: apply to a real MNIST digit at t=0, 250, 500, 750, 999
- Exercise: "Your noise progression should match what you saw in the DiffusionNoiseWidget. Compare: at t=250, the digit is still recognizable. At t=750, it is mostly noise."
- Predict-before-run: "What will the image look like at t=999?" (Pure noise -- the student should know this cold by now.)

**Part 3: Simple U-Net (Guided)**
- Build a minimal U-Net-like network:
  - Encoder: 2-3 downsampling blocks (Conv2d + ReLU + BatchNorm)
  - Decoder: 2-3 upsampling blocks (ConvTranspose2d + ReLU + BatchNorm)
  - Skip connections: concatenate encoder features with decoder features at matching resolutions
  - Timestep embedding: linear layer mapping t to a vector, broadcast-added to feature maps
- Frame: "This is the autoencoder from Module 6.1 with two additions: skip connections (so fine details survive the bottleneck) and timestep conditioning (so the network knows which noise level it is working at)."
- Explain skip connections briefly: "The autoencoder's bottleneck forces compression. But for denoising, we want the decoder to have access to the encoder's high-resolution features. Skip connections pass them through directly -- like giving the decoder a cheat sheet."
- Explain timestep embedding briefly: "The network needs to know what noise level it is working at. At t=50, remove a tiny amount of noise. At t=950, hallucinate structure. We embed t into a vector and add it to the features."
- Count parameters: the student should see this is a small model (<1M params)
- This section has the most scaffolding -- the architecture is provided with clear annotations, not written from scratch by the student

**Part 4: Training Loop (Supported)**
- Implement the full DDPM training loop
- Callback: "These are the 7 steps from Learning to Denoise."
- The student fills in the diffusion-specific parts (sampling t, sampling noise, creating x_t, computing loss) within a provided loop skeleton
- Train for 5 epochs first. Plot the loss curve.
- Exercise: "Is the loss decreasing? It should drop quickly at first (the model learns to predict zero-ish noise for very noisy inputs) and then plateau to a slower descent."
- Train for 15-20 more epochs (total 20-25). Note the wall time.
- Compare loss curves: 5 epochs vs 20+ epochs

**Part 5: Sampling (Supported)**
- Implement the sampling function: the full loop from t=T to t=0
- Callback: "This is the algorithm from Sampling and Generation. The loop, the t=1 special case, the sigma_t * z term -- all of it."
- The student fills in the reverse step computation within a provided loop skeleton
- **Time it.** Before generating, ask: "How long do you think generating 64 images will take?" Let the student predict.
- Generate 64 images. Display as an 8x8 grid. Print the elapsed time.
- Exercise: "Look at your generated digits. Are they recognizable? Are they varied (different digits in different positions)? Are they perfect? (They should not be -- this is a tiny model on 28x28 images.)"
- **The timing reveal:** Compare one training step time vs one sampling run time. "One training step: ~Xms. Generating one image: ~Xs. That is a Yx slowdown."
- Calculate: "At this rate, generating 1000 images would take ___. Generating a 512x512 image (like Stable Diffusion) with a larger model would be even slower."
- WarningBlock: "This slowness is not a bug in your code. It is a fundamental property of pixel-space DDPM. Every image requires T=1000 sequential forward passes. This is why latent diffusion was invented."

**Part 6: Reflection and Bridge (Independent)**
- Compare VAE vs Diffusion generation:
  - "Remember your VAE samples from Exploring Latent Spaces? How do they compare to your diffusion samples in (a) quality and (b) speed?"
  - VAE: one forward pass through the decoder, instant. Blurry.
  - Diffusion: 1000 forward passes, slow. Sharper, more detailed.
  - "The VAE traded quality for speed. Diffusion traded speed for quality. Is there a way to get both?"
- Bridge to Module 6.3: "What if you ran diffusion in the VAE's latent space instead of pixel space? The latent space is 4x smaller. 1000 steps in a compressed space would be much faster. AND you could use the VAE's decoder to upsample the result to full resolution. This is latent diffusion -- the core idea behind Stable Diffusion. That is Module 6.3."
- No formal exercise -- this is a guided reflection that motivates the next module.

### Practice (Notebook Exercises)

The notebook IS the practice. Exercise progression:

| Part | Scaffolding | What the Student Implements | What Tests |
|------|------------|----------------------------|------------|
| 1. Setup | Guided | Noise schedule computation, alpha_bar | Can translate the math notation to code |
| 2. Forward Process | Guided | `q_sample()` function | Can implement the closed-form formula |
| 3. U-Net | Guided (provided with annotations) | Read and understand the architecture | Recognizes encoder-decoder + skip connections |
| 4. Training Loop | Supported (skeleton provided) | Fill in diffusion-specific training steps | Can assemble the 7-step algorithm in code |
| 5. Sampling | Supported (skeleton provided) | Fill in the reverse step computation | Can implement the sampling algorithm |
| 6. Reflection | Independent | Compare VAE vs Diffusion, reason about latent diffusion | Can synthesize the module's key tradeoff |

Exercises are cumulative -- each part builds on the previous. Part 3 (U-Net) is the most scaffolded because architecture is not the focus of this lesson. Parts 4-5 are the core exercises where the student demonstrates they can translate the learned algorithms to working code.

---

## Planning Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (1 small gap: timestep embedding implementation, resolved via guided notebook section)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (capstone payoff + deliberate pain)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: code, concrete, visual, intuitive, verbal/connection)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative + 1 stretch)
- [x] At least 3 misconceptions identified with negative examples (4 misconceptions)
- [x] Cognitive load = 0 new theoretical concepts (CONSOLIDATE)
- [x] Every implementation step connected to its source lesson
- [x] Scope boundaries explicitly stated (minimal architecture, MNIST only, no new theory)

---

## Review — 2026-02-10 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings that would leave the student lost, but the one critical finding (missing negative example) was explicitly planned and its absence weakens a key pedagogical beat. The improvement findings address a misleading U-Net docstring, a missing misconception, and an unused variable that creates confusion. All are fixable without restructuring.

### Findings

#### [CRITICAL] — Missing 5-epoch generation (planned negative example)

**Location:** Notebook, between Phase 1 training (cell-22) and Phase 2 training (cell-23)
**Issue:** The planning document explicitly calls for generating a grid of samples after 5 epochs as a negative example: "Show that insufficient training produces recognizable but poor samples — blobby, digit-like shapes without clear class identity." The built notebook trains for 5 epochs, plots the loss curve, then immediately moves to 15 more epochs of training. Generation only happens after 20 total epochs. The planned negative example is missing entirely.
**Student impact:** The student never sees the progression from "messy blobs" to "recognizable digits." They miss the experience of patience paying off and the visual evidence that diffusion training requires more epochs than an autoencoder. This also means Misconception #2 ("Training should converge quickly since MSE is a simple loss") is only addressed verbally (the text after cell-22 mentions it), not visually. A picture of blobby digits after 5 epochs would be far more convincing than a sentence about it.
**Suggested fix:** Add a generation cell between Phase 1 and Phase 2. Generate 16-64 images after 5 epochs and display them. Label the grid clearly: "After 5 epochs: the model IS learning, but not enough yet." Then after 20 epochs, show another grid side by side. The visual contrast is the lesson.

#### [IMPROVEMENT] — Misconception #3 not addressed ("Low loss = good generation")

**Location:** Notebook Part 4/5 transition
**Issue:** The planning document identifies misconception #3: "If the loss is low, generation must be good." The plan says to address it "after initial training, before generating samples" with the framing "Low loss does not guarantee good generations — you will see this for yourself." The built notebook never makes this point. Training loss decreases, then generation happens, and the connection between loss value and generation quality is never explicitly discussed.
**Student impact:** The student may carry forward the assumption that loss and generation quality are directly correlated. In practice, average MSE loss can look reasonable while samples are poor (errors compound across 1000 steps). This misconception would mislead the student when working with diffusion models in the future.
**Suggested fix:** Add a brief markdown cell before sampling that explicitly frames the gap: "Your training loss has decreased. Does that guarantee good generations? In classification, low loss meant high accuracy. Here the relationship is less direct — errors accumulate across 1000 sampling steps. Let's see what the model actually produces." This also connects naturally to the 5-epoch generation negative example.

#### [IMPROVEMENT] — U-Net docstring claims 3 skip connections but only has 2

**Location:** Notebook Part 3, SimpleUNet docstring (cell-14)
**Issue:** The docstring states "Skip connections: concatenate encoder features at each level" and lists "3 encoder levels... 3 decoder levels." But the architecture only has 2 skip connections: enc1 (32ch, 28x28) concatenated with dec2, and enc2 (64ch, 14x14) concatenated with dec3. The enc3 output (128ch, 7x7) passes through the bottleneck but has no skip connection to the decoder. The docstring implies symmetry ("at each level") that does not exist in the code.
**Student impact:** A student reading the docstring, then reading the forward() method, would expect to find 3 skip connections. Finding only 2 creates a moment of confusion: "Did I miss one? Is the enc3 skip connection implicit?" The student might think they misunderstand skip connections, when actually the docstring is just slightly misleading.
**Suggested fix:** Update the docstring to say "Skip connections at 2 of 3 resolution levels (enc1->dec2, enc2->dec3)" and add a brief comment in the forward method noting that enc3's features flow through the bottleneck rather than being explicitly skipped. This is a deliberate simplification, not an error, and naming it as such prevents confusion.

#### [IMPROVEMENT] — Unused `alpha_bars_device` variable creates confusion

**Location:** Notebook cell-21 (Phase 1 training)
**Issue:** Cell-21 creates `alpha_bars_device = alpha_bars.to(device)` but then passes the CPU tensor `alpha_bars` to `train_epoch()`. The solution for Step 4 (cell-19) tells the student to write `q_sample(x_0, t, alpha_bars.to(device), noise=noise)`, meaning `.to(device)` is called inside every batch iteration. The `alpha_bars_device` variable is created but never actually used anywhere in the notebook.
**Student impact:** A careful student will notice `alpha_bars_device` and wonder why it exists. They might try to use it in their solution (which would be more efficient), leading to a parameter name mismatch with the solution. Alternatively, they might be confused about whether they should be using the device tensor or calling `.to(device)` each time.
**Suggested fix:** Either (a) remove `alpha_bars_device` and let the solution handle the device transfer inside `train_epoch`, or (b) pass `alpha_bars_device` to `train_epoch` and update the solution to simply use the parameter directly. Option (b) is cleaner and more efficient.

#### [POLISH] — t-indexing notation shift between markdown and code not explicitly noted

**Location:** Notebook Part 5, cell-27 (markdown) vs cell-28/29 (code/solution)
**Issue:** The markdown description uses 1-indexed math notation: "$z = 0$ for $t = 1$ (the last step commits without noise)." The code uses 0-indexed: `if t_val > 0`. The solution block says "At the final step ($t = 0$), $z = 0$" — switching to 0-indexed. The shift from 1-indexed (matching the DDPM paper and Lesson 4) to 0-indexed (matching Python) is never explicitly called out.
**Student impact:** Minimal — the student is experienced with Python and would figure it out. But a one-line comment noting the index convention would prevent a brief moment of "wait, the lesson said t=1 but the code says t=0."
**Suggested fix:** Add a brief inline comment in the code or solution: "# Note: t_val=0 here corresponds to t=1 in the paper notation (final step)."

#### [POLISH] — Lesson page "What You Will Build" describes but does not show preview images

**Location:** Lesson page, "What You Will Build" section (line 162-188 of .tsx)
**Issue:** The planning document says: "By the end of this notebook, you will generate images like these: [show a grid of generated MNIST digits — imperfect but recognizable]." The built lesson page describes the grid in text ("you will generate an 8x8 grid of MNIST digits — 64 images that have never existed") but does not show a preview image. This is understandable since the lesson page is a framing page, but the plan called for a visual preview.
**Student impact:** A preview image would increase motivation before launching the notebook. The text-only description is adequate but less compelling than seeing actual generated digits.
**Suggested fix:** Consider adding a small static preview image of generated MNIST digits. This could be a pre-generated grid saved as a static asset. However, this is a low priority since the notebook delivers the visual payoff.

#### [POLISH] — `alpha_bars` passed as CPU tensor but needs device in `q_sample` call

**Location:** Notebook cell-18 (train_epoch function signature) and cell-19 (solution)
**Issue:** The `train_epoch` function receives `alpha_bars` as a parameter. The solution for Step 4 calls `alpha_bars.to(device)` inside the loop, converting the CPU tensor to GPU every batch. This works but is wasteful — the same conversion happens 468 times per epoch (one per batch). The function signature does not indicate whether `alpha_bars` should be on CPU or device.
**Student impact:** Negligible performance impact on MNIST, but sets a bad code habit. A student who later scales up to larger datasets might not realize this is an unnecessary per-batch operation.
**Suggested fix:** This overlaps with the `alpha_bars_device` finding above. Fixing one fixes the other. Either pass the device tensor into the function, or move the `.to(device)` call outside the loop within the function.

### Review Notes

**What works well:**
- The lesson page is an excellent framing document for a notebook-driven capstone. It is focused, motivating, and appropriately brief. The "Everything You Need" TipBlock and "No New Theory" InsightBlock set the right psychological frame for a CONSOLIDATE lesson.
- The notebook's scaffolding progression (Guided -> Supported -> Independent) is well-calibrated. Parts 1-3 build confidence before Parts 4-5 demand implementation.
- Callbacks to source lessons are thorough and specific ("This is the formula from The Forward Process," "These are the 7 steps from Learning to Denoise"). Every code block is grounded in a named lesson.
- The timing experiment is the strongest pedagogical beat in the notebook. Predict-before-run, concrete measurement, scaling calculation — this makes the abstract "sampling is slow" into a visceral experience.
- The VAE vs Diffusion reflection (Part 6) is an elegant bridge to Module 6.3. The question "What if you ran diffusion in the VAE's latent space?" is the perfect setup for latent diffusion.

**Pattern to watch:**
- The one critical finding (missing negative example) is a case where the plan specified something valuable and the build omitted it. The 5-epoch generation would strengthen the lesson significantly — it is both a negative example and a misconception-addressing moment. It is also cheap to add (one generation cell + one display cell between Phase 1 and Phase 2).

**Overall assessment:**
This is a strong capstone notebook with clear scaffolding, excellent callbacks, and a compelling emotional arc. The missing 5-epoch generation is the most impactful gap — adding it would address both the negative example requirement and misconception #3 in one move. The U-Net docstring and unused variable are straightforward fixes. After one revision pass, this lesson should be ready to ship.

---

## Review — 2026-02-10 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 2

### Verdict: NEEDS REVISION

All critical and improvement findings from iteration 1 were addressed solidly. The 5-epoch generation negative example is now present and well-framed. The "Does Low Loss Mean Good Generation?" section effectively addresses misconception #3. The U-Net docstring is now accurate, and the `alpha_bars_device` variable is properly used. One new improvement finding emerged: the `sample_quick()` pre-filled function spoils the Part 5 exercise by showing the student the complete sampling implementation before they are asked to write it.

### Iteration 1 Fixes Verification

All 7 findings from iteration 1 were addressed:

1. **CRITICAL (missing 5-epoch generation):** Fixed well. Cell-23/24 now generate 16 images after 5 epochs with clear framing ("What Does 5 Epochs Get You?"), a predict-before-run prompt, and descriptive print statements guiding the student to evaluate the output. The grid title "After 5 epochs: the model IS learning, but not enough yet" sets the right expectations. The visual contrast between 5 and 20 epochs delivers the planned negative example.

2. **IMPROVEMENT (misconception #3 -- low loss = good generation):** Fixed well. Cell-29 is a dedicated markdown section that explicitly names the misconception, explains error accumulation across 1000 steps, and connects back to the 5-epoch evidence. The framing "Loss is a necessary but not sufficient signal" is precise and memorable.

3. **IMPROVEMENT (U-Net docstring):** Fixed. The docstring now accurately states "2 decoder levels with skip connections: enc1->dec2, enc2->dec3 (enc3 feeds into the bottleneck without an explicit skip connection)." Code comments in the forward method reinforce this.

4. **IMPROVEMENT (unused alpha_bars_device):** Fixed. Cell-21 creates `alpha_bars_device` and passes it to `train_epoch()`. The solution in cell-19 correctly says "The `alpha_bars` parameter is already on the GPU (we moved it to the device before training), so no `.to(device)` call is needed here."

5. **POLISH (t-indexing):** Fixed. Cell-31 code comments now say "(Note: t_val=0 corresponds to the final step in our 0-indexed loop)" and the solution in cell-32 includes a full "Index convention note" paragraph mapping between 0-indexed code and 1-indexed paper notation.

6. **POLISH (preview images on lesson page):** Not addressed, which is fine -- this was the lowest-priority polish item and the lesson page text is adequate.

7. **POLISH (CPU tensor in train_epoch):** Fixed as a consequence of fixing finding #4.

### Findings

#### [IMPROVEMENT] — `sample_quick()` spoils the Part 5 sampling exercise

**Location:** Notebook cell-24 (between Part 4 Phase 1 and Phase 2)
**Issue:** To enable the 5-epoch generation (fixing the iteration 1 critical finding), a pre-filled `sample_quick()` function was added. This function contains the complete DDPM sampling implementation -- the reverse step formula and the noise injection with the t=0 special case -- identical to what the student is asked to fill in as YOUR CODE HERE in Part 5 (cell-31). The student reads the complete working implementation in cell-24 before reaching the exercise in cell-31 where they are supposed to implement it themselves.
**Student impact:** When the student reaches Part 5 and sees "YOUR CODE HERE: compute the denoised estimate," they can simply scroll up and copy the implementation from `sample_quick()`. The exercise loses its value as a test of whether the student can translate the sampling algorithm to code. The student who does scroll up gets the answer without thinking. The student who does not scroll up still saw the code 10-15 minutes earlier and may remember it, reducing the active recall demanded by the exercise.
**Suggested fix:** Replace `sample_quick()` with a version that obscures the implementation. Two options: (a) Import a pre-built sampling function from a helper module (e.g., `from courseai_utils import sample_preview`) so the implementation is not visible in the notebook. Since this is Colab, this would require a `!pip install` or `!wget` of a utility script, which adds friction. (b) Use a different approach entirely: instead of full sampling, show a simpler diagnostic -- generate denoised estimates at a few timesteps (single-step denoising, not the full iterative loop) to show the model is learning without revealing the full sampling loop. For example, take a noisy image at t=500 and show the model's one-step denoising prediction. This demonstrates learning without spoiling the multi-step sampling algorithm. Option (b) is pedagogically stronger because it also reinforces the difference between one-step denoising (what the model was trained to do) and iterative sampling (what produces real generation).

#### [POLISH] — Lesson page describes 6 notebook parts but the Colab link text says 6 parts with slightly different names

**Location:** Lesson page line 219, ConstraintBlock items, and the Colab link description (line 219)
**Issue:** The Colab link description (line 219) says "The notebook has 6 parts: setup and noise schedule, forward process, simple U-Net, training loop, sampling, and reflection." The ConstraintBlock (lines 78-89) lists 9 scope items mixing what the student implements with what they do NOT implement. The notebook itself has 6 clearly named parts matching the Colab link description. There is no inconsistency, but the ConstraintBlock's "NOT" items include "NOT: full U-Net with attention or sinusoidal embeddings -- that is Module 6.3" using an em dash with a space before it ("embeddings---that") which renders correctly as an mdash entity in TSX, but two of the NOT items use a Unicode em dash character directly rather than the `&mdash;` entity.
**Student impact:** Negligible. The rendering is correct either way in the browser. This is a code consistency issue rather than a student-facing issue.
**Suggested fix:** Standardize all em dashes in the ConstraintBlock to use `&mdash;` entities for consistency with the rest of the lesson component. Low priority.

#### [POLISH] — `sample_quick` uses CPU tensors for schedule values while `sample` (Part 5) uses the same pattern

**Location:** Notebook cell-24 (`sample_quick`) and cell-31 (`sample`)
**Issue:** Both sampling functions access `alphas[t_val]`, `alpha_bars[t_val]`, and `betas[t_val]` via scalar indexing on CPU tensors inside a loop running on GPU data. This works because scalar indexing returns Python floats that broadcast with CUDA tensors. However, `sigmas` is explicitly moved to device with `.to(device)` while `alphas`, `alpha_bars`, and `betas` are left on CPU. The inconsistency (some schedule values on device, some on CPU) may briefly confuse a careful student who wonders whether all tensors need to be on the same device.
**Student impact:** Minimal. The code works correctly. A student familiar with PyTorch device semantics might pause to think about why `sigmas` needs `.to(device)` but `alphas` does not (answer: `sigmas[t_val]` is used in `sigma_t * z` where both are tensors, but since scalar indexing extracts a Python float, it would actually work either way -- the `.to(device)` on sigmas is unnecessary but harmless).
**Suggested fix:** Either move all schedule tensors to device at the top of the function for consistency, or remove the `.to(device)` from `sigmas` since scalar indexing makes it unnecessary. This is a minor code clarity issue, not a correctness issue.

### Review Notes

**What works well:**
- All iteration 1 fixes were applied thoughtfully, not just mechanically. The 5-epoch generation is well-integrated with predict-before-run and clear labeling. The "Does Low Loss Mean Good Generation?" section is a standalone pedagogical beat, not just a bolted-on paragraph.
- The lesson page remains excellent. The framing is focused and motivating. The "Four Lessons, One Pipeline" section provides a clear narrative bridge from theory to implementation. The architecture expectations paragraph honestly sets the student up for imperfect but real results.
- The notebook's scaffolding progression (Guided -> Guided -> Guided -> Supported -> Supported -> Independent) is well-calibrated for a CONSOLIDATE capstone. The student builds confidence through Parts 1-3 before facing the two Supported exercises.
- Callbacks to source lessons remain thorough and specific throughout.
- The timing experiment is still the strongest pedagogical beat. The predict-before-run framing, the concrete measurement, the scaling calculation, and the explicit "this is NOT a bug" framing create a visceral experience that motivates latent diffusion.

**The one meaningful finding:**
The `sample_quick()` spoiler is the most significant remaining issue. It was introduced to fix the iteration 1 critical finding (missing 5-epoch generation), which was the right call -- the negative example is pedagogically important. But the implementation choice (pre-filling the full sampling code) has a side effect on the Part 5 exercise. The suggested fix (using single-step denoising diagnostics instead of full iterative sampling) would preserve the negative example while keeping the sampling exercise fresh.

**Overall assessment:**
This lesson has improved significantly from iteration 1. The 5-epoch generation and the misconception #3 section are strong additions. The one remaining improvement finding (sampling spoiler) is fixable without restructuring. After one more targeted fix, this lesson should pass.

---

## Review — 2026-02-10 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All critical and improvement findings from iterations 1 and 2 have been addressed effectively. The remaining findings are polish-level only. The lesson is ready to ship.

### Iteration 2 Fixes Verification

The one improvement finding from iteration 2 was resolved well:

1. **IMPROVEMENT (sample_quick() spoils Part 5 exercise):** Fixed with an elegant alternative. The `sample_quick()` function has been replaced with a `denoise_diagnostic()` function (cell-23/24) that tests the model's learning via single-step denoising predictions at several noise levels. This approach:
   - Shows the model IS learning after 5 epochs (serves the same purpose as the planned 5-epoch generation negative example)
   - Does NOT reveal the iterative sampling loop (the diagnostic uses one-step predictions, not the 1000-step algorithm)
   - Keeps the Part 5 sampling exercise genuinely fresh -- the student has never seen the reverse step loop implemented anywhere in the notebook
   - Adds pedagogical value: the diagnostic reinforces the difference between what the model was trained to do (one-step noise prediction) and what produces real generation (1000-step iterative sampling)
   - Uses the rearranged closed-form formula, which the student has seen before (algebraic equivalence from learning-to-denoise), so no new concepts are introduced

   The framing is also strong: "We haven't built the sampling loop yet (that's Part 5), so we can't generate from scratch. But we can test the model's denoising ability directly." This explicitly acknowledges that the sampling loop has not been built yet, building anticipation for Part 5.

2. **POLISH (lesson page part names vs ConstraintBlock):** Not addressed -- this was correctly identified as negligible in iteration 2. The rendering is correct in the browser.

3. **POLISH (CPU vs device tensor inconsistency in sampling functions):** No longer applicable in its original form. The `sample_quick()` function that had the inconsistency no longer exists. The `denoise_diagnostic()` function handles device placement cleanly. The `sample()` function (cell-31) has a clear comment explaining the scalar indexing pattern: "Schedule tensors stay on CPU -- scalar indexing below extracts Python floats that broadcast with CUDA tensors automatically." This is adequate.

### Findings

#### [POLISH] — Unicode em dash vs &mdash; entity inconsistency in ConstraintBlock

**Location:** Lesson page, ConstraintBlock items (lines 84, 87 of .tsx)
**Issue:** Two of the NOT items use Unicode em dash characters (`\u2014`) directly in string literals, while the rest of the lesson component uses `&mdash;` JSX entities. Lines 84 and 87 contain `\u2014` ("embeddings\u2014that" and "images\u2014MNIST"). All other em dashes in the file use `&mdash;`.
**Student impact:** None. Both render identically in the browser. This is a source code consistency issue only.
**Suggested fix:** Replace the two `\u2014` instances with the pattern used elsewhere in the component. Very low priority.

#### [POLISH] — Denoising diagnostic does not fully replace the planned "negative example" visual contrast

**Location:** Notebook, between Phase 1 and Phase 2 (cell-23/24)
**Issue:** The planning document originally called for generating a grid of samples after 5 epochs as a negative example, with a side-by-side visual contrast against 20-epoch samples: "blobby, digit-like shapes without clear class identity" vs "recognizable, varied digits." The denoising diagnostic is pedagogically sound but shows a different thing: the model's one-step denoising ability at various noise levels on a known digit, not generation from scratch. The student never sees "attempted generation with an undertrained model" as a visual contrast to "generation with a trained model."
**Student impact:** Minimal. The diagnostic effectively shows the model is learning but not enough. The emotional arc is preserved: the student sees imperfect predictions, trains more, then generates successfully. The visual contrast is between "rough denoising predictions at 5 epochs" and "clean generated digits at 20 epochs" -- different outputs but the same narrative trajectory.
**Suggested fix:** No fix needed. The denoising diagnostic is a better tradeoff than the original plan because it preserves the Part 5 exercise. The slight loss in visual contrast (no "blobby generation grid") is outweighed by the gain in exercise integrity. Document this as a deliberate deviation.

### Review Notes

**What works well:**

- The denoising diagnostic (cell-23/24) is the standout fix from this iteration. It is a pedagogically clever solution to the tension between "show early results" and "don't spoil the sampling exercise." The single-step denoising approach tests the model's learning, uses familiar math (rearranged closed-form formula), and explicitly sets up Part 5 by noting "we haven't built the sampling loop yet." This is better than the `sample_quick()` approach it replaced.

- The misconception #3 section (cell-29, "Does Low Loss Mean Good Generation?") reads well as a standalone pedagogical beat. It now connects back to the denoising diagnostic ("you already saw a hint of this"), creating a narrative thread between the 5-epoch check and the 20-epoch generation.

- The notebook's scaffolding progression is well-calibrated across all 6 parts. The student builds confidence through Guided parts (1-3), faces two substantive Supported exercises (4-5) where they fill in the diffusion-specific code, and ends with an Independent reflection (6) that bridges to the next module.

- Callbacks to source lessons remain thorough and specific throughout both the lesson page and the notebook. Every formula, algorithm, and concept is traced to its origin lesson by name.

- The timing experiment (cell-33 through cell-39) remains the strongest pedagogical beat in the notebook. Predict-before-run, concrete measurement, ratio calculation, scaling extrapolation, and the explicit "this is NOT a bug" framing create a visceral experience that will stick.

- The lesson page is appropriately brief for a notebook-driven capstone. It frames, motivates, sets expectations, and launches -- without duplicating notebook content.

**Overall assessment:**

This lesson has matured through three iterations from a strong foundation to a polished capstone. Iteration 1 identified a missing negative example and a sampling spoiler risk. Iteration 2 caught the spoiler that the fix introduced. Iteration 3 confirms that the denoising diagnostic elegantly resolves both issues simultaneously. The lesson is ready to ship. Only two polish items remain, neither of which affects the student experience.
