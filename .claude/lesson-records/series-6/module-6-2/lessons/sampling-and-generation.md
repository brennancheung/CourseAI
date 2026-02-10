# Lesson: Sampling and Generation (sampling-and-generation)

**Module:** 6.2 (Diffusion)
**Series:** 6 (Stable Diffusion)
**Position:** Lesson 4 of 5
**Cognitive load:** BUILD
**Previous lesson:** learning-to-denoise (BUILD)
**Next lesson:** build-a-diffusion-model (CONSOLIDATE)
**Notebook:** `notebooks/6-2-4-sampling-and-generation.ipynb` (numerical exploration of sampling coefficients, reverse steps, one-shot vs multi-step comparison, full sampling loop)

---

## Phase 1: Orient (Student State)

The student has completed three diffusion lessons and all of Module 6.1 (Generative Foundations). Here is their state on the most relevant concepts:

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Forward process (gradual noise destruction) | INTRODUCED | the-diffusion-idea | Knows the intuition: adding noise step by step until pure static. Has not implemented it. |
| Reverse process (learned iterative denoising) | INTRODUCED | the-diffusion-idea | Knows the high-level idea: chain small denoising steps to generate. Has NOT seen the algorithm, math, or any detail of how it actually works. |
| "Small steps make it learnable" | DEVELOPED | the-diffusion-idea | Core mental model. Can explain why breaking generation into many steps works when one-shot generation does not. |
| Closed-form formula q(x_t\|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon | DEVELOPED | the-forward-process | Can use it, understands the derivation, verified it with arithmetic. Knows it "teleports" to any timestep. |
| Alpha-bar (cumulative signal fraction) | DEVELOPED | the-forward-process | Understands as the signal-to-noise dial. Used the interactive widget. Knows alpha_bar starts near 1 and drops to near 0. |
| Noise schedule (beta_t) | DEVELOPED | the-forward-process | Understands as a design choice. Knows linear and cosine schedules. |
| Variance-preserving formulation | DEVELOPED | the-forward-process | Understands why signal is scaled down before noise added. Total variance stays at 1. |
| DDPM training objective (predict epsilon, MSE loss) | DEVELOPED | learning-to-denoise | Knows the model predicts the noise that was added, not the clean image. Knows the loss is MSE between predicted and actual noise. |
| DDPM training algorithm (7-step procedure) | DEVELOPED | learning-to-denoise | Can trace through all 7 steps. Knows the closed-form formula is used in training (step 4). Knows timesteps are sampled randomly, not sequentially. |
| Timestep embedding (network receives t as input) | INTRODUCED | learning-to-denoise | Knows one network handles all timesteps by receiving t. Does not know the architecture internals. |
| MSE loss (from Series 1 through DDPM) | APPLIED | Series 1 + learning-to-denoise | Deeply familiar. Has seen MSE in three contexts (regression, autoencoder, DDPM). |
| Reparameterization trick (z = mu + sigma * epsilon) | INTRODUCED | variational-autoencoders | Knows the formula and why it exists (gradients through randomness). Will recognize the same structural pattern in the reverse process formula. |

**Mental models and analogies already established:**
- "Destruction is easy; creation from scratch is impossibly hard; undoing a small step of destruction is learnable."
- "Alpha-bar is the signal-to-noise dial."
- "Same building blocks, different question" (MSE, conv layers, backprop).
- Open-book exam analogy for training (noisy image = question, epsilon = answer key, MSE = grade).
- Sculpting from marble: rough shape -> details -> polish (iterative refinement, coarse-to-fine).

**What was explicitly NOT covered that is relevant here:**
- The reverse process algorithm (how sampling actually works step by step)
- Why noise is added back during sampling (stochastic sampling)
- The reverse process posterior formula p_theta(x_{t-1} | x_t)
- The mean formula for the reverse step (how predicted epsilon becomes a denoised image)
- Any code or implementation of sampling (deferred to Lesson 5)
- U-Net architecture (deferred to Module 6.3)

**Readiness assessment:** The student is well-prepared. They understand training completely (what the model learns, how, and why) but have a conspicuous gap: they have never seen how that trained model is USED to generate. The emotional setup is perfect -- they have been told "chain denoising steps together" since Lesson 1 but have never seen the actual algorithm. This lesson closes that loop.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to trace through the DDPM reverse process (sampling) algorithm step by step, understanding how a trained noise-prediction network iteratively transforms pure Gaussian noise into a coherent image.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Closed-form formula q(x_t\|x_0) | INTRODUCED | DEVELOPED | the-forward-process | OK | Need to recognize the formula when it appears rearranged in the reverse step. Student has it at DEVELOPED -- more than sufficient. |
| Alpha-bar and alpha_t notation | INTRODUCED | DEVELOPED | the-forward-process | OK | The reverse step formula uses alpha_t, alpha_bar_t, and beta_t. Student knows all three and their relationships. |
| DDPM training objective (predict epsilon) | DEVELOPED | DEVELOPED | learning-to-denoise | OK | Must understand that the trained model outputs a noise prediction epsilon_theta(x_t, t). This is the model's sole capability that the sampling algorithm uses. |
| Variance-preserving formulation | INTRODUCED | DEVELOPED | the-forward-process | OK | Need to understand that x_t has unit variance. The reverse process preserves this property. |
| Noise schedule (beta_t) | INTRODUCED | DEVELOPED | the-forward-process | OK | Beta_t appears directly in the reverse step formulas. Student knows these are design choices with specific values. |
| "Small steps make it learnable" insight | DEVELOPED | DEVELOPED | the-diffusion-idea | OK | Core motivation for why sampling works as an iterative loop rather than one-shot. |
| Reparameterization trick pattern (signal + noise_scale * epsilon) | INTRODUCED | INTRODUCED | variational-autoencoders | OK | The reverse step formula has the same structure: predicted mean + sigma * z. Recognizing the pattern reduces cognitive load. |
| Multi-scale denoising progression (coarse-to-fine) | INTRODUCED | INTRODUCED | the-diffusion-idea | OK | Provides intuition for what the model does at different stages of sampling (early steps: global structure, later steps: fine details). |

**All prerequisites are OK.** No gaps to resolve.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Sampling is just running the forward process in reverse (subtract noise instead of add)" | The phrase "reverse process" strongly implies undoing what was done. The forward process adds noise, so the reverse must subtract it. Reinforced by the sculpting analogy (chip away to reveal). | If you could simply subtract noise, you would not need a neural network at all -- just subtract the known noise schedule. But you do not know what noise was added to THIS specific image, only the statistical properties. The forward process is a stochastic process; you cannot rewind randomness. Also: subtracting Gaussian noise from Gaussian noise is still Gaussian noise -- you get nowhere. | Section 4 (Explain) -- address immediately when introducing the reverse step formula. This is the single most important misconception. |
| "The denoised result is deterministic -- same noise input always produces the same image" | In the training loop, epsilon is sampled once and the loss is deterministic given that sample. The student may assume sampling is equally deterministic. Also, neural networks are deterministic functions (same input -> same output). | Run the sampling algorithm twice from the same x_T. If you inject fresh noise z at each step (stochastic sampling), you get different images. The added noise at each step introduces branching -- the model explores different plausible denoising paths. Only DDIM (deterministic sampling) removes this stochasticity, and that is a later development. | Section 7 (Elaborate) -- after the student has seen the full algorithm and noticed the z term. |
| "All 1000 steps do equally important work" | The student has seen beta_t as a smooth schedule and the iterative loop treats every step identically in code. Nothing so far suggests that different stages of sampling contribute differently. | Visualize the denoising process: the first 200 steps transform pure static into a blob with rough structure. Steps 200-800 refine that structure into recognizable content. The last 200 steps add fine details and textures. Skipping early steps: no image forms. Skipping late steps: image is present but blurry/rough. The contributions are wildly unequal. | Section 6 (Explore) -- the interactive visualization widget makes this visible. |
| "You need to store all 1000 intermediate images during sampling" | The iterative nature of the algorithm suggests keeping track of the full trajectory. In training, they saw x_0, x_t, and epsilon. The student might think sampling needs all intermediates. | The algorithm only ever uses x_t to compute x_{t-1}. Each step overwrites the current image. You only need to store one image at a time (plus the model and schedule parameters). Memory cost is constant regardless of the number of steps. | Section 5 (Check 1) -- predict-and-verify question about memory usage. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| One reverse step at t=500 with concrete numbers | Positive | Show the student exactly what happens in a single step: plug in x_500, get epsilon_theta prediction, compute predicted mean, add scaled noise, get x_499 | Makes the abstract formula concrete. t=500 is in the middle range where the image is recognizably noisy but has some structure. Connects to the t=500 example from learning-to-denoise (same timestep used there for the training walkthrough). |
| One reverse step at t=950 (near pure noise) vs t=50 (near clean) | Positive | Shows how the same algorithm behaves differently at extremes. At t=950, the model is hallucinating structure from almost nothing. At t=50, it is making tiny refinements to a nearly clean image. | Reinforces the multi-scale progression from the-diffusion-idea (coarse-to-fine). Makes the student feel the difference between "creating structure" and "polishing details." Extends beyond the first example to show the concept generalizes. |
| "Just subtract the noise prediction" (skip the formula, no noise injection) | Negative | Shows why naive noise subtraction does not work. If you directly compute x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t), the result is blurry/incoherent because the model's prediction is imperfect and errors accumulate catastrophically without the iterative refinement. | Addresses misconception #1 directly. The student might wonder: "if the model predicts epsilon, why not just solve for x_0 in one shot?" This is the negative example that proves iterative denoising is not just elegant but necessary. |
| Full 1000-step sampling loop from T to 0 (traced conceptually, not computed by hand) | Positive (stretch) | The student traces the algorithm as a whole, understanding the loop structure: start from x_T ~ N(0,1), repeat {predict noise, compute mean, add noise, step t-1}, stop at t=0 (no noise added at final step). | This is the payoff example -- the full picture. After seeing individual steps, the student sees how they compose. Also introduces the t=0 special case (no noise added at the final step, because there is no "next step" to correct errors). |

### Gap Resolution

No gaps found. All prerequisites are met at or above required depth.

---

## Phase 3: Design

### Narrative Arc

The student has spent three lessons building up the pieces: intuition for why diffusion works (Lesson 1), the math of noise destruction (Lesson 2), and the training objective that teaches a network to predict noise (Lesson 3). They know what the model learns but have never seen it CREATE. This lesson is the payoff -- the moment where training meets generation. The student will discover that sampling is not simply "run forward in reverse" but a careful dance: at each step, the model predicts what noise was added, computes a best guess for the slightly-less-noisy image, and then -- counterintuitively -- adds a small amount of fresh noise before moving to the next step. That added noise is not a bug; it is exploration. Without it, the model commits too early to one interpretation and produces blurry, averaged results. The emotional journey is: excitement ("I finally get to see generation!"), surprise ("wait, you ADD noise BACK?"), understanding ("oh, it is for diversity"), and anticipation tinged with impatience ("1000 steps? That is painfully slow -- there must be a faster way").

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Verbal/Analogy | "Stochastic sampling as exploring a branching path" -- at each step, the model points you toward the clean image, but the added noise lets you wander slightly, exploring different plausible interpretations. Like hiking toward a mountain: the compass (model) points the way, but you take slightly different paths each hike (noise injection), arriving at different spots on the mountain. | The counterintuitive "add noise back" step is the hardest part of this lesson to accept. An analogy grounded in physical experience makes the exploration argument intuitive before the math. |
| Symbolic | The reverse step formula: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z. Written out with every symbol defined. Then the full pseudocode algorithm (loop from T to 1). | The student needs to see the actual formula to connect training (predicts epsilon) to sampling (uses epsilon prediction to compute the denoised step). The pseudocode is the deliverable -- the thing they will implement in Lesson 5. |
| Concrete example | Numerical walkthrough of one step at t=500 with specific coefficient values. What goes in, what comes out. Not a full 28x28 image -- a simplified scalar or small-vector example to make the arithmetic visible. | Without concrete numbers, the formula is opaque. The student has seen this pattern work before: the 1D pixel walkthrough in the-forward-process was the lesson's most effective teaching moment. Repeat the pattern. |
| Visual | An interactive denoising visualization widget showing the image evolving from pure static to a recognizable image across the sampling trajectory. Key timestep snapshots (e.g., t=1000, 750, 500, 250, 100, 0) with the image at each stage. | The coarse-to-fine progression is the most visually compelling aspect of diffusion. Seeing structure emerge from noise is the "wow" moment. Also makes misconception #3 (all steps equally important) visually obvious -- the early snapshots show dramatic changes while the late ones show subtle refinements. |
| Intuitive | Why adding noise back HELPS: the model's prediction at any step is imperfect. If you commit fully to that imperfect prediction, errors accumulate and collapse to a blurry average. Adding noise keeps options open -- it is controlled uncertainty that prevents premature convergence. Connect to temperature in language models: temperature > 0 produces diverse text; temperature = 0 produces repetitive, "safe" text. | The student already has deep intuition for temperature from Module 4.1. This connection makes the stochastic sampling argument instantly recognizable: "oh, it is the same idea as temperature." |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 genuinely new concepts:
  1. The reverse step formula (how to use epsilon_theta to compute x_{t-1}) -- genuinely new
  2. Stochastic noise injection during sampling (why you add noise back) -- genuinely new
  3. The full sampling loop structure (iterate from T to 0, special case at t=0) -- procedural, builds on concepts above
- **Previous lesson load:** BUILD (learning-to-denoise)
- **This lesson's load:** BUILD -- appropriate. After a STRETCH (the-forward-process) and two BUILDs (learning-to-denoise, this), the student gets a CONSOLIDATE capstone next. The two new concepts (reverse step formula + stochastic sampling) are within the 2-3 limit, and the formula reuses familiar notation (alpha_bar, beta_t, epsilon_theta) rather than introducing new symbols.

### Connections to Prior Concepts

- **Closed-form formula (the-forward-process):** The reverse step formula is an algebraic rearrangement. The forward process says x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon. The reverse process uses the model's epsilon prediction to approximately solve for x_0, then steps backward. "You derived the formula for destroying an image. Now you are running it in reverse, using the model's noise prediction to undo the destruction."
- **Reparameterization trick (variational-autoencoders):** The reverse step has the same pattern: mean + sigma * z. The student will recognize this structure immediately. "Same pattern as the reparameterization trick: a deterministic component (predicted mean) plus a stochastic component (sigma * z)."
- **Temperature in language models (Module 4.1):** The noise injection during sampling is analogous to temperature. Temperature = 0 gives deterministic, repetitive output. Temperature > 0 gives diverse, creative output. Similarly, sigma_t = 0 gives deterministic (DDIM-like) sampling; sigma_t > 0 gives stochastic (DDPM) sampling with diverse outputs.
- **Multi-scale denoising (the-diffusion-idea):** The sculpting analogy comes alive here. Early sampling steps (high t) make coarse structural decisions. Late steps (low t) add fine details. "The sculpting analogy from Lesson 1 was a preview -- now you can see it happen step by step."
- **Training algorithm (learning-to-denoise):** The training loop uses the forward process to create noisy images; the sampling loop uses the reverse process to denoise them. They are mirror images. The 7-step training algorithm and the sampling algorithm share the same model, the same noise schedule, and the same notation.

**Potentially misleading prior analogies:**
- The "open-book exam" analogy from learning-to-denoise is about training, not sampling. In training, you KNOW the true noise (the answer key). In sampling, there IS no answer key -- the model's prediction is all you have. This distinction should be explicitly called out to prevent confusion.

### Scope Boundaries

**This lesson IS about:**
- The DDPM reverse process (sampling) algorithm, step by step
- The reverse step formula and what each term means
- Why noise is added back during sampling (stochastic sampling vs deterministic)
- The 1000-step sampling loop structure (including the t=0 special case)
- Visualizing the denoising trajectory from pure static to image
- The computational cost of sampling (1000 forward passes through the network)
- Depth target: DEVELOPED for the sampling algorithm (student can trace through it, explain each step, reason about what happens at different stages)

**This lesson is NOT about:**
- Implementing the sampling algorithm in code (that is Lesson 5, build-a-diffusion-model)
- DDIM or other accelerated samplers (out of scope for this module; mentioned only as "faster methods exist")
- Classifier-free guidance or conditional generation (Module 6.3)
- U-Net architecture or how the network internally processes x_t and t (Module 6.3)
- Score matching, SDEs, or continuous-time formulations (out of scope)
- The full variational derivation of the reverse process posterior (out of scope -- we present the formula, not its derivation from the ELBO)
- Any notebook exercises -- the student should internalize the algorithm conceptually before writing code

### Lesson Outline

**1. Context + Constraints**
- What: The DDPM sampling algorithm -- how a trained model generates images from pure noise.
- What not: No code, no implementation, no accelerated samplers. Understand the algorithm; build it next lesson.
- Why now: The student has all the pieces (forward process math, training objective, trained model) but has never seen them assembled into generation.

**2. Recap (brief)**
- Not a full recap section -- prerequisites are solid. Instead, a 2-3 sentence reminder at the start: "You have a trained model that takes a noisy image x_t and timestep t, and predicts the noise epsilon that was added. The forward process formula tells you how any clean image maps to any noise level. Now: how do you use that trained model to generate an image that has never existed?"

**3. Hook: "The Missing Piece"**
- Type: Challenge preview / puzzle.
- Setup: Display the training loop on the left and a question mark on the right. "You have a model that can predict noise at any timestep. How do you turn that into generation?" Let the student sit with this for a moment.
- Then: "Your first instinct might be: predict the noise, subtract it, done. Let's see why that does not work." This leads directly into the negative example (one-shot denoising failure).
- Why this hook: It activates the student's existing knowledge (they know what the model does) and creates a puzzle (how do I use it?). The tension between "I know all the pieces" and "I cannot yet assemble them" is motivating.

**4. Explain: The Reverse Step**
- **Negative example first (Problem before Solution):** One-shot denoising. The model predicts epsilon_theta at t=500. You could solve the forward formula for x_0 directly: x_0_hat = (x_t - sqrt(1-alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t). Show what this looks like: blurry, incoherent, because the model's prediction is imperfect and a single prediction cannot recover all the lost information. "This is why diffusion uses 1000 steps, not 1."
- **The actual reverse step formula:**
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0,1) and sigma_t = sqrt(beta_t).
- Break it down term by term:
  - (1/sqrt(alpha_t)): scaling factor (related to how much signal was preserved at this step)
  - x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta: "take the current noisy image and remove the model's predicted noise, scaled by the right amount"
  - sigma_t * z: the stochastic noise injection (address this separately in section 7)
- **Connection to the forward formula:** The mean term is derived from rearranging the forward process, using epsilon_theta as a stand-in for the true epsilon. "The forward process destroyed the image using known math. The reverse process uses the model's noise prediction to approximately undo one step of that destruction."
- **Connection to reparameterization trick:** "Notice the structure: deterministic_mean + sigma * z. You saw this in the VAE: mu + sigma * epsilon. Same pattern: a learned mean plus controlled randomness."

**5. Check 1: Predict-and-Verify**
- "You are at t=500. The model predicts epsilon_theta. You compute the predicted mean. How much of the original image's information is in that prediction?" (Answer: roughly half, since alpha_bar_500 ~ 0.05 for the linear schedule. The model is hallucinating most of the structure.)
- "How much memory does the sampling algorithm need -- does it grow with T?" (Answer: No. Only x_t at the current timestep is needed. Each step overwrites the previous. Memory is constant.) This addresses misconception #4.

**6. Explore: Denoising Visualization Widget**
- Interactive widget showing the denoising trajectory. A procedural image (same 28x28 T-shirt for visual continuity) with a timeline slider showing the image at various stages of denoising.
- Key design: show snapshots at t = 1000, 900, 800, 600, 400, 200, 100, 50, 0 (or similar). The student can step through them or use a slider.
- Visual emphasis on the coarse-to-fine progression: early steps show dramatic structural changes (blob forming from static), middle steps show recognizable object forming, late steps show subtle detail refinement.
- TryThisBlock experiments: "Step through slowly. Notice how the first few hundred steps do the most dramatic work. When does the T-shirt first become recognizable? When does the outline solidify? When do fine textures appear?"
- This directly addresses misconception #3 (all steps equally important) -- the visualization makes the unequal contributions obvious.

**7. Elaborate: Why Add Noise Back?**
- The sigma_t * z term. This is the counterintuitive part.
- **Temperature analogy:** "Remember temperature in language models? At temperature = 0, the model always picks the most likely next token -- safe, repetitive, boring. At temperature > 0, it explores less likely options -- diverse, creative, surprising. The noise injection in sampling is the same idea."
- **Why it helps:** The model's noise prediction at any step is imperfect. If you follow the predicted mean exactly (sigma = 0), you commit fully to that imperfect estimate. Errors compound across 1000 steps, and the result converges to a blurry, averaged image. Adding noise keeps options open -- the model gets to course-correct in subsequent steps, exploring different plausible interpretations of the noisy image.
- **Hiking analogy:** "Hiking toward a mountain. Your compass (the model) points toward the peak. But you add a small random jitter to each step. Different hikes reach different spots on the mountain. Without jitter, every hike follows the exact same path -- and if the compass is slightly miscalibrated, you end up consistently in the wrong place."
- **Diversity argument:** Run sampling twice from the same x_T (same initial noise). With stochastic sampling (sigma > 0), you get different images. Without it (sigma = 0, DDIM-style), you get the same image. Stochastic sampling is what gives diffusion models their variety.
- **Address misconception #2** directly: "The sampling process is not deterministic. The trained model is a deterministic function, but the noise injection at each step introduces randomness. Different noise draws lead to different images."
- **Mention (not develop):** DDIM (Denoising Diffusion Implicit Models) sets sigma = 0 for deterministic, faster sampling. This is a later development that trades diversity for speed. The student will encounter it when we discuss accelerated samplers.

**8. Check 2: Transfer Questions**
- "Your colleague says: 'Diffusion sampling is just running the forward process backward -- instead of adding noise, you subtract it.' What is wrong with this claim?" (Answer: The forward process is a known mathematical transformation. The reverse requires a trained model to predict the noise, because you do not know what noise was added. Also, the reverse step adds fresh noise for exploration, which has no analog in the forward process.)
- "If you wanted more diverse outputs, would you increase or decrease sigma_t?" (Answer: Increase -- more noise = more exploration. Connects to temperature analogy.)

**9. The Full Sampling Algorithm**
- Pseudocode for the complete algorithm:
  1. Sample x_T ~ N(0, I) (pure noise)
  2. For t = T, T-1, ..., 1:
     a. If t > 1, sample z ~ N(0, I). Else z = 0.
     b. Predict epsilon_theta = model(x_t, t)
     c. Compute x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta) + sigma_t * z
  3. Return x_0
- **The t=0 special case:** At the final step (t=1 -> x_0), z = 0. No noise is added because this is the final image -- there is no subsequent step to correct errors. "The last step commits. Every step before it explores."
- **Computational cost:** T forward passes through the neural network. For T=1000 and a U-Net, this takes seconds to minutes per image depending on resolution. "Imagine waiting 30 seconds to a minute for every single image. Now imagine generating a grid of 64 images. This is the pain that motivates everything in Module 6.3 and beyond."
- **Connection to training:** Side-by-side comparison. Training: sample one random timestep, one forward pass, one gradient update. Sampling: iterate through ALL timesteps, T forward passes, no gradients (inference only). "Training touches one timestep at a time. Sampling must visit every single one."

**10. Concrete Walkthrough: One Step at t=500**
- Use the same t=500 example from learning-to-denoise for continuity.
- Assume linear schedule. At t=500: alpha_bar_500 ~ 0.05, so sqrt(alpha_bar) ~ 0.22, sqrt(1-alpha_bar) ~ 0.97.
- beta_500 ~ 0.01 (from linear schedule), alpha_500 = 1 - beta_500 = 0.99.
- Show the formula with these numbers plugged in:
  x_499 = (1/sqrt(0.99)) * (x_500 - (0.01/sqrt(0.95)) * epsilon_theta) + sqrt(0.01) * z
  x_499 = 1.005 * (x_500 - 0.0103 * epsilon_theta) + 0.1 * z
- Interpretation: "The mean is almost exactly x_500 with a tiny noise correction subtracted. The noise injection (0.1 * z) adds a small random perturbation. Each step makes a small adjustment -- this is why you need 1000 of them."
- Second example at t=50 (near clean): beta_50 ~ 0.0002. The noise correction is tiny. The model is making minute adjustments. "At this stage, the image is nearly clean and the model is polishing details."
- Second example at t=950 (near pure noise): beta_950 ~ 0.019. The noise correction is larger relative to the signal. The model is making bold structural decisions. "At this stage, the model is deciding: is this a shoe or a shirt?"

**11. Summarize**
- The trained model predicts noise. The sampling algorithm uses that prediction to take one small step toward a cleaner image.
- The reverse step formula rearranges the forward process using the model's noise prediction in place of the true noise.
- Noise is added back at each step (except the last) to maintain diversity and prevent premature convergence -- the same principle as temperature in language models.
- The full algorithm iterates from t=T (pure noise) to t=0 (generated image), requiring T forward passes through the network.
- Coarse-to-fine: early steps create structure, late steps refine details.
- 1000 steps is slow. This pain is real and motivates everything that follows.

**12. Next Step**
- "You can now trace the full diffusion pipeline: forward process (destroy), training (learn to predict noise), and sampling (iteratively denoise to generate). Next lesson, you will build it. All of it. From scratch."
- Set expectation: the capstone lesson implements the forward process, training loop, and sampling loop on real data. "You will feel the slowness firsthand -- and understand exactly why latent diffusion was invented."

### Widget Specification

**DenoisingTrajectoryWidget** -- Interactive visualization of the sampling process.

- Shows a timeline/slider from t=T (1000) to t=0
- At each position, displays the image at that stage of denoising (pre-computed using the same procedural 28x28 T-shirt image for visual continuity with prior widgets)
- Key snapshots highlighted on the timeline (e.g., t=1000, 750, 500, 250, 100, 0)
- A "Play" button that animates the denoising process (stepping through at a reasonable pace)
- Signal-vs-noise percentage bar (reusing the same visual pattern from DiffusionNoiseWidget and AlphaBarCurveWidget)
- Context-sensitive description text: at high t ("Pure static -- the model is hallucinating structure"), at mid t ("A shape is emerging"), at low t ("Refining details and textures")
- Optional: show the model's predicted noise at the current step alongside the image, so the student can see what the model "sees" vs what it is producing
- Same seeded PRNG and procedural image as the other three widgets for visual continuity

### Practice (Notebook)

**No notebook for this lesson.** The module plan specifies: "No notebook yet -- understand the algorithm before implementing." The student should internalize the sampling algorithm conceptually. Implementation comes in Lesson 5 (build-a-diffusion-model), which is the CONSOLIDATE capstone.

---

## Planning Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (N/A -- no gaps)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 planned: verbal/analogy, symbolic, concrete example, visual, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (4 identified)
- [x] Cognitive load <= 3 new concepts (2-3 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-10 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical mathematical errors in the formula or algorithm presentation, but one critical self-contradicting answer in a predict-and-verify check would confuse the student. Three improvement findings address pedagogical ordering, a missing plan element, and a weak negative example. Two polish findings on formatting.

### Findings

### [CRITICAL] — Predict-and-Verify answer contradicts itself

**Location:** Section 6 (Check 1), "Predict and Verify" GradientCard
**Issue:** The question asks "How much of the original image's information is in that prediction?" The answer opens with "Roughly half" but then immediately states alpha_bar_500 ~ 0.05, meaning only 5% of the original signal remains. "Roughly half" and "5%" are contradictory. The planning document contains this same contradiction ("roughly half, since alpha_bar_500 ~ 0.05"), suggesting it was copied without noticing the inconsistency.
**Student impact:** The student reads "roughly half," forms a mental model of 50%, then reads "only about 5% of the original signal remains" and is confused. Which is it? They would not know whether to trust the verbal answer or the numerical one. This undermines the predict-and-verify format, which depends on the answer being clearly correct so the student can calibrate their understanding.
**Suggested fix:** Remove "Roughly half." Replace with an answer that is consistent with the 5% figure. Something like: "Very little from the original image itself. At t=500 with a cosine schedule, alpha_bar_500 ~ 0.05 -- only about 5% of the original signal remains. The model is hallucinating most of the structure from learned patterns, not from information still present in x_t. The prediction is useful, but it is mostly the model's best guess about what an image looks like, not a recovery of what was actually there." This is honest and consistent.

### [IMPROVEMENT] — Concrete walkthrough comes too late (concrete-before-abstract violation)

**Location:** Section 11 (Concrete Walkthrough) vs Section 5 (The Reverse Step)
**Issue:** The reverse step formula is introduced in Section 5 with a term-by-term conceptual breakdown, but the student does not see a single computed number until Section 11 -- six sections later. The student must hold the abstract formula through two checks (Sections 6), a widget (Section 7), the "why add noise back" elaboration (Section 8), another check (Section 9), and the full algorithm (Section 10) before grounding it in concrete arithmetic. This violates the "concrete before abstract" ordering principle. The planning document designed this ordering (outline items 4 then 10), but the lesson-planning principles say concrete should come before or alongside abstract.
**Student impact:** The student sees the formula, understands each term conceptually, but has no numerical intuition for the scale of the coefficients. When they encounter the full sampling algorithm (Section 10), they still have not seen what a single step actually computes. The walkthrough would be far more effective immediately after the term-by-term breakdown, giving the student numerical grounding before asking them predict-and-verify questions or viewing the full algorithm.
**Suggested fix:** Move the concrete walkthrough (current Section 11) to immediately after the term-by-term breakdown (after current Section 5, before the first check). This way the student sees: formula -> term-by-term breakdown -> plug in numbers at t=500 -> now do checks with numerical intuition -> widget -> why add noise -> full algorithm. The t=950 vs t=50 comparison cards can stay near the end or move with the walkthrough.

### [IMPROVEMENT] — Missing planned "open-book exam" analogy contrast

**Location:** Sections 3-5 (Recap through Reverse Step)
**Issue:** The planning document explicitly identified a potentially misleading prior analogy: "The open-book exam analogy from learning-to-denoise is about training, not sampling. In training, you KNOW the true noise (the answer key). In sampling, there IS no answer key -- the model's prediction is all you have. This distinction should be explicitly called out to prevent confusion." The built lesson never makes this distinction. The aside in Section 2 says "The model can predict noise at any timestep. Now: how do you use that ability to generate?" but never contrasts the training situation (known epsilon) with the sampling situation (only epsilon_theta available).
**Student impact:** The student who internalized the open-book exam analogy may carry it into sampling and think the model has access to the true noise. This is exactly the misconception that leads to "just subtract the noise" thinking. The one-shot negative example addresses the practical consequence but does not address the conceptual root: in sampling, there is no answer key.
**Suggested fix:** Add a sentence or two in Section 3 (The Missing Piece) or at the start of Section 5 (The Reverse Step): "In training, you knew the true noise -- it was the answer key. In sampling, there is no answer key. The model's prediction epsilon_theta is all you have. This is the fundamental difference."

### [IMPROVEMENT] — Negative example (one-shot denoising) tells but does not show

**Location:** Section 4 (The Tempting Shortcut)
**Issue:** The one-shot denoising negative example explains conceptually why it fails ("blurry, incoherent results") and gives a mathematical formula for the one-shot attempt. But it never shows what the failure looks like. The student is told "this does not work" but has no visual or concrete evidence. The planning document describes showing "what this looks like" but the built lesson only describes the failure in words.
**Student impact:** The student takes on faith that one-shot denoising produces bad results. This is less convincing than seeing the blurry output. The negative example would be stronger with even a brief visual or concrete description of what the blurry output actually looks like (e.g., "At t=500, the one-shot prediction looks like a vague gray smudge where a T-shirt should be").
**Suggested fix:** Add a concrete visual description or (better) use the DenoisingTrajectoryWidget's infrastructure to show what a one-shot prediction looks like at a high noise level. Even without a new widget, a sentence like "At t=500, the one-shot attempt produces a vague smear of gray -- the rough proportions of the T-shirt are there, but all fine detail and texture is lost" would ground the failure concretely.

### [POLISH] — Spaced em dashes in term-by-term breakdown labels

**Location:** Section 5, term-by-term breakdown cards (lines 251, 264, 277)
**Issue:** The InlineMath component for each formula term is followed by ` &mdash;` with a space before the dash (from the JSX whitespace after the self-closing `/>` tag). In rendered HTML, the labels would appear as "formula -- Scaling factor" with spaces around the em dash. The writing style rule requires no spaces around em dashes: "word--word" not "word -- word".
**Student impact:** Minor visual inconsistency. Not confusing, but inconsistent with the rest of the lesson's em dash usage.
**Suggested fix:** Remove the space before `&mdash;` in each of the three label paragraphs. Use a pattern like `<InlineMath math="..." />&mdash;Scaling factor` with no spaces.

### [POLISH] — HTML comment section labels use spaced em dashes

**Location:** HTML comments throughout (e.g., "Section 4: Negative Example -- One-Shot Denoising")
**Issue:** The HTML comments that label sections use spaced em dashes (` — `). While these are not student-facing, they are inconsistent with the project convention and could be copied into student-facing text in future edits.
**Student impact:** None (comments are not rendered).
**Suggested fix:** Use unspaced em dashes in comments for consistency, or ignore since these are never rendered.

### Review Notes

**What works well:**
- The narrative arc is strong. The emotional journey from "I finally get to see generation" through "wait, you add noise BACK?" to "1000 steps is painfully slow" is exactly what the planning document designed. The lesson delivers on this arc.
- The connection to temperature in language models (Section 8) is one of the strongest connections in the lesson. The student already has deep intuition for temperature, and the analogy makes stochastic sampling instantly understandable.
- The DDPM vs DDIM ComparisonRow is well-scoped -- mentions DDIM enough to plant the seed without developing it.
- The DenoisingTrajectoryWidget is well-designed with proper interactive affordances (play/pause, slider, clickable snapshots, context-sensitive descriptions). The coarse-to-fine progression is visually compelling.
- The full sampling algorithm presentation using PhaseCards is clear and well-structured, with the t=1 special case called out in a WarningBlock aside.
- Scope boundaries are respected -- no scope creep into DDIM internals, U-Net architecture, or code implementation.

**Pattern observed:** The lesson is strong on symbolic, visual, and intuitive modalities but the concrete/numerical modality arrives late. Moving the walkthrough earlier would make all subsequent sections more grounded. This is a recurring tension in BUILD lessons -- the temptation to show all the conceptual pieces first and ground them in numbers last. The principles say ground early.

**Widget quality:** The DenoisingTrajectoryWidget simulates denoising by applying the forward process noise at each timestep (using the closed-form formula) rather than actually running a reverse process. This is a simulation -- it shows what the image looks like at each noise level, not what a trained model actually produces at each step. The lesson frames it as "watch structure emerge from pure noise" which is slightly misleading since the widget is computing q(x_t|x_0), not the reverse process output. This is acceptable for pedagogical purposes (the visual pattern is the same), but the lesson should not imply the widget is running the actual sampling algorithm. The current framing ("Step through the denoising process. Watch the same procedural T-shirt image emerge from noise") is borderline but acceptable.

---

## Review — 2026-02-10 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

All Iteration 1 critical and improvement findings have been addressed effectively. One new improvement finding on a factual inconsistency between the concrete walkthrough and the predict-and-verify answer (linear vs cosine schedule attribution). One polish finding on minor redundancy. The lesson is close to passing -- one targeted fix should resolve the remaining issue.

### Findings

### [IMPROVEMENT] — Schedule inconsistency between walkthrough and check answer

**Location:** Section 5b (Concrete Walkthrough, line 331) vs Section 6 (Predict and Verify, line 442)
**Issue:** The concrete walkthrough at t=500 explicitly says "with a linear noise schedule" and labels the setup as "Setup at t=500 (linear schedule)." Two sections later, the Predict-and-Verify answer says "At t=500 with a cosine schedule, alpha_bar_500 ~ 0.05." The student just anchored on "linear schedule" and now reads "cosine schedule" attributed to the same alpha_bar value. This is contradictory framing -- the student does not know which schedule is being used and may wonder whether they missed a schedule change.
**Student impact:** The student who carefully followed the walkthrough ("linear schedule, alpha_bar_500 ~ 0.05") reads the check answer and sees "cosine schedule, alpha_bar_500 ~ 0.05." They are forced to reconcile: did we switch schedules? Are both schedules the same at t=500? Is this a mistake? This breaks the flow of the predict-and-verify format, which should confirm understanding, not introduce new confusion. The numerics happen to be roughly similar for both schedules at t=500, but the student does not know that.
**Suggested fix:** Make the check answer consistent with the walkthrough. Change "with a cosine schedule" to "with a linear schedule" in the Predict-and-Verify answer (line 442), or remove the schedule attribution entirely since the walkthrough already established which schedule is in use. The simplest fix: change "At t=500 with a cosine schedule" to "At t=500" (the schedule was already established).

### [POLISH] — Redundant sentence on stochastic diversity

**Location:** Section 8, paragraph after ComparisonRow (lines 621-636)
**Issue:** The paragraph beginning "This also addresses a subtle point..." restates what was already covered in the preceding paragraphs and ComparisonRow. Specifically, "Different noise draws lead to different images" appears as a ComparisonRow bullet ("Different noise draws -> different images") and is restated nearly verbatim in line 631. The paragraph adds the framing of the model being a deterministic function while the noise injection adds randomness, which is a valid nuance, but most of the paragraph's content is repetitive.
**Student impact:** Minor. The student reads the same point three times (paragraphs, ComparisonRow bullet, post-ComparisonRow paragraph). This does not confuse but slightly dilutes the lesson's otherwise tight pacing.
**Suggested fix:** Either trim the paragraph to just the novel nuance ("The trained model is a deterministic function -- same input, same output. But the noise injection at each step introduces randomness. Run the sampling algorithm twice from the same starting noise x_T with different z draws, and you get different images.") or remove it entirely since the ComparisonRow already makes the point.

### Iteration 1 Fixes Verified

All five fixes from Iteration 1 have been applied and are effective:

1. **CRITICAL (Predict-and-Verify contradiction):** Fixed. Answer now opens with "Very little" and is consistent with the 5% alpha_bar figure throughout. No contradiction.

2. **IMPROVEMENT (Concrete walkthrough ordering):** Fixed. The walkthrough now appears immediately after the term-by-term breakdown (Section 5b), before the first check. The student has numerical grounding before being asked predict-and-verify questions. The coefficient comparison at t=950 vs t=50 also moved with it, providing a second positive example early. This is a significant improvement to the lesson's pedagogical ordering.

3. **IMPROVEMENT (Open-book exam contrast):** Fixed. The opening paragraph of the Reverse Step section (lines 203-210) now reads: "One crucial difference from training: in training, you knew the true noise -- it was the answer key on an open-book exam. In sampling, there is no answer key." This directly addresses the potentially misleading prior analogy identified in the planning document.

4. **IMPROVEMENT (One-shot denoising concrete description):** Fixed. The WarningBlock aside (lines 153-160) now includes a vivid description: "a foggy smear of gray -- vaguely the right brightness but no edges, no texture, no coherent structure. The model tried to jump directly from static to image and produced something between the two." This grounds the negative example concretely.

5. **POLISH (Em dashes):** Fixed. The term-by-term breakdown labels now use `/>&mdash;` without a space before the em dash. The HTML comment em dashes were also cleaned up.

### Review Notes

**What improved since Iteration 1:**
- The lesson's pedagogical ordering is now much stronger. The concrete walkthrough immediately following the formula is exactly what the principles prescribe. The student never has to hold an abstract formula for more than a few paragraphs before seeing real numbers.
- The open-book exam contrast is a clean, effective addition. One sentence that prevents a significant misconception.
- The one-shot denoising description is vivid without being overwrought. "Foggy smear of gray" is the right level of concrete.

**What works well (carried forward from Iteration 1):**
- The narrative arc remains strong. The emotional journey delivers.
- The temperature analogy for stochastic sampling is excellent pedagogy.
- The DenoisingTrajectoryWidget is well-designed with proper interactive affordances.
- The full sampling algorithm presentation using PhaseCards is clear.
- Scope boundaries continue to be respected.
- All five modalities are present and effective.

**Remaining concern:** The schedule inconsistency (IMPROVEMENT finding above) is a small but real issue. It is the only finding that could confuse the student. The fix is a single phrase change. Once resolved, the lesson should pass.
