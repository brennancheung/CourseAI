# Module 6.2: Diffusion -- Record

**Goal:** The student understands the core diffusion mechanism -- gradually destroying images with noise (forward process) and training a neural network to reverse that destruction (reverse process) -- well enough to implement and train a pixel-space DDPM from scratch, generating real images and experiencing both the magic and the slowness that motivates latent diffusion.
**Status:** Complete (5 of 5 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Forward process (gradual noise destruction of images) | INTRODUCED | the-diffusion-idea | Adding Gaussian noise step by step until the image becomes pure static. Taught via ink-in-water analogy and inline noise progression strip. Student understands the concept but has not seen the math (no alpha_bar, no noise schedule formula). |
| Reverse process (learned iterative denoising) | INTRODUCED | the-diffusion-idea | A neural network learns to remove a small amount of noise at each step; chaining ~1,000 steps produces generation. Taught via sculpting analogy and "could YOU denoise a slightly noisy photo?" moment. No math, no sampling algorithm, no architecture details. |
| "Small steps make it learnable" (core diffusion insight) | DEVELOPED | the-diffusion-idea | One-shot generation from pure noise is underdetermined (infinitely many images consistent with noise). But removing a small amount of noise is a tractable, learnable task. The student can explain this in their own words and reason about why diffusion works. Core mental model for the module. |
| Multi-scale denoising progression (coarse-to-fine) | INTRODUCED | the-diffusion-idea | High noise = structural decisions (face vs landscape). Medium noise = refine structure. Low noise = fine details (textures, edges). Connected to CNN feature hierarchy from Series 3. Taught via PhaseCards and sculpting analogy. |
| Image manifold intuition (noise pushes off, denoising walks back) | MENTIONED | the-diffusion-idea | Brief geometric/spatial framing: real images occupy a thin manifold in high-dimensional space. Noise pushes images off this manifold; denoising walks back toward it. Connected to latent space concepts from Module 6.1. Two sentences, not developed. |
| Diffusion has no bottleneck (contrast with autoencoders) | MENTIONED | the-diffusion-idea | Briefly noted that diffusion operates in full image space, not through a compressed representation. The learning mechanism is the noise schedule, not information compression. Student told not to expect mu/sigma parameters. Latent diffusion (which adds a bottleneck) deferred to Module 6.3. |
| Gaussian noise addition property (independent Gaussians add, variances sum) | INTRODUCED | the-forward-process | N(0, sigma1^2) + N(0, sigma2^2) = N(0, sigma1^2 + sigma2^2). Taught as a tool for the closed-form derivation, not as deep probability theory. Visual: static SVG showing two dashed Gaussians (var=0.6, var=1.0) combining into a wider solid Gaussian. Concrete example: N(0,3) + N(0,5) = N(0,8). Connected to reparameterization trick from VAE. |
| Gaussian variance scaling under multiplication (c*X scales variance by c^2) | INTRODUCED | the-forward-process | If X ~ N(0,1), then c*X ~ N(0, c^2). Taught alongside the addition property as a prerequisite tool. Connected to the reparameterization trick: "sigma * epsilon gave you a sample with variance sigma^2 -- that is this property in action." |
| Noise schedule (beta_t) | DEVELOPED | the-forward-process | The sequence {beta_1, ..., beta_T} controlling noise amount at each timestep. Student understands it is a design choice, not given by physics. Taught via linear schedule example (0.0001 to 0.02). Negative example: constant schedule wastes early timesteps. Connected to widget from the-diffusion-idea ("the first few steps barely changed the image -- that was the schedule at work"). |
| Variance-preserving formulation | DEVELOPED | the-forward-process | x_t = sqrt(1-beta_t) * x_{t-1} + sqrt(beta_t) * epsilon. Signal scaled down before noise added, keeping total variance at 1. Motivated by variance-exploding negative example (naive addition blows up pixel values). Mixing/paint analogy: remove old paint before adding new color, total volume stays constant. Verified algebraically: (1-beta_t) + beta_t = 1. |
| Alpha notation (alpha_t = 1 - beta_t, signal fraction) | INTRODUCED | the-forward-process | Renaming for convenience: where beta is noise fraction, alpha is signal fraction. Framed as "progressive simplification, not progressive complexity." Leads to alpha_bar. |
| Alpha-bar (cumulative signal fraction, product of alphas) | DEVELOPED | the-forward-process | alpha_bar_t = product of alpha_i for i=1..t. The one number that encodes the entire history of the noise schedule at timestep t. Starts near 1 (clean), drops to near 0 (pure noise). Interactive widget (AlphaBarCurveWidget) lets student drag along the curve and see image + formula coefficients update in real time. Mental model: "alpha_bar is the signal-to-noise dial." |
| Closed-form shortcut q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon | DEVELOPED | the-forward-process | The lesson's climax. Derived step by step: unroll 2-step recursion, apply Property 2 (scaling), then Property 1 (addition), generalize to t steps. Both Gaussian properties visibly used in the derivation. Connected to reparameterization trick (same structure: signal + noise_scale * epsilon). Verified via 1D pixel walkthrough (pixel=0.8, 3 steps, full round-trip arithmetic proving exact match). The formula is exact, not approximate. |
| Linear vs cosine noise schedules | INTRODUCED | the-forward-process | Linear schedule (DDPM 2020) destroys information too quickly at early timesteps. Cosine schedule (Improved DDPM 2021) spends more time at high alpha_bar. ComparisonRow showing key differences. Widget includes schedule toggle. Student can reason about why schedule choice matters but has not implemented either. |
| DDPM training objective (predict noise, not clean image) | DEVELOPED | learning-to-denoise | The model predicts epsilon (the noise that was added), not x_0 (the clean image). Loss = \|\|epsilon - epsilon_theta(x_t, t)\|\|^2. Motivated by consistent target argument: epsilon is always from N(0,1) regardless of image content, while x_0 varies wildly. Algebraic equivalence shown: x_0 can be recovered from predicted epsilon using the closed-form formula. Ho et al. 2020 confirmed noise prediction gives better sample quality empirically. Addressed misconception #1 (model predicts clean image) directly. |
| Simplified DDPM loss (MSE on epsilon) | DEVELOPED | learning-to-denoise | L = \|\|epsilon - epsilon_theta\|\|^2. The same MSE formula from Series 1 (loss-functions), same nn.MSELoss in PyTorch. Side-by-side comparison of Series 1 MSE, autoencoder MSE, and DDPM MSE formulas shown early (hook) and in "Three Faces of MSE" synthesis table. Student understands the loss is not new math -- same formula, different question. Simplified from full variational lower bound (weighting terms dropped per Ho et al.). |
| DDPM training algorithm (7-step procedure) | DEVELOPED | learning-to-denoise | Complete training algorithm: (1) sample image x_0, (2) sample random timestep t ~ Uniform(1,T), (3) sample noise epsilon ~ N(0,1), (4) create noisy image via closed-form formula, (5) network predicts noise epsilon_theta = network(x_t, t), (6) compute MSE loss, (7) backprop and update. Steps 2-4 are diffusion-specific data preparation; steps 1, 5-7 are the standard training loop. Taught via color-coded PhaseCards (blue=familiar, violet=new) and open-book exam analogy threading through all steps. |
| Random timestep sampling in training (not sequential) | DEVELOPED | learning-to-denoise | Each training iteration samples ONE random timestep, not a sequence from 1 to T. The closed-form formula teleports to that timestep. Different images in the same batch have different timesteps. Addressed misconceptions #2 (model sees entire trajectory) and #3 (timesteps iterated in order). Mini-batch table example: Cat at t=800, Shoe at t=50, Landscape at t=400. |
| Timestep embedding (network receives t as input) | INTRODUCED | learning-to-denoise | One network handles all timesteps by receiving t as an additional input alongside x_t. At low t, it detects subtle noise; at high t, it hallucinates structure from near-pure static. Addressed misconception #5 (separate network per timestep). Architecture details (how the network uses t) explicitly deferred to Module 6.3. |
| Three faces of MSE (regression/autoencoder/DDPM) | INTRODUCED | learning-to-denoise | Synthesis showing MSE loss in three contexts: (1) linear regression: y_hat vs y, (2) autoencoder: x_hat vs x, (3) DDPM: epsilon_theta vs epsilon. Same formula, same gradients, same PyTorch code. Different prediction and target each time. Reinforces "same building blocks, different question" mental model. |
| DDPM reverse step formula (x_{t-1} from x_t using epsilon_theta) | DEVELOPED | sampling-and-generation | x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z. Broken down term by term: scaling factor, noise removal (predicted mean), and stochastic noise injection. Derived by rearranging the forward process formula, substituting epsilon_theta for true epsilon. Connected to reparameterization trick pattern (mean + sigma * z). Concrete walkthrough at t=500 with specific coefficient values immediately after the formula. Second examples at t=950 (bold structural decisions) and t=50 (minute detail refinements). |
| Stochastic noise injection during sampling (sigma_t * z term) | DEVELOPED | sampling-and-generation | Fresh noise added at each reverse step (except the final step t=1 -> x_0). Motivated by temperature analogy from LLMs: sigma_t > 0 gives diverse outputs (like temperature > 0), sigma_t = 0 gives deterministic outputs (DDIM-style). Without noise injection, model commits to imperfect predictions and errors compound to blurry averages. Hiking analogy: compass (model) points toward peak, but jitter (noise) prevents consistently landing in the wrong spot if compass is miscalibrated. z = 0 at t=1 because the last step commits -- no subsequent step to correct errors. |
| DDPM sampling algorithm (full loop from x_T to x_0) | DEVELOPED | sampling-and-generation | Complete algorithm presented via PhaseCards: (1) sample x_T ~ N(0,I), (2) loop from t=T to 1: sample z, predict epsilon_theta, compute reverse step, (3) return x_0. The t=1 special case (z=0) highlighted in WarningBlock. Memory usage is constant (only x_t needed at each step, each step overwrites previous). Requires T forward passes through the network -- seconds to minutes per image. |
| One-shot denoising failure (why naive noise subtraction does not work) | INTRODUCED | sampling-and-generation | Negative example: rearranging the forward formula to solve directly for x_0 from one prediction produces blurry, incoherent results ("foggy smear of gray"). Model's prediction is an approximation; staking everything on one imperfect prediction causes catastrophic error compounding. This is why diffusion uses 1000 iterative steps. |
| DDPM vs DDIM (stochastic vs deterministic sampling) | MENTIONED | sampling-and-generation | DDPM adds noise at each step (sigma_t > 0) for diverse outputs; DDIM sets sigma_t = 0 for deterministic, faster sampling. ComparisonRow contrasts key differences. DDIM mentioned as a later development that trades diversity for speed -- not developed, deferred to accelerated samplers discussion. |
| 1000-step computational cost of DDPM sampling | APPLIED | build-a-diffusion-model | Elevated from INTRODUCED. Student timed one training step vs generating 64 images. Measured the ~1000x slowdown firsthand. Calculated scaling projections for larger models/resolutions. Experienced the wait as visceral motivation for latent diffusion. |
| Coarse-to-fine denoising progression (early steps create structure, late steps refine details) | DEVELOPED | sampling-and-generation | Extended from INTRODUCED in the-diffusion-idea to DEVELOPED via the DenoisingTrajectoryWidget. At t=950: model hallucinating structure from near-pure static, making bold decisions (shoe vs shirt?). At t=500: shape clearly emerging, edges forming. At t=50: polishing textures and fine details. Not all steps equally important -- early steps do the most dramatic work. Visualized interactively with timeline slider, play/pause animation, key timestep snapshots. |
| Forward process implementation (q_sample function) | APPLIED | build-a-diffusion-model | Elevated from DEVELOPED. Student implemented q_sample() from scratch in PyTorch: gather alpha_bar_t per image, reshape for broadcasting, compute signal and noise coefficients, produce noisy image. Verified visually against widget outputs from the-diffusion-idea and the-forward-process. |
| DDPM training algorithm implementation (full training loop) | APPLIED | build-a-diffusion-model | Elevated from DEVELOPED. Student filled in the 4 diffusion-specific parts of the training loop: sample random timesteps (torch.randint), sample noise (torch.randn_like), create noisy images (q_sample), compute MSE loss (F.mse_loss). Trained on MNIST for 20 epochs and observed loss curve behavior. |
| DDPM sampling algorithm implementation (reverse loop from x_T to x_0) | APPLIED | build-a-diffusion-model | Elevated from DEVELOPED. Student implemented the reverse step formula and noise injection inside a loop from T-1 to 0. Handled the t=0 special case (z=0, final step commits). Generated 64 MNIST images from pure noise. |
| DDPM reverse step formula implementation | APPLIED | build-a-diffusion-model | Elevated from DEVELOPED. Student translated x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_hat) + sigma_t * z into one line of PyTorch code. Addressed 0-indexed vs 1-indexed notation shift between paper and Python. |
| Skip connections in U-Net (encoder features concatenated with decoder) | INTRODUCED | build-a-diffusion-model | First encounter with skip connections in the context of denoising. Framed as "the autoencoder's bottleneck forces compression, but for denoising we want the decoder to have access to high-resolution features -- skip connections pass them through directly, like giving the decoder a cheat sheet." Student read and understood annotated code with 2 skip connections (enc1->dec2, enc2->dec3). Did not build from scratch -- architecture was provided. |
| Timestep embedding implementation (simple linear projection) | INTRODUCED | build-a-diffusion-model | Extended from INTRODUCED (concept-level in learning-to-denoise) to INTRODUCED (implementation-level). Student read annotated code: normalize t to [0,1], pass through 2-layer MLP, add resulting 128-dim vector to bottleneck features. Explicitly minimal -- no sinusoidal encoding, no adaptive group norm. Full timestep embedding deferred to Module 6.3. |

## Per-Lesson Summaries

### the-diffusion-idea (Lesson 1)

**Cognitive load:** BUILD (after CONSOLIDATE in exploring-latent-spaces)

**Concepts taught:**
- Forward process as gradual noise destruction (INTRODUCED)
- Reverse process as learned iterative denoising (INTRODUCED)
- "Small steps make it learnable" core insight (DEVELOPED)
- Multi-scale denoising progression: coarse-to-fine (INTRODUCED)
- Image manifold intuition (MENTIONED)
- Diffusion has no bottleneck (MENTIONED)

**Mental models established:**
- "Destruction is easy; creation from scratch is impossibly hard; but undoing a small step of destruction is learnable."
- "A neural network that can remove a little noise is solving an easy problem. Chain a thousand of those together, and you get creation."
- "Same building blocks, different question" extended to diffusion: conv layers, MSE loss, backprop -- the new question is "what noise was added to this image?"

**Analogies used:**
- Ink drop in water (forward process -- physical diffusion as the origin of the name)
- Sculpting from marble: rough shape -> details -> polish (iterative refinement, coarse-to-fine)
- Jigsaw puzzle in a tornado (why one-shot creation from disorder fails)
- Image manifold: noise pushes off a thin surface in high-dimensional space, denoising walks back (brief, geometric/spatial)

**How concepts were taught:**
- Forward process: ink-in-water analogy first, then inline NoiseProgressionStrip (static 5-image strip at t=0/250/500/750/1000), then interactive DiffusionNoiseWidget with slider, signal-vs-noise bar, and clickable preview strip
- Reverse process: motivated by impossibility of one-shot reversal (underdetermined problem), then "could YOU denoise a slightly noisy photo?" moment, then sculpting analogy for iterative refinement
- Multi-scale progression: PhaseCards (high/medium/low noise = composition/structure/details), connected to CNN feature hierarchy via TipBlock
- "Same building blocks" connection: explicitly named conv layers (Series 3), MSE loss (Series 1.1), training loop (Series 2)

**Misconceptions addressed:**
- "Diffusion generates from nothing" -- corrected: each step is "take what I have and make it slightly better," creation emerges from accumulated corrections
- "Denoising is trivially easy like a Photoshop filter" -- corrected: at high noise, the model must hallucinate plausible content; at low noise, it must reproduce precise details. Denoising at any single noise level is learnable, but not trivial
- "Training iterates through all 1,000 steps sequentially" -- briefly noted that each training step picks one random noise level; deferred to the-forward-process lesson
- "Diffusion is a completely new paradigm" -- corrected via "same building blocks, different question" connection

**What is NOT covered (deferred):**
- Mathematical formulation of noise schedules, alpha_bar, closed-form formula (-> the-forward-process)
- Training objective / loss function (-> learning-to-denoise)
- Sampling algorithm (-> sampling-and-generation)
- Code or implementation (-> build-a-diffusion-model)
- U-Net architecture (-> Module 6.3)
- Score matching, SDEs, continuous-time formulations (out of scope)
- Conditioning or text guidance (-> Module 6.3)

**Widget:** DiffusionNoiseWidget -- interactive slider controlling noise level (t=0 to t=1000) on a procedural 28x28 T-shirt silhouette. Cosine noise schedule. Signal-vs-noise percentage bar. Clickable preview strip at 8 fixed timesteps. Reverse-direction annotation. Seeded PRNG for deterministic noise.

### the-forward-process (Lesson 2)

**Cognitive load:** STRETCH (hardest in the module -- 3 new concepts at the upper limit)

**Concepts taught:**
- Gaussian noise addition property (variances add for independent Gaussians) (INTRODUCED)
- Gaussian variance scaling under multiplication (c*X ~ N(0, c^2)) (INTRODUCED)
- Noise schedule beta_t (design choice controlling per-step noise amount) (DEVELOPED)
- Variance-preserving formulation (scale signal down before adding noise) (DEVELOPED)
- Alpha notation (alpha_t = 1 - beta_t as signal fraction) (INTRODUCED)
- Alpha-bar (cumulative signal fraction, product of all alphas) (DEVELOPED)
- Closed-form shortcut q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon (DEVELOPED)
- Linear vs cosine noise schedules (INTRODUCED)

**Mental models established:**
- "Alpha-bar is the signal-to-noise dial. The closed-form formula lets you turn that dial to any position without stepping through all the intermediate values."
- Variance-preserving as paint mixing: remove old paint before adding new color, total volume stays constant.
- Step-by-step vs closed-form as walking vs teleporting: same destination, different path. The forward process definition tells you the route; the closed-form formula tells you the destination.

**Analogies used:**
- Paint mixing for variance-preserving (remove old before adding new, total stays constant)
- Map coordinates vs walking block-by-block for closed-form vs iterative (same place, direct access)
- The closed-form formula as the reparameterization trick in a new context (signal + noise_scale * epsilon)

**How concepts were taught:**
- Gaussian properties: dedicated "Two Properties You Need" section early in the lesson, framed as tools not theory. Property 1 (addition): formula + concrete example (N(0,3) + N(0,5) = N(0,8)) + static SVG (two dashed Gaussians combining into wider solid Gaussian). Property 2 (scaling): formula + connection to reparameterization trick from VAE.
- Variance-preserving: motivated by negative example first (variance-exploding: naive addition without scaling causes pixel values to blow up), then the fix (scale signal down with sqrt(1-beta_t)), then algebraic verification (Var = (1-beta_t) + beta_t = 1).
- Noise schedule: explained as a design choice, not physics. Linear schedule example. Negative example: constant schedule wastes early timesteps. Connected to what the student saw in the Lesson 1 widget.
- Alpha notation: introduced as progressive simplification. Alpha_t = 1 - beta_t (just a renaming). Alpha_bar as cumulative product. Addressed misconception that notation is unnecessarily complex ("alpha_bar is the punchline").
- Closed-form shortcut: derived step by step from 2-step unrolling. Explicitly applied both Gaussian properties (Property 2 for each scaled noise term's variance, Property 1 to combine). Generalized to t steps. Connected to reparameterization trick pattern. Verified via 1D pixel walkthrough with full round-trip arithmetic.
- Schedule comparison: ComparisonRow (linear vs cosine). Widget includes schedule toggle.
- Predict-and-verify exercises: beta=0 gives unchanged image, beta=1 gives pure noise. Transfer check: colleague claims closed-form is "just an approximation" -- student corrects this.

**Misconceptions addressed:**
- "You need to iterate through all steps to get to timestep t" -- the closed-form formula jumps directly. Motivated by the practical problem: 50M operations per epoch without the shortcut.
- "The image gets bigger/louder as you add noise (variance explodes)" -- variance-preserving formulation prevents this. Negative example: naive addition without scaling.
- "The noise added at each step is the same amount" -- beta_t varies by timestep (the noise schedule). Negative example: constant schedule wastes early timesteps.
- "The notation (beta, alpha, alpha_bar) is unnecessarily complex" -- alpha_bar is the punchline, the one number you need. The notation is progressive simplification.
- "The closed-form formula is just an approximation" -- it is mathematically exact. The Gaussian addition property is exact, not approximate. Transfer check exercise addresses this directly.

**What is NOT covered (deferred):**
- Reverse process formula or how to denoise (-> learning-to-denoise)
- Training objective or loss function (-> learning-to-denoise)
- Sampling algorithm (-> sampling-and-generation)
- Code implementation of the forward process (-> build-a-diffusion-model)
- U-Net or any denoising architecture (-> Module 6.3)
- Score matching, SDEs, continuous-time formulations (out of scope)
- Conditioning or text guidance (-> Module 6.3)

**Widget:** AlphaBarCurveWidget -- interactive alpha_bar curve (violet) with draggable marker. Displays procedural 28x28 T-shirt at current noise level using closed-form formula. Live formula coefficients (sqrt(alpha_bar) and sqrt(1-alpha_bar)) update in real time. Signal-vs-noise percentage bar. Slider for fine timestep control. Toggle between cosine and linear schedules. Context-sensitive description text. Seeded PRNG for deterministic noise. Same procedural image as DiffusionNoiseWidget for visual continuity.

### learning-to-denoise (Lesson 3)

**Cognitive load:** BUILD (relief after STRETCH of the-forward-process)

**Concepts taught:**
- DDPM training objective: predict epsilon (noise), not x_0 (clean image) (DEVELOPED)
- Simplified DDPM loss: L = ||epsilon - epsilon_theta||^2, same MSE from Series 1 (DEVELOPED)
- DDPM training algorithm: 7-step procedure from sampling image to backprop (DEVELOPED)
- Random timestep sampling: one random t per iteration, not sequential (DEVELOPED)
- Timestep embedding: one network handles all timesteps by receiving t as input (INTRODUCED)
- Three faces of MSE: regression, autoencoder, DDPM -- same formula, different question (INTRODUCED)

**Mental models established:**
- "Same building blocks, different question" payoff: the DDPM training objective IS MSE loss. The building blocks are MSE loss, backprop, gradient descent. The question is "what noise was added to this image?"
- "Open-book exam": the network receives a noisy image (the question), predicts the noise (its answer), and we check against the actual noise (the answer key). MSE is the grade.

**Analogies used:**
- Open-book exam: network always knows what noise level it is dealing with (timestep is the open book); epsilon is the answer key; MSE is the grade. Threaded through all 7 training algorithm steps via PhaseCards.
- "Same wrongness score from your third lesson": MSE loss callback to Series 1 loss-functions lesson, grounding the DDPM loss in deep familiarity.
- Mirror image: t=50 and t=500 have swapped signal/noise coefficients -- explicitly called out to prevent confusion about the symmetric numbers.

**How concepts were taught:**
- Training objective: Hook first ("after all that math, the training objective is just MSE?"), then immediate formula reveal with side-by-side comparison (Series 1 MSE vs DDPM MSE). Cognitive relief as the emotional trajectory.
- Why predict noise: ComparisonRow (Option A: predict x_0 vs Option B: predict epsilon), consistent target argument, algebraic equivalence showing nothing is lost, Ho et al. empirical confirmation.
- Training algorithm: 7 color-coded PhaseCards (blue=familiar steps 1/5/6/7, violet=diffusion-specific steps 2/3/4). Open-book exam analogy threaded through steps 3/5/6. Aside explicitly highlights what is new vs familiar. Mermaid flow diagram showing data pipeline.
- Random timestep sampling: Dedicated "One Random Timestep, Not a Sequence" GradientCard addressing misconceptions #2/#3. Mini-batch table showing 3 images with different timesteps in one batch.
- Concrete example: Full walkthrough with T-shirt image at t=500 (alpha_bar=0.05, signal=22.4%, noise=97.5%). Second example at t=50 (mirror coefficients) and t=950 (near-pure noise). Interactive exploration via TrainingStepSimulator widget.
- Three Faces of MSE: Synthesis table showing regression/autoencoder/DDPM targets side by side. Three formulas written out to make the identity unmistakable.

**Misconceptions addressed:**
- "The model predicts the clean image, not the noise" -- noise gives consistent target (always N(0,1)); algebraic equivalence means nothing is lost. ComparisonRow + "Predict and Verify" check.
- "The model sees the entire noise trajectory (all 1000 steps)" -- closed-form formula jumps to one random timestep. "One Random Timestep, Not a Sequence" card.
- "Training loops through timesteps 1 to T in order" -- uniform random sampling, mini-batch table showing different timesteps per image.
- "MSE on noise is a different loss function from the MSE I already know" -- side-by-side formula comparison (Series 1 vs DDPM) early in the hook. Three Faces of MSE table as synthesis.
- "The model needs a separate network for each timestep" -- one network conditioned on t. "One Network for All Timesteps" card. Architecture details deferred.

**What is NOT covered (deferred):**
- Reverse/sampling process algorithm (-> sampling-and-generation)
- U-Net architecture or network design (-> Module 6.3)
- How the network receives/uses timestep t internally (-> Module 6.3)
- Code implementation of training (-> build-a-diffusion-model)
- Full DDPM loss with weighting terms (mentioned, not derived)
- Score matching, SDEs, continuous-time formulations (out of scope)
- Classifier-free guidance or conditional generation (-> Module 6.3)

**Widget:** TrainingStepSimulator -- interactive exploration of one training step. Slider (+ preset buttons at t=50/200/500/800/950) selects a timestep. Displays: clean T-shirt image (x_0), noisy image (x_t) at selected timestep, predicted noise (epsilon_hat from simulated partially-trained network), actual noise (epsilon), and MSE loss between them. Live coefficients display (sqrt(alpha_bar) and sqrt(1-alpha_bar)). Signal-vs-noise percentage bar. Context-sensitive description text varying with noise level (easy/moderate/hard/very hard). Same procedural 28x28 T-shirt image and seeded PRNG as DiffusionNoiseWidget and AlphaBarCurveWidget for visual continuity.

**Notebook:** `notebooks/6-2-3-learning-to-denoise.ipynb` -- 4 exercises with scaffolding progression:
- Exercise 1 (Guided): Create noisy images at various timesteps using the closed-form formula. Predict-before-run prompts.
- Exercise 2 (Guided): Compute MSE loss manually between noise vectors, verify with nn.MSELoss.
- Exercise 3 (Supported): Fill in a skeleton of the DDPM training step function (data prep + loss computation, no real model).
- Exercise 4 (Independent): Predict and reason about which timesteps produce higher/lower loss for a trained model.

### sampling-and-generation (Lesson 4)

**Cognitive load:** BUILD (sustained from learning-to-denoise, before CONSOLIDATE capstone)

**Concepts taught:**
- DDPM reverse step formula (DEVELOPED)
- Stochastic noise injection during sampling -- the sigma_t * z term (DEVELOPED)
- DDPM sampling algorithm -- full loop from x_T to x_0 (DEVELOPED)
- One-shot denoising failure -- negative example (INTRODUCED)
- DDPM vs DDIM -- stochastic vs deterministic sampling (MENTIONED)
- 1000-step computational cost of sampling (INTRODUCED)
- Coarse-to-fine denoising progression (DEVELOPED -- extended from INTRODUCED in the-diffusion-idea)

**Mental models established:**
- "Destruction was easy and known. Creation requires a trained guide. At each step the guide points toward the clean image, but a small jitter keeps the path from collapsing to a single boring route. 1,000 tiny corrections compose into something that never existed."
- Noise injection in sampling as the same principle as temperature in language models: sigma_t > 0 = diverse/exploratory, sigma_t = 0 = deterministic/safe.
- "The last step commits. Every step before it explores." (z = 0 only at t=1)

**Analogies used:**
- Temperature in language models for stochastic noise injection (temperature > 0 = diverse, temperature = 0 = repetitive/safe; sigma_t plays the same role in diffusion sampling)
- Hiking toward a mountain with a slightly miscalibrated compass: the model (compass) points toward clean image (peak), noise injection (jitter) prevents consistently ending up in the wrong spot. Different hikes reach different spots on the mountain.
- Open-book exam contrast: in training, epsilon is the answer key (known). In sampling, there is no answer key -- epsilon_theta is all you have. This is the fundamental difference.

**How concepts were taught:**
- Reverse step formula: negative example first (one-shot denoising failure with vivid description: "foggy smear of gray"), then the formula with violet-highlighted box, then term-by-term breakdown (3 cards: scaling factor, noise removal, fresh noise injection). Connection to reparameterization trick pattern noted in TipBlock. Connection to forward formula derivation explained.
- Concrete walkthrough: immediately after term-by-term breakdown (concrete-before-abstract). Full numerical walkthrough at t=500 with linear schedule (alpha_bar=0.05, beta=0.01). Plugged into formula with simplified result. Second examples at t=950 and t=50 in side-by-side comparison cards showing how coefficients change behavior.
- Stochastic noise injection: motivated by temperature analogy, then "why it helps" argument (imperfect predictions + no noise = blurry averages), then hiking analogy, then DDPM vs DDIM ComparisonRow.
- Full sampling algorithm: 3 PhaseCards (sample pure noise, loop from T to 1, return x_0). t=1 special case in WarningBlock aside. Computational cost in amber GradientCard. Training vs sampling contrast in InsightBlock aside.
- Coarse-to-fine: DenoisingTrajectoryWidget with timeline slider, play/pause animation, key timestep snapshots, signal-vs-noise bar, context-sensitive descriptions. TryThisBlock prompts student to notice when structure first becomes recognizable.

**Misconceptions addressed:**
- "Sampling is just running the forward process in reverse (subtract noise)" -- corrected: you cannot rewind randomness; the reverse requires a trained model's prediction. Also, the reverse step ADDS fresh noise, which has no analog in the forward process. Addressed via transfer check GradientCard.
- "The denoised result is deterministic" -- corrected: the trained model is deterministic but the noise injection at each step introduces randomness. Different z draws produce different images from the same starting noise. Only DDIM removes this stochasticity.
- "All 1000 steps do equally important work" -- corrected: visually obvious in the DenoisingTrajectoryWidget. Early steps create structure (dramatic changes), late steps polish details (subtle refinements).
- "You need to store all 1000 intermediate images" -- corrected: memory check predict-and-verify shows each step overwrites the previous. Memory is constant.

**What is NOT covered (deferred):**
- Code implementation of the sampling algorithm (-> build-a-diffusion-model)
- DDIM or accelerated samplers in detail (-> Module 6.4)
- Classifier-free guidance or conditional generation (-> Module 6.3)
- U-Net architecture or how the network processes x_t and t (-> Module 6.3)
- Score matching, SDEs, continuous-time formulations (out of scope)
- Full variational derivation of the reverse process posterior (out of scope)

**Widget:** DenoisingTrajectoryWidget -- interactive visualization of the denoising trajectory from t=1000 (pure noise) to t=0 (generated image). Timeline slider with play/pause animation (150ms per frame, 41 frames). Key timestep snapshots strip (t=1000, 900, 800, 600, 400, 200, 100, 50, 0) as clickable thumbnails. Signal-vs-noise percentage bar. Context-sensitive description text (pure static / hints of structure / shape emerging / recognizable / fine details / nearly done / generated image). Same procedural 28x28 T-shirt image, cosine noise schedule, and seeded PRNG as all prior Module 6.2 widgets for visual continuity.

**Notebook:** None (by design). The student should internalize the sampling algorithm conceptually before implementing. Implementation deferred to Lesson 5 (build-a-diffusion-model).

### build-a-diffusion-model (Lesson 5 -- Capstone)

**Cognitive load:** CONSOLIDATE (zero new theoretical concepts -- pure implementation of existing knowledge)

**Concepts taught (all elevated from prior lessons):**
- Forward process implementation as q_sample() function (APPLIED -- elevated from DEVELOPED)
- DDPM training algorithm as a full training loop on MNIST (APPLIED -- elevated from DEVELOPED)
- DDPM sampling algorithm as a full reverse loop generating images (APPLIED -- elevated from DEVELOPED)
- DDPM reverse step formula as one line of PyTorch (APPLIED -- elevated from DEVELOPED)
- 1000-step computational cost of sampling (APPLIED -- elevated from INTRODUCED, measured firsthand)
- Skip connections in U-Net encoder-decoder (INTRODUCED -- first encounter in code)
- Timestep embedding as simple linear projection (INTRODUCED -- implementation detail, concept already known)

**Mental models reinforced:**
- "Every formula from the module became a line of PyTorch code." The closed-form formula became q_sample(). The 7-step training algorithm became train_epoch(). The reverse step formula became the core of sample().
- "Same building blocks, different question" confirmed experientially: the training loop heartbeat (forward -> loss -> backward -> update) is the same as Series 2. Only the data preparation (sample t, sample noise, create x_t) is diffusion-specific.
- "This slowness is NOT a bug" -- the 1000-step sampling cost is a fundamental property of pixel-space DDPM, not an implementation issue. Measured and felt firsthand.

**Analogies used:**
- No new analogies. The capstone reinforces existing analogies by grounding them in implementation. Callbacks to source lessons at every step: "This is the formula from The Forward Process," "These are the 7 steps from Learning to Denoise," "This is the algorithm from Sampling and Generation."

**How concepts were taught:**
- Notebook-driven capstone: the lesson page provides brief framing and a Colab link. The notebook IS the lesson.
- 6-part notebook with scaffolding progression: Parts 1-3 (Guided) build confidence by implementing the noise schedule, forward process, and reading the annotated U-Net architecture. Parts 4-5 (Supported) are the core exercises: fill in diffusion-specific parts of the training loop and sampling loop within provided skeletons. Part 6 (Independent) is reflection comparing VAE vs diffusion and bridging to Module 6.3.
- Predict-before-run prompts throughout: "What will alpha_bar[999] be?" "What will the image look like at t=999?" "How long will generating 64 images take?"
- Denoising diagnostic (after 5 epochs): single-step denoising predictions at multiple noise levels show the model IS learning but predictions are rough at high noise. This replaces full generation as a progress check without spoiling the Part 5 sampling exercise.
- Misconception #3 ("low loss = good generation") addressed via dedicated markdown section before sampling, explaining error accumulation across 1000 steps.
- Timing experiment: student predicts sampling time, then measures it. One training step: ~10ms. Generating 64 images: minutes. Scaling calculation to Stable Diffusion resolution. The wait is the lesson.
- VAE vs Diffusion reflection: quality (sharper diffusion) vs speed (instant VAE) tradeoff leads to the question "What if you ran diffusion in the VAE's latent space?" -- the bridge to Module 6.3.

**Misconceptions addressed:**
- "The U-Net must be complex and large to produce good results" -- a minimal <1M parameter model generates recognizable (if imperfect) MNIST digits. Architecture sophistication improves quality; it does not gate whether diffusion works.
- "Training should converge quickly since MSE is a simple loss" -- 5 epochs is not enough (shown via denoising diagnostic). The model must learn 1000 noise levels simultaneously with sparse per-level sampling.
- "If the loss is low, generation must be good" -- addressed via dedicated section before sampling. Average MSE across timesteps does not guarantee good generation because errors accumulate across 1000 sequential sampling steps.
- "Sampling should be roughly as fast as training" -- timed comparison reveals ~1000x slowdown. This is fundamental, not a bug.

**What is NOT covered (deferred):**
- Full U-Net with attention, group norm, sinusoidal positional encoding (-> Module 6.3)
- Conditional generation or classifier-free guidance (-> Module 6.3)
- Latent diffusion (-> Module 6.3)
- DDIM or accelerated samplers (-> Module 6.4)
- High-resolution images (MNIST 28x28 only)
- FID scores or formal evaluation metrics

**Notebook:** `notebooks/6-2-5-build-a-diffusion-model.ipynb` -- 6 parts with scaffolding progression:
- Part 1 (Guided): Noise schedule computation -- compute betas, alphas, alpha_bars. Plot and verify against widget from the-forward-process.
- Part 2 (Guided): Forward process -- implement q_sample(). Visualize noise progression on a real MNIST digit. Compare to DiffusionNoiseWidget.
- Part 3 (Guided, provided): Simple U-Net architecture with annotations. Student reads and understands, does not write from scratch. Encoder-decoder with 2 skip connections and timestep embedding via linear projection.
- Part 4 (Supported): Training loop -- fill in 4 diffusion-specific lines (sample t, sample noise, q_sample, MSE loss). Train 5 epochs (denoising diagnostic), then 15 more (total 20). Loss curve analysis.
- Part 5 (Supported): Sampling loop -- fill in reverse step formula and noise injection with t=0 special case. Generate 64 images. Timing experiment.
- Part 6 (Independent): VAE vs Diffusion reflection. Quality/speed tradeoff. Bridge to latent diffusion (Module 6.3).
