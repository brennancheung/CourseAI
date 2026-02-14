# samplers-and-efficiency -- Lesson Planning Document

**Module:** 6.4 (Stable Diffusion)
**Position:** Lesson 2 of 3 (Lesson 16 in Series 6)
**Type:** STRETCH (genuinely new concepts: ODE formulation of diffusion, DDIM deterministic sampling, higher-order solvers)
**Previous lesson:** stable-diffusion-architecture (CONSOLIDATE)
**Next lesson:** generate-with-stable-diffusion (CONSOLIDATE)

---

## Phase 1: Student State

### Relevant concepts the student has

| Concept | Depth | Source Lesson | Notes |
|---------|-------|--------------|-------|
| DDPM reverse step formula (x_{t-1} from x_t using epsilon_theta, with stochastic noise injection) | APPLIED | build-a-diffusion-model (6.2.5) | Student implemented the full sampling loop in PyTorch. Knows the formula: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta) + sigma_t * z. Knows every term: scaling factor, noise removal, fresh noise injection. This is the DDPM baseline that this lesson shows how to surpass. |
| DDPM sampling algorithm (full loop from x_T to x_0, 1000 steps) | APPLIED | build-a-diffusion-model (6.2.5) | Student implemented the full loop. Knows memory is constant (only x_t needed), knows the t=1 special case (z=0), and experienced the computational cost firsthand (minutes to generate 64 MNIST digits). |
| Stochastic noise injection during sampling (sigma_t * z term) | DEVELOPED | sampling-and-generation (6.2.4) | Student understands why noise is added: without it, the model commits to imperfect predictions and errors compound to blurry averages. Connected to temperature in LLMs (sigma > 0 = diverse, sigma = 0 = deterministic). The temperature/sigma analogy is a direct bridge to DDIM's sigma=0 formulation. |
| 1000-step computational cost of DDPM | APPLIED | build-a-diffusion-model (6.2.5) | Student timed the generation, computed scaling projections to larger models. Felt the slowness viscerally. This is the pain point this lesson resolves. |
| Coarse-to-fine denoising progression (early steps create structure, late steps refine details) | DEVELOPED | sampling-and-generation (6.2.4) | Extended from INTRODUCED via the DenoisingTrajectoryWidget. Student saw that not all steps do equally important work. This directly motivates step-skipping: if many intermediate steps are redundant, can we skip them? |
| DDPM vs DDIM (stochastic vs deterministic sampling) | MENTIONED | sampling-and-generation (6.2.4) | ComparisonRow showed key differences. DDIM described as "deterministic, faster sampling" that "trades diversity for speed." Student has the name and the one-sentence summary. This lesson elevates to DEVELOPED. |
| Closed-form formula q(x_t\|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon | DEVELOPED | the-forward-process (6.2.2) | Critical prerequisite. The student derived this step by step and verified it arithmetically. This formula is the foundation of the DDIM derivation: DDIM predicts x_0 from epsilon_theta, then uses the closed-form to jump to any intermediate timestep. |
| Alpha-bar (cumulative signal fraction, product of alphas) | DEVELOPED | the-forward-process (6.2.2) | Student used the AlphaBarCurveWidget. Knows alpha_bar is the "signal-to-noise dial" that can be evaluated at any timestep. This is essential for understanding why DDIM can skip steps: it only needs alpha_bar at the target timestep, not every intermediate alpha. |
| Full SD pipeline data flow (text -> CLIP -> denoising loop -> VAE decode -> pixel image) | DEVELOPED | stable-diffusion-architecture (6.4.1) | Taught one lesson ago. Student can trace the complete pipeline with tensor shapes at every stage. Knows the sampler/scheduler sits inside the denoising loop and decides how to use the U-Net's noise prediction to take a step. |
| Component modularity (swappable components via tensor handoffs) | INTRODUCED | stable-diffusion-architecture (6.4.1) | Student knows components can be swapped independently. The previous lesson explicitly noted "you can swap the scheduler/sampler (DDPM, DDIM, Euler -- Lesson 16)." This lesson delivers on that promise. |
| Compute cost breakdown (denoising loop dominates: 100 U-Net passes with CFG at 50 steps) | INTRODUCED | stable-diffusion-architecture (6.4.1) | Student knows the bottleneck is the denoising loop, not CLIP encoding or VAE decoding. Fewer denoising steps = proportionally less compute. |
| DDPM training objective (predict epsilon, MSE loss) | APPLIED | build-a-diffusion-model (6.2.5) | The model predicts the noise that was added. Critical: the SAME trained model is used by all samplers. DDIM, Euler, DPM-Solver do not require retraining. |

### Mental models and analogies already established

- **"Destruction is easy; creation from scratch is impossibly hard; undoing a small step is learnable."** -- Core diffusion insight from the-diffusion-idea (6.2.1).
- **"Alpha-bar is the signal-to-noise dial."** -- From the-forward-process (6.2.2). The dial can be turned to any position via the closed-form formula.
- **Temperature in LLMs maps to sigma_t in diffusion.** -- From sampling-and-generation (6.2.4). sigma > 0 = diverse/exploratory, sigma = 0 = deterministic.
- **"The last step commits. Every step before it explores."** -- From sampling-and-generation (6.2.4). z=0 at t=1 only.
- **"Not all steps do equally important work."** -- From the DenoisingTrajectoryWidget (6.2.4). Early steps create structure, late steps polish details.
- **"Three translators, one pipeline."** -- From stable-diffusion-architecture (6.4.1). CLIP translates language, U-Net generates in latent space, VAE translates to pixels.
- **"Assembly line"** -- From stable-diffusion-architecture (6.4.1). Components are modular and swappable.
- **"Same building blocks, different question"** -- Recurring meta-pattern throughout Series 6.

### What was explicitly NOT covered in prior lessons (relevant here)

- **DDIM's actual mechanism** -- The student knows DDIM exists and is "deterministic, faster" but has zero understanding of how it works. They do not know that DDIM reinterprets the forward process as non-Markovian, predicts x_0 first, or uses the closed-form formula to jump to an arbitrary earlier timestep.
- **The ODE perspective on diffusion** -- The student has only seen diffusion as a discrete Markov chain (step-by-step, each step depends on the previous). The idea that the same trained model implicitly defines an ODE whose trajectory can be followed with standard numerical methods is entirely new.
- **Higher-order solvers (DPM-Solver, etc.)** -- The student has no concept of solver order. They do not know what "order" means in the context of ODE solvers, or why higher-order methods converge faster.
- **The Euler method as an ODE solver** -- While the student has used forward Euler implicitly (gradient descent updates are Euler steps on the loss surface), this connection has never been made explicit and "Euler method" as a named concept has not been taught.
- **Why step-skipping works** -- The student intuits that not all steps are equally important (DenoisingTrajectoryWidget), but does not have the mathematical framework for why certain steps can be skipped without quality loss.
- **Practical sampler selection guidance** -- Which sampler, how many steps, quality/speed tradeoffs for different use cases.

### Readiness assessment

The student is well-prepared but this will be the hardest lesson in Module 6.4. The foundation is strong: they have the DDPM sampling algorithm at APPLIED depth (implemented it), the closed-form formula at DEVELOPED depth (derived and verified it), and the computational cost pain at APPLIED depth (felt it). They have DDIM at MENTIONED depth (name-dropped in 6.2.4), so the name is familiar even though the mechanism is new.

The key risk is cognitive overload from three genuinely new ideas: (1) the ODE reframing of diffusion, (2) DDIM's predict-x_0-then-jump mechanism, and (3) higher-order solvers. The mitigation is careful sequencing: DDIM first (smallest conceptual delta from DDPM, student already heard the name), then the ODE perspective (reframes what they just learned about DDIM), then higher-order solvers (natural follow-up once ODE is established). Each builds on the previous, and connecting back to the closed-form formula the student already knows well reduces the actual novelty.

The emotional setup is strong. The student experienced the 1000-step pain in Module 6.2 and was told it would be addressed. The previous lesson showed the sampler as a swappable component in the pipeline. This lesson delivers the resolution: "Here is why you can generate in 20 steps with the exact same model."

---

## Phase 2: Analysis

### Target concept

This lesson teaches the student to explain why advanced samplers (DDIM, Euler, DPM-Solver) can generate images in 20 steps using the exact same trained model that DDPM needs 1000 steps for, by understanding that the model's noise predictions implicitly define a trajectory that can be traversed with different step sizes and solver strategies.

### Prerequisites table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| DDPM reverse step formula | DEVELOPED | APPLIED | build-a-diffusion-model (6.2.5) | OK | Student must understand each term in the DDPM step formula to see what DDIM changes. Has it at APPLIED (above requirement). |
| Closed-form formula q(x_t\|x_0) | DEVELOPED | DEVELOPED | the-forward-process (6.2.2) | OK | DDIM's core mechanism: predict x_0, then use the closed-form formula to jump to an earlier timestep. Student derived and verified this formula. |
| Alpha-bar notation and semantics | INTRODUCED | DEVELOPED | the-forward-process (6.2.2) | OK | Appears throughout every sampler formula. Student has it at DEVELOPED. |
| Stochastic noise injection (sigma_t * z in DDPM) | DEVELOPED | DEVELOPED | sampling-and-generation (6.2.4) | OK | Must understand the role of noise injection to appreciate what happens when DDIM removes it (sigma=0). Student knows the temperature analogy and the diversity argument. |
| 1000-step computational cost | APPLIED | APPLIED | build-a-diffusion-model (6.2.5) | OK | Must have felt the pain to care about the solution. Student timed it. |
| DDPM training objective (predict epsilon) | DEVELOPED | APPLIED | build-a-diffusion-model (6.2.5) | OK | Must know what the model outputs to understand how all samplers use that output differently. Student has it at APPLIED. |
| Coarse-to-fine denoising progression | INTRODUCED | DEVELOPED | sampling-and-generation (6.2.4) | OK | Provides intuition for why step-skipping can work: if intermediate steps are doing similar work, you can skip some. Student has it at DEVELOPED. |
| Gradient descent as iterative optimization | APPLIED | APPLIED | Series 1 (1.1) | OK | The Euler method connection: gradient descent IS Euler's method applied to the loss surface. Student has performed thousands of gradient descent steps. This familiar concept becomes the bridge to ODE solvers. |
| Full SD pipeline with sampler as a component | DEVELOPED | DEVELOPED | stable-diffusion-architecture (6.4.1) | OK | Student knows where the sampler sits in the pipeline and that it is swappable. This lesson explains what the different samplers actually do. |

### Gap resolution

No gaps. All prerequisites are at or above required depth. The student has every concept needed at DEVELOPED or APPLIED depth. The genuinely new concepts (DDIM mechanism, ODE perspective, higher-order solvers) are the lesson's core content, not prerequisites.

### Misconceptions table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Better samplers require retraining the model" | Throughout the course, improving performance has required changing the model (better architecture, more training, different loss). The student's instinct is that fundamentally different behavior requires a fundamentally different model. Also, DDIM was published as a separate paper (Song et al., 2020), which might suggest a different training procedure. | DDIM, Euler, and DPM-Solver all use the EXACT same epsilon_theta(x_t, t) function from the trained DDPM. The model's job never changes: given a noisy input and a timestep, predict the noise. The sampler's job is to decide what to DO with that prediction. You can swap samplers after training without changing a single weight. In diffusers, this is literally `pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)` -- one line, no retraining. | Section 3 (Hook), established as the foundational insight before any sampler is introduced. Reinforced with each new sampler. |
| "DDIM is a different generative model from DDPM" | DDIM has a different name, a different paper, and produces different results (deterministic vs stochastic). The student may think DDIM is a distinct approach rather than a different way to use the same trained model. | DDIM uses the same trained model, the same noise prediction, even the same forward process for training. The ONLY difference is in the reverse process (sampling). DDIM reinterprets the trained model's output to enable larger steps and deterministic generation. If you train a DDPM and save the weights, you can sample from it using either DDPM or DDIM without any modification. | Section 4 (DDIM), immediately when introducing DDIM. Framed as "same model, different sampler." |
| "Fewer steps always means worse quality" | The student experienced that 1000 steps in DDPM produces good results, and they may assume quality degrades linearly with fewer steps. The DDPM Markov chain was presented as needing each step, reinforcing the idea that skipping steps loses information. | DDIM at 50 steps produces results comparable to DDPM at 1000 steps. DPM-Solver at 20 steps matches or exceeds DDPM at 1000. The quality-steps relationship is highly nonlinear and sampler-dependent. At very low step counts (5-10), quality does degrade, but the "cliff" is much further out than the student expects. Show a quality-vs-steps comparison: DDPM drops sharply below ~200, DDIM maintains quality down to ~20, DPM-Solver maintains quality down to ~10-15. | Section 7 (Elaborate), after all three samplers are introduced. ComparisonRow or chart showing quality-vs-steps for different samplers. |
| "The ODE/SDE perspective is a completely new way to think about diffusion" | The ODE language (trajectories, vector fields, solvers) sounds very different from the Markov chain language (steps, transitions, probabilities). The student may think these are fundamentally different frameworks. | The ODE perspective describes the SAME process: moving from noise to data. The Markov chain view says "take small random steps." The ODE view says "follow a smooth trajectory." Same landscape, same destination, different description of the path. The trained model defines the direction at every point; the sampler decides the step size. The ODE perspective is a lens that reveals why step-skipping is mathematically justified, not a replacement for the Markov chain view. | Section 5 (ODE perspective), as an explicit bridge. "Same landscape, different lens." |
| "DDIM works by just skipping timesteps in the DDPM algorithm (doing every 50th step instead of every step)" | The student knows DDPM iterates t from T to 1. The simplest mental model of "faster" is "skip steps" -- just do steps 1000, 950, 900, ... instead of every step. This is closer to the truth than the other misconceptions but misses the crucial mechanism: DDIM does not use the DDPM step formula at all. It predicts x_0, then jumps to the target timestep using the closed-form formula. | If you literally skip steps in the DDPM algorithm (apply the DDPM reverse step formula at t=1000, then at t=950, skipping 949 intermediate steps), the result is terrible because the DDPM formula assumes small steps (small beta_t). The coefficients are calibrated for adjacent timesteps, not for jumps of 50. DDIM uses a completely different step formula that is designed for arbitrary step sizes. The key ingredient: predict x_0 first, then use the closed-form formula (which the student derived in 6.2.2) to jump to any timestep. | Section 4 (DDIM), as a negative example before revealing the actual DDIM mechanism. Show what happens when you naively skip steps in DDPM to motivate why a different formula is needed. |

### Examples planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| DDPM at 1000 steps vs DDIM at 50 steps vs DPM-Solver at 20 steps: same model, same prompt, same seed, dramatically different step counts, comparable quality | Positive (anchor) | The opening demonstration that makes the lesson's promise concrete. Before any explanation, show the student that the same trained model produces comparable results at 1000, 50, and 20 steps with different samplers. This creates the "how is that possible?" question that the lesson answers. | Establishes the concrete phenomenon before explaining it. Uses the same model/prompt/seed to make the "same model" claim undeniable. The step count ratio (1000 : 50 : 20 = 50:1 speedup) is dramatic enough to generate genuine curiosity. |
| DDIM's predict-x_0-then-jump mechanism, traced step by step with the closed-form formula at two timesteps: t=1000 -> t=800 (a large jump) and t=200 -> t=0 (the final jump) | Positive (concrete) | Shows exactly how DDIM takes one step: (1) predict epsilon, (2) compute x_0 estimate, (3) use the closed-form formula to compute x_{t-next} at an arbitrary target timestep. The closed-form formula is the student's own tool from 6.2.2, used in a new context. Two examples show the mechanism works at different noise levels. | Concrete-before-abstract. The student must see the mechanism with specific numbers before understanding the general principle. Two timesteps (high noise and low noise) show the formula generalizes. The callback to the closed-form formula from 6.2.2 reduces novelty: "you already have the tool; DDIM just uses it differently." |
| Naive step-skipping with DDPM formula (every 50th step) producing garbage results | Negative | Disproves misconception #5. Shows that step-skipping is not just "do fewer iterations of the same formula." The DDPM step formula assumes small, adjacent steps. Applying it with large gaps produces catastrophic errors because the coefficients are calibrated for beta_t (small), not for alpha_bar ratios spanning 50 timesteps. | This negative example is essential to motivate why DDIM needs a different formula. Without it, the student might think DDIM is "just DDPM with fewer steps" and miss the actual mechanism. The failure of naive skipping creates the need for the predict-x_0-then-jump approach. |
| Gradient descent as Euler's method on the loss surface (bridge analogy) | Positive (bridge) | Connects the ODE perspective to something deeply familiar. Gradient descent says: "compute the gradient (direction), take a step of size lr (step size), repeat." Euler's method on an ODE says: "compute the derivative (direction), take a step of size h (step size), repeat." Same algorithm, different domain. The student has done gradient descent thousands of times; they have been doing Euler's method all along. | The ODE perspective is the biggest conceptual leap in this lesson. Without a bridge to familiar concepts, it risks feeling like abstract math disconnected from practice. Gradient descent is the most deeply practiced algorithm in the student's repertoire (APPLIED from Series 1). Making the connection explicit turns a scary new concept into a reframing of an old friend. |
| DPM-Solver's higher-order steps: looking at the model's predictions at TWO nearby timesteps to estimate curvature, enabling larger accurate steps | Positive (stretch) | Extends the Euler -> higher-order progression. Euler uses the derivative at one point (first-order). DPM-Solver uses derivatives at multiple points to estimate how the trajectory curves, enabling larger steps that stay on track. Analogy: driving on a straight road, you can go faster; on a curvy road, you need to slow down. Higher-order methods read the road ahead. | This is the stretch example that completes the sampler hierarchy. It must be concrete enough to build intuition but does not need the full math (INTRODUCED depth). The driving analogy makes the concept accessible without requiring ODE theory. |

---

## Phase 3: Design

### Narrative arc

The student has a problem: DDPM generation is painfully slow. They felt this in Module 6.2 when generating 28x28 MNIST digits took minutes. The previous lesson showed that Stable Diffusion runs 50-100 denoising steps at 512x512 resolution, each step requiring a full U-Net forward pass (or two with CFG). The pipeline works, but the speed bottleneck is real. The student has been told, since 6.2.4, that faster samplers exist. They have seen "DDIM" mentioned once, and the previous lesson noted that schedulers are swappable. But they do not understand HOW fewer steps can produce comparable results. The Markov chain view of DDPM says each step depends on the previous one -- how can you skip steps? This lesson reveals the answer: DDPM's step formula is not the only way to use a noise-prediction model. The model predicts epsilon; what you DO with that prediction is a separate choice. DDIM reinterprets the same prediction to enable large jumps between timesteps using the student's own closed-form formula from Module 6.2. The ODE perspective reveals WHY this works: the model's predictions at every point in noise-space define a smooth trajectory, and different samplers are just different ways to follow that trajectory -- some careful and slow (DDPM, tiny steps with random jitter), some efficient and direct (Euler, medium steps along the trajectory), some sophisticated and fast (DPM-Solver, large steps that account for curvature). The same trained model, the same learned landscape, three different walking speeds.

### Modalities planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual (diagram) | (1) A trajectory diagram showing the path from noise (z_T) to clean data (z_0) in a 2D projected space. Three overlapping trajectories: DDPM (many tiny steps with random jitter, dense dots), DDIM (fewer evenly-spaced points along a smooth curve, no jitter), DPM-Solver (even fewer points, slightly curved arcs between them). Same start, same end, different paths. (2) A side-by-side comparison of the DDIM step: epsilon_theta -> predicted x_0 -> closed-form jump to x_{t-next}, shown as a two-hop diagram (predict backward to x_0, jump forward to x_{t-next}). | The trajectory view is the core spatial/geometric intuition for this lesson. The student needs to SEE that all samplers traverse the same space but take different-sized steps. DDPM's dense random walk vs DDIM's clean striding vs DPM-Solver's efficient leaping makes the speed difference visually obvious. The DDIM step diagram makes the predict-then-jump mechanism concrete: the detour through x_0 is the key insight. |
| Symbolic (formula/code) | (1) The DDIM step formula: x_{t-1} = sqrt(alpha_bar_{t-1}) * predicted_x_0 + sqrt(1 - alpha_bar_{t-1}) * direction_pointing_to_x_t. With predicted_x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t). (2) Side-by-side: DDPM step formula vs DDIM step formula, with shared terms highlighted and differences marked. (3) The Euler step formula: x_{t-h} = x_t + h * f(x_t, t), with f derived from the model's noise prediction. | Formulas are necessary for precision. The student has the DDPM formula at APPLIED depth, so showing the DDIM formula side-by-side lets them compare term by term. The predicted_x_0 sub-formula is a rearrangement of the closed-form formula the student already knows, reducing novelty. The Euler formula connects to gradient descent's update rule (x <- x - lr * grad), making the ODE perspective concrete. |
| Concrete example | DDIM step traced at t=1000 -> t=800 with specific alpha_bar values: (1) start with x_1000, (2) model predicts epsilon_theta, (3) compute predicted_x_0 using the rearranged closed-form, (4) use the closed-form to jump to t=800 using alpha_bar_800. Show the two alpha_bar values (alpha_bar_1000 near 0, alpha_bar_800 slightly higher), the signal/noise coefficients at both points. Contrast with DDPM which would need 200 individual steps for the same transition. | Makes the mechanism viscerally concrete. Specific alpha_bar values (numbers from the student's own noise schedule) ground the formula in reality. The contrast with DDPM's 200 steps makes the efficiency gain tangible. |
| Verbal/Analogy | (1) "Same vehicle, different route": the model (vehicle) is the same. DDPM takes every back road. DDIM takes the highway. DPM-Solver takes the highway with GPS that reads the road ahead. (2) "Predict-and-leap": DDIM's mechanism. At each step, DDIM asks "if I had to guess the final destination right now, where would it be?" (predict x_0), then says "OK, given that destination, where should I be at timestep t-next?" (closed-form jump). It is a navigator constantly recalculating the route. (3) Gradient descent IS Euler's method: "You have been doing Euler's method since Series 1. Every time you took a gradient step -- compute direction, scale by learning rate, update -- that was one Euler step on the ODE of steepest descent." | Three analogies for three levels of the concept. The vehicle/route analogy establishes the foundational insight (same model, different usage). The predict-and-leap analogy makes DDIM's specific mechanism memorable. The gradient descent bridge connects the ODE perspective to the student's most practiced algorithm, defusing the intimidation of "ODE" terminology. |
| Intuitive | The "of course" moment: the model predicts epsilon at EVERY timestep, including ones far from the current step. Its predictions at t=1000 already contain information about t=0 (that is literally what "predict the noise" means -- predicting the noise IS predicting x_0). DDPM ignores this: it takes a tiny step and asks again. DDIM uses it: it extracts the full x_0 prediction and leaps. Of course you can go faster if you use more of what the model already tells you. | The intuitive modality turns the "how is this possible?" into an "of course." The student knows the model predicts epsilon at each step. The DDIM insight is that this prediction contains MORE information than DDPM uses. Once the student sees this, the speed improvement feels inevitable rather than magical. |

### Cognitive load assessment

- **New concepts in this lesson:** Three genuinely new concepts:
  1. **DDIM's predict-x_0-then-jump mechanism** -- new sampling formula using the closed-form formula in a new way. The conceptual delta from DDPM is moderate: same model, different step formula.
  2. **The ODE perspective on diffusion** -- the model's noise predictions define a continuous trajectory that can be followed with different step sizes. This is a new lens on a familiar process.
  3. **Higher-order solvers (DPM-Solver)** -- using multiple model evaluations to estimate trajectory curvature for larger accurate steps. This is at INTRODUCED depth, not DEVELOPED.

  Three new concepts is at the upper limit. Mitigation: concept 1 (DDIM) heavily leverages the closed-form formula at DEVELOPED depth, concept 2 (ODE) is bridged via gradient descent at APPLIED depth, and concept 3 (DPM-Solver) is kept at INTRODUCED (intuition only, no formula derivation).

- **Previous lesson load:** CONSOLIDATE (stable-diffusion-architecture)
- **This lesson's load:** STRETCH -- appropriate. Sandwiched between CONSOLIDATE on both sides (stable-diffusion-architecture before, generate-with-stable-diffusion after). The module plan explicitly designates this as STRETCH and notes the "comfortable cognitive bookends." Coming after CONSOLIDATE gives the student cognitive capacity. Going to CONSOLIDATE afterward provides relief.

### Connections to prior concepts

| Prior Concept | How It Connects | Risk of Misleading? |
|---------------|----------------|--------------------|
| Closed-form formula (6.2.2) | This is DDIM's secret weapon. The closed-form formula q(x_t\|x_0) lets you compute x_t at ANY timestep given x_0, without iterating through intermediate steps. DDIM reverses this: given a predicted x_0, compute x_{t-next} at any desired timestep. The student derived this formula and verified it -- now they see it used in a new way. | No risk. Direct application of a well-understood formula. |
| Stochastic noise injection / temperature analogy (6.2.4) | DDIM is what happens when you set sigma=0 (no noise injection). The student already saw this framed as a possibility in 6.2.4's DDPM vs DDIM ComparisonRow. The temperature analogy (sigma>0 = diverse, sigma=0 = deterministic) provides direct intuition for DDIM's deterministic nature. | Low risk. The student may initially think sigma=0 in DDPM = DDIM, but that is not quite right. DDPM's formula with sigma=0 is numerically unstable for large steps because it assumes adjacent timesteps. DDIM reformulates to handle arbitrary step sizes. Must clarify this distinction. |
| Gradient descent (Series 1) | Gradient descent IS Euler's method on the loss landscape ODE. The update rule x <- x - lr * grad has the same structure as Euler's method: x_{t+h} = x_t + h * f(x_t). The learning rate is the step size. The gradient is the direction. The student has done thousands of these steps. | Low risk. The analogy is structurally exact but the student may not have thought of gradient descent as "solving an ODE." Framing it as "you have been doing this all along, you just did not know it had a name" reduces intimidation. Must not overstate: gradient descent on a loss surface is not the same ODE as diffusion, but the METHOD (Euler) is the same. |
| "Not all steps do equally important work" (6.2.4) | This DenoisingTrajectoryWidget insight directly supports step-skipping. If many intermediate steps do similar refinement work, a smart sampler can combine those refinements into fewer, larger steps. | No risk. Natural extension of an established observation. |
| "Same building blocks, different question" (recurring) | Extends to samplers: same noise prediction model, same trained weights, but a different question -- "how should I use this prediction to take a step?" DDPM asks "what is the adjacent step?" DDIM asks "what would x_0 look like, and where should I be 200 steps from now?" | No risk. Natural extension of the recurring theme. |
| DDPM vs DDIM ComparisonRow (6.2.4) | The student has seen this comparison at MENTIONED depth. This lesson fulfills the promise: now they learn what DDIM actually does and why it works. | No risk. The prior MENTIONED treatment set up this lesson's delivery. |

### Scope boundaries

**This lesson IS about:**
- Why DDPM requires many steps (Markov chain assumption, formula calibrated for adjacent timesteps)
- DDIM's predict-x_0-then-jump mechanism and why it enables step-skipping (DEVELOPED)
- The ODE perspective: the model's predictions define a trajectory; samplers are different ways to follow it (INTRODUCED, intuition-level)
- Euler method as the simplest ODE solver, connected to gradient descent (INTRODUCED)
- DPM-Solver as a higher-order solver that reads trajectory curvature (INTRODUCED)
- Why all samplers use the same trained model (no retraining)
- Practical sampler comparison: quality vs speed tradeoffs, recommended step counts
- Deterministic vs stochastic sampling tradeoff

**This lesson is NOT about:**
- Deriving the full DDIM or DPM-Solver formulas from first principles (the student sees the formulas and understands the mechanism, but does not derive them from the variational bound or the probability flow ODE)
- Score-based models or the SDE/ODE duality in full mathematical rigor (MENTIONED as context, not developed)
- Implementing samplers from scratch (the notebook uses diffusers schedulers)
- Ancestral sampling, Karras samplers, or UniPC (out of scope; too many sampler variants would overwhelm)
- Training-based acceleration methods (distillation, consistency models -- these are Series 7)
- SD v1 vs v2 vs XL architectural differences
- Any changes to the model architecture or training procedure

**Depth targets:**
- DDIM predict-x_0-then-jump mechanism: DEVELOPED (student can explain the two-step process, trace a step with the formula, and explain why it enables step-skipping)
- ODE perspective on diffusion: INTRODUCED (student understands the trajectory interpretation and can explain why different step sizes work, but does not derive the probability flow ODE)
- Euler method as ODE solver: INTRODUCED (student recognizes the connection to gradient descent and can explain the step formula, but does not implement it)
- DPM-Solver / higher-order methods: INTRODUCED (student understands the concept of using multiple evaluations for curvature estimation, but does not work with the formula)
- Sampler comparison / practical guidance: INTRODUCED (student knows which samplers to use for which situations)

### Lesson outline

**1. Context + Constraints**
- This is Lesson 16, the second lesson in Module 6.4. The student assembled the full pipeline in Lesson 15. This lesson addresses the remaining pain point: the denoising loop is the computational bottleneck, and DDPM needs many steps.
- This is a STRETCH lesson: three genuinely new concepts, but all build on deeply familiar foundations (closed-form formula, gradient descent, the DDPM step formula the student implemented).
- Scope: DDIM, Euler, DPM-Solver. NOT: score-based models in detail, sampler implementation from scratch, training-based acceleration (consistency models = Series 7).
- By the end: the student can explain why 20 steps produces comparable results to 1000, and can choose a sampler with understanding rather than blind defaults.

**2. Recap (brief)**
- Remind the student of the DDPM sampling loop: at each step, the model predicts epsilon, the formula computes x_{t-1}, and fresh noise z is injected. 1000 steps, each requiring a U-Net forward pass. With CFG, that is 2000 forward passes per generation.
- Remind the student of the closed-form formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon. "You can compute the noisy version of any image at ANY timestep directly. No iteration needed."
- Connection to the pipeline: the scheduler/sampler component in the SD pipeline decides how to use the U-Net's prediction. This lesson explains three different strategies.

**3. Hook: "Same Model, Different Speed"**
- Type: Before/after comparison / challenge preview.
- Show the student: three images generated with the SAME model, SAME prompt, SAME seed. One took 1000 steps (DDPM), one took 50 steps (DDIM), one took 20 steps (DPM-Solver). Results are comparable.
- "The model was trained once. These three images used the same weights, the same noise prediction function. The ONLY difference: how those predictions were used to take steps. This lesson explains how."
- Establish the foundational insight early: "The model predicts noise. The sampler decides what to do with that prediction. Swapping samplers is an inference-time decision that requires zero retraining."

**4. Explain: DDIM -- Predict and Leap (DEVELOPED)**

**4a. Why DDPM is slow (the Markov chain constraint)**
- DDPM's reverse step formula assumes adjacent timesteps: x_{t-1} from x_t. The coefficients (1/sqrt(alpha_t), beta_t/sqrt(1-alpha_bar_t)) are calibrated for the tiny transition from t to t-1, where beta_t is small (~0.0001 to 0.02).
- If you try to skip steps (apply the DDPM formula from t=1000 to t=950), the coefficients are wrong. Beta_t does not scale linearly. The result: catastrophic errors, garbage output.
- Negative example: show what happens with naive step-skipping in DDPM. The formula breaks because it was designed for small steps.

**4b. DDIM's insight: predict x_0 first**
- The model predicts epsilon_theta(x_t, t). But there is a direct relationship between epsilon and x_0 (from the closed-form formula): x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t).
- This predicted_x_0 is the model's current best guess for the clean data. It is noisy and imperfect (especially at high t), but it contains information about the destination.
- "DDIM asks: if my destination is predicted_x_0, where should I be at timestep t_next?"
- The answer: use the closed-form formula in the forward direction. x_{t_next} = sqrt(alpha_bar_{t_next}) * predicted_x_0 + sqrt(1 - alpha_bar_{t_next}) * "direction pointing to x_t". (The direction term ensures the step moves toward the data, using the model's prediction to correct course.)

**4c. The DDIM step formula**
- Side-by-side comparison: DDPM step formula vs DDIM step formula. Highlight what changed:
  - DDPM: uses alpha_t, beta_t (adjacent-step quantities). Adds noise z.
  - DDIM: uses alpha_bar_t and alpha_bar_{t_next} (cumulative quantities). No noise (sigma=0 for deterministic). Can jump to ANY t_next, not just t-1.
- ConceptBlock: "DDIM uses alpha_bar, not alpha. That is the difference between a step-by-step formula and a leap formula. Alpha_bar encodes the entire schedule up to timestep t -- the student's 'signal-to-noise dial' from 6.2.2."
- Callback: "The closed-form formula let you DESTROY an image to any noise level in one step. DDIM lets you REVERSE that destruction in large steps, using the same mathematical tool."

**4d. Concrete example: one DDIM step from t=1000 to t=800**
- Specific alpha_bar values (from a cosine schedule, which the student knows). Show the computation step by step.
- Compare: DDPM would need 200 individual steps for the same transition. DDIM does it in one.

**4e. Deterministic generation and the sigma parameter**
- DDIM with sigma=0 is fully deterministic: same z_T, same prompt = same image, every time, regardless of step count.
- DDIM with sigma>0 interpolates between DDIM and DDPM behavior. sigma=1 recovers DDPM.
- Connection to the temperature analogy from 6.2.4: "sigma is the temperature dial. DDPM is always at temperature > 0. DDIM can turn it to zero."
- Why deterministic is useful: reproducibility, interpolation in z_T space (same trajectory from different starting points), and editing workflows.

**5. Check (predict-and-verify)**
- "Your DDIM sampler uses 50 steps for a 1000-timestep model. What is the step size?" (Answer: 20 timesteps per step. The sampler selects 50 evenly-spaced timesteps from [1000, 980, 960, ..., 20] and takes one DDIM step between each pair.)
- "If DDIM predicts x_0 at every step, why not just use the x_0 prediction from the very first step (t=1000)?" (Answer: the prediction at t=1000 is terrible because the input is almost pure noise. The model has very little signal to work with. Each step refines the prediction: at t=800, the model works with a less noisy input and produces a better x_0 estimate. The iterative refinement is still needed, just with fewer, larger steps.)

**6. Explain: The ODE Perspective (INTRODUCED)**

**6a. Bridge from gradient descent**
- "You have been using Euler's method since Series 1." The gradient descent update: theta <- theta - lr * gradient. This is Euler's method applied to the ODE d_theta/dt = -gradient(theta). The learning rate is the step size. The gradient is the direction.
- ConceptBlock: "An ODE says 'at every point, the direction to move is...' A solver (like Euler) says 'OK, I will take a step of size h in that direction, then check again.' Gradient descent is exactly this, on the loss surface."

**6b. Diffusion as an ODE**
- The model's noise prediction at every (x_t, t) defines a direction: "to get closer to clean data, move this way." This is a vector field over the entire noise-data space.
- The collection of all these directions forms an ODE: dx/dt = f(x, t), where f is derived from the model's noise prediction.
- Following this ODE from t=T (noise) to t=0 (data) traces a trajectory through the space. This trajectory IS the generation process.
- Visual: the trajectory diagram. At every point, the model provides a direction arrow. The trajectory follows these arrows from noise to data.

**6c. Why the ODE view enables efficiency**
- If the trajectory is smooth (and it is, because the noise schedule is smooth and the model is a smooth neural network), then you do not need to check the direction at every tiny step. You can take larger steps and still follow the trajectory accurately.
- Euler's method: take a step of size h in the direction the model gives you. Check again. Repeat. Fewer checks = fewer model evaluations = faster.
- "DDIM IS Euler's method on the diffusion ODE." (This is approximately true and a useful simplification. DDIM was derived independently but is equivalent to a first-order ODE solver on the probability flow ODE.)

**6d. Score-based connection (MENTIONED)**
- Brief context: Song et al. (2021) showed that the DDPM forward process can be described by a stochastic differential equation (SDE), and there is a corresponding ODE (the probability flow ODE) that traces the same marginal distributions. This is why DDIM works: it follows the probability flow ODE.
- "You do not need the SDE/ODE duality to use samplers. But if you read papers and see 'probability flow ODE,' this is what they mean: the deterministic trajectory that DDIM follows."
- InsightBlock: keep this brief. Two sentences of context, not a derivation. The student needs the vocabulary for reading papers, not the math.

**7. Explain: Higher-Order Solvers -- DPM-Solver (INTRODUCED)**

**7a. The problem with Euler: linear approximation**
- Euler's method assumes the direction does not change between steps. It takes a straight-line step. If the trajectory curves, Euler overshoots.
- At low step counts (<20 steps with Euler), quality drops because the trajectory DOES curve and Euler cannot account for it.
- Analogy: driving on a curvy road at high speed. If you can only check the road direction once per second (Euler), you miss turns. If you check multiple times (higher-order), you can anticipate curves and stay on the road.

**7b. DPM-Solver: reading the road ahead**
- DPM-Solver (Lu et al., 2022) evaluates the model at multiple nearby timesteps to estimate how the direction is changing (curvature).
- First-order DPM-Solver is essentially Euler (one evaluation).
- Second-order: two evaluations per step, accounts for how the direction changes.
- Third-order: three evaluations, accounts for how the change-in-direction changes.
- Result: DPM-Solver-2 at 20 steps matches DDIM at 50 steps. DPM-Solver-3 at 15-20 steps produces excellent results.
- TipBlock: more evaluations per step, but far fewer steps total. Net result: fewer total model evaluations.
- Framing: keep at intuition level. The student understands WHAT higher-order means (multiple evaluations to estimate curvature) and WHY it helps (larger accurate steps). They do not need the actual multi-step formulas.

**8. Elaborate: Sampler Comparison and Practical Guidance**

**8a. Quality vs steps comparison**
- ComparisonRow or chart comparing step count vs quality for DDPM, DDIM, Euler, DPM-Solver.
- Key takeaways:
  - DDPM: needs 200+ steps for good quality. Below 100, degrades rapidly.
  - DDIM: good quality at 50 steps, acceptable at 20, degrades below ~10.
  - Euler: similar to DDIM (first-order ODE solver).
  - DPM-Solver: good quality at 20 steps, acceptable at 10-15, the current standard.

**8b. When to use which sampler**
- DPM-Solver (or DPM-Solver++): best default for most generation. 20-30 steps.
- DDIM: good when you need deterministic generation (reproducibility, interpolation). 50 steps.
- Euler: simple, well-understood. Good at 30-50 steps. Often used for debugging.
- DDPM: only when you want maximum quality and do not care about speed, or for research purposes. 1000 steps.
- WarningBlock: "More steps does NOT always help with advanced samplers. DPM-Solver at 200 steps is not meaningfully better than at 25. You are paying more compute for diminishing returns. Use the sampler's recommended range."

**8c. Stochastic vs deterministic**
- Deterministic samplers (DDIM sigma=0, Euler): same seed = same image. Useful for reproducibility, A/B testing parameter changes, interpolation.
- Stochastic samplers (DDPM, DDIM sigma>0, some Euler variants): add noise at each step. More diverse outputs but not reproducible. Can sometimes produce higher quality at the cost of consistency.
- Connection to the temperature analogy: this is the same tradeoff as temperature in language models, applied to image generation.

**9. Check (transfer)**
- "Your colleague claims they found a way to make DPM-Solver even faster by increasing the solver order to 10 (evaluating the model 10 times per step). Why might this NOT work as expected?" (Answer: each evaluation costs a full U-Net forward pass. At order 10, each step costs 10 forward passes. You need fewer steps, but the total forward passes may not decrease. Also, very high-order methods can be numerically unstable. In practice, order 2-3 hits the sweet spot where fewer total evaluations produce good results.)
- "If you retrained the same U-Net with a different noise schedule (say, a sigmoid instead of cosine), would your DDIM sampler still work?" (Answer: yes, but you would need to recompute the alpha_bar values. DDIM uses alpha_bar_t at each scheduled timestep. The formula is the same; only the numerical values of alpha_bar change. The sampler works with any noise schedule because it only needs the alpha_bar values, not the specific schedule shape.)

**10. Practice -- Notebook exercises (Colab)**
- **Exercise design rationale:** This is a STRETCH lesson. The exercises should build understanding of the sampler mechanisms, not require implementing them from scratch. Use diffusers schedulers and focus on comparing behavior, inspecting intermediate states, and verifying the "same model, different sampler" insight.
- **Exercise sequence (mostly independent):**
  1. **(Guided) Same model, different sampler:** Load a pre-trained diffusion model (or Stable Diffusion). Generate an image with DDPMScheduler at 1000 steps. Swap to DDIMScheduler and generate with the same seed at 50 steps. Swap to DPMSolverMultistepScheduler at 20 steps. Compare the three images. Time each generation. Predict-before-run: "Will the 20-step DPM-Solver image look significantly worse than the 1000-step DDPM image?"
  2. **(Guided) DDIM determinism:** Generate 3 images with DDIM at the same seed. Verify they are identical (pixel-level comparison). Generate 3 images with DDPM at the same seed. They should differ (stochastic noise injection). Predict-before-run: "Will DDPM with the same seed produce the same image?"
  3. **(Supported) Step count exploration:** Generate images at 5, 10, 20, 50, 100, 200 steps with DPM-Solver. Display in a grid. At what step count does quality become acceptable? At what step count does increasing steps stop improving quality? Plot generation time vs step count.
  4. **(Independent) Predict the intermediate:** Using DDIM at 10 steps, extract and visualize the intermediate x_t at each of the 10 steps (by VAE-decoding the intermediate latent). Observe the coarse-to-fine progression. Compare to DDPM at 10 steps (which should look much worse). Reflection: "Why does DDIM at 10 steps produce recognizable intermediates while DDPM at 10 steps produces garbage?"
- **Solutions should emphasize:** The "same model" insight (weights are identical across all experiments), the quality-vs-steps nonlinearity, and the connection between the intermediates and the trajectory perspective.

**11. Summarize**
- "The model predicts noise. The sampler decides what to do with that prediction."
  - **DDPM:** take a tiny step to the adjacent timestep, add fresh noise. 1000 steps, stochastic.
  - **DDIM:** predict x_0 from the noise prediction, leap to a distant timestep using the closed-form formula. 50 steps, deterministic (or tunable).
  - **DPM-Solver:** follow the trajectory using multiple model evaluations to read the road ahead. 20 steps, current standard.
- "No retraining. Same weights. Different walkers."
- Mental model echo: "The model defines where to go. The sampler defines how to get there."

**12. Next step**
- "You now understand every piece of the Stable Diffusion pipeline AND how to choose a sampler. Lesson 17 puts it all together: you will use the diffusers library to generate images, and you will know what every parameter means because you built each concept from scratch."
- Preview: "Every API parameter maps to a concept you learned. `guidance_scale` is CFG from Lesson 13. `num_inference_steps` is the sampler step count from this lesson. `scheduler` is your sampler choice. You will not be following a tutorial -- you will be driving a machine you built."

---

## Widget Assessment

**Widget needed:** No dedicated interactive widget.

**Rationale:** This lesson's core concepts are best served by static diagrams and notebook exercises rather than an interactive widget.

The trajectory diagram (DDPM vs DDIM vs DPM-Solver paths) is the key visual, but it is a conceptual diagram showing three paths through a 2D projected space, not an interactive manipulation. A draggable slider would not add pedagogical value because the insight is structural ("these are different paths through the same space") not parametric ("what happens when I change X"). The DDIM step diagram (predict x_0, jump to t_next) is similarly a static two-hop visual.

The notebook exercises provide the hands-on component: students compare real outputs from different samplers, observe the quality-vs-steps curve, and inspect intermediate states. This is more pedagogically valuable than a widget simulation because the student works with actual Stable Diffusion outputs.

The lesson uses:
- Trajectory diagram (visual): three paths from noise to data, DDPM (dense/jittery), DDIM (sparse/smooth), DPM-Solver (sparse/curved arcs)
- DDIM step diagram (visual): predict x_0, jump to t_next via closed-form
- Side-by-side formulas: DDPM step vs DDIM step, shared terms highlighted
- ComparisonRow: sampler characteristics (steps, determinism, quality, speed)
- GradientCards: sampler recommendations by use case
- ConceptBlock: "alpha_bar vs alpha" distinction, ODE bridge from gradient descent
- WarningBlock: "more steps does not always help," "no retraining needed"
- InsightBlock: score-based context (brief), "predict-and-leap" mechanism
- Colab notebook with 4 exercises (Guided -> Guided -> Supported -> Independent)

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (zero gaps)
- [x] No multi-concept jumps in widgets/exercises (notebook exercises use diffusers schedulers, not raw implementations)
- [x] All gaps have explicit resolution plans (no gaps to resolve)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (DDPM slowness pain resolved by showing same model, different sampler)
- [x] At least 3 modalities planned for the core concept, each with rationale (visual, symbolic, concrete, verbal/analogy, intuitive -- 5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (4 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 3 new concepts (at the upper limit, mitigated by leveraging APPLIED/DEVELOPED prerequisites)
- [x] Every new concept connected to at least one existing concept (DDIM to closed-form formula, ODE to gradient descent, DPM-Solver to Euler)
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: MAJOR REVISION

Critical finding in the notebook (Exercise 2 teaches the wrong lesson about DDPM stochasticity) must be fixed before the lesson is usable. The three improvement findings address a missing visual modality, a missing second concrete example, and a notebook scaffolding issue that would confuse students.

### Findings

#### [CRITICAL] -- Notebook Exercise 2 DDPM stochasticity claim is factually wrong

**Location:** Notebook cell-10 (DDPM stochastic comparison) and cell-11 (explanation markdown)
**Issue:** The exercise creates a fresh `torch.Generator(device=device).manual_seed(seed)` before EACH of the 3 DDPM runs. In `diffusers`, the generator controls ALL random draws during the pipeline call, including the initial latent z_T AND every per-step noise z. Seeding the generator identically before each run means the sequence of random draws is identical across runs. The 3 DDPM images will be pixel-identical (or near-identical due to floating point nondeterminism), not different as the exercise claims.

The exercise then states: "DDPM: stochastic. Same seed sets the same z_T, but fresh noise is injected at every step... The generator produces different z values at each step across runs, so the paths diverge." This is incorrect when the generator is seeded identically. The generator does NOT produce different z values -- it produces the SAME z values because the internal state is reset to the same seed before each run.

The student would run this exercise, see 3 identical (or near-identical) DDPM images, and be confused by the explanation claiming they should differ. This directly undermines the lesson's teaching about deterministic vs stochastic sampling.

**Student impact:** The student sees identical DDPM images but reads an explanation claiming they should differ. They either (a) lose trust in the lesson's accuracy, (b) form a wrong mental model about what "stochastic" means, or (c) spend time debugging a non-existent problem.

**Suggested fix:** To properly demonstrate DDPM stochasticity, the exercise needs to NOT re-seed the generator between runs. For example:
- Create ONE generator with a seed, generate the first image (consuming random state for z_T + all per-step noise draws), then generate a second and third image WITHOUT re-seeding. The later runs draw different random numbers because the generator state has advanced.
- Or: seed only the initial z_T identically but use a different generator (or no generator) for the per-step noise.
- Or: the simplest fix: use a different seed for each DDPM run to show that DDPM with different noise paths produces different images, while keeping the same seed for DDIM to show determinism.

Also update the explanation text to accurately describe why DDPM is stochastic (different random noise per step IN A SINGLE RUN creates path diversity BETWEEN different seeds or between runs with different generator states) rather than claiming the same seed produces different outputs.

---

#### [IMPROVEMENT] -- Missing visual modality: trajectory diagram and DDIM step diagram are text-only

**Location:** Lesson sections "DDPM vs DDIM / Euler vs DPM-Solver" trajectory cards (lines 684-713) and DDIM PhaseCards (lines 332-354)
**Issue:** The planning document explicitly calls for two visual diagrams: (1) a trajectory diagram showing three paths from noise to data in a 2D projected space (DDPM dense/jittery, DDIM sparse/smooth, DPM-Solver sparse/curved), and (2) a two-hop DDIM step diagram showing predict-backward-to-x_0 then jump-forward-to-t_next. Neither was built as an actual visual. Instead, the trajectory concept is conveyed via three GradientCards with text descriptions ("drunkard's walk," "direct route," "GPS route"), and the DDIM step is shown only as formulas in PhaseCards.

The visual modality is listed as one of the 5 planned modalities, and it is the one the plan describes as "the core spatial/geometric intuition for this lesson." Text descriptions of spatial concepts are really the verbal modality dressed up in cards. For a STRETCH lesson with three new concepts, having the core geometric intuition delivered only verbally is a significant weakness.

**Student impact:** The student understands the DDIM mechanism symbolically (formula) and verbally (analogy) but lacks the geometric/spatial picture that would make the trajectory concept "click." The ODE section talks about vector fields and trajectories, but the student never SEES one. This makes the ODE section more abstract than it needs to be.

**Suggested fix:** Add either (a) a static SVG diagram showing three stylized trajectories in a 2D space (noise at top, data at bottom; DDPM = many small dots with jitter, DDIM = fewer dots on smooth curve, DPM-Solver = fewest dots with curved arcs), or (b) a simple two-panel DDIM step diagram (panel 1: x_t -> predict x_hat_0, panel 2: x_hat_0 -> compute x_{t_next}). Even a simple CSS/HTML visualization with dots and lines would be more effective than text descriptions of spatial concepts. The trajectory diagram should go in the ODE section (around line 680) where the spatial interpretation is introduced. The DDIM step diagram could replace or supplement the PhaseCards.

---

#### [IMPROVEMENT] -- Missing second concrete DDIM example (t=200 -> t=0)

**Location:** Lesson section "One DDIM Step: t = 1000 to t = 800" (lines 414-464)
**Issue:** The planning document specifies two concrete DDIM traced examples: (1) t=1000 -> t=800 (a large jump at high noise), and (2) t=200 -> t=0 (the final jump at low noise). Only the first was built. The second example was planned specifically to show that at lower noise levels, the x_hat_0 prediction is much better and the leap is more accurate -- demonstrating that the mechanism generalizes across noise levels and explaining why the iterative refinement converges.

**Student impact:** The student only sees DDIM working at the hardest case (near-pure noise). They do not see what happens when the prediction is good (low noise) -- which is the case where DDIM's efficiency really shines and where the "refinement converges" intuition becomes concrete. The TipBlock "Why Not Jump Straight to t=0?" partially addresses this gap, but verbally rather than with a worked example.

**Suggested fix:** Add a brief second example after the t=1000 -> t=800 walkthrough. It can be shorter (the mechanism is the same), focusing on the key difference: alpha_bar_200 is much higher (~0.85), so the x_hat_0 prediction is much closer to the true x_0, and the leap to t=0 produces a nearly clean image. This confirms the pattern generalizes and shows why the final steps are smaller/easier.

---

#### [IMPROVEMENT] -- Notebook Exercise 3 scaffolding is inconsistent with "Supported" label

**Location:** Notebook cell-13 (Exercise 3 code) and cell-14 (display/plot code)
**Issue:** Exercise 3 is labeled "Supported" but has two scaffolding problems:

(1) In cell-13, the loop body is mostly written for the student. The scheduler setup, generator creation, and timing are all provided. The student only needs to replace `result = None` with a `pipe()` call -- but the exact function signature is shown in the hint comment directly above (`result = pipe(prompt, num_inference_steps=n, guidance_scale=guidance_scale, generator=generator)`). The student literally copies the line from the hint. This is closer to Guided (run and observe) than Supported (write with scaffolding).

(2) In cell-14, the plotting code has `# TODO: Plot step_counts vs step_timings` but the actual solution line is already present in the cell: `plt.plot(step_counts, step_timings, 'o-', color='cyan', linewidth=2, markersize=8)  # TODO (solution shown)`. The comment "TODO (solution shown)" is confusing -- is this a TODO or not? The student sees the code already written with a comment saying it is the solution.

**Student impact:** For (1), the exercise provides so much scaffolding that the student is not genuinely writing code -- they are copying a line from a comment. For (2), the student is confused by "TODO" comments on lines that already contain the solution code.

**Suggested fix:** For (1), remove the hint that shows the exact `pipe()` call. Keep the high-level comment ("Loop over step_counts, set scheduler, generate, time it") but let the student figure out the pipe() call from Exercise 1's pattern. For (2), either make the plotting a genuine TODO (replace the plot line with `# YOUR CODE HERE` and put the solution only in the `<details>` block) or remove the TODO comment and acknowledge it is provided code. Do not mix "TODO" labels with already-present solution code.

---

#### [POLISH] -- Spaced em dashes in concrete example step labels

**Location:** Lines 434 and 442 of the lesson component
**Issue:** The labels "Step 1 &mdash; Predict the destination:" and "Step 2 &mdash; Leap to t=800:" render with spaces around the em dash ("Step 1 -- Predict"). The Writing Style Rule requires no spaces around em dashes: `word--word` not `word -- word`.

**Student impact:** Minor visual inconsistency with the rest of the lesson (all other em dashes in the lesson are correctly unspaced).

**Suggested fix:** Change to "Step 1&mdash;Predict the destination:" and "Step 2&mdash;Leap to t=800:" (remove the spaces around `&mdash;`). Alternatively, use a colon or period instead of an em dash for the label separator.

---

#### [POLISH] -- Notebook DDPM at 50 steps in Exercise 2 may produce poor-quality images

**Location:** Notebook cell-10 (DDPM part of Exercise 2)
**Issue:** Exercise 2 uses DDPM at 50 steps for the stochastic comparison. The code comment acknowledges: "DDPM was designed for 1000 steps, but we use 50 here for speed. Quality will be lower, but the stochasticity point still holds." While the stochasticity demonstration point is valid (once the critical finding above is fixed), the poor DDPM quality at 50 steps may confuse the student into attributing the quality difference to stochasticity rather than to using DDPM outside its designed step range.

**Student impact:** Minor. The student might conflate "DDPM at 50 steps looks worse" with "stochastic = worse quality." The lesson itself explains that DDPM needs 1000 steps, so this is more of a potential confusion than a real problem.

**Suggested fix:** No change strictly necessary. If addressing, consider adding a brief note that the quality difference is about step count (DDPM at 50 is outside its designed range), not about stochastic vs deterministic per se.

### Review Notes

**What works well:**
- The narrative arc is strong. The problem-before-solution structure is compelling, and the emotional setup (student felt the 1000-step pain, was promised a resolution) pays off effectively.
- The gradient descent -> Euler's method bridge is the strongest pedagogical move in the lesson. It takes a potentially intimidating concept (ODEs) and grounds it in the student's most practiced algorithm.
- The negative example (naive step-skipping) is perfectly placed and effectively motivates why DDIM needs a different formula.
- The lesson stays within its stated scope boundaries. The INTRODUCED-depth treatments of ODE and DPM-Solver are appropriately brief.
- The connections to prior concepts are frequent and specific (naming source lessons, callback to formulas the student derived).
- The summary and mental model echo are clean and memorable.

**Systemic pattern:**
- The visual modality gap is a recurring pattern worth watching. The plan called for diagrams but the build used text cards. For STRETCH lessons, genuine visuals (even simple SVGs) are more important because the student is processing more new information and benefits more from multi-modal encoding. Future builds should prioritize the planned visual modality even if it requires more implementation effort.

**Notebook quality:**
- The notebook is generally well-structured. Exercise 1 is an excellent Guided exercise. Exercise 4 is a well-designed Independent exercise with good scaffolding hints and a thorough solution. The critical issue in Exercise 2 is a factual error that must be fixed. The scaffolding issue in Exercise 3 is a design weakness but not a showstopper.

---

## Review -- 2026-02-14 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All findings from iteration 1 have been correctly addressed. The CRITICAL notebook fix (Exercise 2 DDPM stochasticity) is factually correct. The three IMPROVEMENT fixes (SVG diagrams, second concrete DDIM example, Exercise 3 scaffolding) are well-implemented. No new issues were introduced by the fixes. Two minor polish items noted below.

### Iteration 1 Fix Verification

**[CRITICAL] Exercise 2 DDPM stochasticity -- FIXED CORRECTLY**
The generator is now seeded ONCE before the three DDPM runs (cell-10), and the code comments explicitly explain that each pipeline call advances the generator state. The explanation markdown (cell-11) accurately describes why the three images differ: "the generator was seeded once and its internal state advanced after each run." The note about DDPM quality at 50 steps is present and clear. The DDIM part (cell-9) correctly re-seeds before each run to demonstrate determinism. The exercise now teaches the correct lesson about stochastic vs deterministic sampling.

**[IMPROVEMENT] Missing visual modality -- FIXED CORRECTLY**
Two SVG diagrams added: (1) DDIM two-hop step diagram (lines 370-421) showing x_t -> predict x̂₀ -> leap to x_{t_next} with a timeline axis, three color-coded nodes, dashed blue predict arc, solid violet leap arc, and a caption. (2) Trajectory diagram (lines 789-906) showing three paths from noise to data: DDPM as a polyline with many dots and jitter lines (amber), DDIM as a smooth bezier with 6 dots (blue), DPM-Solver as a bezier with 3 dots (emerald). Legend at bottom. Both have descriptive aria-labels. Marker IDs are unique across the two SVGs (no collision). The diagrams are well-structured and should render correctly.

**[IMPROVEMENT] Missing second concrete DDIM example -- FIXED CORRECTLY**
Second example at t=200 -> t=0 added (lines 522-577). Uses alpha_bar_200 ≈ 0.85 (reasonable for cosine schedule) and alpha_bar_0 = 1.0 (correct by definition). The text accurately conveys that the DDIM formula with ᾱ_0 = 1.0 returns x̂₀ directly (sqrt(1.0) * x̂₀ + sqrt(0) * direction = x̂₀). The comparison paragraph between the two examples ("at t=1000, the prediction was a crude outline... at t=200, it captures fine detail") effectively demonstrates that the mechanism generalizes across noise levels and explains why iterative refinement converges. The accompanying InsightBlock "Refinement Converges" reinforces this.

**[IMPROVEMENT] Exercise 3 scaffolding -- FIXED CORRECTLY**
Cell-13 now has a high-level hint ("Look at Exercise 1 for the pattern") without showing the exact pipe() call signature. The student must figure out the call from Exercise 1's pattern and replace `result = None`. Cell-14 has `# YOUR CODE HERE` for the plotting code without pre-filled solution lines. The solution is only in the `<details>` block (cell-15), which includes reasoning before code, the generation loop fix, the plotting code, and observation notes. This correctly matches the "Supported" scaffolding level.

**[POLISH] Spaced em dashes in concrete example -- FIXED**
Lines 491 and 498 now use unspaced em dashes: "Step 1&mdash;Predict" and "Step 2&mdash;Leap". Consistent with all other em dashes in the lesson.

**[POLISH] DDPM at 50 steps quality note -- ADDRESSED**
Cell-10 includes a code comment and cell-11 includes a dedicated note explaining that the quality difference is about step count (DDPM outside its designed range), not about stochastic vs deterministic.

### Findings

#### [POLISH] -- Concrete example alpha-bar values use inconsistent notation (≈ vs =)

**Location:** Lesson section "One DDIM Step: t = 1000 to t = 800" (lines 481-487 and 493-494)
**Issue:** The bullet list introduces values with "approximately" (ᾱ₁₀₀₀ ≈ 0.0001), but the subsequent text uses exact equals ("Using the rearranged formula with ᾱ₁₀₀₀ = 0.0001"). This is a very minor inconsistency. The same pattern appears in the second example (lines 536-537 use ≈, line 547 uses =).
**Student impact:** Negligible. The student understands these are approximate values from a schedule. No confusion expected.
**Suggested fix:** Use ≈ consistently in both places, or use = consistently and remove ≈ from the bullet list. Either convention works.

---

#### [POLISH] -- Notebook Exercise 2 DDPM section title says "stochastic" but the demonstration mechanism is "different generator state"

**Location:** Notebook cell-8 (Exercise 2 title) and cell-10 (DDPM code)
**Issue:** The exercise title says "DDIM Determinism vs DDPM Stochasticity" and the explanation attributes the different images to DDPM's stochastic noise injection. While this is correct at a conceptual level (DDPM's stochasticity causes the generator to consume more random state per run), the actual demonstration mechanism is "generator state advancement between runs" rather than "same z_T with different per-step noise." A purist might note that the three DDPM runs also have different z_T (not just different per-step noise), so the demonstration conflates two sources of difference. However, the explanation text (cell-11) is honest about this: "Each pipeline call consumed dozens of random draws (one per step for the noise injection, plus one for z_T)."
**Student impact:** Negligible. The explanation is transparent about the mechanism. The conceptual point (DDPM produces diverse outputs, DDIM does not) is correctly demonstrated. A student who reads carefully will understand that both z_T and per-step noise differ across runs.
**Suggested fix:** No change strictly necessary. The explanation is already honest about the mechanism. If addressing, could add one sentence: "To isolate per-step stochasticity alone, you would need to inject the same z_T but different per-step noise -- an advanced exercise beyond our scope here."

### Review Notes

**What works well:**
- The DDIM two-hop SVG diagram is clear and well-labeled. The predict-then-leap mechanism is visually obvious: dashed blue arc backward to x̂₀, solid violet arc forward to x_{t_next}. This is a significant improvement over text-only PhaseCards.
- The trajectory diagram effectively conveys the "same destination, different paths" insight. The three visual styles (dense jittery polyline, smooth curve, minimal curve) immediately communicate the speed/precision tradeoff without requiring any text.
- The second concrete DDIM example (t=200 -> t=0) completes the pedagogical arc. The contrast between the two examples (crude prediction at high noise vs accurate prediction at low noise) directly builds the "refinement converges" intuition.
- The notebook Exercise 2 fix correctly demonstrates the deterministic/stochastic distinction. The approach (seed once, let state advance) is pragmatic and the explanation is transparent about the mechanism.
- Exercise 3 now genuinely requires the student to write code from the Exercise 1 pattern, matching the Supported scaffolding level.

**Overall assessment:**
This is a well-built STRETCH lesson. The three new concepts (DDIM mechanism, ODE perspective, higher-order solvers) are sequenced and scaffolded effectively. The gradient descent -> Euler's method bridge is the pedagogical highlight. The SVG diagrams add genuine visual modality. The notebook exercises provide appropriate hands-on practice with proper scaffolding progression. The lesson stays within its stated scope boundaries and correctly uses prior concepts at appropriate depths. Ready to ship.
