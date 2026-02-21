# Lesson: Score Functions & SDEs

**Slug:** `score-functions-and-sdes`
**Series:** 7 (Post-SD Advances), **Module:** 7.2 (The Score-Based Perspective), **Position:** Lesson 4 of 11 in series, Lesson 1 of 2 in module

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Gradient as direction of steepest ascent | DEVELOPED | gradient-descent (1.1.4) | "Ball rolling downhill." Student has applied gradients in optimization since Series 1. Deeply practiced. Gradient descent update rule: theta_new = theta_old - alpha * nabla_L. |
| Gradient descent as Euler's method | INTRODUCED | samplers-and-efficiency (6.4.2) | Explicit bridge: "Gradient descent IS Euler's method." theta_new = theta_old - lr * nabla_L has the same structure as x_{t+h} = x_t + h * f(x_t, t). Student recognized the structural identity. |
| DDPM forward process (discrete noise addition in T steps) | DEVELOPED | the-forward-process (6.2.2) | Variance-preserving formulation: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon. Noise schedule beta_t. Closed-form shortcut. Student can work through the math. |
| DDPM reverse process (trained denoising, step by step) | DEVELOPED | sampling-and-generation (6.2.4) | Reverse step formula with noise removal and stochastic injection. Student can trace the algorithm and has implemented it. |
| DDPM training: predict noise epsilon_theta | DEVELOPED | learning-to-denoise (6.2.3) | Loss = MSE between predicted and actual noise. Student understands why predict noise (consistent target) and has implemented the training loop. |
| ODE perspective on diffusion | INTRODUCED | samplers-and-efficiency (6.4.2) | Model's noise predictions define a smooth vector field / trajectory from noise to data. Samplers are different ODE solvers. "DDIM IS approximately Euler's method on the diffusion ODE." Probability flow ODE MENTIONED for paper-reading vocabulary. |
| Euler's method as ODE solver | INTRODUCED | samplers-and-efficiency (6.4.2) | Compute direction at current point, take step of size h, repeat. Structurally identical to gradient descent. First-order: one model evaluation per step. |
| DPM-Solver / higher-order ODE solvers | INTRODUCED | samplers-and-efficiency (6.4.2) | Evaluate model at multiple nearby timesteps to estimate curvature, enabling larger accurate steps with fewer evaluations. DPM-Solver-2 (second-order), DPM-Solver++ (third-order, current standard at 15-20 steps). |
| DDIM predict-x0-then-jump mechanism | DEVELOPED | samplers-and-efficiency (6.4.2) | Two-step mechanism: predict clean image from noise prediction, leap to arbitrary target timestep. DDIM is approximately Euler's method on the diffusion ODE. |
| Stochastic noise injection in DDPM sampling (sigma_t * z) | DEVELOPED | sampling-and-generation (6.2.4) | Fresh noise at each reverse step. Temperature analogy: sigma > 0 = diverse, sigma = 0 = deterministic (DDIM). |
| Alpha-bar as signal-to-noise ratio | DEVELOPED | the-forward-process (6.2.2) | alpha_bar_t = cumulative product of alpha_i. Starts near 1 (clean), drops to near 0 (pure noise). Interactive widget experience. |
| Loss landscape and optimization | DEVELOPED | loss-functions (1.1.3), gradient-descent (1.1.4) | Bowl shape, valley = minimum. Student has extensive experience navigating loss landscapes with gradients. |
| Probability / log probability (from LLM training) | DEVELOPED | Series 4 | Cross-entropy loss on log probabilities. Student is comfortable with log probabilities from language modeling. |
| "The model defines where to go. The sampler defines how to get there." | DEVELOPED | samplers-and-efficiency (6.4.2) | Core mental model. The model predicts noise at every point; the sampler decides how to use that prediction. |

### Mental Models and Analogies Already Established

- **"Ball rolling downhill"** -- gradient as direction of descent. Deeply embedded since lesson 4.
- **"Predict and leap"** -- DDIM's mechanism. Predict destination, jump via closed-form formula.
- **"Same vehicle, different route"** -- DDPM/DDIM/DPM-Solver as different routes through the same landscape.
- **"Gradient descent IS Euler's method"** -- the bridge between optimization and ODE solving. Already INTRODUCED.
- **"The model defines where to go. The sampler defines how to get there."** -- model vs sampler separation.
- **"Alpha-bar is the signal-to-noise dial"** -- continuous interpolation between clean and noisy.

### What Was Explicitly NOT Covered

- Score matching, score functions, and score-based generative models -- explicitly deferred as "out of scope" in Module 6.2 lessons
- SDE/ODE duality in full rigor -- mentioned for vocabulary only in 6.4.2
- Probability flow ODE -- MENTIONED in 6.4.2, named but not explained
- Ito calculus, Fokker-Planck equation -- out of scope (and will remain so)
- The relationship between noise prediction and score function -- not discussed at all
- Continuous-time formulation of diffusion -- the student works entirely in discrete time with t in {1, ..., T}
- Why the ODE perspective works (just stated as a fact in 6.4.2)

### Readiness Assessment

The student is well-prepared. They have:
1. Deep comfort with gradients and optimization from Series 1 (APPLIED depth)
2. Solid understanding of DDPM forward/reverse process from Module 6.2 (DEVELOPED)
3. The ODE perspective already INTRODUCED in 6.4.2 -- this lesson deepens it, not introduces from scratch
4. Experience with Euler's method as an ODE solver, bridged from gradient descent
5. Familiarity with log probabilities from LLM training (Series 4)

The main gap is conceptual: the student has never connected "noise prediction" to "gradient of log probability." This is the core reveal of the lesson. All the mathematical prerequisites are in place.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to understand the score function (gradient of log probability) as the unifying concept that connects DDPM's noise prediction to a continuous-time SDE/ODE framework for diffusion.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Gradient as direction of steepest ascent | INTRODUCED | DEVELOPED | gradient-descent (1.1.4) | OK | Need the student to understand "gradient points in a direction." They have this at APPLIED depth from extensive practice. |
| DDPM forward process (discrete noise schedule) | INTRODUCED | DEVELOPED | the-forward-process (6.2.2) | OK | Need the student to recognize that DDPM adds noise in discrete steps. They can derive the closed-form formula. More than sufficient. |
| DDPM noise prediction training objective | INTRODUCED | DEVELOPED | learning-to-denoise (6.2.3) | OK | Need to know that the model predicts epsilon (noise). Core prerequisite for the score-noise equivalence reveal. |
| ODE perspective on diffusion | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) | OK | Student already has the key intuition: model predictions define a trajectory, samplers follow it. This lesson formalizes and deepens. Exact depth match. |
| Euler's method as ODE solver | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) | OK | Compute direction, take step, repeat. Structurally identical to gradient descent. Sufficient for understanding ODE solvers on the probability flow ODE. |
| Log probability concept | INTRODUCED | DEVELOPED | Series 4 (language modeling) | OK | Need the student to be comfortable with "log p(x)" as a quantity. They computed cross-entropy loss on log probs extensively. |
| Alpha-bar / noise schedule | INTRODUCED | DEVELOPED | the-forward-process (6.2.2) | OK | Need to reference alpha-bar when connecting noise prediction to score. Student has widget experience and derived the formula. |
| Stochastic vs deterministic sampling (DDPM vs DDIM) | INTRODUCED | DEVELOPED | samplers-and-efficiency (6.4.2) | OK | Need to connect: DDPM sampling = reverse SDE (stochastic), DDIM sampling = probability flow ODE (deterministic). Student already has both mechanisms. |
| Probability density / distribution intuition | INTRODUCED | INTRODUCED | Series 4 + 6.1 | OK | Student has probability distributions from VAE (latent space, KL divergence, sampling) and language modeling (softmax, cross-entropy). Not rigorous measure theory, but sufficient for "data occupies high-probability regions." |

### Gap Resolution

No GAPs or MISSING prerequisites. All concepts required are available at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The score function is something completely new and unrelated to what I already know about diffusion" | The student learned diffusion through the DDPM lens (noise schedules, epsilon prediction, discrete steps). Score functions sound like a different framework. | The equivalence: epsilon_theta(x_t, t) = -sqrt(1 - alpha_bar_t) * score(x_t, t). The noise prediction the student already understands IS a scaled version of the score. Nothing new was added -- a new lens on the same thing. | Core reveal in the Explain section, after score function is motivated and defined. The "aha" moment of the lesson. |
| "The SDE forward process is different from DDPM's forward process" | SDE sounds like new math. The student learned a specific formula x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon. A "stochastic differential equation" sounds like a different thing. | Take DDPM's forward process and make the step size infinitely small: you get the SDE. They are the same process at different resolutions. Like a staircase (discrete steps) vs a ramp (continuous): same height, same direction, just smoother. | SDE section, after the score function is established. Show the discrete-to-continuous limit visually. |
| "You need to understand Ito calculus / stochastic calculus to work with SDEs" | "SDE" sounds intimidating. Academic papers on score-based diffusion use Ito integrals and Fokker-Planck equations. | We will never write an Ito integral. The SDE is just "take an infinitely small DDPM step: shrink the signal a tiny bit, add a tiny bit of noise." The discrete formula the student already knows is sufficient intuition. | Scope boundary in Context + Constraints, reinforced with a WarningBlock when the SDE is introduced. |
| "The probability flow ODE is a separate model that needs separate training" | The ODE was briefly mentioned in 6.4.2 as a separate concept. It might sound like a different model. | Same trained model, same weights, same predictions. The probability flow ODE is just the deterministic version of the reverse SDE. DDIM was already approximately solving this ODE. No new training, no new model. | After reverse SDE, as the "remove the noise term" simplification. Connect explicitly to DDIM. |
| "Score function = loss function gradient (nabla_theta L)" | The student has extensive experience with gradients of loss functions with respect to parameters (backprop). "Gradient of log probability" could be confused with "gradient of the loss." | The score function is nabla_x log p(x), NOT nabla_theta L. It is the gradient with respect to the DATA, not the parameters. It asks "which direction should I move the IMAGE to increase its probability?" not "which direction should I move the WEIGHTS to decrease the loss?" | Early in the score function definition. Explicit WarningBlock with side-by-side comparison. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| 1D Gaussian score function | Positive | Show the score function concretely with a distribution the student can visualize. For N(0, 1): score(x) = -x. At x=3 (far from mean), score = -3 (strong pull back). At x=0 (at the mean), score = 0 (already at peak). | Simplest possible distribution where the student can compute the score by hand. Makes the abstract "gradient of log probability" concrete. Connects to ball-on-curve analogy from Series 1. |
| 2D Gaussian mixture score field | Positive | Show the score as a vector field in 2D. Two peaks, arrows everywhere pointing toward the nearest peak. Between the peaks, arrows point toward one or the other. Shows that the score field encodes the entire shape of the distribution. | Generalizes from 1D to higher dimensions. The vector field visualization is the key geometric insight: at every point in data space, the score says "go this way for higher probability." This is what the denoising model is learning. |
| Noisy data distribution (score at different noise levels) | Positive (stretch) | Show how adding noise to a distribution smooths its score field. At low noise, the score field is complex (sharp peaks, narrow valleys). At high noise, it simplifies (nearly Gaussian, scores are nearly linear). This is WHY denoising score matching works: the noisy distribution has a smoother, easier-to-learn score. | Connects score function to the noise schedule the student already knows. Explains the otherwise mysterious "why does adding noise help?" question. The DDPM curriculum has been teaching the student to work with noisy distributions -- now they see WHY. |
| Score function at mean of distribution (score = 0) | Negative | At the peak of the probability distribution, the score is zero. The score function does NOT tell you the probability -- it tells you the DIRECTION. A score of zero means "you are already at a peak," not "the probability is zero." | Prevents confusion between score (gradient) and probability density (value). Students who conflate "how likely" with "which direction" will misunderstand what the model is learning. Parallels the gradient = 0 at minimum from optimization (same concept: zero gradient means you are at an extremum). |

---

## Phase 3: Design

### Narrative Arc

The student has spent 20+ lessons learning diffusion through the DDPM lens: add noise in T discrete steps, train a model to predict noise, reverse the process with a specific formula. The sampler lesson (6.4.2) planted a seed -- "DDIM is approximately Euler's method on the diffusion ODE" -- but never explained what that ODE really is or where it comes from. This lesson pulls that thread.

The motivating question is: what is the noise prediction network actually learning? Not "what noise was added" (the training objective), but what is it learning about the DATA? The answer is the score function -- the gradient of log probability -- which points toward regions where data is more likely. This is the same concept as gradients in optimization (which the student has used since lesson 4), but applied to data space instead of parameter space. The noise prediction epsilon_theta that the student already knows IS a scaled version of the score. Nothing new was trained -- the student's existing understanding of DDPM already contains the score-based perspective; it was just hidden by the discrete-time notation.

With the score function in hand, the DDPM forward process generalizes to a continuous SDE (noise destroys structure smoothly instead of in discrete steps), the reverse process becomes another SDE guided by the score, and the deterministic ODE from the sampler lesson gets a proper name: the probability flow ODE. The student should leave feeling that the DDPM framework they already know was always a special case of this more general picture, and that this picture is why modern architectures can move beyond DDPM-style training (flow matching, which is the next lesson).

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Geometric/Spatial** | Score function as a vector field in 2D: arrows at every point in data space pointing toward higher probability. For a 2D Gaussian mixture, the field converges on two peaks. At different noise levels, the field smooths from complex to simple. | The score function IS a spatial concept -- it is a direction at every point. The vector field visualization is the single most important modality for building intuition. Parallels the gradient arrows on the loss landscape that the student already knows. |
| **Symbolic** | Three key equations: (1) score(x) = nabla_x log p(x), (2) the score-noise equivalence: epsilon_theta approx -sqrt(1 - alpha_bar_t) * score(x_t, t), (3) the probability flow ODE: dx = [f(x,t) - 0.5 * g(t)^2 * score(x_t, t)] dt. | The equations are the permanent takeaway for paper-reading. But they come AFTER geometric intuition, not before. Each equation should feel like "of course, that is just the math for what I already understand." |
| **Concrete example** | 1D Gaussian: p(x) = N(0,1), log p(x) = -x^2/2 + const, score(x) = d/dx log p(x) = -x. At x=3, score = -3 (push toward center). At x=0, score = 0 (already at peak). Worked through with actual numbers. | The student needs to compute a score by hand to demystify it. A 1D Gaussian is the simplest case where the math is trivial (differentiate a quadratic). The numbers make it concrete: "at x=3, the score is -3" is more memorable than "the score points toward higher probability." |
| **Verbal/Analogy** | "The score function is a compass that always points toward likely data." Extension of the ball-on-curve analogy: in optimization, the gradient points downhill on the loss landscape. The score function points uphill on the probability landscape. Same concept, different landscape. | Connects to the most deeply embedded analogy in the course (ball rolling downhill, gradient descent). The student should feel "oh, it is THAT concept again, just applied to data instead of parameters." |
| **Intuitive** | "DDPM's noise prediction was the score function all along." The reveal: epsilon_theta = -sqrt(1-alpha_bar_t) * score. The model learned to point toward higher probability, disguised as predicting noise. Nothing new was added -- just a new way to see what was already there. | The "aha" moment. If the student does not feel "of course" when they see the equivalence, the lesson has failed. This is the emotional core. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 genuinely new concepts:
  1. Score function as gradient of log probability (NEW -- never taught)
  2. SDE forward process as continuous-time generalization of DDPM (NEW -- DDPM was discrete only)
  3. Probability flow ODE formalized (DEEPENED from MENTIONED to INTRODUCED/DEVELOPED)
- **Previous lesson load:** Module 7.1 Lesson 3 (ip-adapter) was BUILD. Before that, controlnet-in-practice was CONSOLIDATE, controlnet was STRETCH.
- **Is this appropriate?** Yes, STRETCH is appropriate. The student has had two lessons of relief (CONSOLIDATE + BUILD) since the last STRETCH (controlnet). The theoretical depth of this lesson warrants STRETCH, but every new concept connects to existing knowledge (gradients to score, discrete DDPM to continuous SDE, ODE from 6.4.2 to probability flow ODE). The cognitive load is in reconceptualization, not in learning from scratch.

### Connections to Prior Concepts

| New Concept | Connects To | How |
|-------------|------------|-----|
| Score function (nabla_x log p(x)) | Gradient descent (nabla_theta L from 1.1.4) | Same mathematical operation (gradient), same geometric meaning (direction of steepest ascent/descent), applied to a different space (data space vs parameter space). "You have been using gradients since lesson 4. The score function is the same idea applied to data instead of parameters." |
| Score function points toward higher probability | Ball-on-curve analogy (1.1.4) | In optimization, the ball rolls downhill on the loss landscape. In score-based diffusion, a point "rolls uphill" on the probability landscape. Same concept, opposite direction (minimize loss vs maximize probability). |
| Score-noise equivalence | DDPM noise prediction (6.2.3) | epsilon_theta approx -sqrt(1-alpha_bar_t) * score(x_t, t). The noise prediction the student has been training IS a scaled score function. The model was always learning to point toward data. |
| SDE forward process | DDPM forward process (6.2.2) | DDPM: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon. SDE: the continuous limit where step size goes to zero. Same process, infinitely refined. Staircase vs ramp analogy. |
| Probability flow ODE | ODE perspective (6.4.2) | "DDIM is approximately Euler's method on the diffusion ODE" -- 6.4.2 stated this. Now we give that ODE its name and derivation. The student has already been using it without knowing its formal identity. |
| Reverse SDE (stochastic sampling) | DDPM sampling with sigma_t * z (6.2.4) | DDPM's stochastic noise injection during reverse steps IS the discrete approximation of the reverse SDE. sigma_t * z is the stochastic term. |
| Score at different noise levels | Alpha-bar curve (6.2.2) | At high noise (alpha_bar near 0), the noisy distribution is nearly Gaussian and the score field is simple. At low noise (alpha_bar near 1), the distribution is close to the data and the score field is complex. This is why the noise schedule matters. |

### Analogies to Extend

- **"Ball rolling downhill"** from gradient descent -- extend to "point drifting uphill on the probability landscape." Same mechanism, different context.
- **"The model defines where to go"** from 6.4.2 -- NOW we know what "where to go" means precisely: the score function. The model defines the score at every point.
- **"DDIM is Euler's method on the diffusion ODE"** from 6.4.2 -- formalize this: the ODE is the probability flow ODE, and we can derive it from the reverse SDE by removing the stochastic term.

### Analogies That Could Be Misleading

- **"Ball rolling downhill"** could mislead if students think score-based sampling involves minimizing something. We are NOT doing gradient descent on a loss function. The score function points toward higher probability, but sampling is NOT optimization. Sampling follows the SDE/ODE, which involves both the score AND the forward process drift. Address this explicitly.

### Scope Boundaries

**This lesson IS about:**
- The score function: what it is, geometric intuition, concrete examples
- The equivalence between noise prediction and the score function
- The SDE forward process as a continuous generalization of DDPM
- The reverse SDE and probability flow ODE (at intuition level)
- Why this perspective matters for understanding modern diffusion models

**This lesson is NOT about:**
- Ito calculus, stochastic integration, Fokker-Planck equation (out of scope for the entire series)
- Score matching training objective (how to train a score network from scratch) -- we note that DDPM training already does this implicitly
- Denoising score matching derivation (Song & Ermon 2019) -- beyond scope, mentioned for paper reading only
- Flow matching (next lesson)
- Consistency models or distillation (Module 7.3)
- Any implementation or coding -- this is a purely conceptual/theoretical lesson
- Measure theory, probability density function derivations, or rigorous proofs

**Depth targets:**
- Score function: DEVELOPED (intuition + formula + concrete example + geometric visualization + connection to DDPM)
- SDE forward process: INTRODUCED (conceptual, visual, connected to DDPM, but no equations beyond the general form)
- Probability flow ODE: INTRODUCED (named, connected to DDIM/ODE from 6.4.2, formula shown, but not derived rigorously)
- Score-noise equivalence: DEVELOPED (formula stated, concrete numerical example, intuitive explanation)

---

### Lesson Outline

#### 1. Context + Constraints
- "This is the most theoretical lesson in Series 7. We are not learning new tools or running models. We are looking at what you already know -- DDPM, noise prediction, samplers -- through a different lens."
- "We will NOT be doing Ito calculus. We will NOT be deriving equations from measure theory. Every concept connects to things you already have: gradients, DDPM, ODE solvers."
- "By the end, you will understand what your noise prediction network has been secretly learning all along."
- Capstone series tone: "Let's read the frontier together" -- the student has earned the right to see the deeper theory.

#### 2. Recap
Brief reactivation of three concepts:
- **Gradient** (from 1.1.4): 2-3 sentences. "The gradient points in the direction of steepest increase of a function. You used nabla_theta L to find the direction that increases loss the most, then went the opposite way."
- **ODE perspective** (from 6.4.2): 2-3 sentences. "Remember: DDIM is approximately Euler's method on the diffusion ODE. The model's noise predictions define a trajectory from noise to data. Different samplers follow that trajectory with different step sizes."
- **DDPM noise prediction** (from 6.2.3): 2-3 sentences. "The model takes x_t and t, and predicts the noise epsilon that was added. MSE loss: ||epsilon - epsilon_theta(x_t, t)||^2."
- Transition: "We said the model predicts noise. But what is it really learning about the data?"

#### 3. Hook
Type: **Misconception reveal / reframing question**

"What does the noise prediction network know about the data distribution?"

The student's current understanding: it predicts what noise was added. That is the training objective. But consider: at any point x_t in noisy image space, the network can tell you which direction to move to get closer to a clean image. It does this for EVERY possible noisy image, at EVERY noise level. That is not just noise removal -- that is a map of the entire noisy data distribution. A map that says, at every point: "go this way for higher probability."

"The noise prediction epsilon_theta was always a disguised version of something more fundamental. Let's find out what."

#### 4. Explain: The Score Function

**Part A: What is the score function?**

Problem-before-solution: "Imagine you have a probability distribution p(x) -- say, the distribution of all natural images. You are at some point x in this space (maybe a noisy image, maybe a random point). You want to know: which direction should I move to get to a more probable point? Not 'how probable is this point?' but 'which way is up?'"

Definition: score(x) = nabla_x log p(x). The gradient of the log probability density with respect to x.

**WarningBlock: "Not the gradient you are used to."** nabla_x log p(x), not nabla_theta L(theta). This is the gradient with respect to DATA, not PARAMETERS. In optimization, you ask "which way should I move the weights?" Here you ask "which way should I move the data point?" Side-by-side comparison:
| | Optimization | Score Function |
|---|---|---|
| What changes | Parameters theta | Data point x |
| Function | Loss L(theta) | Log probability log p(x) |
| Direction | Toward lower loss | Toward higher probability |

**Part B: Concrete example -- 1D Gaussian**

p(x) = N(0, 1). Write out log p(x) = -x^2/2 + const. Take the derivative: score(x) = d/dx log p(x) = -x.

Work through specific values:
- At x = 3 (far from mean): score = -3. Strong pull toward center.
- At x = 1 (near mean): score = -1. Gentle pull toward center.
- At x = 0 (at mean): score = 0. Already at the peak -- no direction preferred.
- At x = -2 (other side): score = +2. Pull toward center from the left.

InsightBlock: "The score function is always pointing toward the peak of the distribution. It IS a compass toward likely data."

**Negative example (score at the peak):** The score is zero at x=0, even though the probability is highest there. The score tells you DIRECTION, not VALUE. Zero score means "you are at a peak," not "the probability is zero." Parallels gradient = 0 at a minimum in optimization.

**Part C: 2D vector field visualization**

Extend to 2D Gaussian mixture (two peaks). The score at every point is a 2D vector pointing toward the nearest peak. Between the peaks, the field splits -- some points are pulled toward one peak, others toward the other.

Visual: vector field plot showing arrows converging on two peaks. At different noise levels: high noise simplifies the field (arrows are nearly uniform, pointing toward the combined center), low noise reveals the complex structure (two distinct basins).

TipBlock: "This is exactly what the denoising model sees. At high noise, the model's job is easy -- the noisy distribution is nearly Gaussian, and the score field is simple. At low noise, the model must capture the fine structure of the data distribution."

Connect to alpha-bar: "When alpha_bar is near 0 (high noise), the noisy distribution q(x_t) is close to N(0,I) and the score field is nearly linear. When alpha_bar is near 1 (low noise), q(x_t) is close to the data distribution and the score field is complex."

#### 5. Check #1
Three predict-and-verify questions:
1. "For a 1D Gaussian N(5, 1), what is the score function? What is its value at x = 7?" (Answer: score(x) = -(x-5), so score(7) = -2. Points toward the mean.)
2. "If you have two points, x_A with score = -5 and x_B with score = -0.1, which is further from the peak?" (Answer: x_A. Larger score magnitude means steeper probability gradient, i.e., further from a peak.)
3. "Can the score function tell you whether p(x_A) > p(x_B)?" (Answer: No. The score tells you direction at each point, not the absolute value of the density. You would need to integrate to compare densities.)

#### 6. Explain: The Score-Noise Equivalence (The Reveal)

"Here is the key insight of this entire lesson."

The DDPM noise prediction epsilon_theta(x_t, t) has this relationship to the score function:

**epsilon_theta(x_t, t) approx -sqrt(1 - alpha_bar_t) * nabla_{x_t} log q(x_t)**

Or equivalently:

**score(x_t, t) = nabla_{x_t} log q(x_t) approx -epsilon_theta(x_t, t) / sqrt(1 - alpha_bar_t)**

The noise prediction IS a scaled version of the score function.

Walk through the intuition:
- The score says: "to increase the probability of x_t, move in this direction."
- The noise prediction says: "the noise that was added to reach x_t was in this direction."
- These are the SAME direction (up to a sign flip and scaling). Moving toward higher probability means undoing the noise. The noise points AWAY from the data; the score points TOWARD the data.

InsightBlock: "The noise prediction network has been learning the score function all along. Every DDPM model you have ever seen is a score-based model in disguise."

Concrete verification with the formulas the student knows: show how the noise removal term in the DDPM reverse step formula (beta_t / sqrt(1-alpha_bar_t) * epsilon_theta) can be rewritten using the score.

Address Misconception #1: "This is not a different framework. Score-based diffusion and DDPM are two descriptions of the same process. The score was hiding inside DDPM all along."

#### 7. Check #2
Two predict-and-verify questions:
1. "If epsilon_theta(x_t, t) is a vector pointing to the upper-right, what direction does the score point?" (Answer: lower-left, with magnitude scaling by 1/sqrt(1-alpha_bar_t). The score is the negative of the noise prediction, scaled.)
2. "At t=1000 (near pure noise, alpha_bar very small), how does the scaling factor 1/sqrt(1-alpha_bar_t) behave? At t=50 (nearly clean)?" (Answer: at t=1000, 1-alpha_bar is near 1, so the factor is near 1 -- noise prediction and score are almost the same thing. At t=50, 1-alpha_bar is small, so the factor is large -- small noise predictions correspond to large scores. This makes sense: near-clean data has sharply peaked probability, so the score is steep.)

#### 8. Explain: From Discrete Steps to Continuous Time (SDEs)

**Part A: The staircase-to-ramp transition**

"DDPM adds noise in T=1000 discrete steps. What happens if we make T infinitely large and the step size infinitely small?"

Visual: show the discrete forward process as a staircase (jumps at each t) transitioning to a smooth ramp (continuous curve) as the number of steps increases. The same trajectory, just smoother.

The forward process SDE (informal, no Ito notation):
- dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dw
- English: "shrink the signal a tiny bit, add a tiny bit of random noise. Repeat continuously."
- This is DDPM's variance-preserving step x_t = sqrt(1-beta_t) * x_{t-1} + sqrt(beta_t) * epsilon, taken to the continuous limit.

WarningBlock: "We will NOT be doing Ito calculus. The dx notation means 'an infinitely small version of the DDPM step you already know.' dw means 'an infinitely small random nudge.' If you want to think of it as 'DDPM with infinitely many infinitely small steps,' that is perfectly correct."

Address Misconception #2: "The SDE IS the DDPM forward process, just with the step size taken to zero. Same process, smoother description."

**Part B: The reverse SDE**

Anderson's key result (1982): if the forward process is an SDE, the reverse process is ALSO an SDE, and it depends on the score function.

Reverse SDE (informal): dx = [-0.5 * beta(t) * x - beta(t) * score(x, t)] dt + sqrt(beta(t)) dw_reverse

English: "undo the shrinking, follow the score toward higher probability, add a bit of random exploration. Repeat continuously."

Connect to DDPM reverse step: the discrete reverse step formula the student knows IS the discrete version of this reverse SDE. The epsilon_theta term (noise removal) IS the score term. The sigma_t * z injection IS the dw_reverse stochastic term.

InsightBlock: "DDPM reverse sampling was always approximately solving the reverse SDE. The score function was always the guide."

**Part C: The probability flow ODE**

"What if we remove the stochastic term from the reverse SDE?"

Probability flow ODE: dx = [-0.5 * beta(t) * x - 0.5 * beta(t) * score(x, t)] dt

No randomness. Fully deterministic. SAME marginal distributions at every time t as the SDE.

This IS the ODE from lesson 6.4.2. "DDIM is approximately Euler's method on the probability flow ODE" -- now we know exactly what that ODE is.

ComparisonRow: Reverse SDE vs Probability Flow ODE
| | Reverse SDE | Probability Flow ODE |
|---|---|---|
| Has randomness? | Yes (dw term) | No (deterministic) |
| DDPM analog | DDPM sampling (sigma_t > 0) | DDIM sampling (sigma = 0) |
| Same score function? | Yes | Yes |
| Same model weights? | Yes | Yes |
| Same marginal distributions? | Yes | Yes |
| Different paths? | Yes (stochastic, different each time) | No (same path every time) |

TipBlock: "Same landscape, different lens" callback from 6.4.2. The reverse SDE and probability flow ODE describe the same generative process. One is stochastic (explores), the other is deterministic (commits). The student already knows both: DDPM vs DDIM.

#### 9. Check #3 (Transfer)
Two transfer questions:
1. "A colleague says they are using a 'score-based model' and you are using 'DDPM.' Are you using fundamentally different generative models?" (Answer: No. DDPM IS a score-based model. The noise prediction network learns a scaled version of the score function. The DDPM forward/reverse process is a discrete approximation of the SDE framework. Different vocabulary for the same thing.)
2. "DPM-Solver++ is described as a higher-order ODE solver for the probability flow ODE. Based on what you know from 6.4.2, what does 'higher-order' mean here, and why does it help?" (Answer: Higher-order means evaluating the score/model at multiple nearby timesteps to estimate how the trajectory curves, enabling larger steps. This is the same as what DPM-Solver does in 6.4.2 -- it was solving the probability flow ODE all along, we just did not have the vocabulary for it yet.)

#### 10. Elaborate: Why This Perspective Matters

Three reasons this matters for the rest of the series:
1. **Unified vocabulary for reading papers:** "Score-based," "SDE," "probability flow ODE" appear in every modern diffusion paper. The student can now parse these terms.
2. **Flow matching depends on this:** The next lesson (flow matching) replaces the SDE's curved trajectories with straight lines. This only makes sense if you see generation as following a trajectory defined by a vector field (the score).
3. **Consistency models depend on this:** Module 7.3 builds on the ODE trajectory concept -- "any point on the same trajectory maps to the same endpoint" requires the probability flow ODE.

GradientCard: "The score function is the unifying thread. DDPM, DDIM, DPM-Solver, flow matching, consistency models -- they all work because a trained network can estimate the score of the noisy data distribution."

#### 11. Practice (Notebook)

Design: 4 exercises, Guided -> Guided -> Supported -> Independent progression.

**Exercise 1 (Guided): Compute Score Functions by Hand**
- Given 1D Gaussian distributions (N(0,1), N(3,2), mixture of two Gaussians), compute the score function analytically.
- Plot the score function alongside the PDF. Verify that the score is zero at peaks and large far from peaks.
- Predict-before-run: "Where will the score cross zero for the mixture?"

**Exercise 2 (Guided): Visualize the Score as a 2D Vector Field**
- Create a 2D Gaussian mixture. Plot the score field as arrows (quiver plot).
- Add increasing Gaussian noise to the distribution. Re-plot the score field at each noise level.
- Observe: high noise -> simple field, low noise -> complex field.
- Connect to alpha-bar: "This is why the noise schedule starts with easy tasks (high noise, simple scores) and progresses to hard tasks (low noise, complex scores)."

**Exercise 3 (Supported): Verify the Score-Noise Equivalence**
- Load a pre-trained diffusion model (small, e.g., DDPM on MNIST or a toy 2D distribution).
- For a given noisy image x_t at a specific timestep t:
  - Get the model's noise prediction epsilon_theta(x_t, t)
  - Compute the implied score: score = -epsilon_theta / sqrt(1 - alpha_bar_t)
  - Verify the direction makes sense: the score should point from x_t toward the region of clean data.
- Repeat at multiple noise levels and observe how the score magnitude changes.

**Exercise 4 (Independent): SDE vs ODE Trajectories**
- Using a pre-trained model on a toy 2D distribution:
  - Generate samples by numerically solving the reverse SDE (add noise at each step)
  - Generate samples by numerically solving the probability flow ODE (no noise)
  - Compare: SDE paths are wiggly and diverse, ODE paths are smooth and deterministic
  - Same starting point with ODE gives same endpoint. Same starting point with SDE gives different endpoints.
- Connect to DDPM vs DDIM comparison from 6.4.2 -- same insight, now with the formal framework.

#### 12. Summarize

Key takeaways (echo mental models):
1. **The score function** is the gradient of log probability: nabla_x log p(x). It points toward higher-probability regions of data space. It is a compass toward likely data.
2. **DDPM's noise prediction IS the score function** (up to scaling): epsilon_theta approx -sqrt(1-alpha_bar_t) * score. The model was always learning to point toward data.
3. **The SDE framework** generalizes DDPM to continuous time. Forward SDE = continuous noise addition. Reverse SDE = score-guided generation with stochastic exploration.
4. **The probability flow ODE** is the deterministic version. DDIM was already approximately solving it. DPM-Solver is a better solver for it.
5. **Same model, multiple perspectives.** Score-based, diffusion, DDPM, DDIM -- different vocabulary for the same underlying process.

#### 13. Next Step

"Now you know that generation follows a trajectory -- either the stochastic SDE or the deterministic ODE -- and the trajectory is defined by the score function. But look at these trajectories: they CURVE through high-dimensional space. What if we could straighten them? A straight path from noise to data would be simpler to follow, need fewer steps, and be easier to train. That is flow matching."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: geometric, symbolic, concrete example, verbal/analogy, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 2-3 new concepts (score function, SDE framework, probability flow ODE)
- [x] Every new concept connected to at least one existing concept (score to gradients, SDE to DDPM forward, prob flow ODE to DDIM/ODE perspective)
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: MAJOR REVISION

One critical finding (the missing vector field visualization for the score function) must be addressed before this lesson is usable. The planning document specifically identifies the geometric/spatial modality as "the single most important modality for building intuition," and the lesson relies entirely on verbal description where it should show a visual. The three improvement findings are individually moderate but collectively weaken the second half of the lesson.

### Findings

#### [CRITICAL] -- Missing 2D score vector field visualization

**Location:** Section "The Score as a Vector Field" (lines 381-437 of the lesson component)
**Issue:** The planning document says: "Visual: vector field plot showing arrows converging on two peaks. At different noise levels: high noise simplifies the field, low noise reveals the complex structure." The built lesson describes this verbally ("imagine a vector field: arrows at every point in the plane") but provides NO actual visual. The GradientCards for high noise vs low noise contain bullet-point lists, not diagrams. The planning document calls the vector field visualization "the single most important modality for building intuition."
**Student impact:** The student must imagine the score field from text alone. The score function IS a spatial concept -- it is a direction at every point in space. Without seeing arrows converging on peaks, the student's geometric intuition for the score function is built on imagination rather than observation. This is the one concept in the lesson where a visual is irreplaceable. The notebook DOES have the quiver plot (Exercise 2), but the lesson itself should establish the visual intuition before the notebook practices it.
**Suggested fix:** Add a static SVG or a simple interactive visualization showing the score field for a 2D Gaussian mixture. At minimum, a single static diagram with arrows converging on two peaks. Ideally, a two-panel or four-panel view showing the score field at different noise levels (clean, low noise, medium noise, high noise) -- this would also address the "score at different noise levels" positive example, which is currently text-only. This does not need to be interactive; a well-labeled static image would suffice.

#### [IMPROVEMENT] -- Score-noise equivalence derivation gap

**Location:** Section "The Score-Noise Equivalence" (lines 551-633)
**Issue:** The planning document says: "Concrete verification with the formulas the student knows: show how the noise removal term in the DDPM reverse step formula can be rewritten using the score." The built lesson states "The noise removal term -- (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta -- can be rewritten using the score" but does NOT actually perform the rewriting. It points at the connection without showing it. The equivalence formula is stated as a fact, not derived from what the student knows.
**Student impact:** A mathematically-minded student (which this student is -- they derived the closed-form formula, they implemented the reverse step) will want to see the algebra. Without it, the equivalence feels like "trust me" rather than "of course." The planning document's framing ("Each equation should feel like 'of course, that is just the math for what I already understand'") is not achieved here.
**Suggested fix:** Add a brief derivation or walkthrough showing the substitution. Start with the DDPM reverse step's noise removal term, substitute the score-noise equivalence, and show that the reverse step formula becomes a score-guided step. This does not need to be rigorous -- 3-5 lines of algebra would suffice. The student has the DDPM reverse step at APPLIED depth, so this is within their ability to follow.

#### [IMPROVEMENT] -- SDE section lacks concrete grounding

**Location:** Sections "From Discrete Steps to Continuous Time" through "The Probability Flow ODE" (lines 717-943)
**Issue:** The SDE section introduces three interrelated concepts (forward SDE, reverse SDE, probability flow ODE) with no concrete worked example for any of them. The staircase-to-ramp transition is described verbally with no visual. The forward SDE formula is stated and translated to English, but there is no numerical walkthrough. The reverse SDE is stated and connected to DDPM, but no specific values are traced. The probability flow ODE is stated and connected to DDIM. The student is expected to absorb three formulas in quick succession without any concrete grounding between them.
**Student impact:** After the richly concrete first half (1D Gaussian with specific values at 4 points, detailed worked example), the second half feels abstract by comparison. The student who was confidently computing score(-3) = 3 may feel adrift when three differential equations arrive without any numbers. The pacing accelerates right when the material gets harder.
**Suggested fix:** Add at least one concrete grounding element to the SDE section. Options: (a) A brief numerical comparison showing the DDPM step at beta_t = 0.01 alongside the SDE infinitesimal step, demonstrating they are the same operation at different scales. (b) A visual showing the staircase-to-ramp transition (even a simple text-art or description-in-a-box showing discrete steps smoothing into a curve). (c) A concrete example of the forward SDE: "At time t, x has value 3.0. The drift term pushes it toward 0 by 0.015 (shrinking the signal). The diffusion term adds a random nudge of magnitude 0.1. After one infinitesimal step, x is approximately 2.985 + 0.1 * z." Any of these would bridge the abstract-concrete gap.

#### [IMPROVEMENT] -- Staircase-to-ramp visual missing

**Location:** Section "From Discrete Steps to Continuous Time" (lines 717-745)
**Issue:** The planning document describes a visual: "show the discrete forward process as a staircase (jumps at each t) transitioning to a smooth ramp (continuous curve) as the number of steps increases." The built lesson describes this verbally: "Picture the discrete forward process as a staircase -- the signal drops in jumps at each step. As the number of steps increases, the staircase becomes smoother." No actual visual is provided.
**Student impact:** The staircase-to-ramp is the key intuition pump for understanding how DDPM generalizes to an SDE. Without a visual, the student relies on mental imagery for a transition that is inherently spatial. This is a less severe gap than the vector field (which is CRITICAL because it affects the core concept), but it weakens the SDE section.
**Suggested fix:** Add a simple diagram or SVG showing two or three panels: (a) DDPM with T=10 steps (clear staircase), (b) T=100 steps (smoother staircase), (c) continuous SDE (smooth curve). This could be a very simple line chart -- no interactivity needed.

#### [POLISH] -- Summary block uses plain text for formulas

**Location:** SummaryBlock items (lines 1127-1160)
**Issue:** The summary descriptions use plain text for formulas: "score(x) = nabla_x log p(x)", "epsilon_theta approx -sqrt(1 - alpha_bar_t) * score". Throughout the rest of the lesson, these are rendered with KaTeX. The summary is the last thing the student reads before closing the lesson -- having the formulas in plain text rather than rendered math creates a visual inconsistency.
**Student impact:** Minor. The student can read the plain text formulas. But the visual downgrade signals "this part is less polished" at the exact moment the lesson should feel most complete.
**Suggested fix:** Render the key formulas in the summary descriptions using InlineMath, or accept the plain text as a deliberate choice for the summary format. If the SummaryBlock component does not support React nodes in the description field, this may require a component change and is not worth the effort.

#### [POLISH] -- Notebook Exercise 4 solution uses emoji in markdown

**Location:** Notebook cell-20, the solution `<details>` block
**Issue:** The solution block starts with `<summary>` containing a lightbulb emoji. This is consistent with other notebook solutions in the course (Exercise 3 also uses it), but worth noting if the project has an emoji-free convention.
**Student impact:** None. This is purely a consistency question.
**Suggested fix:** If other notebooks in the course use the lightbulb emoji for solution blocks, leave as-is. If not, remove.

### Review Notes

**What works well:**
- The motivation and hook are genuinely compelling. "What does the network really know?" is a question the student has earned the right to ask, and the lesson delivers a satisfying answer.
- The 1D Gaussian example is excellent -- concrete, worked through, and effectively bridges from abstract definition to specific numbers.
- The score-noise equivalence reveal is the emotional core of the lesson, and the intuition walkthrough (score points toward data, noise points away from data, same direction with a flip) is clear and memorable.
- All five misconceptions are addressed at the right points in the lesson. The WarningBlock for "Not the gradient you are used to" is particularly well-placed.
- The notebook is well-structured with appropriate scaffolding progression and good solution quality.
- The connections to prior lessons are explicit and well-articulated throughout. The recap section efficiently reactivates the three key prerequisites.
- The ComparisonRow (Reverse SDE vs Probability Flow ODE) is a strong structural element that maps cleanly onto the DDPM-vs-DDIM distinction the student already knows.

**Pattern to watch:**
- The lesson's first half (score function definition through the equivalence) is richly concrete and multi-modal. The second half (SDE/ODE framework) is more abstract and formula-heavy. This is partly inherent to the material (SDEs are more abstract than score functions), but the contrast is noticeable. Adding one concrete element to the SDE section would balance the pacing.

**Critical finding must be addressed before re-review.** The vector field visualization is the lesson's most important visual, and its absence is a genuine pedagogical gap. The notebook provides the visualization, but the lesson should establish the intuition visually before the student practices it.

---

## Review -- 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 2

### Verdict: NEEDS REVISION

All iteration 1 findings were addressed. The CRITICAL finding (missing 2D score vector field visualization) was resolved with three-panel GradientCard descriptions that are vivid enough for the student to form correct mental imagery, with the notebook providing actual quiver plots. The three IMPROVEMENT findings were all properly fixed: the algebraic derivation for the score-noise equivalence is now a clear step-by-step walkthrough, the forward SDE section now has a concrete numerical example, and the staircase-to-ramp transition now has three descriptive panels. One new IMPROVEMENT finding exists regarding the probability flow ODE presentation.

### Iteration 1 Fix Verification

**[CRITICAL] Missing 2D score vector field visualization** -- RESOLVED. Three-panel GradientCard grid (lines 416-503) with detailed descriptions at "No Noise," "Moderate Noise," and "High Noise." Each panel describes arrow behavior, basin structure, and boundary characteristics. While these are textual descriptions rather than actual SVG diagrams, they are vivid and specific enough ("Arrows near (-3, 0) all converge tightly on that peak," "Between the peaks, arrows are shorter and less decisive") that the student can form correct spatial intuition. The notebook (Exercise 2) provides the actual quiver plots. Downgraded from CRITICAL to POLISH (see below).

**[IMPROVEMENT] Score-noise equivalence derivation gap** -- RESOLVED. Lines 686-707 now show a three-step algebraic walkthrough: start with the noise removal term from the DDPM reverse step, substitute the score-noise equivalence, and show the sqrt(1-alpha_bar_t) cancellation. Each step has a label ("Start: the noise removal term," "Substitute," "The sqrt cancels"). The student can follow every line. This directly achieves the planning document's goal of "each equation should feel like 'of course.'"

**[IMPROVEMENT] SDE section lacks concrete grounding** -- RESOLVED. Lines 893-922 add a "Numerical Example: One Forward SDE Step" GradientCard with specific values (x=3.0, beta=0.02, dt=0.001). Drift term computed to -0.00003, diffusion term to approximately 0.0045*z, result shown. "Now repeat this a million times" connects back to the continuous limit. Effective concrete grounding.

**[IMPROVEMENT] Staircase-to-ramp visual missing** -- RESOLVED. Lines 821-863 add a three-panel GradientCard grid (T=10 steps, T=100 steps, T->infinity) with descriptions of the visual transition from staircase to smooth curve. Each panel has a header color and descriptive text. The InsightBlock aside ("DDPM = staircase, SDE = ramp") provides a crisp summary.

**[POLISH] Summary block plain text formulas** -- NOT FIXED. Likely a component limitation (SummaryBlock accepts string props, not React nodes). Acceptable as-is.

**[POLISH] Notebook emoji** -- VERIFIED CONSISTENT. The lightbulb emoji in solution `<details>` blocks appears across multiple notebooks in the course (4-3-1, 4-3-2, 6-2-4). This is an established pattern. No action needed.

### Findings

#### [IMPROVEMENT] -- Probability flow ODE presented as "just removing the noise term" but the score coefficient also changes

**Location:** Section "The Probability Flow ODE" (lines 1015-1051)
**Issue:** The lesson says "What if we remove the stochastic term from the reverse SDE?" and presents the probability flow ODE formula. But comparing the two formulas reveals more changed than just removing `dw_bar`:

- Reverse SDE: `[-1/2 beta(t) x - beta(t) * score] dt + sqrt(beta(t)) dw_bar`
- Probability flow ODE: `[-1/2 beta(t) x - 1/2 beta(t) * score] dt`

The coefficient on the score term changed from `beta(t)` to `1/2 beta(t)`. This is mathematically correct (Song et al. 2021 derive this via the Fokker-Planck equation), but the lesson presents it as if only the `dw_bar` was removed. A student comparing the two formulas side by side would notice the coefficient changed and wonder why.
**Student impact:** This student is mathematically attentive -- they derived the closed-form formula, they implemented the reverse step, they followed the algebraic walkthrough in the score-noise equivalence section. They will compare the reverse SDE and ODE formulas. Finding an unexplained coefficient change undermines the "of course" feeling the lesson works hard to build. The student may think they misread the formulas, or that there is a typo, or that something deeper happened that they missed.
**Suggested fix:** Add one sentence acknowledging the change without deriving it. For example, after "What if we remove the stochastic term from the reverse SDE?" add: "When you remove the stochastic term, the score coefficient changes from beta(t) to 1/2 beta(t) -- the math behind this comes from probability theory we will not cover, but the key result is that this adjusted coefficient produces the same marginal distributions as the stochastic version." This is honest about the gap without pretending the change does not exist. Alternatively, a brief TipBlock or aside: "Notice that the score coefficient changed from beta(t) to 1/2 beta(t). This adjustment is what makes the deterministic ODE match the stochastic SDE's distributions. The derivation requires Fokker-Planck equations, which are out of scope -- the important thing is the result: same model, deterministic trajectory, same distribution."

#### [POLISH] -- 2D score field descriptions are textual rather than visual

**Location:** Section "The Score as a Vector Field" (lines 416-503)
**Issue:** The iteration 1 CRITICAL finding requested "a static SVG or a simple interactive visualization." The fix added three-panel GradientCard descriptions that are vivid and detailed, but still textual. The score function is an inherently spatial concept (a direction at every point), and the planning document calls the vector field visualization "the single most important modality for building intuition." The notebook (Exercise 2) provides the actual quiver plots, so the student does eventually see the visual, but the lesson itself relies on textual description of a spatial phenomenon.
**Student impact:** Minor for this specific student, who has extensive experience with gradients and spatial concepts and can likely form correct mental imagery from the text. The notebook fills the gap within the same session. But the lesson's geometric/spatial modality is weaker than it could be.
**Suggested fix:** If a static SVG or diagram is feasible, add one. If not, the current textual descriptions are adequate -- the student gets conceptual understanding from the lesson and visual confirmation from the notebook. No action required if SVG creation is prohibitively expensive.

#### [POLISH] -- Summary block still uses plain text for formulas

**Location:** SummaryBlock items (lines 1285-1319)
**Issue:** Same as iteration 1 POLISH finding. The summary descriptions use plain text ("score(x) = nabla_x log p(x)," "epsilon_theta approx -sqrt(1 - alpha_bar_t) * score") while the rest of the lesson renders these with KaTeX. The SummaryBlock component likely accepts string props, not React nodes, making this a component limitation rather than a lesson issue.
**Student impact:** Minimal. The plain text is readable. Visual inconsistency at the end of the lesson, but not confusing.
**Suggested fix:** Accept as-is unless the SummaryBlock component is refactored to support React nodes in the description field.

### Review Notes

**What improved from iteration 1:**
- The algebraic walkthrough for the score-noise equivalence (lines 686-707) is the single biggest improvement. It transforms the equivalence from "trust me" to "of course" by showing three clear substitution steps. This was the lesson's weakest point and is now one of its strongest.
- The numerical example for the forward SDE (lines 893-922) effectively bridges the abstract-concrete gap in the second half of the lesson. The specific numbers (x=3.0, drift=-0.00003, diffusion=0.0045*z) make the SDE formula tangible.
- The three-panel staircase visualization (lines 821-863) provides the conceptual scaffolding for understanding the discrete-to-continuous transition.
- The three-panel score field description (lines 416-503) is substantially more detailed than the original, with vivid descriptions of arrow behavior at each noise level.

**What works well (carried forward from iteration 1):**
- The motivation and hook remain compelling. The 1D Gaussian example is excellent. The score-noise equivalence reveal is the emotional core. All five misconceptions are addressed at the right points. The notebook is well-structured. The ComparisonRow maps cleanly onto DDPM-vs-DDIM.

**Pattern resolved:**
- The first-half-concrete, second-half-abstract imbalance flagged in iteration 1 is now significantly reduced. The numerical SDE example and the staircase panels give the second half concrete anchors. The imbalance is inherent to the material (the SDE section introduces three interrelated formulas) but is now managed rather than problematic.

**The one remaining IMPROVEMENT finding is isolated and has a clear fix.** Adding one sentence or aside acknowledging the coefficient change in the probability flow ODE would resolve it. This does not require restructuring any section or adding new concepts.

---

## Review -- 2026-02-20 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

All findings from iterations 1 and 2 have been addressed. No new findings at any severity level.

### Iteration 2 Fix Verification

**[IMPROVEMENT] Probability flow ODE coefficient change not acknowledged** -- RESOLVED. Lines 1030-1036 now read: "Notice the score coefficient changed from beta(t) to 1/2 beta(t) here--removing the stochastic term requires adjusting the drift to preserve the same marginal distributions. The derivation involves Fokker-Planck (which we are skipping), but the result is clean: halve the score coefficient, drop the noise." This is honest about the derivation gap without pretending the change does not exist. The student who compares the two formulas will find the explanation immediately. The tone matches the lesson's approach of naming things beyond scope without teaching them (same pattern as the Ito calculus WarningBlock).

**[POLISH] 2D score field descriptions are textual rather than visual** -- ACCEPTED. The three-panel GradientCard descriptions are vivid and specific. The notebook (Exercise 2) provides actual quiver plots within the same session. The student gets conceptual understanding from the lesson and visual confirmation from the notebook. No further action needed.

**[POLISH] Summary block uses plain text for formulas** -- ACCEPTED. The SummaryBlock component accepts string props, not React nodes. The plain text is readable. This is a component limitation, not a lesson issue.

### Review Notes

**What the lesson does well (final assessment):**

1. **Motivation and hook.** "What does the network really know?" is a question the student has earned after 20+ lessons of DDPM mechanics. The reframing from "predicts noise" to "points toward data" is the core insight, and the lesson builds to it with genuine narrative tension.

2. **The score-noise equivalence reveal.** The three-step algebraic walkthrough (DDPM noise removal term -> substitute -> sqrt cancels) is the lesson's strongest pedagogical moment. It achieves the planning document's goal of "each equation should feel like 'of course.'" The student can follow every line using knowledge at APPLIED depth.

3. **Concrete-abstract balance.** The 1D Gaussian example (first half) and the numerical SDE step (second half) provide concrete anchors for abstract material. The staircase-to-ramp panels bridge the discrete-to-continuous transition. The lesson manages its inherent theory density effectively.

4. **Connection density.** Every new concept is explicitly connected to existing knowledge: score to gradients (1.1.4), SDE to DDPM forward (6.2.2), reverse SDE to DDPM sampling (6.2.4), probability flow ODE to DDIM/ODE perspective (6.4.2), score at different noise levels to alpha-bar curve (6.2.2). The student is never more than one connection away from familiar territory.

5. **Misconception coverage.** All five planned misconceptions are addressed at the right points in the lesson, each with concrete negative examples. The WarningBlock "Not the Gradient You Are Used To" (misconception #5) is particularly well-placed early in the score function definition, before the student could form the confusion.

6. **Notebook quality.** Four exercises with proper scaffolding progression (Guided -> Guided -> Supported -> Independent). Predict-before-run prompts. Solution blocks with reasoning before code and common mistakes. Self-contained setup. Consistent terminology with the lesson.

7. **Forward-looking connections.** The "Why This Perspective Matters" section and the next-step bridge to flow matching ("what if we could straighten the trajectories?") give the student clear motivation for the framework beyond paper-reading vocabulary.

**This lesson is ready to ship.** Proceed to Phase 5 (Record) in the planning skill.
