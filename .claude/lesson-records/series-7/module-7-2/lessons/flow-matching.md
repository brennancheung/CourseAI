# Lesson: Flow Matching

**Slug:** `flow-matching`
**Series:** 7 (Post-SD Advances), **Module:** 7.2 (The Score-Based Perspective), **Position:** Lesson 5 of 11 in series, Lesson 2 of 2 in module
**Cognitive Load:** BUILD (follows STRETCH lesson on score functions and SDEs)

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Score function as gradient of log probability | DEVELOPED | score-functions-and-sdes (7.2.1) | "Compass toward likely data." nabla_x log p(x) = direction toward higher probability at every point in data space. Worked through 1D Gaussian concrete example (N(0,1): score(x) = -x). 2D vector field with arrows converging on peaks at different noise levels. |
| Score-noise equivalence | DEVELOPED | score-functions-and-sdes (7.2.1) | epsilon_theta approx -sqrt(1-alpha_bar_t) * score. The noise prediction IS a scaled score function. DDPM was always score-based. Three-step algebraic walkthrough performed. |
| SDE forward process as continuous-time DDPM | INTRODUCED | score-functions-and-sdes (7.2.1) | dx = -1/2 beta(t) x dt + sqrt(beta(t)) dw. "Staircase to ramp" analogy. Concrete numerical example with one SDE step. Student understands the continuous limit conceptually. |
| Reverse SDE (score-guided generation) | INTRODUCED | score-functions-and-sdes (7.2.1) | Stochastic generation guided by the score function. Connected to DDPM reverse sampling. Anderson 1982 result stated without proof. |
| Probability flow ODE | INTRODUCED | score-functions-and-sdes (7.2.1) | Deterministic version of the reverse SDE. Connected to DDIM. "Same landscape, different lens." Formula shown, coefficient change acknowledged honestly. |
| ODE trajectory as generation path | INTRODUCED | samplers-and-efficiency (6.4.2) | Model predictions define a smooth trajectory from noise to data. Samplers are ODE solvers following this trajectory. "The model defines where to go. The sampler defines how to get there." |
| Euler's method as ODE solver | INTRODUCED | samplers-and-efficiency (6.4.2) | Compute direction at current point, take step of size h, repeat. Structurally identical to gradient descent. First-order: one model evaluation per step. |
| DDPM forward process (discrete noise addition) | DEVELOPED | the-forward-process (6.2.2) | x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon. Variance-preserving formulation. Closed-form shortcut. |
| DDPM noise prediction training objective | DEVELOPED | learning-to-denoise (6.2.3) | Loss = MSE between predicted and actual noise. Student implemented the training loop. |
| DDIM predict-x0-then-jump mechanism | DEVELOPED | samplers-and-efficiency (6.4.2) | Two-step: predict clean image from noise prediction, leap to arbitrary target timestep via closed-form formula. |
| Alpha-bar as signal-to-noise ratio | DEVELOPED | the-forward-process (6.2.2) | Starts near 1 (clean), drops to near 0 (pure noise). Interactive widget experience. |
| Gradient as direction of steepest ascent | DEVELOPED | gradient-descent (1.1.4) | "Ball rolling downhill." Deeply embedded since lesson 4. |
| Vector field concept (score as direction at every point) | DEVELOPED | score-functions-and-sdes (7.2.1) | At every point in data space, a vector indicating direction toward higher probability. Complex at low noise, simple at high noise. Notebook quiver plots. |

### Mental Models and Analogies Already Established

- **"Compass toward likely data"** -- the score function at every point tells you which direction to move for higher probability. Zero at the peak (you have arrived).
- **"The score was hiding inside DDPM all along"** -- noise prediction IS the score function (scaled). Every DDPM model is a score-based model in disguise.
- **"Staircase to ramp"** -- DDPM (discrete steps) is a staircase, SDE (continuous time) is a ramp. Same trajectory, smoother description.
- **"Same landscape, different lens"** -- reverse SDE (DDPM sampling, stochastic) vs probability flow ODE (DDIM sampling, deterministic). Different routes, same landscape.
- **"The model defines where to go. The sampler defines how to get there."** -- model vs sampler separation.
- **"Ball rolling downhill"** -- gradient as direction of descent. Deeply embedded.
- **"Predict and leap"** -- DDIM's predict-x0-then-jump mechanism.

### What Was Explicitly NOT Covered

- Flow matching, conditional flow matching, rectified flow -- explicitly deferred as "next lesson" in score-functions-and-sdes
- Velocity prediction parameterization -- not discussed anywhere
- Straight-line interpolation between noise and data -- the student works with curved diffusion trajectories only
- Any training objective other than noise prediction (epsilon parameterization) -- the student knows the MSE noise prediction loss from DDPM
- Why diffusion trajectories curve -- the student knows trajectories exist (from 6.4.2 and 7.2.1) but has not analyzed their shape
- SD3, Flux, or any architecture that uses flow matching -- deferred to Module 7.4

### Readiness Assessment

The student is excellently prepared. They have:
1. The score function and SDE/ODE framework freshly established from the previous lesson (INTRODUCED/DEVELOPED)
2. A strong understanding that generation follows trajectories through data space (from 6.4.2 and 7.2.1)
3. The probability flow ODE as a deterministic trajectory (INTRODUCED) -- flow matching will be presented as a different way to define the trajectory
4. Euler's method as an ODE solver (INTRODUCED) -- relevant for understanding why straighter paths need fewer steps
5. The bridge from score-functions-and-sdes that explicitly motivated flow matching: "What if we could straighten the trajectories?"

The main gap is entirely expected: the student has never encountered the idea of choosing the trajectory shape, or that the diffusion trajectory could be anything other than what DDPM defines. This is the core insight of the lesson.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to understand flow matching as a generative framework that defines straight-line trajectories between noise and data, replacing the curved diffusion paths with simpler, faster-to-follow paths via a velocity prediction training objective.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Probability flow ODE (deterministic trajectory from noise to data) | INTRODUCED | INTRODUCED | score-functions-and-sdes (7.2.1) | OK | Need the student to understand that generation follows a deterministic trajectory. They have this from the previous lesson where the probability flow ODE was formalized. |
| Score function as vector field (direction at every point) | INTRODUCED | DEVELOPED | score-functions-and-sdes (7.2.1) | OK | Need the student to see that the network learns a "direction" at every point. The score function establishes this concept. Flow matching replaces the score with a velocity field, same concept of "direction at every point." |
| ODE trajectory concept | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) + score-functions-and-sdes (7.2.1) | OK | Need the student to have the picture of generation as following a path through space. They have this from both 6.4.2 (informal) and 7.2.1 (formalized as the probability flow ODE). |
| Euler's method (compute direction, take step, repeat) | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) | OK | Need to understand that ODE solving is "evaluate direction, step, repeat." Relevant for understanding why straighter paths need fewer Euler steps. |
| DDPM noise prediction training objective (MSE loss on epsilon) | INTRODUCED | DEVELOPED | learning-to-denoise (6.2.3) | OK | Need to contrast flow matching's velocity prediction objective with DDPM's noise prediction. Student has implemented the DDPM training loop. More than sufficient. |
| DDPM forward process (interpolation between data and noise) | INTRODUCED | DEVELOPED | the-forward-process (6.2.2) | OK | Need the student to recognize that x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon defines a specific (nonlinear) interpolation. Flow matching uses a different (linear) interpolation. |
| Alpha-bar noise schedule | INTRODUCED | DEVELOPED | the-forward-process (6.2.2) | OK | Need to reference the nonlinear relationship between t and alpha_bar. Flow matching replaces this with a linear schedule. |
| Score-noise equivalence | INTRODUCED | DEVELOPED | score-functions-and-sdes (7.2.1) | OK | Need to connect noise prediction, score prediction, and velocity prediction as three parameterizations of the same underlying vector field. |

### Gap Resolution

No GAPs or MISSING prerequisites. All required concepts are available at sufficient depth. The previous lesson was specifically designed to prepare for this one.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Flow matching is a completely new paradigm unrelated to diffusion" | Flow matching uses different notation, a different training objective, and the word "flow" instead of "diffusion." It sounds like a different field. | Flow matching produces a vector field that maps noise to data, just like the probability flow ODE. The trained network takes a noisy input and a timestep and outputs a direction, exactly like a DDPM model. You can even convert between noise prediction, score prediction, and velocity prediction with simple algebra. Same family, different member. | After explaining conditional flow matching. Explicit connection section showing all three parameterizations are interconvertible. |
| "Straight trajectories are just a nice-to-have aesthetic preference" | The student may not immediately see why trajectory shape matters for practical quality. Curved vs straight seems like an abstract geometric distinction. | Compare Euler's method on a curved path vs a straight path. On a curved path, each Euler step overshoots the curve (the trajectory bends after you step), requiring small steps to stay accurate. On a straight path, Euler's method is EXACT in one step -- no overshooting, no curvature error. This is why flow matching models can generate in fewer steps. | In the "Why Straight Matters" section, immediately after showing curved vs straight. Tie to Euler's method the student already knows from 6.4.2. |
| "Velocity prediction is fundamentally different from noise prediction" | Different training target (predict v instead of epsilon), different formula, different name. The student has spent 20+ lessons with noise prediction and may think velocity prediction requires a different kind of network or a different generative process. | The same U-Net (or DiT) architecture works for all three parameterizations. You can convert between them with one line of algebra: v_t = alpha_t * epsilon - sigma_t * x_0, and epsilon = sigma_t * v_t + alpha_t * x_t (at the appropriate parameterization). The network's architecture does not change -- only the training target and loss function change. | After introducing velocity prediction. Show the conversion formulas. |
| "Flow matching requires a new architecture or special model" | "Flow matching" sounds like it might need a fundamentally different neural network. The student's mental model of "U-Net predicts noise" might make them think "U-Net predicts velocity" is a different architecture. | SD3 and Flux use the same basic architecture pattern (a neural network that takes x_t and t as input and outputs a prediction). The change is in the training objective and the interpolation schedule, not the model architecture. In fact, the shift from U-Net to DiT (which SD3/Flux also do) is a separate architectural choice that happened to coincide with the shift to flow matching. | In the SD3/Flux connection section. Separate the training objective change (flow matching) from the architecture change (DiT). |
| "Linear interpolation between noise and data is too simple to work -- surely you need the carefully designed noise schedule" | The student spent time learning about variance-preserving formulations, cosine schedules, alpha-bar curves. A simple linear interpolation x_t = (1-t)*x_0 + t*epsilon seems too naive. | The simplicity IS the advantage. The DDPM noise schedule (sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon) was designed to make the forward process have certain mathematical properties (variance-preserving, tractable posteriors). Flow matching sidesteps all of that by working directly with the conditional vector field. The linear interpolation means the velocity is constant along each conditional path (dx/dt = epsilon - x_0), which makes the training objective trivially simple. | After showing the flow matching interpolation. Contrast with the DDPM schedule explicitly. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| 2D point cloud: curved diffusion trajectory vs straight flow matching trajectory | Positive | Show the core insight visually. The diffusion probability flow ODE traces curved paths from noise points to data points. Flow matching defines straight-line paths between paired noise and data points. The student sees that straight paths are simpler to follow. | The 2D case makes trajectory shape visible. The student has already seen 2D score fields in the notebook from the previous lesson (Exercise 2 quiver plots), so extending to 2D trajectories is natural. The simplest possible example that shows the core idea. |
| Euler's method on a curved path vs a straight path | Positive | Demonstrate WHY straight paths need fewer steps. On a curved path, Euler overshoots at each step because it extrapolates linearly along a curve. On a straight path, Euler's extrapolation is exact -- the true trajectory IS a straight line. One Euler step from any point on a straight path lands exactly on the path. | Connects flow matching's practical benefit (fewer steps) to a concept the student already has (Euler's method from 6.4.2). Makes the abstract "straight paths are better" into a concrete computational advantage. Confirms the pattern generalizes beyond the 2D toy case. |
| Flow matching training: one training sample | Positive (stretch) | Show how a single training step works concretely. Pick a data point x_0, sample noise epsilon, pick a random time t, compute x_t = (1-t)*x_0 + t*epsilon (just a weighted average), compute the target velocity v = epsilon - x_0 (constant, does not depend on t!), train the network to predict v from (x_t, t). | Demystifies the training objective by making it completely concrete. The student has seen the DDPM training loop (sample noise, compute noisy image, predict noise, MSE loss). Showing the flow matching version side by side reveals how much simpler it is. The constant velocity target is the key surprise. |
| Diffusion on a curved path where Euler needs many steps to stay accurate | Negative | Show what happens with too few steps on a curved trajectory. With 5 Euler steps on a curved diffusion ODE path, the trajectory drifts off course and the generated sample is poor. With 5 Euler steps on a straight flow matching path, the trajectory is exact. This is NOT a case where "more steps always helps" -- on a straight path, one step is already exact. | Defines the boundary: the advantage of flow matching is not about being "generally better" -- it is specifically about trajectory geometry and its interaction with ODE solvers. On a genuinely straight path, more steps do not help (one is sufficient for that path). On a curved path, more steps are always needed. |

---

## Phase 3: Design

### Narrative Arc

The previous lesson ended with a promise: "What if we could straighten the trajectories?" The student now knows that generation follows a trajectory -- the probability flow ODE traces a deterministic path from noise to data, guided by the score function. But look at these paths: they curve through high-dimensional space, bending as the score field changes direction at different noise levels. This curvature is a problem. Every bend in the trajectory is a place where Euler's method (or any ODE solver) accumulates error, requiring small steps to stay on track. The student has already felt this: DDPM needs 1000 steps, DDIM improves to 50, DPM-Solver++ gets to 15-20 -- but all of these are fighting the same curved trajectory with increasingly clever solvers.

Flow matching asks a radical question: what if we designed the trajectory to be straight in the first place? Instead of the complex noise schedule that DDPM defines (sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon, with its nonlinear coefficients), flow matching uses the simplest possible interpolation: x_t = (1-t) * x_0 + t * epsilon. A straight line from data to noise. The velocity along this path is constant (epsilon - x_0), and a network trained to predict this velocity produces a vector field that defines straight trajectories. On a straight path, Euler's method is exact -- one step from any point lands you exactly where you need to go. This is why SD3 and Flux can generate high-quality images in far fewer steps than DDPM-style models, and why the field has shifted to flow matching as the standard training objective.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Geometric/Spatial** | Side-by-side comparison of curved diffusion trajectories vs straight flow matching trajectories in 2D. Multiple trajectories from noise points to data points. The diffusion paths curve; the flow matching paths are straight lines. A second diagram showing Euler's method on each: on the curved path, Euler steps overshoot; on the straight path, Euler steps are exact. | Trajectory shape IS the core insight. The geometric difference between curved and straight is the single most important thing the student needs to see. This connects to the 2D score field visualizations from the previous lesson and the trajectory concept from 6.4.2. |
| **Symbolic** | Three key equations: (1) flow matching interpolation: x_t = (1-t)*x_0 + t*epsilon, (2) target velocity: v_t = dx_t/dt = epsilon - x_0, (3) flow matching loss: L = MSE(v_theta(x_t, t), epsilon - x_0). Plus the conversion formulas between velocity, noise, and score parameterizations. | The equations are the permanent takeaway for paper-reading. They come AFTER the geometric intuition, not before. Each equation should feel trivially simple compared to the DDPM equivalents ("this is just a weighted average," "this is just the derivative of a line"). |
| **Concrete example** | One flow matching training step worked through with numbers. x_0 = [3.0, 1.0] (a data point), epsilon = [-1.0, 2.0] (sampled noise), t = 0.3 (random time). Compute x_t = 0.7 * [3.0, 1.0] + 0.3 * [-1.0, 2.0] = [1.8, 1.3]. Target velocity: [-1.0, 2.0] - [3.0, 1.0] = [-4.0, 1.0]. Train network to predict [-4.0, 1.0] from ([1.8, 1.3], t=0.3). | The student needs to compute a flow matching training step by hand to demystify it. The numbers make it concrete and show how much simpler it is than the DDPM training step (no sqrt(alpha_bar), no variance-preserving rescaling, just a weighted average and a subtraction). |
| **Verbal/Analogy** | "GPS recalculating vs a straight highway." On a winding road (curved diffusion trajectory), your GPS must constantly recalculate as the road turns -- miss a turn and you are off course. On a straight highway (flow matching trajectory), you point the car in the right direction and drive -- no recalculation needed, no chance of missing a turn. Fewer steps (GPS recalculations) needed because the path has no curves to navigate. | Connects the abstract trajectory geometry to a familiar experience. The "recalculation" maps directly to ODE solver steps. Makes the computational advantage intuitive before the formal explanation. |
| **Intuitive** | The "of course" chain for conditional flow matching: (1) We want to go from noise to data. (2) The simplest path is a straight line. (3) A straight line has constant velocity. (4) Constant velocity is trivially easy to predict. (5) So train a network to predict the velocity. Each step follows inevitably from the previous one. | Mirrors the "of course" design challenge pattern established in ControlNet (7.1.1) and IP-Adapter (7.1.3). The student should feel they could have invented flow matching given the framework from the previous lesson. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. Conditional flow matching (straight-line interpolation + velocity prediction training objective) -- NEW
  2. Rectified flow (iterative trajectory straightening for learned flow fields) -- NEW
  - Additionally, velocity prediction parameterization is NEW but is a natural consequence of conditional flow matching rather than a separate concept
- **Previous lesson load:** score-functions-and-sdes was STRETCH (2-3 new concepts: score function, SDE framework, probability flow ODE formalized)
- **Is this appropriate?** Yes, BUILD is appropriate. The student just completed the hardest theoretical lesson in the series. This lesson should feel like a payoff -- the theoretical framework from lesson 4 makes flow matching comprehensible as a natural simplification. The two new concepts (conditional flow matching and rectified flow) build directly on the trajectory/vector field framework established in the previous lesson. The cognitive effort is in seeing the simplification, not in absorbing new abstractions.

### Connections to Prior Concepts

| New Concept | Connects To | How |
|-------------|------------|-----|
| Straight-line interpolation x_t = (1-t)*x_0 + t*epsilon | DDPM forward process x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon (6.2.2) | Both are interpolations between data and noise, parameterized by time. The DDPM version uses nonlinear coefficients (sqrt(alpha_bar_t), sqrt(1-alpha_bar_t)) that make the path curve. The flow matching version uses linear coefficients (1-t, t) that make the path straight. "Same idea -- interpolate between data and noise -- but with the simplest possible coefficients." |
| Velocity prediction | Noise prediction (6.2.3) and score prediction (7.2.1) | Three parameterizations of the same underlying concept: at every point (x_t, t), the network outputs a vector. Noise prediction says "what noise was added." Score prediction says "which direction increases probability." Velocity prediction says "which direction should x_t move along the trajectory." They are interconvertible with simple algebra. |
| Flow matching ODE | Probability flow ODE (7.2.1) | Both are deterministic ODEs that define generation trajectories from noise to data. The probability flow ODE uses the score function as its vector field; the flow matching ODE uses the learned velocity field. The key difference is trajectory shape: probability flow ODE trajectories curve (inherited from the diffusion SDE), flow matching trajectories are straight (by construction). |
| "Euler is exact on straight paths" | Euler's method from 6.4.2 | The student learned that Euler's method computes direction and takes a step. On curved paths, this overshoots. On straight paths, the extrapolation is perfect because the true trajectory IS linear. This is why flow matching models need fewer steps: the ODE solver's job is trivially easy. |
| Flow matching training loss (MSE on velocity) | DDPM training loss (MSE on noise) from 6.2.3 | Same structure: sample a random (x_0, t, noise), compute x_t, network predicts a target, MSE loss. The difference is the interpolation (linear vs variance-preserving) and the target (velocity vs noise). The training loop is structurally identical. |
| Rectified flow (straightening) | Higher-order ODE solvers from 6.4.2 | Both address the same problem (curvature in trajectories) from different angles. Higher-order solvers handle curvature at inference time (better stepping). Rectified flow eliminates curvature at training time (straighter paths). Complementary approaches. |

### Analogies to Extend

- **"The model defines where to go. The sampler defines how to get there."** from 6.4.2 -- still true for flow matching. The flow matching model defines the velocity field; the sampler (ODE solver) follows it. But now "where to go" is along straight lines, so the sampler's job is much easier.
- **"Same landscape, different lens"** from 7.2.1 -- extend to include flow matching as a third lens: diffusion SDE (stochastic curved paths), probability flow ODE (deterministic curved paths), flow matching ODE (deterministic straight paths). Same start (noise), same end (data), different routes.
- **"Predict and leap"** from DDIM (6.4.2) -- flow matching is the logical extreme: the entire trajectory IS a leap. No prediction needed mid-path because the path does not curve.

### Analogies That Could Be Misleading

- **"Staircase to ramp"** from 7.2.1 could mislead if the student thinks flow matching is just "even smoother steps." Flow matching is not about step refinement -- it is about trajectory redesign. The ramp (SDE) still curves; flow matching replaces the ramp with a straight line. Address this by contrasting flow matching with the SDE/ODE framework: "It is not a better solver for the same trajectory. It is a different trajectory entirely."
- **"Compass toward likely data"** (score function) could mislead if the student thinks flow matching uses the score function. Flow matching uses a velocity field, not a score field. The velocity field does NOT point toward high probability -- it points along the straight-line trajectory. The two fields happen to agree on endpoints but differ in direction at intermediate points. Address this in the velocity vs score comparison.

### Scope Boundaries

**This lesson IS about:**
- Why diffusion trajectories curve and why that is a problem
- Conditional flow matching: straight-line interpolation, velocity prediction, training objective
- Velocity prediction as a parameterization alongside noise and score prediction
- Rectified flow as trajectory straightening (at intuition level)
- Why flow matching enables fewer sampling steps (connection to Euler's method)
- Why SD3 and Flux use flow matching (mentioned, not architecturally detailed)

**This lesson is NOT about:**
- Optimal transport formulations of flow matching (Lipman et al. 2023 -- mentioned for paper reading, not developed)
- Continuous normalizing flows (CNFs) or their history -- flow matching is taught as a practical method, not as part of the CNF literature
- The DiT architecture (Module 7.4) -- we mention that SD3/Flux use flow matching but do not explain the transformer architecture
- Training a flow matching model from scratch (notebook uses pre-trained models or toy examples)
- Rigorous proofs of why conditional flow matching converges to the marginal flow
- The reparameterization trick for flow matching or its relationship to the ELBO
- Consistency models (Module 7.3) -- mentioned as another speed approach, not developed
- Any implementation beyond toy 2D examples and pre-trained model usage

**Depth targets:**
- Conditional flow matching (straight-line interpolation + velocity prediction): DEVELOPED (intuition + formula + concrete example + geometric visualization + training loop comparison)
- Velocity prediction parameterization: DEVELOPED (formula, conversion to/from noise prediction and score, connection to DDPM training)
- Rectified flow: INTRODUCED (conceptual explanation + visual intuition for iterative straightening, but not worked through in detail)
- Connection to SD3/Flux: INTRODUCED (named, motivated, but architectural details deferred to Module 7.4)

---

### Lesson Outline

#### 1. Context + Constraints

- "The previous lesson gave you the score function and the SDE/ODE framework. This lesson is the payoff: a simpler way to train and sample from generative models."
- "We will stay at the intuition level for rectified flow. The rigorous optimal transport connection is out of scope."
- "By the end, you will understand why SD3 and Flux do not use DDPM-style noise prediction anymore, and what they do instead."
- ConstraintBlock: scope boundaries. This lesson does not cover the DiT architecture (that is Module 7.4) or consistency models (Module 7.3). We focus on the training objective and trajectory geometry.

#### 2. Recap

Brief reactivation of three concepts:
- **Probability flow ODE** (from 7.2.1): 2-3 sentences. "Generation follows a deterministic trajectory from noise to data. The probability flow ODE defines this trajectory using the score function. DDIM was approximately solving this ODE."
- **Euler's method** (from 6.4.2): 2-3 sentences. "The simplest ODE solver: compute direction at current point, take a step, repeat. Accuracy depends on step size and how much the trajectory curves."
- **DDPM training** (from 6.2.3): 2-3 sentences. "Sample data, add noise at random timestep, network predicts the noise, MSE loss. The same training setup will reappear with a different target."
- Transition: "The previous lesson ended with a question: these trajectories CURVE through space. What if we could straighten them? Let's find out what happens."

#### 3. Hook

Type: **Before/after comparison + challenge preview**

"Look at the probability flow ODE trajectories you studied in the previous lesson."

Present a description of diffusion ODE trajectories in 2D: multiple paths from noise points to data points, visibly curving through space. The paths bend because the score field changes direction at different noise levels -- at high noise the field is simple (nearly Gaussian), at low noise it is complex (multi-modal). The trajectory must follow this changing field, creating curves.

"Now imagine this: what if every trajectory were a straight line? Data point to noise point, no curves, no bends."

Present the same start/end points but with straight lines connecting them. Visually simpler. Dramatically fewer bends for an ODE solver to navigate.

"This is not wishful thinking. This is what flow matching does. And the math is embarrassingly simple."

#### 4. Explain: Why Trajectories Curve

**Part A: The problem with curved paths**

Connect to Euler's method: "You know Euler's method from the sampler lesson. Compute the direction at your current point, take a step. But what if the trajectory curves right after you step? Your step overshoots -- you expected the trajectory to keep going straight, but it turned. The more the trajectory curves, the smaller your steps must be to stay on track."

This is WHY diffusion needs many steps. Not because generation is inherently complex, but because the probability flow ODE trajectory CURVES, and ODE solvers need small steps to follow curves accurately. Higher-order solvers (DPM-Solver++) handle curvature better, but they are fighting the symptom, not the cause.

**Part B: Why diffusion trajectories curve**

The probability flow ODE trajectory is defined by the score function, which changes dramatically as noise level changes. At high noise (t near 1), the score field is smooth and nearly Gaussian. At low noise (t near 0), the score field is complex and multi-modal. The trajectory must navigate this changing landscape, and the change in direction creates curvature.

"The curves are not a bug in diffusion -- they are inherent to the score field changing with noise level."

InsightBlock (aside): "This is why DPM-Solver needs 15-20 steps, not 1. The curvature of the diffusion ODE trajectory sets a floor on how few steps any solver can take while staying accurate."

#### 5. Check #1

Two predict-and-verify questions:
1. "If the probability flow ODE trajectory were perfectly straight, how many Euler steps would you need to follow it exactly?" (Answer: One. Euler's method extrapolates linearly from the current point. If the trajectory IS a line, the extrapolation is exact regardless of step size.)
2. "Looking at the score field panels from the previous lesson -- the field changes from simple (high noise) to complex (low noise). How does this relate to where the probability flow ODE trajectory curves the most?" (Answer: The trajectory curves most where the score field changes the fastest -- in the transition from high noise to low noise. This is the middle region where the field structure is emerging. At the extremes, the field is relatively stable.)

#### 6. Explain: Conditional Flow Matching

**Part A: The key idea -- choose the path, then learn to follow it**

"Diffusion starts with a training objective (predict noise) and discovers that the resulting trajectories curve. Flow matching flips the order: CHOOSE the trajectory first (make it straight), then design a training objective to learn it."

This is the conceptual shift. In DDPM, the forward process (add noise) determines the trajectory shape. In flow matching, you define the trajectory shape directly and build a training objective around it.

**Part B: The simplest possible interpolation**

DDPM forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
- Nonlinear coefficients (sqrt of cumulative product)
- Variance-preserving (coefficients squared sum to 1)
- Path from x_0 to noise CURVES because of the nonlinear coefficients

Flow matching interpolation: x_t = (1-t) * x_0 + t * epsilon, where t goes from 0 to 1
- Linear coefficients
- At t=0: x_t = x_0 (pure data)
- At t=1: x_t = epsilon (pure noise)
- Path from x_0 to epsilon is a STRAIGHT LINE

ComparisonRow:
| | DDPM Interpolation | Flow Matching Interpolation |
|---|---|---|
| Formula | x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon | x_t = (1-t) * x_0 + t * epsilon |
| Coefficients | Nonlinear (sqrt of cumulative product) | Linear (1-t, t) |
| Path shape | Curved (variance-preserving constraint) | Straight line |
| At midpoint (t=0.5) | NOT 50/50 mix (depends on noise schedule) | Exactly 50/50 mix |

Address Misconception #5: "The simplicity IS the advantage. The DDPM noise schedule was carefully designed to make certain mathematical properties hold (variance-preserving, tractable posteriors). Flow matching sidesteps all of that -- the linear interpolation is simple and the resulting training objective is simple. No alpha-bar, no cumulative products, no special schedule."

**Part C: The velocity field**

"The path is a straight line from x_0 to epsilon. What is the velocity along this path?"

Take the derivative: v = dx_t/dt = d/dt [(1-t)*x_0 + t*epsilon] = epsilon - x_0

The velocity is CONSTANT. It does not depend on t. At every point along the straight-line path, the velocity is the same: epsilon - x_0 (the direction from data to noise).

InsightBlock (aside): "The velocity is constant because the path is a straight line. A straight line has no curvature, so the tangent direction never changes. This is the same reason constant velocity means zero acceleration in physics."

**Part D: The training objective**

"Train a network v_theta(x_t, t) to predict the velocity epsilon - x_0."

Flow matching training step:
1. Sample a data point x_0
2. Sample random noise epsilon ~ N(0, I)
3. Sample a random time t ~ Uniform(0, 1)
4. Compute x_t = (1-t) * x_0 + t * epsilon
5. Target: v = epsilon - x_0
6. Loss = MSE(v_theta(x_t, t), epsilon - x_0)

ComparisonRow:
| | DDPM Training | Flow Matching Training |
|---|---|---|
| Interpolation | x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon | x_t = (1-t)*x_0 + t*epsilon |
| Target | epsilon (the noise) | epsilon - x_0 (the velocity) |
| Loss | MSE(epsilon_theta, epsilon) | MSE(v_theta, epsilon - x_0) |
| Schedule | alpha_bar_t (carefully designed) | Uniform t in [0, 1] |

WarningBlock (aside): "Wait -- if we need x_0 to compute the target velocity (v = epsilon - x_0), how is this useful? The network gets (x_t, t) as input, NOT x_0. Just like DDPM: the training target uses x_0 and epsilon (both known during training), but the network only sees x_t and t. At inference, we do not have x_0 -- we follow the learned velocity field from noise to generate it."

**Part E: Concrete worked example**

Work through one training step with specific numbers:
- Data point: x_0 = [3.0, 1.0]
- Sampled noise: epsilon = [-1.0, 2.0]
- Random time: t = 0.3
- Interpolated point: x_t = 0.7 * [3.0, 1.0] + 0.3 * [-1.0, 2.0] = [2.1 + (-0.3), 0.7 + 0.6] = [1.8, 1.3]
- Target velocity: v = [-1.0, 2.0] - [3.0, 1.0] = [-4.0, 1.0]
- Network input: ([1.8, 1.3], t=0.3)
- Network target: [-4.0, 1.0]
- Loss: MSE(v_theta([1.8, 1.3], 0.3), [-4.0, 1.0])

"Compare this to the DDPM training step: no sqrt(alpha_bar), no variance-preserving rescaling, no noise schedule lookup. Just a weighted average for x_t and a subtraction for the target."

TipBlock (aside): "Notice: the target velocity [-4.0, 1.0] does not depend on t. The same (x_0, epsilon) pair produces the same target velocity at every timestep. The only thing t changes is WHERE along the straight line (x_t) the network is asked to make its prediction."

#### 7. Check #2

Two predict-and-verify questions:
1. "In flow matching, what happens if t=0.5 in the interpolation x_t = (1-t)*x_0 + t*epsilon?" (Answer: x_t = 0.5*x_0 + 0.5*epsilon -- exactly a 50/50 mix of data and noise. In DDPM, the midpoint depends on the noise schedule and is NOT a 50/50 mix.)
2. "A colleague says: 'Flow matching is simpler because it has no noise schedule.' Is this correct?" (Answer: Partially. Flow matching has a schedule -- it is just the trivial schedule t in [0, 1] with uniform sampling. The key simplification is that this schedule is linear, requiring no alpha_bar computation, no cumulative products, no careful tuning. The "schedule" is implicit in the linear interpolation.)

#### 8. Explain: Velocity, Noise, and Score -- Three Parameterizations

"The previous lesson showed that noise prediction is a scaled score function. Velocity prediction is the third member of this family."

Show the relationship between the three:
- **Noise prediction:** network outputs epsilon_theta -- the noise that was added
- **Score prediction:** network outputs s_theta -- the gradient of log probability (direction toward data)
- **Velocity prediction:** network outputs v_theta -- the tangent direction along the trajectory

All three are vector fields: at every point (x_t, t), the network outputs a vector. The vectors differ in direction and magnitude, but they encode the same information. You can convert between them:

- From noise to score: score = -epsilon / sqrt(1-alpha_bar_t) (from the previous lesson)
- From velocity to noise: epsilon = sigma_t * v_t + alpha_t * noise_component (depends on the specific parameterization)
- Key point: they are all different ways to parameterize the same underlying vector field

GradientCard: "Three Languages for the Same Idea"
- Noise prediction: "What noise was added?" (DDPM's language)
- Score prediction: "Which way is higher probability?" (Score-based diffusion's language)
- Velocity prediction: "Which way is the trajectory heading?" (Flow matching's language)

Same trained network architecture. Same kind of output (a vector at each point). Different training objectives and different interpretations.

Address Misconception #3: "Velocity prediction does not require a different architecture. The same U-Net or transformer that predicts noise can predict velocity instead. You change the training target and the loss function, not the model architecture."

#### 9. Check #3

Two predict-and-verify questions:
1. "If you have a model trained with noise prediction (DDPM-style) and a model trained with velocity prediction (flow matching-style), can you use the same ODE solver for both?" (Answer: Yes. Both models produce a vector field. The solver follows the trajectory defined by that field. The difference is that the flow matching model's trajectories tend to be straighter, so the solver needs fewer steps.)
2. "A paper says a model was trained with 'v-prediction.' Is this the same as flow matching?" (Answer: Not exactly. V-prediction can be used within the DDPM framework too -- it is a parameterization choice. Flow matching specifically refers to the combination of linear interpolation AND velocity prediction AND uniform time sampling. V-prediction within a DDPM noise schedule is different from flow matching's v-prediction with linear interpolation. The terminology is often used loosely in practice.)

#### 10. Explain: Rectified Flow

**Part A: The problem with conditional flow matching**

"We said flow matching defines straight paths between paired (x_0, epsilon) points. But we train the network on MANY such pairs. The aggregate velocity field -- what the network actually learns -- is an average of all these individual straight-line velocities. This average can reintroduce curvature."

Think of it this way: at a given point x_t, many different (x_0, epsilon) pairs could have passed through this point. The network's prediction at x_t is an average of all their velocities. If different pairs have very different velocities at the same point, the average introduces curvature in the aggregate trajectory.

**Part B: Rectified flow -- iterative straightening**

"Rectified flow is a simple but clever idea: take the learned flow, generate (noise, data) pairs using it, and re-train on the new pairs."

The process:
1. Train a flow matching model (round 1)
2. Generate samples: start from noise epsilon, follow the learned velocity field to produce data x_0
3. Now you have new (x_0, epsilon) pairs that are "aligned" -- epsilon and x_0 are connected by the learned flow
4. Re-train flow matching on these new pairs (round 2)
5. The new model's trajectories are straighter because the pairs are more naturally aligned

Each round of rectification straightens the trajectories further. In practice, 1-2 rounds of rectification significantly reduce the number of steps needed.

PhaseCard sequence (3 cards):
1. "Initial Training" -- Train on random (data, noise) pairs. Individual paths are straight, but the aggregate field has curvature from averaging.
2. "Generate Aligned Pairs" -- Use the model to generate (noise, data) pairs. These pairs are connected by the learned flow, not random pairing.
3. "Retrain" -- Train on the aligned pairs. The averaging now introduces less curvature because the pairs are naturally aligned.

InsightBlock (aside): "Rectified flow is to flow matching what distillation is to temperature -- it refines the process. The first model is good; the rectified model is straighter and needs fewer steps."

#### 11. Check #4 (Transfer)

Two transfer questions:
1. "SD3 and Flux use flow matching. Based on what you know, what advantage does this give them over a DDPM-trained model with the same architecture?" (Answer: Straighter trajectories mean fewer sampling steps for the same quality. SD3 can generate in 20-30 steps where a DDPM-trained model of similar size might need 50+. The training objective is also simpler -- no carefully tuned noise schedule, just linear interpolation.)
2. "A colleague proposes: 'Instead of rectified flow, why not just use DPM-Solver++ on the flow matching model?' Would this help? Why or why not?" (Answer: It would help, but less than you might think. DPM-Solver++ handles curvature at inference time by estimating it with multiple function evaluations. But if the trajectories are already nearly straight from flow matching, there is little curvature to handle. Rectified flow eliminates curvature at the source; DPM-Solver++ compensates for it at inference. Both help, but the combination gives diminishing returns compared to either alone.)

#### 12. Elaborate: Why Flow Matching Won

Three reasons flow matching became the standard for modern architectures:

**GradientCard 1 (emerald): "Simpler Training"**
- No noise schedule to tune (no alpha_bar, no beta_t, no cosine vs linear debate)
- Linear interpolation instead of variance-preserving formulation
- Uniform time sampling instead of importance-weighted timestep selection
- The entire training loop fits in fewer lines of code

**GradientCard 2 (blue): "Fewer Steps"**
- Straighter trajectories mean ODE solvers converge in fewer steps
- Euler's method is nearly exact on straight paths
- Practical impact: 20-30 steps for SD3/Flux vs 50+ for DDPM-style models
- Rectified flow can push this even lower

**GradientCard 3 (violet): "Architecture Independence"**
- Flow matching works with U-Nets, transformers, or any architecture
- The training objective does not depend on architectural choices
- This independence made it easy to adopt when the field shifted from U-Net to DiT (Module 7.4)

Address Misconception #1 here: "Flow matching is not a different paradigm from diffusion. It produces a vector field that maps noise to data, just like the probability flow ODE. The difference is HOW the vector field is trained (velocity vs score/noise) and WHAT trajectory it defines (straight vs curved). Same family, different member."

Address Misconception #4 here: "SD3 and Flux made two changes simultaneously: they switched from U-Net to DiT (architecture change) AND from noise prediction to flow matching (training objective change). These are independent choices. You could use flow matching with a U-Net, or noise prediction with a DiT. The changes happened to coincide, but they solve different problems."

#### 13. Practice (Notebook)

Design: 4 exercises, Guided -> Guided -> Supported -> Independent progression. Exercises are cumulative (each builds understanding for the next).

**Exercise 1 (Guided): Flow Matching vs DDPM Interpolation**
- Implement both interpolation schemes in 1D: DDPM (with a cosine schedule) and flow matching (linear).
- Plot the interpolation paths from x_0=5.0 to epsilon=-2.0 for both.
- Predict-before-run: "Which path is straighter?" (The flow matching path is a line; the DDPM path curves because of the nonlinear schedule coefficients.)
- Plot the "velocity" (derivative) along each path. Flow matching: constant. DDPM: varies with t.
- What it tests: understanding of the interpolation difference and why linearity produces straight paths. Depth: DEVELOPED for flow matching interpolation.

**Exercise 2 (Guided): Euler Steps on Curved vs Straight Paths**
- Create a simple 2D ODE with a curved solution trajectory and a straight solution trajectory.
- Apply Euler's method with N=5 steps to both. Plot the Euler approximation vs the true trajectory.
- Observe: Euler on the curved path drifts; Euler on the straight path is exact.
- Repeat with N=1 step. The straight path is still exact. The curved path is way off.
- What it tests: the connection between trajectory geometry and ODE solver accuracy. Depth: DEVELOPED for "why straight paths need fewer steps."

**Exercise 3 (Supported): Train a Flow Matching Model on 2D Data**
- Define a simple 2D target distribution (e.g., two-moons or concentric circles).
- Implement the flow matching training loop: sample x_0, sample epsilon, sample t, compute x_t and target velocity, MSE loss.
- Train a small MLP to predict velocity.
- Generate samples by solving the ODE (Euler's method) from noise.
- Compare: how many Euler steps are needed for good samples?
- Predict-before-run: "With 5 Euler steps, will the samples be recognizable?" (For a flow matching model on simple 2D data, yes -- the trajectories are nearly straight.)
- What it tests: end-to-end flow matching training and generation. Depth: DEVELOPED for the flow matching training loop, INTRODUCED for practical generation.

**Exercise 4 (Independent): Compare Flow Matching to DDPM on the Same Data**
- Use the same 2D distribution from Exercise 3.
- Train a DDPM model (noise prediction, cosine schedule) and a flow matching model (velocity prediction, linear interpolation).
- Generate samples from both at varying step counts (1, 5, 10, 20, 50 steps).
- Plot: sample quality vs number of steps for both models.
- Expected result: flow matching produces recognizable samples at fewer steps.
- Bonus: try one round of rectified flow on the flow matching model and compare.
- What it tests: the practical advantage of flow matching over DDPM in a controlled comparison. Depth: APPLIED for flow matching, REINFORCED for DDPM.

#### 14. Summarize

Key takeaways (echo mental models):

1. **Curved vs straight.** Diffusion ODE trajectories curve because the score field changes with noise level. Flow matching trajectories are straight by construction (linear interpolation between data and noise).
2. **Velocity prediction.** Flow matching trains a network to predict velocity (v = epsilon - x_0) instead of noise or score. Same kind of network, different target, straight trajectories.
3. **Simpler training.** No noise schedule, no alpha_bar, no variance-preserving formulation. Just a weighted average (x_t = (1-t)*x_0 + t*epsilon) and a subtraction (v = epsilon - x_0).
4. **Fewer steps.** On a straight path, Euler's method is exact in one step. Flow matching models generate high-quality samples in 20-30 steps where DDPM-style models need 50+.
5. **Same family.** Noise prediction, score prediction, and velocity prediction are three parameterizations of the same underlying vector field. Flow matching is a member of the diffusion family, not a replacement.

ModuleCompleteBlock: Module 7.2 "The Score-Based Perspective" is complete.
- Achievements: Score function as gradient of log probability, SDE/ODE duality, score-noise equivalence, flow matching, velocity prediction, rectified flow.
- Next module: 7.3 "Fast Generation" -- consistency models, latent consistency, SDXL Turbo.

#### 15. Next Step

"You now understand two ways to define the generation trajectory: the diffusion ODE (curved, score-guided) and flow matching (straight, velocity-guided). Both produce high-quality samples, but flow matching gets there in fewer steps. But what if you could collapse the ENTIRE trajectory into a SINGLE step? Not 20 steps, not 5 steps -- one step, directly from noise to data. That is the promise of consistency models."

---

## Review — 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings -- the student will not be lost or form wrong mental models. Three improvement findings that would meaningfully strengthen the lesson. Two minor polish items.

### Findings

#### [IMPROVEMENT] — Geometric/spatial modality for the core insight is textual, not visual

**Location:** Hook section (ComparisonRow, lines 157-177) and "Why Trajectories Curve" section (lines 194-268)
**Issue:** The planning document specified a geometric/spatial modality as the primary way to communicate the core insight: "side-by-side comparison of curved diffusion trajectories vs straight flow matching trajectories in 2D." The built lesson uses a ComparisonRow with bullet points (text) instead of any visual representation of the trajectories. The GPS analogy provides a verbal substitute, but the student never SEES a curved path next to a straight path in the lesson component itself.
**Student impact:** The student reads ABOUT curved vs straight trajectories rather than seeing them. The core geometric insight -- that diffusion paths visibly curve while flow matching paths are straight lines -- is the single most important visual in this lesson. Textual description weakens the impact. The notebook compensates (Exercises 1-2 produce the plots), but the lesson itself should establish the visual before practice.
**Suggested fix:** This is the same pattern that was accepted in the score-functions-and-sdes review (textual descriptions of 2D score fields, with notebook providing actual quiver plots). If the decision is to accept this pattern consistently (no custom visualization components, rely on notebooks for visuals), this is an acknowledged trade-off, not a bug. However, if a simple static diagram or even a Mafs-based interactive visualization of two trajectories (one curved, one straight) is feasible, it would significantly strengthen this lesson. At minimum, consider adding a brief note like "You will see this in Exercise 1 of the notebook" to bridge the gap.

#### [IMPROVEMENT] — Velocity-to-noise/score conversion formulas are vague

**Location:** Three Parameterizations section, conversion bullet points (lines 787-797)
**Issue:** The lesson states "From velocity to its components: v_t = epsilon - x_0, so with knowledge of either component you can recover the other." This is hand-wavy. The planning document specified showing explicit conversion formulas between all three parameterizations. The noise-to-score conversion is shown with a formula (line 790). The velocity-to-noise conversion is described in words but not given as a formula.
**Student impact:** The student understands that the three parameterizations are interconvertible in principle but cannot perform the conversion in practice. When reading a paper that says "we convert from v-prediction to epsilon-prediction," the student would not know the formula. The asymmetry between the noise-to-score conversion (given as a formula) and the velocity-to-noise conversion (described in words) is noticeable.
**Suggested fix:** Add the explicit conversion. Within the flow matching framework: since x_t = (1-t)*x_0 + t*epsilon and v = epsilon - x_0, the student can derive: epsilon = x_t + (1-t)*v and x_0 = x_t - t*v. These are simple rearrangements. Showing them concretely (perhaps as a small formula block after the existing bullet points) would close the gap without adding cognitive load, since the algebra follows directly from what was just taught.

#### [IMPROVEMENT] — Notebook missing pip install cell for dependencies

**Location:** Notebook setup cell (cell-2)
**Issue:** The setup cell imports `torch`, `numpy`, `matplotlib`, and `sklearn.datasets.make_moons` but does not include a `!pip install` cell. While PyTorch and NumPy are pre-installed in Google Colab, `scikit-learn` is sometimes present and sometimes not depending on the Colab runtime version. The notebook claims to be self-contained for Colab but could fail on the first import if sklearn is missing.
**Student impact:** If the student opens the notebook in Colab and sklearn is not available, they get an ImportError on the first cell and have to debug the environment before starting the exercises. This is a friction point that breaks the "low activation energy" principle.
**Suggested fix:** Add a pip install cell before the import cell: `!pip install -q torch numpy matplotlib scikit-learn`. The `-q` flag keeps the output minimal. This is standard practice for Colab notebooks and costs nothing.

#### [POLISH] — Rectified flow analogy could be stronger

**Location:** Rectified flow section, aside (lines 957-962)
**Issue:** The aside says "Rectified flow is to flow matching what distillation is to temperature -- it refines the process." This analogy is awkward. Distillation is not really "to temperature" in the way that rectified flow is to flow matching. The analogy does not map cleanly: distillation separates liquids by boiling point, while rectified flow straightens trajectories by re-pairing. The connection is too loose to be illuminating.
**Student impact:** Minor. The student might pause on this analogy and try to map it, finding the mapping does not hold. It does not cause confusion but it does not help either.
**Suggested fix:** Consider a simpler framing: "Rectified flow is a refinement step -- like retracing a hand-drawn line with a ruler. The first pass gives you the approximate shape; the second pass straightens it." Or simply drop the analogy and let the PhaseCards speak for themselves, since the three-step process is already clear.

#### [POLISH] — Comment em dashes use spaces

**Location:** Code comments in the lesson file (lines 36, 37, 194, 323, 1023, 1188)
**Issue:** Code comments (e.g., `{/* Section 5: Explain — Why Trajectories Curve */}`) use spaced em dashes. The writing style rule specifies no spaces around em dashes. While these comments are not rendered to the student, maintaining consistency throughout the file prevents accidental copy-paste into rendered text.
**Student impact:** None -- these are JSX comments, not rendered prose. Purely a codebase consistency issue.
**Suggested fix:** Replace ` — ` with `—` in code comments, or leave as-is since they are not rendered. Low priority.

### Review Notes

**Overall assessment:** This is a strong lesson. The narrative arc is compelling, the pacing is good, the worked example is excellent, and the misconceptions are addressed at the right points. The lesson delivers on its promise of being a BUILD-level payoff after the STRETCH lesson on score functions and SDEs. The student will come away understanding why flow matching replaced DDPM-style training in modern architectures.

**Pattern observed:** The geometric/spatial modality gap follows the same pattern as the previous lesson in this module (score-functions-and-sdes), where textual descriptions were accepted as adequate with the notebook providing actual visualizations. If this is an intentional pattern (lesson = text + formulas + worked examples; notebook = plots + code), it should be acknowledged as a deliberate design choice rather than a shortcoming. However, for this particular lesson, the curved-vs-straight trajectory visualization IS the core insight, making the textual-only treatment more costly than the score field descriptions in the previous lesson.

**What works well:**
- The "Of Course" chain is pedagogically excellent -- it gives the student the feeling they could have invented flow matching.
- The WarningBlock about "where is x_0 at inference?" preempts a natural confusion at exactly the right moment.
- The v-prediction vs flow matching distinction in Check #3 prepares the student for real-world paper terminology.
- The notebook is well-structured with clear TODO markers, solutions with reasoning, and a cumulative progression.
- The "Two Independent Changes" card cleanly separates architecture from training objective, preventing a conflation the student would inevitably make.

---

## Review — 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings. All three iteration 1 IMPROVEMENT findings have been addressed effectively. One new improvement finding identified in the notebook. Two minor polish items.

### Iteration 1 Fix Verification

**[FIXED] Geometric/spatial modality for the core insight.** The hook section now includes two side-by-side GradientCards with ASCII art trajectories (amber for curved diffusion paths, emerald for straight flow matching paths). The ASCII art effectively communicates the geometric difference--curved dots vs straight pipe characters. A bridging note ("You will see this difference plotted precisely in Exercises 1 and 2 of the notebook") connects the in-lesson representation to the notebook visualizations. This is a good solution within the project's established pattern (lesson = text/formulas/static diagrams, notebook = actual plots).

**[FIXED] Velocity-to-noise/score conversion formulas.** Lines 829-832 now show explicit BlockMath formulas: `epsilon = x_t + (1-t)*v` and `x_0 = x_t - t*v`. The derivation context is provided (lines 825-827: "since x_t = (1-t)*x_0 + t*epsilon and v = epsilon - x_0, rearranging gives...") and a practical interpretation follows (lines 833-838). The student can now actually perform the conversion when reading papers. Well done.

**[FIXED] Notebook pip install cell.** Cell-2 now contains `!pip install -q torch numpy matplotlib scikit-learn` before the import cell. Self-contained for Colab.

**[FIXED] Rectified flow analogy.** The aside (lines 997-1001) now reads "Rectified flow is like retracing a hand-drawn line with a ruler. The first pass gives you the approximate shape; the second pass straightens it. The first model is good; the rectified model is straighter and needs fewer steps." This maps cleanly: hand-drawn line = initial flow matching model, ruler-traced line = rectified model. Clear and illuminating.

### Findings

#### [IMPROVEMENT] — Notebook Exercise 3 training loop runs with None values

**Location:** Notebook cell-15 (Exercise 3 training loop)
**Issue:** The TODO markers set `t = None`, `x_t = None`, `target_v = None`, and `loss = None`. The training loop then immediately calls `pred_v = fm_model(x_t, t)` with `x_t = None`, which will raise a `TypeError` before the student has a chance to fill in the TODOs. In a well-scaffolded Supported exercise, the student should be able to run the cell to see what happens with the existing code before modifying it, or at minimum the error message should be informative. A `TypeError: linear(): argument 'input' (position 1) must be Tensor, not NoneType` is not helpful for understanding what went wrong.
**Student impact:** The student hits an uninformative Python error on first run. This is a minor friction point but could confuse a student who expects the cell to at least run (with a placeholder error or zero values). The solution block is available, but the error disrupts the flow. This is not a critical issue because the TODOs are clearly marked and the student is expected to fill them in before running.
**Suggested fix:** Two options: (1) Initialize the TODO variables with placeholder values that would run but produce wrong results (e.g., `t = torch.zeros(batch_size, 1)`, `x_t = x_0.clone()`, `target_v = torch.zeros_like(x_0)`, `loss = torch.tensor(0.0, requires_grad=True)`), so the student sees the training run but with wrong behavior, motivating them to fix it. (2) Add a comment above the TODOs: `# Fill in ALL four TODOs before running this cell. Running with None will raise an error.` Option 2 is simpler and sufficient.

#### [POLISH] — Notebook introduction uses triple-dash em dashes

**Location:** Notebook cell-0 (introduction markdown)
**Issue:** The introduction text uses `---` as em dashes in two places: "No new theory---just hands-on practice" and "Exercises 1--2 are pure math". The first is a triple dash (not standard markdown for em dash), and the second is a double dash (en dash convention). These render inconsistently depending on the Colab markdown renderer. In Jupyter/Colab, `---` becomes a horizontal rule if on its own line, but inline it renders as literal dashes. The writing style rule says em dashes should be `word—word` (Unicode em dash, no spaces).
**Student impact:** Minor visual inconsistency in the notebook introduction. The student sees three hyphens instead of an em dash. Does not affect understanding.
**Suggested fix:** Replace `---` with the Unicode em dash character in the notebook markdown cells where it appears inline.

#### [POLISH] — Notebook Exercise 2 plot title uses spaced em dash

**Location:** Notebook cell-12, plot titles
**Issue:** The plot titles use ` -- ` (spaced double hyphen) as separators: `'Curved Path (Spiral) -- {n_steps} Euler Step...'` and `'Straight Path (Line) -- {n_steps} Euler Step...'`. This is a code context (matplotlib title string) and these dashes render directly on the plot. The spaced double hyphen is an informal en dash.
**Student impact:** None. The plot titles are functional and clear. Purely a style consistency issue.
**Suggested fix:** Replace ` -- ` with ` | ` or use the Unicode em dash if maintaining the project's style convention in all rendered text.

### Review Notes

**Overall assessment:** The lesson has improved meaningfully from iteration 1. The four fixes all landed well, with the ASCII trajectory visualization and explicit conversion formulas being the most impactful changes. The lesson's pedagogical structure remains strong: motivation before explanation, concrete before abstract, clear narrative arc from problem (curved paths) to solution (straight paths) to consequence (fewer steps). The notebook is thorough with good scaffolding progression.

**Iteration 1 to 2 delta:** All three IMPROVEMENT findings resolved. The rectified flow analogy (POLISH) was also fixed. The only remaining POLISH item from iteration 1 (code comment em dashes) was deliberately left unfixed as cosmetic-only, which is the right call.

**New finding significance:** The notebook Exercise 3 `None` values issue is the only IMPROVEMENT finding. It is a scaffolding quality issue rather than a pedagogical one--the lesson content and exercises are sound, but the notebook could be slightly more student-friendly with either placeholder values or a clear "fill in before running" note. This is a quick fix.

**What works well (reaffirmed from iteration 1):**
- The "Of Course" chain remains the lesson's best pedagogical moment
- The WarningBlock about x_0 at inference is well-placed
- The v-prediction vs flow matching distinction in Check #3 is valuable for paper-reading
- The new ASCII trajectory visualizations add a genuine spatial modality to the hook
- The explicit conversion formulas close the paper-reading gap
- The "retracing a hand-drawn line with a ruler" analogy is much clearer than the original
- The notebook's cumulative structure (see difference -> understand why -> build it -> compare) mirrors the lesson arc perfectly

---

## Review — 2026-02-20 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

No critical or improvement findings. All iteration 2 findings have been addressed. One minor polish item remains (cosmetic only). The lesson is ready to ship.

### Iteration 2 Fix Verification

**[FIXED] Notebook Exercise 3 training loop runs with None values.** Cell-15 now has explicit `NotImplementedError` guards: `if t is None or x_t is None or target_v is None: raise NotImplementedError("Fill in the three TODOs above (t, x_t, target_v) before running this cell.")` and a second guard for the loss TODO. A comment at the top of the cell reads `# NOTE: Fill in ALL four TODOs before running this cell.` and `# Running with None values will raise an error.` The student now gets a clear, actionable error message that names the specific variables to fill in. This is the right solution--better than placeholder values because it forces the student to engage with all four TODOs before proceeding.

**[FIXED] Notebook introduction triple-dash em dashes.** Cell-0 now uses Unicode em dashes (`\u2014`) and en dashes (`\u2013`) throughout: "No new theory\u2014just hands-on practice" (em dash), "30\u201345 minutes" (en dash), "Exercises 1\u20132" (en dash), "Exercises 3\u20134" (en dash). These render correctly in Colab markdown.

**[FIXED] Notebook Exercise 2 plot title spaced em dash.** Cell-12 plot titles now use Unicode em dashes without spaces: `f'Curved Path (Spiral)\u2014{n_steps} Euler Step...'`. Clean and consistent.

### Findings

#### [POLISH] -- Notebook Exercise 3 comment uses triple-dash as em dash

**Location:** Notebook cell-15, Python code comment (line with `target_v` TODO)
**Issue:** The comment reads `# This is constant for each (x_0, epsilon) pair---it does not depend on t!` using triple hyphens as an inline em dash. This is a Python code comment, not rendered markdown or student-facing prose.
**Student impact:** None. The comment is read in code context where triple hyphens are unremarkable. It does not render in any output.
**Suggested fix:** Replace `---` with `--` or a Unicode em dash for consistency, or leave as-is. Zero priority.

### Review Notes

**Overall assessment:** This lesson is ready to ship. Across three review iterations, the lesson has been refined from a strong initial build into a polished, pedagogically sound teaching artifact. The core structure was sound from the beginning; the revisions addressed genuine gaps (geometric modality, conversion formulas, notebook scaffolding) without disrupting what worked.

**Cumulative improvement across iterations:**
- Iteration 1 (3 IMPROVEMENT, 2 POLISH): Added ASCII trajectory visualizations, explicit conversion formulas, pip install cell, better rectified flow analogy
- Iteration 2 (1 IMPROVEMENT, 2 POLISH): Added NotImplementedError guards, fixed em/en dashes in notebook markdown and plot titles
- Iteration 3 (0 IMPROVEMENT, 1 POLISH): All substantive issues resolved. One cosmetic code comment remains.

**Pedagogical strengths (final assessment):**
- The narrative arc is compelling: problem (curved trajectories) before solution (straight trajectories), culminating in the "Of Course" chain that makes flow matching feel inevitable
- The worked example with concrete numbers (Section "Worked Example: One Training Step") demystifies the training objective completely
- The WarningBlock about x_0 at inference preempts the most natural confusion at exactly the right moment
- The three-parameterization section with explicit conversion formulas prepares the student for real paper-reading
- The v-prediction vs flow matching distinction (Check #3, Question 2) addresses terminology confusion the student will inevitably encounter
- The "Two Independent Changes" card cleanly separates architecture from training objective, preventing a conflation that plagues many explanations of SD3/Flux
- The notebook mirrors the lesson arc perfectly (see difference, understand why, build it, compare head-to-head) with good scaffolding progression (Guided, Guided, Supported, Independent)
- The rectified flow section stays appropriately at INTRODUCED depth without overreaching

**The lesson delivers on its BUILD promise.** After the STRETCH lesson on score functions and SDEs, this lesson provides the payoff: the theoretical framework from the previous lesson makes flow matching feel like an obvious simplification. The student leaves understanding why modern architectures (SD3, Flux) use flow matching, how the training objective works, and how velocity, noise, and score prediction relate to each other.

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
- [x] Cognitive load: 2 new concepts (conditional flow matching, rectified flow)
- [x] Every new concept connected to at least one existing concept (flow matching to DDPM interpolation, velocity to noise/score, straight paths to Euler's method, rectified flow to higher-order solvers)
- [x] Scope boundaries explicitly stated
