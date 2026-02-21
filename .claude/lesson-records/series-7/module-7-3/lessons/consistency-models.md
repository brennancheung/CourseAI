# Lesson: Consistency Models

**Slug:** `consistency-models`
**Series:** 7 (Post-SD Advances), **Module:** 7.3 (Fast Generation), **Position:** Lesson 6 of 11 in series, Lesson 1 of 3 in module
**Cognitive Load:** STRETCH (2 genuinely new concepts: self-consistency property, consistency distillation)

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Probability flow ODE (deterministic trajectory from noise to data) | INTRODUCED | score-functions-and-sdes (7.2.1) | Deterministic version of the reverse SDE. Removes stochastic term, halves score coefficient. Connected to DDIM. Formula shown, coefficient change acknowledged. The student understands that generation follows a smooth, deterministic path from noise to data. |
| ODE trajectory as generation path | INTRODUCED | samplers-and-efficiency (6.4.2), deepened in score-functions-and-sdes (7.2.1) | Model predictions define a smooth trajectory. Samplers are ODE solvers following this trajectory. "The model defines where to go. The sampler defines how to get there." Formalized as the probability flow ODE in 7.2.1. |
| Euler's method as ODE solver | INTRODUCED | samplers-and-efficiency (6.4.2) | Compute direction at current point, take step of size h, repeat. First-order: one model evaluation per step. The student knows that accuracy depends on step size and trajectory curvature. |
| Flow matching (straight-line trajectories, velocity prediction) | DEVELOPED | flow-matching (7.2.2) | x_t = (1-t)*x_0 + t*epsilon. Velocity v = epsilon - x_0. Straight paths by construction. Euler's method is exact on a straight path in one step. The student trained a 2D flow matching model in the notebook. |
| Curved vs straight trajectories | DEVELOPED | flow-matching (7.2.2) | Diffusion ODE paths curve because the score field changes with noise level. Flow matching paths are straight by construction. Straighter paths need fewer ODE solver steps. GPS analogy (winding road vs straight highway). |
| DDPM noise prediction training | DEVELOPED | learning-to-denoise (6.2.3) | Loss = MSE between predicted and actual noise. Student has implemented the training loop. |
| DDPM forward process (variance-preserving noise addition) | DEVELOPED | the-forward-process (6.2.2) | x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon. Closed-form shortcut. Interactive widget experience. |
| Score function as gradient of log probability | DEVELOPED | score-functions-and-sdes (7.2.1) | nabla_x log p(x) = direction toward higher probability. "Compass toward likely data." 2D vector field visualization. |
| Score-noise equivalence | DEVELOPED | score-functions-and-sdes (7.2.1) | epsilon_theta approx -sqrt(1-alpha_bar_t)*score. The noise prediction IS a scaled score function. DDPM was always score-based. |
| Conversion between noise/score/velocity parameterizations | DEVELOPED | flow-matching (7.2.2) | Three parameterizations of the same vector field. Explicit formulas: epsilon = x_t + (1-t)*v, x_0 = x_t - t*v. "Three languages for the same idea." |
| Higher-order ODE solvers (DPM-Solver++) | INTRODUCED | samplers-and-efficiency (6.4.2) | Evaluate model at multiple nearby timesteps to estimate curvature, enabling larger accurate steps. DPM-Solver++ achieves 15-20 steps. |
| Rectified flow (iterative trajectory straightening) | INTRODUCED | flow-matching (7.2.2) | Generate (epsilon, x_0) pairs from a trained model, retrain on aligned pairs. Each round straightens aggregate trajectories. "Retracing a hand-drawn line with a ruler." |
| Knowledge distillation pattern (teacher-student) | MENTIONED | Not formally taught; LoRA (4.4.4) established "learning from a trained model" concept | The student knows LoRA as a way to adapt a pretrained model. The general pattern of a smaller/faster model learning from a larger/slower one has not been formally developed but the intuition is available. |

### Mental Models and Analogies Already Established

- **"Compass toward likely data"** -- the score function at every point tells you which direction to move for higher probability.
- **"The score was hiding inside DDPM all along"** -- noise prediction IS a scaled score function.
- **"Staircase to ramp"** -- DDPM (discrete) to SDE (continuous). Same trajectory, smoother description.
- **"Same landscape, different lens"** extended to three lenses -- diffusion SDE (stochastic, curved), probability flow ODE (deterministic, curved), flow matching ODE (deterministic, straight).
- **"GPS recalculating vs straight highway"** -- curved paths require constant recalculation; straight paths do not.
- **"Curved vs straight"** -- the core geometric insight of flow matching.
- **"The model defines where to go. The sampler defines how to get there."** -- model vs sampler separation.
- **"Predict and leap"** -- DDIM's predict-x0-then-jump mechanism.
- **"Of Course" chain** pattern -- used in ControlNet, IP-Adapter, and flow matching. Design insights that feel inevitable given the right framework.
- **"Symptom vs cause"** -- better ODE solvers treat the symptom (curvature); flow matching treats the cause (the trajectory itself).
- **"Three languages for the same idea"** -- noise, score, and velocity prediction as interconvertible parameterizations.

### What Was Explicitly NOT Covered

- Consistency models, self-consistency property, consistency distillation -- explicitly deferred as "Module 7.3" in flow-matching lesson
- Any method that bypasses the ODE trajectory entirely (all methods so far step along the trajectory)
- Knowledge distillation as a formal technique (only LoRA's "learn from a trained model" pattern exists)
- GAN training or adversarial losses (out of scope for all prior series)
- One-step or few-step generation from a single model call (the student's fastest method is DPM-Solver++ at 15-20 steps, or flow matching at 20-30 steps)

### Readiness Assessment

The student is well-prepared. They have:
1. The probability flow ODE as a deterministic trajectory from noise to data (INTRODUCED in 7.2.1)
2. Deep understanding that generation = following a trajectory with an ODE solver (from 6.4.2 and 7.2)
3. Euler's method and higher-order solvers (INTRODUCED in 6.4.2)
4. The "symptom vs cause" framework: better solvers treat the symptom of curvature, flow matching treats the cause -- now consistency models bypass the trajectory entirely (a natural third level)
5. The bridge from flow-matching that explicitly motivated this lesson: "What if you could collapse the ENTIRE trajectory into a SINGLE step?"

The student has the ODE trajectory picture firmly established, which is the essential prerequisite (as noted in the series plan: "Consistency models need the ODE view"). The main conceptual leap is from "follow the trajectory efficiently" to "bypass the trajectory entirely." This is genuinely new -- nothing in prior lessons has prepared the student for the idea of learning a function that jumps directly to the endpoint. The DDIM "predict and leap" mechanism is the closest analogue (predict x_0 then jump), but DDIM still requires multiple iterations of this predict-and-leap cycle. Consistency models learn to make a single leap from any noise level to the clean image.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to understand the self-consistency property of ODE trajectories (any point on the same trajectory maps to the same clean endpoint) and how this property can be used as a training objective to learn a direct mapping from any noise level to the clean image, bypassing multi-step ODE solving entirely.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Probability flow ODE (deterministic trajectory from noise to data) | INTRODUCED | INTRODUCED | score-functions-and-sdes (7.2.1) | OK | Need the student to understand that generation defines a smooth, deterministic trajectory. The self-consistency property is defined on this trajectory. The student has this from 7.2.1 where the probability flow ODE was formalized and connected to DDIM. |
| ODE trajectory concept (generation as following a path) | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) + score-functions-and-sdes (7.2.1) | OK | Need the student to visualize generation as a path from noise to data with a definite endpoint. The self-consistency property requires this picture. Student has it from both 6.4.2 (informal) and 7.2.1 (formalized). |
| Euler's method (compute direction, step, repeat) | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) | OK | Need to understand that ODE solving is iterative and error-prone on curved paths. Consistency models bypass this iteration. Also needed for understanding consistency training (adjacent timestep comparison). |
| ODE solver step count as the speed bottleneck | INTRODUCED | INTRODUCED | samplers-and-efficiency (6.4.2) + flow-matching (7.2.2) | OK | Need the student to feel the problem: even with better solvers and straighter paths, we still need multiple steps. This motivates "what if we bypassed the trajectory entirely?" |
| Flow matching straight-line trajectories | INTRODUCED | DEVELOPED | flow-matching (7.2.2) | OK | Need the context that even straight paths still require ODE steps (though fewer). Consistency models are the logical next question: "what if we did not need steps at all?" |
| DDIM predict-x0-then-jump mechanism | INTRODUCED | DEVELOPED | samplers-and-efficiency (6.4.2) | OK | Closest existing analogue to consistency models. DDIM predicts x_0 from x_t and leaps--but then has to iterate. Consistency models make the prediction so accurate that no iteration is needed. This comparison anchors the new concept to something familiar. |
| Score function / noise prediction (model outputs direction at every point) | INTRODUCED | DEVELOPED | score-functions-and-sdes (7.2.1), learning-to-denoise (6.2.3) | OK | Need the student to understand that the pretrained diffusion model defines the ODE trajectory. Consistency distillation uses this model as a teacher. |
| Knowledge distillation pattern | MENTIONED | MENTIONED | LoRA (4.4.4), informal | OK (small gap) | The concept of "a student network learning from a teacher network" has not been formally taught. However, the student knows LoRA as adapting a pretrained model, and the general idea of learning from an existing model is intuitive. A brief 2-3 paragraph introduction of distillation is sufficient. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Knowledge distillation (MENTIONED, need INTRODUCED) | Small | Brief recap section (2-3 paragraphs) introducing the teacher-student pattern: a pretrained (teacher) model generates targets, a new (student) model learns to match them. Connect to LoRA's pattern of "learning from a trained model." This is needed because consistency distillation is fundamentally a distillation technique -- the pretrained diffusion model is the teacher, the consistency model is the student. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Consistency models are just diffusion models with fewer ODE steps" | The student has seen a progression: 1000 steps (DDPM) -> 50 (DDIM) -> 15-20 (DPM-Solver++) -> 20-30 (flow matching). It is natural to think consistency models are the next point on this curve -- maybe 5 steps, then 1. | Consistency models do not run the ODE at all (in single-step mode). There is no "direction at this point, take a step" cycle. The network maps x_t directly to x_0 without any intermediate points. You cannot "pause" a consistency model mid-generation -- there is no intermediate trajectory to inspect. In contrast, you CAN pause a DDIM solver at step 3 of 10 and see a partially denoised image. | In the core explanation, immediately after introducing the self-consistency property. Explicit comparison: ODE solvers step along the trajectory; consistency models jump to the endpoint. |
| "The self-consistency property is a new mathematical discovery about ODEs" | The name "consistency models" and the "self-consistency property" sound like a mathematical breakthrough. | The self-consistency property is a trivial fact about deterministic ODEs that mathematicians have known for centuries: if the ODE is deterministic and you start from two different points on the same trajectory, you end at the same place. This is just what "deterministic" means. The insight is not the property itself -- it is USING this property as a training objective for a neural network. | Early in the explanation, when defining the self-consistency property. InsightBlock: "This is not a new mathematical result. It is a new way to USE an old mathematical result." |
| "Consistency training and consistency distillation produce the same quality" | Both use the same self-consistency property as the objective. The student might assume the two training approaches are interchangeable. | Consistency distillation uses a pretrained teacher to provide accurate estimates of where the trajectory goes. Consistency training must estimate the trajectory from scratch using only adjacent timesteps and a running average of the model. The teacher in distillation has been trained for millions of steps and knows the trajectory well; the consistency training model must discover it while training. In practice, consistency distillation converges faster and produces better samples, especially at 1-2 steps. The gap narrows with more training but does not fully close. | After presenting both approaches side by side. ComparisonRow with quality/training cost/dependency tradeoffs. |
| "One-step consistency model output is as good as 50-step diffusion output" | The "1-step generation" claim sounds like a free lunch. If you can generate in 1 step, why would anyone use 50 steps? | One-step consistency model output at 512x512 is noticeably softer and less detailed than 50-step DDPM output. The FID gap is significant (e.g., FID ~3.5 for consistency distillation vs ~2.0 for 50-step class-conditional diffusion on ImageNet 64x64 in the original paper). Multi-step consistency (2-4 steps) narrows this gap substantially. The tradeoff is real: 1 step for speed, more steps for quality. | In the "Multi-step Consistency" section. Honest about the quality-speed tradeoff with concrete comparisons. |
| "Consistency models make flow matching obsolete" | The student just learned flow matching enables fewer steps. Now consistency models enable 1 step. It seems like consistency models strictly dominate. | Flow matching and consistency models address different things. Flow matching makes trajectories straighter (better training, more stable). Consistency models bypass the trajectory at inference (faster generation). You can combine them: a flow matching model can serve as the teacher for consistency distillation, getting the benefits of both (stable training + fast inference). They are complementary, not competing. | In the Elaborate section, positioning consistency models within the speed landscape. Preview of Lesson 8. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| A single ODE trajectory with three highlighted points (high noise, medium noise, low noise) all mapping to the same clean image endpoint | Positive | Visualize the self-consistency property. The trajectory has a definite endpoint (the clean image x_0). Points at t=0.8, t=0.5, and t=0.2 all sit on this trajectory. If we had a function f that maps any of these points to the endpoint, we could skip the entire trajectory-following process. f(x_0.8, 0.8) = f(x_0.5, 0.5) = f(x_0.2, 0.2) = x_0. | The simplest possible visualization of what "self-consistency" means. The student needs to SEE multiple points on one trajectory converging to one endpoint before the concept makes sense. Connects directly to the ODE trajectory they have been working with since 6.4.2. |
| DDIM "predict-and-leap" vs consistency model "single leap" | Positive | Contrast the closest existing method with the new one. DDIM: predict x_0 from x_T, leap to x_T/2, predict x_0 again (different estimate!), leap to x_T/4, ... repeat. Consistency model: predict x_0 from x_T, done. DDIM's estimate of x_0 improves with each step because it gets closer to x_0 along the trajectory. The consistency model's estimate must be good enough in one shot. | Anchors the new concept to something the student already understands deeply (DDIM from 6.4.2). The comparison makes clear what consistency models gain (single call) and what they must overcome (accuracy of a single prediction from far away on the trajectory). Shows that consistency models are not DDIM with 1 step -- the training objective is fundamentally different. |
| Toy 2D consistency training: two points on the same trajectory should predict the same endpoint | Positive (stretch) | Make the training objective concrete with numbers. Take a pretrained 2D diffusion model's ODE. Pick a trajectory. Sample two points on it (x_t1 and x_t2). The consistency model should output the same value for both. The training loss penalizes the difference: L = distance(f_theta(x_t1, t1), f_theta(x_t2, t2)). | Demystifies the training objective by working through one example with concrete inputs and outputs. Parallels the flow matching worked example from the previous lesson. The student has trained 2D models in notebooks before (flow matching, DDPM), so extending to 2D consistency training is a natural next step. |
| Starting from OFF-trajectory noise (random noise not on any particular trajectory) | Negative | Show what the consistency model actually does at inference vs training. At inference, you start from pure Gaussian noise, which does not sit on any particular training trajectory. The consistency model must generalize: it maps ANY noisy point to a clean image, not just points that happen to lie on ODE trajectories it saw during training. This is why the consistency model is not just "memorizing trajectories" -- it must learn the general mapping from noise level to clean data. | Defines the boundary of the self-consistency analogy. The student might think the model only works for points exactly on a specific trajectory. In practice, the model handles arbitrary noisy inputs at any noise level. The self-consistency property is the training signal, not a runtime constraint. |

---

## Phase 3: Design

### Narrative Arc

The previous module ended with a challenge: "What if you could collapse the ENTIRE trajectory into a SINGLE step?" The student has now seen two strategies for faster generation. Better ODE solvers (DPM-Solver++, from 6.4.2) treat the symptom of curvature -- they follow the same curved trajectory but take smarter steps. Flow matching (from 7.2.2) treats the cause -- it replaces the curved trajectory with a straight one, so any solver can follow it in fewer steps. But both strategies share an assumption: generation MEANS following a trajectory. What if it did not?

Consider the probability flow ODE trajectory. It is deterministic -- from any starting point, there is exactly one path to the endpoint. This means something remarkable: every point on the same trajectory maps to the SAME clean image. The point at noise level 0.8 and the point at noise level 0.2 are on the same trajectory, heading to the same destination. What if a neural network could learn this mapping directly? Give it any point on ANY trajectory, and it outputs the endpoint. No stepping, no solver, no trajectory-following. Just one function evaluation: noise in, clean image out.

This is the idea behind consistency models (Song et al., 2023). The self-consistency property -- "any two points on the same ODE trajectory must map to the same clean image" -- becomes a training objective. If the model can learn to satisfy this constraint across all trajectories and all noise levels, it has learned to generate in a single step. The elegance is that this does not require designing a new generative process. It piggybacks on the ODE trajectory that a pretrained diffusion model already defines, and distills the entire trajectory into a single function call.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Geometric/Spatial** | ODE trajectory diagram with three highlighted points (x_t at t=0.8, t=0.5, t=0.2) all connected by the trajectory, all with arrows pointing to the same endpoint x_0. A second diagram showing the consistency function f mapping all three directly to x_0, bypassing the trajectory. Side-by-side with ODE solver stepping along the trajectory. | The self-consistency property IS a geometric statement about trajectories. The student needs to SEE that multiple points on the same trajectory converge to the same place. This extends the trajectory visualizations from 7.2.2 (curved vs straight) with a new operation: jumping from any point to the endpoint. |
| **Symbolic** | Three key equations: (1) Self-consistency constraint: f(x_t, t) = f(x_t', t') for all t, t' on the same ODE trajectory. (2) Boundary condition: f(x, epsilon) = x (near-clean image maps to itself, where epsilon is a small noise level near 0). (3) Consistency distillation loss: L = d(f_theta(x_{t_{n+1}}, t_{n+1}), f_theta_minus(hat_x_{t_n}, t_n)) where hat_x_{t_n} is the one-step ODE estimate from x_{t_{n+1}} to t_n, and f_theta_minus is the EMA target. | The formulas encode the training objective precisely. They come AFTER the geometric intuition so the student reads them as "of course, that is what the picture says in math." The boundary condition is the anchor: at noise level near 0, the trajectory endpoint IS the current point. |
| **Concrete example** | Toy 2D example: a pretrained diffusion model defines ODE trajectories on a 2D distribution (e.g., two-moons). Pick one trajectory. Sample x_t at t=0.7 and x_t at t=0.3. The consistency model should output the same clean point for both. Show the training loss as the distance between these two outputs. Work through with specific 2D coordinates. | The student has trained 2D models in notebooks for both DDPM (6.2) and flow matching (7.2.2). Extending to 2D consistency models is natural. The concrete numbers demystify the abstract "self-consistency" concept. |
| **Verbal/Analogy** | "Shortcut" vs "scenic route." ODE solving is the scenic route: you walk the full path from noise to data, stopping at every viewpoint (timestep). The consistency model learns a shortcut: given any viewpoint along the scenic route, it teleports you directly to the destination. You never need to walk the path. But learning the shortcut requires knowing where the path goes (which is why consistency distillation uses a pretrained model that already knows the path). | Extends the existing trajectory/path metaphor the student has from 6.4.2 and 7.2. The "scenic route" maps to ODE solving, the "shortcut" maps to the consistency function. The "knowing where the path goes" maps to the teacher model in distillation. Simple, memorable, and accurately represents the relationship between consistency models and the underlying ODE. |
| **Intuitive** | The "three levels" framework for speed: (1) Better solver -- follow the same trajectory more cleverly (DPM-Solver++). (2) Straighter trajectory -- change the trajectory shape so any solver works better (flow matching). (3) Bypass the trajectory -- learn to jump directly to the endpoint (consistency models). Each level is more radical than the previous. The student should feel the progression: "we tried smarter walking, then we straightened the road, now we are teleporting." | Organizes the entire speed narrative from Series 6 through Module 7.3 into a clean hierarchy. Makes the student feel where consistency models fit in the conceptual landscape. Echoes the "symptom vs cause" framework from flow matching, extended to a third level. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. Self-consistency property of ODE trajectories and its use as a training objective -- NEW (the property itself is trivially true of deterministic ODEs, but using it as a training signal for neural networks is a novel insight the student has not encountered)
  2. Consistency distillation (teacher-student pattern for consistency models) -- NEW (requires a brief introduction of the knowledge distillation pattern, which is a small gap)
  - Additionally, consistency training (without a teacher) is a variant of the same idea, not a separate concept
  - Multi-step consistency is an extension, not a new concept
- **Previous lesson load:** flow-matching was BUILD (2 concepts: conditional flow matching, rectified flow)
- **Previous previous lesson load:** score-functions-and-sdes was STRETCH (2-3 concepts: score function, SDE framework, probability flow ODE)
- **Is this appropriate?** STRETCH is appropriate but pushing the limit. The student has had a BUILD lesson as a buffer since the last STRETCH. The self-consistency property is conceptually novel--nothing in prior lessons directly prepares for "bypass the trajectory entirely." However, the concept is deeply grounded in the ODE trajectory the student already understands, and the training objective (two points on the same trajectory should produce the same output) is simple once the property is grasped. The "three levels" framing helps the student place this concept relative to what they know. Mitigated by: (a) connecting everything to the ODE trajectory from 7.2, (b) the DDIM predict-and-leap comparison as a familiar anchor, (c) concrete 2D examples throughout, (d) the capstone series tone (guided paper-reading, not hand-holding).

### Connections to Prior Concepts

| New Concept | Connects To | How |
|-------------|------------|-----|
| Self-consistency property (any point on the same trajectory maps to the same endpoint) | Probability flow ODE as deterministic trajectory (7.2.1) | The self-consistency property IS a statement about the probability flow ODE. Because the ODE is deterministic, every point on a trajectory has a unique endpoint. "You already know these trajectories are deterministic. Here is what that buys us." |
| Consistency function f(x_t, t) = x_0 | DDIM predict-x0-then-jump (6.4.2) | DDIM predicts x_0 from x_t at each step, then leaps. The consistency model is like DDIM's x_0 prediction, but trained to be accurate enough that you only need one prediction, not a sequence of improving predictions. "DDIM predicts x_0 and iterates. The consistency model predicts x_0 and stops." |
| Consistency distillation (learn from a pretrained model's trajectory) | LoRA as "learning from a trained model" (4.4.4) + "The model defines where to go" (6.4.2) | The pretrained diffusion model defines the ODE trajectory (it "knows where to go"). The consistency model learns to jump along this trajectory. Same pattern as LoRA: start with a powerful trained model, learn something specific on top of it. |
| "Three levels of speed" (better solver, straighter trajectory, bypass trajectory) | "Symptom vs cause" framework from flow-matching (7.2.2) | Extends the two-level framework (symptom = better solvers, cause = straighter paths) to three levels. Consistency models are the third level: bypass the trajectory entirely. |
| Consistency training loss (adjacent timesteps should agree) | MSE training losses from DDPM (6.2.3) and flow matching (7.2.2) | The structure is similar: sample a random point in time, compute a target, minimize a distance metric. The key difference is that the target comes from the MODEL'S OWN predictions at an adjacent timestep, not from the ground truth noise or velocity. |

### Analogies to Extend

- **"The model defines where to go. The sampler defines how to get there."** from 6.4.2 -- consistency models change this: the model defines where to go AND gets there in one step. The sampler is no longer needed (in single-step mode).
- **"Predict and leap"** from DDIM (6.4.2) -- the consistency model is the ultimate predict-and-leap: one prediction, one leap, done. No iteration.
- **"Symptom vs cause"** from flow-matching (7.2.2) -- extended to three levels: symptom (better solvers), cause (straighter paths), bypass (consistency models).
- **"Same landscape, different lens"** from 7.2.1 -- consistency models are yet another lens on the same landscape: instead of walking through it (SDE), driving through it (ODE), or straightening the road (flow matching), you teleport to the destination.

### Analogies That Could Be Misleading

- **"Predict and leap" (DDIM)** could mislead if the student thinks consistency models ARE DDIM with 1 step. DDIM with 1 step works poorly because the single-step x_0 estimate from far away (high noise) is inaccurate. The consistency model is trained specifically to make this single-step prediction accurate. The training objective is fundamentally different. Address by explicitly comparing DDIM's 1-step output (blurry, inaccurate) with a consistency model's 1-step output (coherent, though softer than multi-step).
- **"Shortcut" (teleportation)** could mislead if the student thinks the consistency model has memorized specific trajectories and can only teleport along those exact paths. In practice, the model generalizes to any noisy input at any noise level, including inputs not on any specific ODE trajectory it saw during training. The self-consistency property is the training signal, not a runtime constraint. Address with the negative example (off-trajectory inference).

### Scope Boundaries

**This lesson IS about:**
- The self-consistency property of deterministic ODE trajectories
- Using self-consistency as a training objective for a neural network
- Consistency training (learning from the ODE directly, no teacher)
- Consistency distillation (learning from a pretrained diffusion model's trajectory)
- The boundary condition f(x, epsilon) = x and the parameterization c_skip/c_out
- Multi-step consistency as a quality-speed tradeoff
- A toy 2D consistency training exercise in the notebook

**This lesson is NOT about:**
- Latent consistency models or LCM-LoRA (Lesson 7)
- Adversarial diffusion distillation / SDXL Turbo (Lesson 7)
- Full mathematical derivation of the consistency training loss (the student sees the training objective and why it works, not the proofs)
- The specific architecture modifications in the original consistency models paper (skip connections, EMA schedule details)
- Production-quality consistency model training (only a toy 2D exercise)
- Comparison of all acceleration approaches (Lesson 8)
- Implementation details of consistency model sampling schedules

**Depth targets:**
- Self-consistency property: DEVELOPED (geometric intuition + formal constraint + concrete example + training objective derivation)
- Consistency distillation: DEVELOPED (teacher-student pattern + training procedure + connection to pretrained models)
- Consistency training (without teacher): INTRODUCED (conceptual explanation, contrasted with distillation, but not worked through in detail)
- Multi-step consistency generation: INTRODUCED (the concept and why it helps, but not the scheduling details)
- Knowledge distillation pattern: INTRODUCED (brief gap fill, enough to understand consistency distillation)

---

### Lesson Outline

#### 1. Context + Constraints

- "This is the first lesson in Module 7.3: Fast Generation. The previous module gave you the theoretical tools -- score functions, ODE trajectories, flow matching. This module uses those tools to answer: how fast can we actually generate?"
- "We focus on the conceptual foundation of consistency models. The next lesson (7.3.2) applies this to real-world latent diffusion models (LCM, SDXL Turbo)."
- "The notebook includes a toy consistency training exercise on 2D data. We are not training production consistency models -- that requires the same GPU scale as diffusion model training."
- ConstraintBlock: scope boundaries. This lesson does not cover LCM or adversarial distillation (Lesson 7), the full speed landscape comparison (Lesson 8), or production training procedures. We focus on the self-consistency idea and how it becomes a training objective.

#### 2. Recap

Brief reactivation of three concepts:
- **Probability flow ODE** (from 7.2.1): 2-3 sentences. "Generation follows a deterministic trajectory from noise to data. The probability flow ODE defines this trajectory. Because it is deterministic, the same starting noise always leads to the same clean image."
- **DDIM predict-and-leap** (from 6.4.2): 2-3 sentences. "DDIM predicts the clean image x_0 from the current noisy image, then leaps toward x_0. But this prediction is approximate -- it gets better as you get closer to x_0. That is why DDIM still needs multiple steps."
- **The speed progression so far** (from 6.4.2 and 7.2.2): "Better solvers (DPM-Solver++, 15-20 steps), straighter paths (flow matching, 20-30 steps). Both still require STEPPING along a trajectory."
- **Knowledge distillation** (gap fill): 2-3 paragraphs introducing the teacher-student pattern. A large, slow model (teacher) generates examples or targets. A smaller/faster model (student) learns to match the teacher's outputs. The student model does not need to discover the solution from scratch -- it learns from the teacher's established knowledge. Connect to LoRA: "You have seen this pattern before -- LoRA adapts a pretrained model's knowledge for a specific task. Distillation is a broader version: a new model learns from a pretrained model's behavior."
- Transition: "The flow matching lesson ended with a question: 'What if you could collapse the ENTIRE trajectory into a SINGLE step?' Not 20 steps, not 5 steps -- one step. Let us find out how."

#### 3. Hook

Type: **Conceptual reveal + "three levels" framework**

"You have learned two strategies for faster generation. Think of them as levels."

Three-level GradientCards:
1. **Level 1: Smarter Walking** (amber) -- DPM-Solver++ follows the same curved path but takes larger, smarter steps. Result: 15-20 steps instead of 1000. "You are still walking the path."
2. **Level 2: Straighten the Road** (emerald) -- Flow matching replaces the curved path with a straight one. Euler's method is nearly exact. Result: 20-30 steps. "The road is straighter, so you walk faster."
3. **Level 3: ???** (violet) -- "What if you did not walk the path at all?"

"Level 3 is the idea behind consistency models. You do not step along the trajectory -- you jump directly to the endpoint."

#### 4. Explain: The Self-Consistency Property

**Part A: A trivial property of deterministic ODEs**

"Look at the probability flow ODE trajectory."

Trajectory diagram description: a single ODE trajectory from x_T (noise) to x_0 (clean data). Highlight three points: x_t at t=0.8, x_t at t=0.5, x_t at t=0.2. All three sit on the same trajectory, heading to the same endpoint x_0.

"Here is a fact so obvious it might not seem useful: all three of these points end up at the same place."

"Of course they do. The ODE is deterministic. If you start at any point on this trajectory and run the ODE forward to t=0, you always arrive at x_0. The point at t=0.8 and the point at t=0.2 are on the same trajectory -- they MUST have the same destination."

InsightBlock: "This is not a new mathematical result. It is the definition of a deterministic ODE. The insight is not the property -- it is what we can DO with it."

**Part B: The consistency function**

"Define a function f(x_t, t) that maps any point on a trajectory to its endpoint:"

f(x_t, t) = x_0 for all t, where x_t lies on the ODE trajectory ending at x_0

"The self-consistency constraint: for any two points on the same trajectory,"

f(x_t, t) = f(x_t', t') (for all t, t' on the same ODE trajectory)

"And the boundary condition:"

f(x, epsilon_min) = x (at noise level near 0, you are already at the endpoint)

"If a neural network could learn this function f, generation would be trivial: sample noise x_T, call f(x_T, T), get a clean image. One function evaluation. No ODE solver. No trajectory stepping."

Address Misconception #1: "This is NOT the same as running an ODE solver with 1 step. An ODE solver with 1 step computes the direction at x_T and takes a single (massive, inaccurate) step. The consistency model does not compute a direction -- it maps x_T directly to x_0. No direction, no step. A direct mapping."

ComparisonRow:
| | ODE Solver (1 step) | Consistency Model |
|---|---|---|
| What it computes | Direction at x_T (score/noise/velocity), then steps once | f(x_T, T) directly -- the endpoint |
| Accuracy | Terrible (one Euler step from pure noise) | Trained to be accurate |
| Training | No special training needed | Requires consistency training |
| Why it works | It does not (too few steps) | Trained specifically for single-step accuracy |

**Part C: DDIM comparison**

"This should remind you of DDIM's predict-and-leap. DDIM also predicts x_0 from x_t. But DDIM's prediction is approximate -- it uses the noise prediction epsilon_theta(x_t, t) to estimate x_0, and this estimate improves as t decreases (as you get closer to the clean image). That is why DDIM still needs multiple predict-and-leap iterations."

"The consistency model is trained specifically so that f(x_t, t) is accurate even when t is large (far from the clean image). DDIM predicts x_0 and iterates. The consistency model predicts x_0 and stops."

#### 5. Check #1

Three predict-and-verify questions:
1. "Two points x_t1 and x_t2 are on DIFFERENT ODE trajectories (heading to different clean images). Should f(x_t1, t1) equal f(x_t2, t2)?" (Answer: No. The self-consistency property only holds for points on the SAME trajectory. Different trajectories have different endpoints. f(x_t1, t1) and f(x_t2, t2) should be different clean images.)
2. "A colleague says: 'The consistency model just memorizes which noise maps to which clean image.' Is this correct?" (Answer: No. The model learns a continuous mapping from (noisy image, noise level) to clean image. It handles noise inputs it has never seen during training, including noise that does not lie on any specific ODE trajectory from the training data. The self-consistency property is the training signal -- at inference, the model generalizes.)
3. "What does the boundary condition f(x, epsilon_min) = x accomplish?" (Answer: It anchors the function at near-clean inputs. If x is already nearly clean, the model should not change it. This prevents the consistency function from "hallucinating" changes to already-clean images. It also provides a stable point that the rest of the function must be consistent with.)

#### 6. Explain: Consistency Distillation

**Part A: The teacher-student setup**

"How do you train a network to learn the consistency function? The self-consistency property says f should be constant along any ODE trajectory. But you need to know where the trajectory goes to enforce this constraint."

"Consistency distillation uses a pretrained diffusion model as the teacher. The teacher already knows the ODE trajectory -- its noise predictions define the vector field that the probability flow ODE follows."

Training procedure:
1. Sample a data point x_0, add noise to get x_{t_{n+1}} (a point at a higher noise level)
2. Use the pretrained teacher to estimate where x_{t_{n+1}} would be at the next lower noise level t_n (one ODE step: hat_x_{t_n} = ODE_step(x_{t_{n+1}}, t_{n+1}, t_n, teacher))
3. Train the consistency model so that f_theta(x_{t_{n+1}}, t_{n+1}) is close to f_{theta_minus}(hat_x_{t_n}, t_n)
4. theta_minus is an exponential moving average (EMA) of theta (stabilizes training, like the target network in DQN)

"In words: two points on the same trajectory (estimated by the teacher) should produce the same output from the consistency model."

Formula:
L_CD = d(f_theta(x_{t_{n+1}}, t_{n+1}), f_{theta_minus}(hat_x_{t_n}, t_n))

where d is a distance metric (e.g., L2 or LPIPS) and hat_x_{t_n} is the teacher's one-step ODE estimate.

**Part B: Why distillation works well**

"The teacher model has been trained for millions of steps. It knows where the ODE trajectory goes at every point. The consistency model learns from this established knowledge rather than discovering the trajectory from scratch. This is why consistency distillation converges faster and produces better results than consistency training."

Connect to the distillation pattern: "Same pattern as knowledge distillation in general: the teacher has expensive knowledge (a full ODE trajectory requiring many steps). The student learns a compressed version (a single-step mapping to the endpoint)."

**Part C: The EMA target (theta_minus)**

"The consistency model appears on BOTH sides of the loss: f_theta on the left (for the higher-noise point) and f_{theta_minus} on the right (for the teacher-estimated lower-noise point). If both sides used the same theta, the model could cheat by making f constant everywhere -- outputting the same image regardless of input."

"The EMA target theta_minus is a slowly-moving copy of theta. It provides a stable target that the model must match, preventing collapse."

TipBlock: "This EMA target pattern is common in self-supervised learning (e.g., momentum contrast, BYOL). The idea is always the same: when both sides of the loss are generated by the same model, make one side lag behind to prevent trivial solutions."

#### 7. Explain: Consistency Training (Without a Teacher)

**Part A: The core idea**

"What if you do not have a pretrained teacher model? Consistency training learns the consistency function directly, without a teacher providing the ODE trajectory."

"Instead of using a teacher's one-step ODE estimate, consistency training uses the model's OWN predictions at adjacent timesteps. The training signal comes from the model's self-consistency: adjacent timesteps should produce similar outputs, and this similarity improves as training progresses."

"The key difference: in consistency distillation, the teacher provides a good estimate of hat_x_{t_n} from the start. In consistency training, the estimate of hat_x_{t_n} starts poor and improves as the model trains. This makes consistency training slower and harder to converge."

**Part B: Comparison**

ComparisonRow:
| | Consistency Distillation | Consistency Training |
|---|---|---|
| Teacher model | Required (pretrained diffusion model) | Not required |
| Where trajectory info comes from | Teacher's ODE step estimates | Model's own predictions + discretization |
| Training speed | Faster (teacher provides good targets) | Slower (must discover trajectory from scratch) |
| Sample quality | Better, especially at 1-2 steps | Good, but lags behind distillation |
| Practical use | Most common in practice | Useful when no teacher is available |
| Dependency | Tied to a specific teacher model | Independent |

Address Misconception #3: "Consistency distillation and consistency training are NOT interchangeable. Distillation has a major advantage: the teacher already knows the trajectory. Consistency training must discover it while learning. In the original paper, consistency distillation achieves FID ~3.5 on ImageNet 64x64 while consistency training achieves FID ~6.2 at similar step counts."

#### 8. Check #2

Two predict-and-verify questions:
1. "In consistency distillation, the teacher model takes one ODE step from x_{t_{n+1}} to estimate hat_x_{t_n}. Why only one step? Why not run the full ODE to get the exact endpoint?" (Answer: Running the full ODE would be prohibitively expensive during training -- you would need 10-50 model evaluations PER training step. One ODE step is cheap (one teacher evaluation) and provides a reasonable estimate, especially between adjacent timesteps where the trajectory changes little. The consistency model learns to make these local consistency constraints imply global consistency.)
2. "If you had a flow matching model instead of a DDPM model as the teacher for consistency distillation, would you expect better or worse results? Why?" (Answer: Likely better. The flow matching model defines straighter ODE trajectories. Straighter trajectories mean the one-step ODE estimate hat_x_{t_n} is more accurate (less curvature error between adjacent timesteps). A more accurate estimate provides a better training signal for the consistency model. This is one reason flow matching and consistency distillation are complementary, not competing.)

#### 9. Explain: Multi-Step Consistency

**Part A: The quality-speed tradeoff**

"One-step consistency model generation is fast but not as good as multi-step diffusion. The single-step prediction from pure noise is a hard problem -- the model must map from a maximally uncertain input to a clean image."

"Multi-step consistency is a middle ground: run the consistency model at 2-4 noise levels instead of just 1. At each step, the model maps from the current noise level directly to x_0 (using the consistency function), then RE-NOISES x_0 to a lower noise level, and runs the consistency function again."

Multi-step procedure:
1. Start at x_T (pure noise)
2. Apply f(x_T, T) to get a clean estimate x_0_hat
3. Add noise to x_0_hat to get x_{t_2} at a lower noise level t_2
4. Apply f(x_{t_2}, t_2) to get a better clean estimate
5. Repeat for 1-3 more noise levels

"Each application of f starts from a LESS noisy input, so its prediction is more accurate. 2-4 steps of this process recovers much of the quality gap between 1-step consistency and 50-step diffusion."

Address Misconception #4: "One-step consistency models are NOT as good as 50-step diffusion. There is a real quality-speed tradeoff. But 4-step consistency often approaches 50-step diffusion quality -- this is still a massive speedup."

**Part B: How this differs from ODE solving**

"Multi-step consistency is NOT the same as running the ODE with 2-4 steps. In ODE solving, each step CONTINUES from where the previous step left off -- you walk along the trajectory. In multi-step consistency, each step RESTARTS: jump to x_0, re-noise, jump to x_0 again. Each step is an independent application of the consistency function, not a continuation of a trajectory."

#### 10. Check #3 (Transfer)

Two transfer questions:
1. "Based on the three levels framework (smarter walking, straighten the road, teleport), where does multi-step consistency fit? Is it Level 2 or Level 3?" (Answer: It is Level 3 with partial Level 2 flavor. Each individual step IS a teleportation -- the consistency function maps directly to x_0, not along a trajectory. But the re-noising and re-applying pattern gives it an iterative flavor. The key distinction: there is no trajectory being followed between steps. Each step is an independent jump to the endpoint. So it is Level 3 with multiple independent teleports for refinement, not Level 1 or 2 with fewer steps along a trajectory.)
2. "A colleague suggests: 'We should just distill any diffusion model into a consistency model and always use 1 step.' What would you say? What tradeoffs should they consider?" (Answer: Several tradeoffs: (a) 1-step quality is noticeably lower than multi-step -- for applications needing high fidelity (art, medical imaging), this matters. (b) Consistency distillation requires a teacher model and training time -- not free. (c) The consistency model is tied to the teacher's quality ceiling. (d) Multi-step consistency (2-4 steps) might be the sweet spot: much faster than 50-step diffusion, close in quality. (e) Flexibility: a consistency model is specialized for low-step generation; a regular diffusion model can generate at any step count.)

#### 11. Elaborate: Positioning Consistency Models

**Part A: The "three levels" revisited (complete)**

Level 3 GradientCard (violet): "Bypass the Trajectory"
- Consistency models learn a direct mapping: any noisy input to the clean endpoint
- No ODE solver needed (in single-step mode)
- Trained using the self-consistency property of the underlying ODE
- Quality-speed tradeoff: 1 step (fast, softer), 2-4 steps (fast, near-diffusion quality)

Address Misconception #5: "Flow matching and consistency models are complementary, not competing. Flow matching makes the trajectory straighter (better training signal, more stable convergence). Consistency models bypass the trajectory at inference (fewer steps needed). A flow matching model can serve as the teacher for consistency distillation, getting the best of both: stable training from straight trajectories, plus single-step inference from the consistency function. This combination is exactly what Latent Consistency Models use (next lesson)."

**Part B: What the consistency model actually learns**

"The consistency model learns a mapping from (noisy image, noise level) to clean image. This is similar to what the noise prediction network already does -- given a noisy input, estimate the clean image. The difference is in HOW it is trained. The noise prediction network is trained on MSE(epsilon_theta, epsilon). The consistency model is trained on the self-consistency constraint: adjacent points on the trajectory should agree."

"In fact, Song et al. show that consistency models can be parameterized using the same architecture as the original diffusion model, with a specific skip connection structure that enforces the boundary condition f(x, epsilon_min) = x."

#### 12. Practice (Notebook)

Design: 4 exercises, Guided -> Guided -> Supported -> Independent progression. The first three are cumulative (each builds on the previous); Exercise 4 is semi-independent.

**Exercise 1 (Guided): Visualize Self-Consistency on ODE Trajectories**
- Use a pretrained 2D diffusion model (provided, from the flow matching notebook or a simple DDPM model on two-moons)
- Generate multiple ODE trajectories by starting from different noise points and solving the ODE to completion
- Plot the trajectories. Pick one trajectory and highlight 5 points at different noise levels.
- Verify: running the ODE from each of these 5 points arrives at the same endpoint (within numerical tolerance)
- Predict-before-run: "If you start from the midpoint of a trajectory and run the ODE, will you reach the same endpoint as starting from the beginning?" (Answer: Yes -- this is the self-consistency property.)
- What it tests: the self-consistency property is a real, verifiable property of ODE trajectories, not just a mathematical abstraction. Depth: DEVELOPED for self-consistency.

**Exercise 2 (Guided): One-Step ODE vs Consistency Model Prediction**
- Using the same 2D model, compare: (a) the ODE endpoint starting from noise x_T, (b) a single Euler step from x_T (terrible approximation), (c) DDIM's 1-step x_0 prediction from x_T
- Plot all three alongside the true endpoint
- Predict-before-run: "How far off will the 1-step Euler step be from the true endpoint?" (Answer: Very far -- one Euler step from pure noise is massively inaccurate on a curved trajectory.)
- What it tests: why we need consistency models -- single-step ODE methods fail because the trajectory curves. A trained consistency model would map x_T directly to x_0 without this error. Depth: DEVELOPED for understanding why consistency models are needed.

**Exercise 3 (Supported): Train a Toy Consistency Model on 2D Data**
- Implement consistency distillation on the 2D two-moons dataset
- Use the pretrained 2D diffusion model from Exercise 1 as the teacher
- Training loop: sample x_0, add noise to get x_{t_{n+1}}, teacher takes one ODE step to estimate hat_x_{t_n}, minimize distance between f_theta(x_{t_{n+1}}, t_{n+1}) and f_{theta_minus}(hat_x_{t_n}, t_n)
- Generate samples using 1-step consistency (from pure noise)
- Compare the generated samples to the teacher's multi-step ODE samples
- What it tests: end-to-end consistency distillation training and single-step generation. Depth: APPLIED for consistency distillation, DEVELOPED for the training procedure.

**Exercise 4 (Independent): Multi-Step Consistency and Quality Comparison**
- Using the consistency model from Exercise 3, generate samples at 1, 2, 4, and 8 consistency steps
- Compare to the teacher model at 1, 5, 10, 20, 50 ODE steps
- Plot the quality progression for both (visual comparison of sample distributions)
- Expected result: consistency model at 2-4 steps approaches or matches the teacher at 20-50 steps
- Bonus: try consistency training (without a teacher) and compare convergence speed to distillation
- What it tests: the quality-speed tradeoff in practice, multi-step consistency vs ODE solving. Depth: APPLIED for multi-step consistency, REINFORCED for the quality-speed tradeoff.

#### 13. Summarize

Key takeaways (echo mental models):

1. **Self-consistency.** Any point on the same ODE trajectory maps to the same clean endpoint. This trivial property of deterministic ODEs becomes a powerful training objective: train a network so that f(x_t, t) is constant along any trajectory.
2. **Three levels of speed.** Better solvers (walk the path smarter). Straighter paths (straighten the road). Consistency models (bypass the road entirely -- teleport to the destination).
3. **Distillation vs training.** Consistency distillation uses a pretrained teacher to provide the trajectory. Consistency training discovers the trajectory on its own. Distillation is faster and better, but requires an existing model.
4. **Quality-speed tradeoff.** One-step consistency is fast but softer. Multi-step consistency (2-4 steps) recovers most of the quality. The right number of steps depends on the application.
5. **Complementary, not competing.** Flow matching makes trajectories straighter; consistency models bypass them. The combination (flow matching teacher + consistency distillation) gets the best of both.

#### 14. Next Step

"You now understand the consistency model idea: collapse an entire ODE trajectory into a single function evaluation. But we have been working with toy 2D data. The next lesson takes this to real scale: Latent Consistency Models (LCM) apply consistency distillation to Stable Diffusion and SDXL, enabling 1-4 step generation from existing checkpoints. We will also see a different approach to the same problem: adversarial diffusion distillation (ADD) in SDXL Turbo, which uses a GAN-style discriminator instead of self-consistency."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (knowledge distillation: small gap, 2-3 paragraph recap)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: geometric, symbolic, concrete example, verbal/analogy, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 2 new concepts (self-consistency property, consistency distillation)
- [x] Every new concept connected to at least one existing concept (self-consistency to probability flow ODE, consistency function to DDIM predict-and-leap, distillation to LoRA/teacher-student pattern, three levels to symptom-vs-cause framework)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: MAJOR REVISION

One critical finding (missing concrete 2D worked example -- a planned modality that was not built) and three improvement findings. The lesson is strong in narrative structure, motivation, misconception handling, and flow. The critical gap is that one of the five planned modalities is entirely absent, leaving the core concept without the concrete grounding that makes it click for this student.

### Findings

#### [CRITICAL] — Missing concrete 2D worked example for the self-consistency property

**Location:** Sections 5-6 (The Self-Consistency Property, The Consistency Function)
**Issue:** The planning document explicitly plans a "Concrete example" modality: "Toy 2D example: a pretrained diffusion model defines ODE trajectories on a 2D distribution (e.g., two-moons). Pick one trajectory. Sample x_t at t=0.7 and x_t at t=0.3. The consistency model should output the same clean point for both. Show the training loss as the distance between these two outputs. Work through with specific 2D coordinates." This modality is entirely absent from the built lesson. The lesson has a geometric/spatial modality (the ASCII trajectory diagram), a symbolic modality (the three equations), a verbal/analogy modality (teleportation, three levels), and an intuitive modality (the "of course" framing). But the concrete worked example with specific numbers -- the one that demystifies the abstract consistency constraint by showing it with real coordinates -- is missing.
**Student impact:** The student has trained 2D models in notebooks for both DDPM and flow matching. They expect concepts to be grounded in specific numbers. The self-consistency property is introduced with a trajectory diagram and formulas, but the student never sees something like: "Point A at t=0.7 has coordinates (1.3, -0.8). Point B at t=0.3 has coordinates (0.5, -0.3). Both are on the same trajectory. The consistency function should output f(1.3, -0.8, 0.7) = f(0.5, -0.3, 0.3) = (0.2, -0.1) -- the clean endpoint." Without this, the formulas remain abstract. The notebook does provide this verification in Exercise 1, but the lesson itself should ground the concept before the student reaches the notebook.
**Suggested fix:** Add a brief worked example between the trajectory diagram (GradientCard "One Trajectory, One Destination") and the formal equations. Use two specific 2D points on a trajectory, show their coordinates, and demonstrate that f maps both to the same endpoint. 4-5 sentences with specific numbers. This does not need to be a full exercise -- just a "for instance" with concrete values. Also show the training loss concretely: "The consistency distillation loss for this pair would be d(f(1.3, -0.8, 0.7), f(0.5, -0.3, 0.3)) -- the distance between the two predictions. If the model is well-trained, this distance is near zero."

---

#### [IMPROVEMENT] — Knowledge distillation recap lacks the depth promised in the plan

**Location:** Section 3 (Quick Recap), fourth paragraph on knowledge distillation
**Issue:** The planning document's gap resolution specifies: "Brief recap section (2-3 paragraphs) introducing the teacher-student pattern: a pretrained (teacher) model generates targets, a new (student) model learns to match them. Connect to LoRA's pattern of 'learning from a trained model.'" The built lesson provides only one paragraph (~4 sentences). The paragraph covers the core idea (teacher generates targets, student matches) and connects to LoRA, which is good. But it compresses what the plan calls a "2-3 paragraph" gap fill into a single paragraph. Knowledge distillation is only MENTIONED in the student's prior learning. Given that consistency distillation is one of the two core new concepts, and the entire training procedure section depends on the student understanding the teacher-student pattern, the gap fill deserves the planned depth.
**Student impact:** The student reaches the "Consistency Distillation" section (Section 7) having seen the word "distillation" once in the recap. The concept "the teacher provides the ODE trajectory and the student learns from it" is the backbone of the training procedure. If the student's grasp of distillation is shaky, the entire training section feels unmotivated -- they may not understand WHY a teacher is needed or what role it plays.
**Suggested fix:** Expand the distillation recap to 2-3 paragraphs as planned. Paragraph 1: the general pattern (teacher generates targets, student matches). Paragraph 2: why this works (the teacher has expensive knowledge that took millions of steps to acquire; the student gets a shortcut). Paragraph 3: connection to LoRA ("you have seen this pattern before"). This sets up the consistency distillation section to land with full force.

---

#### [IMPROVEMENT] — No negative example for the "memorization" misconception

**Location:** Check #1, Question 2
**Issue:** The planning document identifies a negative example: "Starting from OFF-trajectory noise (random noise not on any particular trajectory)... The consistency model must generalize: it maps ANY noisy point to a clean image, not just points that happen to lie on ODE trajectories it saw during training." This is addressed in Check #1 Question 2 as a comprehension question, but it is only in a hidden `<details>` answer. The lesson does not present this as a teaching moment in the main body. The student might form the misconception that the consistency model memorizes trajectory-endpoint pairs, and the only place this is corrected is inside a collapsed reveal block that many students will skip.
**Student impact:** A student who forms the "memorization" mental model and skips the reveal will carry this misconception into the distillation section. They might think the training procedure is about memorizing enough trajectories to cover the space, rather than learning a generalizable mapping. This weakens their understanding of why consistency models work on novel noise at inference time.
**Suggested fix:** Add 2-3 sentences in the main body (after the consistency function definition, before Check #1) explicitly addressing this. Something like: "At inference, you start from pure random noise -- not a point that lies on any specific ODE trajectory from the training data. The consistency model must generalize: the self-consistency property is the training signal, but at inference the model handles arbitrary noisy inputs it has never seen." This mirrors the planned negative example and ensures the misconception is addressed in the main flow, not just in an optional reveal.

---

#### [IMPROVEMENT] — Consistency distillation training procedure could show the "two points on the same trajectory" picture more visually

**Location:** Section 7 (The Training Procedure)
**Issue:** The training procedure is presented as a numbered list of 4 steps followed by the loss formula. This is accurate and complete, but the student has to mentally reconstruct what the procedure LOOKS LIKE geometrically. The planning document describes this as a geometric/spatial modality ("two points on the same trajectory should produce the same output from the consistency model"), but the built lesson only has the numbered steps and the formula. A small diagram or GradientCard showing the geometric picture -- two points on a trajectory, the teacher connecting them, the consistency model mapping both to the same output -- would significantly aid comprehension.
**Student impact:** The student has been learning through geometric trajectory pictures throughout this lesson and the prior two lessons. The training procedure abruptly shifts to a purely textual/symbolic presentation. The student must imagine the geometry themselves. For a STRETCH lesson, reducing the cognitive load of this key section matters.
**Suggested fix:** Add a GradientCard or ASCII diagram between the numbered steps and the loss formula showing: (a) x_{t_{n+1}} on the trajectory, (b) the teacher stepping to hat_x_{t_n}, (c) both being fed through the consistency model, (d) the loss comparing the two outputs. This mirrors the trajectory diagrams used earlier in the lesson for the self-consistency property. Label the arrows: "teacher ODE step" and "consistency model predictions should match."

---

#### [POLISH] — Notebook has spaced em dashes throughout

**Location:** Notebook `notebooks/7-3-1-consistency-models.ipynb`, throughout all markdown cells
**Issue:** The writing style rule specifies: "Em dashes must have no spaces: `word--word` not `word -- word`." The notebook has ~25 instances of spaced em dashes (` — `), for example: "any point on the same trajectory maps to the same clean endpoint — as a training objective", "ODE being deterministic — but it becomes a powerful training objective", "just hands-on practice with the math and models."
**Student impact:** Minor inconsistency with lesson prose style. Not pedagogically harmful.
**Suggested fix:** Find-and-replace ` — ` with `—` throughout the notebook markdown cells.

---

#### [POLISH] — "Samplers and Efficiency" lesson name in recap does not match curriculum

**Location:** Section 3 (Quick Recap), second paragraph
**Issue:** The recap refers to "Samplers and Efficiency" as the source lesson for DDIM predict-and-leap. This should be verified against the actual lesson slug and title. The module record refers to it as lesson from module 6.4.2 with slug "samplers-and-efficiency", which matches. However, the bold text says "Samplers and Efficiency" while the first recap paragraph says "Score Functions & SDEs" with an ampersand. Minor inconsistency in naming convention (ampersand vs "and").
**Student impact:** Negligible. The student will recognize the lesson either way.
**Suggested fix:** Use consistent formatting for lesson name references throughout the recap (either always use "and" or always use "&").

### Review Notes

**What works well:**

1. **Narrative arc is excellent.** The "three levels of speed" framework is a genuinely satisfying conceptual organizer. It connects to the "symptom vs cause" framework from flow matching and extends it naturally. The student feels the progression from "smarter walking" to "straighten the road" to "teleport."

2. **Misconception handling is strong.** The lesson addresses 4 of 5 planned misconceptions in the main body: "not the same as 1-step ODE" (ComparisonRow + WarningBlock), "not a new mathematical result" (InsightBlock), "distillation vs training are not interchangeable" (ComparisonRow with FID numbers), "not as good as 50-step diffusion" (WarningBlock with quality-speed tradeoff). The fifth (flow matching vs consistency as complementary) is addressed in the Elaborate section.

3. **Connection to prior concepts is well-executed.** The DDIM predict-and-leap comparison is particularly effective -- "DDIM predicts x_0 and iterates. The consistency model predicts x_0 and stops." This is exactly the right anchor for a student who has DDIM at DEVELOPED depth.

4. **The notebook is well-structured.** Four exercises with appropriate scaffolding progression (Guided, Guided, Supported, Independent). Exercise 3 has proper `NotImplementedError` guards and detailed solution blocks. The notebook uses the same terminology as the lesson and does not introduce new concepts.

5. **Scope discipline is good.** The lesson stays within its boundaries. It does not drift into LCM, adversarial distillation, or architecture details. The next-step block clearly signals what comes next.

**Pattern to watch:** The lesson is text-heavy in the training procedure sections (Sections 7-8). The earlier sections have good visual/geometric grounding (trajectory diagrams, GradientCards, ComparisonRows), but the distillation and consistency training sections rely more on prose and formulas. Adding one geometric diagram to the training procedure (the critical finding's suggested fix overlaps with this) would balance the lesson's modality mix.

---

## Review — 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All six iteration 1 findings have been properly resolved. The fixes are well-integrated into the lesson flow without introducing new issues. One minor polish item noted below.

### Iteration 1 Fix Verification

| Finding | Severity | Status | Notes |
|---------|----------|--------|-------|
| Missing concrete 2D worked example | CRITICAL | RESOLVED | GradientCard "Worked Example: Two Points, One Endpoint" added between the trajectory diagram and the formal equations. Uses specific 2D coordinates (1.3, -0.8) at t=0.7 and (0.5, -0.3) at t=0.3 mapping to endpoint (0.2, -0.1). Also includes a concrete preview of the training loss. Transition "To make this concrete" reads naturally. The forward-reference to "consistency distillation training loss" is acceptable because it is signaled with "Later" and the concept of "distance between two predictions near zero" is self-explanatory at this point. |
| Knowledge distillation recap too thin | IMPROVEMENT | RESOLVED | Expanded from 1 paragraph to 3 paragraphs. Paragraph 1: the general pattern (teacher generates targets, student matches) with a diffusion-specific example. Paragraph 2: why this works (teacher has expensive knowledge baked into weights, student gets a shortcut). Paragraph 3: connection to LoRA ("you have seen this pattern before"). The depth is appropriate for bridging MENTIONED to INTRODUCED. |
| No negative example for memorization misconception | IMPROVEMENT | RESOLVED | Added a dedicated paragraph in the main body (lines 479-496) between the DDIM comparison and Check #1. Explicitly addresses: "at inference time, you start from pure random noise -- not a point that lies on any specific ODE trajectory from the training data." Uses the word "generalize" and distinguishes training signal from runtime behavior. The misconception is now addressed in the main flow, not just in an optional reveal. |
| Training procedure lacks geometric diagram | IMPROVEMENT | RESOLVED | ASCII diagram "The Distillation Picture" added in a blue GradientCard between the numbered steps and the loss formula. Shows the ODE trajectory, teacher step connecting two points, consistency model predictions on both sides, and the "should match" loss. The diagram is well-labeled and the accompanying text paragraph explains each arrow. |
| Notebook spaced em dashes | POLISH | RESOLVED | No spaced em dashes remain in the notebook. All ~25 instances have been corrected. |
| Lesson name inconsistency in recap | POLISH | RESOLVED | Both lesson references now use "and" consistently ("Score Functions and SDEs", "Samplers and Efficiency"). |

### Findings

#### [POLISH] -- Notebook Exercise 4 re-noising uses flow matching interpolation without explicit comment

**Location:** Notebook Exercise 4, solution code, multi-step consistency sampling function
**Issue:** In the solution's `sample_consistency_multistep` function, the re-noising step uses `x = (1 - t_next) * x_0_hat + t_next * fresh_noise`. This is the flow matching interpolation formula, which makes sense because the teacher is a flow matching model. However, the lesson's multi-step consistency section (lines 905-906) describes re-noising as "Add noise to x_0_hat to get x_{t_2} at a lower noise level t_2" without specifying whether this uses the DDPM or flow matching noise schedule. The solution code silently uses the flow matching formula. A one-line comment in the solution noting "We use the flow matching interpolation for re-noising because the teacher was trained with flow matching" would prevent confusion for a student who might try the DDPM formula instead.
**Student impact:** A student who implements the multi-step sampling from scratch might use `sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise` (the DDPM formula) instead of the flow matching interpolation. This would produce wrong results with no obvious error. The issue is subtle enough that the student could spend significant time debugging.
**Suggested fix:** Add a one-line comment in the solution code: `# Re-noise using flow matching interpolation (matching the teacher's training)`

### Review Notes

**What was well-executed in the fixes:**

1. **The worked example integration is seamless.** The transition "To make this concrete, imagine a 2D diffusion model trained on the two-moons distribution" flows naturally from the abstract "of course they do" paragraph. The coordinates are specific enough to be useful but simple enough not to overwhelm. The forward-reference to the training loss is a nice touch that seeds the distillation section.

2. **The distillation recap has the right depth.** Three paragraphs is the right length. The second paragraph ("Why does this work?") is particularly effective -- it connects the teacher's "expensive knowledge" to the specific context of ODE trajectories, making the general distillation pattern concrete for this lesson's use case.

3. **The memorization clarification is well-placed.** Positioned after the DDIM comparison and before Check #1, it catches the student at exactly the moment they might form the misconception. The paragraph is concise (6 lines) and uses strong language ("must generalize") that makes the point clearly.

4. **The distillation diagram complements rather than duplicates.** The ASCII diagram shows the geometric picture (two points on a trajectory, teacher step, predictions should match) while the surrounding text provides the symbolic details. This is the right modality balance for this section.

**Overall assessment:** The lesson is now pedagogically sound. The narrative arc is strong, all five planned modalities are present for the core concept, all five planned misconceptions are addressed in the main body, the notebook exercises have appropriate scaffolding, and the lesson stays within its stated scope. The one remaining polish item is minor and can be fixed without re-review.
