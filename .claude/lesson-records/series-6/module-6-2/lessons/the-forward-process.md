# Lesson: The Forward Process

**Module:** 6.2 (Diffusion)
**Position:** Lesson 6 of Series 6 (second lesson of Module 6.2)
**Slug:** `the-forward-process`

---

## Phase 1: Student State (Orient)

The student has just completed "The Diffusion Idea" (Lesson 5), a BUILD lesson that established forward and reverse process intuition without any math. They understand that destruction (adding noise) is easy, creation from scratch is impossibly hard, but undoing a small step of destruction is learnable. They have interacted with a noise slider widget and seen images dissolve into static. They know the forward process adds Gaussian noise step by step and the reverse process uses a neural network to undo it. But they have zero formalization -- no formulas, no noise schedules, no alpha notation. They are primed for "the math behind what you just saw."

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Forward process (gradual noise destruction of images) | INTRODUCED | the-diffusion-idea | Knows the concept: add noise step by step until the image becomes pure static. Has seen it in the widget (slider from clean to noise). Has NOT seen the math -- no alpha, no beta, no schedule formula. |
| Reverse process (learned iterative denoising) | INTRODUCED | the-diffusion-idea | Knows a neural network learns to remove a small amount of noise at each step. Chaining ~1,000 steps produces generation. No math, no sampling algorithm. |
| "Small steps make it learnable" (core diffusion insight) | DEVELOPED | the-diffusion-idea | Can explain in own words why breaking generation into many small denoising steps works. This is the student's primary mental model entering this lesson. |
| Multi-scale denoising progression (coarse-to-fine) | INTRODUCED | the-diffusion-idea | High noise = structural decisions, medium = refine, low = fine details. Connected to CNN feature hierarchy. |
| Gaussian distribution / N(0,1) | DEVELOPED | variational-autoencoders | Used N(0,1) for VAE sampling and reparameterization. Knows "draw random values from a bell curve centered at zero." Can write the code. Has NOT seen properties like addition of independent Gaussians or variance manipulation. |
| The reparameterization trick (z = mu + sigma * epsilon) | INTRODUCED | variational-autoencoders | Knows the formula: z = mu + sigma * epsilon, where epsilon ~ N(0,1). Knows it separates randomness from learnable parameters so gradients can flow. The diffusion forward process formula has the same structural pattern. |
| Reconstruction loss (MSE) | DEVELOPED | autoencoders | Has trained with MSE loss. Knows the formula, can implement it. Will become the diffusion training loss in Lesson 7. |
| Image manifold intuition | MENTIONED | the-diffusion-idea | Brief geometric framing: real images on a thin manifold, noise pushes off, denoising walks back. Two-three sentences, not developed. |
| "Same building blocks, different question" mental model | DEVELOPED | from-classification-to-generation, the-diffusion-idea | Extended to diffusion: conv layers, MSE loss, backprop -- same tools, different question ("what noise was added?"). |
| Neural network training loop | APPLIED | Series 1-2 | Deeply familiar. Forward pass, loss, backward, update. Has built many. |

### Mental Models and Analogies Already Established

- "Destruction is easy; creation from scratch is impossibly hard; but undoing a small step of destruction is learnable."
- Ink drop in water (forward process -- physical diffusion, gradual, irreversible without information)
- Sculpting from marble (reverse process -- iterative refinement, coarse to fine)
- Jigsaw puzzle in a tornado (why one-shot creation from disorder fails)
- Image manifold: noise pushes off a thin surface, denoising walks back (brief, geometric)
- "Same building blocks, different question" -- paradigm shift is in the objective
- Reparameterization trick as "separating randomness from learnable parameters"

### What Was Explicitly NOT Covered

- Mathematical formulation of the forward process (noise schedules, alpha, beta, alpha_bar)
- The closed-form shortcut q(x_t|x_0)
- Variance-preserving vs variance-exploding formulations
- Why Gaussian noise specifically (mathematical properties)
- Any formulas or equations for noise addition
- Training objective or loss function (Lesson 7)
- Sampling algorithm (Lesson 8)
- Code or implementation (Lesson 9)

### Readiness Assessment

The student is prepared but this will be genuinely hard. They have strong conceptual intuition from Lesson 5 and practical experience with Gaussians from VAE work, but they have not done mathematical reasoning about noise properties since Series 1. The reparameterization trick formula (z = mu + sigma * epsilon) is the closest structural pattern to the diffusion formula they will see here. The key risk is cognitive overload: this lesson must introduce Gaussian noise properties, the beta noise schedule, the alpha/alpha_bar notation, the variance-preserving constraint, AND the closed-form shortcut. Careful scaffolding is essential -- each piece must be motivated before it appears, and the closed-form shortcut must feel like a revelation (elegant simplification) rather than another formula to memorize.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to **derive and use the closed-form formula q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon that lets you jump to any noise level in one step, and to understand why each design choice in the forward process (Gaussian noise, variance-preserving formulation, noise schedule) exists.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Forward process as gradual noise addition | INTRODUCED | INTRODUCED | the-diffusion-idea | OK | Student needs the intuition (add noise step by step) to motivate formalizing it. Has this from the widget and analogies. |
| Gaussian distribution / N(0,1) | DEVELOPED | DEVELOPED | variational-autoencoders | OK | Must be comfortable drawing from N(0,1) and knowing what it means. Has coded this for VAE. Need to extend to properties (addition, scaling) but the foundation is solid. |
| Reparameterization trick (z = mu + sigma * epsilon) | INTRODUCED | INTRODUCED | variational-autoencoders | OK | The closed-form formula has the same structural pattern (signal + noise_scale * epsilon). The student recognizing this pattern is a key pedagogical moment. Only INTRODUCED depth needed -- they need to recognize the pattern, not re-derive it. |
| Addition of independent Gaussians | INTRODUCED | MISSING | -- | GAP | The derivation of the closed-form shortcut requires knowing that adding two independent Gaussian random variables gives another Gaussian, and that variances add. This was never taught. The student knows what N(0,1) is and can sample from it, but has never seen the property that N(a, sigma1^2) + N(b, sigma2^2) = N(a+b, sigma1^2 + sigma2^2). |
| Variance scaling under multiplication | INTRODUCED | MISSING | -- | GAP | If X ~ N(0, 1), then c*X ~ N(0, c^2). This is needed for the variance-preserving derivation. The student has used sigma * epsilon in the reparameterization trick but was never told that multiplying a Gaussian by c scales its variance by c^2. |
| "Small steps make it learnable" | DEVELOPED | DEVELOPED | the-diffusion-idea | OK | The core motivation for why we define a gradual process at all. Student has this at the right depth. |
| Image manifold intuition | MENTIONED | MENTIONED | the-diffusion-idea | OK | Brief spatial framing. Only MENTIONED depth needed here -- the manifold idea contextualizes why we want images to stay "close" at each step but is not central to the math. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Addition of independent Gaussians | Small (student has N(0,1) practically, needs one property) | Brief dedicated subsection (~3 paragraphs + visual) within this lesson. Frame it as: "You know how to sample from a Gaussian. Here is one property you need: if you add two independent Gaussian samples, the result is also Gaussian, and the variances add." Visual: two narrow bell curves combining into one wider bell curve. Concrete numerical example with actual numbers. This is a tool, not a concept -- present it as a fact the student can verify, not a theorem to prove. |
| Variance scaling under multiplication | Small (student has used sigma * epsilon, needs the principle stated) | Combined with the addition property above in the same subsection. "If epsilon ~ N(0,1), then c * epsilon ~ N(0, c^2). You actually used this in the reparameterization trick: sigma * epsilon gave you a sample with variance sigma^2." One paragraph + connect to existing knowledge. |

Both gaps are small -- the student has the practical experience (sampling, using the reparameterization trick) but lacks the stated property. A brief "Gaussian properties you need" section early in the lesson resolves both.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The noise added at each step is the same amount" | The forward process was described as "add noise at each step" without specifying that the amount varies. The widget in Lesson 5 appeared to add noise uniformly. Natural to assume uniform noise addition. | Show images at early vs late timesteps with the same step size. Early steps (low beta): the image barely changes. Late steps (higher beta): the image degrades faster. If noise were uniform, the image would degrade linearly, but it does not -- the schedule controls the rate. Plot the beta schedule and show it is not flat. | When introducing the noise schedule (beta_t). Show the schedule curve and connect it to what the student saw in the widget. |
| "You need to iterate through all steps to get to timestep t" | The forward process is defined recursively (each step depends on the previous one). Natural to assume you must compute all intermediate steps. The student saw the process as sequential in the widget. | The closed-form formula q(x_t\|x_0) jumps directly to any timestep. You can compute the image at step 500 without computing steps 1-499. This is the key practical insight that makes training feasible -- you sample a random t and jump straight there. | The entire second half of the lesson builds toward this revelation. Address the misconception explicitly right before introducing the closed-form formula: "You might think we need to iterate 500 times. We do not." |
| "The image gets bigger/louder as you add noise (variance explodes)" | Adding random noise to something should make it grow in magnitude. If you keep adding noise to a signal, the combined signal+noise should have ever-increasing variance. This is physically intuitive -- piling more stuff on top of something makes it bigger. | The variance-preserving formulation deliberately prevents this. At each step, the original signal is scaled DOWN before noise is added. The coefficients are designed so that the total variance stays at 1 throughout the process. Show the formula: sqrt(1-beta_t) * x_{t-1} + sqrt(beta_t) * epsilon. The two coefficients are designed so (1-beta_t) + beta_t = 1, keeping variance constant. | When introducing the variance-preserving formulation. This is a key design insight, not just a mathematical convenience. |
| "alpha_bar is just another arbitrary Greek letter / the notation is unnecessarily complex" | The student encounters alpha_t = 1 - beta_t, then alpha_bar_t = product of all alphas. This feels like unnecessary layers of abstraction. Why not just use beta? The notation soup (beta, alpha, alpha_bar) is intimidating. | alpha_bar is the punchline -- it is the one number that tells you the signal-to-noise ratio at any timestep. Without alpha_bar, you need the entire history of betas. With it, you need one number. The closed-form formula only needs alpha_bar. The apparent complexity dissolves once you see why each name exists. | Introduce alpha as "let us name this for convenience" and alpha_bar as "this is the number that matters." Frame the notation as progressive simplification, not progressive complexity. |
| "The noise schedule is a minor implementation detail, not a design choice" | The lesson focuses on the mathematical formulation. The schedule (linear, cosine, etc.) seems like just a hyperparameter. How hard can choosing a schedule be? | The schedule fundamentally shapes what the model learns. A linear schedule destroys information too quickly at early timesteps, leaving the model with very few "easy" denoising steps. A cosine schedule spends more time at low noise levels, giving the model more practice with fine details. The choice of schedule visibly affects generated image quality. Show images at the same timestep under different schedules -- the difference is stark. | After introducing the linear beta schedule, briefly contrast with cosine schedule. Connect to the student's widget experience in Lesson 5 (which used a cosine schedule). |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Adding noise to a single pixel value (1D numerical walkthrough) | Positive | Demonstrate the forward process math step by step on the simplest possible case. Start with a pixel value of 0.8. Apply three steps of noise with specific beta values. Show exactly how the pixel degrades. Then show the closed-form shortcut gives the same result. | Minimally complex: one number, not an image. The student can verify each step with mental math. The 1D case makes the formula concrete before generalizing to images. Also demonstrates that the closed-form shortcut is not approximate -- it gives exactly the same result as iterating step by step. |
| Noise progression on a 28x28 image at specific timesteps | Positive | Show the forward process on a real image. Compare iterative step-by-step result with closed-form jump at t=250, t=500, t=750. Visually identical results confirm the formula works on actual images, not just single numbers. | Extends from 1D to the full image case. The student sees that the math they worked through on a single pixel applies identically to every pixel independently. The visual comparison (step-by-step vs closed-form) is the experiential proof that the shortcut works. This connects to the widget from Lesson 5 -- same images, now with the math behind them. |
| Variance-exploding negative example: "just add noise without scaling" | Negative | Show what happens if you do the naive thing: x_t = x_{t-1} + sqrt(beta_t) * epsilon, without scaling x_{t-1}. The variance grows without bound. After 100 steps, pixel values are wildly out of range. The image does not converge to N(0,1) -- it explodes. | Motivates the variance-preserving formulation by showing the failure mode it prevents. The student sees concretely why scaling the signal down is not optional. Also grounds the "variance-preserving" name -- it is literally about preserving (bounded) variance. |
| Comparing linear vs cosine noise schedules on the same image | Positive (stretch) | Show the same image at t=250 under a linear schedule vs a cosine schedule. The linear schedule has already destroyed too much information; the cosine schedule preserves more detail at the same timestep. Plot both schedules as curves (alpha_bar vs t) on the same axes. | Makes the noise schedule a design choice the student can reason about, not an arbitrary parameter. Connects to the Lesson 5 widget, which used a cosine schedule. Builds intuition that will carry through to the implementation lesson (Lesson 9). |

### Gap Resolution

Both gaps (Gaussian addition property, variance scaling) resolved via a dedicated "Two Gaussian Properties You Need" section early in the lesson. See detailed plan above.

---

## Phase 3: Design

### Narrative Arc

The student knows WHAT the forward process does -- they have watched images dissolve into noise and felt the intuition of why small steps make generation learnable. But they have no idea HOW it works mathematically. Right now, "add noise at each step" is a vague instruction. How much noise? The same amount every time? What does "step 300" actually mean, numerically? And here is the practical problem: if training requires jumping to a random timestep (which the student was briefly told in Lesson 5), and the forward process is defined step by step, do you really need to compute all 300 previous steps just to get a training sample at step 300? That would make training painfully slow. This lesson answers all of these questions, building from simple Gaussian properties through the noise schedule to an elegant mathematical shortcut: a single formula that lets you jump to ANY noise level in one step. The formula turns out to have the same structural pattern as the reparameterization trick the student already knows -- a signal term plus a noise term, weighted by complementary coefficients. The math is not arbitrary; every choice (Gaussian noise, variance-preserving scaling, the specific schedule) exists for a reason, and understanding those reasons gives the student real insight into why diffusion models are designed the way they are.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Symbolic | The core formulas: single step q(x_t\|x_{t-1}), the alpha/alpha_bar definitions, the closed-form q(x_t\|x_0). Each introduced with motivation before the notation. | This is a math-heavy lesson. The formulas ARE the content. But they must be introduced incrementally, each motivated by a problem the student feels. |
| Concrete example | 1D pixel walkthrough: pixel = 0.8, apply 3 steps with specific betas, verify closed-form gives same result. Actual numbers, actual arithmetic. | Without concrete numbers, the formulas are abstract symbol manipulation. Walking through with a specific pixel value grounds every symbol in something the student can compute. The "verify the shortcut gives the same answer" moment is the proof that matters. |
| Visual | Three visuals: (1) Bell curve diagram for Gaussian properties (two curves combining into one wider curve), (2) Noise schedule plot (beta_t and alpha_bar_t vs timestep), (3) Interactive widget showing alpha_bar curve and its effect on images at different timesteps. | The noise schedule IS a visual concept -- it is a curve that shapes the entire process. Seeing alpha_bar decrease from 1 to 0 as a smooth curve, with images placed along it, makes the schedule tangible. The bell curve diagram grounds the Gaussian properties the student needs. |
| Intuitive | The "mixing" intuition for variance-preserving: each step is a weighted blend of "old signal" and "new noise." The weights are chosen so the mix always has the same total variance. Like mixing paint: if you always keep the total volume constant by removing some old paint before adding new color, the volume stays bounded. | The variance-preserving constraint can feel arbitrary or purely mathematical. The mixing intuition makes it feel natural: you are not piling noise on top of signal, you are gradually replacing signal with noise while keeping the total amount constant. |
| Verbal/Analogy | The "zoom lens" analogy for the closed-form shortcut: iterating step by step is like walking from your house to a destination one block at a time. The closed-form formula is like looking at a map and teleporting directly there. Same destination, different path. The forward process definition tells you the route; the closed-form formula tells you the destination. | Motivates why the closed-form formula exists (efficiency) and what it means (same result as iterating). The student needs to feel that the shortcut is not an approximation -- it is mathematically identical. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3 genuinely new concepts: (1) Gaussian noise properties (addition, variance scaling) -- treated as tools, not deep theory, (2) the noise schedule and variance-preserving formulation (why each step blends signal and noise with specific coefficients), (3) the closed-form shortcut (alpha_bar and the direct formula). This is at the upper limit of the 3-concept rule.
- **Previous lesson load:** BUILD (the-diffusion-idea was conceptual intuition-building)
- **Is this appropriate?** Yes, but barely. After a BUILD lesson, STRETCH is appropriate. Three new concepts is the maximum. The mitigation is that concept (1) is a brief tool-building section, not a deep conceptual challenge, and concept (3) is the elegant payoff that makes concept (2) worthwhile. The load is front-heavy (properties + schedule setup) with a satisfying resolution (the shortcut). The emotional arc should feel like climbing a hill and reaching a vista, not like being buried under formulas.

**Cognitive load type: STRETCH**

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| Forward process intuition (Lesson 5) | Direct formalization. "You saw what happens -- now you will see the math behind it." Every formula is connected back to what the student observed in the widget. |
| Reparameterization trick (z = mu + sigma * epsilon) | The closed-form formula has the same structure: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon. Signal term + noise term, where epsilon ~ N(0,1). This is the key "aha" connection. Call it out explicitly: "This looks familiar." |
| Gaussian distribution / N(0,1) | Extended with two properties (addition, variance scaling). Framed as "you already know Gaussians -- here are two facts about them you need." |
| Variance as a measure of spread | From Series 1 foundations. The student knows variance intuitively (how spread out values are). Used without re-teaching. |
| "Same building blocks" mental model | Extended: the formula is not alien -- it has the same structure as something the student already knows (reparameterization trick). |
| DiffusionNoiseWidget (Lesson 5) | The interactive widget showed images at various noise levels. Now the student learns the formula that produces those images. The widget was the visual; this lesson is the math. Connect explicitly: "Remember the slider? Here is what it was computing." |

### Misleading Prior Analogies

- **"Clouds, not points" (VAE):** The student might expect the forward process to involve distributional encoding (mu and sigma per image). Clarify early that the forward process is purely mechanical noise addition -- there is no learned encoding. The image does not get an "uncertainty cloud." Noise is added to it directly.
- **Ink-in-water analogy (Lesson 5):** Useful for intuition but misleading for the math. Physical diffusion is a continuous PDE process; the DDPM forward process is discrete steps with a designed schedule. The ink analogy suggests the process is natural/inevitable; the math reveals it is carefully designed (the schedule is a choice). Briefly note this distinction when introducing the schedule.

### Scope Boundaries

**This lesson IS about:**
- Why Gaussian noise (mathematical convenience: addition property, closed-form shortcuts)
- The noise schedule: what beta_t is, the linear schedule, brief mention of cosine
- The variance-preserving formulation: why we scale the signal down before adding noise
- The alpha and alpha_bar notation: what they mean and why they exist
- The closed-form shortcut: q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
- What images look like at various timesteps (connecting the formula to visual experience)
- Interactive widget: noise schedule curve + image at selected timestep

**This lesson is NOT about:**
- The reverse process formula or how to denoise (Lesson 7)
- The training objective or loss function (Lesson 7)
- The sampling algorithm (Lesson 8)
- Code implementation of the forward process (Lesson 9)
- Score matching, score functions, or continuous-time SDEs (out of scope)
- Deriving the reverse process distribution (out of scope)
- U-Net or any denoising architecture (Module 6.3)
- Conditioning or text guidance (Module 6.3)
- Advanced noise schedules beyond linear and brief cosine mention (out of scope for this lesson)

**Target depths:**
- Gaussian noise properties (addition, variance scaling): INTRODUCED (stated as tools, not deeply explored)
- Noise schedule (beta_t, what it controls): DEVELOPED (student can explain what it does and why it matters)
- Variance-preserving formulation: DEVELOPED (student can explain why we scale down the signal)
- alpha_bar and the closed-form shortcut: DEVELOPED (student can use the formula to compute x_t from x_0 and explain why it works)

### Lesson Outline

1. **Context + Constraints** -- "Last lesson you saw images dissolve into noise. This lesson is about the math behind that process. We will write formulas, but every formula will be motivated -- you will understand WHY each piece exists before seeing it. By the end, you will have one elegant formula that lets you jump to any noise level in a single step. We will not cover the reverse process, the training loss, or any code -- those come in the next lessons."

2. **Hook (Puzzle)** -- Present a practical problem: "The diffusion training loop needs to create noisy versions of training images at random timesteps. If the forward process is defined step by step, and you need step 500, do you really have to compute all 500 previous steps? For 1,000 timesteps and 50,000 training images, that is 50 million noise-addition operations PER EPOCH. There must be a shortcut." This creates a concrete problem the lesson will solve. The student feels the need for the closed-form formula before it appears.

3. **Recap: Two Gaussian Properties You Need** -- Brief dedicated section resolving the two prerequisite gaps. (a) If you add two independent Gaussian samples, the result is Gaussian with variances summed. Visual: two bell curves combining into one wider curve. (b) If you multiply a Gaussian sample by c, the variance scales by c^2. Connect to the reparameterization trick: "sigma * epsilon gave you a sample with variance sigma^2 -- that is this property in action." Concrete numerical example for each property. Frame as "tools you will need in 5 minutes, not deep theory." Keep this section short and purposeful.

4. **Explain: One Step of Noise** -- Formalize a single step: q(x_t | x_{t-1}) = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon. Motivate each piece:
   - "We need to add noise. But how much?" -> introduce beta_t as the noise amount at step t
   - "Why not just x_{t-1} + sqrt(beta_t) * epsilon?" -> The variance-exploding negative example. Show what happens: after 100 steps, pixel values blow up. The image does not converge to manageable noise -- it explodes.
   - "So we scale the signal down: sqrt(1 - beta_t) * x_{t-1}." -> The mixing intuition: each step blends old signal with new noise, keeping total variance constant
   - Verify: Var(sqrt(1-beta_t) * x + sqrt(beta_t) * epsilon) = (1-beta_t) * Var(x) + beta_t * 1 = 1 if Var(x) = 1. "The variance stays at 1. This is why it is called variance-preserving."
   - Connect: "This is not arbitrary. Every coefficient exists for a reason."

5. **Check (Predict-and-Verify)** -- "If beta_t = 0 (no noise added this step), what does the formula give?" Answer: x_t = x_{t-1}. The image does not change. "If beta_t = 1 (maximum noise), what happens?" Answer: x_t = epsilon. The image is replaced entirely by noise. "So beta controls the blend between signal and noise at each step."

6. **Explain: The Noise Schedule** -- Introduce the schedule as a design choice:
   - "beta_t is not a single number -- it changes at each timestep. The sequence {beta_1, ..., beta_T} is the noise schedule."
   - Show the linear schedule: beta starts small (e.g., 0.0001) and grows linearly to a larger value (e.g., 0.02). Plot it.
   - "Why not constant? Because the amount of signal remaining changes. Early on, the image has lots of signal -- a small beta is enough to start the process. Later, the image is mostly noise -- a larger beta finishes the job."
   - Connect to what the student saw: "In the widget from last lesson, the first few steps barely changed the image. The last few steps erased the remaining traces. That is the schedule at work."

7. **Explain: Alpha Notation (Progressive Simplification)** -- Introduce alpha and alpha_bar as notational convenience, not complexity:
   - "Let us define alpha_t = 1 - beta_t. This is just a renaming. Where beta is the noise fraction, alpha is the signal fraction."
   - "Now the single-step formula becomes: x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon."
   - "But the real payoff is alpha_bar: alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t. This is the cumulative signal fraction -- how much of the original image survives after t steps."
   - Plot alpha_bar_t vs timestep. It starts near 1 (image mostly preserved) and drops to near 0 (image destroyed). "This one curve tells you everything about the forward process at any timestep."
   - Place widget here or reference it: show images at points along the alpha_bar curve.

8. **Explore: Interactive Widget** -- A widget showing the alpha_bar curve (smooth decreasing curve from 1 to ~0) with a draggable marker. As the student moves the marker along the curve, they see: (a) the current timestep t, (b) the alpha_bar value (signal fraction remaining), (c) what the image looks like at that noise level. Below the curve, show the formula x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon with the coefficients updating in real time. TryThisBlock experiments: (a) Find the timestep where alpha_bar = 0.5 (equal parts signal and noise). What does the image look like? (b) Move to very early timesteps -- alpha_bar is near 1. Can you tell the image is noisy? (c) Move to very late timesteps -- alpha_bar is near 0. Can you see any trace of the original? (d) Notice how the formula's coefficients change: when alpha_bar is large, the x_0 term dominates; when it is small, the epsilon term dominates.

9. **Explain: The Closed-Form Shortcut** -- The lesson's climax. Build it step by step:
   - "We have x_1 = sqrt(alpha_1) * x_0 + sqrt(1 - alpha_1) * epsilon_1."
   - "And x_2 = sqrt(alpha_2) * x_1 + sqrt(1 - alpha_2) * epsilon_2."
   - "Substitute x_1 into the x_2 equation. After collecting terms and using the Gaussian addition property from Section 3..."
   - Show the derivation for 2 steps, arriving at x_2 = sqrt(alpha_1 * alpha_2) * x_0 + sqrt(1 - alpha_1 * alpha_2) * epsilon.
   - "The product alpha_1 * alpha_2 is alpha_bar_2. And the pattern holds for any t steps."
   - The closed-form: **q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon**, where epsilon ~ N(0,1).
   - Call out the reparameterization trick connection: "This looks familiar. Signal term times x_0, plus noise term times epsilon. The same pattern as z = mu + sigma * epsilon from the VAE. Except here, the 'mu' is sqrt(alpha_bar_t) * x_0, and the 'sigma' is sqrt(1 - alpha_bar_t)."

10. **Concrete Verification: 1D Pixel Walkthrough** -- Take pixel value 0.8. Apply 3 steps with specific beta values (e.g., beta_1=0.01, beta_2=0.02, beta_3=0.03) and specific epsilon draws (e.g., epsilon_1=0.5, epsilon_2=-0.3, epsilon_3=0.8). Compute x_1, x_2, x_3 step by step. Then compute alpha_bar_3 and apply the closed-form formula with the combined epsilon. Show the results match. "Same destination, whether you walk or teleport."

11. **Check (Transfer)** -- "Your colleague says: 'The closed-form formula is just an approximation -- it cannot give exactly the same result as running all the steps.' Is this correct?" Answer: No. The formula is exact, not approximate. The Gaussian addition property is exact: the sum of independent Gaussians IS Gaussian. The only difference is which epsilon you use (step-by-step uses T different epsilons; the shortcut uses one combined epsilon drawn from the same resulting distribution). The distributions are mathematically identical.

12. **Elaborate: Why Gaussian?** -- Now that the student has seen the closed-form formula, explain why Gaussian noise was chosen:
    - Addition property: Gaussians add to Gaussians. This is what made the closed-form shortcut possible. If we used uniform noise, the sum of many steps would not have a clean closed form.
    - Central limit theorem: even if individual steps were not exactly Gaussian, many steps would approximate Gaussian by CLT. Nature converges to Gaussians.
    - Convenient parameterization: a Gaussian is fully described by mean and variance. Two numbers.
    - "Gaussian is not the only possible choice, but it is the choice that makes everything else elegant."

13. **Elaborate: Schedule Comparison** -- Briefly compare linear vs cosine schedules. Show alpha_bar curves for both. "The linear schedule drops alpha_bar too quickly in early timesteps -- the model gets fewer 'easy' training examples. The cosine schedule spends more time at high alpha_bar (low noise), giving the model more practice with subtle denoising. The widget in the last lesson used a cosine schedule." This is a brief elaboration, not a deep dive.

14. **Summarize** -- Key takeaways: (a) Each step of the forward process blends signal and noise with coefficients that keep variance constant (variance-preserving). The noise amount at each step is controlled by the schedule beta_t. (b) alpha_bar_t is the cumulative signal fraction -- the one number that tells you everything about noise level t. It starts near 1 (clean) and drops to near 0 (pure noise). (c) The closed-form formula q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon lets you jump to ANY timestep in one step. This is what makes training practical. (d) Every design choice exists for a reason: Gaussian noise for the addition property, variance-preserving for bounded values, the schedule for controlling the destruction rate.

15. **Next Step** -- "You now have the forward process -- the mathematical machinery for creating noisy training examples at any timestep in one step. Next: what does the model actually learn? The answer is surprisingly simple: predict the noise. The training objective is MSE loss on noise predictions -- the same MSE loss you have used since Series 1."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (2 gaps identified and resolved: Gaussian addition, variance scaling)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (dedicated "Two Gaussian Properties" section)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (practical problem: training efficiency)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: symbolic, concrete, visual, intuitive, verbal/analogy)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (3 new concepts -- at the limit, appropriate for STRETCH)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-10 (Iteration 1/3)

### Summary
- Critical: 2
- Improvement: 5
- Polish: 3

### Verdict: MAJOR REVISION

Two critical findings must be addressed before this lesson is usable. The closed-form derivation skips the explicit application of Property 2 (variance scaling), which is the very property the lesson taught minutes earlier. And the 1D pixel walkthrough computes step-by-step numerical results but then handwaves the closed-form comparison, defeating the purpose of the example. Both issues occur at the lesson's mathematical climax and would leave the student unable to follow the key derivation.

### Findings

#### [CRITICAL] -- Closed-form derivation skips the explicit use of Property 2 (variance scaling)

**Location:** Section 10: The Closed-Form Shortcut, the step from "Distribute and collect terms" to "Using Property 1 (variances add), the combined noise has variance..."
**Issue:** The derivation jumps from the distributed equation to the variance sum without showing HOW Property 2 is applied. After distributing, the student sees two noise terms: `sqrt(alpha_2(1 - alpha_1)) * epsilon_1` and `sqrt(1 - alpha_2) * epsilon_2`. The lesson says "Using Property 1 (variances add)" and shows the variance is `alpha_2(1 - alpha_1) + (1 - alpha_2)`. But the student needs to see Property 2 FIRST: "The first term is `sqrt(alpha_2(1 - alpha_1)) * epsilon_1`. By Property 2, scaling a N(0,1) sample by c gives variance c^2, so this term has variance alpha_2(1 - alpha_1)." Then Property 1 combines the two variances. The lesson taught Property 2 specifically for this moment and then does not show it being used.
**Student impact:** The student spent effort learning Property 2 and is now watching the derivation that supposedly uses it. The step where the scalar coefficients become variances is invisible -- the derivation jumps over it. This is the core mathematical move of the entire lesson. If the student cannot follow this step, the closed-form formula feels like a magic result rather than a derivation they can reconstruct.
**Suggested fix:** Add an intermediate step between the distributed equation and the variance sum. Something like: "Each noise term is a scaled Gaussian. By Property 2, `sqrt(alpha_2(1 - alpha_1)) * epsilon_1` has variance `alpha_2(1 - alpha_1)`, and `sqrt(1 - alpha_2) * epsilon_2` has variance `1 - alpha_2`. By Property 1 (independent Gaussians, variances add)..." This makes both properties visibly active in the derivation.

#### [CRITICAL] -- 1D pixel walkthrough does not complete the closed-form verification

**Location:** Section 11: Verification: 1D Pixel Walkthrough, the "Closed-form shortcut" box
**Issue:** The step-by-step section computes concrete numerical values (x_1 = 0.846, x_2 = 0.795, x_3 = 0.921) using specific epsilon draws. The closed-form section computes alpha_bar_3 = 0.9412 and the coefficients, then says "the closed-form gives a sample from the same distribution." But it never plugs in a combined epsilon value to produce a number. The student has a concrete step-by-step result (0.921) and an abstract claim ("same distribution") with no matching concrete result. This undermines the entire purpose of the worked example: to demonstrate that both methods give the same answer with real numbers. The planning document explicitly says: "Show the results match."
**Student impact:** The student expects the payoff: plug numbers into the closed-form formula and get the same result (or an explanation of why the specific numbers differ but the distributions match). Instead they get a hand-wave. The student who was doing mental math along with each step suddenly has nothing to verify. The "same destination, whether you walk or teleport" subtitle is unfulfilled.
**Suggested fix:** Either (a) show that the three separate epsilon values combine into a single equivalent epsilon (compute the equivalent combined epsilon value and plug it into the closed-form to get 0.921), which would demonstrate exact numerical matching, OR (b) reframe the example explicitly: compute the closed-form with a fresh single epsilon (e.g., epsilon = 0.5), show the result, and then clearly explain that the step-by-step and closed-form produce different specific numbers because they use different random draws, but both draw from q(x_3|x_0) = N(0.9702 * 0.8, 0.2425^2) = N(0.776, 0.059). Either approach grounds the claim in numbers the student can verify.

#### [IMPROVEMENT] -- Reparameterization trick "aha" connection is in an aside, not in the main content

**Location:** Section 10: The Closed-Form Shortcut, Row.Aside InsightBlock titled "Same Pattern"
**Issue:** The planning document identifies the reparameterization trick connection as the key "aha" moment: "This looks familiar." The aside calls it out, but asides are secondary content that the student may not read carefully. The main flow goes from the closed-form formula directly to "This is the formula you saw updating in the widget." The structural parallel between `q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon` and `z = mu + sigma * epsilon` is the single most important connection in the lesson. It deserves to be in the main content body, not relegated to a side panel.
**Student impact:** A student who does not read the aside carefully misses the planned pedagogical payoff. The formula feels like a new thing to memorize rather than a variation on a pattern they already know.
**Suggested fix:** Add a paragraph in the main content (after the highlighted closed-form formula box) that explicitly calls out the structural parallel: "Notice the pattern. This is the reparameterization trick again: signal + noise_scale * epsilon. In the VAE, it was z = mu + sigma * epsilon. Here, the 'mu' is sqrt(alpha_bar_t) * x_0 and the 'sigma' is sqrt(1 - alpha_bar_t). Same structure, different context." The aside can stay for the concise version, but the main content must carry the connection.

#### [IMPROVEMENT] -- Gaussian properties section lacks a visual (planned but not built)

**Location:** Section 4: Two Gaussian Properties You Need
**Issue:** The planning document specifies three visuals, including "Bell curve diagram for Gaussian properties (two curves combining into one wider curve)." The built lesson presents Property 1 and Property 2 with only symbolic formulas and one concrete numerical example. There is no visual representation of two Gaussians combining into a wider Gaussian. The modality rule requires at least 3 modalities for core concepts. The Gaussian properties section has symbolic (formulas) and concrete (the 3+5=8 example) but no visual modality.
**Student impact:** The student who thinks visually has no spatial anchor for these properties. "Variances add" is a statement; seeing two narrow bell curves merge into one wider curve makes the property intuitive. This is also the first mathematical content in the lesson, so a visual here lowers the activation energy for the dense derivation ahead.
**Suggested fix:** Add an inline SVG or a small diagram showing two Gaussians with labeled variances combining into a wider Gaussian. This does not need to be interactive -- a static visual is sufficient for a tool-building section. Alternatively, a brief aside with a visual framing would suffice.

#### [IMPROVEMENT] -- Misconception 1 (noise is same amount each step) is not addressed with a negative example

**Location:** The Noise Schedule section (Section 7)
**Issue:** The planning document identifies misconception 1: "The noise added at each step is the same amount." The suggested negative example is to "show images at early vs late timesteps with the same step size." The built lesson explains that beta_t changes at each timestep and references what the student saw in the widget, but never shows a concrete negative example of what would happen if beta were constant. The misconception is addressed verbally but not with a specific visual or numerical counter-example. The misconception rule requires a concrete negative example for each planned misconception.
**Student impact:** A student who assumed constant noise has the misconception corrected verbally but may not fully feel why it matters. Without seeing what constant noise looks like (linear degradation, poor results), the schedule feels like an implementation detail rather than a critical design choice.
**Suggested fix:** Add a brief contrast: "If beta were constant at 0.01 for all 1,000 steps, the first 100 steps would destroy as much information as the last 100. But the early steps have much more signal to preserve. A constant schedule wastes the early timesteps." This can be 2-3 sentences within the existing section.

#### [IMPROVEMENT] -- Variance-preserving derivation assumes Var(x_{t-1}) = 1 without justification

**Location:** Section 5: One Step of Noise, the variance verification
**Issue:** The variance check says "If x_{t-1} has variance 1" and then shows the algebra. But the student has not been told why x_{t-1} has variance 1. For the FIRST step, x_0 is a clean image. Images normalized to [0, 1] do not have variance 1. The variance-preserving property holds IF the input variance is 1, which is approximately true after enough steps (since the process converges to N(0, I)), but is not exactly true for x_0. The lesson asserts the premise without grounding it.
**Student impact:** A careful student will ask "wait, does my image actually have variance 1?" and feel confused. A less careful student will accept the premise but have a fragile understanding that could break when they see real images with different value ranges.
**Suggested fix:** Add a brief note: "In practice, images are normalized to have roughly unit variance before the forward process begins. As long as the input has variance close to 1, the variance-preserving property ensures it stays near 1 at every subsequent step." This is one sentence that resolves the dangling assumption.

#### [IMPROVEMENT] -- Second positive example from planning doc (28x28 image comparison) is missing

**Location:** The entire lesson
**Issue:** The planning document specifies four examples. Example 2 is: "Noise progression on a 28x28 image at specific timesteps -- show iterative step-by-step result with closed-form jump at t=250, t=500, t=750. Visually identical results confirm the formula works on actual images." The built lesson has the widget (which shows images at various timesteps) and the 1D pixel walkthrough, but never shows side-by-side images comparing step-by-step iteration vs closed-form at the same timestep on an actual image. The example rule requires at least 2 positive examples that demonstrate the concept generalizes. The 1D walkthrough is one positive example; the missing 28x28 comparison was the second.
**Student impact:** The student verifies the formula on a single pixel and is told it works on images, but never sees the visual proof. The widget shows the closed-form result at various timesteps but not the comparison with step-by-step. The generalization from 1D to 2D images is asserted rather than demonstrated.
**Suggested fix:** This example could be difficult to implement without a running model. Two options: (a) Add a ComparisonRow or inline strip showing the same procedural image at t=250 from both methods (the widget already has the procedural image and noise application code, so this is feasible), or (b) add a brief paragraph connecting the widget to the 1D example: "The widget you just used IS the closed-form formula in action on a full image. Every pixel follows exactly the same math you just verified for a single pixel. The formula applies independently to each pixel."

#### [POLISH] -- Widget description text has inaccurate range for alpha_bar near 0.3-0.5

**Location:** `AlphaBarCurveWidget.tsx`, `getDescription()` function (lines 555-568)
**Issue:** The description for `alphaBar > 0.3` says "Alpha-bar near 0.5" but the condition catches values from 0.3 to 0.5. The description for `alphaBar > 0.1` says "Alpha-bar below 0.3" which is accurate. The 0.3-0.5 range description is slightly misleading -- a student at alpha_bar = 0.35 would see "near 0.5" when it is closer to 0.3.
**Student impact:** Minor confusion when the description does not match the displayed value.
**Suggested fix:** Change the description for `alphaBar > 0.3` to something like: "Alpha-bar between 0.3 and 0.5 -- signal and noise are roughly balanced. The image is recognizable but degraded." Or adjust the range breakpoints.

#### [POLISH] -- The map/walking analogy in the closed-form section differs from the planned "zoom lens" analogy

**Location:** Section 10, second Row after the closed-form formula (lines 686-692)
**Issue:** The planning document specifies a "zoom lens" analogy; the built lesson uses a "map and walking" analogy (walking one block at a time vs looking at coordinates on a map). The map analogy is arguably clearer than "zoom lens" -- this is a positive deviation, not a problem. Noting for completeness.
**Student impact:** None negative. The map analogy works well.
**Suggested fix:** No fix needed. The deviation is an improvement. Update the planning document to reflect the actual analogy used.

#### [POLISH] -- Schedule toggle buttons in widget lack cursor-pointer

**Location:** `AlphaBarCurveWidget.tsx`, schedule toggle buttons (lines 517-538)
**Issue:** The "Cosine" and "Linear" toggle buttons do not have `cursor-pointer` in their className. They use standard button elements which typically have default cursor behavior, but explicit `cursor-pointer` is the lesson's interaction design standard per the review checklist. The slider has `cursor-ew-resize` (correct), and the SVG has `cursor-crosshair` (correct), but the buttons rely on default browser behavior.
**Student impact:** Minimal -- most browsers show a pointer on buttons by default. But the inconsistency with the explicit cursor classes elsewhere is notable.
**Suggested fix:** Add `cursor-pointer` to the button className strings.

### Review Notes

**What works well:**
- The narrative arc is excellent. The hook (practical problem: 50 million operations) creates genuine motivation before any formula appears. The student feels the need for the shortcut before it is introduced.
- The variance-exploding negative example is well-placed and well-executed. It motivates variance-preserving before stating the formula. The rose-colored styling makes it visually distinct as a "don't do this" moment.
- The predict-and-verify checks (beta=0 and beta=1 edge cases) are strong. They let the student test the formula against intuition.
- The widget is well-integrated. The TryThisBlock experiments are specific and purposeful, not generic "play around" instructions.
- Scope boundaries are clean. The lesson does not drift into reverse process, training loss, or architecture territory.
- All content uses Row components correctly. Em dash style is consistent (no spaces).
- The alpha notation section subtitle ("Progressive simplification, not progressive complexity") is a good framing that addresses the notation-soup misconception proactively.

**Systemic pattern:**
The two critical findings share a root cause: the lesson sets up promises (teach two Gaussian properties, then use them; do a numerical walkthrough, then verify with closed-form) and partially delivers on them. The properties are taught but their application is compressed into an invisible step. The walkthrough computes numbers but the verification abstracts away. This is a lesson that does 90% of the work and then rushes the payoff. The fixes are all about completing what is already started -- adding the missing intermediate step, adding the missing numbers.

---

## Review -- 2026-02-10 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 2

### Verdict: NEEDS REVISION

All critical and improvement findings from iteration 1 have been addressed. The derivation now explicitly shows Property 2 in action (each scaled noise term's variance is computed individually before Property 1 combines them). The walkthrough now computes an effective epsilon and plugs it into the closed-form formula to demonstrate exact numerical agreement. The reparameterization trick connection is now in the main content body. The bell curve SVG is present. The Var(x)=1 note is present. The constant-schedule paragraph is present. The widget-to-formula connection paragraph is present. Widget cursor-pointer on buttons is fixed. Description ranges are fixed.

One new improvement finding emerged on a fresh-eyes pass. No regressions detected from the fixes.

### Findings

#### [IMPROVEMENT] -- Walkthrough verification logic is sound but has a subtle framing issue

**Location:** Section 11: Verification: 1D Pixel Walkthrough, the "Closed-form shortcut" box
**Issue:** The verification works backward: it takes the step-by-step result (x_3 = 0.921), solves for the effective epsilon (0.598), and says "this epsilon is within 1 standard deviation of N(0,1), so the result is consistent." This is logically valid but pedagogically weaker than the planning document's intent. The student computes a number (0.598), checks that it is "reasonable," and is told this proves the formulas are equivalent. But any single number from N(0,1) would be "reasonable" -- the check does not feel conclusive. The student might think: "Sure, 0.598 is within N(0,1), but so would 2.5 be." The verification would be stronger if it explicitly completed the round-trip: plug the effective epsilon back into the closed-form formula and show that it reproduces 0.921 exactly. The current text says "the closed-form formula would produce [x_3 = 0.921] with this effective epsilon" but does not show the multiplication: 0.9702 * 0.8 + 0.2425 * 0.598 = 0.776 + 0.145 = 0.921. That one line of arithmetic would close the loop and make the verification feel airtight rather than probabilistic.
**Student impact:** The student who is following along with a calculator has a step-by-step result of 0.921 and wants to see 0.921 come out of the closed-form formula. The current path goes: 0.921 -> solve for epsilon -> check epsilon is reasonable. The stronger path would be: 0.921 -> solve for epsilon -> plug epsilon back in -> get 0.921. The "check epsilon is reasonable" framing converts an exact algebraic equivalence into a statistical plausibility argument, which undersells the result.
**Suggested fix:** After the effective epsilon calculation (0.598), add one line: "Plug it back in: 0.9702 * 0.8 + 0.2425 * 0.598 = 0.776 + 0.145 = 0.921. Exactly the step-by-step result." Then the existing text about "same distribution, same answer" lands with full force.

#### [POLISH] -- Gaussian addition visual uses matching variances, weakening the "variances add" message

**Location:** Section 4: Two Gaussian Properties You Need, the GaussianAdditionVisual component
**Issue:** The two dashed Gaussians both have variance 0.8 (var1 = var2 = 0.8). The visual shows two identical narrow curves combining into one wider curve, labeled N(0, sigma^2) and N(0, 2*sigma^2). This is technically correct but the visual does not match the concrete example given in the text (N(0, 3) + N(0, 5) = N(0, 8)). The concrete example uses different variances (3 and 5) specifically to show that "the variances (3 + 5 = 8), not the standard deviations, are what add." But the visual uses identical variances, which means the student cannot see the asymmetric addition that makes the concrete example compelling. Two identical curves combining into a wider one could be interpreted as "the result is twice as wide" rather than "the variances add."
**Student impact:** Minor. The visual still communicates "adding Gaussians gives a wider Gaussian," which is the core message. But the mismatch between the text's numerical example (3 + 5 = 8) and the visual's symmetric setup (sigma^2 + sigma^2 = 2*sigma^2) is a missed opportunity to reinforce the specific non-trivial case.
**Suggested fix:** Either (a) use var1 = 0.6 and var2 = 1.0 in the visual (non-equal, so the visual shows an asymmetric combination matching the text's spirit), or (b) label the visual with the actual concrete numbers from the text (N(0, 3) and N(0, 5) -> N(0, 8)). This is low priority since the core message lands either way.

#### [POLISH] -- Alpha-bar product rounding: 0.9412 vs exact 0.9411

**Location:** Section 11: Verification: 1D Pixel Walkthrough, the closed-form shortcut box
**Issue:** The lesson states alpha_bar_3 = 0.99 * 0.98 * 0.97 = 0.9412. The exact value is 0.941094, which rounds to 0.9411 (four decimal places). The lesson uses 0.9412. This is a tiny rounding difference that does not affect any downstream calculations (all subsequent values are internally consistent with 0.9412), but a student with a calculator would get 0.9411 and wonder about the discrepancy.
**Student impact:** Negligible. The downstream math is internally consistent. A pedantic student might notice; most would not.
**Suggested fix:** Change 0.9412 to 0.9411. One-character fix. Or leave as-is and accept the rounding convention.

### Review Notes

**Verification of iteration 1 fixes:**

1. CRITICAL -- Property 2 in derivation: FIXED. Lines 772-788 now explicitly show "By Property 2 (scaling scales variance)" with two bullet points computing each noise term's variance individually (alpha_2(1 - alpha_1) and 1 - alpha_2) before Property 1 combines them. The derivation now visibly uses both properties the student learned.

2. CRITICAL -- Walkthrough verification: FIXED. Lines 928-944 now compute an effective epsilon (0.598), show it is within N(0,1), and state that the closed-form formula with this effective epsilon reproduces the step-by-step result. This is a substantial improvement over the original hand-wave. The one remaining improvement (plugging epsilon back in to close the arithmetic loop) is noted above.

3. IMPROVEMENT -- Reparam connection in main body: FIXED. Lines 816-827 now contain the full "Notice the structure. This is the reparameterization trick again..." paragraph in the main content body, not just in the aside. The aside (lines 831-836) provides the concise version. Both are present, with the main body carrying the pedagogical weight.

4. IMPROVEMENT -- Bell curve SVG: FIXED. Lines 74-160 implement a GaussianAdditionVisual component that renders two dashed Gaussians combining into a wider solid Gaussian. Placed directly under the Property 1 text and concrete example.

5. IMPROVEMENT -- Var(x)=1 note: FIXED. Lines 459-464 add an italic note explaining that images are normalized to [-1, 1] before the forward process begins, grounding the Var(x_{t-1}) = 1 assumption.

6. IMPROVEMENT -- Constant-schedule example: FIXED. Lines 559-565 add a paragraph explaining what happens with a constant beta of 0.01 for 1,000 steps, and why a constant schedule is the wrong tradeoff.

7. IMPROVEMENT -- Widget-to-formula connection: FIXED. Lines 956-963 connect the widget back to the 1D walkthrough: "The widget you explored earlier IS the closed-form formula in action on a full image. Every pixel follows exactly the same math you just verified."

8. POLISH -- Widget description ranges: FIXED. The getDescription function now uses "between 0.3 and 0.5" for the relevant range.

9. POLISH -- Widget cursor-pointer on buttons: FIXED. Lines 520 and 531 include cursor-pointer in both button classNames.

10. POLISH -- Map analogy deviation: Acknowledged in iteration 1 as a positive deviation. No fix needed.

**What works well (fresh-eyes assessment):**
- The lesson has excellent narrative momentum. The hook creates a genuine problem (50M operations per epoch), the Gaussian properties section feels purposeful (tools for a derivation 5 minutes away), and the closed-form formula arrives as a satisfying payoff.
- The derivation in Section 10 is now one of the strongest parts of the lesson. Each step is labeled (substitute, distribute, apply Property 2, apply Property 1, simplify), and both taught properties are visibly used. A student can reconstruct this derivation from scratch.
- The predict-and-verify exercises (beta=0 and beta=1 edge cases) are well-placed and effective. They give the student agency between the formula introduction and the dense derivation.
- The widget integration is strong: the widget is introduced at the right point (after alpha_bar notation, before the closed-form derivation), the TryThisBlock experiments are specific, and the lesson later connects the widget back to the 1D walkthrough.
- Scope boundaries are clean. The lesson never drifts into reverse process, training loss, or architecture territory.
- Em dash style is consistently correct (no spaces) throughout.
- All content uses Row components correctly.

---

## Review -- 2026-02-10 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All findings from iterations 1 and 2 have been addressed. The lesson is pedagogically sound and ready to ship.

**Iteration 2 fix verification:**

1. **Algebraic round-trip (IMPROVEMENT):** FIXED. Lines 935-945 compute the effective epsilon (0.598), then explicitly plug it back into the closed-form formula: 0.9702 * 0.8 + 0.2425 * 0.598 = 0.776 + 0.145 = 0.921, with a checkmark. The round-trip is complete and the verification feels airtight.

2. **Bell curve visual with non-equal variances (POLISH):** FIXED. GaussianAdditionVisual now uses var1 = 0.6, var2 = 1.0 (asymmetric), with labels "N(0, 3) and N(0, 5)" and "N(0, 8)" matching the text's concrete example. The two dashed curves are visibly different widths.

3. **Alpha-bar rounding 0.9412 -> 0.9411 (POLISH):** PARTIALLY FIXED. The alpha_bar value is displayed as 0.9411 (correct), but the derived coefficients (sqrt(alpha_bar) = 0.9702, sqrt(1-alpha_bar) = 0.2425) are still consistent with the old value of 0.9412. Strictly, sqrt(0.9411) = 0.9701 and sqrt(0.0589) = 0.2427 to 4dp. See the one remaining polish finding below.

**No regressions detected.** All 10 fixes from iteration 1 remain intact.

### Findings

#### [POLISH] -- Walkthrough coefficients inconsistent with corrected alpha_bar

**Location:** Section 11: Verification: 1D Pixel Walkthrough, the closed-form shortcut box (lines 912-945)
**Issue:** The lesson displays alpha_bar_3 = 0.9411 (corrected from 0.9412 per iteration 2). However, the derived coefficients sqrt(alpha_bar_3) = 0.9702 and sqrt(1 - alpha_bar_3) = 0.2425 are computed from the old value of 0.9412, not from 0.9411. The exact values: sqrt(0.941094) = 0.9701 and sqrt(0.058906) = 0.2427 (to 4dp). The displayed coefficients are off by 1 in the 4th decimal place. The round-trip arithmetic still closes (0.921), so the pedagogical impact is negligible.
**Student impact:** A student with a calculator verifying sqrt(0.9411) would get 0.9701, not 0.9702, and might briefly wonder about the discrepancy. The final answer (0.921) is unaffected because the rounding errors cancel in the round-trip.
**Suggested fix:** Change 0.9702 to 0.9701 and 0.2425 to 0.2427, then recompute the round-trip: effective epsilon = (0.921 - 0.9701 * 0.8) / 0.2427 = (0.921 - 0.7761) / 0.2427 = 0.597, and 0.9701 * 0.8 + 0.2427 * 0.597 = 0.7761 + 0.1449 = 0.921. Alternatively, change alpha_bar back to 0.9412 and keep the existing coefficients -- either way, just make them consistent.

### Review Notes

**What works well:**
- The lesson's narrative arc is excellent. The hook (50M operations per epoch) creates genuine motivation, the Gaussian properties section feels purposeful, and the closed-form formula arrives as a satisfying payoff.
- The derivation in Section 10 is the strongest part of the lesson. Both taught Gaussian properties are visibly used (Property 2 in bullet points, Property 1 for combining). A student can reconstruct this derivation from scratch.
- The 1D pixel walkthrough with full round-trip verification (step-by-step -> effective epsilon -> plug back in -> get 0.921) is convincing and complete. The student sees exact numerical agreement, not a statistical plausibility argument.
- The reparameterization trick connection is now in the main body where it belongs, providing the key "aha" moment that the formula is not new -- it is a variation on z = mu + sigma * epsilon.
- The widget is well-integrated: placed after notation but before derivation, with specific TryThisBlock experiments and a later connection paragraph bridging the 1D walkthrough to the full-image case.
- Scope boundaries are clean. The lesson stays within its stated scope throughout.
- All interactive elements have appropriate cursor styles. Em dashes are consistent. Row components used everywhere.

**Overall assessment:** This lesson is ready to ship. It teaches a genuinely hard mathematical concept (the closed-form shortcut) with excellent scaffolding: motivation before formulas, tools before they are needed, concrete examples grounding every abstraction, and multiple modalities reinforcing the core idea. The one remaining polish finding is a 4th-decimal-place rounding inconsistency that does not affect the student's understanding.
