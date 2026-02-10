# Lesson: Learning to Denoise (DDPM)

**Slug:** `learning-to-denoise`
**Module:** 6.2 (Diffusion)
**Series:** 6 (Stable Diffusion)
**Position:** Lesson 3 of 5 (Lesson 7 overall in series)
**Cognitive load:** BUILD (relief after STRETCH of the-forward-process)

---

## Phase 1: Orient — Student State

The student just completed the-forward-process (STRETCH), the hardest lesson in the module. They are mathematically fatigued but have strong forward-process foundations. This lesson should feel like a payoff — "after all that math, the training objective is just MSE on noise?"

### Relevant Concepts with Depths

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Closed-form shortcut q(x_t\|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon | DEVELOPED | the-forward-process (6.2) | Derived step by step, verified with 1D pixel walkthrough. The student can use this formula. This is the formula the training algorithm uses to create noisy images. |
| Alpha-bar (cumulative signal fraction) | DEVELOPED | the-forward-process (6.2) | The one number encoding the entire schedule history. "Signal-to-noise dial." Interactive widget experience. |
| Noise schedule (beta_t) | DEVELOPED | the-forward-process (6.2) | Design choice, not physics. Linear and cosine schedules compared. |
| Variance-preserving formulation | DEVELOPED | the-forward-process (6.2) | Signal scaled down before noise added, total variance stays at 1. |
| "Small steps make it learnable" (core diffusion insight) | DEVELOPED | the-diffusion-idea (6.2) | One-shot generation is underdetermined; removing small amounts of noise is tractable. Core mental model for WHY diffusion works. |
| Reverse process as learned iterative denoising | INTRODUCED | the-diffusion-idea (6.2) | A neural network learns to remove noise. No math, no training details, no architecture. Student knows the WHAT but not the HOW. |
| MSE loss function | DEVELOPED | loss-functions (1.1) | Full formula with KaTeX, squaring rationale. "Wrongness score." Used extensively since Series 1. |
| nn.MSELoss (PyTorch) | DEVELOPED | training-loop (2.1) | Stateless callable object wrapping the same MSE formula. criterion(y_hat, y). |
| Training loop (forward -> loss -> backward -> update) | DEVELOPED | implementing-linear-regression (1.1), training-loop (2.1) | "Heartbeat of training." Student has implemented this many times in PyTorch. |
| Reconstruction loss (MSE, target IS the input) | DEVELOPED | autoencoders (6.1) | Same MSE formula but the target is the input image, not a label. Student saw this in the autoencoder context. |
| Reparameterization trick (z = mu + sigma * epsilon) | INTRODUCED | variational-autoencoders (6.1) | Isolating randomness. The closed-form formula has the same structure: signal + noise_scale * epsilon. Connected explicitly in the-forward-process. |
| Gaussian noise properties (addition, scaling) | INTRODUCED | the-forward-process (6.2) | Tools for the closed-form derivation. Student can use them but hasn't practiced them. |
| Multi-scale denoising progression (coarse-to-fine) | INTRODUCED | the-diffusion-idea (6.2) | High noise = structural decisions, low noise = fine details. |

### Mental Models Already Established
- "Destruction is easy; creation from scratch is impossibly hard; but undoing a small step of destruction is learnable."
- "Alpha-bar is the signal-to-noise dial. The closed-form formula lets you turn that dial to any position."
- "Same building blocks, different question" — conv layers, MSE loss, backprop are shared; the question changes.
- "Training loop = forward -> loss -> backward -> update" — the heartbeat of training.
- "MSE = wrongness score" — the squaring rationale (penalizes big errors more) is deeply familiar.

### Readiness Assessment
The student is well prepared. The two critical prerequisites — the closed-form formula and MSE loss — are both at DEVELOPED depth. The closed-form formula was just taught (previous lesson), and MSE loss has been used continuously since Series 1. The "same building blocks, different question" mental model from the-diffusion-idea is the perfect frame: the training objective IS the same MSE loss the student already knows, applied to a different question (predicting noise instead of predicting labels or reconstructing images).

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain and trace the DDPM training algorithm: given an image, sample a random timestep and random noise, create the noisy image using the closed-form formula, have a neural network predict the noise, and compute MSE loss between predicted and actual noise.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Closed-form formula q(x_t\|x_0) | DEVELOPED (must use in training algorithm) | DEVELOPED | the-forward-process (6.2) | OK | Student derived it, verified it with arithmetic, interacted with it via widget. Can use it as a tool. |
| Alpha-bar | DEVELOPED (must interpret in formula) | DEVELOPED | the-forward-process (6.2) | OK | Understands as "signal-to-noise dial," used in widget. |
| MSE loss | DEVELOPED (must recognize and apply) | DEVELOPED | loss-functions (1.1) | OK | Taught in Series 1, used in autoencoders, used in PyTorch as nn.MSELoss. Deeply familiar. |
| Training loop pattern | DEVELOPED (must map algorithm to pattern) | DEVELOPED | training-loop (2.1) | OK | "Heartbeat of training" used in every PyTorch module since 2.1. |
| Forward process as gradual noise destruction | INTRODUCED (must understand why we noise images) | INTRODUCED | the-diffusion-idea (6.2) | OK | Student understands the conceptual framework. INTRODUCED is sufficient — we don't need them to apply the forward process, just understand why it exists. |
| Reverse process as learned denoising | INTRODUCED (must connect training to eventual sampling) | INTRODUCED | the-diffusion-idea (6.2) | OK | Student knows the model will denoise; this lesson teaches how it learns to. |
| Gaussian noise (epsilon ~ N(0,1)) | INTRODUCED (must understand what the model predicts) | INTRODUCED | the-forward-process (6.2) | OK | Student used epsilon in the closed-form formula. Knows it is standard normal noise. |
| Neural network as function approximator | DEVELOPED (must accept that a network can predict noise) | DEVELOPED | what-is-learning (1.1), nn-module (2.1) | OK | Core mental model from the start of the course. The network approximates a function — here, the function from noisy image to noise. |

**No gaps.** All prerequisites are at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The model predicts the clean image, not the noise" | Intuitively, denoising means recovering the original. Every prior loss compared prediction to a target (labels, reconstructed images). Predicting noise feels backwards. | If the model predicts x_0 directly, its MSE target varies wildly (a cat, a dog, a landscape). If it predicts epsilon, the target is always standard normal noise — same distribution regardless of the image content. The noise prediction is a simpler, more consistent target. Also: given epsilon_predicted, you can algebraically recover x_0 from the closed-form formula — they are equivalent, but noise prediction gives better gradient signal. | Core explain section. Motivate with "which is easier to predict?" before revealing the DDPM choice. Include the algebraic equivalence to show nothing is lost. |
| "The model sees the entire noise trajectory (all 1000 steps)" | The forward process was described as iterative steps 1 through T. Training might seem to require iterating. Also, the previous lesson's "walking vs teleporting" analogy might not have fully landed. | The closed-form formula q(x_t\|x_0) jumps directly to any timestep. Training samples ONE random timestep per image per training step. The model never sees a sequence — it sees a single noisy image and must predict the noise. Reinforce the "teleporting" insight from the-forward-process. | Early in the training algorithm walkthrough. Explicitly state: "one random timestep, not a sequence." |
| "Training loops through timesteps 1 to T in order" | Sequential thinking is natural. The forward process was described as a sequence (step 1, step 2, ..., step T). The student might assume training follows the same order. | Training samples t uniformly at random. This is critical for efficiency and for the model learning all noise levels equally. A concrete example: in one batch, the model might see t=723 for one image, t=42 for another, and t=891 for a third. No sequential order. | Training algorithm walkthrough. Show the random sampling explicitly. |
| "MSE on noise is a different loss function from the MSE I already know" | The context is so different (predicting noise in images vs predicting house prices) that the student might think the math changed. | Write out both: MSE_from_series_1 = (1/n) * sum(y_hat - y)^2 and DDPM_loss = (1/n) * sum(epsilon_hat - epsilon)^2. They are identical. The only difference is what y_hat and y represent. The formula, the gradients, the PyTorch code — all the same. | The "surprisingly simple" reveal moment. Side-by-side comparison. |
| "The model needs a separate network for each timestep" | 1000 different noise levels might seem to require 1000 specialized denoisers, one per noise level. | One network handles ALL timesteps. It receives the timestep t as an input alongside the noisy image. At t=50, it has learned to remove a tiny amount of noise; at t=900, it has learned to hallucinate structure from near-pure static. The timestep conditions the behavior. (Architecture details deferred to Module 6.3, but the single-network fact is stated here.) | After the training algorithm, before scope boundaries. Brief clarification, not a deep dive into conditioning. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Concrete training step walkthrough: one image, one timestep, one noise sample | Positive | The core example. Trace every value through the algorithm: pick image x_0, sample t=500, sample epsilon from N(0,1), compute x_500 via closed-form, feed to network, get epsilon_hat, compute MSE. | Makes the abstract algorithm concrete. Uses a specific timestep (500 = midpoint, intuitive) and shows exact operations. The student can see that every piece is familiar — closed-form formula from last lesson, MSE from Series 1. |
| Same image, different timestep (t=50 vs t=950) | Positive | Shows how the SAME algorithm works at extreme noise levels. At t=50, the noisy image is barely changed and the network must predict tiny perturbations. At t=950, the image is almost pure noise and the network must infer global structure. | Extends the first example to show the range of the problem. Connects to multi-scale denoising from the-diffusion-idea. Prevents the misconception that the model does the same thing at every timestep. |
| "What if we predicted x_0 instead of epsilon?" | Negative | Shows why noise prediction is preferred. The x_0 target varies wildly across images (cat, dog, landscape). The epsilon target is always from the same distribution (N(0,1)). Both are mathematically equivalent (you can recover one from the other), but noise prediction gives a more consistent optimization landscape. | Addresses the #1 misconception head-on. Gives the student the "of course" moment: predicting a consistent target is easier than predicting wildly varying images. Also shows the algebraic equivalence so nothing feels lost. |
| Mini-batch: three images with different random timesteps | Positive (stretch) | Shows what a real training batch looks like: image A at t=200, image B at t=750, image C at t=45. Different timesteps, different noise levels, all in one batch, all contributing to the same MSE loss. | Extends from single-example to batch-level understanding. Concretely shows the random timestep sampling. Connects to the student's experience with mini-batch training from Series 2. |

---

## Phase 3: Design

### Narrative Arc

The student has just survived the hardest lesson in the module. They now have the closed-form formula — the ability to teleport to any noise level — but they don't yet know what to do with it. The forward process was about destruction; this lesson is about teaching a neural network to reverse that destruction. The emotional arc is relief: after the mathematical intensity of the-forward-process, the training objective turns out to be almost embarrassingly simple. Sample an image, pick a random noise level, add noise using the formula you just learned, have a network predict the noise, compute MSE loss. That's it. The student has been computing MSE loss since the third lesson of the entire course. The same formula that measured how wrong a linear regression was now measures how well a neural network predicts noise. The "same building blocks, different question" mental model from the-diffusion-idea delivers its biggest payoff here: the entire DDPM training objective is built from pieces the student has used dozens of times.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Verbal/Analogy | "The training objective is an open-book exam: the network always has the answer key (epsilon) and we measure how close its guess is." + MSE loss callback: "the same wrongness score from your third lesson." | The analogy frames the entire training process: the network is given a noisy image (the question), must predict the noise (its answer), and we check against the actual noise (the answer key). The MSE callback grounds the concept in deep familiarity. |
| Symbolic/Formula | Side-by-side comparison: Series 1 MSE = (1/n) sum(y_hat - y)^2 vs DDPM loss = \|\|epsilon - epsilon_theta(x_t, t)\|\|^2. Highlight that epsilon_theta is just the model's prediction and epsilon is the target. | The formulas being visually identical is the key insight. The student must see that no new math was invented — the "same formula, different letters" moment. |
| Concrete example | Step-by-step walkthrough of one training iteration with specific values: x_0 (T-shirt image), t=500, epsilon (specific noise vector), x_500 computed via closed-form, epsilon_hat from network, MSE computed. | The algorithm is abstract until you trace specific values through each step. Using t=500 (the midpoint) makes the noise level intuitive. Using the T-shirt image provides visual continuity with the widgets from earlier lessons. |
| Visual | Diagram of the training algorithm as a flow: x_0 -> [sample t, sample epsilon] -> closed-form formula -> x_t -> [neural network] -> epsilon_hat -> [MSE loss with epsilon]. Arrows showing information flow. | The training algorithm has clear stages. A flow diagram makes the pipeline visible and shows where each piece connects. The student can trace the data path. |
| Intuitive | "Which target is easier to predict: the noise or the clean image? The noise comes from the same distribution every time (N(0,1)). The clean image could be anything." | Builds the "of course" feeling for why predicting noise is preferred. The student should feel this is obvious in retrospect. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2
  1. The DDPM training objective: predict epsilon (the noise), not x_0 (the clean image)
  2. The DDPM training algorithm: sample image, sample timestep, sample noise, create noisy image (closed-form), predict noise, compute MSE loss
- **Previous lesson load:** STRETCH (the-forward-process — 3 new concepts at the upper limit, heavy math)
- **This lesson's load:** BUILD — appropriate. Two new concepts, both built almost entirely from existing pieces. The training objective is MSE (DEVELOPED from Series 1). The training algorithm is the standard training loop (DEVELOPED from Series 2) with diffusion-specific inputs. The cognitive relief after the STRETCH is intentional and necessary.

### Connections to Prior Concepts

| Existing Concept | Connection | How |
|-----------------|------------|-----|
| MSE loss (Series 1.1) | The DDPM loss IS MSE loss. | Side-by-side formula comparison. "The same wrongness score from your third lesson." |
| Training loop (Series 1.1, 2.1) | The DDPM training loop IS the standard training loop with diffusion-specific data preparation. | Show the standard loop structure, highlight which steps are new (data prep) and which are identical (loss, backward, update). |
| Closed-form formula (the-forward-process) | The formula is used as a tool in the training algorithm to create noisy images at arbitrary timesteps. | The formula the student derived is now the data augmentation step. "That formula wasn't just elegant math — it's what makes training practical." |
| "Same building blocks, different question" (the-diffusion-idea) | Biggest payoff: the training objective uses MSE loss, gradient descent, backprop — the question is "what noise was added?" | Explicitly call back to this mental model. "We told you: same building blocks, different question. Here's the question." |
| Reconstruction loss (autoencoders, 6.1) | Another MSE loss but with a different target. Autoencoder: compare output to input. DDPM: compare predicted noise to actual noise. Three MSE targets now: labels (classification), input (reconstruction), noise (diffusion). | Brief comparison showing three uses of the same formula with different targets. Reinforces that MSE is a general tool. |
| "Alpha-bar is the signal-to-noise dial" (the-forward-process) | The network must learn what to do at every position on the dial. At low alpha-bar (high noise), it must hallucinate structure. At high alpha-bar (low noise), it must detect subtle perturbations. | The t=50 vs t=950 example connects to the dial metaphor and the multi-scale denoising from the-diffusion-idea. |

**Potentially misleading prior analogies:** None significant. The "same building blocks" analogy is reinforced, not contradicted. The "teleporting" metaphor for the closed-form formula is directly used (training teleports to a random timestep).

### Scope Boundaries

**This lesson IS about:**
- The DDPM training objective: predict epsilon, compute MSE loss
- Why predicting noise is preferred over predicting the clean image
- The complete training algorithm (pseudocode level)
- The simplified loss function from Ho et al. 2020
- Connecting MSE loss from Series 1 to the diffusion context

**This lesson is NOT about:**
- The reverse/sampling process algorithm (how to use the trained model to generate) — that is sampling-and-generation (Lesson 8)
- The U-Net architecture or any denoising network design — that is Module 6.3
- How the network receives the timestep t (conditioning mechanism) — Module 6.3
- Implementation in code — that is build-a-diffusion-model (Lesson 9)
- The full DDPM loss (with weighting terms) vs the simplified loss — mentioned but not derived
- Score matching, SDEs, continuous-time formulations — out of scope for the series
- The variational lower bound derivation of the DDPM loss — mentioned as context, not derived
- Classifier-free guidance or conditional generation — Module 6.3

**Target depth:** The training objective and algorithm reach DEVELOPED. The student can explain why predict noise, trace the algorithm step by step, and identify each piece's origin in prior lessons. They have not implemented it (APPLIED comes in Lesson 9).

### Lesson Outline

#### 1. Context + Constraints
What this lesson covers (the DDPM training objective) and what it does not cover (sampling/generation, architecture, implementation). Frame the scope: "You'll learn how the model is trained. How to use the trained model for generation is the next lesson."

#### 2. Recap (brief)
No gaps to fill, but a brief callback to the closed-form formula is warranted since it was just taught and is heavily used here. One sentence + formula reminder, not a re-derivation. "Last lesson, you derived this formula. Today, you'll see why it matters."

#### 3. Hook — "The Surprisingly Simple Objective"
Type: Misconception reveal / cognitive relief.

Set up the expectation: "You've derived the forward process math. The reverse process — learning to undo noise — must be at least as complex, right?" Then the reveal: the training objective is MSE loss on noise prediction. The student has been computing MSE loss since their third lesson in the course. Show the formula immediately and let the relief land.

Why this hook: The student is coming off a STRETCH lesson. The emotional trajectory of the module plan calls for "Wait, the training objective is just MSE on noise?" This hook delivers that moment right at the top.

#### 4. Explain — Why Predict Noise (Not the Clean Image)

The core conceptual section. Two-part structure:

**Part A: The question.** Given a noisy image x_t, the network could try to predict:
- Option 1: The clean image x_0 (seems intuitive — denoising means recovering the original)
- Option 2: The noise epsilon that was added (seems backwards — why predict what you want to remove?)

**Part B: Why noise wins.** Three reasons, in order of intuitive power:
1. **Consistent target distribution:** epsilon is always from N(0,1), regardless of image content. x_0 varies wildly (cats, dogs, landscapes). A consistent target makes optimization easier.
2. **Algebraic equivalence:** Given predicted epsilon, you can recover x_0 from the closed-form formula. Nothing is lost. Show the algebra: x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon_hat) / sqrt(alpha_bar_t).
3. **Empirical result:** Ho et al. 2020 found noise prediction gives better sample quality. This is a practical finding, not just theory.

**Modalities:** Verbal ("which target is easier?"), symbolic (the equivalence algebra), intuitive ("of course — a consistent target is easier to hit").

Address Misconception #1 (predicting x_0 instead of epsilon) directly in this section.

#### 5. Check — Predict and Verify
"Your colleague says predicting noise is wasteful because you have to do extra math to recover the clean image. How would you respond?" (The target consistency argument + the algebraic recovery formula.)

#### 6. Explain — The Training Algorithm

The procedural core of the lesson. Present the DDPM training algorithm step by step:

1. **Sample** a training image x_0 from the dataset
2. **Sample** a random timestep t ~ Uniform(1, T)
3. **Sample** noise epsilon ~ N(0, 1)
4. **Create** the noisy image: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon (the closed-form formula)
5. **Predict**: epsilon_hat = neural_network(x_t, t)
6. **Compute loss**: L = MSE(epsilon_hat, epsilon) = ||epsilon - epsilon_hat||^2
7. **Backpropagate** and update weights (standard training loop)

**Modalities:**
- Visual: flow diagram showing the pipeline (x_0 -> add noise -> x_t -> network -> epsilon_hat -> MSE with epsilon)
- Symbolic: pseudocode for the training loop
- Concrete example: trace one training step with T-shirt image at t=500

Address Misconceptions #2 and #3 here: the model does NOT iterate through timesteps. It jumps to one random timestep (using the formula from last lesson) and trains on that single noise level. Different images in the same batch can have different timesteps.

**Connection moment:** Map each step to the standard training loop. Steps 1-4 are data preparation (like loading a batch and preprocessing). Steps 5-6 are forward pass + loss computation. Step 7 is backward pass + optimizer step. "The heartbeat hasn't changed."

#### 7. Check — Trace a Training Step
Give the student a concrete scenario: x_0 is a sneaker image, t=200, epsilon is a specific noise vector. Ask them to describe each step of the algorithm. Then: "What would change if t=900 instead?" (The noisy image would be almost pure static, the network's job is much harder — it must hallucinate plausible structure.)

#### 8. Elaborate — Three Faces of MSE

Deeper nuance section. Show MSE loss in three contexts the student has now seen:

| Context | Prediction | Target | Source |
|---------|-----------|--------|--------|
| Linear regression | y_hat (predicted price) | y (actual price) | Series 1.1 |
| Autoencoder | x_hat (reconstructed image) | x (input image) | Module 6.1 |
| DDPM | epsilon_hat (predicted noise) | epsilon (actual noise) | This lesson |

Same formula. Same gradients. Same PyTorch code (nn.MSELoss). Different question each time.

Address Misconception #4 (MSE on noise is a different loss) by making the identity explicit.

Also briefly note:
- The DDPM paper derives the simplified loss from a variational lower bound (ELBO). The simplified version drops weighting terms and works better empirically. The student does not need the derivation — the simplified loss is what practitioners use.
- One network handles all timesteps. It receives t as an additional input. How the network uses t (timestep conditioning) is an architecture question deferred to Module 6.3.

Address Misconception #5 (separate network per timestep) briefly here.

#### 9. Check — Transfer Question
"You're explaining DDPM to a friend who knows the VAE from Module 6.1. They ask: 'So the VAE reconstructs images using MSE, and DDPM predicts noise using MSE — what's the fundamental difference?' How do you answer?" (The VAE compresses through a bottleneck and reconstructs. DDPM adds noise at a specific level and predicts what was added. The VAE's target is the input image. DDPM's target is the noise. The VAE learns a latent space; DDPM learns a denoising function. Same loss formula, completely different learning objectives.)

#### 10. Practice — Notebook Exercises (Colab)

No heavy implementation in this lesson — implementation is Lesson 9. But lightweight exercises that reinforce the concepts:

- **Exercise 1 (Guided): Create noisy images.** Use the closed-form formula to create noisy versions of Fashion-MNIST images at various timesteps. This is the data preparation step of training — the student practices using the formula as a tool. Predict-before-run: "What will the image look like at t=100 vs t=800?"

- **Exercise 2 (Guided): Compute the loss by hand.** Given a noisy image, a "predicted noise" tensor (random for now, since we have no trained model), and the actual noise, compute MSE loss manually and verify with nn.MSELoss. The student sees that the loss computation is identical to what they've done before.

- **Exercise 3 (Supported): Write the training pseudocode.** Fill in a skeleton of the DDPM training loop in Python/PyTorch. Not a working model (no network architecture), but the data preparation and loss computation steps. Tests whether the student can translate the algorithm into code structure.

- **Exercise 4 (Independent): Predict the loss landscape.** Given a trained model (hypothetical), which timesteps would have higher loss? (Middle timesteps, where the noise level creates the most ambiguity. Very low timesteps = easy, barely any noise to predict. Very high timesteps = also somewhat predictable because the image is nearly pure noise, so the prediction is close to the input noise.)

Exercises are independent (not cumulative). Each tests a different facet: using the formula (Ex 1), computing the loss (Ex 2), structuring the algorithm (Ex 3), reasoning about behavior (Ex 4).

#### 11. Summarize
Key takeaways:
1. The DDPM training objective is MSE loss between predicted and actual noise — the same loss formula from Series 1.
2. The model predicts noise (epsilon), not the clean image. This gives a consistent target regardless of image content, and you can recover x_0 algebraically.
3. Training samples one random timestep per image. The closed-form formula teleports to that timestep. No iterating through all 1000 steps.
4. The training loop is the standard loop (forward -> loss -> backward -> update) with diffusion-specific data preparation.

Echo the mental model: "Same building blocks, different question. The building blocks: MSE loss, backprop, gradient descent. The question: what noise was added to this image?"

#### 12. Next Step
"You now know how to train the model. But how do you use it? Starting from pure noise, how does the trained network iteratively denoise to create an image? That's the reverse process — the sampling algorithm — and it's the next lesson."

---

## Review — 2026-02-10 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 5
- Polish: 3

### Verdict: MAJOR REVISION

One critical finding (missing notebook) must be resolved before the lesson is usable. The lesson content itself is strong — the narrative arc delivers the intended cognitive relief, the MSE connection is well-made, and the training algorithm walkthrough is clear. However, several improvement findings would make the lesson significantly more effective for the student.

### Findings

#### [CRITICAL] — Notebook missing

**Location:** Practice section (Outline item 10)
**Issue:** The planning document specifies 4 exercises (Exercise 1: Create noisy images, Exercise 2: Compute loss by hand, Exercise 3: Write training pseudocode, Exercise 4: Predict the loss landscape). No notebook exists at `notebooks/6-2-3-learning-to-denoise.ipynb`. The lesson component has no ExercisePanel or notebook link.
**Student impact:** The student reads the lesson, understands the concepts conceptually, but never practices using the closed-form formula as a tool or computing the MSE loss in the diffusion context. Without hands-on practice, the training algorithm remains theoretical. The planning doc explicitly notes "The student practices using the formula as a tool" — that practice does not exist.
**Suggested fix:** Create the notebook with all 4 planned exercises. Exercise 1 (guided: create noisy images at various timesteps) and Exercise 2 (guided: compute MSE loss manually) are the most critical for reinforcing the lesson's core concepts. Add a notebook link and ExercisePanel to the lesson component.

---

#### [IMPROVEMENT] — Missing planned example: mini-batch with different timesteps

**Location:** Section 10 (Tracing a Training Step) and Section 9 (Misconceptions)
**Issue:** The planning document includes a fourth example: "Mini-batch: three images with different random timesteps (image A at t=200, image B at t=750, image C at t=45)." This example is absent from the built lesson. The lesson addresses the "one random timestep, not a sequence" misconception verbally (Section 9) and the check exercise (Section 11) asks about a single image with different timesteps. But the batch-level view — multiple images, each with different timesteps, contributing to one loss — is never concretely shown.
**Student impact:** The student understands that each training iteration picks one random timestep, but may not form a clear picture of what a real training batch looks like. The misconception about sequential timesteps is partially addressed, but the mini-batch example would drive the point home by showing the *batch-level* randomness. The student has experience with mini-batch training from Series 2 — this example would connect to that.
**Suggested fix:** Add a brief concrete example after Section 9 (the "One Random Timestep" misconception card) or at the end of Section 10. Show three images in one batch: "Image A (T-shirt) at t=200, Image B (sneaker) at t=750, Image C (handbag) at t=45. Each gets its own epsilon, its own x_t, its own epsilon_hat. MSE is averaged across the batch. Different noise levels, different images, one gradient update." This can be compact — a small table or a list of three bullet points.

---

#### [IMPROVEMENT] — "What if t=50" example has swapped coefficients

**Location:** Section 10, "What if t=50 instead?" block (lines 592-597)
**Issue:** The lesson states: "At t=50, alpha_bar_50 ≈ 0.95. The noisy image is 97.5% signal, 22.4% noise." These percentages are backwards — they are the coefficients from the t=500 example (sqrt(0.95) ≈ 0.975 and sqrt(0.05) ≈ 0.224), applied to the wrong alphas. For alpha_bar=0.95: sqrt(0.95) ≈ 0.975 for signal, sqrt(0.05) ≈ 0.224 for noise. So "97.5% signal, 22.4% noise" is actually correct for the signal/noise coefficients at t=50 — but the phrasing echoes the exact same numbers from the t=500 example ("0.224 · x_0 + 0.975 · epsilon") and could confuse the student who just read those numbers in the opposite roles. More importantly, the student may wonder why the same numbers appear in both examples.
**Student impact:** The student just read "0.224 · x_0 + 0.975 · epsilon" for t=500 (22.4% signal, 97.5% noise). Now they see "97.5% signal, 22.4% noise" for t=50 and might think it is an error, or might not notice that the roles have exactly swapped. The symmetry is coincidental (sqrt(0.05) and sqrt(1-0.05) vs sqrt(0.95) and sqrt(1-0.95)) but not explained, so it looks like a copy-paste error.
**Suggested fix:** Make the symmetry explicit: "At t=50, alpha_bar ≈ 0.95 — the mirror image of t=500. The coefficients swap roles: now the signal coefficient is 0.975 and the noise coefficient is 0.224." Or simply use different numbers: pick t=100 with alpha_bar ≈ 0.85, giving sqrt(0.85) ≈ 0.922 for signal and sqrt(0.15) ≈ 0.387 for noise, which avoids the confusing symmetry entirely.

---

#### [IMPROVEMENT] — Misconception #4 ("MSE on noise is a different loss") addressed late

**Location:** Section 12 (Three Faces of MSE), lines 674-764
**Issue:** The planning document lists Misconception #4 as "MSE on noise is a different loss function from the MSE I already know" and plans to address it at "The 'surprisingly simple' reveal moment." The built lesson addresses it much later — in Section 12 (Three Faces of MSE), which comes after the training algorithm walkthrough, both checks, and the misconception cards. By Section 12, the student has already been working with the MSE loss for several sections. The side-by-side comparison that makes the identity unmistakable arrives as a late recap rather than an early "of course" moment.
**Student impact:** The lesson does mention MSE familiarity in the hook (Section 4, "the same loss you have been computing since Loss Functions") and in Step 6 of the training algorithm ("the same MSE formula"). But the powerful side-by-side comparison — three rows, same formula, different letters — comes 400+ lines into the lesson. A student who is uncertain about whether the diffusion MSE is "really the same" must wait a long time for definitive confirmation.
**Suggested fix:** Consider placing a compact version of the side-by-side comparison earlier — right after the hook (Section 4), or immediately after the first "Check" (Section 6). It does not need to be the full Three Faces table; even a brief "Write them side by side" moment with the Series 1 MSE and the DDPM MSE would land the insight earlier. The fuller Three Faces table can remain in Section 12 as an elaboration that adds the autoencoder context.

---

#### [IMPROVEMENT] — The "open-book exam" analogy is introduced but not developed

**Location:** Section 4 (The Surprisingly Simple Objective), lines 164-169
**Issue:** The planning document describes a key analogy: "The training objective is an open-book exam: the network always has the answer key (epsilon) and we measure how close its guess is." The built lesson introduces this analogy in one paragraph — "noisy image (the question), guesses which noise was added (its answer), check against the actual noise (the answer key)" — but then does not use it again in the training algorithm walkthrough. The PhaseCards in Section 7 describe Step 3 as "the answer key" (good), but Steps 5-6 do not call back to the exam analogy.
**Student impact:** The analogy is effective but fleeting. If the student is forming their mental model of the training algorithm, the exam analogy could serve as a unifying frame ("Step 3: prepare the answer key. Step 5: the student takes the exam. Step 6: grade the exam."). Without this reinforcement, the analogy is a one-time metaphor rather than a recurring mental model.
**Suggested fix:** Either: (a) Briefly reference the exam analogy in the PhaseCards for Steps 5 and 6 ("The network takes the exam" / "Grade the exam — MSE is the score"), or (b) Add a brief sentence after the PhaseCards: "That is the complete exam cycle: prepare the answer key (Step 3), give the student the question (Step 5), grade (Step 6), learn from mistakes (Step 7)." This is a small change that strengthens the analogy's staying power.

---

#### [IMPROVEMENT] — No interactive widget or visual beyond the Mermaid diagram

**Location:** Entire lesson
**Issue:** The planning document specifies 5 modalities. The built lesson delivers: verbal/analogy (the exam analogy, MSE callback), symbolic (formulas), concrete example (the traced training step), visual (Mermaid flow diagram), and intuitive ("which target is easier?"). However, the visual modality is limited to a single Mermaid diagram (Section 8, lines 458-476). There is no interactive element in the lesson at all — no widget, no slider, no draggable component. Previous lessons in this module (the-diffusion-idea, the-forward-process) had rich interactive widgets. For a BUILD lesson, the lack of interactivity is not fatal, but it is a missed opportunity. The training algorithm walkthrough (sampling noise levels, seeing how MSE changes at different timesteps) lends itself to an interactive experience.
**Student impact:** The student reads a well-written text-and-formula lesson, but never interacts with the concepts. After two lessons with engaging widgets (DiffusionNoiseWidget, AlphaBarCurveWidget), a text-only lesson may feel less engaging. The Mermaid diagram is static.
**Suggested fix:** This is an improvement, not critical. Options: (a) A minimal widget that lets the student pick a timestep t and see the resulting signal/noise coefficients and a visual of the noisy image (could reuse the procedural T-shirt image from earlier widgets). (b) A side-by-side showing the noisy image at the student-selected timestep with the formula coefficients updating live. (c) If widget work is deferred to after the notebook, note it as a future enhancement. The lesson functions without a widget, but a BUILD lesson is the ideal place for a "play with it" moment.

---

#### [POLISH] — Em dash spacing is correct throughout

**Location:** Entire lesson
**Issue:** Checked all em dashes in the lesson. All use the `&mdash;` entity or `\u2014` with no surrounding spaces. This is correct per the Writing Style Rule. No finding here — noted for completeness.
**Student impact:** None.
**Suggested fix:** None needed.

---

#### [POLISH] — Cursor style on `<details>` summary elements

**Location:** Sections 6, 11, 15 (the three check/reveal blocks)
**Issue:** The `<summary>` elements use `cursor-pointer` class (lines 294, 642, 848). This is correct. No finding — noted for completeness per the Interaction Design Rule.
**Student impact:** None.
**Suggested fix:** None needed.

---

#### [POLISH] — "Cosine schedule" in concrete example not previously established at this specificity

**Location:** Section 10 (Tracing a Training Step), line 545
**Issue:** The concrete example states "alpha_bar_500 ≈ 0.05 (using a cosine schedule — only 5% signal remains at the midpoint)." The student knows about cosine vs linear schedules at INTRODUCED depth from the-forward-process. The parenthetical clarification is helpful. However, the specific value "alpha_bar_500 ≈ 0.05 for cosine" is new information the student has not computed or seen before. It is presented as a given rather than derived or connected to the widget they used.
**Student impact:** Minor. The student has seen the alpha-bar curve widget with both schedule types. The specific value is plausible and the lesson is using it as a concrete number, not teaching it. The student might briefly wonder "how do I know alpha_bar_500 is 0.05?" but will likely accept it given their widget experience.
**Suggested fix:** Add a brief connection: "alpha_bar_500 ≈ 0.05 (recall the alpha-bar curve you explored — at the midpoint of a cosine schedule, the signal fraction has already dropped to 5%)." This connects the number to the student's widget experience rather than presenting it as a fact from nowhere.

### Review Notes

**What works well:**
- The narrative arc delivers on the BUILD intent. The lesson genuinely feels like cognitive relief after the-forward-process. The hook ("the surprisingly simple objective") lands well.
- The training algorithm walkthrough (Section 7) is excellent. The PhaseCards with color coding (blue for familiar, violet for new) make the pattern immediately visible. The aside explicitly calls out what is new vs familiar.
- The "Three Faces of MSE" table is a strong closing synthesis. The side-by-side formulas make the identity unmistakable.
- The checks are well-designed. The first check (Section 6) tests the noise-vs-image reasoning. The second check (Section 11) tests algorithm tracing. The transfer check (Section 15) tests comparative understanding. Good scaffolding progression.
- Scope boundaries are well-maintained. The lesson does not drift into architecture, sampling, or implementation. The "Architecture Is Later" aside (Section 13) cleanly defers.
- The "One Network for All Timesteps" card (Section 13) addresses Misconception #5 concisely and at the right location.
- Connection to prior lessons is strong throughout — the closed-form formula, MSE from Series 1, training loop from Series 2, reconstruction loss from autoencoders. The student's prior knowledge is consistently activated.

**Patterns to watch:**
- The notebook gap is the most urgent issue. The lesson teaches the training algorithm conceptually, but the planning document explicitly designed exercises to make the student practice using the closed-form formula as a data preparation tool and computing MSE loss in the new context. Without the notebook, the lesson reaches INTRODUCED depth for the training algorithm, not DEVELOPED.
- The lesson is text-heavy for a module that has been widget-rich. This is acceptable for a BUILD lesson, but worth noting as a pattern — if the next lesson (sampling-and-generation) is also text-only, the module may lose the interactive momentum established in lessons 1-2.

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 2 new concepts (within limit of 3)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated
