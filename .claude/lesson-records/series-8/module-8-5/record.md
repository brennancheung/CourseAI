# Module 8.5: Preference Optimization Deep Dives -- Record

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Bradley-Terry preference model -- converting pairwise score differences to probabilities via sigmoid | DEVELOPED | direct-preference-optimization | P(B preferred over A) = sigma(r_B - r_A). Taught with chess Elo analogy (connected to Chatbot Arena from 5.1.4) and concrete trace using reward scores 0.7 vs 0.3 from 4.4.3. Student can state the model, explain why it is used, and compute preferences from reward differences. |
| Sigma / logistic function as probability mapping | DEVELOPED | direct-preference-optimization | sigma(x) = 1/(1+exp(-x)). Inline definition given at point of use. Connected to sigmoid activations from Series 1. Properties: sigma of large positive ~ 1, large negative ~ 0, zero = 0.5. |
| RLHF objective in formal notation -- KL-constrained reward maximization | DEVELOPED | direct-preference-optimization | max E[r(x,y)] - beta * KL(pi || pi_ref). Each term explained: expected reward, KL divergence (inline definition given), beta as tradeoff hyperparameter. Student can write and explain each term. |
| KL divergence formal definition | INTRODUCED | direct-preference-optimization | sum_y pi(y|x) * log(pi(y|x)/pi_ref(y|x)). Inline definition. Previously only informally understood as "how far the model has drifted." |
| Closed-form optimal policy for KL-constrained RLHF -- pi* = (1/Z) * pi_ref * exp(r/beta) | DEVELOPED | direct-preference-optimization | Student understands the form: reference policy reweighted by exponentiated reward. Derivation shown via four intermediate algebra steps (multiply/divide, take log, rearrange). Knows Z(x) is the partition function. Understands why the reference model appears (falls out of the math from the KL constraint). |
| Partition function Z(x) -- normalization in the optimal policy formula | INTRODUCED | direct-preference-optimization | Brief definition as normalizing constant ensuring probabilities sum to 1. Key property: it depends only on x (the prompt), not on y (the response), so it cancels in the DPO loss derivation. |
| Implicit reward model -- extracting a reward function from any policy | INTRODUCED | direct-preference-optimization | r(y,x) = beta * log(pi(y|x)/pi_ref(y|x)) + beta * log(Z(x)). Any policy paired with a reference implicitly defines a reward function. Student understands the insight but does not use it independently. Taught via "editor analogy extension" -- writer who has learned to write well implicitly embodies editorial standards. |
| DPO loss function -- derivation, structure, and behavior | DEVELOPED | direct-preference-optimization | L_DPO = -E[log sigma(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))))]. Student can write the loss, explain each term, trace through a numerical example. Derivation shown step-by-step: RLHF objective -> optimal policy -> rearrange for reward -> substitute into Bradley-Terry -> Z(x) terms cancel. |
| Log-ratio as the coordinate system for DPO -- how much the policy has shifted from reference | DEVELOPED | direct-preference-optimization | Extension of log-probabilities from 4.1.1/4.3.2. The DPO loss operates on log-ratios (log pi - log pi_ref) not absolute log-probs. Negative example (without reference model) shows why absolute log-probs lead to probability mass collapse. |
| DPO gradient behavior -- focuses on hard cases where model disagrees with preferences | DEVELOPED | direct-preference-optimization | Sigmoid structure means gradient is steep when the model prefers the dispreferred response, near zero when the model already agrees. DpoLossLandscape SVG visualizes this. Taught via predict-before-reveal check and numerical example. |
| Reference model as structural anchor -- prevents probability mass collapse | DEVELOPED | direct-preference-optimization | Negative example shown: without reference model, loss reduces to "maximize preferred minus dispreferred absolute log-probs," causing probability collapse. ComparisonRow (without reference vs with reference) makes the contrast explicit. Taught as a structural requirement, not optional regularization. |
| Beta hyperparameter -- controls conservatism of DPO optimization | DEVELOPED | direct-preference-optimization | Large beta = strict (stay close to reference). Small beta = loose (allow more drift). Setting beta to zero removes KL constraint (enables reward hacking). Worked through in numerical example (beta = 0.1). |
| DPO implementation in PyTorch -- the training loop | APPLIED | direct-preference-optimization | ~15-line loss function (four forward passes -> log-ratios -> loss). Full training loop: load model as policy, frozen copy as reference, iterate over preference pairs, compute loss, backpropagate, update. Reference model is frozen (no gradient). Log-probs of full response = sum of per-token log-probs. |
| DPO is not an approximation of RLHF -- exact equivalence under Bradley-Terry | DEVELOPED | direct-preference-optimization | Explicit misconception addressed. Derivation shows exact equivalence: same optimal policy satisfies both objectives. "Not an approximation" WarningBlock shown immediately after presenting the DPO loss formula. |
| Pi* to pi substitution in the DPO loss -- how gradient descent drives the current policy toward the optimum | DEVELOPED | direct-preference-optimization | Explicit paragraph added after DPO loss formula. Loss is minimized at pi = pi*, so gradient descent drives current policy toward optimum. Addresses the "is this valid?" confusion from substituting pi for pi* in the preference model. |

## Lesson Summaries

### Lesson 1: Direct Preference Optimization (direct-preference-optimization)

**Concepts taught:**
- Bradley-Terry preference model: DEVELOPED
- Sigma / logistic function: DEVELOPED
- RLHF objective in formal notation: DEVELOPED
- KL divergence formal definition: INTRODUCED (inline, at point of use)
- Closed-form optimal policy for KL-constrained RLHF: DEVELOPED
- Partition function Z(x): INTRODUCED
- Implicit reward model: INTRODUCED
- DPO loss function -- derivation, structure, gradient behavior: DEVELOPED
- Log-ratio as coordinate system: DEVELOPED
- Reference model as structural anchor: DEVELOPED
- Beta hyperparameter: DEVELOPED
- DPO implementation in PyTorch: APPLIED
- DPO = RLHF under Bradley-Terry (not an approximation): DEVELOPED
- Pi* to pi substitution validity: DEVELOPED

**Mental models established:**
- "The reward model was always inside the policy" -- the central DPO insight. Any policy paired with a reference implicitly defines a reward function via the log-ratio. The reward model is not missing from DPO; it is absorbed into the policy.
- "The derivation roadmap as scaffold" -- four-step path (RLHF objective -> optimal policy -> implicit reward -> DPO loss) previewed before the derivation. Student always knows where they are in the argument.
- "Log-ratios, not absolutes" -- DPO measures how much the policy has CHANGED relative to the reference, not the absolute probability of any response. The reference model defines the coordinate system.
- "DPO's gradient focuses on the hard cases" -- sigmoid structure means learning effort concentrates where the model most disagrees with human preferences. Gradient near zero when model already agrees.
- "Not an approximation" -- DPO eliminates the reward model by solving for it analytically (exactly), not by ignoring it. The optimal policy is identical to the RLHF optimum under Bradley-Terry.

**Analogies used:**
- Chess Elo ratings as a Bradley-Terry application (grounds the preference model in familiar territory; Chatbot Arena connection from 5.1.4)
- "The editor was always inside the writer" -- extension of the "reward model as experienced editor" analogy from 4.4.3. Once a writer has internalized editorial standards, you can extract the reward from the policy by measuring how much it has shifted from the reference.
- Derivation roadmap (four connected boxes -- spatial scaffold for a multi-step mathematical argument)

**How concepts were taught:**
- **Recap:** Three beats (alignment problem -> RLHF approach -> DPO overview). Re-activated "SFT gives voice, alignment gives judgment," "experienced editor," "continuous freeze the backbone." GradientCard: "You know WHAT. This lesson explains WHY."
- **Bradley-Terry:** Motivated from the existing problem (preference pairs + reward scores), introduced sigma function inline, chess Elo example, concrete trace with reward scores 0.7 vs 0.3 from 4.4.3. GradientCard aside: "Not DPO-specific -- dates to 1952."
- **RLHF objective:** Formal notation with inline KL definition. Term-by-term explanation (reward, KL, beta). Beta direction clarified: large beta = strict, small beta = loose, beta -> 0 = reward hacking (NOT beta -> infinity).
- **Optimal policy:** Closed-form result stated, four intermediate algebra steps shown in styled box. Reference model appears structurally from the KL constraint. "Roadmap: Step 1 Complete" InsightBlock.
- **Implicit reward:** Rearrangement shown explicitly (multiply/divide, take log, multiply by beta). "The reward was always inside" framing. Substitution into Bradley-Terry shown. Z(x) cancellation highlighted. DPO loss formula in prominent styled box.
- **Pi* to pi:** Explicit paragraph after the loss formula explaining why substituting the current policy is valid (loss is minimized at pi = pi*).
- **Not an approximation:** WarningBlock immediately after the DPO loss formula.
- **Check 1 (Predict the Gradient):** Two cases (model agrees vs disagrees). Predict-before-reveal. DpoLossLandscape SVG: loss vs log-ratio difference with shaded regions labeled "Model disagrees / steep gradient" and "Model agrees / flat gradient."
- **Numerical example:** Quantum computing preference pair from 4.4.3. Table of log-probabilities, step-by-step computation (log-ratios -> difference -> multiply by beta -> apply sigma -> loss). Convergence case shown (larger log-ratio difference -> smaller loss).
- **Negative example:** DPO without reference model. Loss formula shown, probability collapse mechanism explained, ComparisonRow (without reference vs with reference) makes contrast explicit. "The Reference Model Is Structural" GradientCard.
- **Implementation:** PhaseCards for four steps (compute log-probs, compute log-ratios, compute loss, backpropagate). ~15-line PyTorch code block. Reference model frozen detail highlighted.
- **Implicit reward:** Revisited as an elaboration after implementation. Connecting back to the design space from 5.1.2 (no EXPLICIT reward model; implicit one defined by the policy).
- **Check 2 (Transfer):** "DPO is just supervised learning -- nothing interesting about the loss." Reveal: four reasons it is different (pairs, log-ratios, sigmoid focuses on hard cases, equivalent to RLHF).

**Visual elements:**
- DerivationRoadmap: Inline SVG, four connected boxes (RLHF Objective -> Optimal Policy -> Implicit Reward -> DPO Loss). Used as scaffold and referenced with "Step N Complete" callouts throughout.
- DpoLossLandscape: Inline SVG, -log(sigma(x)) plotted with red-shaded region (model disagrees, steep gradient) and green-shaded region (model agrees, flat gradient). Axis labels "log-ratio difference (preferred - dispreferred)."
- Log-probability table: HTML table with policy and reference log-probs for preferred and dispreferred responses.
- ComparisonRow: Without reference (probability collapse) vs with reference (log-ratios, anchored optimization).
- PhaseCards: Four steps of DPO implementation (cyan, blue, violet, emerald).

**What is NOT covered:**
- PPO algorithm details (clipping, advantage estimation, value functions) -- only the RLHF objective is used
- DPO variants (IPO, KTO, ORPO) -- INTRODUCED in 5.1.2; the next step card points there
- Constitutional AI or RLAIF
- Training a reward model from scratch
- Implementing PPO for comparison
- Frontier model alignment practices or scaling behavior
- Formal Lagrange multiplier proof in full generality -- key steps and result shown, not measure-theoretic proof

**Notebook:** `notebooks/8-5-1-direct-preference-optimization.ipynb` (4 exercises)
- Exercise 1 (Guided): Verify DPO loss by hand. Pre-computed log-probs for 5 preference pairs. Compute loss for each, vary beta. Insight: beta controls conservatism; falls directly from the formula.
- Exercise 2 (Supported): Implement the DPO loss function in PyTorch. Verify against Exercise 1 hand calculations. Compute gradient via autograd, inspect direction. Insight: ~5-10 lines, each line maps to a derivation step.
- Exercise 3 (Supported): Train GPT-2 small with DPO. Full loop: policy + frozen reference, preference pairs, loss, backprop, update. Compare outputs before and after. Insight: training loop looks like supervised learning; complexity is in the loss function.
- Exercise 4 (Independent): Explore implicit reward. Compute log-ratios for several responses, verify preferred responses have higher implicit reward. Generate a new response not in training data, compute its implicit reward. Insight: implicit reward generalizes beyond training data.

**Review:** Passed at iteration 2/3. Iteration 1 had 1 critical finding (beta direction backwards: "set beta to infinity" -> "set beta to zero"), 2 improvement findings (missing intermediate algebra steps from optimal policy to implicit reward; pi* to pi substitution not explained), and 2 polish findings (Elo example mixed logarithm conventions; Exercise 4 scaffolding label inconsistent). All critical and improvement findings resolved. Iteration 2 had 0 critical/improvement findings and 1 polish finding (notebook Exercise 4 heading still said "Minimal Scaffolding" -- minor, noted for fix independently).

**Cognitive load:** STRETCH. First multi-step mathematical derivation in the course. Three new concepts (Bradley-Terry, closed-form optimal policy, DPO loss). Mitigated by heavy scaffolding: derivation roadmap previewed, step-by-step algebra shown, numerical example grounding the abstract result, exercises start with guided arithmetic.
