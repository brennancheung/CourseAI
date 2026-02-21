# Lesson 4: Reasoning Models (reasoning-models) -- Planning Document

**Module:** 5.2 Reasoning & In-Context Learning
**Position:** Lesson 4 of 4 (module capstone)
**Type:** BUILD (follows STRETCH chain-of-thought)
**Slug:** reasoning-models

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Intermediate tokens as computation (autoregressive loop as computational amplifier) | DEVELOPED | chain-of-thought (5.2.3) | Core mental model: each generated token triggers another forward pass. More tokens = more forward passes = more computation. The student can explain this mechanistically and identify when it helps. |
| Fixed computation budget per forward pass (N transformer blocks, no difficulty knob) | DEVELOPED | chain-of-thought (5.2.3) | The student understands that every token prediction runs through the same N blocks regardless of difficulty, and that CoT is the mechanism for exceeding this budget. |
| When CoT helps vs does not (computational complexity criterion) | DEVELOPED | chain-of-thought (5.2.3) | Student can distinguish problems that benefit from CoT (multi-step, exceeding single-pass capacity) from those that do not (factual recall, classification). |
| Process supervision vs outcome supervision | INTRODUCED | chain-of-thought (5.2.3) | Student has seen the distinction: outcome evaluates final answer only, process evaluates each step. Concrete example of two chains with same correct answer but different reasoning quality. Student recognizes the distinction but has not explored WHY process supervision matters for training or HOW it changes learning dynamics. |
| Self-consistency / majority voting | MENTIONED | chain-of-thought (5.2.3) | Student has heard the term: generate multiple chains, take majority-vote answer. No implementation details, no depth on why it works or when it fails. Name recognition only. |
| CoT error propagation (errors in intermediate steps corrupt downstream) | INTRODUCED | chain-of-thought (5.2.3) | Student has seen that wrong intermediate steps propagate. Connected to ICL ordering sensitivity and context stuffing. Knows the model does not catch its own mistakes. |
| Zero-shot CoT vs few-shot CoT | DEVELOPED | chain-of-thought (5.2.3) | Student can explain both techniques and connect them to prompt engineering (format instruction) and ICL (examples as program). |
| RLHF pipeline (pretraining -> SFT -> RL) | DEVELOPED | constitutional-ai (5.1.1), recap from 4.4 | Student has the full pipeline at DEVELOPED depth across multiple lessons. Knows reward models, preference pairs, PPO mechanics at conceptual level. |
| Reward hacking / proxy optimization | DEVELOPED | rlhf-and-alignment (4.4.3) | Student has seen models optimizing against the reward model proxy rather than the intended objective. Goodhart's law applied in both training (4.4.3) and evaluation (5.1.4) contexts. |
| "The prompt is a program, attention is the interpreter" | DEVELOPED | in-context-learning (5.2.1) | Central mental model for the module. Extended through prompt engineering (programming better programs) and CoT (giving the program more compute steps). |

**Mental models and analogies already established:**
- "A forward pass is the model's thinking budget per token. CoT is spending more budget by generating more tokens." (chain-of-thought)
- Scratchpad/working memory analogy: intermediate tokens as external memory (chain-of-thought)
- "Thinking budget" per token = one forward pass through N blocks (chain-of-thought)
- Autoregressive generation as a feedback loop, grounded in generate() code from building-nanogpt (chain-of-thought, building-nanogpt)
- "Same pipeline, different data source" for alignment variations (constitutional-ai)
- Reward model as "experienced editor" with blind spots (4.4.3)
- Design space axes framework for preference optimization (alignment-techniques-landscape)

**What was explicitly NOT covered in prior lessons (relevant here):**
- Reasoning models trained with RL to use CoT effectively (explicitly deferred from chain-of-thought)
- Test-time compute scaling (explicitly deferred)
- Search during inference, tree-of-thought, or beam search over reasoning paths (explicitly deferred)
- Self-consistency implementation details (only MENTIONED in chain-of-thought)
- How process supervision changes training dynamics (only INTRODUCED the distinction)

**Readiness assessment:** The student is well-prepared. The entire conceptual foundation is in place: tokens-as-computation is DEVELOPED, the RLHF pipeline is DEVELOPED, and process vs outcome supervision has been INTRODUCED with a concrete example. This lesson connects two established frameworks (CoT mechanism + RL training) rather than introducing a fundamentally new paradigm. The BUILD designation is appropriate.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how reinforcement learning trains models to generate effective reasoning chains, why spending more computation at inference time (test-time compute scaling) can substitute for larger models, and how this represents a paradigm shift from scaling model size to scaling inference computation.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Intermediate tokens as computation | DEVELOPED | DEVELOPED | chain-of-thought (5.2.3) | OK | Core mechanism that reasoning models optimize. Student must understand WHY more tokens = more compute to understand WHY training models to generate better tokens matters. |
| Fixed computation budget per forward pass | DEVELOPED | DEVELOPED | chain-of-thought (5.2.3) | OK | The constraint that reasoning models address. Must be at DEVELOPED to understand the paradigm shift. |
| Process supervision vs outcome supervision | INTRODUCED | INTRODUCED | chain-of-thought (5.2.3) | OK | This lesson DEVELOPS it from INTRODUCED. Student has the distinction and a concrete example; this lesson builds on it. No gap. |
| Self-consistency / majority voting | MENTIONED | MENTIONED | chain-of-thought (5.2.3) | OK | This lesson DEVELOPS it from MENTIONED. Student has name recognition; this lesson teaches the mechanism and connects it to search. No gap -- the lesson plan accounts for building up from MENTIONED. |
| CoT error propagation | INTRODUCED | INTRODUCED | chain-of-thought (5.2.3) | OK | Motivation for why training models to reason better matters. INTRODUCED is sufficient -- this lesson uses the concept, does not require the student to apply it independently. |
| RLHF pipeline | DEVELOPED | DEVELOPED | 5.1.1 / 4.4 | OK | Must understand RL training at conceptual level to understand RL for reasoning. |
| Reward hacking / proxy optimization | DEVELOPED | DEVELOPED | 4.4.3 | OK | Connects to outcome reward as proxy and why it can be gamed. |
| When CoT helps vs does not | DEVELOPED | DEVELOPED | chain-of-thought (5.2.3) | OK | Student must know the computational complexity criterion to understand why reasoning models allocate variable compute. |

All prerequisites are at sufficient depth. No gaps.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Reasoning models are just bigger models" | Bigger models have been the dominant scaling paradigm for years (GPT-2 -> GPT-3 -> GPT-4). Natural to assume better performance = more parameters. The student has seen scaling as the answer to many problems. | Same-size model with and without reasoning training: the reasoning-trained model solves problems the base model cannot, despite identical parameter count. A 7B reasoning model outperforming a 70B base model on math benchmarks. Size did not change; inference-time computation did. | Explain section, immediately after introducing RL for reasoning. Address head-on because this is the core paradigm shift of the lesson. |
| "The model 'decides' to think harder on harder problems" | Anthropomorphic framing is natural -- when we face a hard problem, we decide to think more carefully. The previous lesson established that CoT helps on harder problems, which sounds like the model "knows" when to try harder. | The base model generating CoT has no mechanism for calibrating effort. It generates tokens until it produces a stop token or hits a length limit. It cannot introspect on problem difficulty. Reasoning models learn to generate more tokens on harder problems through RL reward signals, not through understanding difficulty. The "decision" is a learned policy, not deliberation. | After explaining RL training for reasoning. The RL mechanism explains HOW variable compute emerges without anthropomorphizing. |
| "Process supervision means a human checks every step" | The student learned process supervision in the previous lesson as "evaluating each step individually." The natural image is a human reading each step and scoring it, which would be impossibly expensive. This makes process supervision seem impractical. | Process reward models (PRMs) are trained on step-level labels (which may initially come from humans or from automated verification) and then evaluate steps at scale without human involvement. Like the RLAIF insight from 5.1.1: the initial human signal gets scaled through AI. Same pattern: human criteria -> AI evaluator. | When developing process supervision from INTRODUCED to DEVELOPED. Connect explicitly to the RLAIF pattern. |
| "Test-time compute scaling means running the model on better hardware" | "Compute" in the student's experience often means hardware (GPUs, TPUs). "Scaling compute" sounds like buying more GPUs. | Test-time compute scaling means generating more tokens (more forward passes through the same model on the same hardware). A model running on identical hardware can use 10x more compute on a hard problem by generating 10x more reasoning tokens. The hardware is the same; the number of forward passes changes. | When introducing test-time compute scaling. Clarify immediately that "compute" here means forward passes, not hardware. Connect to the established "tokens as computation" mental model. |
| "More reasoning tokens are always better" | Natural extension of "more tokens = more computation = better answers." If some CoT helps, more must help more. | Diminishing returns and overthinking. On simple problems, long reasoning chains introduce errors and waste compute (callback to CoT lesson: adding "let's think step by step" to factual recall). On problems within the model's capacity, excessive reasoning can lead to second-guessing correct initial answers. The optimal amount of reasoning compute depends on problem difficulty -- this is precisely what test-time compute scaling calibrates. | Elaborate section. Connects to CoT error propagation (INTRODUCED in previous lesson) and the computational complexity criterion (DEVELOPED). |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Math problem solved by base model with CoT (sometimes right, sometimes wrong) vs reasoning model (consistently right) | Positive | Show that RL training produces reliably better reasoning chains, not just longer ones. Same problem, same architecture, different training. | Directly extends the 17x24 example from the CoT lesson. Student already has the "before" (base model + CoT prompt). Now sees the "after" (RL-trained reasoning). Concrete and comparable. |
| Self-consistency: 5 reasoning chains for a word problem, 3 correct and 2 wrong, majority vote yields correct answer | Positive | DEVELOP self-consistency from MENTIONED. Show how sampling multiple chains and voting reduces error probability. Concrete numbers make the mechanism tangible. | Builds directly on what was MENTIONED. Uses a simple enough example that the student can verify each chain. Connects to CoT error propagation: if one chain can err, multiple chains provide redundancy. |
| Simple factual question where reasoning model wastes tokens overthinking and arrives at wrong answer through second-guessing | Negative | Disproves "more reasoning is always better." Shows that reasoning compute has diminishing and even negative returns on problems that do not need it. | Mirrors the negative example from the CoT lesson ("What is the capital of France?" with CoT). Extends it: even a model TRAINED to reason can over-apply reasoning. The computational complexity criterion still applies. |
| "Bigger model vs more thinking time" tradeoff: 7B reasoning model vs 70B base model on competition math | Stretch | The paradigm shift example. Shows that scaling inference compute can substitute for scaling parameters. Neither approach dominates -- they are different points in a tradeoff space. | This is the "of course" moment for the paradigm shift. The student already knows tokens-as-computation and that RL can train reasoning. The stretch example connects these into the implication: you can trade model size for inference compute. |

### Gap Resolution

No gaps identified. All prerequisites are at sufficient depth.

---

## Phase 3: Design

### Narrative Arc

The previous lesson revealed that chain-of-thought works because intermediate tokens give the model additional forward passes -- more computation per problem. But there is a problem: the base model has no idea how to use this extra computation well. When you add "let's think step by step" to a prompt, you are hoping the model generates useful intermediate tokens, but there is no guarantee. Sometimes the reasoning is correct. Sometimes it is wrong. Sometimes the model rambles. The quality of the reasoning chain is left to chance -- to whatever patterns the model absorbed during pretraining. What if, instead of hoping the model reasons well, you could *train* it to reason well? This is what reasoning models do. They apply reinforcement learning -- the same RL the student already knows from RLHF -- but with a different reward signal: instead of "is this response helpful and harmless?" the reward is "did you get the right answer?" And once you have a model that can reliably reason through problems, a new possibility opens up: instead of building bigger models, you can let the model think longer. This is the paradigm shift from scaling model size to scaling inference computation.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Verbal/Analogy** | "Training the scratchpad" -- the CoT lesson established the scratchpad analogy (intermediate tokens as working memory). This lesson extends it: RL training teaches the model to use the scratchpad effectively, like training a student to show their work in a structured, checkable way rather than scribbling randomly. | Extends an established analogy rather than introducing a new one. The student already has "scratchpad = working memory." Adding "RL trains effective scratchpad use" is a single conceptual step. |
| **Verbal/Analogy** | "Bigger brain vs more thinking time" -- for the paradigm shift. A student who thinks for 30 minutes outperforms a student who glances for 5 seconds, even if the second student is "smarter." The question becomes: when is it better to get a smarter student vs giving the current student more time? | Maps to familiar experience. Makes the tradeoff tangible before formalizing it. The student (software engineer) has experienced this with code review -- sometimes a quick glance suffices, sometimes you need to trace through carefully. |
| **Visual (inline SVG)** | Scaling paradigm diagram: two axes (model size, inference compute), with traditional scaling as movement along the model-size axis and test-time compute scaling as movement along the inference-compute axis. Shows that both increase total computation but through different mechanisms. | The paradigm shift is spatial -- two independent axes of scaling. A diagram makes the independence visible. Connects to the design space axes framework from alignment-techniques-landscape (5.1.2), reinforcing the "axes not ladder" mental model. |
| **Concrete example** | Self-consistency worked example: 5 chains for a word problem, with actual reasoning shown, 3 correct and 2 wrong, majority vote selecting the correct answer. Then: what if you generated 20 chains? The probability of majority-correct increases. | DEVELOPS self-consistency from MENTIONED to DEVELOPED. Concrete numbers (3/5, then extrapolated to 20) make the statistical argument tangible. The student can verify each chain, building trust in the mechanism. |
| **Symbolic/Code** | Pseudocode for the RL training loop: generate chain -> check answer -> compute reward -> update policy. Annotated to show what differs from standard RLHF (reward signal source, what is being optimized). | Connects to the student's DEVELOPED understanding of RLHF. The pseudocode makes the connection explicit: same loop, different reward. No new mechanism, different application. |
| **Intuitive** | The "of course" beat: "You already knew that tokens are computation (CoT lesson). You already knew that RL can optimize behavior toward a reward signal (RLHF). Of course you can use RL to optimize how the model generates reasoning tokens. And of course, if reasoning tokens provide more computation, a model that reasons for longer gets more computation without needing more parameters." | Two established facts combine into the new insight. The "of course" moment collapses the conceptual distance. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. RL for reasoning (applying the known RLHF framework to a new domain -- reasoning quality)
  2. Test-time compute scaling (the paradigm shift: scaling inference compute as alternative to scaling model size)
  3. Search during inference (self-consistency DEVELOPED from MENTIONED, best-of-N, verifier-guided search -- these are variations of one concept: sampling multiple reasoning paths and selecting)
- **Previous lesson load:** STRETCH (chain-of-thought introduced tokens-as-computation and fixed budget as genuinely new concepts)
- **This lesson's load:** BUILD -- appropriate. The conceptual framework (tokens as computation, RL training) is already established. This lesson connects existing pieces rather than introducing a new paradigm. New vocabulary (test-time compute, PRM, ORM) but the underlying ideas are combinations of concepts the student already has.
- **Load trajectory:** STRETCH (ICL) -> BUILD (prompt engineering) -> STRETCH (CoT) -> BUILD (reasoning models). Pattern maintained. Recovery lesson after the CoT stretch.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| Tokens as computation (CoT, 5.2.3) | Direct extension: reasoning models are TRAINED to use tokens-as-computation effectively, via RL. |
| RLHF pipeline (4.4, 5.1.1) | Same RL loop, different reward signal. "Same pipeline, different objective." |
| Reward hacking (4.4.3) | Outcome reward models can be gamed -- a model might learn shortcuts that produce correct answers without genuine reasoning. This is WHY process supervision matters. |
| Process vs outcome supervision (CoT, 5.2.3) | DEVELOPS from INTRODUCED. Process supervision trains better reasoning, not just correct answers. Connected to reward hacking: outcome supervision is the proxy, process supervision is closer to the true objective. |
| Self-consistency (CoT, 5.2.3) | DEVELOPS from MENTIONED. Multiple chains + voting as a form of search. Connects to the broader idea of search during inference. |
| CoT error propagation (CoT, 5.2.3) | Motivates WHY training better reasoning matters: base model CoT is unreliable, errors propagate. RL training reduces (but does not eliminate) this failure mode. |
| Design space axes (alignment-techniques-landscape, 5.1.2) | The "bigger model vs more thinking time" tradeoff echoes the "axes not ladder" framework. Scaling has multiple independent axes, not a single dimension. |
| "Same pipeline, different data source" (constitutional-ai, 5.1.1) | Pattern recurrence: same RL machinery applied to different objectives (helpfulness -> alignment -> reasoning). |

**Analogies from prior lessons that can be extended:**
- "Scratchpad/working memory" -> "Training the scratchpad" (RL teaches effective scratchpad use)
- "Forward pass is thinking budget" -> "Budget allocation" (reasoning models learn to allocate variable budgets)
- "Same pipeline, different data source" -> "Same pipeline, different reward signal"

**Analogies from prior lessons that could be misleading:**
- "The prompt is a program" -- reasoning models shift some of the "programming" from the prompt to the training. The model's learned reasoning policy partially replaces explicit CoT prompting. Important to note that prompting still matters, but the model has internalized reasoning strategies.

### Scope Boundaries

**This lesson IS about:**
- How RL trains models to generate effective reasoning chains (conceptual mechanism)
- The difference between outcome reward models (ORMs) and process reward models (PRMs)
- Test-time compute scaling: trading inference compute for model size
- Self-consistency and best-of-N as search strategies during inference
- The paradigm shift from "scale the model" to "scale the inference compute"
- How reasoning models (o1-style) differ from base models with CoT prompting

**This lesson is NOT about:**
- Implementing RL for reasoning in code (conceptual lesson)
- Specific model architectures or training details of o1, DeepSeek-R1, etc.
- Tree-of-thought or MCTS implementation details (search concept only, not specific algorithms)
- Mathematical formalization of the RL objective
- Distillation of reasoning models into smaller models
- Agentic patterns, tool use, or multi-step planning
- The specific training data or reward model training procedures

**Target depths:**
- RL for reasoning: DEVELOPED (can explain the mechanism, connect to RLHF, identify why it works)
- Test-time compute scaling: DEVELOPED (can explain the paradigm shift, identify the tradeoff)
- Process supervision vs outcome supervision: DEVELOPED (upgraded from INTRODUCED -- can explain WHY process supervision trains better reasoning, connect to reward hacking)
- Self-consistency / search during inference: DEVELOPED (upgraded from MENTIONED -- can explain the mechanism, connect to error propagation, identify when it helps)
- Specific reasoning model architectures (o1, etc.): INTRODUCED (knows they exist and how they differ conceptually, not implementation details)

### Lesson Outline

**1. Context + Constraints**
What this lesson is about: how RL trains models to reason effectively, and the paradigm shift from scaling model size to scaling inference computation. What we are NOT doing: implementing RL, studying specific model architectures, or formalizing the math. This is the module capstone -- connecting the CoT mechanism to training and deployment.

**2. Recap**
Brief reconnection to two key facts from the CoT lesson: (1) intermediate tokens are computation (each triggers another forward pass), and (2) the base model has no training signal for reasoning quality -- it generates reasoning tokens based on pretraining patterns, which are unreliable. Reconnect to RLHF: "You know how RL can optimize model behavior toward a reward signal. What if the reward signal were 'did you solve the math problem correctly?'"

**3. Hook (before/after contrast)**
Same math problem from the CoT lesson (or similar difficulty). Base model + "let's think step by step": sometimes right, sometimes wrong, reasoning quality varies wildly across 5 attempts. Same problem, reasoning model: consistently correct, reasoning chains are structured and checkable. ComparisonRow: 5 base-model attempts (3/5 correct, varied quality) vs 5 reasoning-model attempts (5/5 correct, structured chains). GradientCard puzzle: "Same architecture. Same number of parameters. Same problem. What changed?"

**4. Explain Part 1 -- RL for Reasoning**
The answer: training with reinforcement learning where the reward signal is answer correctness.

Walk through the RL training loop:
1. Model generates a reasoning chain for a problem
2. Check the final answer (is it correct?)
3. Correct -> positive reward; incorrect -> negative reward
4. Update the model's policy to generate chains that lead to correct answers more often

Pseudocode annotated with callbacks to RLHF:
- "This is the same RL loop you saw in RLHF. The difference is the reward signal."
- RLHF reward: "Is this response helpful and harmless?" (human preference model)
- Reasoning reward: "Is the final answer correct?" (verifiable, no human needed for math/code)

"Of course" beat: "You already knew tokens are computation. You already knew RL can shape behavior. Of course you can use RL to shape reasoning behavior."

GradientCard: "Training the scratchpad" -- RL does not teach the model new knowledge. It teaches the model to use the scratchpad (context window) more effectively. Same parameters, better reasoning token generation.

Address misconception 1 here: "Reasoning models are not bigger models." Same architecture, same parameter count. The difference is training, not size.

**5. Check 1 (predict-and-verify)**
A reasoning model is trained with RL where the reward is "final answer is correct." Predict: What kind of reasoning behavior does this reward incentivize? What might go wrong?

Reveal: The model learns chains that reach correct answers, but outcome-only reward means: (a) it might learn shortcuts (lucky guesses that happen to be right), (b) it cannot distinguish correct reasoning from cancelling errors, (c) a chain with 9 correct steps and 1 wrong step that luckily produces the right answer gets the same reward as 10 correct steps. This is the reward hacking pattern from 4.4.3 applied to reasoning.

Transition: "Outcome reward has the same problem as every proxy: it can be gamed. What if we could reward the reasoning process itself?"

**6. Explain Part 2 -- Process Supervision (DEVELOP from INTRODUCED)**
Recall from the CoT lesson: outcome supervision evaluates the final answer, process supervision evaluates each step.

Now develop it:
- **Outcome Reward Model (ORM):** Sees the full chain, scores the final answer. Cheap to train (just need answer labels). But rewards are sparse (one signal for the entire chain) and gameable (correct answer does not imply correct reasoning).
- **Process Reward Model (PRM):** Evaluates each step individually. Richer signal (feedback per step, not per chain). Trains models that reason correctly, not just models that get lucky. But: requires step-level labels (expensive, though automatable -- connect to RLAIF pattern from 5.1.1).

ComparisonRow: ORM vs PRM. Concrete example: two chains for the same problem, both reaching the correct answer. Chain A has correct reasoning throughout. Chain B has a sign error in step 3 that cancels with another error in step 5. ORM gives both the same reward. PRM penalizes Chain B at steps 3 and 5.

Connect to reward hacking: "Outcome supervision is the proxy. Process supervision is closer to the true objective. Same pattern as RLHF: the more precisely you can specify what 'good' means, the better the training signal."

Connect to RLAIF: "Step-level labels can start with humans and scale through AI, just like constitutional AI scaled preference labels."

Address misconception 3: "Process supervision does not mean a human checks every step." PRMs are trained models that evaluate steps automatically. Same scaling pattern as RLAIF.

**7. Explain Part 3 -- Test-Time Compute Scaling**
Now the paradigm shift. If reasoning models can generate effective chains, a new scaling dimension opens up.

Traditional scaling: make the model bigger (more parameters -> more computation per forward pass -> better performance). This is the only scaling axis the student has seen so far.

New scaling axis: make the model think longer (more reasoning tokens -> more forward passes -> more computation per problem). Same hardware, same model, more compute.

Scaling paradigm diagram (inline SVG): two axes. X-axis: model size (parameters). Y-axis: inference compute (reasoning tokens). Traditional scaling moves right. Test-time compute scaling moves up. Both increase total computation, through different mechanisms.

"Bigger brain vs more thinking time" analogy. A student who thinks for 30 minutes outperforms a student who glances for 5 seconds, even if the second student is "smarter." When is it better to get a smarter student vs giving the current student more time?

Address misconception 4: "Test-time compute scaling does not mean better hardware. It means more forward passes through the same model on the same hardware."

The key insight: on many problems, spending 10x more inference compute on a smaller reasoning model outperforms a 10x larger base model that answers in one shot. The stretch example: 7B reasoning model vs 70B base model on competition math.

Address misconception 2 here: "The model does not 'decide' to think harder. It generates tokens according to a learned policy shaped by RL. On harder problems, the policy produces longer chains because longer chains on harder problems were rewarded during training."

**8. Explain Part 4 -- Search During Inference (DEVELOP self-consistency from MENTIONED)**
Self-consistency, developed:

The CoT lesson MENTIONED: "generate multiple chains, take majority vote." Now develop the mechanism:
1. Sample N reasoning chains for the same problem (different random seeds -> different chains)
2. Extract the final answer from each chain
3. Take the majority vote

Worked example: 5 chains for a word problem. Show all 5 chains (abbreviated). 3 reach answer 42, 2 reach answer 38. Majority vote: 42 (correct).

Why it works: if each chain has probability p > 0.5 of being correct, then N independent chains with majority voting have higher probability of correct answer than any single chain. This is the same statistical principle as ensemble methods (if the student has seen these) or simply "asking 5 people and going with the majority."

Connect to CoT error propagation: "One chain can have an error. But different chains make different errors. Voting averages out the noise."

Best-of-N with a verifier: instead of majority vote, use a reward model (ORM or PRM) to score each chain and select the best one. This is search: generate candidates, evaluate, select.

Self-consistency is search with majority-vote selection. Best-of-N is search with verifier selection. Both are forms of test-time compute scaling: generating more chains = spending more inference compute.

Address misconception 5: "More chains are not always better." Diminishing returns: going from 1 to 5 chains is a large improvement. Going from 50 to 100 chains is marginal. And the compute cost is linear. The factual recall negative example returns: generating 100 reasoning chains for "What is the capital of France?" wastes compute and can produce overthinking errors.

**9. Check 2 (transfer question)**
A company wants to deploy an LLM for customer support. They can either: (A) use a large base model (70B) that answers immediately, or (B) use a smaller reasoning model (7B) that generates a reasoning chain before answering. The questions are a mix of simple FAQ lookups (60%) and complex troubleshooting (40%).

Questions: Which approach is better for each type of question? Would you use the same strategy for both? What would a hybrid approach look like? What does process supervision look like in this context?

Reveal: Simple FAQs -- direct answer is fine, reasoning chains add latency and cost with no accuracy benefit (computational complexity criterion from CoT). Complex troubleshooting -- reasoning model outperforms, structured chains are checkable, process supervision can flag uncertain steps. Hybrid: route by estimated complexity. Process supervision: flag steps where the PRM is uncertain, escalate to human review.

**10. Practice -- Notebook Exercises (Colab)**
`notebooks/5-2-4-reasoning-models.ipynb` (4 exercises)

- **Exercise 1 (Guided): Base model CoT vs reasoning model comparison.** Give the same 10 math/reasoning problems to a base model with CoT prompting and a reasoning-focused model. Compare accuracy and inspect reasoning chain quality (not just final answers). Predict-before-run: "Which problems will the base model get wrong that the reasoning model gets right?" First 3 problems fully worked with analysis template. Insight: RL training produces consistently better reasoning, not just longer chains.

- **Exercise 2 (Supported): Self-consistency experiment.** For 5 reasoning problems, generate N chains (N = 1, 3, 5, 10, 20) and compute majority-vote accuracy at each N. Plot accuracy vs N. Identify the point of diminishing returns. Try with both easy and hard problems. First problem fully set up with code for N=1 and N=3. Insight: more chains help up to a point; the benefit is larger for harder problems.

- **Exercise 3 (Supported): Process vs outcome evaluation.** For 5 problems with known step-by-step solutions, generate reasoning chains and evaluate them two ways: (a) does the final answer match? (outcome), (b) is each step correct? (process). Find cases where the outcome is correct but a step is wrong. First problem with evaluation template provided. Insight: outcome evaluation misses reasoning flaws that would fail on harder problems.

- **Exercise 4 (Independent): Test-time compute allocation.** Design an experiment: given a fixed compute budget (total tokens across all problems), compare two strategies: (a) equal allocation (same number of reasoning tokens per problem), (b) adaptive allocation (more tokens for harder problems, fewer for easy ones). Use problem difficulty as a proxy (number of steps in the solution). Measure overall accuracy. No skeleton provided. Insight: adaptive compute allocation outperforms uniform allocation -- the core of test-time compute scaling.

**11. Summarize**
Key takeaways:
1. Reasoning models apply RL to train effective use of chain-of-thought -- same RL loop, different reward signal
2. Process supervision trains better reasoning than outcome supervision -- evaluating steps, not just answers (the reward hacking lesson applied to reasoning)
3. Test-time compute scaling is a new scaling axis: instead of bigger models, let models think longer
4. Search during inference (self-consistency, best-of-N) trades compute for reliability
5. The paradigm shift: from "how big is the model?" to "how much does the model think?"

Echo the mental model: "The forward pass budget is fixed. CoT spends more budget by generating more tokens. RL trains the model to spend that budget wisely. And test-time compute scaling says: if spending more budget helps, why not give it as much budget as it needs?"

**12. Next Step**
"This completes our exploration of what LLMs can do at inference time -- from learning tasks from examples (ICL), to systematic prompting, to chain-of-thought computation, to trained reasoning. Each technique works because of a specific, mechanistic reason, not because the model 'understands' or 'thinks.' Next, we move beyond text to see how these same transformer architectures handle images, audio, and more."

(Or whatever the next module is in the curriculum.)

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
- [x] At least 3 modalities planned for the core concept, each with rationale (6 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (2 positive + 1 negative + 1 stretch)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2-3: RL for reasoning, test-time compute scaling, search during inference)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-19 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 5
- Polish: 3

### Verdict: MAJOR REVISION

One critical finding (planned negative example missing as a concrete, standalone element), five improvement findings, and three polish findings. The lesson is structurally strong -- it follows the plan faithfully, connects to prior concepts extensively, maintains appropriate cognitive load, and uses all Row components correctly. The critical finding and improvement findings center on the same pattern: the lesson is explanation-heavy and example-light. The conceptual explanations are solid, but the student gets fewer concrete, worked-through instances than the plan specifies. Fix the critical issue and address the improvement findings, then re-review.

### Findings

### [CRITICAL] -- Planned negative example missing as concrete demonstration

**Location:** Search During Inference section (Section 9) and Test-Time Compute Scaling section (Section 8)
**Issue:** The planning document specifies a negative example: "Simple factual question where reasoning model wastes tokens overthinking and arrives at wrong answer through second-guessing." This was planned as a concrete, standalone example showing that more reasoning is not always better. In the built lesson, this is handled only as text within Misconception 5 ("More reasoning tokens are always better"), which references the CoT lesson's "What is the capital of France?" example by callback rather than presenting its own concrete, new negative example. The plan says: "Mirrors the negative example from the CoT lesson. Extends it: even a model TRAINED to reason can over-apply reasoning." The lesson does not extend it -- it merely references the old example.
**Student impact:** The student reads the misconception block and understands the principle abstractly, but does not see a concrete demonstration specific to reasoning models. The CoT lesson's capital-of-France example was about a base model with CoT prompting. The student may not transfer the lesson to reasoning models without seeing it demonstrated specifically for that context. Negative examples need to be concrete and specific to the concept being taught.
**Suggested fix:** Add a concrete negative example (GradientCard, rose color) before or after Misconception 5 showing a reasoning model applied to a simple question. Show the reasoning model generating an unnecessarily long chain for something like "What color is the sky?" -- where the chain introduces second-guessing ("Is it blue? Or is it cerulean? But at sunset it is orange...") and either wastes compute or arrives at a confused answer. This extends the CoT negative example to reasoning models specifically, which is what the plan intended.

### [IMPROVEMENT] -- Hook comparison is hypothetical, not concrete

**Location:** Section 4 ("Same Architecture, Different Reliability")
**Issue:** The before/after hook presents a ComparisonRow with generic "Attempt 1: Correct reasoning, right answer" / "Attempt 2: Skipped a step, wrong answer" text. This is a pattern description, not a concrete example. The plan says: "Same math problem from the CoT lesson (or similar difficulty)" and specifies showing actual reasoning chains. The built lesson describes what would happen ("the reasoning quality varies wildly") without showing specific reasoning chains with actual math.
**Student impact:** The student gets an abstract pattern ("sometimes right, sometimes wrong") rather than seeing actual reasoning chains they can inspect. The hook is less compelling because the student cannot verify the claims. Compare to the CoT lesson's hook, which showed the actual "17 x 24" calculation with specific wrong answers ("384"). This hook tells rather than shows.
**Suggested fix:** Use a specific problem (e.g., the 17 x 24 from CoT, or a similar multi-step problem). Show 2-3 abbreviated base model chains with actual reasoning steps (one correct, one with a specific error like "17 x 20 = 350"), then show the reasoning model's structured chain. The ComparisonRow items should contain actual math, not meta-descriptions. This grounds the hook in something the student can trace through.

### [IMPROVEMENT] -- Self-consistency worked example answer is too uniform

**Location:** SelfConsistencyExample component (Section 9)
**Issue:** The self-consistency worked example uses a problem where 3 out of 5 chains produce the correct answer ($144), and 2 produce wrong answers. This demonstrates majority voting. However, the problem is quite simple (hourly rate x hours summed across days), and all 3 correct chains reach $144 via different but straightforward approaches. The errors in Chains 4 and 5 are arithmetic mistakes (4 x 12 = 46, 5 + 3 + 4 = 11). This is fine as a first example, but the lesson has no second example showing self-consistency on a harder problem where the chains might diverge more meaningfully (different reasoning approaches, not just arithmetic variations).
**Student impact:** The student sees self-consistency work on an easy problem but might not generalize to harder problems where chains take genuinely different reasoning paths. The plan says: "Concrete numbers make the mechanism tangible" -- the numbers are concrete but the chains are too similar. All correct chains follow the same approach (multiply rate by hours). A harder problem where chains take different decomposition strategies would strengthen the generalization.
**Suggested fix:** The existing example is fine as Example 1. Add a brief second self-consistency example (does not need to be as detailed -- could be a paragraph summarizing a harder problem where chains take genuinely different approaches and still converge on the same answer). This confirms generalization per the Example Rules.

### [IMPROVEMENT] -- Process supervision concrete example lacks a "correct reasoning throughout" chain

**Location:** Section 7 (Process Supervision), "Concrete Example: Same Answer, Different Reasoning" GradientCard
**Issue:** The concrete example shows Chain A (correct reasoning: 17 x 20 = 340, 17 x 4 = 68, 340 + 68 = 408) and Chain B (cancelling errors: 17 x 20 = 350, 17 x 4 = 58, 350 + 58 = 408). This is good and directly from the plan. However, the ORM vs PRM ComparisonRow above it lists properties abstractly without connecting to this specific example. The student reads "Gameable: correct answer does not imply correct reasoning" in the ComparisonRow and then sees the concrete example below. The flow would be stronger if the concrete example came first (concrete before abstract, per the Ordering Rules) and the ComparisonRow synthesized afterward.
**Student impact:** The student reads abstract ORM/PRM properties, tries to understand them without a concrete anchor, then sees the example. By the time they reach the example, the abstract properties have already been processed (or skipped). Flipping the order would let the concrete example motivate the abstract properties.
**Suggested fix:** Move the "Concrete Example: Same Answer, Different Reasoning" GradientCard above the ComparisonRow. Start with the concrete example (Chain A vs Chain B), then use the ComparisonRow as the generalization ("Now let's see how ORMs and PRMs differ in general"). This follows the ordering rule: concrete before abstract.

### [IMPROVEMENT] -- "Specific reasoning model architectures" INTRODUCED depth not met

**Location:** Entire lesson
**Issue:** The planning document targets "Specific reasoning model architectures (o1, etc.): INTRODUCED (knows they exist and how they differ conceptually, not implementation details)." The built lesson mentions o1-style and DeepSeek-R1 only in the scope boundaries ("NOT: specific model architectures or training details of o1, DeepSeek-R1, etc."), in one aside (references block with DeepSeek-R1 paper), and in the lesson header comment. The lesson body never explicitly introduces o1 or DeepSeek-R1 as named architectures, does not explain what makes them different from base models with CoT at even a high level, and does not give the student name recognition for these models.
**Student impact:** The student finishes the lesson understanding the general concept of reasoning models but cannot name specific reasoning models or say what distinguishes o1-style models from others. The INTRODUCED depth requires the student to "recognize the term" and "explain in own words." The student has the concept but not the vocabulary for the specific models.
**Suggested fix:** Add a brief paragraph (2-3 sentences) either in the RL for Reasoning section or just before the summary, naming o1 and DeepSeek-R1 as concrete instances of reasoning models. Something like: "OpenAI's o1 and DeepSeek-R1 are examples of this approach. o1 generates internal reasoning tokens before producing an answer. DeepSeek-R1 showed that pure RL (without supervised fine-tuning on reasoning examples) can produce effective reasoning behavior." This takes the depth from MENTIONED to INTRODUCED with minimal additional cognitive load.

### [IMPROVEMENT] -- Notebook Exercise 2 Step 2 TODO section has the solution in the hint comments

**Location:** Notebook cell-10 (Exercise 2, Step 2)
**Issue:** The exercise is labeled "Supported" (student writes code with hints, solution in a details block). However, the TODO section in cell-10 includes a multi-line commented-out hint that is essentially the complete solution:
```python
    #   sc_results[p_idx] = {}
    #   for n in N_VALUES:
    #       subset = all_answers[:n]
    #       vote = majority_vote(subset)
    #       ...
```
This is the exact code the student needs to write. A "Supported" exercise should give hints about approach, not the complete implementation as commented-out code. The solution is then repeated in the details block in cell-12. The student effectively has the answer twice.
**Student impact:** The student sees the commented-out solution before attempting the exercise. This undermines the predict-before-run and active implementation goals. The hint is so complete that "fill in the code" becomes "uncomment the code." The scaffolding is too high for a Supported exercise.
**Suggested fix:** Replace the commented-out hint with a higher-level hint that describes the approach without giving the code. For example: "Hint: For each N, take the first N answers from all_answers and use majority_vote(). Store in a dictionary keyed by N." Remove the multi-line code block from the hint. The full solution remains in the details block in cell-12 for students who are stuck.

### [POLISH] -- Module completion next module title may be incorrect

**Location:** Section 13, ModuleCompleteBlock
**Issue:** The ModuleCompleteBlock specifies `nextModule="5.3"` and `nextTitle="Multimodal Models"`. The planning document says "(Or whatever the next module is in the curriculum.)" This may or may not be correct depending on the curriculum structure. If 5.3 is not "Multimodal Models," the student gets a wrong preview.
**Student impact:** Minor confusion if the next module title is wrong. Not a learning issue but a trust issue -- incorrect metadata undermines confidence in the content.
**Suggested fix:** Verify against the curriculum data whether Module 5.3 is "Multimodal Models." If correct, no change needed. If incorrect, update the title.

### [POLISH] -- NextStepBlock description is very long

**Location:** Section 15, NextStepBlock
**Issue:** The NextStepBlock description is 3 sentences totaling approximately 80 words. NextStepBlock descriptions in prior lessons tend to be more concise (1-2 sentences). This one recaps the entire module's arc and previews the next module, which is already done by the ModuleCompleteBlock.
**Student impact:** Minimal -- the student reads a slightly redundant summary. But the ModuleCompleteBlock already provides the module recap and next-module preview.
**Suggested fix:** Shorten the NextStepBlock description to 1-2 sentences focused on what comes next, since the ModuleCompleteBlock handles the recap.

### [POLISH] -- Notebook Exercise 3 Step 1 uses LLM to evaluate process correctness

**Location:** Notebook cell-15 (Exercise 3, Step 1)
**Issue:** The process evaluation step uses the LLM itself (via call_llm) to evaluate whether each step in a reasoning chain is correct. The evaluation prompt asks the model to check against known correct steps. This is a reasonable approach for a notebook exercise, but the lesson should acknowledge that this is using an LLM as a proxy PRM, not a true PRM. The LLM evaluator is imperfect -- it may miss errors or flag correct steps. The notebook does not mention this limitation.
**Student impact:** The student might conclude that building a PRM is as simple as prompting an LLM to evaluate steps. This is a simplification. Real PRMs are trained models, as the lesson explains. The gap between "prompt an LLM to check steps" and "train a process reward model" is not acknowledged.
**Suggested fix:** Add a brief markdown cell or code comment acknowledging that this approach is a simplified proxy for a real PRM, and that in practice, PRMs are trained on step-level labels, not prompted. Something like: "Note: we're using the LLM itself as a simplified step evaluator. A real PRM would be a dedicated model trained specifically on step-level correctness labels, as described in the lesson."

### Review Notes

**What works well:**

- The lesson faithfully follows the planning document's structure. All 12 sections from the plan are present in the correct order. The narrative arc is coherent and well-paced.
- Connections to prior concepts are extensive and specific. Every new concept is explicitly linked to something the student knows (RLHF, CoT tokens-as-computation, reward hacking, Constitutional AI). The "same pipeline, different reward signal" thread runs throughout.
- The "of course" moment is well-executed. The GradientCard in Section 5 genuinely collapses the conceptual distance by reminding the student they already have both pieces (tokens as computation + RL shapes behavior).
- All 5 planned misconceptions are addressed with dedicated GradientCard blocks.
- The scaling paradigm diagram (inline SVG) effectively communicates the two-axis scaling concept visually.
- The RL training loop diagram clearly shows the generate-evaluate-reward-update cycle.
- All layout uses Row components correctly. No manual flex layouts.
- The notebook has good scaffolding progression: Guided -> Supported -> Supported -> Independent.
- The notebook's shared data section (PROBLEMS) is efficient and well-designed.
- Cognitive load is appropriate for a BUILD lesson.

**Patterns to watch:**

- The lesson leans toward explanation over demonstration. Multiple sections explain concepts in paragraph form where a concrete example would be more effective. The plan specified 2 positive + 1 negative + 1 stretch example; the lesson has positive examples (ComparisonRow in hook, self-consistency worked example) but the negative example is embedded in a misconception block rather than being standalone.
- The notebook's "Supported" exercises have hints that are too close to solutions. This pattern appeared in the chain-of-thought notebook review (Exercise 2 was pre-filled) and was fixed. The same pattern recurs here. Worth establishing a convention: hints describe approach in words, solutions provide code.

---

## Review -- 2026-02-19 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

All iteration 1 findings have been addressed. The critical finding (missing standalone negative example) is fully resolved with the "When Reasoning Backfires" GradientCard. The hook now shows concrete math chains, the process supervision ordering is correct (concrete before abstract), specific reasoning model architectures reach INTRODUCED depth, and the notebook hints are appropriate for Supported exercises. Two improvement findings remain: the self-consistency section lacks a second example for generalization, and the ModuleCompleteBlock references a non-existent Module 5.3. Two polish findings round out the review.

### Findings

### [IMPROVEMENT] -- Self-consistency section lacks a second example for generalization

**Location:** Section 9 (Search During Inference), SelfConsistencyExample component
**Issue:** The planning document's Examples Planned table includes a self-consistency worked example as a positive example, and the Example Rules require at least 2 positive examples for core concepts. The lesson has one self-consistency worked example (5 chains for the Alex-earns-$12/hour problem) that demonstrates majority voting well. However, self-consistency is a core concept being DEVELOPED from MENTIONED in this lesson, and there is no second example showing the mechanism on a harder problem where chains take genuinely different reasoning approaches (different decomposition strategies, not just arithmetic variations). The iteration 1 review flagged this, and the fix round did not add a second example. Best-of-N with a verifier is explained as a generalization, but it is a different selection mechanism, not a second self-consistency example.
**Student impact:** The student sees self-consistency work on a straightforward problem where all correct chains use essentially the same approach (multiply rate by hours, sum across days). They may not generalize to harder problems where chains take genuinely different reasoning paths (e.g., a problem solvable via algebra or via guess-and-check, where different chains use different strategies but converge). The Example Rules exist precisely to prevent this: "the second example confirms the pattern generalizes."
**Suggested fix:** Add a brief second self-consistency paragraph (no need for a full worked component) after the SelfConsistencyExample, describing a harder problem where chains take genuinely different approaches. For example: "On a harder problem -- say, a system of equations -- one chain might use substitution, another elimination, another guess-and-check. Despite different approaches, the correct chains converge on the same answer. The majority vote still works because the key property is convergence of correct answers, not similarity of reasoning paths." This confirms generalization with minimal additional length.

### [IMPROVEMENT] -- ModuleCompleteBlock references non-existent Module 5.3

**Location:** Section 13, ModuleCompleteBlock (line 1633-1648) and Section 15, NextStepBlock (line 1697-1700)
**Issue:** The ModuleCompleteBlock specifies `nextModule="5.3"` and `nextTitle="Multimodal Models"`. The NextStepBlock description says "you move beyond text to see how these same transformer architectures handle images, audio, and more." However, the curriculum data (`src/data/curriculum/recent-llm-advances.ts`) defines Series 5 with only two modules: 5.1 (Advanced Alignment) and 5.2 (Reasoning & ICL). There is no Module 5.3. The next series in the curriculum is Series 6 (Stable Diffusion), not "Multimodal Models." This was flagged as a polish finding in iteration 1 with a suggestion to verify against the curriculum data. The curriculum data confirms the reference is incorrect.
**Student impact:** The student finishes the module capstone and sees a preview of "Multimodal Models" as the next module. When they return to the app, the next recommended lesson will not match this preview, breaking the "clear next step on every app open" principle. The mismatch between the lesson's promise and the actual curriculum undermines trust.
**Suggested fix:** Update the ModuleCompleteBlock to either: (a) reference the actual next module/series in the curriculum, or (b) use a generic "next module" message without specifying a title. Since this is the last module in Series 5, the NextStepBlock should reflect completion of the series and transition to the next series (likely Series 6: Stable Diffusion). Update the NextStepBlock description to match whatever the actual next content is.

### [POLISH] -- Notebook code comment uses spaced em dash

**Location:** Notebook cell-15, code comment (line: "it is less reliable than a trained PRM -- it may miss subtle errors")
**Issue:** The NOTE comment in cell-15 uses a spaced em dash pattern (" -- ") rather than an unspaced em dash. The project style rule specifies "Em dashes must have no spaces: word--word not word -- word." While this is in a code comment rather than prose, it is visible to the student and sets an inconsistent style precedent.
**Student impact:** Negligible. The comment is informative and the spacing does not affect comprehension. But consistency matters for a polished product.
**Suggested fix:** Change " -- " to the appropriate em dash in the comment, or accept that code comments follow a different convention (since `--` in code comments is conventional in many codebases). Low priority.

### [POLISH] -- Summary item 5 overlaps with item 3

**Location:** Section 12, SummaryBlock, items 3 and 5
**Issue:** Summary item 3 says: "Test-time compute scaling is a new scaling axis. Instead of bigger models, let models think longer." Summary item 5 says: "The paradigm shift: from 'how big is the model?' to 'how much does the model think?'" These are the same concept stated differently. Five summary items is not excessive, but items 3 and 5 are redundant -- the student reads the paradigm shift statement twice in slightly different phrasing.
**Student impact:** Minor. The student may feel the summary is slightly repetitive. Not a comprehension issue.
**Suggested fix:** Either merge items 3 and 5 into a single item that covers both the mechanism (more tokens = more forward passes) and the framing (paradigm shift), or rephrase item 5 to focus on a distinct insight (e.g., the adaptive compute allocation angle -- the optimal amount of reasoning depends on problem difficulty).

### Review Notes

**Iteration 1 findings -- verification of fixes:**

All 9 iteration 1 findings have been addressed:

1. **[CRITICAL] Missing standalone negative example** -- FULLY RESOLVED. The "When Reasoning Backfires" GradientCard (lines 1354-1381) is a concrete, standalone negative example showing a reasoning model overthinking "What is the capital of France?" with second-guessing through historical capitals. This extends the CoT lesson's example to reasoning models specifically, exactly as the plan intended. Effective and well-placed.

2. **[IMPROVEMENT] Hook comparison hypothetical** -- FULLY RESOLVED. The ComparisonRow now shows actual reasoning chains with specific math (17x20=340, 17x4=68) and specific errors (17x20=350, 17x4=72). The student can trace through each chain. Concrete and verifiable.

3. **[IMPROVEMENT] Self-consistency needs second example** -- NOT RESOLVED. See new Improvement finding above. The existing example is good but a second example on a harder problem is still missing.

4. **[IMPROVEMENT] Process supervision ordering (abstract before concrete)** -- FULLY RESOLVED. The concrete example (Chain A vs Chain B) now appears before the ComparisonRow (lines 1043-1075 before lines 1081-1104). Correct ordering: concrete before abstract.

5. **[IMPROVEMENT] Specific reasoning model architectures INTRODUCED depth** -- FULLY RESOLVED. Paragraph at lines 1240-1247 names o1 and DeepSeek-R1 with brief descriptions of what makes each distinctive. Meets INTRODUCED depth: the student can recognize these names and explain their significance in own words.

6. **[IMPROVEMENT] Notebook Exercise 2 hints too close to solution** -- FULLY RESOLVED. The TODO section in cell-10 now describes the approach in words ("take the first N answers", "use majority_vote()", "dictionary keyed by N") without providing the exact code. The multi-line commented-out solution is removed. The student must write the loop and logic themselves. Appropriate scaffolding for Supported.

7. **[POLISH] Module completion next module title** -- NOT RESOLVED, UPGRADED TO IMPROVEMENT. Verified against curriculum data: Module 5.3 does not exist. See new Improvement finding above.

8. **[POLISH] NextStepBlock description too long** -- RESOLVED. Description is now 2 sentences (shorter than before), though the content still references non-existent multimodal module (covered by the ModuleCompleteBlock finding).

9. **[POLISH] Notebook LLM-as-proxy-PRM acknowledgment** -- FULLY RESOLVED. Cell-15 now has a prominent NOTE comment explaining that the LLM-based step evaluator is a simplified proxy for a real PRM.

**No regressions detected from the fix round.** The fixes were clean and did not introduce new issues in surrounding content.

**What works well (carried forward + new observations):**

- The lesson is structurally sound and pedagogically effective. The BUILD designation is appropriate -- it connects two established frameworks (CoT mechanism + RL training) rather than introducing fundamentally new concepts.
- All 5 misconceptions are addressed with dedicated GradientCard blocks, each with concrete counter-arguments.
- The "of course" moment genuinely collapses conceptual distance.
- The standalone negative example ("When Reasoning Backfires") is one of the strongest negative examples in the module -- it shows a specific, traceable failure mode with a concrete reasoning chain.
- The process supervision concrete example (Chain A vs Chain B with cancelling errors) is clear and well-ordered (concrete before abstract).
- The RL training loop diagram and scaling paradigm diagram are effective visual aids.
- The notebook is well-structured with appropriate scaffolding progression and properly calibrated hints.
- Connections to prior concepts are extensive and specific throughout (RLHF, CoT, reward hacking, Constitutional AI, design space axes).
- The notebook's shared data section (PROBLEMS) and helper functions are well-designed for reuse across exercises.

---

## Review -- 2026-02-19 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

All iteration 2 findings have been resolved. No regressions detected. No new findings at any severity level. The lesson is ready to ship.

### Findings

None.

### Iteration 2 Findings -- Verification of Fixes

All 4 iteration 2 findings have been resolved:

1. **[IMPROVEMENT] Self-consistency section lacks second example** -- FULLY RESOLVED. Lines 1313-1324 contain a second self-consistency example: a train-leaves-City-A problem where three chains use genuinely different reasoning strategies (algebra equation 60t = 90(t-2), rate-of-closing approach with 120-mile gap closing at 30 mph, and a table of positions at each hour). All three converge on 360 miles despite different approaches. This confirms generalization per the Example Rules: "the second example confirms the pattern generalizes." The key insight -- that majority vote works because correct reasoning converges regardless of approach, not because chains are similar -- is stated explicitly.

2. **[IMPROVEMENT] ModuleCompleteBlock references non-existent Module 5.3** -- FULLY RESOLVED. The ModuleCompleteBlock now specifies `nextModule="6.1"` and `nextTitle="Generative Foundations"`. Verified against the curriculum data (`src/data/curriculum/stable-diffusion.ts`): Series 6, Module 6.1 is "Generative Foundations." The NextStepBlock description also updated to match: "Next up: generative foundations, starting with the shift from classification to generation." Both references are now accurate.

3. **[POLISH] Notebook code comment spaced em dash** -- FULLY RESOLVED. No spaced em dashes found in the notebook. The PRM proxy acknowledgment note in cell-15 now uses proper dash conventions.

4. **[POLISH] Summary items 3 and 5 overlap** -- FULLY RESOLVED. Item 5 now reads: "The optimal amount of reasoning depends on problem difficulty. Simple tasks need less thinking; complex tasks benefit from more. Adaptive compute allocation outperforms uniform allocation." This is a distinct insight from item 3 (which covers the scaling axis paradigm shift and the 7B-vs-70B comparison). Item 5 focuses on adaptive allocation, not the paradigm shift itself.

### Review Notes

**Overall assessment:** The lesson is pedagogically strong and implementation-complete. Over three review iterations, 1 critical, 7 improvement, and 5 polish findings were identified and all resolved without regressions. The revision process was clean -- each fix addressed its finding precisely without introducing new issues.

**What works well (final assessment):**

- The lesson faithfully implements the BUILD designation: it connects two established frameworks (CoT tokens-as-computation + RLHF) rather than introducing fundamentally new concepts. Cognitive load is appropriate.
- All 5 misconceptions are addressed with dedicated GradientCard blocks, each with concrete counter-arguments grounded in established concepts (reward hacking, RLAIF, tokens-as-computation).
- The "of course" moment genuinely collapses conceptual distance by reminding the student they already have both pieces.
- The standalone negative example ("When Reasoning Backfires") is concrete and traceable -- the student sees a specific reasoning chain that introduces doubt and second-guessing on a simple factual question.
- The self-consistency section now has two examples: a simple worked example (Alex-earns-$12/hour with 5 chains) and a harder generalization example (train problem with three different reasoning strategies). Both confirm the mechanism from different angles.
- The process supervision section follows correct ordering (concrete Chain A/Chain B example before abstract ORM/PRM ComparisonRow).
- Connections to prior concepts are extensive and specific: RLHF, CoT, reward hacking, Constitutional AI, RLAIF, design space axes. Every new concept is explicitly linked to something the student already knows.
- The notebook has appropriate scaffolding progression (Guided -> Supported -> Supported -> Independent) with properly calibrated hints (approach described in words, not code).
- The ModuleCompleteBlock and NextStepBlock accurately reference the next module in the curriculum.
- All layout uses Row components correctly. No manual flex layouts.
- Em dashes follow the project convention (no spaces) throughout student-facing content.
- The 6 modalities (verbal/analogy x2, visual/SVG x2, concrete example, symbolic/code, intuitive) well exceed the minimum 3.

**Module 5.2 completion status:** This lesson is the module capstone (lesson 4 of 4). With this lesson passing review, all 4 lessons in Module 5.2 (Reasoning & In-Context Learning) have been reviewed and passed. The module record should be updated to include this lesson's concept contributions.
