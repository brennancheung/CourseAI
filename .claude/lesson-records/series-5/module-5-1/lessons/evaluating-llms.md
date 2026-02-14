# Lesson Plan: Evaluating LLMs

**Module:** 5.1 (Advanced Alignment), Lesson 4 of 4 (Module Capstone)
**Slug:** `evaluating-llms`
**Status:** Planned

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Red teaming as systematic adversarial process | DEVELOPED | red-teaming-and-adversarial-evaluation (5.1.3) | Systematic, adversarial, probing. Six attack categories. Pen-testing analogy. Alignment surface framing: alignment holds at some points in input space, has gaps at others. Student can explain the methodology and articulate why systematic probing is necessary. |
| Attack-defense dynamic / asymmetry | DEVELOPED | red-teaming-and-adversarial-evaluation (5.1.3) | Alignment is never "done." Each defense creates new attack surfaces. Attackers need ONE gap; defenders need ALL gaps. DAN progression as concrete example. Student can predict that patching creates new surfaces. |
| Automated red teaming (LLMs probing LLMs at scale) | INTRODUCED | red-teaming-and-adversarial-evaluation (5.1.3) | Generate-test-classify-iterate pipeline. Perez et al. (2022): 154,000 prompts finding failures humans missed. Same scaling insight as RLAIF. Student understands WHY it is necessary and HOW it works at high level. |
| Constitutional AI principles as explicit alignment criteria | DEVELOPED | constitutional-ai (5.1.1) | Principles define "better." Made explicit and auditable. Used only during TRAINING. Student understands that alignment quality depends on constitution quality. |
| RLAIF (AI-generated preference labels replacing human labels) | DEVELOPED | constitutional-ai (5.1.1) | AI applies constitutional principles to generate preference data. Same pipeline as RLHF, different data source. Scales to millions vs ~33K for InstructGPT. |
| Design space axes framework for preference optimization | DEVELOPED | alignment-techniques-landscape (5.1.2) | Four independent axes (data format, reference model, online/offline, reward model). Central mental model: "alignment techniques are points in a design space, not steps on a ladder." Student can classify a method they have not seen. |
| Alignment techniques as tradeoffs, not upgrades | DEVELOPED | alignment-techniques-landscape (5.1.2) | Each method solves a specific constraint at a cost. No universally best method. "Constraints drive choice." |
| Reward hacking (exploiting imperfections in learned reward model) | INTRODUCED | rlhf-and-alignment (4.4.3) | "Editor with blind spots." Excessive verbosity, confident filler, formatting tricks. Motivates KL penalty. The model optimizes against a proxy for what we actually want. |
| SFT failure modes (harmful helpfulness, sycophancy, confident incorrectness) | DEVELOPED | rlhf-and-alignment (4.4.3) | Three concrete failure modes. Motivated by "SFT teaches format but has no signal for quality." |
| Generalization from training distribution | DEVELOPED | Series 1 (the-learning-problem) | Learned in foundations: models generalize from training data, performance degrades on out-of-distribution inputs. The core insight that all ML is pattern matching on samples. Re-activated in 5.1.3 as "alignment generalizes from alignment data." |
| Defense-in-depth for alignment | INTRODUCED | red-teaming-and-adversarial-evaluation (5.1.3) | Five layers: training-time alignment, input filtering, output filtering, monitoring, regular re-evaluation. No single layer is sufficient. |

### Established Mental Models and Analogies

- **"The reward model is an experienced editor"** -- learned judgment from comparisons, has blind spots that can be exploited (from 4.4.3). Critical for this lesson: the reward model is a PROXY for what we want, and optimizing against a proxy is the core evaluation problem.
- **"The editor gets a style guide"** -- constitutional AI makes criteria explicit; extends editor analogy (from 5.1.1). Relevant here: evaluation metrics are another form of "style guide" -- explicit criteria that may not capture what we actually care about.
- **"Alignment techniques are points in a design space, not steps on a ladder"** -- tradeoffs, not upgrades (from 5.1.2). Relevant: benchmarks are also points in a measurement space, not a single score that captures everything.
- **"The challenge shifts, not disappears"** -- from annotator bottleneck to constitution design (5.1.1), to maintaining alignment under adversarial pressure (5.1.3). Extends here: the challenge shifts from "make the model better" to "measure whether the model is better."
- **"Blind spots move"** -- in RLHF, blind spots are in the annotator pool; in CAI, in the constitution; after red teaming, to whatever the patch does not cover (5.1.1, 5.1.3). Extends here: benchmarks have blind spots too, and optimizing for benchmarks moves the blind spots.
- **Alignment surface** -- alignment as a surface over input space; holds at some points, has gaps at others (from 5.1.3). Relevant: benchmarks sample a few points on this surface and claim to represent the whole thing.
- **"Tradeoffs, not upgrades"** -- constraints drive choice, not novelty (from 5.1.2). Extends to evaluation: different evaluation methods have different tradeoffs.

### What Was Explicitly NOT Covered

- Benchmarks or evaluation metrics for alignment (explicitly deferred to this lesson in Lessons 1, 2, and 3)
- How to measure whether alignment worked (conceptual gap this lesson fills)
- Contamination and data leakage in evaluation
- Goodhart's law applied to ML metrics
- Human evaluation methodology (inter-annotator agreement, cost, bias)
- LLM-as-judge / automated evaluation approaches
- The gap between benchmark performance and real-world behavior

### Readiness Assessment

The student is exceptionally well-prepared. This is a CONSOLIDATE lesson -- the capstone of the Build-Break-Measure arc. The student has spent three lessons building up a sophisticated understanding of alignment: what it does (Lessons 1-2), how it fails (Lesson 3), and the ongoing dynamic between attack and defense (Lesson 3). Critically, the student already has several pieces of conceptual infrastructure that this lesson integrates rather than introduces:

1. **Reward hacking from 4.4.3** is Goodhart's law in miniature -- the model optimizes against a proxy (reward model) and the proxy diverges from the actual goal. This lesson generalizes that insight: benchmarks are proxies for capability, and optimizing for the proxy diverges from actual capability.

2. **The alignment surface from 5.1.3** -- the idea that alignment holds at some points and fails at others -- directly transfers to benchmarks: a benchmark samples a few points on the capability surface and claims to represent the whole thing.

3. **The "blind spots move" mental model** has been reinforced across three lessons and extends naturally to evaluation: when you optimize for a benchmark, performance improves on measured dimensions and potentially degrades on unmeasured ones.

No gaps need resolution. The student has all the conceptual tools; this lesson connects them to the evaluation domain.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to **critically assess LLM evaluation by understanding what benchmarks actually measure vs what they claim to measure, why contamination undermines evaluation, why optimizing for metrics diverges from actual capability (Goodhart's law), and why evaluation may be fundamentally harder than training.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Reward hacking as optimizing against an imperfect proxy | INTRODUCED | INTRODUCED | rlhf-and-alignment (4.4.3) | OK | Goodhart's law for evaluation is the same mechanism as reward hacking: optimizing against a proxy that diverges from the real goal. The student has the intuition ("editor with blind spots"); this lesson applies it to benchmarks. INTRODUCED is sufficient -- the student needs the concept, not implementation details. |
| Alignment surface (alignment holds at some points, gaps at others) | INTRODUCED | DEVELOPED | red-teaming-and-adversarial-evaluation (5.1.3) | OK | Benchmarks sample a few points on the capability/alignment surface. The student understands the surface metaphor. This lesson extends it: benchmarks are sparse samples on a surface too large to fully measure. INTRODUCED sufficient; student has DEVELOPED. |
| "Blind spots move" mental model | INTRODUCED | INTRODUCED | constitutional-ai (5.1.1), red-teaming (5.1.3) | OK | The student knows that fixing one blind spot creates another. This extends to evaluation: optimizing for one metric moves the blind spot to unmeasured dimensions. INTRODUCED is sufficient. |
| Red teaming as discovering failures that benchmarks miss | INTRODUCED | DEVELOPED | red-teaming-and-adversarial-evaluation (5.1.3) | OK | The student has already seen that a model can pass benchmarks but fail on red teaming (demographic bias example from 5.1.3 hook). This is the entry point for "benchmarks have blind spots." INTRODUCED sufficient; student has DEVELOPED. |
| Automated evaluation using LLMs (scaling argument) | INTRODUCED | INTRODUCED | red-teaming-and-adversarial-evaluation (5.1.3) | OK | Automated red teaming established the pattern of "LLMs evaluating LLMs." This lesson extends it to LLM-as-judge for general evaluation. Same scaling insight, different application. INTRODUCED is sufficient. |
| Human annotation bottleneck (cost, consistency, scale) | DEVELOPED | DEVELOPED | constitutional-ai (5.1.1) | OK | The same three problems (cost, consistency, scale) that motivated RLAIF apply to human evaluation of LLM outputs. The student has vivid examples (lock-picking annotator disagreement). Direct callback. |
| Generalization from training data | DEVELOPED | DEVELOPED | Series 1 (the-learning-problem) | OK | The concept that models generalize from a sample and degrade on out-of-distribution inputs. Applied to evaluation: benchmarks are samples, and performance on the sample may not generalize to the full distribution of real-world use. Deeply established. |

All prerequisites are at sufficient depth. No gaps to resolve.

### Gap Resolution

No gaps. All prerequisites met.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **"Higher benchmark score = better model"** | Leaderboards rank models by single scores. The student has seen benchmark comparisons in blogs, papers, and marketing materials. The implicit message is always "higher number = better." This conflates performance on a specific test with general capability. | GPT-4 scores higher than GPT-3.5 on MMLU, but GPT-3.5 often gives more concise, practical answers for simple questions. A model trained specifically to maximize MMLU performance (through contamination or narrow optimization) could score higher than GPT-4 on MMLU while being worse at everything else. The benchmark measures one dimension; "better" is multidimensional. A model that scores 90% on a contaminated benchmark may genuinely understand less than a model scoring 80% on a clean version. | Hook section. Open with two models and their benchmark scores, then show real-world performance that contradicts the ranking. |
| **"Benchmarks test what they claim to test"** | Benchmark names sound authoritative: "Massive Multitask Language Understanding" (MMLU), "TruthfulQA," "HumanEval." The student assumes a benchmark named "TruthfulQA" actually measures truthfulness. In reality, benchmarks test specific proxy behaviors (pattern matching on multiple-choice questions, matching pre-selected "truthful" answers) that correlate with but do not equal the named capability. | TruthfulQA measures whether a model's answer matches a pre-written "truthful" response. A model that gives a nuanced, correct answer phrased differently from the reference answer scores LOWER than a model that memorized the reference phrasing. The benchmark measures reference-matching, not truthfulness. Similarly, MMLU tests multiple-choice performance, which requires recognition (picking from options) rather than generation (producing an answer from scratch) -- a fundamentally different cognitive task. | Core concept section, when analyzing what benchmarks actually measure. The gap between benchmark name and benchmark mechanism is the key insight. |
| **"Contamination is just cheating -- clean it up and the problem is solved"** | The student's software engineering background suggests contamination is like a test data leak: find the leak, fix it, re-run the test. But contamination in LLM evaluation is structural, not accidental. Models are trained on internet-scale data that includes benchmark questions, answers, and discussions. You cannot fully decontaminate because you cannot fully audit the training data, and new benchmarks become contaminated the moment they are published online. | Researchers created a new "clean" benchmark and published it as a paper. Within months, the benchmark questions appeared in web crawls used for pretraining. Models trained after publication showed suspiciously high performance on those exact questions. The benchmark became contaminated by the act of being published. Decontamination is a temporary state, not a permanent fix -- every public benchmark eventually leaks into training data. | Contamination section. The structural nature of contamination (it is not a bug to fix but a property of training on internet-scale data) is the key insight. |
| **"Human evaluation is the gold standard"** | Human judgment feels like the ultimate arbiter -- surely a human can tell if a model's response is good. The student may assume that when benchmarks disagree, human evaluation provides the ground truth. But human evaluation has its own deep problems: inter-annotator disagreement, systematic biases, cost that prevents scale, inability to evaluate specialized domains. | Two expert annotators evaluating the same model response to a complex coding question. Annotator A rates it 4/5 (correct approach, minor style issues). Annotator B rates it 2/5 (would not pass code review in their codebase). Both are "right" by their own standards. Inter-annotator agreement on open-ended quality judgments is often surprisingly low (Cohen's kappa of 0.3-0.5 for nuanced evaluations). If the "ground truth" method disagrees with itself, it is not actually ground truth. | Human evaluation section. The lock-picking annotator disagreement from 5.1.1 is a direct callback -- the same disagreement problem that motivated RLAIF also undermines human evaluation as ground truth. |
| **"We just need better benchmarks"** | The student may conclude that current benchmarks are flawed but solvable -- if we design better benchmarks, we will have reliable evaluation. This is the optimistic engineering response: identify the problem, build the fix. But Goodhart's law means that ANY benchmark, no matter how well-designed, becomes less useful as a measure once it becomes a target for optimization. The problem is not bad benchmarks; the problem is the relationship between measurement and optimization. | Imagine a perfectly designed benchmark that captures exactly what you care about today. You publish it. Labs begin optimizing for it. Within a year, models exploit patterns in the benchmark format (e.g., learning that multiple-choice distractors follow certain linguistic patterns) rather than developing the underlying capability. The benchmark score goes up while the capability it was designed to measure does not improve proportionally. This is not a flaw in the benchmark design -- it is Goodhart's law: "When a measure becomes a target, it ceases to be a good measure." The same mechanism as reward hacking, applied to evaluation. | Goodhart's law section. This is the deepest misconception -- it reframes the evaluation problem from "build better tools" to "understand the fundamental limits of measurement." |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Two models with inverted benchmark vs real-world rankings** | Positive | Show that benchmark rankings do not reliably predict real-world usefulness. Model A scores higher on MMLU and HumanEval but produces verbose, over-hedged answers. Model B scores lower but gives concise, actionable responses. Users overwhelmingly prefer Model B. The benchmark measures capability; the user measures usefulness. These are different things. | This is the simplest, most visceral demonstration that "higher score is not always better." It grounds the entire lesson: if benchmarks do not reliably predict what users care about, what do they actually measure? The two-model comparison is a pattern the student knows from Lesson 2 (PPO vs DPO comparison). |
| **Contamination detection: suspiciously high performance on a specific subset** | Positive | Show how contamination manifests. A model scores 95% on one section of a benchmark but only 70% on another section of similar difficulty. The high-scoring section overlaps with known training data. The score is real (the model got the answers right) but the capability is not (the model memorized the answers, not the reasoning). | Makes contamination concrete and detectable. The student can see the forensic evidence -- uneven performance across supposedly equivalent sections. Connects to the student's programming intuition: this is like a test suite where one module has 95% coverage because someone hardcoded the expected outputs. |
| **Goodhart's law: reward hacking revisited as evaluation hacking** | Positive | Connect reward hacking (from 4.4.3) to evaluation. In reward hacking, the model learns to exploit the reward model (verbose answers, confident filler). In evaluation hacking, the ecosystem learns to exploit benchmarks (training on benchmark-similar data, formatting outputs to match expected patterns, selecting benchmarks where a model performs well). Same mechanism, different scale: model vs ecosystem. | This is the deepest connection in the lesson. The student already understands reward hacking as "optimizing against a proxy diverges from the actual goal." Goodhart's law IS this insight, generalized. Extending a known concept to a new domain is more powerful than introducing Goodhart's law as a new idea. The "of course" moment: "Of course benchmarks suffer from the same problem as reward models -- they are both proxies." |
| **LLM-as-judge giving high scores to verbose, confident responses** | Negative (for "automated eval solves human eval problems") | Show that LLM judges inherit and amplify the biases of their training. An LLM-as-judge systematically rates longer responses higher, even when the shorter response is more accurate. It rates confident-sounding responses higher, even when the hedging response is more honest. The automated judge has "blind spots" just like the reward model -- same pattern from 4.4.3. | This defeats the assumption that automating evaluation solves the problems of human evaluation. The student has seen this pattern three times now: human annotators have biases (5.1.1), red team models have blind spots (5.1.3), and now LLM judges have biases too. The consistent pattern -- "the evaluator's limitations become the evaluation's limitations" -- is the meta-insight. |
| **A model that "passes" safety benchmarks but fails red teaming** | Negative (for "benchmarks measure what they claim") | Direct callback to the 5.1.3 hook: the model that passed safety tests but failed on demographic bias probing, sycophancy, and indirect requests. The student has already seen this example. Re-presenting it through the evaluation lens reinforces: the benchmark measured safety on a narrow sample, not safety in general. The alignment surface has gaps that the benchmark did not sample. | This is a callback, not a new example. It connects Lesson 3 to Lesson 4 -- the same failure that motivated "why red teaming is necessary" now motivates "why benchmarks are insufficient." Seeing the same example through two different lenses (adversarial testing lens in Lesson 3, evaluation lens in Lesson 4) deepens understanding. |

---

## Phase 3: Design

### Narrative Arc

The student has spent three lessons in the Build-Break-Measure arc. Lessons 1-2 built alignment techniques: constitutional AI, RLAIF, DPO variations. Lesson 3 broke them: red teaming revealed that alignment is never done, that models pass standard tests while failing on adversarial probing, that blind spots move with every patch. Now comes the final question, and it is the hardest one: how do you actually measure whether any of it worked? Not "did the model refuse this one harmful prompt" but "is this model aligned, capable, truthful, and useful?" The student who has been reading benchmark comparisons in blog posts and papers ("Model X achieves 92.4% on MMLU, beating Model Y's 89.1%") needs to understand what those numbers actually mean -- and what they hide. The uncomfortable truth is that evaluation may be fundamentally harder than training. Training has a clear objective: minimize a loss function. Evaluation requires answering "what do we actually want from this model?" -- a question that is multidimensional, context-dependent, and changes over time. Every benchmark is a proxy, and Goodhart's law guarantees that optimizing against any proxy eventually diverges from the thing you actually care about. The student already has this intuition from reward hacking in Series 4: the model learned to game the reward model with verbose, confident answers. Evaluation hacking is the same mechanism at a larger scale -- the entire ecosystem learns to game benchmarks. This lesson does not teach the student to distrust all benchmarks. It teaches them to read evaluation results critically: to ask what the benchmark actually measures (not what it claims to measure), whether contamination is plausible, what dimensions are not being measured, and whether the human or automated judges have their own blind spots. The capstone insight is that evaluation is not a solved problem -- it is an active research frontier that shapes how the field develops, because what you measure determines what you optimize for.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Verbal/Analogy** | "Benchmarks are standardized tests for LLMs. Standardized tests have the same problems in education that benchmarks have in ML: teaching to the test (optimization pressure), test prep companies (contamination), the SAT measuring 'test-taking ability' more than 'intelligence' (proxy vs reality), and schools being ranked by test scores rather than learning outcomes (Goodhart's law). The student who aces the SAT and bombs college is the model that tops the leaderboard and disappoints users." | The student is familiar with standardized testing and its well-known limitations. This analogy maps precisely to every major evaluation problem: contamination = test prep, Goodhart's law = teaching to the test, proxy gap = SAT vs actual intelligence. The analogy makes abstract evaluation concepts immediately graspable and grounds them in lived experience. |
| **Visual (benchmark anatomy diagram)** | A diagram breaking down what a benchmark score actually consists of: (1) the benchmark's design choices (question format, answer format, scoring rubric), (2) the overlap between benchmark questions and training data (contamination), (3) the gap between what the benchmark measures and what users actually care about (proxy gap), (4) the coverage of the benchmark relative to the full capability surface (sampling). Each component visually shows how the final number is a product of many layers of indirection, not a direct measurement of capability. | The concept "a benchmark score is not what it appears to be" is abstract. A visual decomposition makes it concrete -- the student can see the layers between "model capability" and "benchmark number." This is similar to how the pipeline comparison diagram in Lesson 1 made the RLHF vs CAI difference visible. |
| **Concrete examples (worked evaluation comparison)** | A worked comparison of two models on the same benchmark: one scores higher through memorization (contamination), the other scores lower but demonstrates genuine understanding. Walk through the evidence: uneven performance across sections, suspiciously high scores on publicly available questions vs lower scores on held-out questions, answers that match reference phrasing exactly (memorization signal). | This makes contamination concrete and detective-like. Instead of abstractly discussing contamination as a concept, the student traces forensic evidence in benchmark results. This engages the student's analytical skills and makes the problem feel like something they could detect in practice. |
| **Intuitive ("of course" framing)** | "Of course benchmarks have the same problem as reward models -- they are both proxies for what we actually want. The reward model is a proxy for human preferences. The benchmark is a proxy for model capability. And we already know what happens when you optimize against a proxy: reward hacking. Benchmark optimization is reward hacking at a different scale." | This connects the lesson's deepest insight (Goodhart's law for evaluation) to an existing concept the student has at INTRODUCED depth (reward hacking from 4.4.3). The "of course" moment reframes Goodhart's law not as a new concept but as a familiar one in a new costume. This is CONSOLIDATE at its best -- making connections, not introducing new mechanisms. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. **Contamination** as a structural property of training on internet-scale data (not just "data leakage" but an unavoidable consequence of web-crawled pretraining)
  2. **Goodhart's law applied to LLM evaluation** -- when a benchmark becomes a target, it ceases to be a good measure. (Note: reward hacking is Goodhart's law applied to training; this extends it to evaluation. The mechanism is not new, but the application domain is.)

  The other topics (benchmark limitations, human evaluation challenges, LLM-as-judge) are applications and extensions of existing concepts, not genuinely new mechanisms.

- **Previous lesson load:** BUILD (red-teaming-and-adversarial-evaluation -- adversarial thinking was new but accessible, organized around a taxonomy with structural breadth)
- **This lesson's load:** CONSOLIDATE -- appropriate. This is the module capstone. The lesson integrates concepts from all three prior lessons rather than introducing radically new mechanisms. Contamination is genuinely new but conceptually simple (the training data includes the test). Goodhart's law is an extension of reward hacking, which the student already has. Human evaluation challenges are a callback to the annotation bottleneck from 5.1.1. LLM-as-judge extends automated red teaming from 5.1.3. The cognitive challenge is synthesis and critical thinking, not absorbing new mechanisms. This is lower load than any of the previous three lessons, which is the right trajectory for a capstone.

### Connections to Prior Concepts

- **Reward hacking from 4.4.3:** This is the most important connection. Reward hacking = optimizing against a proxy (reward model) that diverges from the real goal (human satisfaction). Goodhart's law for evaluation = the ecosystem optimizing against a proxy (benchmark) that diverges from the real goal (model capability). Same mechanism, different scale. "Remember reward hacking -- the model learned to game the reward model with verbose, confident answers. Now imagine the entire ML ecosystem gaming benchmarks the same way."
- **Alignment surface from 5.1.3:** Benchmarks sample a small number of points on the capability/alignment surface. Red teaming showed that the surface has gaps between the sampled points. Benchmarks, by design, only measure at the sampled points. A model that performs well at sampled points may have significant gaps between them.
- **Human annotation bottleneck from 5.1.1:** The same three problems (cost, consistency, scale) that motivated RLAIF apply to human evaluation. Human evaluation is expensive, annotators disagree, and manual evaluation does not scale. The lock-picking annotator disagreement is a direct callback: if annotators disagree on whether a response is harmful, they will also disagree on whether a response is "good."
- **"Blind spots move" from 5.1.1 and 5.1.3:** When you optimize for a benchmark, performance improves on measured dimensions. But unmeasured dimensions may degrade (or at least not improve). The blind spots move to whatever the benchmark does not measure. This is the same pattern: RLHF blind spots are in the annotator pool, CAI blind spots are in the constitution, benchmark blind spots are in the question set.
- **Automated red teaming from 5.1.3:** The scaling argument for LLM-as-judge is the same as for automated red teaming: humans cannot evaluate enough outputs at scale. The solution (use AI) is the same. And the limitation is the same: the evaluating AI has its own blind spots.
- **"The challenge shifts, not disappears" from 5.1.1:** Extends to evaluation. The challenge shifts from "make the model better" to "measure whether the model is better." And measuring is not easier than making -- it may be harder, because measurement requires defining what "better" means across all dimensions simultaneously.

**Potentially misleading analogies:** The "alignment surface" analogy from 5.1.3 could suggest that benchmarks just need to sample more points. The student might think "if we had enough benchmarks, we'd cover the surface." This is partially true but misses Goodhart's law: the more benchmarks you create, the more optimization targets you create, and each target becomes corrupted by optimization pressure. More measurement does not linearly solve the evaluation problem. The lesson should explicitly address this.

### Scope Boundaries

**This lesson IS about:**
- What benchmarks actually measure vs what they claim to measure (the proxy gap)
- Contamination as a structural property of internet-scale training (not just "data leakage")
- Goodhart's law applied to evaluation metrics -- when a measure becomes a target
- Human evaluation challenges (inter-annotator disagreement, cost, bias, inability to scale)
- LLM-as-judge as a scaling strategy and its limitations (inheriting biases, verbosity bias, sycophancy)
- The fundamental difficulty of evaluation -- why evaluation may be harder than training
- How to critically read benchmark results (what to look for, what to question)

**This lesson is NOT about:**
- Specific current benchmark scores or leaderboard positions (outdated before the lesson is finished)
- Implementing evaluation pipelines in code (the notebook has lightweight exercises)
- Designing new benchmarks or evaluation frameworks
- Statistical methodology for evaluation (significance testing, confidence intervals)
- Evaluation of non-LLM models (focus is LLM-specific evaluation challenges)
- Specific company evaluation practices or internal processes
- The full history of NLP benchmarks (GLUE, SuperGLUE, etc.) as a chronological narrative
- Constitutional AI, preference optimization, or red teaming details (Lessons 1-3)
- Political debate about what models "should" be evaluated on

**Target depths:**
- Benchmark limitations (proxy gap, what benchmarks actually measure): DEVELOPED (student can articulate why a benchmark score does not equal capability, can identify specific ways a benchmark may mislead)
- Contamination: INTRODUCED (student understands the mechanism, can explain why it is structural rather than accidental, but has not analyzed contamination in practice)
- Goodhart's law for evaluation: DEVELOPED (student can explain the mechanism, connect it to reward hacking, and articulate why "better benchmarks" does not fully solve the problem)
- Human evaluation challenges: INTRODUCED (student knows the three problems -- cost, consistency, scale -- and can connect them to the annotation bottleneck from 5.1.1)
- LLM-as-judge: INTRODUCED (student understands the approach, its scaling advantage, and its key limitations, but has not implemented or compared evaluation methods)
- Evaluation as harder than training: INTRODUCED (student can articulate the argument but this is a framing insight, not a technical skill)

### Lesson Outline

#### 1. Context + Constraints
What this lesson covers (critically assessing LLM evaluation) and what it does not (specific benchmark scores, implementing evaluation systems, statistical methodology). Position this as the "Measure it" capstone in the Build-Break-Measure arc. Explicit framing: this lesson is not about memorizing which benchmarks exist. It is about developing the critical thinking to read any evaluation result and understand what it actually tells you (and what it does not). Constraint: the goal is to make the student a sophisticated consumer of evaluation results, not a benchmark designer.

#### 2. Recap
Brief re-activation of the key concepts from Lessons 1-3 that evaluation integrates: (a) alignment techniques are diverse and make different tradeoffs (Lesson 2) -- but how do you know which tradeoff is best for a given use case? (b) Red teaming reveals failures that standard tests miss (Lesson 3) -- the model that passed benchmarks but failed on demographic bias, sycophancy, and indirect requests. (c) Reward hacking from 4.4.3 -- the model learned to game the proxy. Set up the question: "You built alignment (Lessons 1-2). You tested it adversarially (Lesson 3). Now: how do you measure whether it actually worked? And how much should you trust that measurement?"

#### 3. Hook (Misconception Reveal)
Present two models with their benchmark scores. Model A: MMLU 92.4%, HumanEval 87.2%, TruthfulQA 71.8%. Model B: MMLU 88.1%, HumanEval 82.5%, TruthfulQA 68.4%. Ask the student: "Which model is better?" The obvious answer is Model A -- it wins on every benchmark. Then reveal three real-world comparisons: (a) Model B gives more concise, actionable answers while Model A hedges excessively; (b) Model A's MMLU score is suspiciously high on publicly available question subsets vs held-out questions; (c) users in a blind evaluation prefer Model B's responses 63% of the time. The punchline: "Model A has higher scores on every benchmark. Model B is the model users actually prefer. What went wrong?"

**Why this hook type:** The misconception reveal (higher score = better model) is the entry point for the entire lesson. The two-model comparison is concrete, immediate, and creates the right question: if benchmarks do not reliably predict user preference, what do they actually measure? This parallels the 5.1.3 hook (three passes, three failures), adapted for the evaluation domain.

#### 4. Explain: The Benchmark Zoo
Introduce the major families of LLM benchmarks, NOT as a catalog to memorize but as categories with different measurement strategies:

- **Knowledge/reasoning benchmarks** (MMLU, ARC, HellaSwag): Multiple-choice format. Measure recognition (selecting the right answer from options), not generation (producing an answer). High performance may reflect test-taking skill more than understanding.
- **Code benchmarks** (HumanEval, MBPP): Functional correctness (does the code pass tests?). Closest to objective evaluation, but narrow -- measures ability to write functions in isolation, not real-world software engineering.
- **Safety/alignment benchmarks** (TruthfulQA, ToxiGen, BBQ): Measure specific safety properties. But safety is multidimensional, and each benchmark covers a narrow slice (as the student saw in Lesson 3 -- the model that passed safety benchmarks but failed red teaming).
- **Open-ended generation benchmarks** (AlpacaEval, MT-Bench, Chatbot Arena): Measure response quality via human or LLM judgments. Most ecologically valid but most subjective and expensive.

The key insight is NOT the specific benchmarks but the pattern: every benchmark makes design choices (format, scoring, coverage) that determine what it actually measures, and those design choices may differ significantly from what the benchmark name implies.

Introduce the benchmark anatomy diagram here: the layers between "model capability" and "benchmark number."

#### 5. Check 1: What Does This Benchmark Actually Measure?
Present a benchmark description (e.g., MMLU: "tests knowledge across 57 subjects using multiple-choice questions"). Ask the student: "What capability does this benchmark actually test? What capability does the name suggest? What is the gap?" The student should identify: MMLU tests multiple-choice selection (recognition, pattern matching on distractors) across a broad range of topics. It does NOT test the ability to explain a concept, reason through a novel problem, or apply knowledge in context. A student who is good at multiple-choice tests is not the same as a student who deeply understands the material. Same for models.

#### 6. Explain: Contamination -- When the Test Is in the Training Data
The core mechanism: LLMs are trained on internet-scale data. Benchmark questions, answers, and discussions exist on the internet. Models may have seen the exact questions (or close paraphrases) during training. This is not "cheating" in the intentional sense -- it is a structural consequence of training on web crawls.

Three forms of contamination:
- **Direct contamination:** The exact benchmark question and answer appear in training data. The model memorizes the answer.
- **Indirect contamination:** Discussions, explanations, or paraphrases of benchmark questions appear in training data. The model has been "tutored" on the material without seeing the exact question.
- **Benchmark saturation:** A benchmark has been used so widely that labs inadvertently optimize for it -- choosing architectures, data mixes, and training procedures that perform well on the benchmark, even without directly training on its questions.

The structural argument: "Contamination is not a bug to fix. It is a property of the training paradigm. Any benchmark that is published becomes part of the internet, which becomes part of training data. Decontamination is a temporary state -- it expires when the benchmark goes public." The student's software engineering parallel: this is like publishing your test suite and being surprised that the code passes all tests.

Worked example: Show uneven performance across benchmark sections. The model scores 95% on questions that appear in Common Crawl and 72% on questions that do not. Both sections are the same difficulty. The 23-point gap is the contamination signal.

#### 7. Explain: Goodhart's Law -- When Measuring Changes Behavior
The core insight, connected to reward hacking: "When a measure becomes a target, it ceases to be a good measure." The student already knows this mechanism from 4.4.3 -- the model learned to game the reward model with verbose, confident answers. That was Goodhart's law inside training. Now apply it to evaluation:

- **Within a lab:** A team optimizes for MMLU. They select architectures, data mixes, and training schedules that improve MMLU scores. MMLU performance improves. But does the model actually understand more, or did they just optimize the test-taking pipeline? The answer: probably some of both, but you cannot tell from the MMLU score alone.
- **Across the ecosystem:** Leaderboards drive attention, funding, and talent. Labs optimize for leaderboard benchmarks. Benchmarks become optimization targets. The correlation between benchmark score and actual capability degrades as optimization pressure increases.

The "of course" moment: "Of course this happens -- it is the same mechanism as reward hacking. The reward model was a proxy for human preferences, and the model learned to exploit it. Benchmarks are a proxy for capability, and the ecosystem learns to exploit them. The proxy diverges from reality under optimization pressure."

GradientCard: "Goodhart's law does not mean benchmarks are useless. It means benchmark scores require interpretation. A 5-point improvement on a well-known benchmark by a lab that has been optimizing for that benchmark is less impressive than a 2-point improvement on a new benchmark no one has optimized for."

#### 8. Explain: Human Evaluation -- The Imperfect Gold Standard
Human evaluation is often presented as the ground truth -- when benchmarks disagree, ask a human. But human evaluation has its own deep problems, and the student has already seen most of them:

- **Inter-annotator disagreement** (callback to 5.1.1 lock-picking example): If annotators cannot agree on whether a response is good, "human evaluation" is not a single signal but a distribution of opinions. Cohen's kappa for open-ended quality judgments is often 0.3-0.5 -- barely above chance for binary agreement.
- **Cost** (callback to 5.1.1): Human evaluation is expensive. Meaningful evaluation of a model requires thousands of judgments across diverse prompts. At $15-30 per hour for skilled annotators, evaluating a single model update can cost tens of thousands of dollars.
- **Scale** (callback to 5.1.1): Humans cannot evaluate enough outputs to cover the capability surface. This is the same bottleneck that motivated RLAIF -- and the solution for evaluation is the same: use AI.
- **Bias:** Annotators have systematic biases -- they prefer verbose responses (length bias), confident responses (authority bias), responses that agree with their priors (confirmation bias). These biases are invisible in the evaluation results.

The Chatbot Arena as a partial solution: pairwise blind comparisons from real users, Elo-style ranking. Why it works better than scored evaluation: relative comparisons are more reliable than absolute ratings (the student knows this from preference pairs in RLHF). Why it still has problems: selection bias (users choose which queries to submit), demographic bias (who participates), cost of participation.

#### 9. Explain: LLM-as-Judge -- Scaling Evaluation with AI
The scaling argument: the same insight that drove RLAIF (use AI instead of humans for preference labels) and automated red teaming (use AI instead of humans for adversarial probing) applies to evaluation. Use an LLM to judge the quality of another LLM's outputs.

How it works: present the judge model with a prompt, one or two model responses, and a rubric. The judge provides a rating or comparison. Scale this to thousands of evaluations.

Why it is attractive: cost (orders of magnitude cheaper than humans), speed (evaluate thousands of outputs in minutes), consistency (same judge, same criteria every time -- no inter-annotator disagreement).

But the blind spots return. LLM judges have systematic biases:
- **Verbosity bias:** Rate longer responses higher, even when brevity is better
- **Confidence bias:** Rate confident-sounding responses higher, even when hedging is more honest
- **Self-preference bias:** Rate responses from similar models higher (GPT-4 judging GPT-4 outputs more favorably)
- **Format sensitivity:** Rate well-formatted (markdown, bullet points) responses higher regardless of content

The pattern: "The evaluator's limitations become the evaluation's limitations." This is the third time the student has seen this pattern in this module: human annotators have biases (5.1.1), red team models have blind spots (5.1.3), and now LLM judges have biases. The consistent insight: no single evaluation source is reliable. The best evaluation combines multiple methods -- the same defense-in-depth principle from 5.1.3, applied to evaluation.

#### 10. Check 2: Evaluate the Evaluation
Present a scenario: a lab claims "our model achieves state-of-the-art performance" based on three benchmarks. Give the student the benchmark names, scores, and brief descriptions of how the evaluation was done. Ask: "List three questions you would ask before trusting this claim." The student should identify questions like: Was the training data decontaminated for these benchmarks? Were these benchmarks chosen because the model performs well on them (selection bias)? What does the benchmark format (multiple-choice, generation, comparison) actually test? Were the scores compared to other models evaluated under the same conditions? What dimensions of model quality are NOT measured by these three benchmarks?

#### 11. Elaborate: Why Evaluation May Be Harder Than Training
The capstone insight. Training has a well-defined objective: minimize loss. The loss function is imperfect (it is a proxy), but the optimization procedure is clear. Evaluation asks: "did the training work?" -- which requires answering "work for what?" This is a fundamentally harder question because:

- **Multidimensionality:** Models must be helpful, harmless, honest, concise, creative, accurate, and more. These dimensions conflict (more cautious = less helpful, more creative = less accurate). No single number captures all dimensions.
- **Context-dependence:** "Good" depends on who is asking, what they need, and when. A medical professional wants different things from a model than a student or a creative writer. The same response can be excellent for one user and dangerous for another.
- **Moving targets:** What "good" means changes as models improve. Benchmarks that were challenging three years ago are now saturated. The evaluation must evolve with the capability, but the capability is moving faster than the evaluation.
- **The proxy problem is recursive:** You evaluate with benchmarks (proxies), and when you evaluate the benchmarks themselves, you use meta-benchmarks (proxies for the proxies). There is no ground truth at the bottom -- only increasingly indirect measurements.

Connect to the module arc: "In Lesson 1, the challenge was getting enough alignment data. Constitutional AI shifted the challenge to designing the right principles. In Lesson 3, the challenge shifted to finding failures adversarially. Now the challenge shifts again: measuring whether any of it worked. Each shift does not make the problem easier -- it reveals deeper layers of difficulty. The challenge shifts, not disappears."

#### 12. Practice (Notebook -- lightweight, 3 exercises)

- **Exercise 1 (Guided): Benchmark Autopsy.** Given a model's scores on 5 benchmark categories, identify which scores are suspicious (uneven performance suggesting contamination), which benchmarks test recognition vs generation, and which dimensions of quality are not measured. Scaffolded: first two categories have guided questions ("What format does this benchmark use? What does that format actually test?"), last three are unscaffolded. Insight: reading benchmark results critically is a skill, not just skepticism.

- **Exercise 2 (Supported): LLM-as-Judge Bias Detection.** Use an LLM API to judge pairs of responses where one is longer but less accurate and the other is concise but correct. Vary response length and confidence level systematically. Track how the judge's ratings correlate with length and confidence vs accuracy. Visualize the bias. Insight: the judge has systematic biases that are measurable and predictable -- the evaluator's blind spots become the evaluation's blind spots.

- **Exercise 3 (Supported): Design an Evaluation.** Given a specific use case (e.g., "a medical Q&A assistant for patients"), design an evaluation strategy. What benchmarks would you use? What would you evaluate with human judges? What would you evaluate with an LLM judge? What dimensions require red teaming rather than benchmarking? Produce a one-page evaluation plan. Insight: there is no single right evaluation -- evaluation design requires the same tradeoff thinking as alignment technique selection (callback to Lesson 2's design space framework). This exercise integrates the entire module.

Exercises are cumulative: Exercise 1 builds critical reading skills, Exercise 2 demonstrates automated evaluation biases empirically, Exercise 3 synthesizes the full module into an evaluation design task. Solutions should emphasize the reasoning process (why each evaluation choice was made, what it catches and what it misses) rather than a single "right answer."

#### 13. Summarize
Key takeaways:
- Benchmark scores are proxies, not measurements. Every score passes through layers of design choices, potential contamination, and optimization pressure before reaching you.
- Contamination is structural, not accidental. Training on internet-scale data means any public benchmark eventually leaks into training data. Decontamination is temporary.
- Goodhart's law applies to evaluation: when a benchmark becomes a target, it ceases to be a good measure. This is reward hacking at the ecosystem level -- the same mechanism from 4.4.3, applied to evaluation rather than training.
- Human evaluation is imperfect: expensive, inconsistent, biased, and does not scale. But pairwise comparisons (Chatbot Arena) are more reliable than absolute ratings.
- LLM-as-judge scales evaluation but inherits its own biases (verbosity, confidence, self-preference). The evaluator's limitations become the evaluation's limitations.
- Evaluation may be harder than training because it requires defining "what do we actually want?" -- a multidimensional, context-dependent, moving target with no ground truth.

Echo the module arc: "Build it (Lessons 1-2). Break it (Lesson 3). Measure it (this lesson). And measuring may be the hardest part."

Echo the meta-pattern: "The challenge shifts, not disappears." From "build alignment" to "test alignment" to "measure alignment" -- each step reveals deeper difficulty. The student now has the tools to think critically about every evaluation claim they encounter.

#### 14. Module Wrap-Up
Brief reflection on the full module arc:
- Lesson 1 (Constitutional AI): How to align models at scale -- the "build" phase
- Lesson 2 (Alignment Landscape): The design space of alignment techniques -- "build, with tradeoffs"
- Lesson 3 (Red Teaming): How to find where alignment fails -- the "break" phase
- Lesson 4 (Evaluating LLMs): How to measure whether alignment worked -- the "measure" phase

The recurring patterns across all four lessons: blind spots move, the challenge shifts, tradeoffs are unavoidable, proxies diverge under optimization pressure, scaling requires automation (with its own blind spots).

What the student can now do: read an alignment paper or a benchmark result and critically assess the claims. Not by dismissing everything, but by asking the right questions: What was actually measured? How might optimization pressure distort the results? What is not being measured? What are the blind spots of the evaluation method?

#### 15. Next Step
Preview Module 5.2 (Reasoning & In-Context Learning): "You now understand how models are aligned, tested, and evaluated. Module 5.2 shifts focus from alignment to capability: how do language models learn from examples in the prompt without updating their weights? How does 'let's think step by step' turn a mediocre model into a strong reasoner? The next module explores the surprising capabilities that emerge from next-token prediction -- capabilities that benchmarks struggle to capture."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (4 modalities: verbal/analogy standardized testing, visual benchmark anatomy diagram, concrete worked contamination example, intuitive "of course" framing connecting to reward hacking)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load ≤ 3 new concepts (2 genuinely new: contamination as structural, Goodhart's law for evaluation)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings -- the student would not get lost or form wrong mental models. The lesson is well-structured, hits all major planned content, and the CONSOLIDATE approach works. Three improvement findings exist that would make the lesson significantly stronger.

### Findings

#### [IMPROVEMENT] — Negative example for "we just need better benchmarks" is implicit, not explicit

**Location:** Goodhart's Law section (Section 7, lines 726-799)
**Issue:** The planning document identifies five misconceptions, including the deep one: "We just need better benchmarks." The plan specifies Goodhart's law should explicitly disprove this misconception with a concrete negative example -- a perfectly designed benchmark that degrades as optimization pressure builds. The lesson does address Goodhart's law well and includes the "Interpretation, Not Dismissal" GradientCard, but it never explicitly states and then disproves the "better benchmarks" misconception. The student might read the Goodhart's section and still think: "OK, current benchmarks are gamed, but if we designed better ones..." The plan's negative example (imagine a perfect benchmark → labs optimize → it degrades within a year) is not present.
**Student impact:** The student understands Goodhart's law as a concept but may not fully internalize that it applies to ALL benchmarks, including future better-designed ones. The "of course" moment -- "there is no escape from Goodhart's law via better benchmark design" -- is weakened.
**Suggested fix:** Add a paragraph or GradientCard between the "within a lab / across the ecosystem" paragraphs and the "Interpretation, Not Dismissal" card that explicitly names the misconception and walks through the negative example: "You might think the solution is better benchmarks. Imagine a perfectly designed benchmark..." This can be 4-6 sentences. The "Interpretation, Not Dismissal" card then lands as the correct framing after the misconception is defeated.

#### [IMPROVEMENT] — The planned negative example "model passes safety benchmarks but fails red teaming" is underused

**Location:** Benchmark families section (Section 4, line 542) and Hook (Section 3)
**Issue:** The planning document lists this as a key negative example: "Direct callback to the 5.1.3 hook: the model that passed safety tests but failed on demographic bias probing, sycophancy, and indirect requests." In the built lesson, this appears only as a subordinate clause within the Safety/alignment benchmarks bullet point: "as you saw in Red Teaming & Adversarial Evaluation, the model that passed safety benchmarks failed on demographic bias, sycophancy, and indirect requests." This is a callback in passing, not a developed negative example. The plan intended it to be re-presented through the evaluation lens to deepen understanding. The Hook section does touch on this pattern (Model A vs Model B), but the specific red teaming callback from Lesson 3 is not made vivid.
**Student impact:** The student misses the connection that the exact same model from Lesson 3's hook is an example of the proxy gap in action. The "seeing the same example through two different lenses" insight is diluted to a brief mention.
**Suggested fix:** After the benchmark families list and before or after the Evaluation Stack diagram, add a short callback paragraph (3-4 sentences) that explicitly re-frames the Lesson 3 hook example through the evaluation lens: "Remember the model from Red Teaming that passed all three standard safety benchmarks? Those benchmarks sampled a few points on the alignment surface. The model performed well at those specific points. But the gaps between the sampled points -- demographic bias, sycophancy, indirect requests -- were where the failures lived. The benchmark score was real. The safety it implied was not." This makes the negative example do real work rather than being an aside.

#### [IMPROVEMENT] — Misconception 5 ("we just need better benchmarks") and the "more benchmarks = more coverage" potentially misleading analogy are not explicitly addressed

**Location:** Throughout the lesson, but specifically missing after the alignment surface callback in the Contamination aside (Section 6, lines 713-719) and in the "Why Evaluation May Be Harder" section (Section 11)
**Issue:** The planning document's "Potentially misleading analogies" section identifies a specific risk: "The alignment surface analogy could suggest that benchmarks just need to sample more points. The student might think 'if we had enough benchmarks, we'd cover the surface.' This is partially true but misses Goodhart's law." The InsightBlock "The Alignment Surface Returns" in the Contamination aside (line 713) makes the alignment surface callback but does not address this misleading inference. The "Why Evaluation May Be Harder" section has "Recursive Proxy Problem" but does not explicitly address the "more benchmarks = more coverage" misconception. The student, having the alignment surface model firmly in mind, is likely to draw this incorrect conclusion.
**Student impact:** The student leaves with a potentially incomplete mental model: "benchmarks sample sparse points, so the fix is denser sampling." They miss the deeper insight that Goodhart's law corrupts each new sample point under optimization pressure, so more benchmarks does not linearly improve evaluation quality.
**Suggested fix:** Add 1-2 sentences to the InsightBlock "The Alignment Surface Returns" or to the "Why Evaluation May Be Harder" section that explicitly addresses this: "You might think the fix is more benchmarks -- denser sampling on the surface. But Goodhart's law means each new benchmark becomes an optimization target. More measurements under optimization pressure does not linearly improve measurement quality."

#### [POLISH] — Spaced em dashes in SummaryBlock strings and SVG text

**Location:** SummaryBlock description strings (lines 1220, 1226, 1244) and SVG annotation text (line 288)
**Issue:** Three SummaryBlock description strings use ` — ` (spaced em dash) instead of the project convention of no-space em dashes. The SVG text annotation on line 288 also uses a spaced em dash. The lesson prose correctly uses `&mdash;` without spaces throughout, but the JS string literals in the SummaryBlock and SVG use the spaced form.
**Student impact:** Minor visual inconsistency in the summary section and diagram. Does not affect comprehension.
**Suggested fix:** Replace ` — ` with `—` (no spaces) in the three SummaryBlock description strings and the SVG text annotation. Four edits total.

#### [POLISH] — Cohen's kappa value presented without grounding for the student

**Location:** Human Evaluation section (Section 8, line 826)
**Issue:** The lesson states "Cohen's kappa for open-ended quality judgments is often 0.3-0.5 -- barely above chance for binary agreement." The student has not been taught what Cohen's kappa is or what the scale means. The aside (WarningBlock, lines 866-873) gives a vivid example of two annotators disagreeing, which helps, but the main text drops a statistical term without definition. The parenthetical "barely above chance" helps but is vague -- "chance" for kappa is 0, not 0.5, so "barely above chance" is slightly misleading for 0.3-0.5 (which is actually "fair to moderate agreement").
**Student impact:** Minor confusion. The student understands the point (annotators disagree a lot) from context and the example, but the specific statistical claim is opaque. The characterization "barely above chance" is slightly inaccurate.
**Suggested fix:** Either (a) add a brief parenthetical: "Cohen's kappa (a measure of agreement beyond what chance alone would produce, where 0 is pure chance and 1 is perfect agreement) for open-ended quality judgments is often 0.3-0.5 -- fair to moderate agreement, far from the reliability you would want for a 'gold standard.'" Or (b) drop the kappa reference entirely and just use the annotator disagreement example, which is more vivid and sufficient.

#### [POLISH] — Exercise 3 in notebook is labeled "Supported" but has no code scaffolding

**Location:** Notebook Exercise 3 (cells 16-21)
**Issue:** Exercise 3 is labeled "Supported" in the lesson and planning document, but the notebook cells for this exercise are entirely print statements with TODO fields in Python dictionaries. The student fills in string values, not code. This is closer to a "Guided" text exercise than a "Supported" coding exercise. The distinction matters because the scaffolding progression expectations (Guided -> Supported -> Independent) are about code scaffolding, and this exercise provides none.
**Student impact:** The student is not confused, but the exercise is misclassified. The scaffolding progression label "Supported" sets an expectation for code scaffolding with solution hints, but the exercise is a structured text response.
**Suggested fix:** Either (a) relabel as "Guided" since it is a structured fill-in-the-blank with a detailed solution, or (b) accept that "Supported" can apply to design exercises (not just code exercises) and note this in the planning document as a deliberate deviation.

### Review Notes

**What works well:**
- The Build-Break-Measure arc is clearly articulated and the lesson genuinely feels like a capstone. The recap section efficiently re-activates all three prior lessons.
- The hook is excellent. The two-model comparison is concrete, immediate, and creates the right cognitive tension. The "What the Numbers Hide" reveal is well-paced.
- The Evaluation Stack diagram is a strong visual artifact that concretizes the proxy gap concept.
- The Goodhart's law section makes a strong connection to reward hacking from 4.4.3 -- the "of course" moment lands well with the "same mechanism, different scale" framing.
- The module arc summary and ModuleCompleteBlock effectively close out the module.
- The notebook is thorough: Exercise 1 is well-scaffolded with concrete MMLU questions, Exercise 2 empirically demonstrates judge biases (good connection to the lesson), and Exercise 3 synthesizes the full module.
- Callbacks to prior lessons are well-distributed: human annotation bottleneck, alignment surface, blind spots move, defense-in-depth. The lesson feels like genuine consolidation, not new material with callbacks bolted on.

**Patterns observed:**
- The three improvement findings share a common pattern: the lesson addresses concepts but does not always explicitly name and disprove the misconceptions the planning document identified. The plan has excellent misconception analysis (five misconceptions with negative examples), but the lesson implicitly addresses some of them rather than making them explicit. Misconceptions 1, 3, and 4 are well-addressed. Misconception 2 ("benchmarks test what they claim") is addressed via the proxy gap / MMLU analysis. Misconception 5 ("we just need better benchmarks") is the one most weakened by implicit treatment.
- The notebook is strong overall. The TODO-based Exercise 3 is a reasonable pedagogical choice for a design exercise, even if it does not fit the Guided/Supported/Independent scaffolding model perfectly.

---

## Review -- 2026-02-14 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All three improvement findings and two of three polish findings from iteration 1 have been resolved. The lesson is pedagogically sound, well-structured, and effective as a CONSOLIDATE capstone. No critical or improvement-level issues remain.

### Iteration 1 Resolution Check

1. **[IMPROVEMENT] "We just need better benchmarks" misconception implicit** -- RESOLVED. A dedicated GradientCard "We Just Need Better Benchmarks" (Section 7) now explicitly names the misconception, walks through the planned negative example (imagine a perfect benchmark, publish it, labs optimize, it degrades within a year), and lands the key conclusion: "The problem is not bad benchmarks. The problem is the relationship between measurement and optimization." The "Interpretation, Not Dismissal" card follows correctly as the balanced framing after the misconception is defeated. This is exactly what the plan specified.

2. **[IMPROVEMENT] Negative example "model passes safety benchmarks but fails red teaming" underused** -- RESOLVED. A dedicated GradientCard "When Passing Means Nothing" (Section 4) now re-frames the Lesson 3 hook example through the evaluation lens with vivid, specific language: "Those benchmarks sampled a few points on the alignment surface...The benchmark score was real. The safety it implied was not. That is the proxy gap in action." The negative example now does real work rather than being a subordinate clause.

3. **[IMPROVEMENT] "More benchmarks = more coverage" misleading analogy not addressed** -- RESOLVED. The InsightBlock "The Alignment Surface Returns" (Section 6 aside) now explicitly addresses the misleading inference: "The fix is not 'more benchmarks'--more samples on a corrupted surface do not improve coverage. Worse, Goodhart's law means the act of measuring changes the surface itself, because each new benchmark becomes an optimization target." This directly defeats the "denser sampling" misconception.

4. **[POLISH] Spaced em dashes in SummaryBlock and SVG** -- RESOLVED. All four instances now use unspaced em dashes consistent with project convention.

5. **[POLISH] Cohen's kappa not grounded** -- RESOLVED. Parenthetical definition added: "Cohen's kappa (a measure of agreement beyond chance, where 0 is random and 1 is perfect)." Characterization corrected from "barely above chance" to "fair to moderate agreement, far from the reliability you would want for a gold standard."

6. **[POLISH] Exercise 3 notebook labeling** -- Intentionally skipped (requires notebook modification, accepted as deliberate deviation in iteration 1).

### Findings

#### [POLISH] -- SVG diagram text may be too small on mobile devices

**Location:** EvaluationStackDiagram component (lines 59-293)
**Issue:** The SVG diagram uses font sizes of 8-13px for labels, subtitles, notes, and annotations. The right-side notes (fontSize="8") and the contamination annotation (fontSize="8") are particularly small. On mobile devices or smaller viewports, these will be difficult to read. The diagram has `overflow-x-auto` on its container, so it scrolls rather than scales, which means the small text stays small.
**Student impact:** Minor. The diagram communicates its core idea (layers between capability and leaderboard) through the layer structure and labels, which are legible at fontSize 11-13. The notes and annotations are supplementary. On desktop (the primary use context), the text is readable.
**Suggested fix:** If mobile is a future concern, the right-side notes could be moved into tooltips or shown below the diagram at smaller breakpoints. No action needed for the current single-user desktop context.

#### [POLISH] -- Notebook Exercise 2 TODO comment says "4-8 lines" but solution is ~12 lines

**Location:** Notebook cell 11 (Exercise 2, Part B)
**Issue:** The TODO comment says "YOUR CODE HERE (4-8 lines)" but the solution involves the `judge_pair` calls (2 lines), the `verbose_wins`/`concise_wins` tracking (6 lines of if/elif), and the final preference determination (4 lines of if/elif). The actual solution is closer to 12 lines. The TODO already has the solution filled in (not left as a true TODO for the student), but the "4-8 lines" estimate understates the complexity.
**Student impact:** Minimal. The solution is already present in the cell. If the student reads the hint and tries to implement before looking at the filled-in code, the line estimate might set a slightly wrong expectation, but the task is straightforward regardless.
**Suggested fix:** Change the TODO comment to "(8-14 lines)" or remove the line estimate entirely. This is a notebook edit.

### Review Notes

**What works well (reinforced from iteration 1, confirmed still strong):**

- The three improvement fixes from iteration 1 are all well-executed. The "We Just Need Better Benchmarks" GradientCard is particularly strong -- it explicitly names the misconception, provides the concrete negative example, and then the "Interpretation, Not Dismissal" card provides the balanced conclusion. This is the correct pedagogical sequence: confront the misconception, then provide the nuanced framing.
- The "When Passing Means Nothing" callback GradientCard creates a genuine "seeing through two lenses" moment. The student saw this example in Lesson 3 as motivation for red teaming; now they see it as evidence of the proxy gap. The re-framing deepens understanding.
- The "Alignment Surface Returns" InsightBlock now addresses both the contamination angle AND the Goodhart's angle in a single paragraph, creating a tight connection between the two sections.
- The lesson's strongest quality is its CONSOLIDATE character. Every major concept is an extension of something the student already knows: Goodhart's law extends reward hacking, human evaluation challenges extend the annotation bottleneck, LLM-as-judge extends automated red teaming, benchmark blind spots extend the alignment surface. The lesson introduces two genuinely new concepts (contamination as structural, Goodhart's law for evaluation) and otherwise connects, extends, and synthesizes. This is exactly right for a module capstone.
- The notebook is well-designed. Exercise 1 (Benchmark Autopsy) is an excellent guided exercise that makes the proxy gap concrete with real MMLU questions. Exercise 2 (Bias Detection) is empirically grounding -- the student measures biases rather than just reading about them. Exercise 3 (Evaluation Design) synthesizes the full module into a practical task.

**Remaining minor items (both polish, both in notebook):**
- The two polish findings are both minor and both involve the notebook rather than the lesson component. Neither affects the student's learning experience meaningfully. They can be addressed at any time without requiring re-review.
