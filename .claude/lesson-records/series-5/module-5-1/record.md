# Module 5.1: Advanced Alignment -- Record

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Constitutional AI principles as explicit alignment criteria | DEVELOPED | constitutional-ai | Principles written down define "better" -- the criteria human annotators had in their heads but never wrote down. Made explicit and auditable. Used only during TRAINING to generate data, not at inference time. |
| Critique-and-revision as a data generation mechanism | DEVELOPED | constitutional-ai | Model critiques its own response using a selected principle, then revises. The (prompt, revised response) pair becomes SFT training data. This is a DATA GENERATION process, not inference-time behavior. |
| RLAIF (Reinforcement Learning from AI Feedback) | DEVELOPED | constitutional-ai | AI model applies constitutional principles to generate preference labels (prompt + two responses + which is better), replacing human annotators. Same data format as RLHF, different source. Scales to millions of comparisons vs ~33K for InstructGPT. |
| Human annotation bottleneck (cost, consistency, scale) | DEVELOPED | constitutional-ai | Three problems: cost (months of labeling), consistency (inter-annotator disagreement on ambiguous cases), scale (can't hire enough humans for next-gen models). Sharpened with lock-picking annotator disagreement example. |
| Constitution as training-time artifact (not inference-time rules) | DEVELOPED | constitutional-ai | Critical misconception addressed: principles are baked into weights during training. You could delete the constitution after training and behavior wouldn't change. Analogized to "style guide used to train the editor, not a rulebook consulted per decision." |
| Principle failure modes (vague, conflicting, missing) | INTRODUCED | constitutional-ai | Three failure cases shown: vague principles produce random labels, conflicting principles depend on selection logic, missing principles leave blind spots. Challenge shifts from "enough annotators" to "right principles." |
| RLHF pipeline (recap) | DEVELOPED | constitutional-ai (recap from 4.4) | Reconnected across series boundary: pretraining gives knowledge, SFT gives voice, alignment gives judgment. Reward model as "experienced editor" with blind spots. ~33K human comparisons for InstructGPT. |

| Design space axes framework for preference optimization (data format, reference model, online/offline, reward model) | DEVELOPED | alignment-techniques-landscape | Four independent axes that map any preference optimization method. Taught axes-first, methods-second. Progressive reveal: sparse diagram (PPO + DPO with dashed gaps) shown before variations, complete diagram after. Central visual artifact is an inline SVG multi-axis map. The student can use the axes to classify a method they haven't seen. |
| IPO (Identity Preference Optimization) -- bounded preference signal | INTRODUCED | alignment-techniques-landscape | Same as DPO on most axes (paired data, reference model, offline, no reward model). Differs on bounded vs unbounded preference signal. DPO treats "slightly preferred" and "overwhelmingly preferred" the same way; IPO caps the signal, trading optimization strength for stability. Analogized as: DPO editor says "this draft is better," IPO editor says "this draft is slightly better." |
| KTO (Kahneman-Tversky Optimization) -- single-response preference data | INTRODUCED | alignment-techniques-landscape | Moves along the data format axis: works with single responses labeled good/bad (thumbs-up/down), not pairs. Named after prospect theory finding that humans are more sensitive to losses than gains -- model penalized MORE for bad responses than rewarded for good ones. Solves the "thumbs-up/down data" scenario. Still requires reference model. |
| ORPO (Odds Ratio Preference Optimization) -- no reference model | INTRODUCED | alignment-techniques-landscape | Eliminates reference model entirely. Folds preference optimization into SFT using odds ratio penalty. Single-pass instead of two-phase (SFT then alignment). Solves the "memory-constrained" scenario. Key teaching point: "simplification in one dimension creates complexity in another" -- removing the reference model removes the KL stability mechanism, so ORPO needs the odds ratio to compensate. |
| Online vs offline preference optimization | INTRODUCED | alignment-techniques-landscape | Offline: train on pre-collected static dataset (DPO default). Online: generate responses with current model during training (PPO native). Online DPO combines DPO's simpler optimization with online data generation. Key tradeoff: online trains on actual distribution but has cold-start problem (early bad outputs can reinforce bad behavior). Practical reality: most deployed models use offline because cost/quality tradeoff favors it. |
| Iterative alignment / self-play | INTRODUCED | alignment-techniques-landscape | Multiple rounds of generate-label-train. Each round produces a better model whose outputs feed the next round. Connected to RLAIF from constitutional AI -- AI can label preferences in each round, making iteration practical at scale. Does not require a single long training run; can be multiple discrete offline rounds. |
| Online DPO (DPO with online data generation) | INTRODUCED | alignment-techniques-landscape | Combines DPO's single-model simplicity with online learning's distribution match. Addresses the "learn from current mistakes" scenario. Positioned as a middle ground between offline DPO and full PPO. |

| Red teaming as systematic adversarial process | DEVELOPED | red-teaming-and-adversarial-evaluation | Systematic, adversarial, probing -- not random guessing or just "jailbreaks." Covers safety, consistency, fairness, factual accuracy, privacy, robustness. Pen-testing analogy: systematic probing, enormous attack surface, ongoing process. Alignment surface framing: alignment holds at most points in input space but has gaps that red teaming maps. |
| Attack-defense dynamic / asymmetry | DEVELOPED | red-teaming-and-adversarial-evaluation | Alignment is an ongoing dynamic, not a one-time fix. Each defense creates new attack surfaces. Attackers need to find ONE gap; defenders need to cover ALL gaps. DAN progression as concrete worked example of escalation across multiple generations. Connected to "the challenge shifts, not disappears" and "blind spots move" mental models from constitutional-ai. |
| Attack taxonomy: direct harmful requests | INTRODUCED | red-teaming-and-adversarial-evaluation | Category 1 -- the baseline that alignment handles well. Not interesting for red teaming because alignment explicitly covers these patterns. If a model fails here, alignment training was inadequate. |
| Attack taxonomy: indirect / reframing attacks | INTRODUCED | red-teaming-and-adversarial-evaluation | Category 2 -- same content, different framing (fiction, educational, hypothetical). Exploits surface pattern matching. Model responds to framing cues, not intent. Lock-picking example as callback from Series 4. |
| Attack taxonomy: multi-step compositional attacks | INTRODUCED | red-teaming-and-adversarial-evaluation | Category 3 -- each individual step is innocuous, composite is harmful. Exploits limited cross-turn reasoning. Related to but distinct from the three structural reasons (its own mechanism). |
| Attack taxonomy: encoding and format tricks | INTRODUCED | red-teaming-and-adversarial-evaluation | Category 4 -- Base64, ROT13, unusual Unicode, reversed text. Pure out-of-distribution failure. Alignment training data did not include encoded harmful requests. Connected to generalization failure from Series 1. |
| Attack taxonomy: persona and role-play attacks | INTRODUCED | red-teaming-and-adversarial-evaluation | Category 5 -- "You are DAN," "pretend no restrictions." Exploits the model's instruction-following ability against its safety training. Capability becomes vulnerability. |
| Attack taxonomy: few-shot jailbreaking | INTRODUCED | red-teaming-and-adversarial-evaluation | Category 6 -- provide examples of a compliant model, then ask harmful question. In-context learning picks up the adversarial pattern, overriding safety training. Model's learning ability turned against alignment. |
| Structural reasons for alignment failure: surface pattern matching | INTRODUCED | red-teaming-and-adversarial-evaluation | Reason 1 -- alignment teaches "refuse requests that LOOK harmful." Reframing changes surface while preserving intent. Model is pattern-matching, not reasoning about harm. Maps to attack categories 1-2. |
| Structural reasons for alignment failure: training distribution coverage | INTRODUCED | red-teaming-and-adversarial-evaluation | Reason 2 -- alignment training data covers a sample. Inputs sufficiently different from the sample produce unaligned behavior. Same generalization problem from Series 1 applied to alignment. Maps to attack category 4. |
| Structural reasons for alignment failure: capability-safety tension | INTRODUCED | red-teaming-and-adversarial-evaluation | Reason 3 -- model capabilities (instruction following, in-context learning) are also vulnerabilities. More capable = harder to align. Fundamental tension, not a bug. Maps to attack categories 5-6. |
| Automated red teaming (LLMs probing LLMs at scale) | INTRODUCED | red-teaming-and-adversarial-evaluation | Generate-test-classify-iterate pipeline. Red team model generates adversarial prompts, target responds, classifier judges failures, patterns guide next round. Perez et al. (2022): 154,000 prompts finding failures humans missed. Same scaling insight as RLAIF -- humans cannot cover the space. Limitations: red team model has own blind spots, automated classification imperfect, breadth not depth. |
| Defense-in-depth for alignment | INTRODUCED | red-teaming-and-adversarial-evaluation | Five layers: training-time alignment (baseline), input filtering (detect adversarial prompts), output filtering (check responses), monitoring (detect novel attack patterns), regular re-evaluation (repeat red teaming after updates). No single layer is sufficient. Each layer catches what others miss. |

| Benchmark limitations / proxy gap (what benchmarks actually measure vs what they claim) | DEVELOPED | evaluating-llms | Benchmarks are proxies for capability, not direct measurements. Every score passes through design choices (task format, prompt formatting, scoring criteria, aggregation) that shape the final number. The name on the benchmark (e.g., "understanding" in MMLU) does not match what the benchmark mechanism tests (multiple-choice recognition). Standardized testing analogy maps precisely: SAT score vs actual readiness. Evaluation Stack diagram (inline SVG) visualizes the layers between capability and leaderboard number. |
| Goodhart's law for evaluation (when a benchmark becomes a target, it ceases to be a good measure) | DEVELOPED | evaluating-llms | Extension of reward hacking from 4.4.3 to the evaluation domain. Same mechanism, different scale: reward hacking = model optimizing against a proxy (reward model), Goodhart's law for evaluation = ecosystem optimizing against a proxy (benchmark). Explicitly addressed the "we just need better benchmarks" misconception with negative example: a perfect benchmark published today degrades under optimization pressure within a year. Framed as "interpretation, not dismissal" -- benchmark scores require context about optimization pressure, not blanket skepticism. |
| Contamination as structural property of internet-scale training | INTRODUCED | evaluating-llms | Three forms: direct (exact question in training data), indirect (discussions/paraphrases), benchmark saturation (inadvertent optimization). Key insight: contamination is NOT a bug to fix but a property of the training paradigm. Any published benchmark becomes part of internet, which becomes part of training data. Decontamination is temporary -- expires when benchmark goes public. Software engineering analogy: publishing your test suite and being surprised code passes. Forensic evidence: uneven performance across equivalent sections (95% on crawled vs 72% on held-out questions). |
| Human evaluation challenges (cost, consistency, scale, bias) | INTRODUCED | evaluating-llms | Callback to annotation bottleneck from constitutional-ai. Inter-annotator disagreement quantified with Cohen's kappa (0.3-0.5 for open-ended quality judgments = "fair to moderate agreement"). Systematic biases: length bias, authority bias, confirmation bias. Chatbot Arena as partial solution (pairwise blind comparisons, Elo ranking -- same insight as preference pairs in RLHF). Human evaluation is another proxy, not ground truth. |
| LLM-as-judge (scaling evaluation with AI) | INTRODUCED | evaluating-llms | Same scaling argument as RLAIF and automated red teaming: humans cannot evaluate enough outputs. Four systematic biases: verbosity bias, confidence bias, self-preference bias, format sensitivity. Pattern: "the evaluator's limitations become the evaluation's limitations" -- third time this pattern appeared in the module (human annotators, red team models, LLM judges). Defense-in-depth principle applied to evaluation: combine benchmarks, human evaluation, LLM judges, and red teaming. |
| Evaluation as fundamentally harder than training | INTRODUCED | evaluating-llms | Training has a clear objective (minimize loss). Evaluation requires answering "work for what?" -- multidimensional (helpful/harmless/honest/concise/creative/accurate, dimensions conflict), context-dependent (different users want different things), moving targets (benchmarks saturate as models improve), recursive proxy problem (evaluating benchmarks requires meta-benchmarks). Capstone insight for the module: the challenge shifts from "build alignment" to "test alignment" to "measure alignment" -- each step reveals deeper difficulty. |

## Per-Lesson Summaries

### Lesson 1: Constitutional AI (constitutional-ai)

**Concepts taught:**
- Constitutional AI principles as explicit alignment criteria (DEVELOPED)
- Critique-and-revision as a data generation mechanism (DEVELOPED)
- RLAIF -- AI-generated preference labels replacing human labels (DEVELOPED)
- Human annotation bottleneck -- cost, consistency, scale (DEVELOPED)
- Principle failure modes -- vague, conflicting, missing principles (INTRODUCED)

**Mental models established:**
- "The editor gets a style guide" -- extends the Series 4 "reward model is an experienced editor" analogy. The editor now has explicit written principles instead of implicit judgment from examples. Same editor, better tools.
- "Same pipeline, different data source" -- constitutional AI does NOT replace the RLHF pipeline. It replaces WHERE the preference labels come from (AI applying principles vs humans applying intuition). The RL optimization is identical.
- "Cutting out the middleman" -- the "of course" insight: if the reward model learns from human preferences, and we can write those preferences as explicit principles, we can have AI apply them directly. The criteria were always there, just never written down.
- "The challenge shifts, not disappears" -- alignment difficulty moves from a labor problem (enough annotators) to a design problem (right principles). Principles can be iterated, version-controlled, and audited.
- "Blind spots move" -- in RLHF, blind spots are in the annotator pool. In CAI, blind spots are in the constitution. The pattern is the same: training data source determines alignment quality.

**Analogies used:**
- Editor + style guide (extends "editor with blind spots" from Series 4)
- Implicit to explicit criteria (annotators' heads -> written principles)
- Style guide trains the editor vs rulebook consulted per decision (training-time vs inference-time distinction)

**How concepts were taught:**
- **RLHF recap:** 3-paragraph reconnection across series boundary (alignment problem -> RLHF solution -> limitation). Re-activated "voice vs judgment" and "experienced editor" mental models.
- **Human bottleneck:** Lock-picking annotator disagreement example (extending from Series 4), GradientCard with three problems (cost, consistency, scale), then pivoting to "what if you could write down what 'good' means?"
- **Constitution:** Example principles shown (4 concrete principles), then the key insight that these ARE what annotators had implicitly. GradientCard addressing inference-time misconception immediately after.
- **Critique-and-revision:** 5-step walkthrough with PhaseCards using the lock-picking example (prompt -> principle selection -> AI critique -> AI revision -> SFT pair). CritiqueRevisionDiagram (inline SVG). GradientCard reinforcing "data generation, not inference."
- **RLAIF:** Explained as Stage 2 of CAI. PipelineComparisonDiagram (inline SVG, side-by-side RLHF vs CAI with CHANGED/SAME labels). Scale comparison: 33K vs millions.
- **Principle failures:** Three PhaseCards showing vague, conflicting, and missing principles. GradientCard: "challenge shifts, not disappears."

**Visual elements:**
- PipelineComparisonDiagram: Side-by-side RLHF vs CAI pipeline showing 4 stages (SFT data, preference labels, reward model training, RL training) with CHANGED/SAME labels highlighting what differs
- CritiqueRevisionDiagram: 5-step vertical flow (prompt -> principle -> critique -> revision -> SFT pair) with constitution annotation and "DATA GENERATION / Not inference-time behavior" callout
- Lock-picking annotator disagreement panel (grid with Annotator 1 vs Annotator 2 decisions)

**What is NOT covered:**
- Implementing CAI in code (conceptual lesson)
- DPO variations or other alignment techniques (Lesson 2)
- Red teaming or adversarial evaluation (Lesson 3)
- Alignment benchmarks or evaluation (Lesson 4)
- The political/philosophical debate about what principles should be
- The specific principles used by Anthropic or any other company
- Training a reward model or running PPO
- PPO algorithm details

**Notebook:** `notebooks/5-1-1-constitutional-ai.ipynb` (3 exercises)
- Exercise 1 (Guided): Write a principle and use LLM API to critique a response. Try different principles on same response. Insight: principles steer feedback.
- Exercise 2 (Supported): Generate revised response from critique, construct preference pair (revised > original). Insight: this IS the training data CAI generates at scale.
- Exercise 3 (Supported): Write deliberately vague principle, observe critique quality degradation. Insight: constitution quality determines alignment quality.

**Review:** Passed at iteration 2/3. Iteration 1 had 2 improvement findings (missing "of course" intuitive beat, notebook exercise dependency) and 3 polish findings, all resolved. Iteration 2 had 0 improvement findings and 2 minor polish findings (typography in SummaryBlock, hint labeling in notebook).

### Lesson 2: The Alignment Techniques Landscape (alignment-techniques-landscape)

**Concepts taught:**
- Design space axes framework for preference optimization (DEVELOPED) -- four independent axes: data format, reference model, online/offline, reward model
- IPO -- Identity Preference Optimization, bounded preference signals (INTRODUCED)
- KTO -- Kahneman-Tversky Optimization, single-response data (INTRODUCED)
- ORPO -- Odds Ratio Preference Optimization, no reference model (INTRODUCED)
- Online vs offline preference optimization (INTRODUCED)
- Iterative alignment / self-play (INTRODUCED)
- Online DPO (INTRODUCED)

**Mental models established:**
- "Alignment techniques are points in a design space, not steps on a ladder" -- the central mental model. Methods occupy different positions along four axes; no method dominates all axes. Newer does not mean better, it means different tradeoffs.
- "Tradeoffs, not upgrades" -- each variation solves a specific constraint at the cost of something else. IPO trades optimization strength for stability. KTO trades relative signal for data availability. ORPO trades the KL stability mechanism for memory savings.
- "Constraints drive choice" -- the right method depends on your constraints (data format, memory budget, compute budget, data quality), not on which paper is newest.
- "Simplification in one dimension creates complexity in another" -- removing the reference model (ORPO) removes the KL penalty mechanism, requiring an alternative stability mechanism (odds ratio).
- "The map is more durable than any specific method" -- when encountering a new alignment technique, ask: where does it sit on these four axes? What constraint does it relax? What does it give up?

**Analogies used:**
- Vehicle selection analogy: car (DPO) for most trips with good roads, motorcycle (KTO) carries less but goes places the car cannot, bus (PPO) carries the most but needs infrastructure. No "best vehicle," only best for your constraints.
- Editor calibration analogy: DPO editor says "this draft is better," IPO editor says "this draft is slightly better." Calibration prevents overfitting on weak preferences.

**How concepts were taught:**
- **Recap:** ComparisonRow of PPO vs DPO properties, grounded with the quantum computing preference pair from the RLHF lesson (4.4.3). Re-activates existing mental models before extending.
- **Hook:** Three concrete scenarios (thumbs-up/down data, memory-constrained, learn from current mistakes) that PPO and DPO cannot cleanly solve. Creates the need for the design space map before presenting it.
- **Design space axes:** Four GradientCards (one per axis), then PPO and DPO placed on the map via ComparisonRow. Progressive reveal: SparseDesignSpaceMap (inline SVG) with PPO/DPO plotted and dashed "?" markers at empty positions.
- **Variations:** Each method presented as a GradientCard with three parts: what it changes, position on the map, connection to prior knowledge. InsightBlocks and WarningBlocks in asides provide depth without bloating the main content.
- **Complete map:** Full DesignSpaceMap (inline SVG) with all five methods across all four axes, shown AFTER variations are explained. Comparison table (HTML table) provides scannable symbolic reference.
- **Online vs offline:** Dedicated section with ComparisonRow, Online DPO explanation, iterative alignment connected to RLAIF, and a cold-start problem GradientCard as negative example.
- **Choosing a method:** Returns to the three hook scenarios, solves each with the map, then adds a fourth scenario (well-resourced team -> DPO) to defeat "newer = better" misconception. "The Landscape Keeps Growing" card explicitly positions preference optimization as one region of a broader alignment landscape.
- **Checkpoints:** Two predict-before-explain checks -- (1) predict what a single-response method would gain/lose before learning about KTO, (2) place a hypothetical method on the map using the framework.

**Visual elements:**
- SparseDesignSpaceMap: Inline SVG with four horizontal axes, PPO and DPO plotted, dashed circle markers at empty positions with "?" labels. Creates curiosity by showing gaps before filling them.
- DesignSpaceMap: Complete inline SVG with all five methods (PPO, DPO, IPO, KTO, ORPO) plotted across all four axes. Color-coded (amber, indigo, violet, cyan, emerald). Legend at bottom.
- Comparison table: HTML table with all five methods as rows, four axes + "Key Insight" as columns.
- ComparisonRows: PPO vs DPO (recap), PPO vs DPO on the map, offline vs online.

**What is NOT covered:**
- Mathematical loss function derivations (no Bradley-Terry, no loss equations)
- Implementing any technique in code (conceptual lesson)
- Benchmarking or performance comparisons between methods
- Constitutional AI (previous lesson) or red teaming (next lesson)
- RL formalism (policy gradient, advantage estimation)
- Specific company choices

**Notebook:** `notebooks/5-1-2-alignment-techniques-landscape.ipynb` (3 exercises)
- Exercise 1 (Guided): Preference data format conversion -- convert between paired comparison format and single-response thumbs-up/down format. See what information is lost when dropping from relative to absolute labels. Insight: this is why KTO exists.
- Exercise 2 (Supported): Reference model drift -- compute log-probability ratios between policy and reference model, visualize KL divergence growth as policy drifts. Insight: the reference model is a stability mechanism.
- Exercise 3 (Supported): Online vs offline distribution mismatch -- see how a policy's output distribution shifts after an update, making pre-collected preference data stale. Insight: this is the core motivation for online methods.

**Review:** Passed at iteration 2/3. Iteration 1 had 4 improvement findings (missing quantum computing recap example, missing comparison table, missing progressive reveal of design space diagram, fifth misconception not explicitly addressed) and 3 polish findings. All improvement findings resolved. Iteration 2 had 0 improvement/critical findings and 3 minor polish findings (SVG em dash spacing, GradientCard title em dashes, notebook KL direction phrasing).

### Lesson 3: Red Teaming & Adversarial Evaluation (red-teaming-and-adversarial-evaluation)

**Concepts taught:**
- Red teaming as systematic adversarial process (DEVELOPED) -- systematic, adversarial, probing; covers safety, consistency, fairness, factual accuracy, privacy, robustness
- Attack-defense dynamic / asymmetry (DEVELOPED) -- alignment is never "done"; each defense creates new attack surfaces; attackers need one gap, defenders need to cover all
- Attack taxonomy: six categories organized by mechanism exploited (INTRODUCED) -- direct, indirect/reframing, multi-step, encoding, persona, few-shot
- Three structural reasons for alignment failure (INTRODUCED) -- surface pattern matching, training distribution coverage, capability-safety tension
- Automated red teaming (INTRODUCED) -- LLMs probing LLMs at scale; generate-test-classify-iterate pipeline
- Defense-in-depth (INTRODUCED) -- five layers: training-time alignment, input filtering, output filtering, monitoring, regular re-evaluation

**Mental models established:**
- "Alignment surface" -- alignment as a surface over the input space; holds at most points, has gaps at others; red teaming maps the surface to find gaps. Too large to test exhaustively, so strategies needed to find gaps efficiently.
- "Pen-testing analogy" -- red teaming an LLM parallels penetration testing a network: systematic probing, enormous attack surface, defense must be comprehensive, ongoing process not a one-time audit.
- "Capability = vulnerability" -- the model's capabilities (instruction following, in-context learning) are also its attack surfaces. More capable models have larger attack surfaces, not smaller ones.
- "Blind spots move" (extended) -- patching one gap moves the blind spot elsewhere; blind spots never vanish. Extends the mental model from constitutional-ai into the adversarial domain.
- "The challenge shifts, not disappears" (extended) -- from "build alignment" to "maintain alignment against adversarial pressure." Same pattern as 5.1.1 (annotator bottleneck to constitution design).

**Analogies used:**
- Pen-testing / penetration testing analogy (maps to software engineer background: systematic probing, enormous attack surface, ongoing)
- Alignment surface (spatial metaphor: alignment holds at some points, breaks at others)
- The DAN progression as concrete example of attack-defense co-evolution (DAN 1.0 -> patch -> DAN 2.0 -> patch -> DAN 3.0 -> persona variants)

**How concepts were taught:**
- **Recap:** Brief re-activation of Lessons 1-2 key concepts: alignment trains on a sample, blind spots move, alignment has gaps. Sets up: "how do you find those blind spots?"
- **Hook:** Three passes (direct harmful request refused, balanced answer to sensitive question, acknowledges uncertainty) followed by three failures from the SAME model (reframing attack: lock-picking via fiction, sycophancy: nuclear energy framing-dependent answers, demographic bias: different medical advice by gender). Immediately broadens beyond "red teaming = jailbreaks." WarningBlock aside: "Not Just Jailbreaks."
- **What red teaming is:** Three key words (systematic, adversarial, probing). Pen-testing analogy. Alignment surface framing. Explicit breadth paragraph: six dimensions red teams probe (safety, consistency, fairness, factual accuracy, privacy, robustness). "What Red Teaming Is Not" GradientCard distinguishing from benchmarking, adversarial training, and general QA.
- **Attack taxonomy:** Six PhaseCards, one per category, each with mechanism label and concrete example. AttackTaxonomyDiagram (inline SVG, 2x3 grid organized by sophistication). InsightBlock "Not a Flat List" + WarningBlock "Capability = Vulnerability" in asides.
- **Check 1 (Classify the Attack):** Three novel attacks for student to classify (encoding/format trick, multi-step/compositional, persona/role-play). Predict-then-reveal with detailed explanations.
- **Why aligned models fail:** Bridging paragraph explicitly mapping six taxonomy categories to three structural reasons. Three GradientCards (surface pattern matching, training distribution coverage, capability-safety tension). InsightBlock: capability-safety tension means red teaming must scale WITH model capability.
- **Automated red teaming:** Scaling argument connected to RLAIF from Lesson 1. Four PhaseCards (generate, test, classify, analyze & iterate). Perez et al. (2022) reference: 154,000 prompts. Limitations GradientCard. InsightBlock "Same Scaling Insight" + TipBlock "Breadth + Depth" in asides.
- **Check 2 (Predict the Defense):** Few-shot jailbreaking on Llama 2. Student predicts defenses and their costs. Reveal: input classifier (can be fooled), output classifier (over-refuse), additional RLHF (hurts capability). Pattern: every defense creates a new attack surface.
- **Cat-and-mouse dynamic:** DAN progression as concrete example. GradientCard "The Fundamental Asymmetry." AttackDefenseCycleDiagram (inline SVG, four-node cycle with escalation labels). Defense-in-depth: five PhaseCards (training-time alignment, input filtering, output filtering, monitoring, regular re-evaluation). Connection to "the challenge shifts, not disappears" and "blind spots move."

**Visual elements:**
- AttackTaxonomyDiagram: Inline SVG, 2x3 grid of six attack categories with color-coded borders by sophistication, mechanism labels, and "increasing sophistication" arrow along the bottom
- AttackDefenseCycleDiagram: Inline SVG, four-node cycle (Deploy -> Red Team Finds Gaps -> Patch Defenses -> Attackers Adapt -> Deploy) with escalation labels between nodes and "each cycle escalates" center annotation. Bottom note: "Attackers need to find ONE gap. Defenders need to cover ALL gaps."
- Worked pass/fail pairs in the hook: three GradientCards (emerald) for passes, three GradientCards (rose) for failures

**What is NOT covered:**
- Implementing red teaming tools or running adversarial attacks in code (notebook has lightweight exercises)
- Specific current jailbreaks in detail (patterns, not recipes)
- Political or ethical debate about AI safety (mechanisms, not policy)
- Benchmarks or evaluation metrics for safety (Lesson 4)
- Constitutional AI or preference optimization details (Lessons 1-2)
- Red teaming for non-LLM systems
- Responsible disclosure processes or red teaming governance

**Notebook:** `notebooks/5-1-3-red-teaming-and-adversarial-evaluation.ipynb` (3 exercises)
- Exercise 1 (Guided): Classify 10 adversarial prompts into the six-category attack taxonomy. Identify which mechanism each exploits. First 5 have hints, last 5 are unscaffolded. Insight: the taxonomy is a classification tool, not just a list.
- Exercise 2 (Supported): Test a model with direct request, fiction reframe, and encoded version of same question. Then invent three additional reframings. Insight: alignment holds at some points on the input surface and fails at others.
- Exercise 3 (Supported): Use an LLM to generate 20 variations of a sensitive prompt, send to target model, classify responses, visualize distribution. Insight: even at toy scale, automated probing reveals inconsistency that manual testing would miss.

**Review:** Passed at iteration 2/3. Iteration 1 had 4 improvement findings (misconception 5 not explicitly addressed, sycophancy example ambiguous, missing taxonomy-to-structural-reasons bridge, no negative example for what red teaming is NOT) and 3 polish findings. All improvement findings resolved. Iteration 2 had 0 improvement/critical findings and 2 minor polish findings (SVG em dash spacing, notebook Exercise 1 predict-then-reveal format).

### Lesson 4: Evaluating LLMs (evaluating-llms)

**Concepts taught:**
- Benchmark limitations / proxy gap -- what benchmarks actually measure vs what they claim (DEVELOPED)
- Goodhart's law for evaluation -- when a benchmark becomes a target, it ceases to be a good measure (DEVELOPED)
- Contamination as structural property of internet-scale training (INTRODUCED)
- Human evaluation challenges -- cost, consistency, scale, bias (INTRODUCED)
- LLM-as-judge -- scaling evaluation with AI and its biases (INTRODUCED)
- Evaluation as fundamentally harder than training (INTRODUCED)

**Mental models established:**
- "Benchmarks are standardized tests for LLMs" -- the SAT analogy maps precisely to every evaluation problem: contamination = test prep, Goodhart's law = teaching to the test, proxy gap = SAT score vs actual readiness. Makes abstract evaluation concepts graspable through familiar experience.
- "Same mechanism, different scale" -- reward hacking (model games proxy) and Goodhart's law for evaluation (ecosystem games proxy) are the same mechanism operating at different levels. Connected explicitly to the "editor with blind spots" from 4.4.3.
- "The evaluator's limitations become the evaluation's limitations" -- pattern that appeared three times in this module: human annotators have biases (5.1.1), red team models have blind spots (5.1.3), LLM judges have biases (this lesson). No single evaluation source is sufficient.
- "The challenge shifts, not disappears" (extended) -- from "build alignment" (Lessons 1-2) to "test alignment" (Lesson 3) to "measure alignment" (this lesson). Each shift reveals deeper difficulty. The challenge of defining "good" may be harder than optimizing for it.

**Analogies used:**
- Standardized testing analogy (SAT measures test-taking ability, not intelligence -- benchmarks measure benchmark-taking ability, not capability)
- Published test suite analogy (contamination = publishing your test suite and being surprised code passes -- maps to software engineering background)
- Editor analogy callback (reward model as editor with blind spots from 4.4.3 -- benchmarks as "editors" with the same blind spots, at ecosystem scale)

**How concepts were taught:**
- **Recap:** Brief re-activation of three prior lessons' key concepts: alignment techniques are diverse with tradeoffs (Lesson 2), red teaming reveals failures benchmarks miss (Lesson 3), reward hacking as proxy optimization (4.4.3). Sets up: "You built it, you broke it -- now how do you measure whether it worked?"
- **Hook (misconception reveal):** Two models with benchmark scores (Model A wins every benchmark). Then three reveals: Model B gives more concise answers, Model A has suspicious contamination-pattern scores, users prefer Model B 63% of the time. Punchline: higher score on every benchmark, but users prefer the other model. Creates the question: "What went wrong?"
- **Evaluation Stack:** Benchmark families presented as categories with different measurement strategies (knowledge/reasoning, code, safety/alignment, open-ended generation). Key insight: every benchmark makes design choices that determine what it measures, which may differ from what its name implies. "When Passing Means Nothing" GradientCard: callback to 5.1.3 hook -- model that passed safety benchmarks but failed on demographic bias, sycophancy, indirect requests. Re-framed through evaluation lens: "The benchmark score was real. The safety it implied was not." EvaluationStackDiagram (inline SVG): six layers from "Actual Model Capability" to "Leaderboard Position" with contamination annotation.
- **Contamination:** Three forms (direct, indirect, saturation) presented via GradientCards. Structural argument explicitly made: contamination is a property of the paradigm, not a bug to fix. Published test suite analogy. Forensic evidence worked example: 95% vs 72% on equivalent sections = contamination signal. "The Alignment Surface Returns" InsightBlock addresses misleading "more benchmarks = more coverage" inference.
- **Goodhart's law:** Connected to reward hacking ("Remember reward hacking? That was Goodhart's law inside training. Now apply it to evaluation."). Within-lab and across-ecosystem dynamics. "We Just Need Better Benchmarks" GradientCard explicitly names and disproves the misconception with negative example (perfect benchmark degrades under optimization in a year). "Interpretation, Not Dismissal" GradientCard provides balanced framing.
- **Human evaluation:** Callback to annotation bottleneck from 5.1.1 (lock-picking disagreement). Cohen's kappa defined and contextualized (0.3-0.5 = fair to moderate, far from gold standard reliability). Four problems: inter-annotator disagreement, cost, scale, bias. Chatbot Arena as partial solution (pairwise comparisons more reliable than absolute ratings -- same insight as preference pairs in RLHF).
- **LLM-as-judge:** Same scaling argument as RLAIF and automated red teaming. Four bias cards (verbosity, confidence, self-preference, format sensitivity). Pattern explicitly identified: third time the module shows "evaluator's limitations = evaluation's limitations." Defense-in-depth principle applied to evaluation.
- **Why evaluation may be harder than training:** Four GradientCards (multidimensionality, context-dependence, moving targets, recursive proxy problem). Connected to module arc: each lesson shifted the challenge, never resolved it.
- **Module arc summary:** Build-Break-Measure arc explicitly summarized. Recurring patterns identified: blind spots move, the challenge shifts, tradeoffs are unavoidable, proxies diverge under optimization pressure, scaling requires automation with its own blind spots.
- **Checkpoints:** Two predict-before-reveal checks -- (1) "What does MMLU actually measure?" (recognition vs generation, proxy gap), (2) "List three questions before trusting a SOTA claim" (contamination, selection bias, judge bias, proxy gap, comparability).

**Visual elements:**
- EvaluationStackDiagram: Inline SVG with six stacked layers from "Actual Model Capability" (bottom, indigo) through "Task Design," "Prompt Formatting," "Scoring Criteria," "Aggregation" to "Leaderboard Position" (top, red). Each layer has label, subtitle, and right-side note. Contamination annotation on right side (dashed red bracket). Bottom annotation: "The gap between bottom and top is the proxy gap."
- Two-model benchmark comparison cards (GradientCards: blue Model A, cyan Model B) followed by "What the Numbers Hide" reveal (rose GradientCard)
- Three contamination form cards (amber, orange, rose)
- Four LLM judge bias cards (amber, orange, 2x2 grid)
- Four "why evaluation is harder" cards (violet, purple, blue, rose)

**What is NOT covered:**
- Specific current benchmark scores or leaderboard positions
- Implementing evaluation pipelines in code
- Designing new benchmarks or evaluation frameworks
- Statistical methodology for evaluation (significance testing, confidence intervals)
- Evaluation of non-LLM models
- Constitutional AI, preference optimization, or red teaming details (Lessons 1-3)
- Full history of NLP benchmarks (GLUE, SuperGLUE) as chronological narrative

**Notebook:** `notebooks/5-1-4-evaluating-llms.ipynb` (3 exercises)
- Exercise 1 (Guided): Benchmark Autopsy -- given a model's scores on 5 benchmark categories, identify suspicious scores (contamination signals), distinguish recognition vs generation benchmarks, identify unmeasured quality dimensions. First two categories have guided questions, last three are unscaffolded. Insight: reading benchmark results critically is a skill, not just skepticism.
- Exercise 2 (Supported): LLM-as-Judge Bias Detection -- use an LLM API to judge response pairs where one is longer but less accurate and the other is concise but correct. Systematically vary length and confidence. Track correlation of judge ratings with length/confidence vs accuracy. Visualize bias. Insight: judge biases are measurable and predictable.
- Exercise 3 (Supported): Design an Evaluation -- given a specific use case (medical Q&A assistant), design a multi-method evaluation strategy integrating benchmarks, human judges, LLM judges, and red teaming. Produce a one-page evaluation plan. Insight: evaluation design requires the same tradeoff thinking as alignment technique selection.

**Review:** Passed at iteration 2/3. Iteration 1 had 3 improvement findings ("we just need better benchmarks" misconception implicit, "model passes benchmarks but fails red teaming" negative example underused, "more benchmarks = more coverage" misleading analogy not addressed) and 3 polish findings (spaced em dashes, Cohen's kappa ungrounded, Exercise 3 labeling). All improvement findings resolved. Iteration 2 had 0 improvement/critical findings and 2 minor polish findings (SVG text size on mobile, notebook Exercise 2 line count estimate).
