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
