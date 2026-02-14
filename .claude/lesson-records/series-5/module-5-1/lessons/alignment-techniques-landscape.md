# Lesson Plan: The Alignment Techniques Landscape

**Module:** 5.1 (Advanced Alignment), Lesson 2 of 4
**Slug:** `alignment-techniques-landscape`
**Status:** Planned

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| DPO as preference optimization without a separate reward model | INTRODUCED | rlhf-and-alignment (4.4.3) | Directly increases probability of preferred responses, decreases dispreferred. One model instead of two. Comparable results to PPO (Llama 2 used PPO; Zephyr, Mistral used DPO). Implicit KL penalty built into formulation. ComparisonRow: PPO pipeline (4 steps, two models) vs DPO pipeline (4 steps, one model). |
| PPO for language models (generate-score-update loop with KL penalty) | INTRODUCED | rlhf-and-alignment (4.4.3) | Three-step loop: generate, score with reward model, update policy. Two models involved (policy + reward model). "For the first time, the training loop changes shape." |
| KL penalty as soft constraint preventing reward hacking | INTRODUCED | rlhf-and-alignment (4.4.3) | Objective: maximize reward minus KL divergence from SFT model. Connected to catastrophic forgetting. "KL penalty is the continuous version of 'freeze the backbone'." |
| Reward hacking (exploiting imperfections in learned reward model) | INTRODUCED | rlhf-and-alignment (4.4.3) | Editor with blind spots. Excessive verbosity, confident filler, formatting tricks. Motivates KL penalty as essential. |
| Human preference data format (comparison pairs) | DEVELOPED | rlhf-and-alignment (4.4.3) | Prompt + two responses + which is better. Relative signal more reliable than absolute. InstructGPT ~33K comparisons. |
| Reward model architecture (pretrained LM + scalar head) | INTRODUCED | rlhf-and-alignment (4.4.3) | Same backbone + head pattern as classification finetuning. Outputs a single scalar quality score. |
| Constitutional AI principles as explicit alignment criteria | DEVELOPED | constitutional-ai (5.1.1) | Principles written down define "better." Made explicit and auditable. Used only during training to generate data. |
| RLAIF (AI-generated preference labels replacing human labels) | DEVELOPED | constitutional-ai (5.1.1) | AI model applies constitutional principles to generate preference labels. Same data format as RLHF, different source. Scales to millions of comparisons. |
| "Same pipeline, different data source" mental model | DEVELOPED | constitutional-ai (5.1.1) | CAI does NOT replace the RLHF pipeline -- it replaces WHERE the preference labels come from. The RL optimization is identical. |
| Human annotation bottleneck (cost, consistency, scale) | DEVELOPED | constitutional-ai (5.1.1) | Three problems motivating CAI. Sharpened with lock-picking annotator disagreement example. |

### Established Mental Models and Analogies

- **"The reward model is an experienced editor"** -- learned judgment from comparisons, has blind spots (from 4.4.3)
- **"The editor gets a style guide"** -- extends the editor analogy for constitutional AI; principles make criteria explicit (from 5.1.1)
- **"KL penalty is the continuous version of 'freeze the backbone'"** -- soft constraint on drift (from 4.4.3)
- **"Frozen backbone -> KL penalty -> LoRA" spectrum** -- three approaches to adapting without forgetting (from 4.4.4)
- **"Same pipeline, different data source"** -- CAI changes data source, not optimization mechanism (from 5.1.1)
- **"DPO partially restores the familiar training loop shape"** -- one model, closer to supervised learning (from 4.4.3)
- **"For the first time, the training loop changes shape"** -- PPO breaks the "same heartbeat" pattern (from 4.4.3)

### What Was Explicitly NOT Covered

- DPO's mathematical formulation (the Bradley-Terry model, the specific loss function)
- Any DPO variations (IPO, KTO, ORPO)
- Online vs offline preference optimization distinction
- Iterative alignment (using model's own generations for further training)
- Formal RL beyond minimum needed (policy = model behavior, reward = score)
- PPO algorithm details (clipping, value function, advantage estimation)
- Multi-objective alignment in depth

### Readiness Assessment

The student is well-prepared. DPO was INTRODUCED as "PPO but simpler, one model instead of two" with a ComparisonRow showing the pipeline difference. The student has a concrete mental model of what DPO does (directly adjusts probabilities of preferred vs dispreferred responses) but has NOT seen the mathematical formulation, which means we can discuss variations without needing to first teach the DPO loss function in detail. This is actually ideal for a "landscape" lesson -- the student has the intuition for what these methods do without being anchored to one specific formula. Constitutional AI from the previous lesson gives the student a fresh sense that the alignment design space is wider than PPO-vs-DPO, priming them for a broader mapping.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to **map the design space of preference optimization techniques along key axes (data requirements, reference model, reward model, online vs offline), understanding each variation as a different answer to the same question rather than a linear progression from worse to better.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| DPO as preference optimization without a separate reward model | INTRODUCED | INTRODUCED | rlhf-and-alignment (4.4.3) | OK | Lesson uses DPO as the anchor point for the design space. Student needs to know WHAT DPO does (directly adjusts probabilities, no reward model) but not the math. INTRODUCED is sufficient. |
| PPO for language models (generate-score-update loop) | INTRODUCED | INTRODUCED | rlhf-and-alignment (4.4.3) | OK | PPO is the "other end" of the design space -- the online, reward-model-based approach. Student needs the pipeline picture, not PPO internals. INTRODUCED is sufficient. |
| KL penalty as soft constraint | INTRODUCED | INTRODUCED | rlhf-and-alignment (4.4.3) | OK | Several variations modify or eliminate the KL penalty. Student needs to know its purpose (prevent drift from SFT model) to understand why removing it matters. INTRODUCED is sufficient. |
| Human preference data format (comparison pairs) | DEVELOPED | DEVELOPED | rlhf-and-alignment (4.4.3) | OK | Core data format that most variations assume. Student has worked through concrete preference pair examples. |
| Reward hacking | INTRODUCED | INTRODUCED | rlhf-and-alignment (4.4.3) | OK | Motivates why different methods exist -- each addresses different failure modes. INTRODUCED gives enough intuition. |
| RLAIF (AI-generated preference labels) | DEVELOPED | DEVELOPED | constitutional-ai (5.1.1) | OK | RLAIF is a data-source variation that the student already knows. This lesson can reference it as another axis of variation (who generates the labels). |
| Constitutional AI principles | DEVELOPED | DEVELOPED | constitutional-ai (5.1.1) | OK | Context from previous lesson. Not directly needed for this lesson's core concepts but establishes that alignment is a design space, not a single technique. |

All prerequisites are at sufficient depth. No gaps to resolve.

### Gap Resolution

No gaps. All prerequisites met.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **"Each new technique is strictly better than DPO"** | The lesson presents IPO, KTO, ORPO after DPO, which implies chronological/quality progression. The human instinct is "newer = better." | KTO requires no paired data but performs worse when high-quality paired data IS available. ORPO eliminates the reference model but is more sensitive to data quality. Each method optimizes for different constraints, not universally better performance. Show a concrete scenario where DPO outperforms a "newer" method. | During the design space mapping. After presenting the axes, explicitly frame as "tradeoffs, not upgrades." Reinforce in the Check section with a "which method for which situation" exercise. |
| **"Online is always better than offline"** | Online methods (like online DPO / iterative RLHF) use the model's own current generations, which sounds more principled -- "train on what you actually produce." The student may infer that offline methods are an inferior shortcut. | Online methods are significantly more expensive (need to generate during training), and for many practical cases the performance gap is small. Papers show diminishing returns from online training when offline preference data is high quality. The practical reality: most deployed models use offline methods because the cost/quality tradeoff favors it. | After explaining online vs offline. Use a concrete cost comparison (online = generate N responses per training step = N forward passes per batch). |
| **"Removing the reference model (as in ORPO) is always a simplification"** | Removing a component from a pipeline sounds like progress -- fewer parts = simpler. Student already saw DPO as "simpler than PPO because no reward model." They'll expect the same pattern: remove more = better. | The reference model in DPO/IPO serves as an anchor preventing catastrophic drift (it IS the KL penalty mechanism). Removing it means ORPO needs another mechanism to prevent the model from degenerating. Simplicity in one dimension creates complexity in another. | When introducing ORPO. Explicitly callback to KL penalty mental model: "remember why the KL penalty exists? What happens when you remove it?" |
| **"These techniques only differ in their loss functions"** | The student has DPO at INTRODUCED depth as "directly adjusts probabilities." If the variations are also "directly adjusting probabilities," the differences feel cosmetic -- just different math under the hood. | KTO changes the fundamental DATA REQUIREMENT (single responses, not pairs). This is not a loss function tweak -- it changes what data you can use. Similarly, online vs offline changes WHEN data is generated (before training vs during training). The design axes are orthogonal: data format, reference model, online/offline, reward model. | Structure the entire lesson around axes, not loss functions. Open with the axis framework before introducing any specific method. |
| **"Preference optimization is the only alignment approach"** | This lesson focuses on preference optimization variations, which could give the impression that alignment = preference optimization. The student may forget that constitutional AI (previous lesson) and SFT (Series 4) are also alignment-relevant. | Constitutional AI is alignment without preference pairs in the traditional sense (AI generates them). SFT alone produces some alignment (format teaches some norms). RLHF uses a reward model trained on preferences, not direct preference optimization. The landscape is broader than this lesson's focus area. | In the scope boundaries (Context + Constraints section). Explicitly state what this lesson covers and what it does not, positioning preference optimization as one region of the full alignment landscape. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **DPO as anchor: the quantum computing preference pair** | Positive (recap) | Ground the design space in something concrete the student has already seen. DPO takes this pair and directly optimizes. | Reuses the exact preference pair from the RLHF lesson (4.4.3), creating continuity. The student remembers this pair -- now they see it through the lens of "what if we changed the method?" |
| **KTO with a single thumbs-up/thumbs-down: user feedback on chatbot responses** | Positive | Show that KTO's innovation is the DATA FORMAT, not just the loss function. A single response rated good/bad, no pair needed. | This is the most intuitive departure from DPO. The student has internalized "preference = comparison pair." KTO breaks that assumption. Real-world connection: app users click thumbs-up/thumbs-down, not "compare these two responses." |
| **ORPO training a small model without any reference model** | Positive | Show ORPO's tradeoff: simpler pipeline but needs another mechanism for stability. | Demonstrates the "remove a component, gain and lose something" pattern. Extends the DPO-simplified-PPO pattern one step further. |
| **Choosing DPO for a well-resourced team with high-quality paired data** | Negative (for "newer = better" misconception) | Shows that the "oldest" variation (DPO) is the right choice in specific circumstances. Defeats the linear progression assumption. | Concrete decision scenario. The student must reason about constraints (data available, compute budget, team expertise) rather than just picking the newest method. |
| **Online DPO generating obviously bad responses early in training** | Negative | Shows that online methods have a cold-start problem: early in training, the model generates low-quality responses, and training on those can be counterproductive. | Grounds "online is always better" misconception. The abstract advantage of "train on what you produce" has concrete failure modes. |

### Design Space Axes (core organizing framework)

This lesson's central artifact is a mapping of the preference optimization design space along four axes:

| Axis | Question It Answers | Range |
|------|-------------------|-------|
| **Data format** | What does your training data look like? | Comparison pairs (DPO, IPO) -> Single responses (KTO) -> No preference data at all (ORPO uses odds ratio during SFT) |
| **Reference model** | Do you need a frozen copy of the SFT model? | Required (DPO, IPO, KTO) -> Not required (ORPO) |
| **Online vs Offline** | When are responses generated? | Offline (pre-collected, static dataset) -> Online (generated by current model during training) |
| **Reward model** | Is there a separate learned reward function? | Yes (PPO, iterative RLHF) -> No, implicit (DPO, IPO, KTO, ORPO) |

---

## Phase 3: Design

### Narrative Arc

Series 4 gave the student two options for alignment: PPO (complex, two models, reward hacking risk) and DPO (simpler, one model, comparable results). The student left that lesson with the impression that DPO was probably the practical choice and PPO the conceptual foundation. But "PPO or DPO?" is not the question researchers are actually asking. The real questions are: What if you do not have paired comparison data? What if you cannot afford to keep a frozen reference model in memory? What if you want the model to learn from its own mistakes, not just a static dataset? Each of these constraints leads to a different method -- not because the field is fragmented, but because different situations demand different tradeoffs. This lesson gives the student a map of that design space, so that when they encounter a new alignment technique (and they will -- the field produces new ones constantly), they can immediately locate it on the map: what data does it need, what models does it require, where does it sit on online-vs-offline? The map is more durable than any individual method.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual (design space map)** | A 2D or multi-axis diagram plotting PPO, DPO, IPO, KTO, ORPO along the four design axes (data format, reference model, online/offline, reward model). Each method is a point in this space. | The core concept IS a spatial relationship -- "these methods occupy different positions in a design space." A table lists methods; a diagram shows the space between them and reveals gaps. This is the central visual artifact of the lesson. |
| **Verbal/Analogy** | "Choosing an alignment technique is like choosing a vehicle for a trip. A car (DPO) works for most trips with good roads. A motorcycle (KTO) carries less (single responses, not pairs) but goes places the car cannot (user feedback data where you only have thumbs-up/down). A bus (PPO) carries the most but needs infrastructure (reward model, more compute). There is no 'best vehicle' -- only best for your constraints." | The "design space as choice among tradeoffs" concept maps well to tool selection. The student is a software engineer and understands choosing tools for constraints. Keeps the framing practical, not academic. |
| **Concrete example (worked comparison)** | Take the same alignment goal (a chatbot that gives helpful, harmless responses) and show how three different methods would approach it given three different data constraints: (1) paired comparisons available -> DPO, (2) only thumbs-up/down -> KTO, (3) want to fold alignment into SFT -> ORPO. | Grounds abstract design axes in a single concrete scenario. The student sees that the goal is identical -- only the constraints differ. This is the "of course" moment: different constraints naturally lead to different methods. |
| **Symbolic (comparison table)** | Side-by-side table comparing methods on axes: data format, reference model needed, reward model needed, online/offline, relative complexity, key insight. | Compresses the design space into a scannable reference. After the visual map and worked examples, the table serves as a summary artifact the student can return to. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 genuinely new concepts:
  1. The design space axes framework itself (organizing preference optimization along data format, reference model, online/offline, reward model)
  2. The online vs offline distinction (responses generated before training vs during training)
  3. The specific innovations of IPO/KTO/ORPO (counted as one concept because they are variations on a theme, not independent ideas -- each is a movement along one axis)
- **Previous lesson load:** STRETCH (constitutional-ai introduced the new paradigm of AI-supervised alignment)
- **This lesson's load:** BUILD -- appropriate. After a STRETCH lesson, a BUILD lesson that organizes and maps rather than introduces a new paradigm gives the student room to consolidate. The challenge is breadth (many methods) not depth (one hard concept). The design space framework gives the breadth structure.

### Connections to Prior Concepts

- **DPO from Series 4.4.3:** The anchor point for the entire design space. Every method is positioned relative to DPO (what does it change? what does it keep?). The ComparisonRow from the RLHF lesson (PPO vs DPO pipeline) extends to a multi-method comparison.
- **PPO from Series 4.4.3:** The "full pipeline" endpoint -- reward model, online generation, explicit KL penalty. PPO is on the design space map as the high-infrastructure, high-control approach.
- **KL penalty from Series 4.4.3:** The "freeze the backbone" analogy. Central to understanding what happens when methods remove the reference model (ORPO) or modify the KL constraint (IPO). "Remember why the KL penalty exists?" is a key callback.
- **RLAIF from 5.1.1:** Another axis of variation (who generates the preference data). Constitutional AI changed the data SOURCE; this lesson changes the data FORMAT and optimization MECHANISM. They are complementary, not competing, dimensions.
- **"Same pipeline, different data source" from 5.1.1:** Extends naturally. Constitutional AI was "same pipeline, different source." KTO is "same goal, different data format." ORPO is "same goal, folded into SFT." The pattern: alignment innovations often modify one component while holding others fixed.

**Potentially misleading analogies:** The "DPO is simpler than PPO" framing from Series 4 could lead to "simpler is always better." This lesson needs to explicitly break that assumption -- simpler methods have fewer knobs but also fewer ways to control behavior.

### Scope Boundaries

**This lesson IS about:**
- Mapping the preference optimization design space along concrete axes
- Understanding what each major variation (IPO, KTO, ORPO) changes relative to DPO and why
- The online vs offline distinction and its practical implications
- Iterative alignment (using model outputs to generate new preference data for the next round)
- Developing the "map" mental model so the student can locate any future technique

**This lesson is NOT about:**
- Mathematical loss function derivations (no Bradley-Terry model, no IPO/KTO/ORPO loss equations beyond high-level intuition)
- Implementing any of these techniques in code
- Benchmarking or comparing performance numbers across methods (too ephemeral, and the student cannot verify)
- Constitutional AI (previous lesson) or red teaming (next lesson)
- Reward modeling in depth (covered conceptually in Series 4)
- RL formalism (policy gradient, advantage estimation, etc.)
- Specific company choices ("Anthropic uses X, OpenAI uses Y")

**Target depths:**
- Design space axes framework: DEVELOPED (student can use the axes to classify a method they haven't seen)
- IPO (Identity Preference Optimization): INTRODUCED (bounded preferences, addresses DPO overfitting, keeps reference model)
- KTO (Kahneman-Tversky Optimization): INTRODUCED (single-response signal instead of pairs, loss-aversion-inspired, keeps reference model)
- ORPO (Odds Ratio Preference Optimization): INTRODUCED (no reference model, folds alignment into SFT, uses odds ratio)
- Online vs offline preference optimization: INTRODUCED (distinction is clear, tradeoffs understood, not practiced)
- Iterative alignment / self-play: INTRODUCED (concept of using model's own outputs for next training round, motivation clear)

### Lesson Outline

#### 1. Context + Constraints
What this lesson covers (the preference optimization design space beyond PPO-vs-DPO) and what it does not (loss function math, implementation, benchmarks). Position this as one region of the full alignment landscape -- constitutional AI (previous lesson) was another region, red teaming (next lesson) will be another. Explicit scope framing to prevent the "alignment = preference optimization" misconception.

#### 2. Recap
Brief recap of DPO and PPO as the two endpoints the student knows. Not a re-teach -- a re-activation. Use the exact framing from the RLHF lesson: PPO = generate-score-update with two models; DPO = directly adjust probabilities with one model. State the mental models: "editor" (PPO has one), "one model instead of two" (DPO). Set up the question: "What if even this binary is too narrow?"

#### 3. Hook (Before/After + Puzzle)
Present three real-world alignment scenarios that the PPO/DPO binary cannot cleanly solve:
- **Scenario A:** You have a chatbot app. Users click thumbs-up or thumbs-down on individual responses. You do NOT have paired comparisons. Neither PPO nor DPO can directly use this data.
- **Scenario B:** You want to align a model but cannot afford to keep a frozen reference model in GPU memory alongside the training model (you are memory-constrained).
- **Scenario C:** You want the model to learn from its own current mistakes, not from a static dataset that was generated by an earlier version of the model.

Ask: "Which of the two methods you know (PPO, DPO) solves each scenario?" Answer: neither cleanly does. This creates the need for the design space map.

#### 4. Explain: The Design Space Axes
Introduce the four axes as the core framework:
1. **Data format:** Paired comparisons vs single responses vs no preference data
2. **Reference model:** Required vs not required
3. **Online vs Offline:** Static pre-collected data vs generated during training
4. **Reward model:** Separate learned model vs implicit in the loss

Place PPO and DPO on this map first (the student already knows these). PPO: paired data, reference model (for KL), online, explicit reward model. DPO: paired data, reference model (implicit in loss), offline, no reward model. The student sees that PPO and DPO already differ on multiple axes.

**Design space diagram** (central visual artifact): Multi-axis comparison with PPO and DPO placed, and empty positions where IPO/KTO/ORPO will go.

#### 5. Check 1: Predict
Before introducing any new method, ask the student: "Look at the empty positions on the map. What would a method look like that uses single responses instead of pairs? What would you gain? What would you lose?" This activates prediction-before-explanation.

#### 6. Explain: The Variations
Introduce each variation as a movement along one or more axes:

**IPO (Identity Preference Optimization):**
- What it changes: Addresses DPO's tendency to overfit on preference strength. DPO treats "slightly preferred" and "overwhelmingly preferred" the same way. IPO bounds the preference signal -- it says "this response is better, but only by this much."
- Position on map: Same as DPO on most axes (paired data, reference model, offline, no reward model). Differs on: bounded vs unbounded preference signal.
- Connection: "If DPO is the editor saying 'this draft is better,' IPO is the editor saying 'this draft is slightly better.' The calibration matters."

**KTO (Kahneman-Tversky Optimization):**
- What it changes: Does not require paired comparisons at all. Works with single responses labeled good or bad (thumbs-up/thumbs-down). Inspired by prospect theory -- humans are more sensitive to losses than gains.
- Position on map: Moves along the data format axis. Single responses instead of pairs. Still needs reference model.
- Connection: Solves Scenario A from the hook. "Of course" moment: most real user feedback IS thumbs-up/thumbs-down, not paired comparisons. KTO fits the data you actually have.
- Loss aversion insight: The asymmetry (losses weighted more than gains) mirrors human psychology. The model is penalized more for generating a bad response than it is rewarded for generating a good one.

**ORPO (Odds Ratio Preference Optimization):**
- What it changes: Eliminates the reference model entirely. Folds preference optimization into the SFT training process using an odds ratio penalty.
- Position on map: Moves along the reference model axis. No reference model needed. Paired data still required.
- Connection: Solves Scenario B from the hook. But callback to KL penalty: "Remember why the reference model exists -- it prevents catastrophic drift. ORPO must handle this differently." Addresses the "simplification = free" misconception.

#### 7. Explain: Online vs Offline
Dedicated section for the online/offline axis because it is orthogonal to the method choice:
- **Offline:** Train on a pre-collected static dataset of preferences. This is what DPO does by default. Cheaper, simpler, but the data was generated by an older model version.
- **Online:** Generate responses with the current model during training, get preference labels (from humans or AI), and train on those. This is what PPO does. More expensive but trains on the distribution the model actually produces.
- **Online DPO:** Apply DPO's optimization but generate fresh data during training. Combines DPO's simplicity with online learning's distribution match.
- **Iterative alignment:** Multiple rounds of generate -> label -> train. Each round produces a better model whose outputs are used for the next round. Connection to RLAIF from constitutional AI -- AI can label the preferences in each round, making iteration practical.
- Negative example: Online methods early in training generate low-quality responses. Training on the model's own bad outputs can reinforce bad behavior before it improves (cold-start problem).

#### 8. Check 2: Design Space Navigation
Present a new method the student has NOT seen (e.g., a hypothetical or a recent paper's approach described in plain language) and ask them to place it on the design space map. Which axes does it change? What tradeoffs does it make? This tests transfer -- can the student use the framework, not just recall the specific methods?

#### 9. Elaborate: Choosing a Method (The Map as Decision Tool)
Return to the three hook scenarios and solve each one using the design space map:
- Scenario A (thumbs-up/down data) -> KTO
- Scenario B (memory-constrained, no reference model) -> ORPO
- Scenario C (learn from own mistakes) -> Online DPO or iterative alignment

Then add a fourth scenario: "You have a well-funded team with high-quality paired comparisons from expert annotators." Answer: DPO. The "oldest" method is the right choice. This directly defeats the "newer = better" misconception.

Address the module-level misconception: "More alignment techniques != more aligned models." The landscape exists because different teams face different constraints. The field is not converging on one method; it is mapping a design space.

#### 10. Practice (Notebook -- lightweight)
A lightweight Colab notebook with 3 exercises demonstrating alignment concepts on small-scale proxies:

- **Exercise 1 (Guided): Preference Data Formats.** Given a set of model responses and human ratings, convert the same feedback into three formats: (a) paired comparisons (for DPO), (b) single good/bad labels (for KTO), (c) SFT data with quality weighting (for ORPO-style). See how data availability determines which methods are viable. Insight: the data format axis is the most practical constraint in real alignment work.

- **Exercise 2 (Supported): Reference Model Drift.** Using a small model (GPT-2 scale), compute the KL divergence between a fine-tuned model and its reference (the pre-SFT checkpoint) at different training steps. Visualize how the model drifts from the reference over training. Then remove the reference constraint and observe what happens (faster learning but potential degeneration). Insight: the reference model is not just a mathematical convenience -- it is a stability mechanism.

- **Exercise 3 (Supported): Online vs Offline Distribution Mismatch.** Generate responses from a base model, label them, and fine-tune. Then generate from the fine-tuned model and compare the response distribution to the original training data. Observe the distribution shift -- the model is now being "tested" on a different distribution than it was "trained" on. Insight: this is why online methods exist -- they close the train/test gap.

Exercises are independent (can be done in any order). Each demonstrates one axis of the design space. Solutions should emphasize the design tradeoff, not just the code.

#### 11. Summarize
Key takeaways:
- The PPO/DPO binary from Series 4 was a starting point, not the full picture
- The design space has four key axes: data format, reference model, online vs offline, reward model
- Each method makes different tradeoffs -- there is no universally "best" technique
- The map is more durable than any specific method. When you encounter a new technique, ask: where does it sit on these axes?

Echo the central mental model: "Alignment techniques are points in a design space, not steps on a ladder."

#### 12. Next Step
Preview Lesson 3 (Red Teaming & Adversarial Evaluation): "Lessons 1 and 2 built alignment techniques. Now: how do you find what they missed? Red teaming asks 'how would you break this?' -- the adversarial complement to everything we have built." Connects to the module arc: Build it (Lessons 1-2) -> Break it (Lesson 3) -> Measure it (Lesson 4).

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
- [x] At least 3 modalities planned for the core concept, each with rationale (4 modalities: visual diagram, verbal/analogy, concrete example, symbolic table)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load ≤ 3 new concepts (2-3 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings -- the student would not be lost or form wrong mental models. However, four improvement findings identify places where the lesson is significantly weaker than it could be. Another pass is needed after addressing these.

### Findings

#### [IMPROVEMENT] — Planned "quantum computing preference pair" recap example missing from lesson

**Location:** Section 2 (Recap: "Where We Left Off")
**Issue:** The planning document specifies a positive example: "DPO as anchor: the quantum computing preference pair" -- reusing the exact preference pair from the RLHF lesson (4.4.3) to create continuity. The built lesson has a ComparisonRow listing PPO and DPO properties in the recap but does NOT include the concrete preference pair. The quantum computing pair does appear in the notebook (Exercise 1), but not in the lesson component where the student first encounters the recap.
**Student impact:** The recap is abstract -- bullet-point properties of PPO and DPO. Without a concrete example grounding these properties, the recap is a list of facts rather than a re-activation of the student's existing mental model. The student remembers the quantum computing preference pair from the RLHF lesson; seeing it again here would immediately re-activate their understanding. Without it, the transition from "here's what you know" to "here's the design space" is less visceral.
**Suggested fix:** Add a brief concrete callback to the quantum computing preference pair before or after the ComparisonRow in the recap section. Something like: "Remember the quantum computing preference pair? Response A gave a specific, age-appropriate explanation; Response B was vague but not wrong. DPO took that pair and directly adjusted probabilities -- no reward model needed. PPO would have scored each response with a reward model first. Same data, different mechanism." This grounds the bullet-point comparison in the student's actual memory.

#### [IMPROVEMENT] — Symbolic modality (comparison table) planned but not built

**Location:** Overall lesson structure
**Issue:** The planning document specifies four modalities, one of which is "Symbolic (comparison table) -- Side-by-side table comparing methods on axes: data format, reference model needed, reward model needed, online/offline, relative complexity, key insight." The built lesson does NOT include this table. The design space diagram (SVG) serves the visual modality, and the GradientCards for each method serve verbal/analogy and concrete example modalities. But the scannable reference table that compresses the entire design space into a single artifact is absent.
**Student impact:** After reading through the individual method cards (IPO, KTO, ORPO) and the online/offline section, the student has seen each method presented sequentially. But there is no single artifact where all five methods are compared side-by-side on all four axes. The SVG diagram shows positions but not the textual descriptions of each position. The student cannot easily cross-reference "which methods need a reference model?" without re-scanning multiple sections. The table would serve as both a summary and a reference artifact.
**Suggested fix:** Add a comparison table (using a styled HTML table or a series of structured elements) after the design space diagram section and before the online/offline deep-dive. The table should have rows for each method (PPO, DPO, IPO, KTO, ORPO) and columns for each axis (Data Format, Reference Model, Online/Offline, Reward Model) plus a "Key Insight" column. This completes the planned symbolic modality and gives the student a scannable reference.

#### [IMPROVEMENT] — Check 1 predict-before-explain positioned after axes but misses the progressive reveal opportunity

**Location:** Section 6 (Check 1: Predict), in relation to the design space diagram placement
**Issue:** The planning document says to present the design space diagram with PPO and DPO placed and "empty positions where IPO/KTO/ORPO will go," then ask the student to predict. The built lesson introduces the four axes, places PPO and DPO, asks the prediction question, then introduces all three variations, and THEN shows the complete design space diagram with all five methods already plotted. The student never sees the diagram with gaps -- they only see the completed version. The progressive reveal (axes first, PPO/DPO placed, gaps visible, then fill them in) is lost.
**Student impact:** The design space diagram arrives after all methods have been explained. The student sees it as a summary, not as a discovery tool. The planned experience was: "Here's the map with two points. Where do the new ones go?" -- which creates active engagement. The built experience is: "Here's all the methods. Here's a diagram showing what you just read." -- which is passive confirmation. The diagram should drive curiosity, not just confirm knowledge.
**Suggested fix:** Either (a) show a simplified version of the diagram with only PPO and DPO before the variations section, with visible empty space, and then show the complete diagram after; or (b) restructure the SVG component to accept a `methods` prop so it can be rendered twice -- once sparse, once complete. Option (a) is simpler and accomplishes the pedagogical goal.

#### [IMPROVEMENT] — Fifth misconception ("preference optimization is the only alignment approach") not explicitly addressed

**Location:** Context + Constraints section
**Issue:** The planning document identifies five misconceptions. Four are clearly addressed in the built lesson: (1) "newer = better" is addressed by the DPO Scenario D card, (2) "online always better" is addressed by the cold-start card, (3) "removing reference model is always simplification" is addressed in the ORPO section and aside, (4) "techniques only differ in loss functions" is addressed by the axes-first structure. But misconception 5 -- "preference optimization is the only alignment approach" -- is only implicitly addressed. The scope section lists what the lesson does NOT cover, and the ObjectiveBlock mentions "the full design space of preference optimization," but there is no explicit statement that preference optimization is one region of a broader alignment landscape that also includes SFT norms, constitutional AI principles, and other approaches.
**Student impact:** After two lessons focused on alignment techniques (constitutional AI, then this landscape), the student may unconsciously equate "alignment = preference optimization." The scope boundaries list exclusions but do not explicitly state "preference optimization is one approach among several." The student might not form this misconception strongly, but the plan identified it as worth addressing, and the built lesson does not.
**Suggested fix:** Add one sentence to the ObjectiveBlock or the opening of the lesson that explicitly positions preference optimization as one region of the broader alignment landscape. Something like: "Preference optimization -- adjusting a model's behavior using human or AI feedback on its outputs -- is the region of alignment we focus on here. Constitutional AI (previous lesson) showed that the data source can change; SFT and prompting are other regions entirely. This lesson maps the preference optimization region specifically."

#### [POLISH] — Design space SVG uses spaced em dashes in subtitle text

**Location:** DesignSpaceMap SVG component, subtitle text (line 113)
**Issue:** The SVG subtitle reads "Each method occupies a different position -- tradeoffs, not upgrades" using a plain hyphen-hyphen, which renders as two hyphens rather than an em dash. This is inconsistent with the rest of the lesson which correctly uses `&mdash;` for em dashes.
**Student impact:** Minor visual inconsistency. The student would not be confused, but it breaks the typographic consistency of the lesson.
**Suggested fix:** Change the SVG text to use a Unicode em dash character directly (since SVG text does not support HTML entities): `Each method occupies a different position\u2014tradeoffs, not upgrades` or simply use the literal `—` character.

#### [POLISH] — Notebook Exercise 2 solution mentions direction convention but the exercise itself does not specify direction

**Location:** Notebook Exercise 2 (Reference Model Drift), cell-10 and cell-12
**Issue:** The TODO comment in cell-10 says "Compute the per-token log-probability ratio: policy - reference" and the solution mentions "Common mistake: Computing ref_log_probs - policy_log_probs (reversed)." But the hint in cell-8 already states the convention: "ratio = log_prob_policy(token) - log_prob_reference(token)." The direction is specified but scattered across cells. This is a minor clarity issue, not a pedagogical problem.
**Student impact:** Minimal. The TODO comment is clear enough. The solution's "common mistake" callout is helpful.
**Suggested fix:** No change needed. This is a very minor observation.

#### [POLISH] — Notebook Exercise 3 solution KL direction note may confuse

**Location:** Notebook Exercise 3 solution (cell-19)
**Issue:** The solution says "Common mistake: Computing KL(base || updated) instead of KL(updated || base). The direction matters: we want to measure how surprising the base data is to the updated model, which is KL(updated || base)." However, KL(updated || base) actually measures how surprising the base distribution is when viewed from the updated distribution, and the TODO code computes `sum(updated_probs * log(updated_probs / base_probs))` which IS KL(updated || base). The explanation is correct but the phrasing "how surprising the base data is to the updated model" could be clearer -- it is the expected surprise when using the base distribution to approximate the updated distribution.
**Student impact:** Minor. A student who already understands KL divergence direction would not be confused; a student who does not would not parse the difference. The code is correct regardless.
**Suggested fix:** Simplify the KL direction note to: "Common mistake: Computing KL(base || updated) instead of KL(updated || base). We want to measure how much the updated policy has diverged from the base -- the code computes this correctly."

### Review Notes

**What works well:**
- The axes-first, methods-second structure is excellent pedagogy. It gives the student a framework before populating it, which is exactly the "parts before whole" ordering rule.
- The three hook scenarios are concrete, memorable, and each maps to a different method later. The payoff in Section 9 ("Choosing a Method") closes the loop satisfyingly.
- The "tradeoffs, not upgrades" framing is consistent throughout and effectively prevents the "newer = better" misconception.
- The notebook exercises are well-designed -- each targets a different axis and uses small-scale proxies to make abstract concepts concrete. The scaffolding progression (Guided, Supported, Supported) is appropriate.
- The aside content is strong. The WarningBlocks and InsightBlocks add genuine value rather than repeating what the main content says.
- The vehicle analogy (car/motorcycle/bus) is effective and connects to the student's software engineering background of choosing tools for constraints.
- The cold-start problem card is a strong negative example that genuinely defeats the "online is always better" misconception.

**Patterns to watch:**
- The lesson runs long -- 12 sections plus notebook. For a BUILD lesson this is acceptable (the content is breadth, not depth) but the pacing should be checked after revisions to ensure it does not sag.
- The SVG diagram is a static image. An interactive version (click a method to highlight its positions across all four axes) would be significantly more effective but may not be worth the implementation cost for this lesson's scope. Note for future consideration, not a finding.

---

## Review — 2026-02-14 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 3

### Verdict: PASS

All four improvement findings from iteration 1 have been properly addressed. The lesson now includes: (1) the quantum computing preference pair callback in the recap, (2) a full comparison table (symbolic modality), (3) a sparse-then-complete progressive reveal of the design space diagram, and (4) an explicit statement positioning preference optimization as one region of a broader alignment landscape. No new critical or improvement-level issues found. Three minor polish items remain, none of which affect the student's learning experience.

### Findings

#### [POLISH] — SVG subtitle text still uses spaced em dashes

**Location:** SparseDesignSpaceMap line 114 and DesignSpaceMap line 330
**Issue:** The SVG subtitle text reads "Two methods placed — where do the gaps lead?" (line 114) and "Each method occupies a different position — tradeoffs, not upgrades" (line 330). Both use spaced em dashes (" — "). The writing style rule requires no spaces around em dashes. This was partially flagged in iteration 1 (line 330 only); line 114 has the same issue.
**Student impact:** Negligible. Minor typographic inconsistency within SVG text that the student is unlikely to notice.
**Suggested fix:** Change to unspaced em dashes: "Two methods placed—where do the gaps lead?" and "Each method occupies a different position—tradeoffs, not upgrades". Since these are in SVG `<text>` elements, use the literal Unicode em dash character (—) directly.

#### [POLISH] — GradientCard titles for IPO/KTO/ORPO use spaced em dashes

**Location:** Lines 890, 929, 976 (GradientCard title props)
**Issue:** The three method card titles use spaced em dashes as separators: "IPO — Identity Preference Optimization", "KTO — Kahneman-Tversky Optimization", "ORPO — Odds Ratio Preference Optimization". The writing style rule says em dashes must have no spaces. However, this is a title/heading context where spaced em dashes serve as visual separators between an acronym and its expansion, which is a common typographic convention.
**Student impact:** Negligible. The spaced format is arguably more readable in a title context. This is a style consistency question, not a clarity issue.
**Suggested fix:** Change to unspaced em dashes for consistency: "IPO—Identity Preference Optimization", etc. Alternatively, use a colon or en dash if the unspaced em dash feels too cramped in a title: "IPO: Identity Preference Optimization". Either would resolve the inconsistency.

#### [POLISH] — Notebook Exercise 3 KL direction explanation still uses slightly confusing phrasing

**Location:** Notebook cell-19 (Exercise 3 solution)
**Issue:** Carried over from iteration 1. The solution says "how surprising the base data is to the updated model, which is KL(updated || base)." The phrasing is technically imprecise -- KL(updated || base) measures the expected extra information cost when using the base distribution to approximate the updated distribution. The code is correct; only the English explanation of what the KL direction means is slightly unclear.
**Student impact:** Minimal. Students who understand KL divergence direction will not be confused; students who do not will not parse the difference. The code computes the right thing.
**Suggested fix:** Simplify to: "Common mistake: Computing KL(base || updated) instead of KL(updated || base). We want to measure how much the updated policy has diverged from the base -- the code computes this correctly."

### Iteration 1 Findings Resolution Check

| Iteration 1 Finding | Severity | Status | How Resolved |
|---------------------|----------|--------|-------------|
| Quantum computing preference pair missing from recap | IMPROVEMENT | FIXED | Added concrete callback at lines 614-621, reusing the exact preference pair from the RLHF lesson with PPO vs DPO mechanism comparison. |
| Symbolic modality (comparison table) not built | IMPROVEMENT | FIXED | Full HTML table added at lines 1040-1108 with all five methods across all four axes plus "Key Insight" column. |
| Progressive reveal of design space diagram missing | IMPROVEMENT | FIXED | Sparse diagram (SparseDesignSpaceMap) with PPO/DPO and dashed "?" gaps shown before variations; complete diagram (DesignSpaceMap) shown after all methods explained. Two separate SVG components. |
| Fifth misconception not explicitly addressed | IMPROVEMENT | FIXED | "The Landscape Keeps Growing" card (lines 1362-1365) explicitly states "preference optimization is one approach to alignment among several" and names Constitutional AI and red teaming as other regions. |
| SVG spaced em dashes | POLISH | PARTIALLY FIXED | Line 330 still has spaced em dash; line 114 also has the same issue (not flagged in iteration 1). Carried forward as Polish. |
| Notebook Exercise 2 direction convention | POLISH | N/A | Iteration 1 said no change needed. Confirmed. |
| Notebook Exercise 3 KL direction note | POLISH | NOT FIXED | Phrasing unchanged. Carried forward as Polish with same suggested fix. |

### Review Notes

**What works well (reinforced from iteration 1):**
- The progressive reveal of the design space diagram is now excellent. The sparse version with dashed "?" markers creates genuine curiosity, and the complete version after the variations provides a satisfying payoff. This is significantly better than the iteration 1 version where the student only saw the completed diagram.
- The comparison table completes the symbolic modality and serves as an effective scannable reference. The "Key Insight" column is a good addition that gives each method a one-line identity.
- The quantum computing preference pair in the recap section effectively re-activates the student's memory from the RLHF lesson. It transforms the recap from an abstract list of properties into a concrete reconnection.
- The fifth misconception is now addressed naturally in the "Landscape Keeps Growing" card, which reads as a concluding reflection rather than a forced disclaimer.
- All four modalities (visual, verbal/analogy, symbolic, concrete example) are now present and working together. The lesson is modality-complete.

**Overall assessment:**
The lesson is pedagogically sound. It teaches the design space framework effectively, connects every new concept to prior knowledge, addresses all five planned misconceptions, and maintains a clear narrative arc from problem (PPO/DPO binary is too narrow) through framework (four axes) to application (choose a method for your constraints). The pacing is appropriate for a BUILD lesson. The notebook exercises are well-designed and target different axes. The three remaining Polish findings are minor typographic issues that do not affect the student's learning experience. The lesson is ready to ship.
