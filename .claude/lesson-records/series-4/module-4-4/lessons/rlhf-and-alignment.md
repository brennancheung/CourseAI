# Lesson: RLHF & Alignment

**Module:** 4.4 (Beyond Pretraining)
**Position:** Lesson 3 of 5
**Type:** Conceptual (no notebook)
**Cognitive load:** BUILD

---

## Phase 1: Orient -- Student State

The student arrives from Lesson 2 (instruction-tuning), a STRETCH lesson where they learned SFT mechanics, implemented loss masking, ran SFT on a small instruction dataset, and established the central insight "SFT teaches format, not knowledge." They have a deep understanding of the complete pipeline from pretraining through classification finetuning through SFT, and can articulate the structural differences between classification finetuning (new head, narrow task) and SFT (same head, broad task, instruction-response data). They have the SftPipelineDiagram fresh in memory, which previewed RLHF as the grayed-out "NEXT LESSON" third stage. The Lesson 2 "Next Step" section explicitly set up this lesson: "SFT produces a model that follows instructions. But following instructions is not enough. The model can be helpful but also harmful, verbose, sycophantic, or confidently wrong."

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| SFT teaches format, not knowledge (base model already has knowledge; SFT teaches instruction-following format) | DEVELOPED | instruction-tuning (4.4.2) | Central insight from previous lesson. Expert-in-monologue analogy. Dual-prompt "capital of France" evidence. |
| SFT training mechanics (same cross-entropy loss, same training loop, different data) | APPLIED | instruction-tuning (4.4.2) | Student ran SFT in notebook. Side-by-side loop comparison proves identical mechanics. |
| Instruction dataset format (instruction/response pairs, diverse task types) | DEVELOPED | instruction-tuning (4.4.2) | Multiple concrete examples. Alpaca 52K, LIMA 1K as size references. |
| Chat templates and special tokens (<\|im_start\|>, <\|im_end\|>, role markers) | DEVELOPED | instruction-tuning (4.4.2) | Functional structure, not cosmetic. Wrong-template negative example proved this concretely. |
| Loss masking / prompt masking (label = -100 for instruction tokens) | DEVELOPED | instruction-tuning (4.4.2) | Color-coded LossMaskingDiagram. PyTorch ignore_index=-100. |
| Classification finetuning vs SFT structural distinction | DEVELOPED | instruction-tuning (4.4.2) | ComparisonRow: new head vs same head, narrow task vs broad task. "Feature extractor" mental model does NOT apply to SFT. |
| Classification finetuning (add head, freeze backbone, train) | DEVELOPED | finetuning-for-classification (4.4.1) | CNN transfer learning callback. GPT2ForClassification. |
| Complete GPT training loop (forward, loss, backward, step) | APPLIED | pretraining (4.3.2) | Written from scratch. "Same heartbeat, new instruments." Extended to SFT in 4.4.2. |
| Cross-entropy loss for next-token prediction | DEVELOPED | pretraining (4.3.2) | Reshape (B, T, V) to (B*T, V). Same formula regardless of vocabulary size. |
| Language modeling as next-token prediction | DEVELOPED | what-is-a-language-model (4.1.1) | The defining objective. Self-supervised labels. |
| Catastrophic forgetting | INTRODUCED | finetuning-for-classification (4.4.1) | Frozen backbone preserves abilities. Full finetuning risks destroying general capabilities. |
| Scaling laws (Chinchilla compute-optimal training) | INTRODUCED | scaling-and-efficiency (4.3.3) | N_opt ~ sqrt(C), D_opt ~ sqrt(C). Predictability of training outcomes. |

**Mental models already established:**
- "SFT teaches format, not knowledge" (instruction-tuning)
- "Same heartbeat, new instruments" / "Same heartbeat, third time" (training loop structure, extended through pretraining, classification, SFT)
- "A pretrained transformer is a text feature extractor" (classification finetuning -- NOTE: does not apply to SFT or RLHF)
- "The architecture is the vessel; the weights are the knowledge" (loading-real-weights)
- "Cross-entropy doesn't care about vocabulary size" (pretraining)

**What was explicitly NOT covered that is relevant here:**
- RLHF, DPO, or any preference-based alignment (explicitly deferred from Lessons 1 and 2)
- Reward models or reward signals
- The concept of "alignment" -- what makes a model helpful, harmless, honest
- Reinforcement learning of any kind (the student has never encountered RL in this course)
- Why SFT alone produces models that can be harmful, sycophantic, or confidently wrong
- Human preference data (the SftPipelineDiagram previewed "Human preferences" as RLHF's data source, grayed out)

**Readiness assessment:** The student is well-prepared. They understand what SFT produces (an instruction-following model) and the Lesson 2 "Next Step" section planted the specific question this lesson answers: "The model follows the FORMAT of being helpful, but has no training signal for what 'helpful' actually means." The conceptual foundation is solid. This lesson introduces new concepts (reward models, PPO intuition, DPO) but at an intuitive level -- no implementation, no math beyond what is needed for understanding. After a STRETCH lesson, BUILD is appropriate. The challenge is that reinforcement learning is entirely new territory for this student, so PPO must be kept at an intuitive level and not over-formalized.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to explain why SFT alone is insufficient for producing reliably helpful models, how RLHF uses human preference data to train a reward model and then optimize the language model's policy via PPO, and how DPO achieves a similar result without a separate reward model.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| SFT teaches format, not knowledge | DEVELOPED | DEVELOPED | 4.4.2 | OK | The student must deeply understand what SFT produces to understand why it is not enough. This is the foundation the entire lesson builds on. |
| SFT training mechanics (same loop, same loss, different data) | INTRODUCED | APPLIED | 4.4.2 | OK | Student needs to understand HOW SFT works to understand what RLHF adds on top. Student exceeds requirement. |
| Instruction dataset format | INTRODUCED | DEVELOPED | 4.4.2 | OK | Student needs to understand instruction/response pairs to understand why paired preference data is different. Student exceeds requirement. |
| Cross-entropy loss | INTRODUCED | DEVELOPED | 4.3.2 | OK | Student needs to know the standard loss to understand that RLHF introduces a different kind of training signal. Student exceeds requirement. |
| Language modeling as next-token prediction | INTRODUCED | DEVELOPED | 4.1.1 | OK | Student needs the base framing to understand that RLHF changes the optimization objective from "predict the next token" to "generate text humans prefer." |
| Loss masking | INTRODUCED | DEVELOPED | 4.4.2 | OK | Needed to understand DPO's implicit reward model formulation, where the log-probability of preferred vs dispreferred completions is the key quantity. |
| Gradient descent / training loop | INTRODUCED | APPLIED | 4.3.2 | OK | Student needs to understand optimization to grasp PPO at an intuitive level. |

All prerequisites OK. No gaps to resolve.

**Note on reinforcement learning:** The student has never encountered RL concepts (policy, reward, value function, etc.) in this course. This is not a gap because this lesson teaches RLHF at an intuitive level -- the student does not need prior RL knowledge. The lesson introduces RL concepts only as needed and only at INTRODUCED depth: "policy" = "the language model's behavior," "reward" = "a score for how good a response is." The goal is intuition, not RL formalism.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "RLHF teaches the model what is true" | The student knows SFT teaches format. Natural inference: the next training stage must teach content/truth. Also, aligned models are often more factually accurate, which reinforces this assumption. | Show a case where RLHF optimizes for preference over truth: a model that gives a confident, well-structured but wrong answer that humans rate higher than a hedging, uncertain but correct answer. RLHF optimizes for what humans PREFER, which correlates with truth but is not identical to it. Sycophancy: "You're right!" is often preferred over "Actually, you're wrong because..." | Core concept section, after introducing the reward model. Frame: RLHF teaches what humans prefer. Preferences correlate with truth, helpfulness, safety -- but they are not the same thing. |
| "RLHF completely replaces SFT -- you go from pretraining directly to RLHF" | The student might think RLHF is an alternative to SFT, not a complement. The pipeline diagram from Lesson 2 shows three stages sequentially, but the student might interpret "RLHF comes after SFT" as "RLHF replaces SFT." | Without SFT first, the base model produces document-continuation text. RLHF cannot meaningfully compare two document continuations for "helpfulness" -- neither is trying to be helpful. You need SFT to get the model into "instruction-following mode" before RLHF can refine the quality of that instruction following. SFT is necessary scaffolding. | Early in the lesson, when establishing the pipeline. The SftPipelineDiagram callback reinforces the sequential nature. |
| "The reward model is a simple rule-based system or a set of human-written rules" | "Reward model" sounds like a scoring rubric. The student might imagine a set of handcrafted rules (e.g., "deduct points for profanity, add points for complete sentences"). This is reinforced by the word "reward" suggesting a simple scoring function. | The reward model is itself a neural network (often a language model) trained on human preference data. It learns to predict which of two responses a human would prefer. It handles nuance, context, and tradeoffs that no set of rules could capture. A rule-based system cannot judge whether an explanation is "clearer" or a joke is "funnier." | When introducing the reward model. Show that it is a model trained on data, not a list of rules. |
| "RLHF requires millions of human preference labels" | The student just learned SFT is data-efficient (1K-52K examples). They might swing to the opposite assumption: RLHF must need enormous human effort to work. Or they might think each response needs a quality score from a human. | InstructGPT used ~33K preference comparisons (pairs, not individual labels). The comparison format (A vs B) is easier for humans than absolute scoring, making data collection more efficient. The reward model generalizes from these comparisons to score new responses the humans never saw. | After explaining the preference data format. |
| "DPO is just a worse/simplified version of RLHF that sacrifices quality" | "Direct" and "simpler" often imply "less capable." The student might assume DPO is a shortcut that produces inferior results. This is reinforced by the general pattern that simpler methods are often less powerful. | DPO achieves comparable results to PPO-based RLHF on many benchmarks while being significantly simpler to implement and train. Llama 2 used PPO; many subsequent models (Zephyr, etc.) achieved strong results with DPO. DPO is not an approximation -- it is a mathematically equivalent reformulation that eliminates the reward model as a separate training stage. | When introducing DPO, after the student understands what PPO does. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| SFT model producing a harmful, sycophantic, or confidently wrong response to a reasonable instruction | Positive | The hook. Demonstrates concretely WHY SFT alone is insufficient. The student sees that following instructions and following them WELL are different things. An SFT model told "How do I pick a lock?" might give detailed instructions. An SFT model asked "Am I right that the Earth is flat?" might agree sycophantically. | Makes the alignment problem visceral. The student should feel uncomfortable with the SFT model's behavior. This motivates the entire lesson: we need a signal for QUALITY, not just FORMAT. |
| Human preference comparison: two responses to the same prompt, one preferred over the other | Positive | Shows what RLHF training data looks like. A prompt ("Explain quantum computing to a 10-year-old"), two responses (one verbose and jargon-heavy, one clear and age-appropriate), and a human preference label. The student sees that the training signal is relative (A > B), not absolute. | The comparison format is the key innovation. It is easier for humans to compare than to score, and the relative signal is more informative than a thumbs-up/thumbs-down. Connects to the intuition that "I don't know what a perfect response looks like, but I can tell you which of these two is better." |
| The reward model as a modified language model (classification head on hidden states, outputs a scalar score) | Positive | Shows that the reward model is architecturally similar to what the student built in Lesson 1 (classification head on pretrained transformer). Instead of classifying sentiment, it scores response quality. This callback makes the reward model feel familiar rather than magical. | Grounds the reward model in known architecture. The student has already built a pretrained-transformer-plus-classification-head model. The reward model is the same pattern: pretrained LM + scalar output head. |
| PPO without KL penalty: the model finds a degenerate strategy that gets high reward but produces garbage | Negative | Shows why unconstrained optimization against a reward model produces reward hacking. The model learns to exploit quirks in the reward model rather than genuinely improving quality. The KL penalty is the solution: stay close to the SFT model so you do not drift into degenerate territory. | The "reward hacking" failure mode is the most important concept in understanding why RLHF is hard. It also motivates the KL penalty, which is the key design choice in PPO for language models. Without this negative example, the student might think optimization is straightforward. |
| DPO eliminating the reward model by working directly with preference pairs | Positive (boundary) | Contrasts DPO with PPO. Shows that you can achieve similar results without training a separate reward model, by directly adjusting the language model's probabilities on preferred vs dispreferred responses. The contrast makes both approaches clearer. | Defines the boundary between two approaches. The student sees that the same goal (align with preferences) can be achieved through two different paths. DPO is the modern, simpler path that many practitioners use. |

---

## Phase 3: Design

### Narrative Arc

Your instruction-tuned model follows instructions. You ask it to write a poem, it writes a poem. You ask it to explain quantum computing, it explains quantum computing. Mission accomplished -- or is it? Try asking it how to pick a lock. It helpfully provides step-by-step instructions. Ask it "Am I right that the Sun orbits the Earth?" and it agreeably confirms your misconception. Ask it to summarize a research paper and it produces a confident, fluent summary that misrepresents the paper's conclusions. The model learned the FORMAT of being helpful from SFT, but it has no concept of what "helpful" actually means. It does not know that some instructions should be refused, that agreeing with the user is not always helpful, or that confidence should correlate with accuracy. SFT gave the model a voice; alignment gives it judgment. But how do you teach judgment? You cannot write a loss function for "be helpful but not harmful." You cannot label training data with "this response is 7.3 out of 10 helpful." What you CAN do is show the model two responses and say "this one is better." That simple comparative signal -- human preferences -- is the foundation of RLHF, and it turns out to be remarkably powerful.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | Side-by-side responses: SFT model giving a harmful/sycophantic answer vs an aligned model refusing or correcting. Also: a concrete preference comparison pair showing two responses to the same prompt with a human preference label. Real text that the student reads and evaluates. | The behavioral difference between SFT and aligned models is the motivating phenomenon. Just as Lesson 2 showed base vs SFT, this lesson shows SFT vs aligned. Text output is the natural modality. The preference pair demystifies the training data. |
| Verbal/analogy | "SFT gives the model a voice; alignment gives it judgment." The model went from mute (base) to speaking (SFT) to speaking wisely (aligned). Also: reward model as "an experienced editor" -- the editor does not write the article but can tell you which draft is better. | The voice/judgment distinction cleanly separates what SFT does from what alignment does. The editor analogy makes the reward model's role intuitive without requiring RL vocabulary. |
| Visual | A pipeline diagram extending the SftPipelineDiagram from Lesson 2: three stages (pretraining -> SFT -> RLHF) with this lesson's stage highlighted. Also: a flow diagram showing the RLHF training loop: generate response -> reward model scores it -> PPO updates policy -> KL penalty prevents drift. | The pipeline diagram provides continuity with the visual language from Lesson 2. The RLHF loop diagram makes the multi-component training process concrete. Without a visual, PPO is an alphabet soup of components. |
| Intuitive | "I cannot define a perfect response, but I can tell you which of these two is better." This is the human experience of preference -- we are better at comparing than absolute scoring. RLHF leverages this natural human ability. Also: the KL penalty as "don't forget everything SFT taught you while chasing the reward." | The comparison intuition grounds the entire approach in something the student already does daily. It explains WHY preference data is the format, not just that it is. The KL penalty connects to catastrophic forgetting, which the student already understands. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3
  1. The alignment problem: why SFT produces models that follow instructions but not necessarily well (harmful, sycophantic, confidently wrong)
  2. Reward models: training a model on human preference pairs to predict which response humans prefer
  3. Policy optimization: PPO (at intuitive level) and DPO as two approaches to optimizing the language model against preferences
- **Previous lesson load:** STRETCH (instruction-tuning was genuinely demanding with 3 new concepts and a notebook)
- **This lesson's load:** BUILD -- appropriate. No notebook, no implementation. Three new concepts but all taught at intuitive level rather than implementation depth. After a STRETCH, BUILD provides recovery while maintaining progress. The student does not need to write code or do math -- they need to understand the WHY and the WHAT of alignment, not the HOW in detail.

### Connections to Prior Concepts

| Prior Concept | How It Connects | Source |
|--------------|----------------|--------|
| "SFT teaches format, not knowledge" | Extended: SFT teaches FORMAT, but format alone does not guarantee quality. The model can follow the format of being helpful while being harmful. RLHF adds a signal for QUALITY. | 4.4.2 |
| Classification finetuning (add head, freeze backbone, train on labeled data) | The reward model IS a classification-style model: pretrained LM backbone + scalar output head, trained on labeled preference data. Same architectural pattern the student used in Lesson 1, applied to a different task (preference prediction instead of sentiment). | 4.4.1 |
| Cross-entropy loss as the standard training objective | RLHF introduces a fundamentally different objective: maximize reward subject to a KL constraint. This is the first time in the course the student sees an objective other than cross-entropy (or a direct variant of it). DPO brings it back closer to cross-entropy by reformulating as a log-probability loss. | 4.3.2 |
| Catastrophic forgetting | The KL penalty in PPO serves the same purpose as frozen backbone in classification finetuning: prevent the model from forgetting what it learned in the previous stage. It is the continuous version of "freeze the backbone." Instead of hard-freezing, the KL penalty soft-constrains the model to stay near the SFT policy. | 4.4.1 |
| "Same heartbeat" training loop | Broken for the first time. RLHF's training loop is NOT the same heartbeat: generate -> score -> update (with two models involved). This is a genuine departure from the pattern. DPO partially restores the familiar loop shape. | 4.3.2 |
| The SftPipelineDiagram (pretraining -> SFT -> RLHF grayed out) | The grayed-out third stage from Lesson 2 is now highlighted. The student has already seen this pipeline; now they understand the third stage. Continuity of visual language. | 4.4.2 |

**Potentially misleading prior analogies:**
- **"Same heartbeat"** -- This analogy was extended three times (pretraining, classification, SFT). The student may expect RLHF to follow the same pattern. It does NOT: RLHF involves generating text, scoring it with a reward model, and updating with PPO -- a fundamentally different loop structure. The lesson should explicitly note: "For the first time, the training loop changes shape." DPO then partially restores familiar territory (closer to standard supervised learning).
- **"A pretrained transformer is a text feature extractor"** -- Already flagged in Lesson 2 as not applying to SFT. Remains inapplicable here. The reward model does use a pretrained transformer as a feature extractor (with a scalar head), so this analogy partially resurfaces in a new context.

### Scope Boundaries

**This lesson IS about:**
- Why SFT alone is insufficient (the alignment problem, concretely)
- What human preference data looks like (comparison pairs, not absolute scores)
- Reward models: what they are, how they are trained, what they output
- PPO for language models at an intuitive level: the generate-score-update loop, KL penalty, reward hacking
- DPO as a simpler alternative: same goal, no separate reward model, preference pairs directly train the policy
- Why alignment matters beyond safety: it makes models genuinely more useful

**This lesson is NOT about:**
- Implementing RLHF or DPO in code (no notebook)
- PPO algorithm details (no value function, advantage estimation, clipping ratio math)
- RL formalism beyond the minimum needed (policy = model, reward = score, that is enough)
- Constitutional AI or RLAIF (Series 5)
- Red teaming, adversarial evaluation, or safety benchmarks
- The political/philosophical aspects of alignment (what values SHOULD be optimized)
- Specific model comparisons (InstructGPT vs ChatGPT vs Claude details)
- Multi-objective alignment (balancing helpfulness vs harmlessness vs honesty in depth)

**Depth targets:**
- The alignment problem (why SFT is insufficient): DEVELOPED (student can explain with concrete examples why instruction-following and helpful-instruction-following are different)
- Human preference data format: DEVELOPED (student understands comparison pairs, can explain why comparison is easier than absolute scoring)
- Reward models: INTRODUCED (student knows it is a neural net trained on preferences that outputs a scalar score, knows the architectural callback to classification finetuning, but does not implement it)
- PPO for language models: INTRODUCED (student understands the generate-score-update loop and why KL penalty prevents reward hacking, but does not know the algorithm details)
- DPO: INTRODUCED (student knows it achieves similar results without a reward model, using preference pairs directly, but does not know the math)
- Reward hacking: INTRODUCED (student understands the failure mode and why constraints are needed)

### Lesson Outline

1. **Context + Constraints** (Row)
   - What we are doing: understanding why aligned models (ChatGPT, Claude) behave differently from SFT-only models, and how human preferences become a training signal
   - What we are NOT doing: implementing RLHF, PPO algorithm details, RL formalism, constitutional AI
   - No notebook -- this is a conceptual lesson. The ideas are essential for understanding why modern LLMs behave the way they do, but the implementation complexity of RLHF exceeds what we can do with GPT-2 in a notebook
   - The bridge from Lesson 2: "SFT taught the model to follow instructions. Now: how do we teach it to follow them WELL?"

2. **Hook -- The Problem With SFT** (Row + Aside)
   - Three concrete examples of SFT failure modes:
     - **Harmful helpfulness:** "How do I pick a lock?" -> SFT model gives detailed instructions (it learned to follow instructions, not to refuse dangerous ones)
     - **Sycophancy:** "Am I right that the Earth is flat?" -> SFT model agrees with the user (it learned to be helpful, and agreeing feels helpful)
     - **Confident incorrectness:** A model summarizing a paper but misrepresenting a key conclusion, doing so fluently and confidently
   - The common thread: the model learned the FORMAT of being helpful but has no training signal for what "helpful" actually means. Cross-entropy on instruction-response pairs rewards producing plausible-sounding responses, not producing GOOD responses.
   - GradientCard with the key question: "SFT teaches format. What teaches quality?"

3. **Explain -- From Format to Quality: The Need for a New Signal** (Row + Aside)
   - Cross-entropy loss treats all correct next tokens equally. It cannot express "this response is better than that response." It can only say "this token was the right next token."
   - What we need: a training signal that says "this COMPLETE response is better than that COMPLETE response." Not token-by-token, but holistic evaluation of the full output.
   - Problem: you cannot write a loss function for "be helpful." Helpfulness is subjective, context-dependent, and impossible to specify as a formula.
   - Solution: ask humans. Humans cannot define helpfulness precisely either, but they CAN compare two responses and say "this one is better." Introduce the preference comparison format.
   - Pipeline callback: extend the Lesson 2 SftPipelineDiagram. Pretraining (text prediction) -> SFT (instruction following) -> RLHF (preference alignment). Now the third stage is active. Each stage adds something: knowledge -> format -> quality.
   - Address misconception early: RLHF does not replace SFT. SFT is necessary first. Without SFT, the base model produces document continuations -- you cannot meaningfully compare two document continuations for "helpfulness." SFT gets the model into instruction-following mode; RLHF refines the quality of that following.

4. **Explain -- Human Preference Data** (Row + Aside)
   - Show a concrete preference comparison:
     - Prompt: "Explain quantum computing to a 10-year-old"
     - Response A (dispreferred): "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be intractable for classical Turing machines..." (jargon-heavy, not age-appropriate)
     - Response B (preferred): "Imagine you have a magic coin that can be both heads and tails at the same time..." (clear, age-appropriate, uses analogy)
     - Human label: B > A
   - Key insight: the signal is RELATIVE, not absolute. Humans do not score responses on a 1-10 scale (unreliable, inconsistent across annotators). They compare pairs (more consistent, easier, leverages natural human ability).
   - "I cannot define a perfect response, but I can tell you which of these two is better" -- this is the core intuition.
   - Scale: InstructGPT used ~33K preference comparisons. Small compared to pretraining (billions of tokens) and comparable to SFT (thousands-tens of thousands). But each comparison is expensive (requires a human to read two full responses and decide).
   - Misconception address: RLHF does not need millions of labels. The comparison format is efficient, and the reward model generalizes from the comparisons.

5. **Check 1 -- Predict and Verify** (Row)
   - "You have preference pairs: (prompt, response A, response B, which is better). You want to train a model that can score ANY response to ANY prompt. What architecture would you use?"
   - Expected: The student should recognize this as similar to classification finetuning from Lesson 1: take a pretrained language model, add a head, train on labeled data. Instead of a sentiment head, it is a "quality score" head. The output is a scalar (one number) rather than class probabilities.
   - Reveal: Exactly right. The reward model is a pretrained language model with a scalar output head. Same pattern as Lesson 1. The callback is explicit.

6. **Explain -- The Reward Model** (Row + Aside)
   - Architecture: pretrained language model + linear head that outputs a single scalar. Feed in (prompt + response), get a score.
   - Training: for each preference pair, compute reward(preferred) - reward(dispreferred). The loss pushes this difference to be positive. The reward model learns to assign higher scores to responses humans preferred.
   - Editor analogy: the reward model is like an experienced editor. It does not write the article, but it can tell you which draft is better. It learned this judgment from seeing thousands of human comparisons.
   - Key point: the reward model is imperfect. It is a learned approximation of human preferences, not a perfect oracle. It has biases, blind spots, and failure modes. This matters for understanding reward hacking.
   - Misconception address: the reward model is NOT a rule-based system. It is a neural network trained on data, just like every other model in this course. It learns nuance that no set of rules could capture.

7. **Explain -- PPO: Optimizing Against the Reward** (Row + Aside)
   - Now we have a reward model that scores responses. How do we use it to improve the language model?
   - The generate-score-update loop:
     1. The language model generates a response to a prompt
     2. The reward model scores the response
     3. The language model is updated to increase the probability of high-scoring responses and decrease the probability of low-scoring ones
   - This is where RL vocabulary enters, minimally: the language model is the "policy" (it decides what to generate), the reward model provides the "reward signal."
   - **Critical:** "For the first time in this course, the training loop changes shape." Pretraining, classification, and SFT all followed the same heartbeat: forward, loss, backward, step. PPO breaks this pattern: generate (many tokens), score (whole response), update (the policy). Two models are involved (policy + reward model). The loop operates at the response level, not the token level.
   - The KL penalty -- the essential constraint:
     - Without constraint: the model finds degenerate strategies that game the reward model. It discovers that certain patterns of text get high reward scores even though they are not genuinely good responses. This is reward hacking.
     - Negative example: a model that learns to be excessively verbose because the reward model slightly prefers longer responses. Or a model that learns to repeat confident-sounding phrases because the reward model associates confidence with quality.
     - The KL penalty says: "Optimize the reward, but do not drift too far from the SFT model." It measures the divergence between the current policy and the SFT baseline. Large divergence is penalized.
     - Connection to catastrophic forgetting: the KL penalty is the continuous version of "freeze the backbone." Instead of hard-freezing (binary: frozen or unfrozen), KL gives a gradient (soft: stay close but can drift a little). Same purpose: preserve what was learned before.
   - Why PPO specifically? It is a stable RL algorithm that works well with large models. But the student does not need the PPO algorithm details (clipping, value function, advantage estimation). The intuition is sufficient: generate, score, update, stay close to SFT.

8. **Check 2 -- Transfer Question** (Row)
   - "What happens if you remove the KL penalty and train with PPO for many steps?"
   - Expected: the model reward-hacks. It finds degenerate strategies that maximize the reward model's score without producing genuinely better responses. It drifts far from the SFT model and may produce nonsensical but high-scoring text. The KL penalty prevents this by anchoring the model to the SFT baseline.
   - This reinforces the reward hacking concept and the necessity of the KL constraint.

9. **Explain -- DPO: The Simpler Alternative** (Row + Aside)
   - The PPO pipeline is complex: train a reward model, then use it in a multi-model training loop with KL penalties.
   - DPO insight: you do not need a separate reward model. You can directly adjust the language model using preference pairs.
   - Intuition: instead of (1) train a reward model, (2) generate responses, (3) score them, (4) update the policy -- just directly increase the probability of preferred responses and decrease the probability of dispreferred responses. The preference data IS the training signal, without the reward model intermediary.
   - DPO partially restores the familiar training loop shape: it looks more like supervised learning (forward pass on preference pairs, compute a loss, backward, step). Still not exactly "same heartbeat" -- the loss function is different and operates on pairs of responses -- but closer to familiar territory than PPO.
   - Results: DPO achieves comparable quality to PPO on many benchmarks. Llama 2 used PPO; many subsequent models (Zephyr, Mistral instruct variants) used DPO. Neither approach is universally better -- they are two paths to the same goal.
   - DPO still implicitly has a KL penalty built into its formulation (the reference model appears in the loss), so it does not suffer from unbounded reward hacking the way PPO-without-KL would.

10. **Elaborate -- Why Alignment Matters Beyond Safety** (Row + Aside)
    - Alignment is often framed as a safety concern (preventing harmful outputs). This matters, but it is not the only reason alignment matters.
    - Aligned models are genuinely more USEFUL: they give clearer explanations, admit uncertainty, ask clarifying questions, refuse gracefully when they cannot help. These are quality improvements, not just safety features.
    - Aligned models are more HONEST: they are less likely to confidently hallucinate. (Though this is imperfect -- RLHF teaches what humans prefer, not what is true. Misconception callback.)
    - The student should leave understanding that alignment is not "adding safety rails" but "teaching the model judgment about what makes a response genuinely good."
    - Brief forward reference: Constitutional AI (Series 5) extends this further -- what if AI provides the preference signal instead of humans?

11. **Summarize** (Row)
    - Pipeline complete: pretraining (knowledge) -> SFT (format) -> RLHF (quality/judgment). Each stage adds something essential.
    - The key innovation: human preferences as a training signal. Not rules, not loss functions -- just "which of these two is better?"
    - Reward models: pretrained LM + scalar head, trained on preference pairs. Same pattern as classification finetuning applied to response quality.
    - PPO: generate, score, update -- with KL penalty to prevent reward hacking. First time the training loop changed shape.
    - DPO: same goal, no reward model, directly optimize on preference pairs. Simpler, comparable results.
    - Mental model: "SFT gives the model a voice; alignment gives it judgment."
    - Misconception correction: RLHF teaches what humans PREFER, not what is TRUE. Preferences correlate with truth but are not identical.

12. **Next Step** (Row)
    - You now understand the full pipeline: pretrain, finetune (SFT), align (RLHF/DPO). But there is a practical problem: full finetuning (whether SFT or RLHF) requires storing all model parameters plus their gradients and optimizer states. For GPT-2 (124M parameters) this is manageable; for a 7B or 70B model, it requires expensive hardware. Next: LoRA and quantization -- how to make finetuning and inference accessible on real hardware.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via records (SFT mechanics from 4.4.2, classification finetuning from 4.4.1, training loop from 4.3.2, next-token prediction from 4.1.1)
- [x] Depth match verified: all OK, student exceeds requirements in most areas
- [x] No untaught concepts remain
- [x] No multi-concept jumps (lesson is conceptual, no widget or notebook exercises that could overload)
- [x] No gaps to resolve
- [x] RL concepts explicitly noted as taught within this lesson at INTRODUCED depth, not assumed as prerequisites

### Pedagogical Design
- [x] Narrative motivation stated as coherent paragraph (problem before solution: SFT model follows instructions but can be harmful/sycophantic/wrong, need a signal for quality)
- [x] At least 3 modalities: concrete example (SFT failure modes, preference pair), verbal/analogy (voice/judgment, editor), visual (pipeline diagram, RLHF loop), intuitive ("I can't define perfect but I can compare")
- [x] At least 2 positive examples (preference comparison pair, reward model architecture callback) + 1 negative (reward hacking without KL penalty) + 1 boundary example (DPO vs PPO)
- [x] At least 3 misconceptions (5 identified) with negative examples
- [x] Cognitive load = 3 new concepts (alignment problem, reward models, policy optimization), all at intuitive level
- [x] Every new concept connected to existing concept (reward model -> classification head from 4.4.1; KL penalty -> catastrophic forgetting from 4.4.1; preference data -> instruction data from 4.4.2; alignment problem -> "SFT teaches format, not knowledge" from 4.4.2)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-13 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues. The lesson is well-structured, pedagogically sound, and faithfully follows its planning document. The student would learn the core concepts effectively. However, four improvement-level findings would make the lesson meaningfully stronger, particularly around the reward model training explanation which uses a subtle formulation the student may not fully grasp, and the DPO section which lacks a concrete example.

### Findings

#### [IMPROVEMENT] — Reward model training loss is described abstractly without grounding

**Location:** Section 7 (The Reward Model), paragraph starting "Training: For each preference pair..."
**Issue:** The lesson says "compute reward(preferred) minus reward(dispreferred). The loss pushes this difference to be positive." This is the correct intuition, but it is stated at an abstract level without a concrete worked example. The student has never encountered a pairwise loss before. Every previous loss function in the course (MSE, cross-entropy) was grounded in concrete numbers. This one is not. The student reads "the loss pushes this difference to be positive" and has to take it on faith rather than seeing it work.
**Student impact:** The student understands the reward model architecturally (pretrained LM + scalar head -- callback to classification finetuning) but has a vague understanding of HOW it learns. They can parrot "it learns from preference pairs" but cannot trace through even one training step mentally.
**Suggested fix:** Add a brief concrete trace. Example: "The reward model scores Response A at 0.3 and Response B at 0.7. The human preferred B. reward(preferred) - reward(dispreferred) = 0.7 - 0.3 = 0.4 > 0. Good -- the model already agrees with the human on this pair. If the scores were reversed (A at 0.7, B at 0.3), the difference would be -0.4, and the loss would push the model to adjust." Two sentences, no formulas, grounds the abstract description in numbers the student can verify.

#### [IMPROVEMENT] — DPO section lacks a concrete example

**Location:** Section 10 (DPO: The Simpler Alternative)
**Issue:** The DPO section explains the concept well at the intuitive level ("directly increase the probability of preferred responses and decrease the probability of dispreferred responses") and uses a ComparisonRow to contrast with PPO. But it does not provide a concrete example of how this works. Every other major concept in the lesson has at least one concrete grounding: the alignment problem has three PhaseCards with specific prompts and responses, human preference data has the quantum computing example, the reward model has the architecture diagram with a specific score (0.82), and PPO has the three-step loop diagram. DPO is the only core concept explained purely in abstract prose.
**Student impact:** The student understands DPO as "simpler PPO" but does not have a concrete mental image of what happens during a DPO training step. The ComparisonRow lists steps but they remain procedural descriptions, not grounded in specific data.
**Suggested fix:** Add a brief concrete trace after the ComparisonRow. Something like: "Given the same preference pair (quantum computing prompt, jargon response A, age-appropriate response B, human prefers B), DPO computes the log-probability of both responses under the current model. If the model assigns higher log-probability to the preferred response B, the loss is already small. If it assigns higher probability to the dispreferred A, the loss is large and the gradient pushes the model to favor B." This reuses the existing preference example from Section 4, which reinforces it.

#### [IMPROVEMENT] — The "experienced editor" analogy is introduced but not fully developed

**Location:** Section 7 (The Reward Model), paragraph starting "Think of the reward model as an experienced editor..."
**Issue:** The editor analogy is stated in two sentences: "The editor does not write the article, but they can tell you which draft is better. They learned this judgment from seeing thousands of human comparisons." This is a good analogy but it is underdeveloped compared to the lesson's other analogies. The "voice vs judgment" analogy from the aside is used three times across the lesson (aside in Section 3, summary, and elaborate section). The editor analogy appears once and is never referenced again. Given that the reward model is one of the three core concepts, its primary analogy deserves more weight.
**Student impact:** Minor -- the student gets the general idea. But the editor analogy has unused potential. For instance, it could be extended to explain reward hacking: "An editor who only saw articles from one genre might overly reward stylistic features of that genre. The reward model, having learned from a finite set of human preferences, can be similarly gamed." This would provide a second angle on reward hacking beyond the current technical explanation.
**Suggested fix:** Either (a) extend the editor analogy to also explain reward hacking (editor has blind spots, the model exploits them), or (b) reference it again in the KL penalty section to create continuity. Option (a) is stronger because it gives a second modality for reward hacking.

#### [IMPROVEMENT] — "RLHF teaches what humans prefer, not what is true" misconception addressed late

**Location:** Section 11 (Why Alignment Matters Beyond Safety), "Important Caveat" GradientCard
**Issue:** The planning document identifies "RLHF teaches the model what is true" as a misconception and says to address it "in the Core concept section, after introducing the reward model." In the built lesson, this misconception is not addressed until Section 11, after the student has already gone through the reward model, PPO, KL penalty, DPO, and a section on why alignment matters. The student spends 5+ sections potentially holding the misconception that RLHF teaches truth, only to be corrected near the end. The planning document's placement was deliberate: correct the misconception while the reward model is fresh, before building more concepts on top of it.
**Student impact:** The student may be building a mental model throughout the middle of the lesson where RLHF is "teaching the model to be correct." When the caveat arrives in Section 11, it requires retroactive revision of that model. Earlier placement would prevent the misconception from forming in the first place.
**Suggested fix:** Add a brief mention in the reward model section (Section 7), after explaining how the reward model learns from human preferences. Something like: "Note what the reward model learns: what humans prefer, not what is objectively true. A confident, well-structured explanation may be preferred over a hedging, uncertain one even when the hedging response is more accurate. We will revisit this limitation later, but keep it in mind: preference and truth are correlated, not identical." Then keep the full elaboration in Section 11 as a callback.

#### [POLISH] — Spaced em dashes in SummaryBlock description strings

**Location:** SummaryBlock items, lines 1240, 1246, 1250, 1264
**Issue:** The description strings in the SummaryBlock use ` — ` (space-em-dash-space) in several places:
- `'"Which of these two is better?" — that simple comparative...'`
- `'Same pattern as classification finetuning — a pretrained backbone...'`
- `'PPO: generate, score, update — with KL penalty.'`
- `'Preferences correlate with truth, helpfulness, and safety — but they are not identical.'`

The writing style rule requires no-space em dashes (`word—word`). The rest of the lesson correctly uses `&mdash;` in JSX (which renders with no spaces), but the SummaryBlock strings use literal ` — ` which will render as spaced.
**Student impact:** Purely visual inconsistency. The main lesson prose uses tight em dashes; the summary uses spaced em dashes.
**Suggested fix:** Replace ` — ` with `—` (or `\u2014`) in the summary description strings to match the rest of the lesson.

#### [POLISH] — The "What comes next" block uses a manual div instead of NextStepBlock

**Location:** Section 13 (Next Step), lines 1276-1289
**Issue:** There are two next-step elements in the lesson: a manually styled `<div>` with "What comes next" and then a `<NextStepBlock>` component. The manually styled div handles the narrative bridge to the next lesson (LoRA/quantization motivation). The NextStepBlock handles the "when you're done" action. This works, but the manual div introduces a one-off styling pattern that may drift from the design system over time. Other lessons appear to handle both the "what's next" narrative and the action in a single block or in two blocks that both use the component system.
**Student impact:** None functionally. The content is good and the transition to PEFT is well-motivated.
**Suggested fix:** Low priority. If a `PreviewBlock` or similar component exists or is created, use it. Otherwise, leave as is -- this is a minor consistency point, not a quality issue.

#### [POLISH] — Section 11 grid of GradientCards uses bullet points with `{'•'}` instead of `<ul>/<li>`

**Location:** Section 11 (Why Alignment Matters Beyond Safety), the "More Useful" and "More Honest" GradientCards
**Issue:** The bullet points are created with `{'•'}` characters inside `<li>` tags. While this renders correctly, it is a manual bullet approach. If other lessons use `<ul className="list-disc">` or rely on CSS list styling, this is an inconsistency. The items are also wrapped in `<li>` elements inside a `<ul>` but the `<ul>` has no `list-style` class, so the `{'•'}` is being manually added.
**Student impact:** None.
**Suggested fix:** Either remove the `{'•'}` and add appropriate Tailwind list styling, or leave as is. This is a code style preference, not a student-facing issue.

### Review Notes

**What works well:**

1. **Motivation is excellent.** The three PhaseCards showing SFT failure modes (harmful helpfulness, sycophancy, confident incorrectness) are concrete, vivid, and genuinely motivating. The student feels the problem before getting the solution. This is one of the stronger hooks in the course.

2. **Connections to prior concepts are strong and explicit.** The reward model callback to classification finetuning (same architecture pattern: backbone + head) is exactly the right connection. The KL penalty callback to catastrophic forgetting is well-executed -- "the continuous version of freeze the backbone" is a genuinely clarifying framing. The "for the first time, the training loop changes shape" moment correctly signals a departure from the "same heartbeat" pattern.

3. **Visual diagrams are effective.** The three inline SVGs (AlignmentPipelineDiagram, RlhfLoopDiagram, RewardModelDiagram) each serve a clear purpose and match the lesson's conceptual flow. The pipeline diagram provides continuity with Lesson 2's SftPipelineDiagram. The RLHF loop diagram makes the multi-component training process concrete. The reward model diagram explicitly connects to the classification finetuning architecture.

4. **Scope is well-managed.** For a conceptual lesson covering RLHF, DPO, reward models, and reward hacking, the lesson stays appropriately bounded. RL formalism is kept to the bare minimum (policy = model behavior). PPO details are correctly deferred. The student leaves with the right mental models without being overwhelmed by implementation details.

5. **The two checkpoints are well-placed.** Check 1 (predict the reward model architecture) leverages the student's classification finetuning experience. Check 2 (what happens without KL penalty) reinforces the most important failure mode concept. Both are predict-then-reveal format.

6. **All five misconceptions from the planning document are addressed.** "RLHF replaces SFT" (Section 4 GradientCard + aside), "reward model is rule-based" (Section 7 WarningBlock), "RLHF needs millions of labels" (Section 5 aside), "DPO is inferior" (Section 10 aside), "RLHF teaches truth" (Section 11 caveat). The timing of the last one is the improvement finding noted above.

**Pattern observation:** This lesson is notably strong on the motivation and "why" dimensions but slightly weaker on the "concrete grounding" dimension for its later concepts (DPO and reward model training mechanics). The earlier concepts (alignment problem, preference data) are grounded in vivid, specific examples. The later concepts rely more on abstract description and comparison. This is likely an energy/pacing issue -- after the strong hook, the lesson maintains prose quality but the concrete examples thin out.

---

## Review — 2026-02-13 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 3

### Verdict: PASS

All four improvement findings from iteration 1 have been addressed effectively. The concrete reward model training trace (0.3 vs 0.7) grounds what was previously abstract. The DPO section now reuses the quantum computing preference pair for a concrete trace. The editor analogy is extended to explain reward hacking via "blind spots." The "preference not truth" misconception is now seeded in Section 7 (reward model) and fully elaborated in Section 11, matching the planning document's intended placement. The spaced em dashes in the SummaryBlock have been fixed. The lesson is pedagogically sound with no critical or improvement-level issues remaining.

### Findings

#### [POLISH] — Confident incorrectness PhaseCard is described rather than shown

**Location:** Section 3 (The Problem With SFT), PhaseCard 3 (Confident Incorrectness), lines 557-573
**Issue:** The first two PhaseCards show concrete text (the lock-picking response, the flat Earth agreement). The third says "A fluent, well-structured summary that misrepresents the paper's main conclusion--stated with complete confidence" without actually showing the summary text. This is a description of what the model would do, not a demonstration. The first two PhaseCards are more vivid because the student reads actual model output.
**Student impact:** Minor. The pattern is clear from the first two PhaseCards. The third communicates the failure mode but does not hit as hard as a concrete fabricated summary would. The student understands the concept; they just do not feel it as viscerally for this third case.
**Suggested fix:** Add a brief concrete summary text. For example, a 1-2 sentence "model output" that confidently states a wrong conclusion from a well-known paper, followed by a note that the real conclusion was different. This would make the third PhaseCard as vivid as the first two. Low priority because the lesson's motivation is already strong with the first two PhaseCards.

#### [POLISH] — Manual "What comes next" div remains outside the component system

**Location:** Section 13 (Next Step), lines 1319-1332
**Issue:** Carried over from iteration 1. The manually styled `<div>` with `bg-primary/10 border border-primary/30` handles the narrative bridge to the next lesson. This works but is a one-off styling pattern outside the component system. Other lessons in the codebase use the same pattern (e.g., PretrainingLesson lines 1160-1175), so this is a codebase-wide consistency question rather than a lesson-specific issue.
**Student impact:** None.
**Suggested fix:** No change needed for this lesson. If a `PreviewBlock` component is ever created for the design system, apply it retroactively to all lessons. This is a systemic issue, not specific to this lesson.

#### [POLISH] — Section 11 GradientCards use manual `{'•'}` bullet characters

**Location:** Section 11 (Why Alignment Matters Beyond Safety), lines 1206-1220
**Issue:** Carried over from iteration 1. The "More Useful" and "More Honest" GradientCards use `{'•'}` characters inside `<li>` tags rather than CSS list styling. This is a minor code style inconsistency.
**Student impact:** None. Renders correctly.
**Suggested fix:** Low priority. If a code style pass is done across all lessons, standardize on either `list-disc` or manual bullets, but do not change just this lesson in isolation.

### Review Notes

**Iteration 1 fixes verified:**

1. **Reward model training trace (0.3 vs 0.7):** Now present in Section 7, lines 837-845. The concrete numbers ground the previously abstract "pushes the difference to be positive" formulation. The student can now trace through one training step mentally. Effective fix.

2. **DPO concrete example (quantum computing pair reuse):** Now present in Section 10, lines 1138-1148. Reusing the quantum computing preference pair is the right choice -- it reinforces the earlier example while grounding DPO in something specific. The student can follow "compute log-probability of A and B, if model prefers A (dispreferred), loss is large, gradient pushes toward B." Effective fix.

3. **Editor analogy extended for reward hacking:** Now present in Section 8, lines 973-981. "An editor who only read articles from one genre might overly reward that genre's stylistic conventions. A clever writer could game the editor by mimicking those conventions..." This gives a second angle on reward hacking beyond the technical description, using familiar non-technical language. Effective fix.

4. **"RLHF teaches preference not truth" seeded early:** Now present in Section 7, lines 862-869. The seeding paragraph ("Note what the reward model learns: what humans prefer, not what is objectively true") is placed immediately after the reward model explanation, matching the planning document's intended location. The full elaboration in Section 11's "Important Caveat" GradientCard now serves as a callback rather than a first introduction. Effective fix.

5. **Spaced em dashes in SummaryBlock:** All four instances now use unspaced `—` characters. Verified at lines 1283, 1289, 1293, 1307. Fixed.

**What works well (confirmed from iteration 1, still holds):**

1. **Motivation remains the lesson's strongest dimension.** The three SFT failure PhaseCards are vivid and the "SFT teaches format. What teaches quality?" framing creates genuine curiosity.

2. **Concrete grounding is now consistently strong across all sections.** The iteration 1 improvements to the reward model training trace and DPO section eliminated the "concrete examples thin out in later sections" pattern noted in the iteration 1 review.

3. **All five planned misconceptions are addressed at appropriate locations.** The timing improvement for the "preference not truth" misconception means all five are now addressed at or before the point where the student might form them.

4. **Connections to prior concepts remain strong.** The reward model -> classification finetuning, KL penalty -> catastrophic forgetting, and "training loop changes shape" callbacks are all well-executed.

5. **The lesson faithfully implements its planning document.** No unplanned deviations. All planned examples, misconceptions, modalities, and scope boundaries are present and correctly positioned.
