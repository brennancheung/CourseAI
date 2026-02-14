# Lesson Plan: In-Context Learning

**Module:** 5.2 (Reasoning & In-Context Learning)
**Position:** Lesson 1 of 4 (first lesson of module)
**Slug:** `in-context-learning`
**Status:** Planning complete

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Attention as data-dependent weighted average (weights computed from input, not fixed parameters) | DEVELOPED | the-problem-attention-solves (4.2.1) | Core mechanism. "The input decides what matters." Student traced every number in a worked example, used the interactive AttentionMatrixWidget, and explicitly contrasted with CNN fixed filters. |
| QK^T as learned relevance (not raw similarity) | DEVELOPED | queries-and-keys (4.2.2) | Student understands projections create asymmetric, learned relevance scores. Can trace the full formula. |
| Full single-head attention formula: output = softmax(QK^T / sqrt(d_k)) V | DEVELOPED | values-and-attention-output (4.2.3) | Student built this formula over three lessons. Understands V separates matching from contributing. |
| Multi-head attention (parallel heads in subspaces, W_O for cross-head mixing) | DEVELOPED | multi-head-attention (4.2.4) | "Multiple lenses, pooled findings." Student understands heads capture diverse relationship types. |
| Autoregressive generation as feedback loop (sample, append, repeat) | DEVELOPED | what-is-a-language-model (4.1.1) | The loop is well established. Student implemented it in building-nanogpt (4.3.1). |
| Language modeling as next-token prediction (P(next token | context)) | DEVELOPED | what-is-a-language-model (4.1.1) | Core framing. The student has trained a model from scratch and loaded real GPT-2 weights. |
| SFT teaches format, not knowledge (base model has knowledge, SFT teaches conversational behavior) | DEVELOPED | instruction-tuning (4.4.2) | Expert-in-monologue analogy. Critical for understanding that ICL doesn't require SFT. |
| Causal masking (lower-triangular attention, each token sees only past) | DEVELOPED | decoder-only-transformers (4.2.6) | Student understands the mechanism and its purpose. |
| Transformer block: "Attention reads, FFN writes" | DEVELOPED | the-transformer-block (4.2.5) | Core mental model. Attention gathers context, FFN processes and transforms. |
| Few-shot jailbreaking as an attack category (examples of compliant model override safety training) | INTRODUCED | red-teaming-and-adversarial-evaluation (5.1.3) | In-context learning was mentioned as the mechanism exploited by few-shot jailbreaking. The student knows it exists but has NOT been taught what it is or why it works. |
| Capability-safety tension (model capabilities are also vulnerabilities) | INTRODUCED | red-teaming-and-adversarial-evaluation (5.1.3) | In-context learning was explicitly named as a capability that becomes a vulnerability. |

### Established Mental Models and Analogies

- "Attention is a weighted average where the input determines the weights" (4.2.1)
- "The input decides what matters" / data-dependent weights vs fixed parameters (4.2.1)
- "Three lenses, one embedding" / "Multiple lenses, pooled findings" (4.2.3, 4.2.4)
- "Next-token prediction is a universal training signal" (4.1.1)
- "Autoregressive generation is a feedback loop" (4.1.1)
- "SFT teaches format, not knowledge" (4.4.2)
- "Capability = vulnerability" (5.1.3)
- "The reward model is an experienced editor" (4.4.3)
- "Benchmarks are standardized tests for LLMs" (5.1.4)

### What Was Explicitly NOT Covered

- Prompt-based classification (zero-shot or few-shot) was explicitly listed as out of scope in finetuning-for-classification (4.4.1) and instruction-tuning (4.4.2)
- Prompt engineering or in-context learning was explicitly listed as out of scope in instruction-tuning (4.4.2)
- The GPT-3 paper and the few-shot prompting discovery
- Why examples in the prompt work mechanically (attention as the mechanism)
- The "transformer as a learning algorithm" framing

### Readiness Assessment

The student is well prepared. They have deep understanding of the attention mechanism (DEVELOPED through multi-lesson arc in 4.2), autoregressive generation (DEVELOPED and APPLIED), and the distinction between knowledge and behavior (SFT lesson). They have encountered in-context learning tangentially (few-shot jailbreaking in 5.1.3) and know it exists as a capability, but have never been taught what it is or why it works. The conceptual leap -- that attention can perform gradient-free task learning -- will be surprising but grounded in mechanisms they already understand.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain why a transformer can learn new tasks from examples placed in the prompt without any weight update, and to identify attention as the specific mechanism that makes this possible.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Attention as data-dependent weighted average | DEVELOPED | DEVELOPED | the-problem-attention-solves (4.2.1) | OK | Core mechanism. ICL works because attention weights are computed from the input (including examples). Student needs to understand this deeply to connect it to ICL. |
| Autoregressive generation feedback loop | DEVELOPED | DEVELOPED | what-is-a-language-model (4.1.1), building-nanogpt (4.3.1) | OK | ICL happens during generation. The student must understand that each generated token becomes part of the context for the next. |
| Next-token prediction as the training objective | DEVELOPED | DEVELOPED | what-is-a-language-model (4.1.1) | OK | The surprise of ICL is that a model trained only on next-token prediction can learn tasks at inference time. The student must understand the objective to feel the surprise. |
| Multi-head attention (parallel heads capturing diverse relationships) | INTRODUCED | DEVELOPED | multi-head-attention (4.2.4) | OK | Multiple heads explain how the model can simultaneously attend to structural patterns (input-output mapping) and semantic content. Only INTRODUCED depth needed. |
| SFT vs base model distinction | INTRODUCED | DEVELOPED | instruction-tuning (4.4.2) | OK | ICL works on base models (pre-SFT). Student needs to understand this distinction to see that ICL is not a result of instruction tuning. |
| QK^T as learned relevance function | INTRODUCED | DEVELOPED | queries-and-keys (4.2.2) | OK | The projections allow the model to match query tokens to relevant examples. Needed at INTRODUCED to explain the mechanism. |

All prerequisites are met at sufficient depth. No gaps.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **"ICL is just pattern matching / template filling -- the model saw this exact pattern in training data"** | The student has seen the model predict "5" after "2+3=" and might think all capabilities come from memorized patterns. The word "learning" seems too strong for what is really just retrieval. | Give the model examples of a made-up function (e.g., reversing words, or a novel symbol mapping like "sdag" -> "happy") that could not have appeared in training data. The model still follows the pattern from examples. If it were just template matching from training, it would fail on truly novel mappings. | Section 7 (Elaborate), after the mechanism is explained. Frame as: "If ICL were just memorized pattern retrieval, it would fail on patterns it has never seen. It does not always fail." |
| **"ICL works because the model 'understands' the task description and 'decides' to follow the examples"** | Natural anthropomorphic framing. The model "reads" the examples, "figures out" the pattern, and "applies" it. This implies comprehension and intentionality. | The model performs ICL even when the task labels are randomly flipped (Brown et al. 2020, Min et al. 2022 findings). If it truly "understood" the task, flipped labels should reverse its behavior completely. Instead, the format and structure of the examples matter more than the label content for many tasks. This is inconsistent with comprehension but consistent with attention-based retrieval. | Section 5 (Explain, core mechanism), embedded in the attention-based explanation. |
| **"Few-shot is always better than zero-shot"** | More examples = more information = better performance. The student may assume monotonic improvement with more examples. | Adding examples in the wrong order, with misleading format, or with subtle errors can degrade performance below zero-shot. Example ordering effects are dramatic (accuracy can swing 20-30 percentage points). If "more = better" were true, ordering would not matter. | Section 7 (Elaborate), as a limitation of ICL. |
| **"ICL requires instruction-tuned / chat models"** | The student has only used chat models (ChatGPT, Claude) for few-shot prompting. They may assume the instruction-following behavior is what enables ICL. | GPT-3 demonstrated ICL as a *base model* capability, before any SFT or RLHF was applied. The original few-shot results were on a base model. SFT helps with format but ICL is a property of the pretrained transformer. Callback to "SFT teaches format, not knowledge." | Section 4 (Hook / initial explanation), when introducing the GPT-3 discovery. Make explicit: base model, no SFT. |
| **"The model updates its weights when it sees the examples (some kind of fast learning / gradient step happens internally)"** | The word "learning" implies weight updates. The student has trained models and associates learning with gradient descent. ICL seems magical -- how can it learn without gradients? | Show the same model, same weights, with two different prompts (different examples). The model produces different outputs for the same test input depending on which examples are in the prompt. If weights were updated, the model's behavior on an empty prompt would also change. It does not -- the weights are frozen. The "learning" happens in the activations, not the parameters. | Section 3 (Hook) and Section 5 (Explain). This is THE core misconception to address early and definitively. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Sentiment classification with few-shot examples** (e.g., "This movie was great" -> Positive, "The food was terrible" -> Negative, "The hotel was nice" -> ???) | Positive | First example: simplest possible ICL demo. Shows the input-output pattern clearly. | Accessible, immediately graspable. The student can see the pattern without any domain knowledge. The task (sentiment) is familiar from finetuning-for-classification (4.4.1), creating a direct contrast: same task, no weight updates. |
| **Novel symbol mapping** (e.g., "foo" -> "bar", "baz" -> "qux", "hello" -> ???) or word reversal function (e.g., "cat" -> "tac", "dog" -> "god", "hello" -> ???) | Positive | Shows ICL on a task that demonstrably was NOT in training data. Confirms ICL is more than memorized patterns. | Disproves the "just pattern matching from training data" misconception. A genuinely novel mapping forces the model to learn the function from the examples alone. |
| **Same test input, different examples yielding different outputs** (e.g., give examples of translation to French vs translation to Spanish, then same test word) | Positive (stretch) | Shows that the "learning" is in the context, not the weights. Different prompts, same weights, different behavior. | Directly disproves "weights update during ICL." Makes the mechanism vivid: the prompt IS the program. |
| **Example ordering sensitivity** (same examples in different orders producing different accuracy) | Negative | Shows ICL is not robust pattern comprehension. Performance depends on surface features (recency, ordering) that would not matter if the model truly "understood." | Defines the boundary: ICL is powerful but fragile. Prevents overestimation. Grounds the mechanism in attention (recency bias from causal masking). |

### Gap Resolution

No gaps to resolve. All prerequisites are at sufficient depth.

---

## Phase 3: Design

### Narrative Arc

The student just finished Module 5.1 (Advanced Alignment), which was about controlling model behavior -- making models safe, honest, and well-behaved. This module shifts perspective: what can models actually *do*?

Here is the puzzle. You trained your GPT model on next-token prediction. Nothing in the training objective says "learn to classify sentiment from examples." Nothing says "learn to translate when given input-output pairs." The model was never told to do any of this. And yet: put three examples of sentiment classification into the prompt, and the model classifies the fourth. Put translation pairs, and it translates. Put a pattern it has never seen in training, and -- sometimes -- it follows that pattern too.

This is in-context learning, and it was the central discovery of the GPT-3 paper. It should not work -- at least, not according to the mental model that says "learning = gradient descent = weight updates." No weights change. No optimizer runs. The model sees examples in its context window and produces the right answer. How?

The answer is attention. The same mechanism the student spent six lessons building in Module 4.2 -- the data-dependent weighted average that computes relevance from the input -- is what makes in-context learning possible. Examples in the prompt create retrieval structures in the attention weights: the model's query for the test input matches keys from the examples, and the values carry the answer pattern. The transformer does not "learn" in the gradient descent sense. It *computes* a task-specific function from whatever is in its context. Every prompt is a different program.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example** | Three worked examples of ICL: sentiment classification (familiar task), novel symbol mapping (proves not just memorization), same input with different example sets (proves no weight update) | ICL is an empirical phenomenon. The student must SEE it work before understanding why. The examples progress from "ok, pattern matching" to "wait, this should not work" to "the prompt is the program." |
| **Visual** | Attention heatmap diagram (inline SVG) showing how a test input's query vectors attend to example inputs and outputs in the prompt. Highlight the retrieval pattern: test query -> example input (match) -> example output (retrieve answer pattern). | The mechanism is attention. The student has seen attention heatmaps before (AttentionMatrixWidget in 4.2.1). Showing the ICL-specific attention pattern -- where the model looks at examples -- makes the mechanism concrete and connects to existing visual vocabulary. |
| **Verbal/Analogy** | "The prompt is a program, and attention is the interpreter" -- the examples are data, the format is the instruction set, and the attention mechanism executes the program at inference time. Extends the existing "data-dependent weights" mental model. | Bridges from "attention computes relevance" (which the student knows) to "attention computes task-specific functions" (which is the new insight). The programming analogy connects to the student's software engineering background. |
| **Intuitive** | The "of course" beat: if attention weights are computed from the input, and examples are part of the input, then OF COURSE the examples influence the output. The mechanism is not new -- it is the same attention the student already understands, operating on a longer context. Nothing magical happened; the context just got more informative. | Collapses the surprise. After the initial "how can it learn without gradients?" puzzle, the student should feel "of course -- I already knew attention is data-dependent. I just did not think about what happens when the data includes examples." |
| **Symbolic** | Brief notation connecting ICL to the attention formula. With examples in context, the QK^T computation creates high scores between the test input's query and the example inputs' keys. The softmax-weighted sum of V vectors then blends the example outputs' representations into the test position's output. No new math -- just the existing formula operating on a prompt that contains examples. | Grounds the verbal explanation in the formula the student already knows. Prevents ICL from feeling like a separate, unexplained capability. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2
  1. In-context learning as gradient-free task learning from examples in the prompt
  2. The attention-based mechanism that explains why ICL works (attention over examples creates retrieval patterns)
- **Previous lesson load:** The previous lesson (evaluating-llms) was the capstone of Module 5.1, conceptually dense but synthesis-oriented. Not a STRETCH lesson.
- **This lesson's load:** STRETCH. This is the first lesson in a new module, introducing a paradigm that challenges the student's understanding of what "learning" means. The core surprise -- that a model trained on next-token prediction can learn tasks at inference time -- is genuinely novel. However, the mechanism (attention) is deeply familiar, which anchors the novelty.
- **Load is appropriate:** Following the capstone of Module 5.1, a STRETCH lesson is appropriate. The student has had four BUILD/CONSOLIDATE-ish lessons in Module 5.1 (the final lesson was evaluating-llms, which was synthesis). STRETCH here is earned.

### Connections to Prior Concepts

- **Attention as data-dependent computation (4.2.1):** The foundational connection. ICL works BECAUSE attention weights are computed from the input. If examples are in the input, they influence the weights. This is not a new mechanism -- it is the existing mechanism applied to a specific input structure. "You already know why this works; you just have not connected it to this phenomenon yet."
- **Autoregressive feedback loop (4.1.1):** ICL operates within the autoregressive generation loop. Each generated token becomes part of the context for the next. The model generates the answer token by token, with each token benefiting from attention over the examples.
- **SFT teaches format, not knowledge (4.4.2):** ICL is a base model capability, not something added by SFT. This connects to the student's understanding that the base model has enormous capability compressed into its weights from pretraining. ICL is one way that capability manifests at inference time.
- **Few-shot jailbreaking (5.1.3):** The student already saw ICL used as an attack vector. This lesson explains the mechanism behind what the student observed: adversarial examples in the prompt steer the model's behavior through attention, overriding safety training. The "capability = vulnerability" mental model is directly relevant.
- **Finetuning for classification (4.4.1):** The student classified sentiment by adding a head and training on labeled data. ICL classifies sentiment by putting examples in the prompt. Same task, dramatically different approach. This contrast makes the lesson's point vivid.

**Potentially misleading prior analogies:**
- "Learning = gradient descent = weight updates" is the mental model that ICL challenges. The student has internalized this from Series 1-3. The lesson must explicitly acknowledge and reframe: ICL is "learning" in the sense of adapting behavior to a task, but NOT in the sense of updating parameters.

### Scope Boundaries

**This lesson IS about:**
- What in-context learning is (zero-shot, few-shot prompting)
- Why it is surprising (no weight updates, not in the training objective)
- How attention mechanistically enables ICL
- The key limitations of ICL (context window, ordering sensitivity, fragility)
- The GPT-3 discovery as historical context

**This lesson is NOT about:**
- Systematic prompt engineering techniques (Lesson 2: prompt-engineering)
- Chain-of-thought or step-by-step reasoning (Lesson 3: chain-of-thought)
- Retrieval-augmented generation (Lesson 2: prompt-engineering)
- Implementing ICL in code beyond notebook exercises
- The theoretical framework for why ICL emerges during pretraining (e.g., the "transformers learn to implement gradient descent" line of research -- MENTIONED at most, not developed)
- Comparing ICL performance to finetuning quantitatively
- Role prompting or system prompts (Lesson 2)

**Target depth:**
- In-context learning (the phenomenon): DEVELOPED
- Attention as ICL mechanism: DEVELOPED
- ICL limitations (ordering, fragility): INTRODUCED
- "Transformers as learning algorithms" theoretical perspective: MENTIONED

### Lesson Outline

#### 1. Context + Constraints
What this lesson is about: how transformers can learn new tasks from examples in the prompt, without any weight update. What we are NOT covering: systematic prompt engineering (next lesson), chain-of-thought reasoning (Lesson 3), or implementing ICL from scratch.

#### 2. Recap (brief)
Quick reconnection to attention as data-dependent computation: "In Module 4.2, you built attention from scratch. The defining insight was that attention weights are computed from the input -- not fixed parameters. Every new input produces new weights." One paragraph, not a full re-teach. Also reconnect to finetuning for classification: "In Module 4.4, you classified sentiment by adding a head and training on labeled data. That required gradient descent, labeled examples, and weight updates." Set up the contrast.

#### 3. Hook (demo / before-after)
**Type:** Demo + puzzle.

Present the student with a concrete scenario: sentiment classification. "You know how to classify sentiment -- you did it in Module 4.4 by finetuning. That required a dataset, a training loop, and changing the model's weights. Now watch this."

Show a few-shot prompt:

```
Review: "This movie was amazing" -> Positive
Review: "The food was terrible" -> Negative
Review: "The scenery was breathtaking" -> Positive
Review: "The service was slow" -> ???
```

The model outputs "Negative." No training. No weight updates. No classification head. The same frozen model that generates Shakespeare can now classify sentiment -- simply because of what is in the prompt.

The puzzle: "The model's parameters did not change. No optimizer ran. No gradients were computed. And yet it performed a task it was never trained to do. How?"

#### 4. Explain Part 1: The GPT-3 Discovery
Brief historical grounding. Brown et al. (2020) showed that a large enough language model can perform tasks it was never explicitly trained on, simply by conditioning on examples in the prompt. Introduce terminology:
- **Zero-shot:** Task instruction only, no examples ("Classify the sentiment of this review:")
- **Few-shot:** Task instruction + examples (the demo above)

Key point: GPT-3 was a BASE MODEL. No SFT, no RLHF. ICL is a property of pretraining, not instruction tuning. Callback: "Remember 'SFT teaches format, not knowledge'? ICL shows that the base model already has remarkable capability -- SFT just makes it more accessible."

ComparisonRow: finetuning approach (new head, labeled data, gradient descent, weight updates, task-specific model) vs ICL approach (same model, examples in prompt, no gradients, no weight updates, general-purpose).

#### 5. Explain Part 2: Why It Works (Attention as the Mechanism)
This is the core of the lesson.

**Problem before solution:** "We said no weights change. So where does the task-specific behavior come from? It comes from the same mechanism you already know: attention."

Walk through the mechanism:
1. The prompt contains examples (input-output pairs) followed by a test input.
2. When the model processes the test input, its query vectors are computed from the test input's tokens.
3. These query vectors compute dot products with the key vectors of ALL tokens in the context -- including the example inputs and outputs.
4. If W_Q and W_K have learned to project inputs and outputs into a shared relevance space, the test input's queries will produce high scores against the example inputs' keys (because they are structurally similar).
5. The attention weights then blend the value vectors of the nearby tokens -- including the example *output* tokens.
6. The blended representation carries information about what output should follow this type of input.

**Visual:** Inline SVG attention pattern diagram. Show a simplified attention heatmap for a few-shot prompt. Highlight the critical pattern: the test input position attends strongly to the example input positions (structural matching) and the model uses the adjacent output tokens to predict the answer.

**The "of course" beat:** "You already knew this. Attention weights are data-dependent. If examples are in the context, they are part of the data. Of course the model attends to them. Of course similar inputs match. The mechanism is not new. You just had not thought about what happens when the context contains examples of a task."

Connect to the formula: softmax(QK^T / sqrt(d_k)) V. The Q comes from the test input. The K and V come from the entire context, including examples. "The formula has not changed. The context has."

#### 6. Check 1 (Predict-and-verify)
Present the student with two prompts for the same task. Prompt A has 3 sentiment examples. Prompt B has 3 translation examples. Both end with the same input: "The weather is beautiful."

Predict: What will the model output for each prompt?

Reveal: Prompt A outputs "Positive" (sentiment label). Prompt B outputs "Le temps est beau" (French translation). Same model, same weights, same test input. Different examples, different behavior. The prompt is the program.

#### 7. Elaborate: What ICL Is and Is Not

**Positive example 2 (novel mapping):** Show the model examples of a made-up mapping: "sdag" -> "happy", "trel" -> "sad", "blix" -> "angry", "wump" -> ???. The model produces an emotion word, following the pattern. This mapping was NOT in training data. ICL is more than memorized template matching.

**Negative example (ordering sensitivity):** Show that reordering the same examples can change accuracy by 20-30 percentage points. If the model truly "understood" the task, order would not matter. ICL performance depends on surface features that attention is sensitive to: recency (causal masking means later tokens attend to everything, earlier tokens attend to less), format consistency, and prompt structure.

**Misconception 1 address (just pattern matching):** The novel mapping example disproves pure memorization. But the ordering sensitivity shows ICL is also NOT human-like comprehension. The truth is in between: ICL is attention-based computation that can generalize to novel patterns within the scope of what the attention mechanism can compute in a single forward pass.

**Misconception 5 address (weights update):** Recap the Check 1 result. Same weights, different prompts, different behavior. If weights updated, behavior on an empty prompt would also change. It does not. The "learning" is in the activations (attention patterns), not the parameters.

**The label-flipping finding (Min et al. 2022, light touch):** Briefly mention that ICL performance is sometimes robust to randomly flipping labels in the examples. This suggests that the format and distribution of inputs matter more than the actual input-output mapping for some tasks. The model picks up "this is a classification task with these categories" from the structure, not just "input X maps to output Y." This is consistent with the attention mechanism: the structural pattern (input -> label -> input -> label) is a strong signal even when the labels are wrong.

**GradientCard: "Capability = Vulnerability" callback.** Connect to few-shot jailbreaking from 5.1.3: "You saw in Module 5.1 that few-shot jailbreaking works by putting adversarial examples in the prompt. Now you understand the mechanism: the model's attention treats those examples the same way it treats any examples. ICL does not distinguish between helpful and harmful patterns."

#### 8. Check 2 (Transfer question)
A developer puts 50 examples of a classification task into a prompt to get better accuracy. The model's context window is 4096 tokens, and the 50 examples use 3800 tokens, leaving 296 tokens for the test input and output.

What problems might arise?

Expected answer: (1) The context window is nearly full, leaving almost no room for the test input. (2) With causal masking, only the last few tokens of the test input attend to ALL 50 examples; earlier test tokens attend to fewer. (3) More examples do not always help -- ordering and selection matter. (4) The test input may be truncated if it is long. The limitations of ICL are fundamentally the limitations of the context window and the attention mechanism.

#### 9. Practice (Notebook exercises)
**Notebook:** `notebooks/5-2-1-in-context-learning.ipynb`

**Exercise structure:** 4 exercises, cumulative theme (exploring ICL behavior empirically), but individually completable.

- **Exercise 1 (Guided): Zero-shot vs Few-shot comparison.** Use an LLM API to classify 10 sentiment examples zero-shot and few-shot (3 examples). Compare accuracy. Observe the improvement from examples. Insight: examples in the prompt genuinely change model behavior. Scaffolding: API call pattern provided, student fills in prompts and tallies results.

- **Exercise 2 (Supported): Novel task ICL.** Create a made-up mapping (e.g., word -> its length as a word, or word -> first letter repeated N times). Test the model with 3-5 examples + a novel input. Try mappings of increasing complexity. Insight: ICL can generalize to tasks not in training data, but has limits. Scaffolding: first mapping provided, student creates 2 more.

- **Exercise 3 (Supported): Ordering sensitivity experiment.** Take 5 sentiment examples. Test accuracy on 10 test inputs with 5 different orderings of the same examples. Plot accuracy per ordering. Insight: ICL is sensitive to example order, consistent with attention-based mechanism (recency, position effects). Scaffolding: permutation generation and accuracy computation provided, student designs the experiment and interprets results.

- **Exercise 4 (Independent): ICL vs Finetuning contrast.** Compare few-shot ICL accuracy vs the finetuned classifier from Module 4.4 on the same sentiment task. The finetuned model had gradient descent and thousands of examples. ICL has 3-5 examples and no weight updates. When does ICL win? When does finetuning win? Insight: ICL and finetuning are complementary, not competing. ICL is fast and flexible; finetuning is accurate and robust. Different tools for different constraints.

#### 10. Summarize
Key takeaways:
- In-context learning is the transformer's ability to learn tasks from examples in the prompt without weight updates
- The mechanism is attention: examples create retrieval patterns in the attention weights
- ICL is a base model capability (not from SFT or RLHF), emerging from pretraining on diverse text
- ICL is powerful but fragile: sensitive to example ordering, format, and context window limits
- The prompt is a program; attention is the interpreter

Mental model echo: "Attention is data-dependent. Examples are data. Of course examples influence the output."

#### 11. Next Step
"You now understand that examples in the prompt steer the model's behavior through attention. But putting examples in a prompt is ad hoc -- which examples? In what format? What if you need specific output structure? The next lesson systematizes this: prompt engineering as programming, not conversation."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: concrete example, visual, verbal/analogy, intuitive, symbolic)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 2 new concepts (within limit)
- [x] Every new concept connected to at least one existing concept (attention from 4.2, finetuning from 4.4, few-shot jailbreaking from 5.1.3)
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical conceptual errors, but one critical structural issue (missing notebook) and three improvement-level findings that would meaningfully strengthen the lesson. The lesson's pedagogical design is strong -- the narrative arc, ordering, motivation, connections, and modality coverage are all well executed. The issues are specific and fixable.

### Findings

#### [CRITICAL] -- Notebook missing

**Location:** Practice section (Section 9), references `notebooks/5-2-1-in-context-learning.ipynb`
**Issue:** The lesson links to a Colab notebook at `notebooks/5-2-1-in-context-learning.ipynb`, but this file does not exist in the repository. The planning document specifies 4 exercises (Guided, Supported, Supported, Independent) with detailed descriptions. The lesson component describes all four exercises and links to the notebook. The student clicks "Open in Google Colab" and gets a 404.
**Student impact:** The student reads descriptions of four exercises designed to build empirical intuition for ICL -- the critical practice phase -- and then cannot do any of them. This breaks the lesson's DEVELOPED depth target for both core concepts, since DEVELOPED requires "work through guided examples" and the exercises are where that happens.
**Suggested fix:** Create the notebook `notebooks/5-2-1-in-context-learning.ipynb` with the four exercises specified in the planning document. Exercise 1 (Guided) must use predict-before-run format. Exercises 2-3 (Supported) need `<details>` solution blocks with reasoning before code. Exercise 4 (Independent) needs a solution block. First cell must install dependencies and import everything needed for Colab. Set random seeds for reproducibility.

#### [IMPROVEMENT] -- Misconception 2 ("model understands/decides") not explicitly addressed in the lesson body

**Location:** Section 5 (Explain Part 2) and Section 7 (Elaborate)
**Issue:** The planning document identifies 5 misconceptions. Misconception 2 -- "ICL works because the model 'understands' the task description and 'decides' to follow the examples" -- is planned to be addressed in Section 5 via the label-flipping finding (Min et al. 2022). In the built lesson, the label-flipping finding appears in Section 7 (Elaborate) rather than Section 5, which is fine as a structural choice. However, the misconception is never stated as something the student might believe and then disproven. The label-flipping finding is presented as an interesting observation ("Min et al. showed that ICL performance is sometimes robust to randomly flipping labels") rather than as a misconception-busting move ("You might think the model reads the examples and 'understands' the task. If that were true, flipping all the labels should completely reverse its behavior. It does not."). The student could read the label-flipping paragraph without connecting it to their own anthropomorphic intuition.
**Student impact:** The student may continue to think of ICL in anthropomorphic terms ("the model figured out the task from the examples") because the lesson never explicitly names and challenges that framing. The ordering sensitivity addresses the "not comprehension" angle, but from a different direction (fragility rather than the label-insensitivity angle).
**Suggested fix:** Before presenting the label-flipping finding, add one sentence that names the misconception: "A natural way to think about ICL is that the model 'reads' the examples, 'figures out' the pattern, and 'applies' it. If that were true, what would happen if you flipped the labels?" Then present the finding. This makes the label-flipping result a direct counterexample to a specific wrong model, rather than an interesting tangential fact.

#### [IMPROVEMENT] -- Planned positive example 3 (same test input, different examples = different tasks) appears only in Check 1, not as a teaching example

**Location:** Section 6 (Check 1) vs the planned "Same test input, different examples yielding different outputs" positive example
**Issue:** The planning document lists three positive examples: (1) sentiment classification, (2) novel symbol mapping, (3) same test input with different example sets yielding different outputs. The third example is planned as a positive (stretch) example that "directly disproves 'weights update during ICL'" and "makes the mechanism vivid: the prompt IS the program." In the built lesson, this example appears only as Check 1 (a predict-and-verify exercise). The student predicts and then reveals the answer. This is valuable, but the pedagogical function is different: a checkpoint tests whether the student already grasps the idea, while a positive example teaches it. The student encounters this idea for the first time in a checkpoint rather than in a teaching context. If the student gets the prediction wrong (which is entirely possible at this point -- they have only seen the attention mechanism once), the reveal section provides the answer but does not re-teach the mechanism in this new context.
**Student impact:** The student who gets Check 1 wrong sees the answer but may not fully internalize why "different examples, same weights, different behavior" proves the learning is in activations rather than parameters. The explain section (Section 5) establishes the attention mechanism, and Check 1 immediately tests transfer to a new scenario. For a STRETCH lesson, this may be too fast. The student has not yet had a chance to consolidate the mechanism before being tested on it.
**Suggested fix:** This is a judgment call. The current structure (explain mechanism, immediately test transfer) is aggressive but defensible for a STRETCH lesson. Two options: (a) Add one brief paragraph after the "of course" beat in Section 5 that explicitly walks through the "same input, different examples" scenario as a teaching example before Check 1 tests it. This way Check 1 confirms rather than teaches. (b) Accept the current structure but enhance the Check 1 reveal to include a brief mechanism re-explanation (e.g., "The Q vectors from 'The weather is beautiful' are the same in both prompts. But the K and V vectors are different because the examples are different. Different K/V means different attention weights, different blended output -- different task."). Option (b) is lower-effort and likely sufficient.

#### [IMPROVEMENT] -- The "Transformers as learning algorithms" theoretical perspective is not MENTIONED as planned

**Location:** Throughout the lesson; planned target depth is MENTIONED
**Issue:** The planning document lists "Transformers as learning algorithms" theoretical perspective at target depth MENTIONED. The scope boundaries state this perspective should be "MENTIONED at most, not developed." The references section includes von Oswald et al. (2023) "Transformers Learn In-Context by Gradient Descent" with the note "Advanced reading -- mentioned in the lesson but not developed." However, searching the lesson body, this perspective is never mentioned in the prose. The reference appears in the ReferencesBlock, but the lesson text never introduces the idea that transformers might implement something like gradient descent in their forward pass. The reference hangs without any connection to the lesson content.
**Student impact:** Minimal -- the student simply does not encounter this idea. But the reference to a paper about "transformers learning by gradient descent" could be confusing without any lesson context. The student might click through to the paper and be confused about its relevance. More importantly, the planning document explicitly committed to MENTIONED depth, and the built lesson does not deliver it.
**Suggested fix:** Add one sentence in the "of course" beat or in the summary section that mentions this perspective: "Some researchers have gone further, showing that transformer attention can implement something resembling gradient descent within the forward pass (von Oswald et al., 2023) -- but that theoretical framework is beyond our scope here." This satisfies MENTIONED depth without developing the concept.

#### [POLISH] -- Attention diagram visual could be more self-explanatory

**Location:** ICLAttentionDiagram component (Section 5)
**Issue:** The diagram uses attention lines from the test input "slow" to the example tokens. All attention lines originate from the same x-position (480, the test input), making them horizontal-ish rather than the curved arcs typical of attention visualizations. The attention weights shown (25%, 15%, 15%, 35%) sum to 90%, not 100%. While the subtitle explains this is a simplified view, the student who traces the numbers and tries to verify against the softmax formula (which sums to 1) will notice the 10% discrepancy. The student has been trained to trace numbers carefully in the attention lessons.
**Student impact:** Minor -- the diagram communicates the core idea effectively. A detail-oriented student might momentarily wonder where the remaining 10% of attention goes (presumably to the arrow tokens or other positions not shown). Not confusing enough to derail understanding.
**Suggested fix:** Either adjust weights to sum to 100% (e.g., 25%, 15%, 20%, 40%) or add a brief note that these are approximate weights and the remaining attention goes to other tokens in the context (arrows, formatting tokens). The latter is more honest.

#### [POLISH] -- Aside InsightBlock in Recap section front-runs the lesson's "of course" moment

**Location:** Section 2 (Recap), aside InsightBlock titled "The Key Connection"
**Issue:** The aside text reads: "Attention weights are data-dependent. If examples are in the input, they are part of the data. This single fact explains everything in this lesson." This is the lesson's core "of course" insight, delivered in the aside of the Recap section -- before the puzzle is even presented. The "of course" beat in Section 5 is designed to collapse the surprise ("You already knew this..."), but if the student reads the aside first, the surprise is pre-collapsed. The lesson's narrative arc is: puzzle (Section 3) -> mechanism (Section 5) -> "of course" resolution. The aside in Section 2 gives away the resolution before the puzzle.
**Student impact:** For a student who reads asides carefully, the surprise of the "how does it work without weight updates?" puzzle in Section 3 is diminished because they already read the answer in the Section 2 aside. The "of course" moment in Section 5 loses its punch. For a student who skims asides, no impact.
**Suggested fix:** Move this InsightBlock to the aside of Section 5 (alongside the "Prompt Is a Program" and "Software Analogy" blocks), or change the Recap aside to something that reconnects to the attention mechanism without giving away its application to ICL. For example: "These two ideas -- attention as data-dependent computation and finetuning as weight updates -- are about to collide in a way you do not expect."

### Review Notes

**What works well:**
- The narrative arc is excellent. The progression from puzzle ("how?") to mechanism (attention) to "of course" resolution to boundaries (what ICL is not) follows the motivation rule and ordering rules precisely. Problem before solution throughout.
- Connection density is high. Every new idea is grounded in something the student already knows. The callback to finetuning for classification, the callback to SFT teaches format not knowledge, the callback to few-shot jailbreaking -- all land precisely because the student has these at DEVELOPED or INTRODUCED depth.
- The ComparisonRow (finetuning vs ICL) is effective. Side-by-side contrast makes the surprise vivid.
- Both checkpoints are well-designed. Check 1 tests the core mechanism in a transfer context. Check 2 tests application of limitations to a realistic scenario.
- Modality coverage is strong: concrete examples (sentiment, novel mapping), visual (SVG diagram), verbal/analogy (prompt as program, config vs recompile), intuitive ("of course" beat), symbolic (attention formula). All five planned modalities are present and genuine.
- Scope discipline is tight. The lesson does not drift into prompt engineering, CoT, or RAG.
- Em dash formatting is correct throughout (no spaced em dashes in rendered content).
- Cursor styles are correct on interactive elements (both `<summary>` elements have `cursor-pointer`).

**Patterns to watch:**
- The notebook gap is a recurring pattern worth monitoring. If lessons are being built before notebooks, the practice phase is missing. Consider creating notebook stubs during lesson building to prevent this.
- The aside front-running issue is subtle but worth being aware of in future lessons. Asides should deepen or contextualize, not spoil upcoming narrative beats.

---

## Review -- 2026-02-14 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

All iteration 1 findings have been correctly resolved. The lesson is strong. One improvement-level finding in the notebook and one polish-level finding remain.

### Iteration 1 Fix Verification

All five iteration 1 findings were verified as resolved:

1. **[CRITICAL] Notebook missing:** Resolved. Notebook exists at `notebooks/5-2-1-in-context-learning.ipynb` with 4 exercises matching the planning document's specifications. Exercises follow the correct scaffolding progression (Guided, Supported, Supported, Independent). Solutions include reasoning before code. Setup cell is self-contained for Colab with random seeds set.

2. **[IMPROVEMENT] Misconception 2 not explicitly named:** Resolved. The label-flipping paragraph in Section 7 now begins: "A natural way to think about ICL is that the model 'reads' the examples, 'figures out' the pattern, and 'applies' it -- as if it understands the task. If that were true, what would happen if you flipped the labels?" The misconception is explicitly named before the evidence. The fix reads naturally and does not feel bolted on.

3. **[IMPROVEMENT] Check 1 reveal lacks mechanism re-explanation:** Resolved. The Check 1 reveal now includes a full Q/K/V trace: "the Q vectors from 'The weather is beautiful' are the same in both prompts. But the K and V vectors are different because the examples are different. Different K means different attention weights. Different V means different blended output. Different task." It also adds: "If the weights had been updated by the examples, the model's behavior on an empty prompt would also change. It does not -- the weights are frozen. The 'learning' is in the activations (attention patterns), not the parameters." This addresses misconception 5 directly in the reveal and provides the re-teaching that iteration 1 recommended.

4. **[IMPROVEMENT] "Transformers as learning algorithms" not MENTIONED:** Resolved. After the "of course" beat in Section 5, the lesson now includes: "Some researchers have taken this further, showing that transformer attention can implement something resembling gradient descent within the forward pass (von Oswald et al., 2023) -- but that theoretical framework is beyond our scope here. What matters for us is the practical mechanism." This satisfies MENTIONED depth exactly as planned, and connects to the von Oswald reference in the ReferencesBlock.

5. **[POLISH] Attention diagram weights sum to 90%:** Resolved. Weights are now 25%, 15%, 20%, 40% = 100%.

6. **[POLISH] Recap aside front-runs the "of course" moment:** Resolved. The aside is now titled "Two Ideas, One Collision" with text: "These two ideas -- attention as data-dependent computation and finetuning as weight updates -- are about to collide in a way you do not expect." This creates anticipation without giving away the mechanism. The "of course" moment in Section 5 retains its full impact.

### Findings

#### [IMPROVEMENT] -- Exercise 3 TODO code is pre-filled, undermining Supported scaffolding

**Location:** Notebook `notebooks/5-2-1-in-context-learning.ipynb`, cell-17 (Exercise 3: Ordering Sensitivity)
**Issue:** The cell contains TODO comments ("YOUR CODE HERE (4-6 lines)" and "YOUR CODE HERE (1 line)") indicating the student should write the code, but the solution code is already present in the cell directly below the TODO comments. The student does not need to write anything -- they just run the cell. The solution block in cell-19 contains the same code, making both the TODOs and the solution redundant. By contrast, Exercise 2 correctly has empty placeholder dicts that the student must fill in, and Exercise 4 correctly has an empty code cell.
**Student impact:** Exercise 3 functions as a Guided exercise (predict-then-run) rather than the planned Supported exercise (write code with hints). The student misses the practice of writing the classification loop themselves, which is the core coding pattern for ICL experimentation. Since Exercises 2 and 4 do require the student to write code, the inconsistency is also noticeable.
**Suggested fix:** Remove the pre-filled code from cell-17, leaving only the TODO comments and the scaffolding code around them (the `for i, order in enumerate(orderings)` loop, the `reordered_examples` line, and the `ordering_results.append(...)` block). The student should write the inner loop (classify all test examples) and the accuracy computation. The existing solution in cell-19 then serves its intended purpose.

#### [POLISH] -- Notebook imports `itertools` and `json` but never uses them

**Location:** Notebook `notebooks/5-2-1-in-context-learning.ipynb`, cell-1 (Setup)
**Issue:** The setup cell imports `itertools` and `json`, but neither is used anywhere in the notebook. The `itertools` import was likely intended for generating permutations in Exercise 3 (which instead uses a manual shuffle approach), and `json` was likely intended for API response parsing (which is handled by the OpenAI SDK).
**Student impact:** Negligible. An observant student might wonder what these imports are for, but it does not affect understanding or functionality.
**Suggested fix:** Remove `import json` and `import itertools` from the setup cell.

### Review Notes

**What works well:**
- All iteration 1 fixes are correctly implemented. None of the fixes introduced new issues or disrupted the lesson's flow. The von Oswald mention is placed naturally. The misconception 2 naming reads as if it was always there. The Check 1 reveal enhancement is the strongest fix -- it transforms the checkpoint from a test-only moment into a teaching-then-confirming moment.
- The notebook is well-structured overall. Exercise 1 (Guided) with predict-before-run is effective. Exercise 2 (Supported) with the progressive mapping complexity is well-designed. Exercise 4 (Independent) has a clear specification and a thorough solution. The key takeaways section mirrors the lesson.
- The lesson component itself is in excellent shape. Narrative arc, modality coverage, misconception handling, scope discipline, and connection density are all strong. The diagram weights summing to 100% and the non-spoiling recap aside are small details that matter for a student who pays attention.

**The one remaining issue** is a notebook scaffolding error (Exercise 3 pre-filled code), not a lesson content issue. Once this is fixed, the lesson and notebook form a cohesive, pedagogically sound unit.

---

## Review -- 2026-02-14 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

Both iteration 2 findings have been correctly resolved. No new issues were introduced by the fixes. The lesson component and notebook together form a cohesive, pedagogically sound unit that meets all planned specifications.

### Iteration 2 Fix Verification

Both iteration 2 findings were verified as resolved:

1. **[IMPROVEMENT] Exercise 3 TODO code pre-filled:** Resolved. Cell-17 now contains only the TODO comments ("YOUR CODE HERE (4-6 lines)" and "YOUR CODE HERE (1 line)") with blank lines where the student must write code. The surrounding scaffolding is intact -- the `for i, order in enumerate(orderings)` loop, the `reordered_examples` line, and the `ordering_results.append(...)` block are all present, giving the student structure without giving them the answer. The solution in cell-19 serves its intended purpose as a reference after attempting. Exercise 3 is now a genuine Supported exercise, consistent with Exercises 2 and 4 which also require the student to write code.

2. **[POLISH] Unused imports (`json`, `itertools`):** Resolved. The setup cell (cell-1) now imports only `os`, `textwrap`, `random`, `OpenAI`, `matplotlib.pyplot`, and `numpy`. No unused imports remain. Clean and self-contained.

Neither fix introduced new issues. The notebook runs correctly (assuming the student fills in the TODOs), and the scaffolding progression (Guided, Supported, Supported, Independent) is now consistently implemented across all four exercises.

### Full Review (Steps 1-8)

#### Step 1: Student Simulation

**Student's entering state:** DEVELOPED understanding of attention as data-dependent weighted average, QK^T as learned relevance, full attention formula, multi-head attention, autoregressive generation, language modeling as next-token prediction, SFT teaches format not knowledge, causal masking. From Module 5.1: constitutional AI, design space axes, red teaming (including few-shot jailbreaking at INTRODUCED), evaluating LLMs. The student has never been taught what in-context learning is or why it works.

**Sequential read-through:** No points of confusion, unexplained terminology, or cognitive leaps. Every concept is grounded in something the student already knows. The progression from puzzle ("how does it work without weight updates?") to mechanism (attention) to "of course" resolution to boundaries (what ICL is not) reads naturally. The two checkpoints test at appropriate points -- Check 1 after the mechanism is explained, Check 2 after limitations are developed.

#### Step 2: Plan Alignment

All planned elements are present and correctly implemented:
- Target concepts at planned depths (ICL phenomenon DEVELOPED, attention mechanism DEVELOPED, limitations INTRODUCED, "transformers as learning algorithms" MENTIONED)
- All 5 misconceptions addressed at planned locations
- All 4 examples (3 positive + 1 negative) present
- All 5 modalities present (concrete example, visual, verbal/analogy, intuitive, symbolic)
- Scope boundaries respected (no drift into prompt engineering, CoT, or RAG)
- Lesson outline matches the planned 11-section structure
- Notebook has 4 exercises with correct scaffolding progression (Guided, Supported, Supported, Independent)

No undocumented deviations.

#### Step 3: Pedagogical Principles

- **Motivation Rule:** Problem stated before solution throughout. "We said no weights change. So where does the task-specific behavior come from?" precedes the attention explanation. The GPT-3 discovery section frames "why it is surprising" before "how it works."
- **Modality Rule:** 5 modalities for the core concept -- concrete examples (sentiment, novel mapping, same-input-different-task), visual (SVG attention diagram), verbal/analogy ("prompt is a program," "config vs recompile"), intuitive ("of course" beat), symbolic (attention formula box). All genuine, not rephrased versions of each other.
- **Example Rules:** 3 positive examples (sentiment classification, novel mapping, same input with different examples) + 1 negative example (ordering sensitivity). First example is the simplest (familiar sentiment task). Second confirms generalization (novel mapping). Third proves mechanism (same weights, different behavior). Negative example defines boundaries (ordering matters, so it is not comprehension).
- **Misconception Rule:** 5 misconceptions identified and addressed. Each has a concrete counterexample. Misconception 5 ("weights update") is addressed earliest (Hook, Section 3). Misconception 4 ("requires instruction tuning") addressed in Section 4. Misconception 2 ("model understands") explicitly named and addressed in Section 7 with label-flipping evidence. Misconception 1 ("just pattern matching") addressed in Section 7 with novel mapping. Misconception 3 ("more examples = better") addressed in Section 7 with ordering sensitivity.
- **Ordering Rules:** Concrete before abstract (demo before mechanism). Problem before solution (puzzle before attention explanation). Parts before whole (individual attention components before integrated view). Simple before complex (sentiment before novel mapping before limitations).
- **Load Rule:** 2 genuinely new concepts. Within limit.
- **Connection Rule:** Every new concept explicitly linked to prior knowledge. ICL linked to attention (4.2), finetuning (4.4.1), SFT (4.4.2), few-shot jailbreaking (5.1.3).
- **Reinforcement Rule:** Attention concepts from 4.2 are more than 3 lessons ago, but the recap in Section 2 explicitly re-activates them. Appropriate.
- **Interaction Design Rule:** Both `<summary>` elements have `cursor-pointer` class. No other interactive elements (no sliders, draggable elements, or custom interactions). Correct.
- **Writing Style Rule:** Em dashes are unspaced throughout (`word—word` via `&mdash;`). Checked in lesson prose and aside text. No violations.

#### Step 4: Examples Evaluation

All examples are concrete, well-placed, and serve stated purposes:
- **Sentiment classification (positive 1):** Immediately graspable, uses familiar task from 4.4.1. Creates direct contrast with finetuning approach.
- **Novel symbol mapping (positive 2):** "sdag" -> "happy" is genuinely novel. Effectively disproves memorization misconception. Placed in Section 7 after mechanism is established.
- **Same input, different examples (positive 3):** Appears in Check 1 with full mechanism re-explanation in the reveal. The Q/K/V trace and "weights are frozen" proof make the reveal a teaching moment, not just an answer.
- **Ordering sensitivity (negative):** Concrete claim (20-30 percentage points), connected to causal masking mechanism. Effectively defines the boundary between "powerful" and "fragile."

No missing examples detected. No abstract statements without grounding.

#### Step 5: Narrative and Flow

- **Hook:** Compelling. "The model's parameters did not change. No optimizer ran. No gradients were computed. And yet it performed a task it was never explicitly trained to do. How?" -- strong question that leverages the student's existing mental model.
- **Flow:** Sections connect via explicit transitions. "We said no weights change. So where does the task-specific behavior come from?" (Section 4 to 5). "The sentiment example was familiar -- you might think the model just memorized..." (Section 5 to 7).
- **Pacing:** Appropriate for STRETCH. The densest section (Section 5, attention mechanism) is balanced by the "of course" beat that collapses complexity. No section too thin or too dense.
- **Conclusion:** Summary captures all 5 key mental models. Next step clearly motivates the transition to prompt engineering.

#### Step 6: Notebook Evaluation

- **Existence and coverage:** Notebook exists, 4 exercises matching planning doc specifications.
- **Scaffolding progression:** Exercise 1 Guided (predict-before-run), Exercise 2 Supported (first mapping provided, student creates 2 more with TODO markers and empty dicts), Exercise 3 Supported (permutation generation provided, student writes classification loop with TODO markers and blank spaces), Exercise 4 Independent (empty code cell with specification). Correct progression from more to less support. At most one Independent exercise, at the end.
- **Solution quality:** Exercises 2, 3, 4 all have `<details>` solution blocks. Solutions include reasoning before code (e.g., Exercise 3 solution explains "Why ordering matters from the attention perspective" before showing code). Solutions mention common findings and alternative approaches (Exercise 2 provides two alternative mapping examples).
- **Self-contained setup:** First cell installs `openai`, imports all dependencies, sets random seeds, verifies API connection. Would run in Colab without local setup. No references to local files.
- **Concept alignment:** Notebook uses same terminology as lesson (Q/K/V, attention-based mechanism, causal masking, recency bias). Does not introduce new concepts beyond what the lesson teaches.
- **Exercise 3 fix verified:** TODO markers present, blank spaces for student code, no pre-filled solution. Scaffolding code (loop structure, result collection) intact.
- **Setup cell fix verified:** No unused imports.

#### Step 7: No findings.

After thorough review across all 8 steps, no Critical, Improvement, or Polish findings remain. The iteration 2 fixes were correctly implemented and did not introduce new issues.

### Review Notes

**What works well:**
- The lesson's narrative arc is the strongest element. The progression from puzzle to mechanism to "of course" resolution to boundaries is textbook problem-before-solution pedagogy. The student is motivated at each stage.
- Connection density is exceptionally high. Every new idea is anchored in something the student already knows at DEVELOPED depth. The callbacks to finetuning (4.4.1), attention (4.2), SFT (4.4.2), and few-shot jailbreaking (5.1.3) are precise and well-placed.
- The notebook exercises form a coherent arc (basic observation, boundary testing, fragility measurement, comparative analysis) that mirrors the lesson's progression. Each exercise builds a more nuanced understanding of ICL.
- The iteration process worked as designed. Iteration 1 caught 1 critical (notebook missing), 3 improvement, and 2 polish findings. Iteration 2 caught 1 improvement and 1 polish remaining after fixes. Iteration 3 confirms all fixes are correct and the lesson is ready to ship.
- Scope discipline is tight throughout. The lesson never drifts into prompt engineering, chain-of-thought, or RAG -- all explicitly deferred to later lessons.

**Ready for Phase 5 (Record).** The lesson and notebook are complete and can be documented in the module record.
