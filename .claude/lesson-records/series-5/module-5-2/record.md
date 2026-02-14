# Module 5.2: Reasoning & In-Context Learning -- Record

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| In-context learning as gradient-free task learning from examples in the prompt | DEVELOPED | in-context-learning | A transformer trained on next-token prediction can classify, translate, or follow novel patterns simply by placing examples in the prompt. No gradients, no optimizer, no parameter changes. The "learning" happens in the activations (attention patterns), not the parameters. GPT-3 demonstrated this as a BASE MODEL capability -- pre-SFT, pre-RLHF. |
| Attention as the mechanism enabling ICL (Q from test input, K/V from context including examples) | DEVELOPED | in-context-learning | The test input's query vectors compute dot products with key vectors of all tokens in the context, including example inputs and outputs. High Q*K scores against structurally similar example inputs route attention to example output values. Same formula (softmax(QK^T / sqrt(d_k)) V), longer context. No new mechanism -- the existing attention mechanism applied to a prompt containing examples. |
| Zero-shot vs few-shot prompting | DEVELOPED | in-context-learning | Zero-shot: task instruction only, no examples. Few-shot: task instruction + input-output examples before the test input. Both are forms of ICL. Terminology introduced via the GPT-3 discovery. |
| ICL limitations: ordering sensitivity, format fragility, context window constraints | INTRODUCED | in-context-learning | Accuracy swings 20-30 percentage points from reordering the same examples. More examples do not always help -- ordering and selection matter more than quantity. Context window limits how many examples can be provided. Causal masking creates asymmetric attention (later tokens attend to more context). ICL is powerful but fragile. |
| ICL as not memorized pattern retrieval (novel mapping capability) | INTRODUCED | in-context-learning | Model can follow made-up mappings ("sdag" -> "happy") that could not have appeared in training data. Disproves "just template matching" misconception. But ordering sensitivity shows ICL is also not human-like comprehension. Something in between: attention-based computation that generalizes within the scope of a single forward pass. |
| Label-flipping robustness (Min et al. 2022) | INTRODUCED | in-context-learning | ICL performance sometimes survives randomly flipping labels in examples. Format and input distribution matter more than the actual input-output mapping for many tasks. The model picks up "this is a classification task with these categories" from structure, not just from the mapping. Evidence against "model understands/decides" anthropomorphic framing. |
| ICL as a capability that is also a vulnerability (callback to 5.1.3) | INTRODUCED | in-context-learning | Connected to few-shot jailbreaking from red-teaming-and-adversarial-evaluation. The same attention mechanism that lets the model learn sentiment classification from 3 examples also lets it learn harmful behaviors from 3 adversarial examples. ICL does not distinguish between helpful and harmful patterns. Every capability is also a vulnerability. |
| "Transformers as learning algorithms" theoretical perspective (von Oswald et al. 2023) | MENTIONED | in-context-learning | Transformer attention can implement something resembling gradient descent within the forward pass. Mentioned as advanced theoretical work but not developed. Referenced in lesson prose and ReferencesBlock. |

## Per-Lesson Summaries

### Lesson 1: In-Context Learning (in-context-learning)

**Concepts taught:**
- In-context learning as gradient-free task learning from examples in the prompt (DEVELOPED)
- Attention as the mechanism enabling ICL -- Q from test input, K/V from entire context including examples (DEVELOPED)
- Zero-shot vs few-shot prompting (DEVELOPED)
- ICL limitations: ordering sensitivity, format fragility, context window constraints (INTRODUCED)
- ICL as not memorized pattern retrieval -- novel mapping capability (INTRODUCED)
- Label-flipping robustness as evidence against anthropomorphic "understands" framing (INTRODUCED)
- ICL as capability that is also a vulnerability (INTRODUCED)
- "Transformers as learning algorithms" theoretical perspective (MENTIONED)

**Mental models established:**
- "The prompt is a program; attention is the interpreter" -- different examples, different behavior; same weights, different programs. Each prompt configures the model for a different task through attention over the context. Extended to software analogy: the prompt is configuration, not code; the model's weights are the compiled binary; you change behavior by changing the config file (prompt), not recompiling (retraining).
- "Attention is data-dependent. Examples are data. Of course examples influence the output." -- the "of course" moment that collapses the surprise. ICL is not a new mechanism; it is the same attention the student already understands, operating on a longer context that happens to contain examples of a task.
- "Between retrieval and comprehension" -- ICL is not memorized pattern matching (novel mappings work) and not human-like understanding (ordering matters). It is attention-based computation -- powerful but constrained by what a single forward pass can compute.
- "Capability = vulnerability" (extended from 5.1.3) -- the attention mechanism that enables task learning from examples also enables learning harmful behaviors from adversarial examples.

**Analogies used:**
- Prompt as configuration file vs code (maps to software engineering background: change config, not recompile)
- "The prompt is a program, attention is the interpreter" (examples are data, format is the instruction set)
- Contrast with finetuning: same task (sentiment classification), dramatically different approach (weight updates vs prompt examples)

**How concepts were taught:**
- **Recap:** Brief reconnection to attention as data-dependent computation (4.2) and finetuning for classification (4.4.1). "Hold both in mind. This lesson is about what happens when you skip the second one entirely."
- **Hook (puzzle):** Sentiment classification few-shot prompt -- same task as finetuning lesson, no weight updates. GradientCard: "The model's parameters did not change. No optimizer ran. No gradients were computed. And yet it performed a task it was never explicitly trained to do. How?"
- **GPT-3 discovery:** Historical grounding. Zero-shot vs few-shot terminology. Critical point: base model capability, not instruction tuning. Callback to "SFT teaches format, not knowledge." ComparisonRow: finetuning approach (5 steps) vs ICL approach (5 steps).
- **Attention mechanism:** Step-by-step walkthrough of Q/K/V during few-shot prompting. ICLAttentionDiagram (inline SVG): test input "slow" attends to example inputs (structural matching via Q*K) and example outputs (answer retrieval via V blending). Attention weights shown (25%, 15%, 20%, 40% = 100%). Connected to the existing formula: softmax(QK^T / sqrt(d_k)) V -- "The formula has not changed. The context has." The "of course" beat: "You already knew this. Attention weights are data-dependent. If examples are in the context, they are part of the data." Von Oswald et al. (2023) MENTIONED in one sentence.
- **Check 1 (predict-and-verify):** Same test input ("The weather is beautiful") with sentiment examples (Prompt A) vs translation examples (Prompt B). Student predicts outputs. Reveal includes full Q/K/V mechanism trace and proof that weights are frozen (empty prompt behavior unchanged).
- **What ICL Is and Is Not:** Novel mapping example ("sdag" -> "happy") disproves pure memorization. Ordering sensitivity (20-30 point accuracy swings) disproves human-like comprehension. Label-flipping finding (Min et al. 2022) explicitly named as misconception-buster: "A natural way to think about ICL is that the model 'reads' the examples, 'figures out' the pattern, and 'applies' it. If that were true, what would happen if you flipped the labels?" GradientCard: "Capability = Vulnerability" connecting to few-shot jailbreaking from 5.1.3.
- **Check 2 (transfer question):** Developer puts 50 examples in a 4096-token context window. Student identifies problems (context nearly full, causal masking asymmetry, ordering effects amplified, diminishing returns).

**Visual elements:**
- ICLAttentionDiagram: Inline SVG showing test input "slow" attending to example tokens. Color-coded: indigo for example inputs, green for example outputs, amber for test input, red for test output. Attention lines with varying thickness (weight 25%, 15%, 20%, 40%). Legend with Q*K (input matching) and V blending (output retrieval). Subtitle and key insight annotation.
- ComparisonRow: Finetuning approach (5 items) vs In-Context Learning (5 items)
- GradientCards for the puzzle (orange), the "of course" moment (violet), novel mapping proof (blue), ordering sensitivity (rose), capability = vulnerability (amber), both checkpoints (emerald)

**What is NOT covered:**
- Systematic prompt engineering techniques (Lesson 2: prompt-engineering)
- Chain-of-thought or step-by-step reasoning (Lesson 3)
- Retrieval-augmented generation (Lesson 2)
- Implementing ICL in code beyond notebook exercises
- Theoretical framework for why ICL emerges during pretraining (MENTIONED, not developed)
- Comparing ICL performance to finetuning quantitatively
- Role prompting or system prompts (Lesson 2)

**Notebook:** `notebooks/5-2-1-in-context-learning.ipynb` (4 exercises)
- Exercise 1 (Guided): Zero-shot vs few-shot comparison. Use an LLM API to classify 10 sentiment examples zero-shot and few-shot (3 examples). Compare accuracy. Predict-before-run format. Insight: examples in the prompt genuinely change model behavior.
- Exercise 2 (Supported): Novel task ICL. Create made-up mappings the model has never seen. Test with 3-5 examples + novel input. Try mappings of increasing complexity. Insight: ICL can generalize to tasks not in training data, but has limits.
- Exercise 3 (Supported): Ordering sensitivity experiment. Take 5 sentiment examples, test accuracy on 10 test inputs with 5 different orderings. Plot accuracy per ordering. Insight: ICL is sensitive to example order, consistent with attention-based mechanism (recency, position effects).
- Exercise 4 (Independent): ICL vs finetuning contrast. Compare few-shot ICL accuracy vs a finetuned classifier on the same sentiment task. Insight: ICL and finetuning are complementary -- ICL is fast and flexible, finetuning is accurate and robust.

**Review:** Passed at iteration 3/3. Iteration 1 had 1 critical (notebook missing), 3 improvement (misconception 2 not explicitly named, Check 1 reveal lacks mechanism re-explanation, "transformers as learning algorithms" not MENTIONED), and 2 polish findings (attention weights sum to 90%, recap aside front-runs "of course" moment). All resolved. Iteration 2 had 1 improvement (Exercise 3 TODO code pre-filled) and 1 polish (unused imports in notebook). Both resolved. Iteration 3: 0 findings across all categories. Clean pass.
