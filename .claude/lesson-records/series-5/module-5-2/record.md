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
| Prompt engineering as structured programming (systematic prompt construction) | DEVELOPED | prompt-engineering | The prompt is a composable program with identifiable components (system/role, task instruction, format specification, few-shot examples, context/retrieved docs, user input), each shaping the attention pattern in a specific way. Engineering discipline, not conversational phrasing or magic incantations. Extended "the prompt is a program" mental model from ICL into a systematic methodology. |
| Format specification and output constraints | DEVELOPED | prompt-engineering | Explicit output schemas (JSON, markdown, structured text) constrain the output distribution through two mechanisms: (1) format tokens in the prompt create structural anchors for attention, and (2) autoregressive generation maintains format consistency once the first format token is generated (the first curly brace constrains all subsequent tokens). |
| Role and system prompting as attention bias | DEVELOPED | prompt-engineering | Role/system prompt tokens bias the attention pattern toward domain-relevant features of the input. "You are a security engineer" makes security-related code patterns more salient to attention. Critical distinction: role prompts shape focus and style, NOT knowledge. Connected to "SFT teaches format, not knowledge" from instruction-tuning (4.4.2) -- role prompts are the same principle at inference time. Concrete negative example: expert role on unknown facts produces more confident wrong answers. |
| Few-shot example selection principles (diversity, format consistency, difficulty calibration) | DEVELOPED | prompt-engineering | Three principles for systematic example selection: (1) diversity over quantity -- examples should cover the input space, not repeat one pattern; 3 diverse examples outperform 10 similar ones; (2) format consistency -- all examples should use the exact output format desired, because format is a stronger signal than labels (callback to label-flipping from ICL); (3) difficulty calibration -- examples should match real task difficulty. Grounded in attention: diverse examples create richer K/V patterns. |
| Retrieval-Augmented Generation (RAG) as context augmentation for attention | INTRODUCED | prompt-engineering | RAG is a prompt engineering pattern, not a model feature: retrieve relevant documents, place them in the prompt as context, and the model's attention attends to them via the same mechanism as few-shot examples. Software analogy: dependency injection (provide data at runtime instead of hardcoding in parameters). Two-step process: retrieve (search, outside the model) then place in prompt (attention, inside the model). |
| Context stuffing failure (retrieval quality over quantity) | INTRODUCED | prompt-engineering | Irrelevant documents in the context dilute attention -- softmax distributes weight across all tokens, so irrelevant tokens "steal" attention from relevant ones. 5 relevant documents outperform 5 relevant + 15 irrelevant. Connected to ICL finding that 50 examples in a 4096-token window caused problems. |
| RAG does not eliminate hallucination | INTRODUCED | prompt-engineering | The model can hallucinate despite having correct information in the retrieved context. It may blend retrieved information with parametric priors from pretraining. Concrete example: document says "$4.2M revenue, 3% decline" but model outputs "$5M+ revenue, strong growth." RAG reduces hallucination by making information available to attention; it does not eliminate it because attention is not guaranteed to weight retrieved passages over parametric knowledge. |
| No universal prompt template (principles transfer, specific prompts do not) | INTRODUCED | prompt-engineering | Different models and tasks benefit from different technique combinations. A prompt optimized for one model version may perform worse on the next. The principles (structure, specificity, format consistency) transfer across models; specific templates do not. Engineering judgment is knowing which tools to use for which job. |
| Intermediate tokens as computation (the core CoT mechanism) | DEVELOPED | chain-of-thought | Each generated intermediate token triggers another forward pass through the same N transformer blocks. The autoregressive feedback loop is not just a generation mechanism -- it is a computational amplifier. More tokens = more forward passes = more computation per problem. The model does not "think harder"; it runs more forward passes, each building on the context of previous ones. Scratchpad analogy: intermediate tokens are external memory that decompose a problem exceeding single-forward-pass capacity. |
| Fixed computation budget per forward pass (architectural constraint CoT addresses) | DEVELOPED | chain-of-thought | Every token prediction runs through the same N transformer blocks (N=12 for GPT-2, N=96 for GPT-3), regardless of problem difficulty. The model cannot allocate more compute to harder problems within a single forward pass. There is no "difficulty knob." CoT is the mechanism for exceeding this fixed budget. Connected to the generate() method the student implemented in building-nanogpt -- each loop iteration is one forward pass with one fixed computation budget. |
| When CoT helps vs does not (computational complexity criterion) | DEVELOPED | chain-of-thought | CoT helps when the problem requires more computation than one forward pass provides: multi-step arithmetic, word problems, logical reasoning, constraint satisfaction. CoT does NOT help (and can hurt) on tasks that fit within a single pass: factual recall, sentiment classification, simple text completion. The criterion is computational complexity, not difficulty in the human sense. Adding "let's think step by step" to "What is the capital of France?" wastes context and can introduce overthinking errors. |
| Zero-shot CoT vs few-shot CoT | DEVELOPED | chain-of-thought | Zero-shot CoT: add "Let's think step by step" (Kojima et al. 2022) -- a prompt engineering technique (format instruction that triggers reasoning-style tokens from pretraining data). Few-shot CoT: provide examples with reasoning chains (Wei et al. 2022, the original CoT paper) -- an ICL technique (examples with reasoning chains as the "program"). Both work for the same mechanistic reason: causing the model to generate intermediate tokens that provide additional forward passes. |
| Process supervision vs outcome supervision | INTRODUCED | chain-of-thought | Outcome supervision evaluates only the final answer -- a chain with correct reasoning and a calculation error at the last step is marked the same as completely wrong logic. Process supervision evaluates each step individually -- is step 1 correct? Is step 2 a valid follow-up? Harder but more informative. Matters because models can reach correct answers through flawed reasoning (cancelling errors), and flawed reasoning patterns fail on harder problems. Full development deferred to reasoning-models lesson. |
| CoT error propagation (errors in intermediate steps corrupt downstream) | INTRODUCED | chain-of-thought | The model does not "catch" its own mistakes. If an intermediate step contains an error (e.g., "17 x 20 = 350"), all subsequent forward passes build on the wrong context. Connected to ICL's ordering sensitivity and prompt-engineering's context-stuffing finding -- the model continues from whatever context exists. Longer chains are not always better: irrelevant intermediate tokens add noise, diluting attention to relevant intermediate results. Quality over quantity. |
| Self-consistency / majority voting (Wang et al. 2022) | MENTIONED | chain-of-thought | Generate multiple reasoning chains, take the majority-vote answer. If one chain can have errors, running several and voting reduces error probability. Mentioned only -- no implementation details or depth. More principled version deferred to reasoning-models lesson. |
| RL for reasoning (RL training with answer correctness as reward) | DEVELOPED | reasoning-models | Same RL loop as RLHF, different reward signal: "did you get the right answer?" instead of "is this helpful?" The model generates reasoning chains, gets positive/negative reward based on correctness, and updates its policy to produce better chains. Connected to established RLHF pipeline -- "same pipeline, different reward signal." Pseudocode walkthrough showing the loop: generate chain -> check answer -> compute reward -> update policy. RL does not teach new knowledge; it trains the model to use the scratchpad (context window) effectively. |
| Test-time compute scaling (inference compute as new scaling axis) | DEVELOPED | reasoning-models | Paradigm shift from "bigger model" to "more thinking time." Traditional scaling: more parameters = more computation per forward pass. Test-time compute scaling: more reasoning tokens = more forward passes = more computation per problem. Same model, same hardware, variable compute per problem. A 7B reasoning model can outperform a 70B base model on math benchmarks by spending more inference compute. Two independent axes of scaling (model size and inference compute), echoing the "axes not ladder" framework from alignment-techniques-landscape. Inline SVG diagram showing the two axes. "Bigger brain vs more thinking time" analogy. |
| Process supervision vs outcome supervision (ORM vs PRM) | DEVELOPED | reasoning-models | Upgraded from INTRODUCED in chain-of-thought. Outcome Reward Models (ORMs) score the final answer only -- cheap to train but sparse reward, gameable via cancelling errors. Process Reward Models (PRMs) evaluate each step individually -- richer signal, trains correct reasoning not just correct answers. Concrete example: two chains both reaching 408, Chain A with correct steps, Chain B with cancelling errors (17x20=350, 17x4=58, 350+58=408). ORM rewards both equally; PRM penalizes Chain B. Connected to reward hacking from 4.4.3 -- outcome supervision is the proxy, process supervision is closer to the true objective. Connected to RLAIF pattern: step-level labels can start with humans and scale through AI. |
| Self-consistency / search during inference | DEVELOPED | reasoning-models | Upgraded from MENTIONED in chain-of-thought. Full mechanism developed: sample N chains (different random seeds), extract final answers, take majority vote. Worked example with 5 chains for a word problem (3 correct at $144, 2 wrong). Second generalization example: train problem solvable via algebra, rate-of-closing, or position table -- different approaches converge on same answer. Why it works: if each chain has p > 0.5 of being correct, N independent chains with majority voting have higher probability than any single chain. Best-of-N with verifier (ORM or PRM) as alternative selection mechanism. Both are forms of test-time compute scaling. Connected to CoT error propagation: different chains make different errors, voting averages out the noise. |
| Outcome reward models (ORMs) vs process reward models (PRMs) | DEVELOPED | reasoning-models | Specific vocabulary for the two supervision approaches. ORM: sees full chain, scores final answer, cheap but sparse and gameable. PRM: evaluates each step, rich signal, requires step-level labels (expensive but automatable via RLAIF pattern). ComparisonRow treatment with 5 properties each. Misconception addressed: "process supervision means a human checks every step" -- PRMs are trained models that evaluate at scale. |
| Specific reasoning model architectures: o1, DeepSeek-R1 | INTRODUCED | reasoning-models | Named as concrete instances of reasoning models. o1 generates internal reasoning tokens before producing a visible answer -- user sees only the final result. DeepSeek-R1 demonstrated that pure RL (without SFT on reasoning examples) can produce effective reasoning behavior from scratch. Name recognition and high-level differentiation, not implementation details. |
| "More reasoning is always better" overthinking problem | INTRODUCED | reasoning-models | Negative example: reasoning model applied to "What is the capital of France?" generates unnecessarily long chain with second-guessing through historical capitals, wasting compute and risking confused answer. Diminishing returns: going from 1 to 5 chains is a large improvement, 50 to 100 is marginal, compute cost is linear. The optimal amount of reasoning depends on problem difficulty -- simple tasks need less thinking, complex tasks benefit from more. Adaptive compute allocation outperforms uniform allocation. |

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

### Lesson 2: Prompt Engineering (prompt-engineering)

**Concepts taught:**
- Prompt engineering as structured programming -- systematic prompt construction with identifiable, composable components (DEVELOPED)
- Format specification and output constraints -- explicit output schemas constrain the output distribution via format token anchoring and autoregressive consistency (DEVELOPED)
- Role and system prompting as attention bias -- role tokens shift what the model attends to in the input, not what the model knows (DEVELOPED)
- Few-shot example selection principles: diversity over quantity, format consistency, difficulty calibration (DEVELOPED)
- Retrieval-Augmented Generation (RAG) as context augmentation for attention (INTRODUCED)
- Context stuffing failure -- irrelevant documents dilute attention, quality over quantity (INTRODUCED)
- RAG does not eliminate hallucination -- model can blend retrieved info with parametric priors (INTRODUCED)
- No universal prompt template -- principles transfer, specific prompts do not (INTRODUCED)

**Mental models established:**
- "Prompt engineering is programming, not conversation" -- extends the "prompt is a program" model from ICL into a systematic engineering discipline. Structure produces reliability. Phrasing does not. Not finding magic phrases; designing reliable systems.
- Software engineering mapping for prompt components: format specification = type signature (constrains return type), role prompt = import (brings context into scope), few-shot examples = unit tests (show expected contract), system prompt = config file (global behavior settings), context/RAG = dependency injection (data at runtime instead of hardcoded in parameters).
- "The prompt is a program; attention is the interpreter. Prompt engineering is writing better programs." -- the synthesis statement connecting all techniques through the attention mechanism.

**Analogies used:**
- Prompt as composable function: `prompt(role, task, format, examples, context, input) -> output` -- each parameter is a design choice
- Format specification = type signature (constrains what the function can return)
- Role prompt = import statement (brings relevant context into scope)
- Few-shot examples = unit tests (show the expected input-output contract)
- System prompt = configuration file (global behavior settings)
- RAG = dependency injection (provide data at runtime, not hardcoded at training time)

**How concepts were taught:**
- **Recap:** Brief reconnection to ICL from Lesson 1. "You learned that the prompt is a program and attention is the interpreter. But ICL is fragile -- ordering matters, format matters, the wrong examples can hurt. That fragility is exactly why prompt engineering exists."
- **Hook (before/after contrast):** Invoice data extraction with conversational prompt (inconsistent format every run) vs structured prompt with format spec + role + example (consistent parseable JSON). GradientCard: "Prompt engineering is not about finding the right words. It is about designing the right structure."
- **Prompt anatomy (inline SVG diagram):** Six color-coded sections (system/role, task instruction, format specification, few-shot examples, context/retrieved docs, user input), each annotated with its function in attention and a software engineering analogy. Flow arrow showing token order. Explicitly connected to causal masking: "Because of causal masking, each token can only attend to tokens before it. System and role tokens placed first are visible to every subsequent token."
- **Technique 1 -- Format specification:** Deepened from hook. Two mechanisms explained: (1) format tokens anchor attention, (2) autoregressive generation maintains consistency once the first format token is generated. Callback to "autoregressive generation is a feedback loop." CodeBlock showing without/with format specification.
- **Technique 2 -- Role/system prompts:** Code review with generic vs security-focused role. ComparisonRow showing qualitatively different issues found. Negative example inside GradientCard: factual question about Acme Corp earnings, no-role model is honest ("I don't have access"), expert-role model confabulates ($847M, 12% growth). Callback to "SFT teaches format, not knowledge."
- **Technique 3 -- Few-shot example selection:** Three principles as GradientCards (diversity > quantity, format consistency, difficulty calibration). Concrete example: sentiment classification with 3 same-polarity examples (all positive) vs 3 diverse examples. Same-polarity classifies a mixed-sentiment input wrong; diverse set gets it right. Connected to attention: diverse examples create richer K/V patterns.
- **Check 1 (predict-and-verify):** Summarize a technical document for non-technical audience. Three prompt variants ranked least-to-most reliable. Key insight: format specification provides the biggest single improvement; not every prompt needs every component.
- **RAG section:** Problem (model knowledge frozen at cutoff) -> solution (put relevant info in prompt). ComparisonRow: closed-book confabulates vs open-book with retrieved docs answers accurately. RAG as two-step process: retrieve (outside model) + place in prompt (inside model). Dependency injection analogy.
- **Negative example -- context stuffing:** ComparisonRow of 5 relevant docs vs 5 relevant + 15 irrelevant. Accuracy drops. Connected to softmax distributing weight across all tokens. Callback to ICL's 50-example problem.
- **RAG hallucination misconception:** ComparisonRow inside GradientCard. Document says "$4.2M, 3% decline" but model outputs "$5M+, strong growth." Model blended retrieved info with parametric priors. RAG reduces, does not eliminate, hallucination.
- **No universal template:** Different models and tasks benefit from different combinations. Principles transfer; templates do not. GradientCard: "The techniques are tools. Prompt engineering is knowing which tools to use for which job."
- **Check 2 (transfer question):** Design a customer support chatbot prompt system. Student identifies needed components (role, format spec with JSON schema, RAG for product-specific articles, user question) and risks (context stuffing, hallucination despite RAG, format breakage). Composition of learned techniques.

**Visual elements:**
- PromptAnatomyDiagram (inline SVG): Six color-coded prompt sections with annotations connecting each to its attention function and software analogy. Token order arrow with causal masking connection.
- ComparisonRows: conversational vs structured prompt (hook), generic vs role-framed code review, no-role vs expert-role factual question, same-polarity vs diverse few-shot examples, closed-book vs RAG, 5 relevant docs vs 5+15 mixed docs, retrieved document vs model output (hallucination)
- GradientCards: core reframe (orange), role prompt limitations (rose), diversity/consistency/calibration principles (blue), RAG-hallucination misconception (amber), tools-not-templates (violet), both checkpoints (emerald)

**What is NOT covered:**
- Chain-of-thought prompting or step-by-step reasoning (Lesson 3: chain-of-thought)
- Reasoning models or test-time compute (Lesson 4: reasoning-models)
- Building a RAG pipeline (vector databases, embedding models, retrieval systems)
- Prompt optimization or automated prompt search (DSPy, etc.)
- Specific model-dependent prompt formatting (ChatML, special tokens)
- Agentic patterns or tool use

**Notebook:** `notebooks/5-2-2-prompt-engineering.ipynb` (4 exercises)
- Exercise 1 (Guided): Format specification. Given a paragraph, construct prompts extracting structured data in three formats (bullets, JSON, markdown table). Start conversational, observe inconsistency, add format specification progressively. Predict-before-run. First format (bullet points) fully worked.
- Exercise 2 (Supported): Role prompting effects. Code snippet with security, performance, and style issues. Three different role prompts, then a combined role. Compare which issues each surfaces. First role prompt provided.
- Exercise 3 (Supported): Few-shot example selection. Text classification with 4 example-set variants (3 random, 3 diverse, 3 same-category, 5 random). 5 trials each. Plot accuracy. Discover diversity > quantity.
- Exercise 4 (Independent): Build a structured prompt. Design a complete prompt for a real task (meeting summaries from notes) using at least 3 techniques. Test on 3 inputs, evaluate consistency. No skeleton provided.

**Review:** Passed at iteration 2/3. Iteration 1 had 0 critical, 4 improvement (misconception 3 missing concrete negative example, few-shot section missing concrete worked example, RAG hallucination claim not concretely demonstrated, causal masking not mentioned for prompt ordering), and 3 polish (notebook em dashes, summary item 6 redundant, diagram "=" notation). All improvements resolved in iteration 2. Iteration 2 had 0 critical, 0 improvement, 1 polish (GradientCard density in role section, deemed acceptable). Clean pass.

### Lesson 3: Chain-of-Thought Reasoning (chain-of-thought)

**Concepts taught:**
- Intermediate tokens as computation -- each generated token triggers another forward pass, making the autoregressive loop a computational amplifier (DEVELOPED)
- Fixed computation budget per forward pass -- same N transformer blocks regardless of problem difficulty, no "difficulty knob" (DEVELOPED)
- When CoT helps vs does not -- computational complexity criterion, not human-perceived difficulty (DEVELOPED)
- Zero-shot CoT vs few-shot CoT -- two ways to trigger reasoning tokens, mapped to prompt engineering and ICL frameworks (DEVELOPED)
- Process supervision vs outcome supervision -- evaluating reasoning chains step-by-step vs final-answer-only (INTRODUCED)
- CoT error propagation -- errors in intermediate steps corrupt downstream, model does not catch mistakes (INTRODUCED)
- Self-consistency / majority voting (MENTIONED)

**Mental models established:**
- "A forward pass is the model's thinking budget per token. CoT is spending more budget by generating more tokens. The budget is fixed; the spending is not." -- The core mental model. Reframes the autoregressive loop from a generation mechanism to a computational amplifier.
- Scratchpad analogy: asking a model to answer "17 x 24" in one forward pass is like asking a human to multiply two-digit numbers without writing anything down. Writing intermediate results is not "trying harder" -- it is using external memory to decompose a problem that exceeds working memory. The context window is the model's scratchpad.

**Analogies used:**
- Mental math without writing anything down = direct answer (fixed compute, no external memory)
- Writing intermediate results on paper = CoT (external memory via context window)
- "Thinking budget" per token = one forward pass through N blocks (fixed, cannot be increased)
- "Spending more budget" = generating more tokens (each triggers another forward pass)

**How concepts were taught:**
- **Recap:** Brief reconnection to two facts the student already has at DEVELOPED depth: (1) autoregressive generation is a feedback loop (generate() from building-nanogpt), (2) every forward pass runs through the same N transformer blocks regardless of difficulty. Connected to prompt-engineering's explicit cliffhanger: "problems that require more computation than a single forward pass provides."
- **Hook (before/after contrast + puzzle):** 17 x 24 with direct prompt (wrong: "384") vs CoT prompt (correct: 340 + 68 = 408). ComparisonRow. GradientCard puzzle: "The model did not become smarter. Its weights did not change. What changed?"
- **Explain Part 1 -- Fixed computation budget:** Established the constraint. Walked through one forward pass: embedding -> N blocks -> output projection -> sample. Same for "2+2" and "prove Riemann hypothesis." Callback to generate() code from building-nanogpt. Scratchpad/working-memory analogy introduced here.
- **Explain Part 2 -- Tokens as computation (core mechanism):** Step-by-step walkthrough of forward passes #1 through #15 for a CoT response. Context grows with each step. "Of course" beat: "You already knew this. Autoregressive generation is a feedback loop. Each token gets N blocks. More tokens = more forward passes. Of course they help."
- **Misconceptions 1 & 2:** Explicitly named and addressed immediately after the core mechanism. (1) "The model thinks during CoT" -- disproved by corrupted intermediate step experiment: model continues from wrong context, not checking reasoning. (2) "CoT is showing internal work" -- if the model could compute internally, it would get the right answer without CoT. Intermediate tokens ENABLE computation, not DISPLAY it.
- **Visual (inline SVG):** ComputationDiagram showing direct answer (1 forward pass, wrong) vs CoT (4 visible steps with growing context, correct). Each step labeled with forward pass number. Bottom comparison: "Same model, same architecture, same weights."
- **Symbolic callback:** generate() code annotated: each loop iteration = one forward pass. Direct answer: ~1 iteration. CoT: ~15 iterations. Same code, more iterations, more computation.
- **Check 1 (predict-and-verify):** Farmer problem (3 fields x 7 rows x 12 plants). Predict which prompt (direct vs CoT) succeeds, how many forward passes, what intermediate results appear. Details/summary reveal.
- **When CoT helps and when it does not:** Second positive example (word problem: 3 shelves x 8 books - 5). Misconception 3 explicitly named: "Always use CoT" -- GradientCard with factual recall negative example ("What is the capital of France?" -- CoT adds nothing). Criterion GradientCard: computational complexity, not human-perceived difficulty.
- **Error propagation + quality over quantity:** Negative example showing wrong intermediate step (17 x 20 = 350) propagating. Connected to ICL ordering sensitivity and prompt-engineering context-stuffing. Longer chains not always better -- irrelevant tokens dilute attention.
- **Not a new mechanism:** CoT is the same generate() loop. Small models fail not because they lack the loop but because they cannot generate useful intermediate steps. Callback to ICL: "the formula has not changed; the context has."
- **Zero-shot CoT vs few-shot CoT:** ComparisonRow. Zero-shot CoT mapped to prompt engineering (format instruction). Few-shot CoT mapped to ICL (examples as program). Self-consistency MENTIONED.
- **Process supervision vs outcome supervision:** ComparisonRow. Concrete example: two chains reaching the correct answer, Chain A with correct reasoning, Chain B with cancelling errors. Outcome supervision rates both equally; process supervision catches Chain B's flawed reasoning. Explicitly noted as INTRODUCED -- full development in reasoning-models.
- **Check 2 (transfer question):** LLM grading math homework. Should it use CoT? (Yes -- multi-step evaluation.) Process or outcome supervision? (Process.) Failure modes? (Error propagation in evaluation, model "continuing" rather than evaluating.)

**Visual elements:**
- ComputationDiagram (inline SVG): Direct answer (1 forward pass, wrong "384") vs CoT (4 steps with growing context, correct "408"). Forward pass arrows labeled pass 1-4. Growing context bracket. Bottom comparison text: "Same model, same architecture, same weights."
- ComparisonRows: Direct vs CoT prompt (hook), direct vs CoT word problem, without/with CoT on factual recall, zero-shot vs few-shot CoT, outcome vs process supervision
- GradientCards: puzzle (orange), fixed budget question (violet), misconception 1 "model thinks during CoT" (rose), misconception 2 "CoT is showing internal work" (rose), misconception 3 "always use CoT" (rose), criterion (blue), error propagation (rose), quality over quantity (violet), process/outcome distinction (violet), both checkpoints (emerald), mental model echo (violet)
- CodeBlock: generate() method annotated with forward pass counts

**What is NOT covered:**
- Reasoning models trained with RL to use CoT effectively (Lesson 4: reasoning-models)
- Test-time compute scaling (Lesson 4)
- Search during inference, tree-of-thought, or beam search over reasoning paths (Lesson 4)
- Self-consistency implementation details (MENTIONED only)
- Automated CoT generation or optimization
- Implementing CoT in production systems

**Notebook:** `notebooks/5-2-3-chain-of-thought.ipynb` (4 exercises)
- Exercise 1 (Guided): Direct vs CoT comparison. Solve 10 arithmetic problems with and without CoT. Predict which benefit from CoT before running. Compare accuracy. Both single-step and multi-step problems. Insight: CoT helps on multi-step but not single-step.
- Exercise 2 (Supported): Token counting as computation measurement. Count intermediate tokens for 5 problems solved with CoT. Plot tokens vs problem complexity. Insight: more complex problems generate more intermediate tokens; each token is an additional forward pass. Solution in details block (not pre-filled).
- Exercise 3 (Supported): Error propagation experiment. Corrupt one intermediate step, ask model to continue. Corrupt at different positions (early vs late), compare impact. Insight: model does not catch errors; earlier corruption has larger downstream impact.
- Exercise 4 (Independent): Find the CoT boundary. Design problems of increasing complexity, test with and without CoT, find the threshold where CoT starts helping. No skeleton provided.

**Review:** Passed at iteration 2/3. Iteration 1 had 0 critical, 4 improvement (notebook Exercise 2 pre-filled despite "Supported" label, "DEVELOPED depth" internal terminology in aside, scratchpad/working-memory analogy missing from lesson, misconception 3 not explicitly named), and 3 polish (aside title "STRETCH Lesson" uses internal terminology, summary 6 items too many, computation diagram "~15N blocks" label inconsistent with visual). All improvements resolved. Iteration 2 had 0 critical, 0 improvement, 2 polish (notebook spaced em dashes, misconception 4 not explicitly labeled as misconception). Clean pass.

### Lesson 4: Reasoning Models (reasoning-models)

**Concepts taught:**
- RL for reasoning -- same RL loop as RLHF with answer correctness as reward signal (DEVELOPED)
- Test-time compute scaling -- paradigm shift from scaling model size to scaling inference computation (DEVELOPED)
- Process supervision vs outcome supervision -- upgraded from INTRODUCED, with ORM vs PRM distinction, cancelling-errors example, reward hacking connection (DEVELOPED)
- Self-consistency / search during inference -- upgraded from MENTIONED, full mechanism with worked examples, majority vote and best-of-N as search strategies (DEVELOPED)
- Outcome reward models (ORMs) vs process reward models (PRMs) -- specific vocabulary and tradeoffs (DEVELOPED)
- Specific reasoning model architectures: o1, DeepSeek-R1 -- named instances with high-level differentiation (INTRODUCED)
- "More reasoning is always better" overthinking problem -- negative example, diminishing returns, adaptive compute allocation (INTRODUCED)

**Mental models established:**
- "Training the scratchpad" -- RL does not teach new knowledge, it teaches the model to use the scratchpad (context window) effectively. Like training a student to show structured, checkable work rather than scribbling randomly. Same paper, same pen, better use of the scratchpad. Extends the scratchpad analogy from chain-of-thought.
- "Bigger brain vs more thinking time" -- a student who thinks for 30 minutes outperforms a student who glances for 5 seconds, even if the second student is "smarter." Maps to model size vs inference compute tradeoff.
- "The forward pass budget is fixed. CoT spends more budget by generating more tokens. RL trains the model to spend that budget wisely. And test-time compute scaling says: if spending more budget helps, why not give it as much budget as it needs?" -- the synthesis statement for the entire module.

**Analogies used:**
- Training the scratchpad (extends scratchpad from CoT lesson)
- Bigger brain vs more thinking time (model size vs inference compute)
- "Same pipeline, different reward signal" (extends "same pipeline, different data source" from Constitutional AI)
- Asking 5 people and going with the majority (self-consistency as ensemble voting)

**How concepts were taught:**
- **Recap:** Reconnection to two key facts: (1) intermediate tokens are computation (each triggers another forward pass), (2) base model has no training signal for reasoning quality -- chains are unreliable. Reconnection to RLHF: "What if the reward signal were 'did you solve the math problem correctly?'" Two established frameworks explicitly named before connecting them.
- **Hook (before/after contrast):** 5 attempts at 17 x 24 with base model + CoT prompt (3/5 correct, inconsistent chains with specific arithmetic errors like 17x20=350, 17x4=72) vs 5 attempts with reasoning model (5/5 correct, structured decompositions). ComparisonRow with actual math chains. GradientCard puzzle: "Same architecture. Same number of parameters. Same problem. What changed?"
- **RL for Reasoning (core mechanism):** RL training loop walked through: generate chain -> check answer -> compute reward (+1/-1) -> update policy. RLTrainingLoopDiagram (inline SVG) showing the cycle visually. Pseudocode annotated with RLHF callbacks. "Of course" beat: "You already knew tokens are computation. You already knew RL can shape behavior." GradientCard: "Training the scratchpad." Misconception 1 addressed: "Reasoning models are not bigger models."
- **Check 1 (predict-and-verify):** Predict what goes wrong with outcome-only reward. Reveal: shortcuts, cancelling errors, reward hacking pattern from 4.4.3 applied to reasoning. Transition to process supervision.
- **Process Supervision (DEVELOP from INTRODUCED):** Concrete example first (concrete before abstract): Chain A with correct reasoning vs Chain B with cancelling errors (17x20=350, 17x4=58, 350+58=408), both reaching 408. ORM rewards both; PRM penalizes Chain B. Then ComparisonRow generalizing ORM vs PRM properties. Connected to reward hacking (outcome = proxy), RLAIF (step labels scale through AI). Misconception 3: "Process supervision does not mean a human checks every step."
- **Test-Time Compute Scaling:** ScalingParadigmDiagram (inline SVG) with two axes: model size (x) and inference compute (y). Traditional scaling moves right, test-time compute moves up. ComparisonRow: traditional vs test-time scaling (5 properties each). "Bigger brain vs more thinking time" analogy. Misconception 4: "Test-time compute scaling does not mean better hardware." Key insight: 7B reasoning model outperforming 70B base model. o1 and DeepSeek-R1 named as concrete instances (achieving INTRODUCED depth). Misconception 2: "The model does not 'decide' to think harder."
- **Search During Inference (DEVELOP self-consistency from MENTIONED):** Three-step mechanism (sample N chains, extract answers, majority vote). SelfConsistencyExample component: 5 chains for Alex-earns-$12/hour problem, 3 correct ($144), 2 wrong ($142, $132 with specific arithmetic errors shown). Majority vote analysis. Second generalization example: train problem solvable via algebra, rate-of-closing, or position table -- different approaches converge. Why it works: p > 0.5 per chain -> majority voting amplifies correctness. Best-of-N with verifier. Connected to CoT error propagation. Negative example: "When Reasoning Backfires" -- reasoning model overthinking "What is the capital of France?" with second-guessing through historical capitals. Misconception 5: "More reasoning tokens are not always better" -- diminishing returns.
- **Check 2 (transfer question):** Company deploying LLM for customer support -- 70B base model vs 7B reasoning model for FAQ (60%) + troubleshooting (40%). Hybrid approach, process supervision for uncertain steps.

**Visual elements:**
- ScalingParadigmDiagram (inline SVG): Two axes (model size, inference compute) with traditional scaling (horizontal arrow) and test-time compute scaling (vertical arrow). Base model, larger model, and reasoning model points. Key insight box: "A 7B reasoning model can outperform a 70B base model."
- RLTrainingLoopDiagram (inline SVG): Five-step cycle showing math problem -> generate reasoning chain -> check answer -> compute reward -> update policy, with annotation: "Same RL loop as RLHF. The only difference: the reward signal."
- SelfConsistencyExample component: 5 chains color-coded (emerald for correct, rose for wrong), majority vote summary.
- ComparisonRows: base model CoT vs reasoning model (hook), ORM vs PRM, traditional scaling vs test-time compute scaling
- GradientCards: puzzle (orange), "of course" (violet), training the scratchpad (cyan), misconception 1 "bigger models" (rose), misconception 2 "decides to think harder" (rose), misconception 3 "human checks every step" (rose), misconception 4 "better hardware" (rose), misconception 5 "always better" (rose), "When Reasoning Backfires" negative example (rose), process supervision concrete example (violet), both checkpoints (emerald), mental model echo (violet)

**What is NOT covered:**
- Implementing RL for reasoning in code
- Specific model architectures or training details of o1, DeepSeek-R1
- Tree-of-thought or MCTS implementation details
- Mathematical formalization of the RL objective
- Distillation of reasoning models
- Agentic patterns, tool use, or multi-step planning

**Notebook:** `notebooks/5-2-4-reasoning-models.ipynb` (4 exercises)
- Exercise 1 (Guided): Base model CoT vs reasoning model comparison. 10 math/reasoning problems, compare accuracy and reasoning chain quality. Predict-before-run: which problems will base model get wrong? First 3 problems fully worked with analysis template.
- Exercise 2 (Supported): Self-consistency experiment. 5 problems, generate N chains (N=1,3,5,10,20), compute majority-vote accuracy at each N. Plot accuracy vs N. Identify diminishing returns. Try easy and hard problems.
- Exercise 3 (Supported): Process vs outcome evaluation. 5 problems with known solutions, evaluate chains two ways: (a) final answer correct? (b) each step correct? Find cases with correct outcome but wrong steps. LLM used as simplified proxy PRM (acknowledged limitation).
- Exercise 4 (Independent): Test-time compute allocation. Design experiment comparing equal allocation vs adaptive allocation (more tokens for harder problems) given fixed compute budget. No skeleton provided.

**Review:** Passed at iteration 3/3. Iteration 1 had 1 critical (planned negative example missing as standalone element), 5 improvement (hook hypothetical not concrete, self-consistency needs second example, process supervision ordering abstract-before-concrete, specific reasoning architectures not at INTRODUCED depth, notebook Exercise 2 hints too close to solution), and 3 polish (ModuleCompleteBlock references non-existent module, NextStepBlock too long, notebook LLM-as-proxy-PRM unacknowledged). All resolved. Iteration 2 had 0 critical, 2 improvement (self-consistency still missing second example, ModuleCompleteBlock references non-existent Module 5.3), 2 polish (notebook spaced em dash, summary items 3/5 overlap). All resolved. Iteration 3: 0 findings across all categories. Clean pass.
