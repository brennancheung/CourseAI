# Module 5.2: Reasoning & In-Context Learning -- Plan

**Status:** In progress
**Prerequisites:** Module 5.1 (Advanced Alignment) complete, Series 4 (LLMs & Transformers) complete

## Module Goal

The student understands how transformers perform computation at inference time without weight updates -- from in-context learning through chain-of-thought reasoning to dedicated reasoning models -- and can explain why each technique works mechanically rather than magically.

## Narrative Arc

Module 5.1 answered "how do we make models behave well?" This module asks a different question: "what can models actually *do*, and why?" The student knows that a language model predicts the next token. But something surprising happens when you put examples in the prompt: the model appears to *learn* the task without any weight update. How?

The arc progresses through four levels of inference-time capability:

1. **In-context learning** (Lesson 1): The transformer can learn tasks from examples in the prompt. This is the foundational surprise -- the model performs gradient-free learning at inference time. The mechanism is attention: examples create a retrieval structure that steers output.

2. **Prompt engineering** (Lesson 2): Once you understand that examples in the prompt steer the model, you can be systematic about it. Prompting becomes programming, not conversation. RAG extends the pattern: put relevant information in the context, let attention do the rest.

3. **Chain-of-thought** (Lesson 3): The model's forward pass has a fixed computational budget per token. For problems that require more computation than one forward pass provides, intermediate tokens serve as external working memory. "Let's think step by step" works because it gives the model more computation, not because the model "decides" to think harder.

4. **Reasoning models** (Lesson 4): What if you trained the model to use chain-of-thought effectively? Reinforcement learning for reasoning, test-time compute scaling, and the paradigm shift from "bigger model" to "more thinking time."

The connecting thread: each lesson reveals that the transformer's capabilities extend beyond what the training objective (next-token prediction) seems to promise, and each technique works because of a specific, mechanistic reason -- not because the model "understands" or "thinks."

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| in-context-learning | The transformer performs gradient-free task learning from examples in the prompt via attention | STRETCH | First lesson introduces the most foundational surprise. Must come before prompt engineering (which systematizes it) and chain-of-thought (which extends it). Transition from alignment module to capabilities module. |
| prompt-engineering | Systematic prompting as programming the model's inference-time behavior | BUILD | Builds on ICL understanding: once you know examples steer the model, you can be systematic. Lower cognitive load -- practical application of the prior lesson's insight. |
| chain-of-thought | Intermediate tokens as computation: why explicit reasoning steps expand the transformer's computational capacity | STRETCH | Requires ICL understanding (few-shot CoT uses in-context examples). Introduces a genuinely new concept: tokens as computation, not just communication. |
| reasoning-models | RL-trained reasoning and test-time compute scaling | BUILD | Applies chain-of-thought insight to training: if CoT expands capacity, train the model to use it. Capstone of the module, not a new paradigm shift. |

## Rough Topic Allocation

- **Lesson 1 (in-context-learning):** Zero-shot vs few-shot prompting, the GPT-3 discovery, why in-context learning is surprising (no weight updates), attention as the mechanism (examples create retrieval patterns), the transformer as a learning algorithm at inference time, limitations of ICL (context window, sensitivity to example ordering)
- **Lesson 2 (prompt-engineering):** Structured output formats, role/system prompting, RAG overview (retrieval as context augmentation), output constraints, moving from "talking to the model" to "programming the model"
- **Lesson 3 (chain-of-thought):** Why "let's think step by step" works mechanically, intermediate tokens as computation, fixed computation per forward pass as a limitation, process supervision vs outcome supervision, the limits of single-forward-pass reasoning
- **Lesson 4 (reasoning-models):** RL for reasoning (reward correct final answers), test-time compute scaling (spend more inference compute on harder problems), search during inference, the "bigger model vs more thinking time" paradigm shift, how models like o1 differ from standard chain-of-thought

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| in-context-learning | STRETCH | New paradigm: inference-time learning without weight updates. Conceptually surprising. Follows a CONSOLIDATE-equivalent (evaluating-llms was the capstone of 5.1). |
| prompt-engineering | BUILD | Practical application of ICL insight. No new paradigm, systematic organization of techniques. |
| chain-of-thought | STRETCH | Genuinely new concept: tokens as computation, fixed compute budget per forward pass. |
| reasoning-models | BUILD | Applies chain-of-thought to training. New vocabulary (test-time compute, RL for reasoning) but the conceptual framework is established. |

No two STRETCH lessons are adjacent. The STRETCH-BUILD-STRETCH-BUILD pattern gives the student recovery between paradigm shifts.

## Module-Level Misconceptions

- **"In-context learning is just pattern matching / nearest-neighbor retrieval"** -- It is more than template matching; the model can learn genuinely novel input-output mappings from examples, including ones not in its training data. But it is also less than full learning -- it cannot update weights or generalize beyond what attention over the context can compute.

- **"The model 'understands' the examples and 'decides' to follow the pattern"** -- The mechanism is attention and the forward pass, not comprehension or decision-making. Examples create retrieval patterns in the attention weights. The model processes the examples the same way it processes any context.

- **"Chain-of-thought works because the model 'thinks' when it generates reasoning tokens"** -- The model generates one token at a time, each via a fixed-size forward pass. CoT works because each intermediate token feeds back into the context, providing additional information for subsequent forward passes. The "thinking" is the token-by-token feedback loop, not an internal deliberation.

- **"Prompt engineering is just phrasing your question nicely"** -- Prompting is programming: you are constructing the input to a function (the forward pass) to produce desired output. The structure, format, and content of the prompt all shape the attention patterns and therefore the output.

- **"Reasoning models are just better base models"** -- They are architecturally identical but trained differently (RL on reasoning tasks). The key innovation is spending more computation at inference time, not having better weights per se. The paradigm shift is from "bigger model" to "more thinking time."
