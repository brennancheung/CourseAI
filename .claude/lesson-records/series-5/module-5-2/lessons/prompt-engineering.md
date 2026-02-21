# Lesson Plan: Prompt Engineering

**Module:** 5.2 (Reasoning & In-Context Learning)
**Position:** Lesson 2 of 4
**Slug:** `prompt-engineering`
**Status:** Planning complete

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| In-context learning as gradient-free task learning from examples in the prompt | DEVELOPED | in-context-learning (5.2.1) | Core foundation. The student understands that a transformer can classify, translate, or follow novel patterns from examples placed in the prompt. No gradients, no weight updates. The "learning" happens in activations (attention patterns), not parameters. |
| Attention as the mechanism enabling ICL (Q from test input, K/V from context including examples) | DEVELOPED | in-context-learning (5.2.1) | The student can trace how test input query vectors attend to example inputs via Q*K scores and retrieve output patterns via V blending. Same formula, longer context. |
| Zero-shot vs few-shot prompting | DEVELOPED | in-context-learning (5.2.1) | Terminology is solid. Zero-shot: instruction only. Few-shot: instruction + input-output examples. Both are forms of ICL. |
| "The prompt is a program; attention is the interpreter" | DEVELOPED (mental model) | in-context-learning (5.2.1) | Central mental model from Lesson 1. Different examples = different behavior. Same weights, different programs. Prompt is configuration, not code; model's weights are the compiled binary; you change behavior by changing the config file (prompt), not recompiling (retraining). |
| ICL limitations: ordering sensitivity, format fragility, context window constraints | INTRODUCED | in-context-learning (5.2.1) | Student knows ICL is powerful but fragile. Accuracy swings 20-30 percentage points from reordering. More examples do not always help. Context window limits how many examples fit. |
| Attention as data-dependent weighted average | DEVELOPED | the-problem-attention-solves (4.2.1) | Core mechanism. "The input decides what matters." Reinforced in 5.2.1. |
| Autoregressive generation as feedback loop (sample, append, repeat) | DEVELOPED | what-is-a-language-model (4.1.1) | The student understands generation. Relevant because structured prompts shape what gets generated token-by-token. |
| SFT teaches format, not knowledge | DEVELOPED | instruction-tuning (4.4.2) | "Expert-in-monologue" analogy. Relevant because system prompts and role prompts interact with the model's SFT-shaped behavior, but the knowledge comes from pretraining. |
| Causal masking (lower-triangular attention, each token sees only past) | DEVELOPED | decoder-only-transformers (4.2.6) | Relevant for understanding why prompt structure and ordering matter -- earlier tokens attend to less context than later ones. |
| Label-flipping robustness (format and structure matter more than mapping for many tasks) | INTRODUCED | in-context-learning (5.2.1) | Student knows the format of examples can matter more than the labels themselves. Directly relevant: format is a powerful lever. |
| ICL as not memorized pattern retrieval (novel mapping capability) | INTRODUCED | in-context-learning (5.2.1) | ICL generalizes beyond training data, within the scope of a single forward pass. |

### Established Mental Models and Analogies

- "The prompt is a program; attention is the interpreter" (5.2.1) -- the central model this lesson extends
- "Attention is data-dependent. Examples are data. Of course examples influence the output." (5.2.1)
- "Between retrieval and comprehension" -- ICL is not memorization and not understanding; it is attention-based computation (5.2.1)
- "Capability = vulnerability" (5.1.3, extended in 5.2.1)
- "SFT teaches format, not knowledge" (4.4.2)
- "Autoregressive generation is a feedback loop" (4.1.1)
- "The input decides what matters" / data-dependent weights (4.2.1)
- "Benchmarks are standardized tests for LLMs" (5.1.4)
- "The evaluator's limitations become the evaluation's limitations" (5.1.4)

### What Was Explicitly NOT Covered

- Systematic prompt engineering techniques were explicitly deferred from in-context-learning (5.2.1)
- Role/system prompting was explicitly listed as out of scope in in-context-learning (5.2.1)
- Retrieval-augmented generation (RAG) was explicitly deferred to this lesson (5.2.1)
- Structured output formats were not covered
- Output constraints (JSON mode, format enforcement) were not covered
- The "programming the model" framing was previewed in the "next step" section of 5.2.1 but not developed

### Readiness Assessment

The student is well prepared and motivated. Lesson 1 established the "prompt is a program" mental model and explicitly set up this lesson in its "Next Step" section: "The next lesson systematizes this: prompt engineering as programming, not conversation." The student understands WHY examples in the prompt work (attention mechanism) and has experienced ICL's fragility (ordering sensitivity, format dependence). They are primed to ask "how do I control this reliably?" -- which is exactly the question this lesson answers. No gaps need bridging. This is a natural BUILD lesson: applying and systematizing the insight from Lesson 1 rather than introducing a new paradigm.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to treat prompt construction as structured programming -- selecting and combining specific techniques (format specification, role framing, example selection, output constraints, context augmentation) to reliably control model behavior, grounded in the attention mechanism they already understand.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| In-context learning as gradient-free task learning | DEVELOPED | DEVELOPED | in-context-learning (5.2.1) | OK | The entire lesson builds on this. The student must deeply understand that examples in the prompt shape behavior through attention, so that prompt engineering techniques feel like principled extensions rather than arbitrary tricks. |
| "The prompt is a program; attention is the interpreter" | DEVELOPED (mental model) | DEVELOPED | in-context-learning (5.2.1) | OK | This lesson extends the mental model from "prompt is a program" (informal) to "prompt engineering is programming" (systematic). The student needs to have internalized the informal version. |
| Attention as the mechanism enabling ICL | INTRODUCED | DEVELOPED | in-context-learning (5.2.1) | OK | The student must understand that attention is WHY prompt structure matters -- format, ordering, and content all shape attention patterns. Only INTRODUCED depth required because this lesson does not extend the mechanism, just applies it. |
| ICL limitations (ordering sensitivity, format fragility) | INTRODUCED | INTRODUCED | in-context-learning (5.2.1) | OK | The limitations motivate prompt engineering: if ICL is fragile, you need systematic techniques to make it reliable. The student must know ICL is fragile to appreciate why these techniques exist. |
| Autoregressive generation as feedback loop | INTRODUCED | DEVELOPED | what-is-a-language-model (4.1.1) | OK | Relevant for understanding structured outputs: the model generates one token at a time, so output format constraints work by shaping what token is likely at each step. Only INTRODUCED needed. |
| SFT teaches format, not knowledge | INTRODUCED | DEVELOPED | instruction-tuning (4.4.2) | OK | Relevant for understanding system/role prompts: they interact with SFT-shaped behavior. The model follows instructions partly because SFT trained it to, and the prompt content shapes what instructions it follows. |

All prerequisites are met at sufficient depth. No gaps.

### Gap Resolution

No gaps to resolve. All prerequisites are at sufficient or above-sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **"Prompt engineering is just phrasing your question nicely / being polite to the model"** | The popular framing of prompting is conversational ("ask nicely," "say please," "be specific"). The student has used ChatGPT conversationally and may think prompt engineering is about better natural language phrasing. The word "engineering" sounds like branding rather than substance. | Take a task (e.g., extract structured data from a paragraph) and show two prompts: (A) a well-phrased conversational request ("Please extract the key information from this text and organize it nicely"), and (B) a structured prompt with explicit format specification, a role frame, and an output schema. Prompt A produces inconsistent, variable-format outputs. Prompt B produces consistent, parseable outputs. The difference is not "niceness" -- it is structure. Phrasing is about natural language; engineering is about controlling computation. | Section 3 (Hook). This is the opening misconception to shatter. The entire lesson's motivation hinges on the student seeing that prompting is programming, not conversation. |
| **"There's one 'best prompt' for each task that you discover through trial and error"** | The student may treat prompting as search for a magic incantation. Trial and error is how most people currently do prompt engineering. The framing of "prompt optimization" in popular discourse reinforces this -- as if there is a single optimal point. | Show the same extraction task with two different valid structured prompts that use different techniques (one uses few-shot examples, the other uses a detailed schema with no examples). Both work well. Then change the model (or model version) and show that the relative performance of the two prompts shifts. There is no fixed "best prompt" because the prompt interacts with the model's learned representations, which differ across models. Prompt engineering is about understanding the principles (what attention responds to), not memorizing templates. | Section 7 (Elaborate). After the student has learned multiple techniques, address the temptation to combine them into one "ultimate prompt." |
| **"System prompts / role prompts give the model special abilities it doesn't otherwise have"** | The student may think "You are an expert data scientist" makes the model compute differently or access special knowledge. The ChatGPT system prompt feels like it changes the model's capabilities. The "role" framing implies the model becomes something different. | Give the model a factual question it cannot answer (something not in its training data). Try it with no role, then with "You are a world-class expert in [topic]." The model does not suddenly know the answer -- it may generate a more confident-sounding wrong answer. The role prompt shapes the distribution of outputs (style, confidence, framing) but does not add knowledge. Callback to "SFT teaches format, not knowledge" -- role prompts teach format within a single inference, not knowledge. | Section 5 (Explain), when introducing role/system prompts. Address immediately after explaining what role prompts do, before the student over-generalizes. |
| **"More context is always better -- put everything in the prompt"** | Natural intuition: more information = better answers. The student knows that more examples in ICL sometimes help. RAG involves adding context. The temptation is to stuff the prompt with as much relevant information as possible. | Show a retrieval task where the prompt contains 5 relevant paragraphs and 15 irrelevant ones. The model's accuracy drops compared to a prompt with only the 5 relevant paragraphs. Irrelevant context creates noise in the attention pattern -- the model's Q*K scores distribute across irrelevant tokens, diluting the signal from relevant ones. This connects directly to attention mechanics: more tokens means the softmax distributes weight across more positions. Callback to the ICL lesson's finding that 50 examples in a 4096-token window caused problems. | Section 7 (Elaborate), in the RAG section. After introducing RAG as "retrieval as context augmentation," immediately show that retrieval quality matters as much as retrieval quantity. |
| **"RAG solves the hallucination problem"** | RAG adds real documents to the context, so the model has factual grounding. The popular framing of RAG is "gives the model access to your data." The student may think this eliminates hallucination because the facts are right there in the prompt. | Show a RAG scenario where the retrieved document is present in the context, but the model's answer contradicts or goes beyond what the document says. The model can hallucinate DESPITE having the correct information in context -- it may blend retrieved information with parametric knowledge (from pretraining), it may over-generalize from the retrieved text, or it may simply not attend to the relevant passage strongly enough. RAG reduces hallucination by giving the model relevant context, but the model is still running the same forward pass, and attention is not guaranteed to attend to the retrieved passages more than to its parametric knowledge. | Section 7 (Elaborate), in the RAG section. This is the most important misconception to address for RAG specifically. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Data extraction: conversational prompt vs structured prompt** (extract name, date, amount from an invoice paragraph) | Positive | First example: demonstrates that structure in the prompt produces structure in the output. The simplest possible case that shows prompting-as-programming vs prompting-as-conversation. | Accessible, immediately graspable. Invoice data extraction is a real task the student can imagine building. The contrast between "please extract the info" (inconsistent output) and a formatted prompt with an explicit output schema (consistent, parseable output) makes the core point vivid. Maps to software engineering background: structured input -> structured output, like a function signature. |
| **Role-framed code review: generic vs role-specific responses** (ask for a code review with no role, then with "You are a senior engineer who prioritizes security vulnerabilities") | Positive | Shows that role prompts shape WHAT the model attends to, not just HOW it phrases the answer. The security-focused role produces a qualitatively different review (finds different issues) than the generic prompt. | Connects to the student's software engineering background. Makes role prompting tangible: the role acts as an attention bias toward certain features of the input. The student can see that "role" is not just style -- it is a soft filter on what the model prioritizes. Second example because it introduces a new technique (role prompting) distinct from format specification. |
| **RAG vs closed-book: answering a question about a recent event** (question about something after the model's training cutoff, with and without a retrieved document) | Positive (stretch) | Shows RAG as attention over retrieved context -- the model can answer questions about information not in its parameters when that information is in the prompt. Extends the "prompt is a program" model: the retrieved document is additional data for the program. | Demonstrates the most impactful technique in real-world prompt engineering. Grounds RAG in the attention mechanism the student already understands: the query attends to the retrieved document's tokens just like it attends to few-shot examples. Makes RAG feel like a natural extension of ICL rather than a separate system. |
| **Context stuffing failure: relevant + irrelevant documents in RAG** (5 relevant paragraphs + 15 irrelevant paragraphs vs 5 relevant only) | Negative | Shows that more context is not always better. Irrelevant context dilutes attention, reducing accuracy. Defines the boundary: RAG's value depends on retrieval quality, not just retrieval quantity. | Directly connects to the attention mechanism. The student can reason about why this happens: softmax distributes weight across all tokens, so irrelevant tokens "steal" attention from relevant ones. Prevents the "just add more context" overgeneralization. Also connects to the ICL lesson's 50-example failure scenario. |

---

## Phase 3: Design

### Narrative Arc

The student finished Lesson 1 knowing something remarkable: put examples in a prompt, and the model learns the task. Attention is the mechanism. The prompt is a program. But Lesson 1 also revealed that ICL is fragile -- ordering matters, format matters, and the context window is finite. The student can explain WHY prompting works, but they cannot yet control it reliably. They are in the position of someone who understands that a programming language is Turing-complete but has never learned its syntax, idioms, or best practices.

This lesson bridges that gap. If the prompt is a program, then prompt engineering is programming -- with all the rigor, structure, and deliberate design that implies. This is not about finding magic phrases. It is about understanding which structural elements of a prompt shape attention patterns, and combining them systematically. The student already knows the "why" (attention). This lesson teaches the "how": format specifications that constrain the output space, role frames that bias attention toward relevant features, example selection principles that make few-shot prompting reliable, and retrieval augmentation that gives the model access to information beyond its parameters. By the end, the student should feel that prompt engineering is a legitimate engineering discipline -- closer to API design than to writing polished prose.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example** | Four worked examples: (1) data extraction with conversational vs structured prompt, (2) code review with generic vs role-specific framing, (3) RAG closed-book vs open-book question answering, (4) context stuffing failure. Each example shows a specific technique and its effect on output. | Prompt engineering is an empirical practice. Every technique must be grounded in a concrete before/after demonstration. The student needs to SEE the difference between ad-hoc and structured prompting before generalizing to principles. |
| **Verbal/Analogy** | "Prompt engineering is programming, not conversation" -- extended from "the prompt is a program." Specific mappings: format specification = type signature (constrains the output), role prompt = import statement (brings relevant context into scope), few-shot examples = unit tests (show the expected input-output contract), system prompt = configuration file (sets global behavior), RAG = dependency injection (provide data at runtime instead of hardcoding). | Maps directly to the student's software engineering background. Each prompt engineering technique maps to a programming concept the student already uses daily. This transforms prompt engineering from "tricks with words" to "structured program design" -- which is the entire thesis of the lesson. |
| **Visual** | Prompt anatomy diagram (inline SVG): a structured prompt broken into labeled sections (system/role, task instruction, format specification, few-shot examples, context/retrieved documents, user input). Each section is color-coded and annotated with its function ("biases attention toward...", "constrains output space to...", "provides retrieval targets for..."). | The student needs a spatial/structural view of what a well-designed prompt contains. Without this, the individual techniques feel like a disconnected list. The diagram shows how the parts compose into a whole, and the attention annotations connect each part back to the mechanism the student already understands. |
| **Intuitive** | The "of course" continuation: "In Lesson 1, you learned that attention is data-dependent and examples are data. Prompt engineering is just being deliberate about WHAT data you put in the context and HOW you structure it. Of course structure matters -- you already knew that format affects ICL performance. Prompt engineering is making that deliberate instead of accidental." | Collapses the distance between "ICL works via attention" (which the student already finds intuitive) and "prompt engineering is a systematic discipline" (which the student might find surprising or overhyped). The "of course" beat makes the connection feel inevitable rather than novel. |
| **Symbolic/Code** | Prompt template as pseudocode. Show a structured prompt written as a function with named parameters: `prompt(role, task, format, examples, context, input) -> output`. This is not real code but pseudocode that makes the programming analogy literal. Show how changing each parameter changes the output -- the function signature of a prompt. | For a software engineer, seeing the prompt as a function with parameters is the most natural representation. It makes the lesson's thesis (prompting = programming) literally visible. The pseudocode format also makes it clear which parts of a prompt are independent design choices vs which depend on each other. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 (depending on granularity)
  1. Systematic prompt construction as a structured engineering discipline (the organizing concept -- techniques are components, not tricks)
  2. RAG as retrieval-augmented context for attention (extends ICL to external knowledge)
  3. The individual techniques (role prompting, format specification, output constraints) are not independent concepts but facets of concept #1. They are practical skills, not paradigm shifts.
- **Previous lesson load:** STRETCH. In-context-learning introduced a genuinely surprising paradigm (gradient-free task learning) and required the student to reconcile "learning" with "no weight updates."
- **This lesson's load:** BUILD. No new paradigm. The student already understands why prompting works (attention). This lesson systematizes the practical application. Lower activation energy, more hands-on, less conceptual surprise. The cognitive work is organizational (how do these techniques relate?) rather than paradigm-shifting (how is this possible?).
- **Load is appropriate:** BUILD after STRETCH follows the module plan's intended trajectory. The student needs recovery after the ICL paradigm shift. A practical, application-focused lesson provides that recovery while still advancing understanding. RAG is the only concept that might feel "new," but it is grounded directly in the ICL mechanism the student already has.

### Connections to Prior Concepts

- **"The prompt is a program; attention is the interpreter" (5.2.1):** The foundational connection. This lesson takes the informal insight and makes it systematic. If the prompt is a program, then prompt engineering is the discipline of writing good programs. Every technique maps to a programming concept: format = type signature, role = import, examples = unit tests, RAG = dependency injection.
- **ICL limitations (ordering sensitivity, format fragility) from 5.2.1:** The MOTIVATION for this lesson. ICL is fragile. Prompt engineering is the practice of making it reliable. Each technique addresses a specific fragility: format specification reduces output variance, example selection reduces ordering sensitivity, role prompts reduce attention diffusion.
- **Attention as data-dependent computation (4.2.1, reinforced in 5.2.1):** Every technique in this lesson works because it shapes the attention pattern. Format tokens create structural anchors for attention. Role tokens bias attention toward relevant features. Retrieved documents provide new K/V entries for the model to attend to. The student should be able to explain WHY each technique works in terms of attention, not just THAT it works.
- **SFT teaches format, not knowledge (4.4.2):** Directly relevant for role prompts and system prompts. The model follows role instructions partly because SFT trained it to follow instructions, but the role does not add new knowledge. This prevents the misconception that "You are an expert in X" makes the model an actual expert.
- **Label-flipping robustness (5.2.1):** Format matters more than labels for many tasks. This is evidence that prompt structure is a powerful lever -- the lesson builds on this finding.
- **Causal masking (4.2.6):** Explains why prompt ordering matters. Earlier tokens attend to less context. Important context should come early (system prompt, role) so that all subsequent tokens can attend to it.

**Potentially misleading prior analogies:**
- "Prompt is a program" could lead the student to think prompts should be literally written in a programming language or that there is a formal syntax. The analogy is structural (prompts have components that compose), not literal (there is no compiler, no type checker, no guaranteed behavior). The lesson must keep the analogy productive without over-formalizing it.

### Scope Boundaries

**This lesson IS about:**
- Prompt engineering as a systematic discipline (not tricks or magic phrases)
- Format specification and output constraints (JSON schemas, structured output)
- Role and system prompts (what they do mechanistically, not just what they are)
- Few-shot example selection principles (which examples, how many, in what order)
- RAG overview: retrieval as context augmentation, grounded in the attention mechanism
- Why each technique works, connected to attention
- The software engineering analogies that make these techniques legible

**This lesson is NOT about:**
- Chain-of-thought prompting or step-by-step reasoning (Lesson 3: chain-of-thought)
- Reasoning models or test-time compute (Lesson 4: reasoning-models)
- Building a RAG pipeline (retrieval systems, vector databases, embedding models) -- only the conceptual overview of why putting retrieved text in the prompt works
- Prompt optimization or automated prompt search (AutoGPT, DSPy, etc.)
- Specific model-dependent prompt formatting (ChatML, special tokens)
- Implementing a production prompt engineering workflow
- Agentic patterns or tool use
- Fine-tuning vs prompting tradeoff analysis (touched in 5.2.1)

**Target depth:**
- Prompt engineering as structured programming: DEVELOPED
- Format specification and output constraints: DEVELOPED
- Role/system prompting: DEVELOPED
- Few-shot example selection principles: DEVELOPED
- RAG as retrieval-augmented context: INTRODUCED (conceptual overview only, not implementation)

### Lesson Outline

#### 1. Context + Constraints
What this lesson is about: moving from "the prompt is a program" (the informal insight from Lesson 1) to "prompt engineering is programming" (a systematic discipline). What we are NOT covering: chain-of-thought prompting (next lesson), building RAG pipelines, prompt optimization tools, or agentic patterns.

#### 2. Recap (brief)
One paragraph reconnecting to the core insight from Lesson 1: "You learned that the prompt is a program and attention is the interpreter. Examples in the prompt steer the model's behavior through attention patterns. But you also learned that ICL is fragile -- ordering matters, format matters, and the wrong examples can hurt more than help. That fragility is exactly why prompt engineering exists."

No full re-teach needed. All prerequisites are solid from the immediately preceding lesson.

#### 3. Hook (before/after)
**Type:** Before/after contrast.

Present a real task: extract structured data from an invoice paragraph (vendor name, date, amount, line items). Show two prompts:

**Prompt A (conversational):** "Please extract the key information from this invoice and organize it nicely."

**Prompt B (structured):** A prompt with a system role, explicit output schema (JSON with named fields), one example showing the expected format, and the invoice paragraph.

Show the outputs side by side. Prompt A gives a different format each time (sometimes bullet points, sometimes a paragraph, sometimes a table). Prompt B consistently produces parseable JSON.

The question: "Both prompts ask for the same thing. What makes one reliable and the other not?"

Address misconception 1 here: "This is not about phrasing your request more politely. Prompt A is perfectly clear English. The difference is structural. Prompt B constrains the output space. Prompt B is a program. Prompt A is a wish."

GradientCard: "Prompt engineering is not about finding the right words. It is about designing the right structure."

#### 4. Explain Part 1: The Prompt as a Composable Program

Introduce the organizing framework: a well-designed prompt has identifiable components, each with a specific function. Like a well-designed function, each component is there for a reason.

**Prompt anatomy diagram (inline SVG):** A structured prompt broken into labeled, color-coded sections:
1. **System/Role** -- Sets the behavioral frame ("biases attention toward domain-relevant features")
2. **Task Instruction** -- States the objective ("defines the task for this forward pass")
3. **Format Specification** -- Constrains the output shape ("narrows the output distribution")
4. **Few-shot Examples** -- Shows the expected input-output contract ("creates retrieval patterns in attention")
5. **Context / Retrieved Documents** -- Provides information for this specific query ("adds K/V entries for attention")
6. **User Input** -- The actual query ("the test input whose Q vectors attend to everything above")

Connect each section to attention: "Every section you add to the prompt changes the K and V matrices. The model's query vectors for the user input will attend to ALL of these sections. The structure of the prompt determines the structure of the attention pattern."

**Pseudocode representation:**
```
prompt(role, task, format, examples, context, input) -> output
```

"Each parameter is a design choice. Prompt engineering is choosing these parameters deliberately."

The software engineering mapping (brief, in an aside):
- Format specification = type signature (constrains what the function can return)
- Role prompt = import statement (brings relevant context into scope)
- Few-shot examples = unit tests (show the expected contract)
- System prompt = configuration file (global behavior settings)
- Context/RAG = dependency injection (provide data at runtime)

#### 5. Explain Part 2: The Techniques

Walk through each technique with concrete demonstration. For each: what it is, why it works (attention mechanism), and when to use it.

**A. Format Specification and Output Constraints**

The hook already demonstrated this. Deepen: format tokens in the prompt create structural anchors. When the model sees `"output_format": "JSON"` or an example with `{"name": "...", "date": "..."}`, the attention mechanism picks up these format tokens. The model's autoregressive generation is constrained at each step -- once it generates `{`, the distribution over next tokens heavily favors JSON-valid continuations.

Key insight: "Format specification works because of two mechanisms: (1) attention anchors on format tokens in the prompt, and (2) autoregressive generation makes each token consistent with previous tokens. The first curly brace constrains all subsequent tokens."

**B. Role and System Prompts**

Present the code review example: generic prompt vs "You are a senior engineer who prioritizes security vulnerabilities." Show qualitatively different outputs -- the security-focused role finds different issues.

**Why it works:** The role text adds tokens to the context that bias attention. When the model processes the code, its Q vectors attend to both the code tokens AND the role tokens. The role tokens shift the attention distribution -- "security" in the role makes security-related code patterns more salient to the attention mechanism.

**What it does NOT do:** Address misconception 3. The role does not give the model new knowledge. "You are a world-class expert in quantum computing" does not make the model know more about quantum computing. It shifts the output distribution toward the style and focus of what an expert might say. Callback: "Remember 'SFT teaches format, not knowledge'? Role prompts are the same principle at inference time. They shape how the model presents what it already knows, not what it knows."

Example of role prompt failure: ask a factual question outside the model's training data with and without an expert role. The role produces a more confident-sounding wrong answer. "The role prompt made the model more confidently wrong, not more correct."

**C. Few-Shot Example Selection**

Connect to ICL from Lesson 1. The student already knows few-shot prompting works. Now: how to choose examples systematically.

Three principles:
1. **Diversity over quantity** -- Examples should cover the space of expected inputs, not just repeat one pattern. Connect to the ordering sensitivity finding: if ordering matters, then which examples you include matters even more.
2. **Format consistency** -- All examples should follow the exact output format you want. The label-flipping finding showed format matters more than labels. Use this to your advantage: consistent format is a stronger signal than many examples.
3. **Difficulty calibration** -- Examples should be representative of the real task difficulty. Too-easy examples set the wrong baseline. Too-hard examples may confuse.

Brief example: sentiment classification with 3 diverse examples (one clearly positive, one clearly negative, one nuanced/mixed) vs 3 examples that are all clearly positive or negative. The diverse set handles edge cases better.

#### 6. Check 1 (Predict-and-verify)

Present a prompt construction challenge. Given a task (summarize a technical document for a non-technical audience), the student sees three prompt variants:
- Variant A: Just the task instruction + document
- Variant B: Task instruction + format specification (3 bullet points, no jargon) + document
- Variant C: Task instruction + role ("technical writer for a general audience") + format specification + one example summary + document

Predict: rank the three from least to most reliable output. Which components matter most for this task?

Reveal: C > B > A, but the gap between B and C is smaller than between A and B. Format specification provides the biggest single improvement for this task. The role and example add incremental value. Key insight: not every prompt needs every component. "Knowing which components to include for a given task is the engineering judgment."

#### 7. Elaborate: RAG and Context Augmentation

**Transition:** "Every technique so far works with information already in the model's parameters -- knowledge from pretraining, behavior from SFT. But what if the answer requires information the model does not have?"

**Problem:** The model's knowledge is frozen at its training cutoff. It cannot answer questions about recent events, private documents, or domain-specific data not in its training corpus.

**Solution:** Put the relevant information in the prompt. If the prompt is a program, retrieved documents are the data the program operates on. The model's attention attends to the retrieved text just like it attends to few-shot examples -- same mechanism, different content.

**RAG example (positive 3):** Question about a recent event, with and without a retrieved document. Without: the model confabulates or says it does not know. With: the model answers correctly, citing the retrieved text. The Q vectors from the question attend to relevant K/V entries in the retrieved document.

**Conceptual diagram:** RAG as a two-step process: (1) Retrieve relevant documents (search step -- outside the model), (2) Place them in the prompt as context (attention step -- inside the model). "RAG is not a model feature. It is a prompt engineering pattern. You are augmenting the prompt with retrieved context so that attention has relevant tokens to attend to."

Software engineering analogy: dependency injection. Instead of hardcoding data into the model (pretraining), you provide it at runtime (in the prompt). The model's "function" (forward pass) operates on whatever data you inject.

**Negative example (context stuffing):** Show the 5-relevant + 15-irrelevant scenario. Accuracy drops. Why: attention distributes weight across all tokens. Irrelevant tokens dilute the attention to relevant passages. "Retrieval quality matters more than retrieval quantity. The best RAG system retrieves few, highly relevant documents -- not everything tangentially related."

**Misconception 5 (RAG solves hallucination):** Show a case where the retrieved document contains the correct answer but the model's response contradicts or goes beyond it. "The model still runs the same forward pass. Attention may attend to the retrieved document, but it also attends to its parametric knowledge. When parametric knowledge and retrieved context conflict, the model does not always prefer the context. RAG reduces hallucination by making relevant information available to attention. It does not eliminate hallucination because attention is not guaranteed to attend to the right tokens with sufficient weight."

**Misconception 2 (one best prompt):** Now that the student has seen multiple techniques, address the temptation to create one "ultimate prompt template." Show that different models and different tasks benefit from different technique combinations. The principles (structure, specificity, format consistency) transfer; specific templates do not. "The techniques are tools. Prompt engineering is knowing which tools to use for which job. There is no universal prompt -- there is understanding of what each component does and when it helps."

#### 8. Check 2 (Transfer question)

A developer is building a customer support chatbot. They have:
- A database of 10,000 support articles
- A requirement to respond in a specific JSON format (status, answer, sources, confidence)
- Users who ask questions about products the model was not trained on

Design the prompt structure. Which components do you need? Why? What are the risks?

Expected answer: (1) System/role prompt to set customer support behavior and tone. (2) Format specification with the JSON schema, ideally with one example. (3) RAG to retrieve relevant support articles based on the user's question. (4) The user's question as input. Risks: retrieval may return irrelevant articles (context stuffing problem), the model may hallucinate beyond what the articles say (RAG does not eliminate hallucination), the JSON format may break for unusual questions (format constraints are probabilistic, not guaranteed). The student should recognize this as a composition of the techniques they just learned, not a new concept.

#### 9. Practice (Notebook exercises)

**Notebook:** `notebooks/5-2-2-prompt-engineering.ipynb`

**Exercise structure:** 4 exercises, independently completable but sharing a theme (systematic prompt construction). Each exercises one or more techniques from the lesson.

- **Exercise 1 (Guided): Format specification.** Given a paragraph of text, construct prompts that extract structured data in three formats: (a) bullet points, (b) JSON, (c) markdown table. Start with a conversational prompt, observe inconsistency. Then add format specification progressively. Predict-before-run: predict the output format before running each prompt variant. Insight: format tokens in the prompt directly constrain the output distribution. Scaffolding: first format (bullet points) is fully worked; student constructs the JSON and markdown table variants.

- **Exercise 2 (Supported): Role prompting effects.** Take a code snippet with multiple issues (security vulnerability, performance problem, style issue). Write three prompts with different roles: (a) security auditor, (b) performance engineer, (c) code style reviewer. Compare which issues each role identifies. Then write a "combined" role and compare. Insight: role prompts bias attention toward specific features of the input; combining roles is possible but dilutes focus. Scaffolding: first role prompt is provided, student writes the other two.

- **Exercise 3 (Supported): Few-shot example selection.** Given a text classification task, compare accuracy with: (a) 3 random examples, (b) 3 diverse examples covering different categories, (c) 3 examples all from the same category, (d) 5 random examples. Run 5 trials each. Plot accuracy. Insight: example diversity matters more than quantity, consistent with the attention mechanism (diverse examples create richer K/V patterns for retrieval). Scaffolding: example selection function and evaluation loop provided; student designs the example sets and interprets results.

- **Exercise 4 (Independent): Build a structured prompt.** Given a real task (e.g., generate a meeting summary from raw notes), design a complete structured prompt using at least 3 techniques from the lesson. Write the prompt, test it on 3 different inputs, evaluate consistency. Reflect: which components contributed most? What would you change for a different task? Insight: prompt engineering is composing techniques deliberately, with each component serving a specific function.

#### 10. Summarize

Key takeaways:
- Prompt engineering is programming, not conversation. The prompt is a structured program with identifiable components, each shaping the attention pattern.
- Format specification constrains the output distribution. Explicit output schemas produce consistent, parseable results.
- Role/system prompts bias attention toward relevant features of the input. They shape focus, not knowledge.
- Few-shot example selection is about diversity and format consistency, not quantity.
- RAG extends the prompt's information by adding retrieved documents to the context. The model attends to them via the same attention mechanism. Retrieval quality matters more than quantity.
- Every technique works because of attention. Understanding the mechanism lets you reason about which techniques will help for a given task.

Mental model echo: "The prompt is a program; attention is the interpreter. Prompt engineering is writing better programs."

#### 11. Next Step

"You can now construct structured prompts that reliably control model behavior. But there is a class of problems where even the best-structured prompt fails: problems that require more computation than a single forward pass provides. A sentiment classification fits in one pass. A multi-step math problem does not. What happens when you ask the model to 'think step by step'? The next lesson explains why chain-of-thought works -- and it is not because the model 'decides' to think harder."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: concrete example, verbal/analogy, visual, intuitive, symbolic/code)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 2-3 new concepts (within limit)
- [x] Every new concept connected to at least one existing concept (ICL from 5.2.1, attention from 4.2, SFT from 4.4.2, ICL limitations from 5.2.1, causal masking from 4.2.6)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-16 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings. The student would not be lost or form an incorrect mental model. However, four improvement findings identify places where the lesson is significantly weaker than it could be. Another pass after fixes.

### Findings

#### [IMPROVEMENT] — Misconception 3 lacks a concrete negative example in the lesson

**Location:** Section 5B (Role and System Prompts), the GradientCard "What Role Prompts Do NOT Do"
**Issue:** The planning document specifies misconception 3 ("System prompts / role prompts give the model special abilities") should be addressed with a concrete negative example: "Give the model a factual question it cannot answer... Try it with no role, then with 'You are a world-class expert.' The model does not suddenly know the answer -- it may generate a more confident-sounding wrong answer." The built lesson describes this scenario in prose ("Try asking a factual question outside the model's training data with and without an expert role") but does not show the actual example. It is told, not shown. The student reads a description of what would happen rather than seeing a concrete before/after demonstration.
**Student impact:** The student intellectually understands the claim but has not seen it concretely. The misconception is addressed at the verbal level but lacks the concrete negative example that would make it stick. This is the difference between "I was told role prompts don't add knowledge" and "I saw a role prompt produce a more confident wrong answer." The latter is more durable.
**Suggested fix:** Add a brief before/after concrete example: show a specific factual question (e.g., "What were the exact revenue figures in Acme Corp's Q3 2024 earnings report?"), show the model without a role producing "I don't have access to that information," then show the model with an "expert financial analyst" role producing a confident-sounding but fabricated answer. Could be a ComparisonRow or inline code-style example. Keep it brief -- 4-6 lines total.

#### [IMPROVEMENT] — Few-shot example selection section lacks a concrete worked example

**Location:** Section 5C (Few-Shot Example Selection)
**Issue:** The planning document specifies a concrete example for this section: "Brief example: sentiment classification with 3 diverse examples (one clearly positive, one clearly negative, one nuanced/mixed) vs 3 examples that are all clearly positive or negative. The diverse set handles edge cases better." The built lesson presents the three principles (diversity, format consistency, difficulty calibration) as GradientCards with explanatory text, but never shows a concrete worked example demonstrating the principles. The format specification section had a code example. The role prompting section had a ComparisonRow. This section has only abstract explanation.
**Student impact:** The student reads about diversity > quantity but does not see it demonstrated in the lesson itself. The notebook exercises will demonstrate it empirically, but the lesson's own explanation lacks the concrete grounding that makes the other technique sections effective. The section feels thinner than the preceding two technique sections because it lacks a concrete example at the same level of specificity.
**Suggested fix:** Add a brief concrete example between the three GradientCards and the aside. Show sentiment classification with 3 diverse examples (positive, negative, mixed) vs 3 same-polarity examples. Even a 3-4 sentence description with specific example texts would ground the principle. Does not need to be a full ComparisonRow -- could be a short inline comparison.

#### [IMPROVEMENT] — RAG section's "hallucination despite context" claim is not demonstrated concretely

**Location:** Section 7 continued, the GradientCard "RAG Does Not Solve Hallucination"
**Issue:** The planning document specifies misconception 5 with a concrete negative example: "Show a RAG scenario where the retrieved document is present in the context, but the model's answer contradicts or goes beyond what the document says." The built lesson describes this in abstract terms ("It may blend retrieved information with parametric knowledge, over-generalize from the retrieved text, or simply not attend to the relevant passage strongly enough") but does not show a concrete scenario where this actually happens. All three mechanisms are listed but none is demonstrated.
**Student impact:** The student learns that RAG does not eliminate hallucination as an abstract claim, but lacks a concrete instance that makes the claim visceral. Contrast with the "context stuffing" section immediately above, which has a concrete ComparisonRow (5 relevant vs 5+15). The hallucination misconception is arguably more important for real-world practice but is treated more abstractly.
**Suggested fix:** Add a brief concrete scenario: e.g., a retrieved document says "Revenue was $4.2M in Q3" and the model's response says "Revenue exceeded $5M in Q3, showing strong growth" -- blending the retrieved figure with parametric priors about what "strong growth" looks like. Could be a 2-3 sentence inline example or a small ComparisonRow showing "Document says X, model outputs Y."

#### [IMPROVEMENT] — Causal masking not mentioned despite being relevant to prompt ordering

**Location:** Section 4 (The Prompt as a Composable Program) and the prompt anatomy diagram
**Issue:** The planning document lists causal masking (4.2.6) as a relevant connection: "Explains why prompt ordering matters. Earlier tokens attend to less context. Important context should come early (system prompt, role) so that all subsequent tokens can attend to it." The built lesson shows the prompt anatomy diagram with sections ordered top-to-bottom (system/role first, user input last) and labels "token order" with a downward arrow, but never explains WHY this ordering matters. The student might wonder: "Does the order of these sections matter? Could I put the user input first and the system prompt last?"
**Student impact:** The student sees the ordered structure but lacks the mechanistic explanation for why that order is preferred. The connection to causal masking is a valuable reinforcement opportunity -- the student has this concept at DEVELOPED depth and it directly explains why system prompts go first. Without this connection, the prompt anatomy feels like a convention rather than a consequence of the mechanism.
**Suggested fix:** Add 1-2 sentences near the prompt anatomy diagram or in the aside connecting the ordering to causal masking: "Because of causal masking, each token can only attend to tokens before it. System/role tokens placed first are visible to every subsequent token. User input tokens placed last can attend to everything above. This is why prompt order matters -- it is not convention, it is a consequence of the attention mechanism."

#### [POLISH] — Notebook markdown cells use spaced em dashes throughout

**Location:** Notebook `5-2-2-prompt-engineering.ipynb`, multiple markdown cells
**Issue:** The writing style rule requires no spaces around em dashes (`word—word` not `word — word`). The notebook's markdown cells consistently use spaced em dashes: "systematic prompt construction — treating prompt design," "format tokens anchor the output distribution — and observe," "self-contained for Google Colab — self-contained," etc. Over 30 instances across the notebook.
**Student impact:** Minor style inconsistency between lesson and notebook. Not pedagogically harmful.
**Suggested fix:** Replace ` — ` with `—` throughout notebook markdown cells (not in code comments or string literals where they serve as visual separators in terminal output).

#### [POLISH] — Summary item 6 is redundant with the lesson's thesis statement

**Location:** Section 10 (Summary), the last SummaryBlock item
**Issue:** The sixth summary item ("The prompt is a program; attention is the interpreter. Prompt engineering is writing better programs.") restates the lesson's header description and the core thesis verbatim. It does not add a new takeaway. The five preceding items are each about specific techniques or principles. The sixth is a meta-statement that wraps them up but reads as repetitive rather than synthesizing.
**Student impact:** Minor -- the student may feel the summary is one item too long. The first five items are strong and actionable. The sixth feels like a closing flourish rather than a distinct takeaway.
**Suggested fix:** Consider removing item 6 and letting the mental model echo live only as the implicit thread connecting items 1-5. Alternatively, make item 6 more of a synthesis: "These techniques are not independent tricks -- they all work through the same attention mechanism, which means you can reason about which techniques will help for a given task from first principles."

#### [POLISH] — Prompt Anatomy Diagram: software analogies displayed as "= import statement" may confuse

**Location:** Section 4, PromptAnatomyDiagram SVG
**Issue:** Each section in the diagram shows a software analogy as `= import statement`, `= function name`, `= type signature`, etc. The `=` prefix makes these read as mathematical equivalence statements rather than loose analogies. Combined with the aside that also lists the analogies, the student encounters the same mapping twice in slightly different wording (e.g., diagram says "= import statement", aside says "Role prompt = import (brings context into scope)").
**Student impact:** Minor. The double-presentation could be seen as reinforcement. The `=` notation is slightly overstrong for an analogy.
**Suggested fix:** Consider changing the diagram prefix from `=` to `~` or removing it and using a subtitle "Software analogy:" prefix instead. Low priority.

### Review Notes

**What works well:**
- The narrative arc is strong. The lesson flows naturally from the ICL recap through the hook, the anatomy framework, the individual techniques, the checkpoints, RAG, and the summary. The student never feels lost about where they are or where they are going.
- The hook (before/after contrast with the invoice extraction task) is effective and directly addresses misconception 1 ("prompting is phrasing") at the right moment.
- The "No Universal Template" section is well-placed and addresses misconception 2 at exactly the right point (after the student has seen all techniques and might be tempted to combine them into one template).
- The two checkpoints are genuinely predict-before-reveal and test transfer, not just recall.
- The software engineering analogies (format = type signature, role = import, examples = unit tests, RAG = dependency injection) land well for the target student and are consistently threaded through the lesson.
- Cognitive load is appropriate for a BUILD lesson. No paradigm shift, just systematic application of existing understanding.
- The notebook is thorough, well-scaffolded, and follows the Guided -> Supported -> Supported -> Independent progression correctly. Exercise 1 is genuinely guided with the bullet-point format fully worked. Exercise 4 is genuinely independent with no skeleton.

**Patterns in findings:**
- Three of the four improvement findings share the same pattern: the lesson describes or claims something that the planning document intended to be demonstrated with a concrete example, but the built lesson presents it abstractly instead. The lesson is strong on framework and explanation but occasionally misses the "show, don't tell" standard that makes the better sections (format specification, role prompting ComparisonRow, RAG before/after) effective. Adding 2-4 concrete sentences to each of these three locations would meaningfully strengthen the lesson without increasing cognitive load.

**Modality check:**
- Verbal/Analogy: Present (software engineering mapping, strong)
- Visual: Present (PromptAnatomyDiagram SVG)
- Symbolic/Code: Present (pseudocode, CodeBlock)
- Concrete example: Present but uneven (strong for format spec and role prompting, weaker for example selection and RAG hallucination)
- Intuitive: Present in recap and asides ("you already know the why, this teaches the how")
- Count: 5 modalities present. Meets the 3+ requirement.

---

## Review — 2026-02-16 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All four improvement findings from iteration 1 have been resolved properly. No new critical or improvement issues were introduced by the fixes. The lesson is ready to ship.

### Iteration 1 Resolution Check

**Finding 1 (IMPROVEMENT): Misconception 3 lacks concrete negative example**
**Status: RESOLVED.** A ComparisonRow was added inside the "What Role Prompts Do NOT Do" GradientCard (Section 5B). The left side shows a factual question about Acme Corp's Q3 earnings with no role -- model responds honestly ("I don't have access to Acme Corp's specific financial data"). The right side shows the same question with an "Expert Financial Analyst" role -- model confabulates a confident, specific, entirely fabricated answer ("$847M in Q3 revenue, reflecting 12% YoY growth"). The contrast is vivid and concrete. The callback to "SFT teaches format, not knowledge" is explicit. Fix quality: strong.

**Finding 2 (IMPROVEMENT): Few-shot section lacks concrete worked example**
**Status: RESOLVED.** A concrete sentiment classification example was added inside a muted background container with a ComparisonRow. Left side: 3 same-polarity examples (all positive) classify a mixed-sentiment test input as "Positive" (wrong). Right side: 3 diverse examples (positive, negative, mixed) correctly classify the same input as "Mixed" or "Negative." An explanatory paragraph below connects back to attention (K/V patterns for nuanced inputs). The section now matches the specificity level of the format specification and role prompting sections. Fix quality: strong.

**Finding 3 (IMPROVEMENT): RAG hallucination claim not demonstrated concretely**
**Status: RESOLVED.** A ComparisonRow was added inside the "RAG Does Not Solve Hallucination" GradientCard. Left (emerald): retrieved document says "Q3 revenue of $4.2M, a 3% decline from Q2." Right (rose): model outputs "Q3 revenue exceeded $5M, reflecting strong growth driven by new product launches." The explanation identifies the mechanism: "the model blended the retrieved figure with parametric priors about what earnings reports typically say." The follow-up paragraph clearly distinguishes "reduces" from "eliminates." Fix quality: strong.

**Finding 4 (IMPROVEMENT): Causal masking not mentioned despite relevance**
**Status: RESOLVED.** A paragraph was added in Section 4 after the attention mechanism paragraph and before the pseudocode. It explicitly connects prompt ordering to causal masking: "Because of causal masking, each token can only attend to tokens before it. System and role tokens placed first are visible to every subsequent token... This is why prompt order matters: it is not convention, it is a consequence of the attention mechanism you learned in Decoder-Only Transformers." The connection is explicit, concise, and flows naturally. Fix quality: strong.

**Polish 1 (notebook em dashes): Resolved.** Notebook markdown cells now use unspaced em dashes consistently.

**Polish 2 (summary item 6): Partially resolved.** The description was revised to be more synthesis-oriented ("Understanding the mechanism lets you reason about which techniques will help for a given task, rather than memorizing templates"), which is an improvement. The headline still echoes the thesis verbatim, but the description now adds reasoning value rather than being purely redundant.

**Polish 3 (diagram "=" notation): Not changed.** This was marked low priority and the "=" notation can be read as a mapping rather than strict equivalence. Acceptable as-is.

### Findings

#### [POLISH] — GradientCard "What Role Prompts Do NOT Do" is content-dense

**Location:** Section 5B (Role and System Prompts), lines 584-619
**Issue:** The GradientCard now contains a prose introduction, a nested ComparisonRow with 3 items per side, and two follow-up paragraphs with a callback to SFT. This is the densest single card in the lesson. It works because the content is well-organized (setup → demonstration → explanation), but it is noticeably heavier than any other GradientCard.
**Student impact:** Minor. The student may need to slow down through this card, but the content is sequential and well-signposted. The density is justified because the misconception (roles add knowledge) is important and the concrete demonstration requires showing two different model behaviors.
**Suggested fix:** No change needed. If the density ever feels problematic, the ComparisonRow could be moved outside the GradientCard into its own Row, but this would break the narrative flow of "here's the misconception, here's the disproof, here's why." Current structure is justified.

### Review Notes

**What works well (building on iteration 1 observations):**
- All four iteration 1 fixes strengthen the lesson in exactly the ways predicted. The "show, don't tell" pattern identified in iteration 1's review notes has been addressed: the three technique sections (format, role, examples) and the RAG section now all have concrete demonstrations at comparable levels of specificity. The lesson's concrete example coverage is now even across all sections.
- The causal masking paragraph in Section 4 is a particularly clean fix. It adds a single paragraph that connects prompt ordering to a concept the student already has at DEVELOPED depth (causal masking from 4.2.6), turning what felt like convention into a mechanistic consequence. This makes the prompt anatomy diagram's "token order" arrow meaningful rather than decorative.
- The nested ComparisonRows inside GradientCards (role misconception, RAG hallucination) are well-structured. Each follows the same pattern: setup prose → ComparisonRow demonstration → explanatory prose. This consistency helps the student process the denser sections.
- The notebook em dash fixes improve consistency between the lesson and notebook. The writing style is now uniform across both artifacts.

**Modality check (updated from iteration 1):**
- Verbal/Analogy: Present (software engineering mapping, strong)
- Visual: Present (PromptAnatomyDiagram SVG)
- Symbolic/Code: Present (pseudocode, CodeBlock)
- Concrete example: Now evenly present across all sections (format spec, role prompting, example selection, RAG, RAG hallucination)
- Intuitive: Present in recap and asides
- Count: 5 modalities present. Meets the 3+ requirement. Concrete examples now at full strength.

**Overall assessment:** The lesson is pedagogically sound. It teaches prompt engineering as structured programming grounded in the attention mechanism, using the student's software engineering background as a bridge. All 5 misconceptions are concretely addressed. All 3 positive examples and 3 negative examples are specific and well-placed. The notebook is well-scaffolded with appropriate progression. The cognitive load is appropriate for a BUILD lesson. Ready to ship.
