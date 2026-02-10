# Lesson: Instruction Tuning (SFT)

**Module:** 4.4 (Beyond Pretraining)
**Position:** Lesson 2 of 5
**Type:** Hands-on (notebook: `4-4-2-instruction-tuning.ipynb`)
**Cognitive load:** STRETCH

---

## Phase 1: Orient -- Student State

The student arrives from Lesson 1 of this module having adapted GPT-2 for classification by adding a classification head, freezing the backbone, and training on SST-2 sentiment data. They have the mental model "a pretrained transformer is a text feature extractor" and understand frozen-backbone vs full finetuning tradeoffs. They also have extensive background from Modules 4.1-4.3: building GPT from scratch, training it, loading real weights, and generating coherent text.

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Classification finetuning (add head, freeze backbone, train) | DEVELOPED | finetuning-for-classification (4.4.1) | Core pattern from previous lesson. Student implemented GPT2ForClassification. Explicit bridge from CNN transfer learning. |
| Last-token hidden state as sequence representation for causal models | DEVELOPED | finetuning-for-classification (4.4.1) | Motivated by causal masking: last token has full sequence context. BERT contrast (bidirectional uses first token). |
| Frozen backbone vs full finetuning tradeoffs | INTRODUCED | finetuning-for-classification (4.4.1) | ComparisonRow. Overfitting argument (124M params on small dataset). Catastrophic forgetting risk with full finetuning. |
| Catastrophic forgetting | INTRODUCED | finetuning-for-classification (4.4.1) | Frozen backbone preserves generation ability. Aggressive full finetuning can destroy general language capabilities. Observable in notebook. |
| Autoregressive generation (next-token prediction + sampling loop) | DEVELOPED | building-nanogpt (4.3.1) | The student's generate() method. torch.no_grad(), crop, forward, sample last position, append. |
| Language modeling as next-token prediction P(x_t | x_1,...,x_{t-1}) | DEVELOPED | what-is-a-language-model (4.1.1) | The defining objective. Self-supervised labels. Connected to supervised learning framework. |
| Cross-entropy loss for next-token prediction | DEVELOPED | pretraining (4.3.2) | Reshape (B, T, V) to (B*T, V). Initial loss sanity check. Same formula regardless of vocabulary size. |
| Complete GPT training loop (forward, loss, backward, step) | APPLIED | pretraining (4.3.2) | Student wrote the full loop with LR scheduling and gradient clipping. "Same heartbeat, new instruments." |
| Tokenization (BPE, tiktoken) | APPLIED | tokenization (4.1.2) | Built BPE from scratch. Merge table IS the tokenizer. |
| HuggingFace transformers library (minimal) | INTRODUCED | loading-real-weights (4.3.4) | from_pretrained("gpt2"). Used only as weight source. Not deeply explored. |
| Weight tying (embedding = output projection) | DEVELOPED | building-nanogpt (4.3.1), finetuning-for-classification (4.4.1) | data_ptr() verified. Classification finetuning breaks the tie by replacing lm_head. |
| "A pretrained transformer is a text feature extractor" | -- | finetuning-for-classification (4.4.1) | Key mental model. Same pattern as CNN transfer learning. |
| "The architecture is the vessel; the weights are the knowledge" | -- | loading-real-weights (4.3.4) | Same code, different weights, different behavior. |

**Mental models already established:**
- "A pretrained transformer is a text feature extractor" (classification finetuning)
- "Hire experienced, train specific" (CNN transfer learning, extended to transformers)
- "Same heartbeat, new instruments" (training loop structure)
- "The architecture is the vessel; the weights are the knowledge"
- "A language model approximates P(next token | context)"
- "Cross-entropy doesn't care about vocabulary size"
- "Causal masking simulates the inference constraint during training"
- "Of course you use the last token" (causal model sequence representation)

**What was explicitly NOT covered that is relevant here:**
- Instruction tuning / SFT (explicitly deferred from Lesson 1's "Next Step" teaser: "what if you want a model that can follow ANY instruction?")
- Instruction dataset format (instruction-response pairs)
- Chat templates and special tokens
- The conceptual distinction between narrow-task adaptation (classification) and general instruction-following
- How ChatGPT differs from GPT-2 at the training level
- The idea that SFT teaches FORMAT not knowledge

**Readiness assessment:** The student is well-prepared but this is a genuine conceptual stretch. They know finetuning mechanics (same training loop, different data), but the SHIFT from "narrow task with a new head" to "general instruction following with the SAME head (lm_head)" is genuinely new. The insight that you do not need a new head -- you just need different training data -- is the conceptual leap. The mechanics are familiar; the framing is novel. This justifies STRETCH.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to explain how supervised finetuning on instruction-response pairs transforms a text-completing base model into an instruction-following assistant, and to perform SFT on a small instruction dataset using the same training loop they already know.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Classification finetuning (add head, freeze backbone, train) | INTRODUCED | DEVELOPED | 4.4.1 | OK | Student needs to understand how classification finetuning works so they can contrast it with SFT. They exceed the requirement. |
| Language modeling as next-token prediction | DEVELOPED | DEVELOPED | 4.1.1 | OK | The core framing. SFT is just next-token prediction on a different dataset. Student needs to deeply understand the base model objective. |
| Cross-entropy loss for next-token prediction | DEVELOPED | DEVELOPED | 4.3.2 | OK | Same loss function used in SFT. Student trained GPT from scratch with this loss. |
| Complete GPT training loop | DEVELOPED | APPLIED | 4.3.2 | OK | SFT uses the same loop. Student exceeded requirement by implementing it from scratch. |
| Autoregressive generation | DEVELOPED | DEVELOPED | 4.3.1 | OK | Student needs to understand generation to appreciate the before/after behavioral difference. |
| Tokenization (BPE) | INTRODUCED | APPLIED | 4.1.2 | OK | Student needs to understand tokenization to understand chat templates and special tokens. |
| Catastrophic forgetting | INTRODUCED | INTRODUCED | 4.4.1 | OK | SFT carries catastrophic forgetting risk. Student was introduced to it in classification context. Will be referenced, not re-taught. |
| HuggingFace transformers | INTRODUCED | INTRODUCED | 4.3.4 | OK | Notebook will use HuggingFace for loading model. Minimal familiarity is sufficient. |

All prerequisites OK. No gaps to resolve.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Instruction-tuned models know more than base models" | The behavioral difference is dramatic -- ChatGPT answers questions, GPT-2 just completes text. It LOOKS like ChatGPT has more knowledge. And the student just saw finetuning ADD something (a classification head). Natural inference: SFT adds knowledge. | Same factual question posed to both base model (generates relevant continuation) and SFT model (gives a direct answer). The knowledge was already there -- the base model just expressed it as text completion rather than as a direct answer. Show a base model completing "The capital of France is" -> "Paris" -- it KNOWS the answer, it just does not follow the instruction "What is the capital of France?" | Core concept section, after showing base model vs chat model behavior comparison. This is the central misconception the whole lesson must defeat. |
| "SFT requires a different training objective or loss function" | Classification finetuning replaced the head and used a different loss target (class labels instead of next tokens). The student might expect SFT to also require a different mechanism -- perhaps a special "instruction loss" or "helpfulness loss." | The training code is identical to pretraining: next-token prediction with cross-entropy. The ONLY difference is the training data. Show the training loop side-by-side: pretraining loop and SFT loop. They are the same. | After establishing what SFT data looks like, when showing the actual training mechanics. |
| "You need to train on millions of instruction examples" | Pretraining uses billions of tokens. The student might assume SFT also needs massive data. This is reinforced by classification finetuning using thousands of examples (SST-2 ~67K). | Alpaca used only 52K instruction examples. LIMA showed 1,000 high-quality examples can match much larger datasets. The entire behavioral transformation from "text completer" to "instruction follower" happens with surprisingly little data, because the model already has the knowledge -- SFT only teaches it the format. | After showing the instruction dataset format, when discussing dataset size. |
| "Chat templates are cosmetic formatting, not functionally important" | Chat templates look like string formatting -- just wrapping text in XML-like tags. The student may think they are documentation conventions rather than essential structure the model depends on. | Wrong template experiment: use the wrong template format with an instruction-tuned model and observe degraded responses. The model was trained to recognize specific token boundaries. Using wrong delimiters means the model cannot distinguish where the instruction ends and the response should begin. | During the chat template section. |
| "SFT and classification finetuning are fundamentally different processes" | Different behavioral outcomes (narrow classification vs general instruction following), different data formats (text/label pairs vs instruction/response pairs), different model architectures (classification head vs lm_head). Feels like two completely different techniques. | The training loop is identical: forward pass, compute loss, backward, step. Classification finetuning trains a NEW head on a NARROW task. SFT trains the EXISTING lm_head on a BROAD task. Both are supervised learning on a pretrained backbone. The distinction is what you train (new head vs existing head) and what the data looks like (labels vs response text). | Hook section, as the bridge from Lesson 1. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Base GPT-2 vs instruction-tuned model on "Write a haiku about machine learning" | Positive | The hook. Shows the dramatic behavioral difference that motivates the entire lesson. Base model continues the text ("Write a haiku about machine learning and submit it to..."). Instruction-tuned model produces a haiku. Same weights (mostly), radically different behavior. | Maximum contrast with minimum complexity. A creative task makes the difference visceral -- the base model clearly does not understand it is being asked to DO something. |
| "The capital of France is ___" as base completion vs "What is the capital of France?" as instruction | Positive | Proves the base model already HAS the knowledge but expresses it differently. This is the key evidence for "SFT teaches format, not knowledge." The base model completes "Paris" after the right prompt; it just cannot follow the question format. | Directly attacks the central misconception. Two prompts, same underlying knowledge, different expression format. The student should have an "aha" moment: the knowledge was always there. |
| An instruction dataset entry with system/user/assistant roles and chat template tokens | Positive | Shows the concrete data format the model trains on. Demystifies instruction datasets. The student sees it is just text with special structure. | The student needs to see what an actual training example looks like before they can understand SFT mechanics. Concrete before abstract. |
| Classification finetuning vs SFT side-by-side comparison | Negative (boundary) | Clarifies what SFT is NOT: it is not adding a new head. It is training the existing next-token prediction head on differently formatted data. The architecture does not change. This is the key contrast with Lesson 1. | Draws the boundary between two forms of finetuning. The student just did classification finetuning -- they need to understand how SFT differs structurally (no new head, same loss, different data). |
| Wrong chat template producing degraded output | Negative | Proves chat templates are functional, not cosmetic. The model's behavior degrades when the template does not match what it was trained on, because the special tokens serve as structural boundaries the model learned to use. | Directly disproves the "templates are just formatting" misconception. Observable failure makes the point concrete. |

---

## Phase 3: Design

### Narrative Arc

You just taught GPT-2 to classify movie reviews. You added a classification head, froze the backbone, and trained on labeled data. It works -- but it is a narrow adaptation. Your model can answer exactly one question: "Is this review positive or negative?" You cannot ask it to summarize a text, write a poem, or explain a concept. For each new task, you would need a new head, a new labeled dataset, and a new round of finetuning. Yet when you use ChatGPT, you type anything and it follows your instructions. It does not have a separate classification head for each possible task. How? The answer is both simpler and more surprising than you might expect. ChatGPT is not a fundamentally different architecture. It is GPT -- the same next-token predictor you built from scratch -- trained on a curated dataset of instruction-response pairs. No new head. No new loss function. The same cross-entropy on next-token prediction you used in the pretraining lesson. The only thing that changed is the data. This lesson is about that transformation: how a dataset of (instruction, response) pairs teaches a text completer to become an instruction follower, and why it works with surprisingly little data -- the model already knows everything from pretraining, it just needs to learn a new FORMAT for expressing what it knows.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | Side-by-side: send the same prompt to a base model and an instruction-tuned model. Base model continues text; instruction-tuned model follows the instruction. Real text showing the behavioral difference. | The behavioral gap between base and chat models is the motivating phenomenon. The student must SEE it before they can understand what causes it. Text output is the most natural modality for a language model lesson. |
| Verbal/analogy | "SFT teaches format, not knowledge." The base model is like a brilliant expert who only speaks in monologue -- they know everything but cannot hold a conversation. SFT does not give them new knowledge; it teaches them conversational form. Like teaching an expert to answer questions instead of giving lectures. | This analogy directly targets the central misconception (SFT adds knowledge). The expert/conversation framing is intuitive and memorable. |
| Visual | Diagram showing the three-stage pipeline: pretraining data (web text, books, code) -> SFT data (instruction/response pairs) -> resulting model behavior. The architecture stays the same across all three stages; only the data changes. Arrows showing that the architecture box is identical in pretraining and SFT. | The visual makes the "same model, different data" point structural. The student can see that no architectural change happens. The diagram also previews the pipeline that continues in Lesson 3 (RLHF adds a third stage). |
| Symbolic/code | The SFT training loop shown side-by-side with the pretraining loop from 4.3.2. Identical structure. The data loading and formatting are different, but loss = F.cross_entropy(logits, targets) is the same line. Also: chat template code showing how special tokens structure the training data. | The code comparison is the definitive proof that SFT uses the same training objective. The student can verify it line by line. Chat template code makes the abstract concept of "structured training data" concrete. |
| Intuitive | "Of course the data is enough." The student already knows the model has vast knowledge from pretraining. They already know finetuning can reshape behavior (classification). The leap is small: if you show the model thousands of examples of question -> answer format, it learns to produce answers when given questions. No magic -- just more of the same supervised learning. | The "of course" moment should follow naturally from the student's existing understanding. SFT is demystified: it is not a new technique, it is an application of the technique they already know. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3
  1. Instruction datasets as a data FORMAT (instruction/response pairs that turn pretraining's next-token prediction into instruction following)
  2. Chat templates and special tokens (structural markers that delimit roles and turns in conversation)
  3. The conceptual shift: SFT teaches format, not knowledge (the model already has the knowledge from pretraining; SFT teaches it HOW to express that knowledge in a conversational format)
- **Previous lesson load:** BUILD (finetuning-for-classification was low novelty, high connection to CNN transfer learning)
- **This lesson's load:** STRETCH -- appropriate. Three new concepts, but #1 and #3 are deeply intertwined (understanding the data format naturally leads to understanding that it teaches format, not knowledge). The mechanics (#1, #2) are familiar (same training loop, just different data). The conceptual shift (#3) is the genuinely demanding part. After a BUILD lesson, a STRETCH is appropriate per the load trajectory.

### Connections to Prior Concepts

| Prior Concept | How It Connects | Source |
|--------------|----------------|--------|
| Classification finetuning | The direct contrast. Classification finetuning = new head, narrow task, labeled data. SFT = same head, broad task, instruction-response data. Both are supervised learning on a pretrained backbone. | 4.4.1 |
| Next-token prediction as the training objective | SFT uses the SAME objective. This is the most important connection. The student might expect a new loss function; they get the same one. | 4.1.1, 4.3.2 |
| "Same heartbeat, new instruments" | Extended again. The SFT training loop is the same heartbeat. The only new instrument is the data format. | 4.3.2 |
| Cross-entropy loss | Same loss, same code, same mechanism. The only change: the targets are response tokens instead of "predict any next token from web text." | 4.3.2 |
| Catastrophic forgetting | SFT carries this risk too. Full finetuning on instruction data can degrade the model's general language abilities, just as classification finetuning can. But SFT mitigates it because instruction-response data is still natural language (unlike narrow classification). | 4.4.1 |
| Tokenization / BPE | Chat templates introduce special tokens. The student knows tokens are the atomic units the model sees. Special tokens are just new entries in the vocabulary that the model learns to recognize as structural markers. | 4.1.2 |
| "The architecture is the vessel; the weights are the knowledge" | Extended: same architecture, same vessel. SFT changes the weights slightly to reshape HOW the knowledge is expressed. | 4.3.4 |

**Potentially misleading prior analogies:**
- "A pretrained transformer is a text feature extractor" -- This framing from Lesson 1 emphasizes the backbone-as-feature-extractor + separate-head pattern. SFT does NOT use a separate head. The lesson must explicitly address this: in SFT, the model keeps its original lm_head because the task IS still next-token prediction -- just on a different kind of data. The "feature extractor" framing is a Lesson 1 concept that does not transfer directly here.
- "Hire experienced, train specific" -- This analogy implies training for a SPECIFIC task. SFT is training for a GENERAL capability (follow any instruction). The lesson should note that SFT is more like "teach the expert to hold conversations" than "teach the expert a specific job."

### Scope Boundaries

**This lesson IS about:**
- The behavioral difference between base models and instruction-tuned models
- Instruction dataset format (instruction/response pairs, system/user/assistant roles)
- Chat templates and special tokens as structural delimiters
- SFT mechanics: same training loop, same loss, different data
- Why SFT works: it teaches format, not knowledge
- The surprising efficiency of SFT (small datasets can produce large behavioral changes)
- Hands-on SFT on a small instruction dataset in the notebook

**This lesson is NOT about:**
- RLHF, DPO, or preference-based alignment (Lesson 3)
- LoRA or parameter-efficient finetuning (Lesson 4)
- Building production-quality instruction datasets (data curation is mentioned but not deep-dived)
- Multi-turn conversation handling in depth (mentioned that templates support it, not implemented)
- Constitutional AI or RLAIF (Series 5)
- Evaluation of instruction-tuned models (beyond qualitative observation)
- Prompt engineering or in-context learning
- Reinforcement learning from any signal

**Depth targets:**
- Instruction dataset format: DEVELOPED (student sees multiple examples, understands the structure, implements formatting in notebook)
- Chat templates and special tokens: DEVELOPED (student understands why they exist and uses them in code)
- "SFT teaches format, not knowledge": DEVELOPED (student can explain this with evidence, not just recite it)
- SFT training mechanics: APPLIED (student runs SFT in the notebook)
- Classification finetuning vs SFT distinction: DEVELOPED (student can articulate the structural differences)

### Lesson Outline

1. **Context + Constraints** (Row)
   - What we are doing: understanding how instruction-tuned models (like ChatGPT) differ from base models (like GPT-2), and performing SFT on a small instruction dataset
   - What we are NOT doing: RLHF, alignment, LoRA, evaluation -- those come in later lessons
   - The bridge from Lesson 1: "Classification finetuning adapts a model for ONE narrow task. Now: what if the task is 'follow any instruction'?"

2. **Hook -- The Before/After** (Row + Aside)
   - Demo: same prompt ("Write a haiku about machine learning") sent to GPT-2 base vs an instruction-tuned model
   - GPT-2 base: continues the text as if it were a document ("Write a haiku about machine learning. Haiku is a traditional form of Japanese poetry that...")
   - Instruction-tuned model: produces a haiku
   - The question: "These models have the same architecture. GPT-2 has 124M parameters. The instruction-tuned model started as a base model just like GPT-2. What changed?"
   - Prediction checkpoint: "Based on Lesson 1, you might guess: a new head was added and trained for instruction following. That is wrong. No new head. No new architecture. The answer is simpler."

3. **Explain -- What Base Models Actually Do** (Row + Aside)
   - A base model is a text completer. It predicts the next token given context. That is ALL it does.
   - "The capital of France is" -> base model completes "Paris." It KNOWS the answer.
   - "What is the capital of France?" -> base model continues as if this were a document: "What is the capital of France? This question is commonly asked in geography classes..."
   - Key insight: the base model has the knowledge but not the behavior. It does not understand that a question expects an answer. It understands that text continues with more text.
   - "SFT teaches format, not knowledge." The expert-in-monologue analogy: brilliant but cannot hold a conversation. SFT teaches conversational form.

4. **Explain -- The Instruction Dataset** (Row + Aside)
   - What SFT data looks like: instruction/response pairs
   - Show a concrete example with all fields visible:
     - Instruction: "Explain why the sky is blue in simple terms."
     - Response: "The sky appears blue because of a phenomenon called Rayleigh scattering..."
   - Show 3-4 diverse examples: a factual question, a creative task, a coding task, a summarization task
   - The dataset teaches the model a PATTERN: when you see text structured as an instruction, produce text structured as a response
   - Dataset size: Alpaca had 52K examples. LIMA showed 1,000 high-quality examples can suffice. Compare to pretraining: billions of tokens vs thousands of instruction pairs. The asymmetry reinforces "format, not knowledge."
   - Brief mention: data quality matters enormously. A few thousand high-quality examples can outperform millions of low-quality ones.

5. **Check 1 -- Predict and Verify** (Row)
   - "The SFT training loop uses a special instruction-following loss function, right? What loss function do you think SFT uses?"
   - Expected: The student should predict cross-entropy on next-token prediction (the same loss from pretraining), because SFT is still generating text token by token. The "response" tokens are the targets.
   - Reveal: Correct. Same cross-entropy, same objective. The only change is the data.

6. **Explain -- SFT Training Mechanics** (Row + Aside)
   - Side-by-side: pretraining loop (from 4.3.2) vs SFT loop
   - Identical structure: forward pass, cross-entropy loss, backward, step
   - What changes: the data pipeline. Instead of random web text chunks, each training example is a formatted instruction-response pair
   - Key detail: during SFT, loss is typically computed only on the RESPONSE tokens (not the instruction tokens). The model should learn to GENERATE responses, not to predict instruction tokens. This is called "masking the prompt" or "loss masking." Show concretely: tokens of the instruction get ignored in the loss, tokens of the response contribute to the loss.
   - No new head: lm_head stays. The model is still predicting the next token. It is just learning to predict instruction-appropriate next tokens.
   - "Same heartbeat" callback: the training loop is the same heartbeat you have written three times now (pretraining, classification, SFT).
   - Architecture diagram: same GPT architecture as pretraining. Token embedding -> transformer blocks -> layer norm -> lm_head -> logits. Nothing added, nothing removed. The classification head from Lesson 1 is GONE. We are back to the original architecture.

7. **Explain -- Chat Templates and Special Tokens** (Row + Aside)
   - Problem: the model needs to know where the instruction ends and the response should begin. In raw text, there is no boundary.
   - Solution: special tokens that act as structural delimiters
   - Show a concrete chat template (ChatML format):
     ```
     <|im_start|>system
     You are a helpful assistant.<|im_end|>
     <|im_start|>user
     Write a haiku about machine learning.<|im_end|>
     <|im_start|>assistant
     ```
   - These are real tokens in the vocabulary. The model learns that after seeing `<|im_start|>assistant\n`, it should generate a response.
   - Connection to tokenization: special tokens are added to the vocabulary (the student knows vocabulary is just a lookup table from 4.1.2). They do not exist in pretraining data, so they have no pretrained meaning -- they acquire meaning entirely from SFT.
   - Multiple template formats exist (ChatML, Llama format, Alpaca format). Each model family uses its own. The format matters because the model was trained to recognize specific delimiters.
   - Brief mention: multi-turn conversations are just longer templates with alternating user/assistant blocks. Same structure, repeated.

8. **Check 2 -- Transfer Question** (Row)
   - "You train a base model with SFT using the ChatML template. At inference time, you accidentally use Llama's template format instead. What happens and why?"
   - Expected: The model produces degraded or incoherent responses, because it was trained to recognize ChatML's special tokens as structural boundaries. Llama's tokens are unknown or have different learned associations. The model does not know where the instruction ends and the response begins.
   - This directly addresses the "templates are cosmetic" misconception.

9. **Explore -- Notebook** (Row)
   - Notebook exercises:
     - Load a small pretrained model (GPT-2 or a small open-source model)
     - Explore an instruction dataset (Alpaca or similar): examine entries, observe the instruction/response format
     - Implement chat template formatting: write the function that converts instruction/response pairs into properly templated text with special tokens
     - Tokenize formatted examples and inspect the token IDs (see the special tokens as integer IDs)
     - Implement prompt masking: create the labels tensor where instruction tokens are set to -100 (ignored by cross-entropy) and response tokens are the targets
     - Run SFT training for a small number of steps
     - Generate from the model before and after SFT: observe the behavioral shift from text completion to instruction following
     - Qualitative evaluation: try several different instructions and compare base vs SFT responses

10. **Elaborate -- Why Does SFT Work With So Little Data?** (Row + Aside)
    - The paradox: pretraining needs billions of tokens, but SFT works with thousands of examples. Why?
    - The answer connects to "format, not knowledge": pretraining must teach the model EVERYTHING about language, facts, reasoning, code, etc. This requires enormous data. SFT only needs to teach the model a new format for expressing what it already knows. Format is a much simpler pattern than knowledge.
    - LIMA result: 1,000 carefully curated examples can produce an instruction-following model competitive with models trained on 52K+ examples. Data quality > data quantity for SFT.
    - Connection to classification finetuning (Lesson 1): the classification head had only 1,536 parameters. It learned to classify with thousands of examples because the backbone already extracted good features. SFT is the same principle at the behavior level: the model already has good representations, SFT teaches it a new way to USE them.
    - Catastrophic forgetting callback: SFT on instruction data is less risky than classification finetuning, because instruction-response data is still natural language. The model does not forget how to generate text; it learns a new FORMAT for generating text. But excessive SFT or low-quality data can still cause problems.

11. **Elaborate -- Classification Finetuning vs SFT: The Full Picture** (Row)
    - ComparisonRow contrasting the two approaches:
      - Classification: adds a NEW head (nn.Linear), narrow task (sentiment, topic), data = (text, label), architecture CHANGES, trains the head on frozen backbone
      - SFT: keeps the ORIGINAL lm_head, broad task (follow any instruction), data = (instruction, response), architecture UNCHANGED, trains (some or all of) the original model on new data
    - Both are supervised learning on a pretrained backbone. The distinction: classification changes WHAT the model outputs (class labels instead of tokens). SFT changes HOW the model uses its existing output mechanism (instruction-appropriate tokens instead of document-continuation tokens).
    - This sets up Lesson 3: "SFT produces an instruction-following model. But does it follow instructions WELL? Does it refuse harmful requests? Is it truthful? SFT alone does not optimize for these qualities. That is what alignment addresses."

12. **Summarize** (Row)
    - Mental model: "SFT teaches format, not knowledge." The base model already has vast knowledge from pretraining. SFT on instruction-response pairs teaches it to express that knowledge in a conversational, instruction-following format.
    - Key insight: no new architecture, no new loss function. Same cross-entropy on next-token prediction. The only change is the data.
    - Chat templates are functional structure, not cosmetic formatting. Special tokens are the boundaries the model learns to recognize.
    - SFT is surprisingly data-efficient because format is simpler than knowledge.

13. **Next Step** (Row)
    - SFT produces a model that follows instructions. But following instructions is not enough. The model can be helpful but also harmful, verbose, sycophantic, or confidently wrong. It learned to follow the FORMAT of being helpful, but it has no training signal for what "helpful" actually means. Next: RLHF and alignment -- training the model to produce responses that humans actually prefer.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via records (classification finetuning from 4.4.1, training loop from 4.3.2, next-token prediction from 4.1.1, tokenization from 4.1.2)
- [x] Depth match verified: all OK
- [x] No untaught concepts remain
- [x] No multi-concept jumps in exercises (notebook builds incrementally: format data -> tokenize -> implement loss masking -> train -> evaluate)
- [x] No gaps to resolve

### Pedagogical Design
- [x] Narrative motivation stated as coherent paragraph (problem before solution: narrow classification -> general instruction following)
- [x] At least 3 modalities: concrete example (base vs chat model behavior), verbal/analogy (expert-in-monologue), visual (pipeline diagram), symbolic/code (training loop comparison, chat template code), intuitive ("of course the data is enough")
- [x] At least 2 positive examples (base vs chat model, instruction dataset entries, "capital of France" knowledge demo) + 1 negative (wrong chat template producing degraded output) + 1 boundary example (classification vs SFT comparison)
- [x] At least 3 misconceptions (5 identified) with negative examples
- [x] Cognitive load = 3 new concepts (at limit, but #1 and #3 are deeply intertwined)
- [x] Every new concept connected to existing concept (instruction datasets -> classification data format; SFT training -> pretraining loop; chat templates -> tokenization vocabulary; "format not knowledge" -> "feature extractor" mental model)
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues -- the lesson is structurally sound, covers all planned concepts, follows the narrative arc, and the student would not be lost or form a wrong mental model. However, four improvement-level findings would make the lesson significantly stronger.

### Findings

#### [IMPROVEMENT] -- Wrong-template negative example is hypothetical, not concrete

**Location:** Check 2 -- Transfer Question (Section 8, line 920-959)
**Issue:** The planning document calls for a negative example where the student observes degraded output from using the wrong chat template. In the built lesson, this is handled as a checkpoint question with a `<details>` reveal. The answer tells the student what *would* happen ("degraded or incoherent responses"), but the student never sees an actual example of wrong-template output. The misconception "chat templates are cosmetic" is addressed verbally but without a concrete negative example demonstrating the failure.
**Student impact:** The student reads that wrong templates cause problems and intellectually accepts it, but does not viscerally see the failure. The misconception is addressed at INTRODUCED depth (words only) rather than DEVELOPED depth (evidence-based). Compare to the "capital of France" example, which shows two concrete outputs. This negative example is weaker.
**Suggested fix:** Add a short concrete example showing what a model actually produces when given a mismatched template -- even a fabricated but realistic snippet (e.g., the model outputting garbled text or continuing the template tokens as if they were regular text). Alternatively, move this to the notebook as an exercise where the student tries it themselves and observes the degradation. Either approach would turn a hypothetical into a concrete demonstration.

#### [IMPROVEMENT] -- The "intuitive" modality is underdeveloped

**Location:** Section 10/11, "Why Does SFT Work With So Little Data?" and "Classification vs SFT" (lines 1028-1148)
**Issue:** The planning document lists an "intuitive" modality: *"'Of course the data is enough.' The student already knows the model has vast knowledge from pretraining. They already know finetuning can reshape behavior. The leap is small: if you show the model thousands of examples of question -> answer format, it learns to produce answers when given questions. No magic -- just more of the same supervised learning."* In the built lesson, the "Why Does SFT Work" section explains the data efficiency logically but never delivers a genuine "of course" moment. It repeats the "format is simpler than knowledge" argument (already stated twice before). The student reads the same reasoning for the third time rather than arriving at an independent "of course" insight.
**Student impact:** The section feels like a recap of the central insight rather than a deepening of it. The student already understood "format, not knowledge" from Section 3. Repeating it without a new angle does not push understanding deeper. The planned "of course" moment -- where the student independently recognizes that SFT is obvious given what they already know -- does not land.
**Suggested fix:** Reframe the section to deliver the intuition from the student's perspective. Something like: "You have trained models three times now. Each time, the model learned what its data taught it. Pretraining data was web text, so the model learned to complete web text. Classification data was labeled sentiment, so the model learned to output sentiment labels. Instruction data is instruction-response pairs, so... of course the model learns to produce responses to instructions. Same mechanism every time. The only question is why so little data suffices -- and you already know the answer: the knowledge is already there." This uses the student's own trajectory as the intuitive argument rather than restating the lesson's thesis.

#### [IMPROVEMENT] -- Narrative bridge from Lesson 1 could be more prominent

**Location:** Context section (lines 401-439)
**Issue:** The bridge from classification finetuning to SFT is in a TipBlock aside: "Classification finetuning adapts a model for one narrow task -- positive vs negative. Now: what if the task is 'follow any instruction'?" This is the motivating question for the entire lesson, but it lives in a sidebar that the student might skim or skip. The main ObjectiveBlock states the lesson's goal but does not frame it as the natural next question after Lesson 1. The planning document's narrative arc opens with this question front and center.
**Student impact:** A student reading quickly would see the objective ("understand SFT") without the motivating tension ("classification is narrow, what about general instruction-following?"). The aside is easy to miss, and the hook section (Section 2) does not explicitly reference the limitation of classification finetuning either -- it jumps to base model vs instruction-tuned model. The "why do I need this?" bridge is present but buried.
**Suggested fix:** Move the bridge into the main content flow rather than the aside. Before or within the hook, add a brief paragraph that explicitly states: "In the last lesson, you adapted GPT-2 for one specific task: sentiment classification. That required a labeled dataset and a new head for each task. But ChatGPT does not have a separate head for each possible task. How does one model follow any instruction?" This creates the motivating tension before the before/after demo.

#### [IMPROVEMENT] -- SFT Pipeline Diagram is placed after the concept, not before or alongside

**Location:** Section 10, "Why Does SFT Work With So Little Data?" (line 1069)
**Issue:** The SftPipelineDiagram is placed at the end of the "Why SFT Works" elaboration section. The planning document describes this visual as showing "the three-stage pipeline: pretraining data -> SFT data -> resulting model behavior" with "the architecture staying the same." This visual is most valuable early -- when the student first needs to understand the relationship between pretraining and SFT. By Section 10, the student has already read about this relationship through six prior sections. Placing the diagram this late means it confirms what the student already knows rather than helping build understanding.
**Student impact:** The diagram would have been most useful in Section 3 (What Base Models Actually Do) or Section 6 (SFT Training Mechanics), when the student is actively building the mental model of "same architecture, different data." By Section 10, it functions as decoration rather than scaffolding.
**Suggested fix:** Move the SftPipelineDiagram to Section 6 (SFT Training Mechanics), right after or alongside the ComparisonRow showing pretraining loop vs SFT loop. The diagram reinforces the same point visually at the moment the student is absorbing the mechanical details. In Section 10, replace it with nothing or with a simple textual callback to the diagram they already saw.

#### [POLISH] -- Spaced em dash in SVG diagram text

**Location:** SftPipelineDiagram, line 111
**Issue:** The SVG text reads: `Same GPT architecture throughout — only the data changes`. The em dash has spaces on both sides. The writing style rule requires no spaces around em dashes: `word—word`.
**Student impact:** Minor visual inconsistency. The rest of the lesson's prose correctly uses `&mdash;` without spaces.
**Suggested fix:** Change to `Same GPT architecture throughout—only the data changes`.

#### [POLISH] -- Code string em dash also spaced

**Location:** Instruction dataset CodeBlock, line 628
**Issue:** The response text `"A list is mutable — you can add, remove, and change ..."` uses a spaced em dash. This is inside a code block showing example dataset content, so it is a representation of training data rather than lesson prose.
**Student impact:** Negligible. This is inside a code example representing JSON data, and spaced em dashes are common in real datasets. Could be left as-is.
**Suggested fix:** Optional. If consistency is desired, change to an unspaced em dash, but this is defensible either way since it represents external data.

#### [POLISH] -- The SFT pipeline diagram's RLHF stage could cause premature questions

**Location:** SftPipelineDiagram (lines 57-265)
**Issue:** The diagram shows three stages: Pretraining, SFT, and RLHF. The RLHF stage is grayed out and labeled "NEXT LESSON," which is a good scope boundary signal. However, it labels RLHF's behavior as "Aligned assistant" and its data as "Human preferences" -- terms the student has not encountered yet. The student might wonder what "aligned" means or what "human preferences" refers to.
**Student impact:** Minimal, since the grayed-out visual treatment signals "not now." The student might be mildly curious but is unlikely to be confused. The Next Step section at the end does reference RLHF, so the preview is consistent.
**Suggested fix:** No change needed. The visual dimming and "NEXT LESSON" label adequately signal that these terms will be explained later. This is a reasonable preview.

### Review Notes

**What works well:**
- The central insight ("SFT teaches format, not knowledge") is hammered home effectively through multiple angles. The "capital of France" dual-prompt example is the best single piece of evidence in the lesson -- it concretely proves the base model has knowledge that it expresses differently.
- The Prediction Checkpoint in the hook (guessing a new "instruction head" then being told that guess is wrong) is pedagogically excellent. It leverages the student's classification finetuning mental model, lets them predict, then corrects the prediction. This is exactly how misconceptions should be addressed.
- The ComparisonRow showing pretraining loop vs SFT loop (Section 6) is the proof that SFT uses the same objective. Five steps, four identical, one different. This is clear and unambiguous.
- The LossMaskingDiagram is well-designed. Showing the actual tokens with color-coded masking (purple for masked, green for loss-contributing) makes the concept visually immediate. The `label: -100` annotations connect directly to the PyTorch code below.
- Scope boundaries are explicit and well-communicated. The ConstraintBlock lists both what IS covered and what IS NOT, with forward references to specific future lessons.
- Checkpoint questions (Check 1: loss function prediction; Check 2: wrong template) are well-placed and test genuine understanding rather than recall.
- The lesson hits all 5 planned misconceptions: (1) "SFT adds knowledge" is addressed centrally; (2) "SFT needs a different loss" is addressed by the checkpoint and loop comparison; (3) "You need millions of examples" is addressed by the Alpaca/LIMA data; (4) "Templates are cosmetic" is addressed by Check 2; (5) "SFT and classification are fundamentally different" is addressed by the comparison in Section 11.

**Patterns to watch:**
- The lesson is long. 13 sections with substantial content in each. For an ADHD-friendly course, consider whether the notebook pointer (Section 9) could come earlier to give the student an action break before the two Elaborate sections. The student reads ~8 sections of concepts before getting to do anything. The notebook is the main hands-on element but arrives late.
- The repetition of the central insight is appropriate (it IS the core concept), but the third and fourth restatements (Sections 10, 11) need to add genuinely new perspective each time rather than re-argue the same point. The Classification vs SFT section (11) does add new perspective (structural comparison). The "Why So Little Data" section (10) is where the repetition feels flattest -- the suggested improvement above addresses this.

---

## Review -- 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All four improvement findings from iteration 1 were addressed correctly. The lesson is now pedagogically sound, with no critical or improvement-level issues remaining. Two minor polish items exist but do not require re-review.

### Iteration 1 Fix Verification

1. **Wrong-template negative example (IMPROVEMENT):** Fixed. Check 2 (lines 944-968) now includes a concrete side-by-side snippet showing correct template producing a coherent response vs wrong template producing garbled template-fragment repetition (`[INST] Explain gravity briefly. [/INST] [INST] What is the meaning of life?...`). The misconception is now addressed with concrete evidence, not just verbal assertion.

2. **Intuitive modality underdeveloped (IMPROVEMENT):** Fixed. Section 10 (lines 1050-1070) now uses the student's own three-training trajectory (pretraining -> classification -> instruction) to build the "of course" moment. The `&hellip;` pause before "of course the model learns to produce responses to instructions" creates space for the student to arrive at the insight independently. The section no longer feels like a recap.

3. **Narrative bridge from Lesson 1 buried (IMPROVEMENT):** Fixed. The ObjectiveBlock (lines 407-415) now opens with "In the last lesson, you adapted GPT-2 for one specific task: sentiment classification. That required a labeled dataset and a new classification head--one head per task. But ChatGPT does not have a separate head for every possible task. How does one model follow any instruction?" The motivating tension is front and center in the main content flow.

4. **SFT Pipeline Diagram placed too late (IMPROVEMENT):** Fixed. The diagram is now in Section 6 (SFT Training Mechanics, line 754), right after the pretraining-vs-SFT ComparisonRow. It reinforces "same architecture, different data" at the moment the student is absorbing the mechanical details, functioning as scaffolding rather than decoration.

5. **Spaced em dash in SVG (POLISH):** Fixed. Line 111 now reads `Same GPT architecture throughout—only the data changes` with no spaces around the em dash.

### Findings

#### [POLISH] -- Code block em dash remains spaced

**Location:** Instruction dataset CodeBlock, line 626
**Issue:** The string `"A list is mutable — you can add, remove, and change ..."` still uses a spaced em dash. This was flagged in iteration 1 as optional since it represents external JSON data, and the builder reasonably left it as-is.
**Student impact:** Negligible. Inside a code example representing training data. Spaced em dashes are common in real datasets.
**Suggested fix:** Optional. Defensible as-is.

#### [POLISH] -- Notebook section arrives after 8 conceptual sections

**Location:** Section 9, "Build It Yourself" (lines 980-1038)
**Issue:** Iteration 1's review notes observed that the student reads approximately 8 sections of concepts before reaching the notebook. For an ADHD-friendly course, the action break comes late. However, the notebook requires understanding of instruction datasets, chat templates, and loss masking to be meaningful -- moving it earlier would require the student to implement concepts they have not yet learned.
**Student impact:** A student with low energy might disengage before reaching the notebook. The lesson is dense but well-chunked, and each section is necessary scaffolding for the notebook exercises.
**Suggested fix:** No change to notebook placement (it depends on prior sections). If pacing is a concern in practice, consider adding a brief callout earlier (e.g., in Section 4 after showing the instruction dataset) that says "You will implement all of this in the notebook later" to signal that hands-on work is coming. This is a design-level decision, not a fix.

### Review Notes

**What improved since iteration 1:**
- The wrong-template negative example is now one of the lesson's strongest moments. The concrete garbled output makes the "templates are functional" point visceral rather than intellectual. The student sees what failure looks like, not just reads about it.
- The "Why Does SFT Work" section is no longer a repetitive recap. The three-training trajectory (pretraining, classification, instruction) reframes the central insight from the student's own experience, making it feel earned rather than asserted.
- The lesson bridge from classification finetuning to SFT is now prominent and motivating. The student immediately understands why this lesson exists and what question it answers.
- The pipeline diagram now serves as active scaffolding in Section 6, where the student is building the "same architecture, different data" mental model.

**Overall assessment:**
The lesson is ready to ship. It covers all planned concepts at the planned depths. All 5 misconceptions are addressed with concrete evidence. The narrative arc flows from motivating question through conceptual explanation to hands-on implementation to deepened understanding. The central insight ("SFT teaches format, not knowledge") is developed through 5 modalities (concrete example, verbal/analogy, visual, symbolic/code, intuitive) and reinforced without being repetitive. Scope boundaries are clear and consistently maintained. The two remaining polish items are genuinely minor and do not affect the student's learning experience.
