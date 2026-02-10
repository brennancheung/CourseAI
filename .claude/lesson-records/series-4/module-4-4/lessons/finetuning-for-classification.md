# Lesson: Finetuning for Classification

**Module:** 4.4 (Beyond Pretraining)
**Position:** Lesson 1 of 5
**Type:** Hands-on (notebook: `4-4-1-finetuning-for-classification.ipynb`)
**Cognitive load:** BUILD

---

## Phase 1: Orient — Student State

The student arrives from Module 4.3 having built, trained, and verified a complete GPT implementation. They loaded real GPT-2 weights and generated coherent text. They also have deep transfer learning experience from CNNs (Series 3). Here is their relevant concept state:

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Transfer learning (reusing pretrained weights for new tasks) | DEVELOPED | transfer-learning (3.2.3) | "Hire experienced, train specific." Two strategies: feature extraction and fine-tuning. |
| Feature extraction (freeze backbone, replace head) | DEVELOPED | transfer-learning (3.2.3) | Three-step pattern: load pretrained, freeze all, replace head. Applied on CIFAR-10 subset. |
| Fine-tuning with differential learning rates | DEVELOPED | transfer-learning-project (3.3.2) | Student wrote optimizer with parameter groups (layer4 at 1e-4, fc at 1e-3). Upgraded from INTRODUCED to DEVELOPED through practice. |
| Feature transferability spectrum (early universal, later task-specific) | DEVELOPED | transfer-learning (3.2.3) | conv1 = universal edges, layer4 = task-specific, fc = always replaced. |
| requires_grad=False for freezing parameters | DEVELOPED | transfer-learning (3.2.3) | Connected to autograd. Used in practice. |
| Cross-entropy loss for classification | INTRODUCED | transfer-learning (3.2.3) | nn.CrossEntropyLoss combines log-softmax + NLL. Brief gap resolution. |
| GPT architecture in PyTorch (full implementation) | APPLIED | building-nanogpt (4.3.1) | Student wrote all five classes. Shape verification. |
| GPT-2 weights loaded into student model | APPLIED | loading-real-weights (4.3.4) | Weight mapping, Conv1D transposition, logit verification. |
| Weight tying (embedding = output projection) | DEVELOPED | building-nanogpt (4.3.1) | Same tensor. data_ptr() confirmed. |
| HuggingFace transformers library | INTRODUCED | loading-real-weights (4.3.4) | from_pretrained("gpt2"). Minimal — used only as weight source. |
| Autoregressive generation (generate method) | DEVELOPED | building-nanogpt (4.3.1) | torch.no_grad(), crop, forward, sample last position, append. |
| Cross-entropy for next-token prediction (50K vocab) | DEVELOPED | pretraining (4.3.2) | Reshape (B, T, V) to (B*T, V). |
| "The architecture is the vessel; the weights are the knowledge" | — | loading-real-weights (4.3.4) | Same code, different weights, different behavior. |
| Complete GPT training loop | APPLIED | pretraining (4.3.2) | Forward, loss, backward, step, LR scheduling, gradient clipping. |

**Mental models already established:**
- "Hire experienced, train specific" (CNN transfer learning)
- Feature extraction vs fine-tuning as a spectrum
- "Start with the simplest strategy, add complexity only if needed"
- Transfer learning is the default, not a workaround
- "The architecture is the vessel; the weights are the knowledge"
- "Same heartbeat, new instruments" (training loop structure)

**What was explicitly NOT covered that is relevant here:**
- Finetuning a transformer/LLM (explicitly deferred from loading-real-weights)
- Classification with a transformer (only generative next-token prediction so far)
- How to extract a single representation from a sequence model (the "which hidden state?" question)
- HuggingFace beyond minimal weight loading

**Readiness assessment:** The student is highly prepared. They have deep transfer learning experience from CNNs (DEVELOPED feature extraction, DEVELOPED fine-tuning with differential LR), a working GPT-2 model with loaded weights, and strong intuition for "pretrained features + task head." The conceptual bridge is straightforward. This is BUILD territory.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to adapt a pretrained GPT-2 model for text classification by adding a classification head, choosing which hidden state to use as the sequence representation, and training with frozen or unfrozen parameters.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Transfer learning (freeze backbone, add head) | DEVELOPED | DEVELOPED | 3.2.3 | OK | Core pattern. Student practiced on CNNs. |
| GPT-2 architecture in PyTorch | DEVELOPED | APPLIED | 4.3.1 | OK | Student built it. Exceeds requirement. |
| Loading pretrained GPT-2 weights | DEVELOPED | APPLIED | 4.3.4 | OK | Student mapped weights manually. |
| Cross-entropy loss for classification | INTRODUCED | INTRODUCED | 3.2.3 | OK | Brief gap resolution covered it. Will use nn.CrossEntropyLoss directly. |
| requires_grad for parameter freezing | DEVELOPED | DEVELOPED | 3.2.3 | OK | Practiced in CNN context. |
| Training loop (forward, loss, backward, step) | DEVELOPED | APPLIED | 4.3.2 | OK | Exceeds requirement. |
| What GPT-2 hidden states represent | INTRODUCED | INTRODUCED | 4.2 (various) | OK | Student knows residual stream, attention output. Lesson will address "which position's hidden state?" as new content. |
| Tokenization / tiktoken | DEVELOPED | APPLIED | 4.1.2 | OK | Student built BPE from scratch. |

All prerequisites OK. No gaps to resolve.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example | Where to Address |
|---------------|----------------------|------------------|-----------------|
| "You need to retrain the entire model for classification" | From CNN experience, full fine-tuning was one option. But GPT-2 has 124M params -- full finetuning on a small classification dataset is wasteful and can overfit badly. | Show parameter count comparison: classification head has ~768 x num_classes params (~2.3K for 3 classes) vs 124M total. Training 124M params on 1000 examples = massive overfitting risk. | Core concept section, after introducing frozen-backbone approach |
| "Use the first token's hidden state as the sequence representation" | Intuitive -- first token seems like a "starting point." In BERT, [CLS] is the first token. But GPT-2 uses causal masking: the first token has seen NOTHING (only itself). The LAST token has seen everything. | Show attention mask diagram: first token attends only to itself (1 token of context), last token attends to all tokens (full context). "Which token has more information about the entire sequence?" | Hook/core concept, before introducing the classification head |
| "Finetuning an LLM is totally different from finetuning a CNN" | Different modality (text vs images), different architecture (transformer vs CNN), different task. Feels like a different world. | Side-by-side comparison: CNN (freeze ResNet, replace model.fc, train on labels) vs GPT-2 (freeze transformer, add linear head, train on labels). The pattern is identical. Only the feature extractor changed. | Hook -- explicit callback at the start |
| "The pretrained language model loses its language abilities after classification finetuning" | Catastrophic forgetting is a real phenomenon. But with frozen backbone, the model's representations do not change at all. With full finetuning on a narrow task, yes, it can lose generality. | Frozen backbone: generate text before and after finetuning -- identical output. The model's internal representations have not changed. Only the head has learned. | After introducing frozen vs unfrozen comparison |
| "You need a special tokenizer or input format for classification" | Classification feels different from generation. But the tokenizer and input format are identical -- feed text, get hidden states. The only change is what you do with the output. | Same tiktoken encoding, same model forward pass, same hidden states. The only difference is: generation takes the logits over vocab, classification takes the last hidden state to a new linear layer. | During the "how to get a sequence representation" section |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Sentiment classification (SST-2: positive/negative movie reviews) | Positive | Core example: binary classification is the simplest possible task. Student can see the full pipeline with minimal complexity. | Two classes, short texts, clear labels. No ambiguity about what "classification" means. Most intuitive starting point. |
| Same frozen model, different head for topic classification (e.g., 4 classes) | Positive | Shows that the same pretrained features support multiple tasks by just swapping the head. Reinforces "the backbone is a general feature extractor." | Demonstrates generality. A second example confirms the pattern works beyond sentiment. Student can swap nn.Linear(768, 2) to nn.Linear(768, 4) and retrain. |
| Using the first token's hidden state instead of the last token's (poor results) | Negative | Concretely demonstrates why the LAST token matters for causal models. The first token has minimal context due to causal masking. | Makes the "which hidden state?" decision concrete with measurable consequences. Student should predict this after understanding causal masking. |

---

## Phase 3: Design

### Narrative Arc

You have built GPT-2 from scratch and loaded real pretrained weights. It generates coherent text. But you cannot ask it "Is this movie review positive or negative?" It will just continue the text. The pretrained model is a powerful feature extractor that understands language -- but it was trained for one task only: predict the next token. To make it useful for classification, you need to adapt it. You already know how to do this. In Series 3, you took a ResNet pretrained on ImageNet and adapted it for flower classification by freezing the backbone and adding a new classification head. The same strategy works here. The pretrained transformer is your "experienced employee" -- it already understands language. You just need to teach it a new output format. The question is: which part of the transformer's output should you use? In a CNN, global average pooling gave you one vector per image. In a causal transformer, you have one hidden state per token. Which one represents the whole sequence?

### Modalities Planned

| Modality | What Specifically | Why This Modality |
|----------|------------------|-------------------|
| Visual | Inline SVG diagram showing GPT-2 as a backbone with the lm_head replaced by a classification head. Side-by-side with the CNN transfer learning diagram from 3.2.3 (ResNet backbone + new fc layer). | Makes the structural analogy between CNN and transformer transfer learning visually explicit. The student can SEE they are the same pattern. |
| Concrete example | Worked example with a specific movie review: tokenize "This movie was terrible" -> 5 tokens -> forward pass -> 5 hidden states of shape (5, 768) -> take last token -> linear(768, 2) -> logits [1.2, -0.8] -> cross-entropy loss with label 0 (negative). | Traces the full pipeline with real numbers. Makes abstract "add a classification head" concrete. |
| Symbolic/code | The classification head implementation: `nn.Linear(config.n_embd, num_classes)` applied to `hidden_states[:, -1, :]` (last token). The training loop adapted from pretraining with nn.CrossEntropyLoss. | The code is the primary modality for a BUILD lesson. Student needs to see the exact implementation. |
| Verbal/analogy | Extend "hire experienced, train specific" from CNN transfer learning. The pretrained GPT-2 is the experienced employee. The classification head is the job-specific training. Freezing the backbone = "don't retrain your general skills, just learn the new output format." | Explicitly bridges to the established mental model. Student should feel the callback. |
| Intuitive | Causal masking means the last token has "read" all previous tokens -- it is the only position with full sequence context. Of course you use that one. | The "of course" moment. Once stated, it feels obvious -- but only because the student deeply understands causal masking from Module 4.2. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2
  1. Extracting a sequence representation from a causal transformer (last token's hidden state)
  2. Adding a classification head to a pretrained language model
- **Previous lesson load:** CONSOLIDATE (loading-real-weights was the Module 4.3 capstone)
- **This lesson's load:** BUILD -- appropriate. Two new concepts, but both connect directly to established patterns (CNN transfer learning, causal masking). The student is applying a familiar strategy in a new domain.

### Connections to Prior Concepts

| Prior Concept | How It Connects | Source |
|--------------|----------------|--------|
| CNN transfer learning (freeze backbone, add head) | Identical pattern applied to a transformer. The lesson makes this explicit. | 3.2.3 |
| Causal masking (each position only sees previous tokens) | Directly motivates using the LAST token: it is the only position with full context. | 4.2.5, 4.2.6 |
| "The architecture is the vessel; the weights are the knowledge" | The pretrained weights are the knowledge. The classification head is a new vessel for a different output. | 4.3.4 |
| Feature transferability spectrum | For transformers: early layers capture syntax/grammar (universal), later layers capture semantics (more task-specific). Same principle, different features. | 3.2.3 |
| "Hire experienced, train specific" | Extended from CNNs to LLMs. Same analogy, same strategy. | 3.2.3 |
| Cross-entropy loss | Same loss function used for next-token prediction (50K classes) and classification (2-4 classes). Reinforces "cross-entropy doesn't care about vocabulary size." | 4.3.2, 3.2.3 |

**Potentially misleading prior analogies:**
- The CNN "global average pooling" approach does NOT directly transfer. In CNNs, GAP averages all spatial positions. In causal transformers, you cannot average all positions equally because early positions have limited context. The lesson should explicitly address why averaging does not work as well as taking the last token.

### Scope Boundaries

**This lesson IS about:**
- Adding a classification head to pretrained GPT-2
- Choosing which hidden state to use (last token, with explanation)
- Frozen backbone vs full finetuning comparison
- Training on a classification dataset (sentiment)
- The conceptual bridge between CNN and LLM transfer learning

**This lesson is NOT about:**
- Instruction tuning or SFT (Lesson 2)
- RLHF or alignment (Lesson 3)
- LoRA or parameter-efficient finetuning (Lesson 4)
- Token classification (NER, POS tagging -- each token gets a label)
- Regression tasks
- Multi-task finetuning
- Prompt-based classification (zero-shot or few-shot with the generative model)
- HuggingFace Trainer API (keep it manual for understanding)
- Padding / batching variable-length sequences in depth (keep to minimum needed for the notebook)

**Depth target:** Classification head pattern at DEVELOPED, last-token representation at DEVELOPED, frozen vs unfrozen comparison at INTRODUCED.

### Lesson Outline

1. **Context + Constraints** (Row)
   - What we are doing: adapting pretrained GPT-2 for text classification
   - What we are NOT doing: instruction tuning, LoRA, prompt-based methods (those come in later lessons)
   - The bridge: "You know how to do this with CNNs. Now you will do it with a transformer."

2. **Hook — The Bridge** (Row)
   - Side-by-side ComparisonRow: CNN transfer learning (3.2.3) vs what we are about to do
   - CNN: Load ResNet -> freeze backbone -> replace model.fc -> train on flowers
   - GPT-2: Load GPT-2 -> freeze transformer -> replace lm_head -> train on sentiment
   - "The pattern is identical. Only the feature extractor changed."
   - Prediction checkpoint: "What is the one thing that is genuinely different between these two?" (Expected: how to get a single representation from a sequence of hidden states vs a spatial feature map)

3. **Explain — Which Hidden State?** (Row + Aside)
   - The new concept: GPT-2 produces one hidden state per token position. A CNN produces a spatial feature map. In CNNs, GAP pools the spatial map into one vector. What do we do for a sequence?
   - Causal masking callback: the first token has seen only itself. The last token has seen everything. Inline SVG showing the causal attention pattern with each position's context highlighted.
   - "Of course you use the last token." This is the new concept of this lesson, and it follows directly from what the student already knows.
   - Negative example: prediction about using the first token (poor results because limited context)
   - Brief mention: BERT uses [CLS] at position 0 because BERT is bidirectional (every position sees everything). GPT is causal, so position matters. This contrast reinforces WHY last-token is the right choice for GPT.

4. **Check 1 — Predict and Verify** (Row)
   - "If you average ALL hidden states instead of taking the last one, would it work better or worse? Why?"
   - Expected: Worse, because early positions have limited context. The average dilutes the full-context representation of the last token with the limited-context representations of early tokens.

5. **Explain — The Classification Head** (Row + Aside)
   - Code: the classification model wraps the pretrained GPT
   - Remove lm_head (or ignore it), add nn.Linear(768, num_classes)
   - Forward pass: tokens -> transformer -> hidden_states -> last token -> classification head -> logits
   - Worked example with "This movie was terrible" traced through with shapes at each step
   - Weight tying observation: we no longer need lm_head. The embedding matrix stays (we still need to embed tokens), but the output projection changes purpose entirely.

6. **Explain — Freezing and Training** (Row)
   - Freeze the transformer backbone (requires_grad=False, callback to CNN)
   - Train only the classification head
   - Training loop: nearly identical to pretraining loop, but with nn.CrossEntropyLoss on labels instead of next-token targets, and no LR warmup needed for a small head
   - "Same heartbeat" callback

7. **Check 2 — Transfer Question** (Row)
   - "You trained a sentiment head (2 classes). Now you want topic classification (4 classes). What changes in the code? What stays the same?"
   - Expected: Only nn.Linear(768, 2) -> nn.Linear(768, 4) and the dataset. Everything else identical. The backbone features support both tasks.

8. **Explore — Notebook** (Row)
   - Notebook exercises:
     - Load pretrained GPT-2 (via HuggingFace, callback to loading-real-weights)
     - Tokenize SST-2 examples with tiktoken
     - Implement the classification head
     - Train with frozen backbone
     - Evaluate accuracy
     - Compare: unfreeze last N transformer blocks and retrain with differential LR
     - Generate text before and after finetuning (frozen: identical. Unfrozen: degraded or unchanged depending on amount of finetuning)

9. **Elaborate — Frozen vs Unfrozen** (Row + Aside)
   - ComparisonRow: frozen backbone (fast, small memory, safe from catastrophic forgetting) vs full finetuning (slower, more memory, risk of forgetting, potentially higher accuracy with enough data)
   - Connection to CNN decision framework: dataset size x domain similarity
   - For transformers: the "domain" consideration is different -- a language model pretrained on web text has broad domain coverage, so feature extraction (frozen) often works well even for specialized text
   - Partial unfreezing: unfreeze last few transformer blocks (like unfreezing layer4 in ResNet)
   - Catastrophic forgetting concretized: generate text after aggressive full finetuning -- the model forgets how to generate

10. **Summarize** (Row)
    - Mental model: "A pretrained transformer is a text feature extractor. Add a head, freeze the backbone, train the head. Same pattern as CNNs."
    - Key insight: use the LAST token because causal masking means it has the most context
    - The classification head is the simplest adaptation. Next lesson: what if the "task" is not classification but "follow any instruction"?

11. **Next Step** (Row)
    - Classification fine-tuning adapts a model for ONE specific narrow task. But what if you want a model that can follow ANY instruction? That requires a different kind of adaptation -- not a new head, but a new training dataset. Next: instruction tuning.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via records (transfer learning from 3.2.3, GPT from 4.3.1, causal masking from 4.2.5/4.2.6)
- [x] Depth match verified: all OK
- [x] No untaught concepts remain
- [x] No multi-concept jumps in exercises
- [x] No gaps to resolve

### Pedagogical Design
- [x] Narrative motivation stated as coherent paragraph (problem before solution)
- [x] At least 3 modalities: visual (SVG diagram), concrete (worked example), symbolic/code, verbal/analogy, intuitive
- [x] At least 2 positive examples (sentiment, topic classification) + 1 negative (first-token representation)
- [x] At least 3 misconceptions (5 identified) with negative examples
- [x] Cognitive load = 2 new concepts (within limit)
- [x] Every new concept connected to existing concept (classification head -> CNN transfer learning, last-token -> causal masking)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings -- the student would not be lost or form a wrong mental model. But four improvement-level findings would make the lesson significantly stronger if addressed.

### Findings

#### [IMPROVEMENT] — Weight tying observation dropped from outline

**Location:** Section 5 (The Classification Head), outline item 5
**Issue:** The planning document's outline for section 5 includes: "Weight tying observation: we no longer need lm_head. The embedding matrix stays (we still need to embed tokens), but the output projection changes purpose entirely." This is absent from the built lesson. The student has DEVELOPED understanding of weight tying from building-nanogpt (4.3.1) and loading-real-weights (4.3.4) -- they know lm_head.weight and wte.weight are the same tensor. Replacing lm_head with a classification head breaks that tie. This is a meaningful observation: the student should understand they are severing a connection they verified with data_ptr() just one module ago. Without it, they might wonder what happens to the shared weight.
**Student impact:** The student may be confused about what happens to the tied lm_head when they replace it. They verified weight tying with data_ptr() in loading-real-weights. Now they are silently discarding lm_head. A brief note connecting this would reinforce the mental model and prevent a dangling question.
**Suggested fix:** Add 1-2 sentences in the Classification Head section or its aside: "Notice that we are not using lm_head at all. In the pretrained model, lm_head shared its weights with the token embedding (weight tying). By replacing lm_head with a classification head, we break that tie -- the embedding stays (we still need to embed tokens), but the output projection now maps to class labels instead of vocabulary logits."

#### [IMPROVEMENT] — Misconception 5 ("special tokenizer needed") addressed only in aside, not in main flow

**Location:** "Same Input Pipeline" aside (after the worked example, around line 778-784) and the worked example itself
**Issue:** The planning document identifies misconception 5: "You need a special tokenizer or input format for classification." The plan says to address this "During the 'how to get a sequence representation' section" with a concrete negative example showing "Same tiktoken encoding, same model forward pass, same hidden states." In the built lesson, this is handled only as a brief InsightBlock aside ("You do not need a special tokenizer..."). The main content does mention "Notice what did not change: the tokenizer, the forward pass..." (line 766-773), which partially covers this. However, the planned negative example format is missing -- there is no concrete "what if you thought you needed a different tokenizer?" moment with a disproof.
**Student impact:** Minor -- the student probably absorbs this from the worked example and the brief mention. But the misconception is specifically about expectations, and a direct address ("You might think classification needs a different input format. It does not.") would be more effective than an aside the student might skim.
**Suggested fix:** Add one sentence to the main flow before or after the worked example that directly addresses the expectation: "You might expect classification to require a different tokenizer or special input format. It does not -- the exact same tokenization and forward pass from generation applies. The only change is what you do with the output."

#### [IMPROVEMENT] — Second positive example (topic classification) is embedded in a checkpoint rather than shown as an independent example

**Location:** Check 2 -- Transfer Question (Section 7, line 867-905)
**Issue:** The planning document lists the second positive example as: "Same frozen model, different head for topic classification (e.g., 4 classes). Shows that the same pretrained features support multiple tasks by just swapping the head." In the built lesson, this appears inside a checkpoint question ("You trained a sentiment head. Now you want topic classification..."). This is good as a checkpoint, but the student never sees the topic classification example worked through -- they only see the answer ("Changes: nn.Linear(768, 2) -> nn.Linear(768, 4) and the dataset"). The pedagogical purpose of a second positive example is to confirm the pattern generalizes beyond the first instance. A checkpoint with a hidden answer is weaker at this than a visible worked example.
**Student impact:** The student who opens the spoiler sees the generalization confirmed. The student who does not (or who opens it too quickly) may not fully internalize that the same backbone supports arbitrary classification tasks. The generalization message lands, but less firmly than a visible example would.
**Suggested fix:** Consider either (a) showing a brief visible code snippet for the 4-class case before the checkpoint (the checkpoint then becomes "predict what changes" with the student already having seen it), or (b) adding 1-2 sentences after the checkpoint's reveal that make the generalization point explicit and visible: "This is the same principle as CNN transfer learning: the backbone is a general feature extractor. Any classification task only requires changing the head dimensions and the dataset."

#### [IMPROVEMENT] — Misconception 1 ("retrain the entire model") not concretely addressed with a negative example

**Location:** Planned misconception 1 in the planning document; the built lesson's "Freezing and Training" section (lines 790-860)
**Issue:** The planning document identifies misconception 1: "You need to retrain the entire model for classification" with a planned negative example showing the parameter count comparison (classification head ~1.5K params vs 124M total, training 124M on 1000 examples = massive overfitting risk). In the built lesson, the TipBlock aside (line 712-718) mentions the parameter count comparison (1,536 vs 124M, 0.001% of total), and the Frozen vs Unfrozen section lists "Risk of catastrophic forgetting" and "Slow training" as downsides of full finetuning. However, the **overfitting** argument -- which is the strongest concrete disproof -- is never stated. The student sees "why frozen is good" but not "why full finetuning on a small classification dataset is actively dangerous."
**Student impact:** The student understands frozen is simpler and faster, but may still think "full finetuning would be better if I had the compute." The overfitting argument (124M params on ~67K SST-2 examples, or even fewer for many real tasks) would make the case more concrete and convincing.
**Suggested fix:** Add to the Frozen vs Unfrozen section or the ComparisonRow's right column: "Risk of overfitting: 124M trainable parameters on a classification dataset of thousands of examples means the model has far more capacity than the task requires. The head has 1,536 parameters -- much better matched to the task."

#### [POLISH] — Notebook link description could mention GPU requirement

**Location:** Notebook section (lines 927-950)
**Issue:** The notebook description mentions "tokenizing SST-2 examples" and "training with frozen backbone" but does not mention whether a GPU is needed. The student has used Colab before (from Series 2/3), but finetuning GPT-2 (even frozen) may take meaningfully longer on CPU. A brief note about runtime type would reduce friction.
**Student impact:** Minor -- the student may start the notebook on CPU and find training slow, then have to switch to GPU runtime and restart.
**Suggested fix:** Add a brief note: "Use a GPU runtime in Colab -- even frozen-backbone training benefits from GPU acceleration for the transformer forward pass."

#### [POLISH] — Architecture diagram missing the "lm_head (REMOVED)" annotation

**Location:** ClassificationHeadDiagram component (lines 188-358)
**Issue:** The architecture diagram shows the new classification pipeline clearly (frozen transformer -> last token selection -> new linear head). But it does not show what was removed -- the lm_head that previously projected to 50,257 vocabulary classes. Since the lesson's text explains "you replace lm_head," showing the removed component (perhaps grayed out or struck through) would make the visual tell the same story as the prose.
**Student impact:** Minor -- the student understands the replacement from text and code. But the diagram alone does not communicate that something was removed; it only shows what was added.
**Suggested fix:** Add a grayed-out or dashed box labeled "lm_head (removed)" next to or replacing the current "Class logits" box, or add a small annotation noting what was replaced.

#### [POLISH] — "Pretraining on Real Text" reference not formatted as a lesson name

**Location:** Freezing and Training section, line 828
**Issue:** The text says "Compare this to the pretraining loop you wrote in Pretraining on Real Text." This refers to lesson 4.3.2 (pretraining) but is not clearly formatted as a lesson reference. In other places, the lesson uses descriptive callback language ("In the transfer learning lesson..."). This reference is clear enough but slightly less polished.
**Student impact:** Negligible -- the student knows which lesson is meant.
**Suggested fix:** Either italicize or quote the lesson name, or rephrase to match the pattern used elsewhere: "Compare this to the pretraining loop you wrote in the pretraining lesson (Module 4.3)."

### Review Notes

**What works well:**
- The CNN transfer learning callback is excellent. The ComparisonRow showing side-by-side CNN vs GPT-2 transfer learning steps is the strongest moment in the lesson. A student arriving from Series 3 will immediately feel grounded.
- The causal masking diagram is well-designed. The color coding (green for last row), context count annotations, and bottom legend make the key insight visually obvious. The "of course you use the last token" moment lands well.
- The prediction checkpoints are well-placed and ask the right questions. The "average all positions?" checkpoint directly addresses a misconception that the student WOULD form based on their CNN experience.
- The worked example with concrete shapes at each step (lines 729-763) is thorough and traces the full pipeline. This is where BUILD lessons succeed -- the student can follow every step.
- The narrative arc is clear: bridge from familiar (CNN transfer learning) -> one new concept (which hidden state?) -> implementation -> practice -> elaboration. The lesson does not wander.
- Scope boundaries are explicit and well-placed. The student knows exactly what this lesson covers and what comes next.
- The BERT mention (lines 595-603) is appropriately brief -- just enough to prevent the student from confusing bidirectional and causal conventions, without derailing into BERT architecture.

**Patterns to watch:**
- The lesson leans heavily on the "this is the same as CNNs" callback, which is correct and effective. But ensure that the GENUINELY new concept (last-token selection) gets enough independent development that the student does not reduce it to "just like CNNs but with the last token." The current treatment is adequate but is the thinnest part of the lesson -- the causal masking diagram and two paragraphs are the only independent development before moving to implementation.
- All five planned misconceptions are addressed, but misconceptions 1 and 5 could be stronger (see findings above). Misconceptions 2 (first token), 3 (different from CNNs), and 4 (catastrophic forgetting) are well-handled.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All four improvement-level findings from iteration 1 have been addressed effectively. The weight tying observation is now in the main flow (lines 752-765), misconception 5 is addressed directly in prose (lines 824-828), the second positive example has visible reinforcement after the checkpoint (lines 971-978), and the overfitting argument is concrete and convincing (lines 1086-1096). The three polish items from iteration 1 (GPU note, lm_head diagram annotation, lesson name reference) are also fixed. No new critical or improvement findings emerged. Two minor polish observations remain.

### Findings

#### [POLISH] — CausalAttentionDiagram uses `<end>` token not present in the worked example

**Location:** CausalAttentionDiagram (lines 55-180) vs worked example (lines 787-820)
**Issue:** The causal attention diagram uses 5 tokens: `['This', 'movie', 'was', 'terrible', '<end>']`. The worked example later traces "This movie was terrible" as 4 tokens `[1212, 3807, 373, 7818]` with no end token. GPT-2 does not use an explicit `<end>` token in classification. The diagram is conceptual and the `<end>` token is useful for illustrating that the "last position has seen everything," but a careful student comparing the two might briefly wonder whether they need to add an end token to their classification pipeline.
**Student impact:** Minimal. The student understands the concept from the diagram and the worked example independently. The `<end>` token in the diagram reads as a generic "final position" marker. But a very literal student might ask "do I need to append an end token?"
**Suggested fix:** Either (a) remove `<end>` from the diagram and use 4 tokens matching the worked example, or (b) add a brief note near the diagram: "The diagram uses a generic `<end>` marker to highlight the last position. In practice, the last token of your input text serves this role -- no special token is needed."

#### [POLISH] — Post-checkpoint generalization paragraph slightly repeats the spoiler content

**Location:** Lines 971-978 (paragraph after Check 2 -- Transfer Question)
**Issue:** The paragraph "The only change: nn.Linear(768, 2) becomes nn.Linear(768, 4). The backbone is a general text feature extractor..." closely mirrors the checkpoint's spoiler content. For a student who opened the spoiler, this reads as a near-verbatim repetition. This was intentionally added in iteration 1 to ensure the second positive example's generalization point is visible in the main flow (not hidden behind a spoiler), which is the right call. The repetition serves a pedagogical purpose -- students who skipped the spoiler need it.
**Student impact:** Negligible. A student who opened the spoiler might briefly feel "I just read this," but the reinforcement is not harmful. The trade-off (visible generalization point vs minor repetition) is the correct one.
**Suggested fix:** No action needed. The current approach is the right trade-off. If desired, the visible paragraph could be rephrased slightly to feel like a synthesis rather than a repeat (e.g., leading with the takeaway rather than the mechanics), but this is purely cosmetic.

### Review Notes

**Iteration 1 fixes verified:**

| Finding | Status | Assessment |
|---------|--------|------------|
| Weight tying observation (IMPROVEMENT) | Fixed | Lines 752-765 integrate it naturally into the Classification Head section. The prose connects to the student's data_ptr() experience from loading-real-weights. Well done. |
| Misconception 5 in main flow (IMPROVEMENT) | Fixed | Lines 824-828 add "You might expect classification to require a different tokenizer or special input format. It does not..." directly in the main content, not just the aside. Clear and direct. |
| Second positive example visibility (IMPROVEMENT) | Fixed | Lines 971-978 add a visible paragraph after the checkpoint restating the generalization point. Students who skip the spoiler still see the key insight. Minor repetition for spoiler-openers is an acceptable trade-off. |
| Overfitting argument for misconception 1 (IMPROVEMENT) | Fixed | Lines 1086-1096 add "massive capacity mismatch" language and explicitly frame 124M params on a small dataset as memorization risk. This is the strongest single sentence in the Frozen vs Unfrozen section. |
| GPU note in notebook (POLISH) | Fixed | Lines 1022-1024. |
| lm_head diagram annotation (POLISH) | Fixed | Lines 346-386 add a grayed-out dashed box with strikethrough "lm_head (removed)" and a dashed connector line. The diagram now tells the same story as the prose. |
| Lesson name reference (POLISH) | Fixed | Line ~893 now uses "the pretraining lesson" matching the callback pattern used elsewhere. |

**What works well (reinforced from iteration 1):**
- The CNN transfer learning bridge remains the lesson's strongest moment. The ComparisonRow is immediate, concrete, and makes the student feel competent before any new material appears.
- The causal masking diagram with per-row context counts is pedagogically effective. The "of course" moment is well-earned.
- All 5 planned misconceptions are now addressed at appropriate strength levels. The overfitting argument (iteration 1 fix) significantly improved misconception 1's treatment.
- The lesson maintains BUILD-level cognitive load throughout. Two genuinely new concepts, both connected to established patterns. No section overwhelms.
- The narrative arc (familiar -> one new idea -> implement -> practice -> elaborate) is clean and well-paced.

**Overall assessment:**
This lesson is ready for the student. The two remaining polish items are genuinely minor and do not affect the learning experience. The lesson achieves its goal: the student understands how to adapt a pretrained transformer for classification, knows WHY the last token is used (from causal masking), and has a clear path to hands-on practice in the notebook. The heavy reliance on CNN transfer learning callbacks is the right strategy for a BUILD lesson -- it keeps cognitive load low while extending established patterns to a new domain.
