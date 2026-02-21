# Lesson 1: SigLIP 2 (siglip-2) -- Planning Document

**Module:** 8.1 Vision & Vision-Language Models
**Position:** Lesson 1 of 2+
**Type:** BUILD
**Slug:** siglip-2

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Contrastive learning paradigm (push matching pairs together, non-matching apart, using batch structure for negatives) | DEVELOPED | clip (6.3.3) | Taught via conference analogy (name tag matching), 4x4 similarity matrix heatmap with walkthrough, symmetric cross-entropy loss formula. Student can explain the training setup, loss function, why negatives come from the batch, and why scale matters. |
| CLIP dual-encoder architecture (separate image encoder + text encoder, no shared weights, loss-only coupling) | INTRODUCED | clip (6.3.3) | Two GradientCards showing encoder paths. Student knows the structure and data flow but has not built it. |
| Shared embedding space (text and image vectors in the same geometric space, enabling cross-modal cosine similarity) | DEVELOPED | clip (6.3.3) | Before/after SVG diagram showing alignment. Connected to VAE latent space and token embeddings. Student deeply understands why the shared space matters. |
| Contrastive loss formula (symmetric cross-entropy on cosine similarity matrix, labels = diagonal) | INTRODUCED | clip (6.3.3) | Full formula in LaTeX + PyTorch pseudocode. Student knows the formula structure and can read the code, but has not worked through the mathematical properties deeply. |
| Batch size as task difficulty in contrastive learning (more negatives = harder task = more discriminative features) | INTRODUCED | clip (6.3.3) | Three GradientCards (batch=2 vs 32 vs 32,768). CLIP's actual batch size of 32,768 as concrete reference. Student knows batch size matters but has not analyzed WHY larger batches are computationally required by the loss function. |
| Zero-shot classification via shared embedding space | INTRODUCED | clip (6.3.3) | 4-step procedure + pseudocode. Student understands how and why it works. |
| CLIP limitations (typographic attacks, spatial relationships, counting, novel compositions) | INTRODUCED | clip (6.3.3) | Student knows CLIP is "statistical co-occurrence, not deep understanding." |
| Vision Transformer / ViT (patchify, process patches as tokens with standard transformer) | DEVELOPED | diffusion-transformers (7.4.2) | Full tensor shape trace of patchify. "Tokenize the image" analogy. Student has deep understanding of ViT as an architecture from the DiT lesson, even though it was taught in the diffusion context. |
| Softmax for probability distributions | DEVELOPED | Multiple (1.x, 4.2) | Extensive experience: attention weights, temperature-controlled generation, classification outputs. Student knows softmax deeply. |
| Cross-entropy loss for classification (nn.CrossEntropyLoss) | INTRODUCED | transfer-learning (3.2.3) | Brief gap resolution: combines log-softmax + negative log-likelihood. Standard loss for classification. |
| Sigmoid / logistic function | DEVELOPED | activation-functions (1.2) | Taught as activation function with formula, derivative (max 0.25), saturation behavior. Student knows the shape and properties. |
| Transfer learning (reusing pretrained model weights for new tasks) | DEVELOPED | transfer-learning (3.2.3) | Feature extraction and fine-tuning strategies. "Hire experienced, train specific" analogy. |
| Temperature parameter in softmax (scaling logits before softmax to control distribution sharpness) | DEVELOPED | what-is-a-language-model (4.1.1), queries-and-keys (4.2.2) | Temperature slider experience. Student knows that dividing logits by temperature controls sharpness: low T = peaked, high T = uniform. Also understands CLIP's inverted temperature convention. |
| Self-distillation / knowledge distillation | NOT TAUGHT | N/A | The student has not encountered distillation as a formal technique. Mentioned briefly in some contexts but never taught. Will need brief inline explanation. |

**Mental models and analogies already established:**
- "Two encoders, one shared space--the loss function creates the alignment, not the architecture" (CLIP core mental model)
- "Negatives come from the batch" (batch structure IS the supervision signal)
- "The shared space enables zero-shot transfer" (matching any text to any image)
- "Name tag matching at a conference" (contrastive learning analogy)
- "Tokenize the image" (ViT patchify analogy from DiT lesson)
- Temperature as a "sharpness knob" for softmax distributions

**What was explicitly NOT covered in prior lessons (relevant here):**
- Sigmoid loss for contrastive learning (the core topic)
- Why CLIP's softmax cross-entropy loss creates batch-size dependency
- Per-pair binary classification as an alternative to row/column-wise softmax
- SigLIP or any CLIP variant (explicitly listed as "not covered" in the CLIP lesson)
- Self-distillation as a training technique
- Multi-resolution training strategies
- How vision encoders plug into VLMs (PaliGemma, etc.)

**Readiness assessment:** The student is well-prepared. Contrastive learning at DEVELOPED depth is the critical prerequisite, and the CLIP lesson provided exactly the right foundation. The student already understands the conference analogy, the similarity matrix, symmetric cross-entropy, and the role of batch size. The key gap is understanding WHY softmax cross-entropy creates batch-size dependency at a mechanistic level--the student knows batch size matters but not the precise reason. This is the "problem" that motivates SigLIP's solution. The BUILD designation is appropriate: sigmoid loss is a targeted replacement within a framework the student already understands deeply, not a paradigm shift.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain why sigmoid loss removes CLIP's batch-size dependency by replacing global softmax normalization with independent binary classification per image-text pair, and how SigLIP 2 builds on this with multi-stage training improvements.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Contrastive learning paradigm | DEVELOPED | DEVELOPED | clip (6.3.3) | OK | SigLIP is a contrastive method. The student must understand matching pairs, negatives from batch, and the shared embedding space to appreciate what changes. |
| Contrastive loss formula (symmetric cross-entropy on similarity matrix) | DEVELOPED | INTRODUCED | clip (6.3.3) | GAP | The student needs to understand the softmax cross-entropy loss deeply enough to see WHY it creates batch-size dependency. INTRODUCED depth means they know the formula but have not analyzed its properties. |
| Batch size as task difficulty in contrastive learning | INTRODUCED | INTRODUCED | clip (6.3.3) | OK | INTRODUCED is sufficient. The student knows batch size matters; this lesson explains the mechanistic reason. |
| Shared embedding space | DEVELOPED | DEVELOPED | clip (6.3.3) | OK | SigLIP uses the same shared embedding space concept. No change needed. |
| CLIP dual-encoder architecture | INTRODUCED | INTRODUCED | clip (6.3.3) | OK | INTRODUCED is sufficient. SigLIP uses the same dual-encoder architecture. The student needs to recognize it, not build it. |
| Softmax properties (normalization across a set, competition between entries) | DEVELOPED | DEVELOPED | Multiple | OK | Understanding that softmax creates competition between entries is critical for understanding why it causes batch-size dependency. |
| Sigmoid function (independent per-element, maps to [0,1]) | DEVELOPED | DEVELOPED | activation-functions (1.2) | OK | The sigmoid function is the core alternative to softmax in SigLIP. Student knows its shape and properties well. |
| ViT / patchify | INTRODUCED | DEVELOPED | diffusion-transformers (7.4.2) | OK | SigLIP 2 uses a ViT image encoder. Student has deep understanding from DiT lesson. |
| Temperature parameter | INTRODUCED | DEVELOPED | Multiple | OK | SigLIP uses a learned temperature parameter. Student understands temperature deeply. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Contrastive loss formula at DEVELOPED (need to understand the softmax normalization's mechanistic effect on batch-size dependency) | Small (has INTRODUCED, needs DEVELOPED for this specific property) | Dedicated recap section that walks through the softmax cross-entropy formula on a small similarity matrix, highlighting the normalization denominator. Show concretely: when you compute softmax over a row, every score in that row affects every other score. The denominator sums ALL exponentials. With batch size 4, the denominator has 4 terms. With batch size 32,768, it has 32,768 terms. The quality of the probability distribution (and therefore the gradient signal) depends on how many negatives are in the denominator. This makes the loss function batch-size dependent. 2-3 paragraphs + a small worked example with the formula. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "SigLIP is a completely different model from CLIP (different architecture, different training, different purpose)" | The name "SigLIP" sounds like a separate system. The student might assume it is as different from CLIP as, say, a diffusion model is from a classifier. | SigLIP uses the SAME dual-encoder architecture (image encoder + text encoder), the SAME shared embedding space, the SAME cosine similarity computation, and produces the SAME kind of output (aligned image/text embeddings for zero-shot tasks). The ONLY fundamental change is the loss function: sigmoid instead of softmax. Everything upstream and downstream is identical. Show the training pseudocode side-by-side: CLIP vs SigLIP differ in exactly one line (the loss computation). | Context section and recap, before introducing sigmoid loss. Set expectations: "one targeted change to the loss function." |
| "Sigmoid loss is less effective than softmax because it does not use the full batch as negatives" | The student learned that more negatives = harder task = better features (CLIP lesson). Sigmoid loss scores each pair independently, which sounds like it "wastes" the negative information. If each pair is scored independently, how can the model learn to distinguish similar-but-different items? | Sigmoid loss DOES use the full batch as negatives. For a batch of N image-text pairs, there are N matching pairs (diagonal) and N^2 - N non-matching pairs (off-diagonal). Each non-matching pair is scored as a negative (label = 0). The difference is HOW: softmax normalizes across the row (competition), sigmoid scores each cell independently (binary classification). The negative signal is still present--it comes from the off-diagonal cells being pushed toward 0. The total number of training signals per batch is actually higher: N^2 binary classifications vs 2N softmax classifications (N rows + N columns). | After explaining sigmoid loss mechanism. Address head-on because the "more negatives = better" lesson from CLIP could lead here. |
| "The batch-size dependency in CLIP is just about having enough negatives (a data quantity issue)" | The CLIP lesson taught batch size as task difficulty ("more negatives = harder discrimination"). The student might think the issue is purely about having enough negative examples to learn from, like needing more training data. | The batch-size dependency is a MATHEMATICAL property of the softmax normalization, not just a data quantity issue. Even if you could somehow provide the same number of negative examples, softmax would still behave differently at different batch sizes because the softmax denominator sums over ALL items in the batch. With small batches, the denominator is a sum of few terms--each term has a large relative influence. With large batches, each term is a tiny fraction of the sum. This changes the gradient dynamics: at small batch sizes, the gradients are noisy and the learning signal is unreliable. The fix is not "just use more data"--it is changing the loss function so that each pair's score does not depend on what else is in the batch. | In the recap/motivation section, when explaining WHY softmax creates the dependency. |
| "SigLIP 2 is mainly about the sigmoid loss (the '2' just means a minor update)" | The student might focus on the sigmoid loss (the name suggests it) and dismiss the "2" improvements as incremental tweaks. | SigLIP 2's improvements beyond the loss function are substantial and independently important: multi-stage training with self-distillation, multi-resolution processing, multilingual data, and decoder-based pretraining tasks. The sigmoid loss was the original SigLIP contribution (2023). SigLIP 2 (2025) keeps the same loss but adds training methodology innovations that significantly improve performance. The "2" represents a different kind of advance: not a new loss function, but a better recipe for training the same architecture. | Elaborate section, after covering sigmoid loss. Frame the "2" improvements as a separate set of innovations. |
| "Self-distillation means training a smaller model from a larger one (like knowledge distillation)" | The student has not formally encountered distillation. If they have heard the term informally, they likely picture a large teacher training a small student--the standard knowledge distillation setup. | In self-distillation, the teacher and student are the SAME model (or the same architecture). Specifically in SigLIP 2, the model trained in an earlier stage serves as the teacher for a later stage. There is no separate larger model. The model teaches itself by using its own predictions from a previous training checkpoint as soft targets for the next training stage. This is more like "refining your own understanding by revisiting material" than "learning from a more knowledgeable teacher." | When introducing multi-stage training. Brief inline explanation since distillation is NOT TAUGHT. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| CLIP vs SigLIP loss on a 4x4 similarity matrix (same matrix, different loss computation) | Positive | Show the concrete difference between softmax cross-entropy (row-wise normalization, column-wise normalization, average) and sigmoid loss (independent binary classification per cell). Same similarity scores, different loss values, different gradient behavior. Walk through both computations on the same 4x4 matrix. | Directly extends the 4x4 similarity matrix from the CLIP lesson. The student already traced through this exact matrix structure. Showing both losses on the same matrix makes the difference tangible and isolates the loss function as the only variable. |
| Batch size experiment: CLIP loss at batch=4 vs batch=64 vs batch=32768 (showing how the softmax denominator changes) | Positive | Demonstrate concretely how the softmax denominator grows with batch size and why this changes the learning dynamics. At batch=4, the denominator has 4 terms. At batch=32768, it has 32768 terms. The gradient of any single negative is diluted by factor of ~8000x. SigLIP: the gradient for each pair is the same regardless of batch size. | Makes the batch-size dependency visceral with concrete numbers. Connects to the three GradientCards from the CLIP lesson (batch=2 vs 32 vs 32768) but now explains the mechanistic reason, not just the task-difficulty framing. |
| Sigmoid loss on a single off-diagonal cell (scoring one non-matching pair) | Positive (stretch) | Trace the sigmoid loss for a single non-matching pair: cosine similarity = 0.3, label = 0, sigmoid(0.3 * temperature) = some probability, binary cross-entropy loss = -log(1 - p). Show that this computation depends ONLY on the similarity between these two specific items, not on any other pair in the batch. Then show the equivalent softmax computation for the same cell: the probability depends on ALL other similarities in the row. | Isolates the independence property of sigmoid loss at the single-cell level. The student can see that sigmoid(s_ij) depends only on s_ij, while softmax(s_ij) depends on s_i1, s_i2, ..., s_iN. This is the mechanistic core of the lesson. |
| SigLIP with batch=256 matching CLIP with batch=32768 (practical payoff) | Negative (boundary) | Show that SigLIP does NOT need a smaller batch size to be useful--it CAN use large batches too. The point is that it does not REQUIRE them. CLIP at batch=256 performs poorly because the softmax normalization is unreliable with only 256 negatives per row. SigLIP at batch=256 works fine because each pair is scored independently. But SigLIP at batch=32768 is even better--more negative pairs means more training signal. The improvement from larger batches is gradual and graceful, not a hard threshold. | Prevents the misconception that "SigLIP = small batch training." The advantage is flexibility and graceful scaling, not avoidance of large batches. Also addresses the "sigmoid does not use negatives" misconception by showing that more negatives still help, just without the hard dependency. |

---

## Phase 3: Design

### Narrative Arc

The student learned CLIP eight modules ago: two encoders, one shared space, contrastive learning with a conference analogy and a 4x4 similarity matrix. CLIP works beautifully, and the student saw it power text-conditioned image generation throughout Series 6 and 7. But there is a detail from the CLIP lesson that was stated but not deeply examined: CLIP was trained with a batch size of 32,768. That number was presented as "more negatives = harder task = better features," and the student accepted it. But why 32,768 specifically? What happens if you train CLIP with batch size 256? The answer is: it breaks. Not gradually degrades--it fundamentally fails to learn good representations. This is not a "more data is better" situation; it is a structural property of the loss function. The softmax cross-entropy loss normalizes across every item in the batch, making each pair's gradient depend on every other pair. With small batches, the normalization is unreliable and the gradients are noisy. CLIP does not just prefer large batches--it requires them. This is expensive, limiting, and architecturally inelegant. SigLIP's fix is beautifully simple: replace softmax with sigmoid. Instead of asking "which of these N items is the correct match?" (a competition across the batch), ask "is this specific pair a match?" (a binary yes/no per cell). Each cell in the similarity matrix becomes an independent binary classification. The sigmoid function maps the similarity score to a probability. Binary cross-entropy provides the loss. No normalization across the batch. No competition between entries. Each pair's gradient depends only on that pair's similarity, not on what else happened to be in the batch. The result: batch-size independence. The model learns effectively at batch=256 or batch=32768.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Verbal/Analogy** | "Multiple choice vs true/false"--CLIP's softmax loss is a multiple-choice exam (pick the correct match from N options). SigLIP's sigmoid loss is a true/false exam (for each pair, answer: match or not?). In multiple choice, the difficulty depends on how many options there are (batch size). In true/false, each question stands alone. | Maps to a universally familiar experience. The "multiple choice depends on number of options" insight is immediately intuitive. The student can feel why true/false is batch-size independent without any math. |
| **Visual (inline SVG)** | Side-by-side 4x4 similarity matrix comparison. Left: CLIP softmax loss with arrows showing row-wise and column-wise normalization (every cell in a row connected to the softmax denominator). Right: SigLIP sigmoid loss with each cell independently colored (matching=green, non-matching=red) and sigmoid applied per cell. No arrows connecting cells--each is self-contained. | The visual makes the independence property immediately visible. Connected cells (softmax) vs independent cells (sigmoid) is the core architectural difference, and a diagram communicates it faster than any formula. Extends the 4x4 heatmap from the CLIP lesson. |
| **Symbolic/Code** | PyTorch pseudocode for both losses side-by-side. CLIP: `logits = similarity * temperature; loss = (F.cross_entropy(logits, labels, dim=1) + F.cross_entropy(logits, labels, dim=0)) / 2`. SigLIP: `logits = similarity * temperature; labels = 2 * eye(N) - 1; loss = -F.logsigmoid(labels * logits).mean()`. Annotated: the CLIP loss uses cross_entropy (row/column normalization), the SigLIP loss uses logsigmoid (per-element). | Code is the student's strongest modality. Side-by-side code makes the difference precise and shows it is a single-line change. The `labels * logits` trick (positive for matches, negative for non-matches) is elegant and worth showing. |
| **Concrete example** | Trace sigmoid loss on one matching and one non-matching pair from the 4x4 matrix. Matching: similarity = 0.9, label = +1, loss = -log(sigmoid(0.9 * t)). Non-matching: similarity = 0.2, label = -1, loss = -log(sigmoid(-0.2 * t)). Show that neither computation references any other cell. | Grounds the formula in specific numbers. The student can verify independence by checking: "does this number depend on anything outside this cell?" The answer is no. |
| **Geometric/Spatial** | Batch-size scaling diagram: show how the softmax denominator grows with batch size (small bucket with 4 items vs large bucket with 32768 items, each item's "share" of the probability mass shrinking). Contrast with sigmoid: N independent gauges, each reading stays the same regardless of how many other gauges exist. | Makes the scaling behavior spatial and intuitive. The "bucket" metaphor for softmax normalization (fixed total probability mass, divided among more and more items) vs independent gauges (each reads its own value) communicates the core mathematical difference. |
| **Intuitive** | The "of course" beat: "CLIP's batch size of 32,768 was not a hyperparameter choice--it was a structural requirement of the loss function. The softmax denominator sums over every item in the batch, making the gradient signal depend on batch composition. Of course a loss function where each pair is scored independently would remove this dependency. Of course that loss function is the sigmoid, which you already know maps a single real number to [0,1] without referencing anything else." | Two "of course" moments chained: (1) the dependency is structural (not a tuning issue), (2) sigmoid is the natural fix (the student already knows it is element-wise). Both premises are things the student already has at DEVELOPED depth. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2
  1. Sigmoid loss for contrastive learning (replacing softmax cross-entropy with per-pair binary classification via sigmoid + BCE--the core innovation)
  2. SigLIP 2 training improvements (multi-stage training, self-distillation, multi-resolution--a cluster of related training methodology innovations)
- **Previous lesson load:** N/A (this is the first lesson in a new module, coming after Series 7 completion). The last lesson (z-image) was BUILD.
- **This lesson's load:** BUILD--appropriate. The student already has contrastive learning at DEVELOPED depth, knows the CLIP framework well, and knows the sigmoid function deeply. Sigmoid loss is a targeted replacement within a known framework. The SigLIP 2 improvements are conceptual (multi-stage training, self-distillation) rather than technically demanding. No new architecture to learn--same dual-encoder.
- **Self-contained lesson note:** As a Special Topics lesson, this must work even if the student's CLIP knowledge has faded somewhat. The recap section is heavier than in structured series lessons.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| Contrastive learning / CLIP (6.3.3) | Direct extension. SigLIP uses the same contrastive framework with a different loss function. The conference analogy, similarity matrix, and shared embedding space all carry over unchanged. |
| Softmax normalization and competition (Series 1, 4.2) | The student has deep experience with softmax creating competition between entries (attention weights, classification logits). This lesson reveals that this competition property is EXACTLY what causes batch-size dependency in CLIP. |
| Sigmoid function (1.2) | The sigmoid function, taught as an activation function in Series 1, becomes the per-element scoring function in SigLIP. Same function, new context. |
| Temperature parameter (4.1, 4.2, 6.3.3) | SigLIP uses a learned temperature (like CLIP). The student's temperature intuition carries over directly. |
| Transfer learning / pretrained encoders (3.2.3) | SigLIP 2 is used as the vision encoder in PaliGemma and other VLMs. Same "hire experienced, train specific" pattern from transfer learning. |
| ViT / patchify (7.4.2) | SigLIP 2 uses a ViT image encoder. The student's "tokenize the image" mental model from DiT applies directly. |
| Batch size in contrastive learning (6.3.3) | The CLIP lesson's "more negatives = better" is refined here: more negatives help, but the relationship should be graceful (SigLIP) not a hard requirement (CLIP). |
| CLIP's role in Stable Diffusion (6.3.3-6.3.4) | The student saw CLIP as the text encoder in SD. SigLIP shows that the vision-language encoder itself can be improved, with downstream benefits for models like PaliGemma. |

**Analogies from prior lessons that can be extended:**
- "Name tag matching at a conference" -> extend to: CLIP = multiple-choice matching (pick the right name from all attendees), SigLIP = yes/no matching (for each person-nametag pair, is this a match?)
- "Two encoders, one shared space" -> unchanged, carries over directly
- "Temperature as sharpness knob" -> carries over, SigLIP learns it end-to-end

**Analogies from prior lessons that could be misleading:**
- "More negatives = better" from CLIP lesson could suggest sigmoid loss is weaker because it "does not use negatives properly." Need to address: SigLIP still uses all negatives, just scores them independently.
- "Batch size as task difficulty" could suggest the issue is about task difficulty rather than mathematical normalization. Need to reframe: the issue is not difficulty but the softmax denominator creating coupling between pairs.

### Scope Boundaries

**This lesson IS about:**
- Why CLIP's softmax cross-entropy loss creates batch-size dependency (the problem)
- How sigmoid loss removes this dependency by scoring each pair independently (the solution)
- The mathematical and intuitive difference between softmax (global normalization) and sigmoid (per-element scoring)
- SigLIP 2's training improvements: multi-stage training, self-distillation, multi-resolution
- How SigLIP connects to downstream tasks: zero-shot classification, VLM vision encoders (PaliGemma)

**This lesson is NOT about:**
- Implementing SigLIP from scratch (conceptual lesson with optional notebook exploration)
- Training a contrastive model from scratch
- Detailed benchmark comparisons or specific accuracy numbers beyond illustrative examples
- The full PaliGemma or Gemini architecture (SigLIP as a component is covered, not the full VLM)
- Other contrastive learning variants (MoCo, SimCLR, BYOL--mentioned at most for context)
- Image generation or diffusion (SigLIP is for understanding/retrieval, not generation)
- Detailed multilingual training methodology
- Mathematical proof of batch-size independence (intuitive argument + concrete examples, not formal proof)

**Target depths:**
- Sigmoid loss for contrastive learning (per-pair binary classification removing batch-size dependency): DEVELOPED (can explain the mechanism, trace through an example, articulate why it removes the dependency)
- Batch-size dependency in softmax contrastive loss (why CLIP needs large batches): DEVELOPED (can explain the mathematical mechanism via the softmax denominator, not just "more negatives = better")
- SigLIP 2 training improvements (multi-stage, self-distillation, multi-resolution): INTRODUCED (knows the concepts and why they help, but not implementation details)
- Self-distillation: INTRODUCED (knows the concept: model teaches itself from a previous checkpoint, distinct from standard knowledge distillation)
- SigLIP as downstream building block (PaliGemma vision encoder): INTRODUCED (knows SigLIP encoders are used in VLMs, understands why contrastive pretraining produces useful vision encoders)

### Lesson Outline

**1. Context + Constraints**
What this lesson is about: how SigLIP replaces CLIP's softmax loss with sigmoid loss, removing the batch-size dependency that makes CLIP expensive to train, and how SigLIP 2 adds training methodology improvements. What we are NOT doing: implementing SigLIP, training from scratch, or covering the full PaliGemma architecture. This is a single, targeted change to a framework the student already knows--the goal is to understand why it matters and how it works.

**Self-contained note:** This is a Special Topics lesson. Prerequisites are recapped inline. The student may not have touched CLIP content in months.

**2. Recap (heavier than structured series)**
Reconnect to the CLIP lesson from Series 6, covering:

1. **The setup:** Two encoders (image + text), one shared embedding space, cosine similarity between embeddings. "Two encoders, one shared space--the loss function creates the alignment." Quick reminder of the conference analogy.

2. **The loss function:** Symmetric cross-entropy on the similarity matrix. Walk through the formula on a 4x4 matrix:
   - Compute cosine similarities for all 16 pairs
   - Apply softmax row-wise: each row becomes a probability distribution over text items (image i asking "which text is my match?")
   - Apply softmax column-wise: each column becomes a probability distribution over image items (text j asking "which image is my match?")
   - Cross-entropy loss on both, averaged
   - **Key observation (gap resolution):** The softmax denominator sums over ALL items in the row/column. When you compute P(text_j | image_i), the denominator includes e^(sim(i,1)) + e^(sim(i,2)) + ... + e^(sim(i,N)). Every similarity in the row affects the probability of every match. This is the normalization that creates coupling.

3. **Batch size requirement:** CLIP trained with batch size 32,768. Not a hyperparameter choice--a structural requirement. Brief setup: "Why 32,768? What happens at smaller batch sizes? We will answer this concretely."

**3. Hook (the batch-size problem, concretely)**
Present the problem with numbers.

CLIP at batch=32,768: excellent zero-shot performance. CLIP at batch=256: dramatically worse. Not "slightly worse"--qualitatively different representations.

Why? Walk through the softmax denominator at both scales:
- Batch=4: softmax over 4 items. Each negative gets ~25% of the probability mass. The gradient for any one negative is 1/4 of the normalization.
- Batch=32,768: softmax over 32,768 items. Each negative gets ~0.003% of the probability mass. The gradient for any one negative is diluted by factor of ~8000x.

The "multiple choice exam" analogy: CLIP's loss asks "which of these N items is the correct match?" With 4 options, it is easy. With 32,768 options, the model must make fine-grained distinctions. But here is the problem: you NEED 32,768 options for the softmax to produce a meaningful probability distribution. With only 4 options, the softmax outputs are all close to 0.25, and the gradient signal is weak.

GradientCard: "CLIP's batch size is not a preference--it is a structural requirement of the softmax loss. Can we change the loss function so that batch size is a choice, not a constraint?"

**4. Explain Part 1 -- Sigmoid Loss (Core Concept)**
The solution: replace softmax (global normalization) with sigmoid (per-element scoring).

**The key insight:** Instead of asking "which of these N items is the correct match?" (multiple choice), ask "is this specific pair a match?" (true/false). Each cell in the NxN similarity matrix becomes an independent binary classification problem.

For a matching pair (diagonal): label = +1, target = sigmoid(similarity * t) should be close to 1.
For a non-matching pair (off-diagonal): label = -1, target = sigmoid(-similarity * t) should be close to 1 (equivalently, sigmoid(similarity * t) should be close to 0).

The loss for each cell: binary cross-entropy.
- Matching pair: -log(sigmoid(s_ij * t))
- Non-matching pair: -log(1 - sigmoid(s_ij * t)) = -log(sigmoid(-s_ij * t))

Compact form: loss_ij = -log(sigmoid(y_ij * s_ij * t)), where y_ij = +1 for matches, -1 for non-matches.

**Why this removes batch-size dependency:** The loss for cell (i,j) depends ONLY on s_ij and y_ij. It does not reference s_i1, s_i2, ..., s_iN. No denominator summing over the batch. No normalization. Each pair's gradient is self-contained.

**Side-by-side 4x4 matrix visual (SVG):** Left side shows softmax with arrows connecting all cells in each row (global normalization). Right side shows sigmoid with each cell independently colored (no connections between cells). The visual contrast communicates the core difference.

**Side-by-side code (PyTorch pseudocode):**
```
# CLIP loss
logits = sim * temperature
loss = (F.cross_entropy(logits, targets, dim=1)
      + F.cross_entropy(logits, targets, dim=0)) / 2

# SigLIP loss
logits = sim * temperature
labels = 2 * torch.eye(N) - 1  # +1 diagonal, -1 off-diagonal
loss = -F.logsigmoid(labels * logits).mean()
```

Annotated: the only difference is the loss computation. Same encoders, same similarity matrix, same temperature.

"Multiple choice vs true/false" analogy formalized: CLIP = N-way classification per row, SigLIP = N^2 binary classifications for the whole matrix.

**5. Check 1 (predict-and-verify)**
A batch has 4 image-text pairs. The similarity matrix is:

```
       text_0  text_1  text_2  text_3
img_0:  0.9     0.1     0.2     0.3
img_1:  0.2     0.8     0.1     0.4
img_2:  0.1     0.3     0.7     0.2
img_3:  0.3     0.2     0.1     0.6
```

Questions:
- In CLIP's loss, does the gradient for cell (0,1) depend on cell (0,3)? (Yes--both are in the same softmax row)
- In SigLIP's loss, does the gradient for cell (0,1) depend on cell (0,3)? (No--each cell is independent)
- If we add 1000 more pairs to the batch (now batch=1004), which loss function's computation for cell (0,1) changes? (Only CLIP's--the softmax denominator now has 1004 terms instead of 4)
- How many loss terms does CLIP compute? (2N = 8: 4 row-wise + 4 column-wise) How many does SigLIP compute? (N^2 = 16: one per cell)

**6. Explain Part 2 -- SigLIP in Practice**
Practical details of the original SigLIP (2023):

- Same ViT image encoder as CLIP (callback to "tokenize the image" from DiT lesson)
- Same transformer text encoder
- Learned temperature parameter (same as CLIP, callback to temperature intuition)
- Training: standard contrastive setup with sigmoid loss
- Key result: matches or exceeds CLIP performance at much smaller batch sizes. Works well from batch=256 to batch=32,768. CLIP only works well at the high end.
- Zero-shot classification works identically to CLIP (same shared space, same procedure)

**Brief practical note:** SigLIP has a subtle training consideration--with sigmoid loss, the ratio of positive to negative examples in each batch is 1:N (one match per image, N-1 non-matches). A bias term is learned to handle this class imbalance. This is a standard technique from binary classification.

**7. Explain Part 3 -- What SigLIP 2 Adds (2025 Improvements)**
SigLIP 2 keeps the sigmoid loss but adds training methodology improvements.

**Multi-stage training:**
- Stage 1: standard contrastive pretraining on large-scale image-text data (like original SigLIP)
- Stage 2: continued training with self-distillation targets from Stage 1 model
- Stage 3: multi-resolution fine-tuning (process images at multiple resolutions)

**Self-distillation (brief inline explanation since NOT TAUGHT):**
"Self-distillation uses the model from a previous training stage as a teacher for the next stage. Unlike standard knowledge distillation (where a large teacher trains a small student), self-distillation uses the same architecture. The model's earlier version provides soft targets--probability distributions rather than hard labels--that capture learned relationships the binary labels miss. Think of it as 'the model refining its own understanding by revisiting the same material with better judgment from its first pass.'"

**Multi-resolution:**
Images processed at multiple resolutions during the later training stages. This helps the model develop robust features that work across different image sizes and aspect ratios.

**Multilingual support:**
Training data includes multilingual image-text pairs, producing a vision encoder that works with text in many languages.

**Key takeaway for SigLIP 2:** The sigmoid loss was the original SigLIP contribution. SigLIP 2's contribution is the training recipe--multi-stage, self-distillation, multi-resolution. Different kinds of innovation: loss function design vs training methodology.

**8. Check 2 (transfer question)**
A colleague is training a vision-language model for a specialized medical imaging domain. They have 50,000 image-report pairs. They are using CLIP's softmax loss with batch_size=64 (limited by GPU memory with large medical images).

Questions:
- Why might CLIP's softmax loss perform poorly at batch_size=64?
- Would switching to SigLIP's sigmoid loss help? Why or why not?
- What else from SigLIP 2's recipe might be useful for this domain?

Expected: Softmax at batch=64 means each row's probability distribution is over only 64 items--the denominator is too small for reliable normalization, weak gradient signal. Sigmoid loss would help because each pair is scored independently, no batch-size minimum. For the domain: multi-resolution training is especially useful for medical images (which come in many sizes), and self-distillation could help the model refine features on the small dataset. Starting from a SigLIP 2 pretrained model and fine-tuning (transfer learning callback) would be the practical approach.

**9. Elaborate -- SigLIP as a Building Block**
SigLIP's real-world impact: it is the vision encoder in PaliGemma (Google's open VLM) and other models.

**Why contrastive pretraining produces good vision encoders:**
The shared embedding space learned during contrastive training means the vision encoder has learned to extract features that are MEANINGFUL in terms of language. When you use this encoder in a VLM (connecting it to a language model), the vision features are already partially aligned with the language model's representation space. This is why contrastive pretraining (CLIP/SigLIP) is the default recipe for vision encoders in VLMs.

**Connection to transfer learning:** "Hire experienced, train specific." The SigLIP encoder is the "experienced hire"--it understands images in terms that relate to language. Plug it into a VLM and train the connection.

**Brief mention of other downstream uses:**
- Zero-shot classification (same as CLIP)
- Image retrieval (find images matching a text query, or texts matching an image)
- Open-vocabulary detection (use the shared space to detect objects described by arbitrary text)

**10. Summarize**
Key takeaways:
1. CLIP's softmax cross-entropy loss normalizes across the entire batch, making batch size a structural requirement (not a preference)
2. SigLIP replaces softmax with sigmoid: each image-text pair is scored independently as a binary classification (match or not match)
3. This makes the loss batch-size independent--SigLIP works well at batch=256 or batch=32,768
4. SigLIP 2 adds training methodology improvements: multi-stage training, self-distillation, multi-resolution
5. SigLIP encoders are used as the vision backbone in VLMs like PaliGemma--contrastive pretraining produces vision features that are naturally aligned with language

Echo the mental model: "CLIP asks 'which of these N items is my match?' (multiple choice--harder with more options). SigLIP asks 'is this specific pair a match?' (true/false--same difficulty regardless of batch size). One line of code changes. Everything else stays the same."

**11. Next Step**
"SigLIP solved a specific problem in vision-language pretraining: making contrastive learning work without requiring enormous batch sizes. Next, we will look at a different kind of vision model: SAM (Segment Anything Model), which takes the 'foundation model' approach to image segmentation--training one model that can segment any object in any image, guided by a prompt."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (self-distillation gap addressed with inline explanation)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (contrastive loss INTRODUCED->DEVELOPED gap resolved in recap)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (6 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative/boundary)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2: sigmoid loss, SigLIP 2 training improvements)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-21 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: MAJOR REVISION

Critical finding exists (Check 2 asks about untaught concepts). Must fix before this lesson is usable.

### Findings

### [CRITICAL] — Check 2 Question 3 asks about SigLIP 2's training recipe before it has been taught

**Location:** "Check Your Understanding" section (Check 2), Question 3
**Issue:** Check 2 is placed BEFORE the "What SigLIP 2 Adds" section. Question 3 asks: "What else from SigLIP 2's recipe might be useful for this domain?" The reveal answer references multi-resolution training, self-distillation, and starting from a pretrained SigLIP 2 model. None of these concepts have been introduced at this point in the lesson. The student has only learned about the sigmoid loss so far, not SigLIP 2's training methodology.
**Student impact:** The student encounters a question they cannot answer. When they reveal the answer, they read about concepts (self-distillation, multi-resolution training) that have no meaning yet. This undermines the predict-and-verify pattern: the student cannot predict because they lack the knowledge. It also spoils the SigLIP 2 section by previewing its content in an answer block before the section builds the motivation and explanation.
**Suggested fix:** Either (a) move Check 2 to after the "What SigLIP 2 Adds" section, which would place the transfer question after all relevant concepts are taught, or (b) remove Question 3 from Check 2 and add a separate Check 3 after the SigLIP 2 section that tests transfer of the training improvements. Option (a) is simpler and preserves the medical imaging scenario as a unified transfer exercise. Note: this is also a plan-level issue (the outline places Check 2 before Explain Part 3 but includes Question 3 about SigLIP 2's recipe).

### [IMPROVEMENT] — Missing geometric/spatial modality (bucket vs independent gauges metaphor)

**Location:** Explain Part 1 and Hook sections
**Issue:** The planning document specifies 6 modalities, including a "Geometric/Spatial" modality: "Batch-size scaling diagram: show how the softmax denominator grows with batch size (small bucket with 4 items vs large bucket with 32768 items, each item's 'share' of the probability mass shrinking). Contrast with sigmoid: N independent gauges, each reading stays the same regardless of how many other gauges exist." This modality is absent from the built lesson. The GradientCards at batch=4 vs batch=32,768 partially cover the scaling behavior, but they present it as a bulleted list of facts, not as a spatial/geometric metaphor that communicates the fixed-probability-mass-divided-among-more-items intuition.
**Student impact:** The lesson still has 5 modalities (well above the minimum of 3), so this is not critical. However, the "bucket" metaphor would have provided a distinct spatial intuition that the other modalities do not: the idea that softmax has a fixed total (probability sums to 1) that must be divided among all items, while sigmoid has no such constraint. This is a qualitatively different way of understanding the distinction.
**Suggested fix:** Add a small inline SVG or a brief verbal paragraph with the bucket/gauge metaphor either in the Hook section (after the GradientCards) or in the "Why this removes batch-size dependency" subsection. The bucket metaphor: "Imagine a bucket containing exactly 1 unit of probability. With 4 items, each gets a substantial share. With 32,768 items, each gets a tiny drop. Now imagine N independent gauges, each reading between 0 and 1 independently. Adding more gauges does not change any existing gauge's reading." This can be 2-3 sentences; it does not need to be a major addition.

### [IMPROVEMENT] — Self-distillation explanation lacks a concrete example

**Location:** "Self-distillation: the model teaches itself" section (lines 967-1003)
**Issue:** The self-distillation explanation is purely verbal and abstract. It explains the concept well ("the model from a previous training stage as a teacher for the next stage," "soft targets rather than hard labels"), but it never grounds this in a concrete example. What does a "soft target" from Stage 1 actually look like? What does the Stage 2 training step look like concretely? The student is told self-distillation captures "learned relationships the binary labels miss" but is not shown what kind of relationship. This is a new concept (NOT TAUGHT) being introduced inline, which makes concrete grounding even more important.
**Student impact:** The student can parrot "the model teaches itself using soft targets from a previous checkpoint" but may not actually understand what this means in practice. Without a concrete example, self-distillation remains an abstraction rather than an intuition. At INTRODUCED depth, the student should be able to "explain in own words," which requires grounding beyond the verbal explanation.
**Suggested fix:** Add a brief concrete example. For instance: "In Stage 1, the model learns that 'a golden retriever on a beach' and 'a dog playing in sand' should have similar embeddings. The hard label says only the exact pair is a match (binary: 1 or 0). But the Stage 1 model's soft predictions capture that these descriptions are quite similar (0.85) even though they are not an exact match. In Stage 2, the model trains with these soft similarities as targets, learning that near-matches are not as wrong as complete mismatches." 3-4 sentences would suffice.

### [IMPROVEMENT] — Misconception 1 ("SigLIP is completely different from CLIP") not addressed as explicitly as planned

**Location:** Context/Recap sections
**Issue:** The planning document calls for Misconception 1 to be addressed in the "Context section and recap, before introducing sigmoid loss" with an explicit callout: "SigLIP uses the SAME dual-encoder architecture, SAME shared embedding space, SAME cosine similarity computation... The ONLY fundamental change is the loss function." The lesson addresses this implicitly through the "one line of code" framing and the code side-by-side, but it never has an explicit "SigLIP is NOT a different model" statement early in the lesson. The TipBlock (line 520-526) says "The entire architectural difference between CLIP and SigLIP is the loss function" but this appears deep in Explain Part 1, not in the context/recap where the misconception would form.
**Student impact:** A student who has not touched CLIP in months might approach this lesson assuming SigLIP is an entirely different system. The "one line of code" header description hints at the connection, but the student might still carry the "different model" assumption through the recap section until they reach the code comparison. Setting expectations earlier would prevent this.
**Suggested fix:** Add a brief statement in the Context section (near the ObjectiveBlock or ConstraintBlock) or at the start of the Recap section: "SigLIP uses the same dual-encoder architecture, the same shared embedding space, and the same training setup as CLIP. The only change is the loss function. Everything you know about CLIP's architecture and applications carries over directly." This can be 1-2 sentences.

### [POLISH] — SigLIP code example uses `F.cross_entropy` without `dim` argument, while plan specifies `dim=1` and `dim=0`

**Location:** Side-by-side code comparison, CLIP loss
**Issue:** The CLIP loss code shows `F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)` (using transpose). The planning document specifies `F.cross_entropy(logits, labels, dim=1) + F.cross_entropy(logits, labels, dim=0)`. The lesson's version using `.T` is arguably cleaner and more readable (transpose the matrix and apply the same operation twice, rather than specifying dim), so this is a reasonable deviation. However, the `.T` approach transposes the matrix and then applies cross-entropy along the default dimension, which achieves the same result. This is fine.
**Student impact:** Minimal. Both are correct representations of symmetric cross-entropy. The `.T` version may actually be clearer for someone reading the code.
**Suggested fix:** No change needed. The lesson's version is a reasonable simplification. Noting for the record that this is a deliberate deviation from the plan.

### [POLISH] — "SAM 3" next step may be confusing if the student does not know what SAM is

**Location:** NextStepBlock (lines 1225-1233)
**Issue:** The next step says "Up Next: SAM 3" with description "The Segment Anything Model--promptable image segmentation as a foundation model approach to vision." The "3" in "SAM 3" might confuse the student into thinking they missed SAM 1 and SAM 2. The lesson text above (lines 1216-1222) provides more context ("SAM (Segment Anything Model)") which helps, but the NextStepBlock title alone says "SAM 3" without clarification.
**Student impact:** Minor. The student might momentarily wonder if they should have studied SAM 1 and 2 first, but the description clarifies this is about image segmentation, and the course structure (Special Topics) implies standalone lessons.
**Suggested fix:** Consider changing the NextStepBlock title to "Up Next: SAM 3 (Segment Anything)" or adding a brief note that SAM 3 covers the latest version of the model with relevant context from earlier versions built in. Alternatively, leave as-is if the SAM 3 lesson itself handles the version context.

### Review Notes

**What works well:**
- The core pedagogical arc is strong. The "problem before solution" structure (batch-size dependency -> sigmoid loss) follows the Motivation Rule perfectly.
- The "multiple choice vs true/false" analogy is the lesson's strongest contribution. It communicates the core insight in one sentence and holds up throughout the lesson.
- The side-by-side SVG diagrams (connected vs independent cells) are effective visual communication.
- The concrete cell tracing with real numbers (sigma(9.0) = 0.00012, sigma(-2.0) = 2.13) grounds the formula in verifiable computation.
- The code side-by-side is clean and minimal, showing that the architectural difference is genuinely one line.
- The ComparisonRow (CLIP: 2N loss terms vs SigLIP: N^2 loss terms) effectively addresses the "sigmoid wastes negatives" misconception.
- Aside content is consistently relevant and well-targeted. The InsightBlock connecting softmax competition across attention/classification/generation is particularly strong.
- All Row components are used correctly. No manual flex layouts.
- Em dash formatting is consistent (no spaces).

**Patterns to watch:**
- The Check 2 sequencing issue (asking about untaught concepts) is a plan-level problem that carried through to implementation. When revising, the builder should verify that every check question tests only concepts that have been taught BEFORE that check in the lesson flow.
- Self-distillation is introduced inline as a NOT TAUGHT concept. The verbal explanation is adequate but would benefit from one concrete example to ground it. This pattern (inline introduction of a new concept) deserves extra attention on concreteness.

**Overall assessment:**
The lesson is pedagogically sound in its core arc (sections 1-8 covering sigmoid loss). The SigLIP 2 section is adequate for INTRODUCED depth. The critical issue is a sequencing error in Check 2 that would confuse a student. Once that is fixed and the improvements are addressed, this should be a clean PASS.

### Fix Pass — 2026-02-21

**Findings fixed:**

1. **[CRITICAL] Check 2 Question 3 asks about SigLIP 2 before taught.** FIXED. Removed Question 3 from Check 2 (which now has only Questions 1-2, testing sigmoid loss concepts). Added a separate Check 3 after the "SigLIP 2 = Two Kinds of Innovation" card, continuing the medical imaging scenario and asking about SigLIP 2's training recipe. The student now encounters this question only after multi-stage training, self-distillation, and multi-resolution have all been taught.

2. **[IMPROVEMENT] Missing geometric/spatial modality.** FIXED. Added the bucket vs independent gauges metaphor in the "Why this removes batch-size dependency" subsection. Brief paragraph: softmax = bucket with fixed 1 unit of probability divided among items; sigmoid = N independent gauges each reading 0-1 on their own.

3. **[IMPROVEMENT] Self-distillation lacks concrete example.** FIXED. Added a concrete example block after the verbal explanation showing hard target [0, 0, 1, 0] vs soft target [0.05, 0.10, 0.70, 0.15] for a "golden retriever on a beach" image, explaining that Stage 1's soft predictions capture that "a dog playing in sand" is similar even though it is not an exact match.

4. **[IMPROVEMENT] Misconception 1 not addressed early enough.** FIXED. Added an explicit statement at the start of the Recap section (before CLIP details): "SigLIP uses the same dual-encoder architecture, the same shared embedding space, and the same cosine similarity computation as CLIP. The only change is the loss function."

**Findings intentionally NOT addressed:**

5. **[POLISH] Code uses `.T` instead of `dim=` argument.** Reviewer noted this is a reasonable simplification. No change needed.

6. **[POLISH] "SAM 3" next step may confuse.** Left as-is. The SAM 3 lesson itself will handle version context, and the surrounding text already says "SAM (Segment Anything Model)."

---

## Review — 2026-02-21 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All four findings from iteration 1 (1 critical + 3 improvement) have been properly resolved. The fixes introduced no new issues. The lesson is pedagogically sound and ready to ship.

### Verification of Iteration 1 Fixes

1. **[CRITICAL] Check 2 Question 3 asked about SigLIP 2 before taught.** VERIFIED FIXED. Check 2 (lines 851-901) now contains only two questions testing sigmoid loss concepts (why softmax fails at batch 64, whether sigmoid loss helps). A separate Check 3 (lines 1075-1111) continues the medical imaging scenario and asks about SigLIP 2's training recipe. Check 3 appears AFTER the "What SigLIP 2 Adds" section, the self-distillation explanation, the multi-resolution section, and the "Two Kinds of Innovation" summary card. All concepts referenced in Check 3's answer (multi-resolution, self-distillation, transfer learning) have been taught before the student encounters the question. The fix is clean and the medical imaging scenario now threads naturally across two checks.

2. **[IMPROVEMENT] Missing geometric/spatial modality.** VERIFIED FIXED. The bucket vs independent gauges metaphor appears in the "Why this removes batch-size dependency" subsection (lines 355-363). It reads naturally: "softmax is a bucket containing exactly 1 unit of probability... Sigmoid is N independent gauges, each reading between 0 and 1 on its own." The metaphor is brief (one paragraph) and provides a genuinely distinct spatial intuition from the other modalities. All 6 planned modalities are now present.

3. **[IMPROVEMENT] Self-distillation lacks concrete example.** VERIFIED FIXED. A concrete example block (lines 989-1013) shows hard target `[0, 0, 1, 0]` vs soft target `[0.05, 0.10, 0.70, 0.15]` for "a golden retriever on a beach." The example effectively grounds the abstract concept: "near-matches are not as wrong as complete mismatches." The student can now understand self-distillation concretely, not just verbally.

4. **[IMPROVEMENT] Misconception 1 not addressed early enough.** VERIFIED FIXED. An explicit statement at the start of the Recap section (lines 110-115) says: "SigLIP uses the same dual-encoder architecture, the same shared embedding space, and the same cosine similarity computation as CLIP. The only change is the loss function." This appears before any CLIP details, setting expectations immediately. Misconception 1 is now addressed at the earliest possible point.

### Findings

### [POLISH] — Open-vocabulary detection omitted from downstream uses

**Location:** "Downstream uses of SigLIP encoders" section (lines 1155-1185)
**Issue:** The planning document's Elaborate section calls for a "brief mention" of open-vocabulary detection as a downstream use ("use the shared space to detect objects described by arbitrary text"). The built lesson has three GradientCards for downstream uses (Zero-Shot Classification, Image Retrieval, VLM Vision Backbone) but omits open-vocabulary detection. This is a minor scope deviation.
**Student impact:** Negligible. The three included downstream uses cover the most important applications. Open-vocabulary detection is a niche application that would add a term the student has not encountered (object detection) without enough context to make it meaningful.
**Suggested fix:** No change needed. The omission arguably improves focus. Noting for the record that this is a deliberate (or at least reasonable) deviation from the plan.

### Review Notes

**What the fixes improved:**
- The Check 2/Check 3 split is clean. The medical imaging scenario now threads across two checks naturally, testing sigmoid loss first and SigLIP 2's training recipe second. This is actually better than the plan's original single Check 2, because it provides two distinct checkpoints for two distinct concept groups.
- The early misconception callout ("SigLIP uses the same architecture...") sets expectations immediately and eliminates the risk of the student approaching the lesson with wrong assumptions.
- The bucket vs gauges metaphor adds a genuinely distinct modality. The lesson now has 6 modalities for the core concept, each providing a different angle of understanding.
- The self-distillation concrete example makes the difference between hard and soft targets tangible. The student can now "explain in own words" at INTRODUCED depth.

**Overall assessment:**
The lesson is ready to ship. The core pedagogical arc (problem before solution, batch-size dependency -> sigmoid loss) is strong. The "multiple choice vs true/false" analogy carries the entire lesson and communicates the core insight in one sentence. All 6 modalities are present and effective. All 5 misconceptions are addressed at appropriate points. The SigLIP 2 section is appropriately lighter (INTRODUCED depth). Check questions test only concepts that have been taught before they appear. No critical or improvement-level issues remain.

**Patterns that worked well in the revision:**
- Splitting one check into two (Check 2 + Check 3) rather than moving the entire check is a better pattern than the plan's original design. It tests concepts closer to where they were taught.
- Adding the early misconception callout as a brief paragraph at the start of the recap (rather than a separate block component) is unobtrusive and effective.
