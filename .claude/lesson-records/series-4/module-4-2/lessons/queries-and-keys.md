# Lesson: Queries and Keys

## Phase 1: Orient — Student State

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Dot-product attention on raw embeddings (Attention(X) = softmax(XX^T) X) | DEVELOPED | the-problem-attention-solves | Full step-by-step worked example: 4 tokens, 3-dim embeddings. All pairwise dot products, softmax row-wise, weighted average. Student traced every number. Formula built incrementally. |
| Weighted average as a mechanism for context (blend embeddings with weights that sum to 1) | DEVELOPED | the-problem-attention-solves | Built through three escalating attempts. Explicit formula and interpretation ("if w_3 is large, the output is mostly embed_3"). |
| Attention weight matrix (square matrix, row i = token i's weights over all tokens, rows sum to 1) | DEVELOPED | the-problem-attention-solves | Computed explicitly in worked example. Interactive AttentionMatrixWidget with presets, hover, toggle between raw scores and softmax. |
| Dual-role limitation (one embedding must serve as both "what I'm seeking" and "what I'm offering") | INTRODUCED | the-problem-attention-solves | Cocktail party analogy. ComparisonRow: "When This Token Is Looking" vs "When Other Tokens Look At It." Clearly stated but NOT resolved. |
| Attention matrix symmetry as design flaw (a dot b = b dot a, but real relationships are asymmetric) | INTRODUCED | the-problem-attention-solves | "The cat chased the mouse" example. Same pair, different reasons, same score. WarningBlock: raw scores symmetric, softmax weights not perfectly symmetric (different denominators). |
| Similarity vs relevance distinction (dot product measures similarity, but relevance is a different question) | INTRODUCED | the-problem-attention-solves | "The bank was steep" — "bank" and "steep" have low embedding similarity but "steep" is highly relevant to disambiguating "bank." |
| Data-dependent weights (attention weights freshly computed from each input, not fixed parameters) | DEVELOPED | the-problem-attention-solves | Explicitly contrasted with CNN fixed filters. InsightBlock: "The weight matrix W is not a learned parameter." |
| Dot product as similarity measure (large positive = same direction, zero = perpendicular, negative = opposite) | DEVELOPED | the-problem-attention-solves | Gap resolved from INTRODUCED. Three-panel geometric SVG plus numerical examples. Connected to cosine similarity formula. |
| Softmax (converting scores to probabilities) | DEVELOPED | what-is-a-language-model, Series 1 | Used repeatedly: classification, temperature slider, LM output, attention scores. Student is fluent with softmax mechanics. |
| Matrix multiplication | APPLIED | Series 1-2 | Used in training loops, forward passes, embedding lookup (one-hot x matrix = row selection). Student is fluent. |
| nn.Linear (learned weight matrix, input x W^T + b) | DEVELOPED | Series 2 (nn-module, 2.1.3) | Student has built models with nn.Linear layers, knows weight shape is (out, in), can predict parameter counts. |
| Vanishing gradients (quantitative) | DEVELOPED | training-dynamics (1.3) | Layer-by-layer table: 0.25^N through 10 layers. Sigmoid derivative max=0.25. Telephone game analogy. Unifying frame: products of local derivatives are unstable unless each factor near 1.0. |
| Softmax saturation (extreme inputs push softmax toward one-hot) | INTRODUCED | what-is-a-language-model | Temperature explorer showed T -> 0 concentrates mass on top token. Student understands softmax sharpens with larger inputs but hasn't formalized "saturation" or connected it to gradient problems. |
| Residual connections (skip connections) | DEVELOPED | Series 3 (modern-architectures) | "Add input to output so gradients can flow." Seed planted in the-problem-attention-solves: "attention output is ADDED to original embedding, not substituted." |
| Token embeddings as learned lookup | DEVELOPED | embeddings-and-position | nn.Embedding maps integer IDs to dense vectors. Parameter count: 50K x 768 = 38.4M. Embeddings are learned parameters, not preprocessing. |
| Complete input pipeline: text -> tokens -> IDs -> embeddings + PE -> tensor | DEVELOPED | embeddings-and-position | Full pipeline traced. Student knows what the transformer receives as input. |

**Mental models and analogies already established:**
- "Attention is a weighted average where the input determines the weights" — the defining insight from Lesson 1
- "The input decides what matters" (data-dependent weights vs fixed parameters) — paradigm shift from CNNs
- "Similarity is not the same as relevance" — raw dot products measure embedding similarity, but relevance requires more
- "One embedding can't serve two roles" (seeking vs offering) — the limitation that motivates Q/K
- Cocktail party analogy: what you're searching for vs what you advertise are different
- "Embeddings are a learned dictionary" — one definition per word
- "Products of local derivatives: the stability question" — vanishing/exploding gradients framework

**What was explicitly NOT covered:**
- Q, K, V projections — not even named in student-facing text (only the concept of "two separate vectors" in the transfer question reveal)
- Scaled dot-product attention (division by sqrt(d_k))
- Multi-head attention
- The transformer block architecture
- Causal masking
- How to fix the dual-role limitation (left as a cliffhanger)

**Readiness assessment:** The student is well-prepared. They have dot-product attention at DEVELOPED depth and viscerally felt the dual-role limitation. The transfer question at the end of Lesson 1 asked: "If every token had TWO vectors — one for seeking, one for advertising — would the matrix still be symmetric?" and the student reasoned that no, because s_A dot a_B != s_B dot a_A. This lesson delivers the answer they're primed for. The mathematical prerequisites (matrix multiplication, softmax, dot products) are all at DEVELOPED or APPLIED depth. nn.Linear is at DEVELOPED — important because Q and K are linear projections. Vanishing gradients at DEVELOPED provides the foundation for understanding why scaling by sqrt(d_k) matters.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand how learned linear projections (Q = W_Q @ embedding, K = W_K @ embedding) let each token create separate "seeking" and "offering" vectors, breaking the symmetry limitation of raw dot-product attention, and why scaling by sqrt(d_k) is necessary to prevent softmax saturation as embedding dimensions grow.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Dot-product attention on raw embeddings | DEVELOPED | DEVELOPED | the-problem-attention-solves | OK | Student must understand the full raw attention mechanism to see what Q and K change. Has computed it step-by-step. |
| Dual-role limitation (one embedding for both seeking and offering) | INTRODUCED | INTRODUCED | the-problem-attention-solves | OK | Student needs to recognize the problem so Q/K feels like the solution. INTRODUCED is sufficient — this lesson develops the fix. |
| Attention matrix symmetry as a flaw | INTRODUCED | INTRODUCED | the-problem-attention-solves | OK | Student recognizes that symmetric scores can't represent asymmetric relationships. This lesson shows how Q/K projections break symmetry. |
| Similarity vs relevance distinction | INTRODUCED | INTRODUCED | the-problem-attention-solves | OK | Provides the framing for why projections compute relevance rather than raw similarity. |
| Dot product as similarity | DEVELOPED | DEVELOPED | the-problem-attention-solves | OK | Student computes Q_i dot K_j — must understand what dot product measures. Fully developed with geometric SVG. |
| Softmax (scores to probability distribution) | DEVELOPED | DEVELOPED | Series 1, what-is-a-language-model | OK | Applied to attention score rows to produce weights. Student is fluent. |
| Matrix multiplication | APPLIED | APPLIED | Series 1-2 | OK | Q = W_Q @ X and QK^T are matrix multiplications. Student is fluent. |
| nn.Linear as learned matrix multiplication | DEVELOPED | DEVELOPED | Series 2 (nn-module) | OK | Q and K are computed via nn.Linear layers. Student knows nn.Linear applies a learned weight matrix. |
| Vanishing gradients (why small derivatives are problematic) | DEVELOPED | DEVELOPED | training-dynamics (1.3) | OK | Needed to understand why softmax saturation is dangerous — gradients vanish when softmax outputs are near 0 or 1. |
| Softmax saturation (extreme inputs -> near-one-hot output) | INTRODUCED | INTRODUCED | what-is-a-language-model | GAP | Student saw this with the temperature slider (T -> 0 concentrates mass) but hasn't connected it to the specific problem of "large dot products push softmax toward one-hot, killing gradients." Needs a bridge from "temperature slider behavior" to "dot products grow with dimension." |
| Linear projection / learned transformation (Wx as rotating/stretching a vector into a new space) | INTRODUCED | Not explicitly taught as "projection" | — | GAP | Student knows nn.Linear computes Wx + b and has used it in models. But the concept of a linear projection as "a learned lens that emphasizes certain features" has not been articulated. "Matrix multiplication transforms a vector" is implicit in everything but never framed as "projection into a new space." |

### Gap Resolution

| Concept | Gap Size | Resolution |
|---------|----------|------------|
| Softmax saturation and gradient connection | Small (has both pieces: softmax behavior from temperature widget + vanishing gradients from 1.3; needs the bridge) | Brief dedicated section (2-3 paragraphs) after introducing QK^T. Callback to temperature slider: "Remember how T=0.1 made softmax concentrate nearly all mass on one token? Large dot products have the same effect." Then connect to vanishing gradients: "When softmax outputs are near 0 and 1, gradients are near zero — the model can't learn." This bridges two existing concepts into the new context. Show with a concrete numerical example: d_k=3 vs d_k=512, average dot product magnitude scales with d_k. |
| Linear projection as a transformation concept | Small (has nn.Linear at DEVELOPED, has matrix multiplication at APPLIED; needs the "why" framing for projections) | Build into the main explanation of Q and K. When introducing W_Q, explicitly frame it: "W_Q is a learned matrix that transforms the embedding into a different vector — one that represents what this token is looking for. The same embedding, multiplied by a different matrix W_K, produces a different vector representing what this token offers. The matrix is the lens; the same input looks different through different lenses." Connect to nn.Linear: "In PyTorch, this is just nn.Linear(d_model, d_k, bias=False)." Keep this integrated, not a separate section. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Q and K are properties of the token itself (like part of speech or semantic role)" | Analogies like "database query" or "library lookup" suggest Q/K are inherent to the word. Names like "query" and "key" suggest fixed roles. | The same token "bank" produces DIFFERENT Q and K vectors depending on the learned W_Q and W_K matrices. Change the weights (i.e., train longer), and the same token's Q and K change. Moreover, the same token in different layers of the same model gets different Q and K vectors because each layer has its own projection matrices. Q and K are not about the token — they're about what the layer has learned to look for. | After introducing Q = W_Q @ embedding. Explicit WarningBlock: "Q and K are not properties of the token. They're properties of the learned projection matrices applied to the token's embedding." |
| "Scaling by sqrt(d_k) is just a normalization trick (cosmetic, not essential)" | Scaling factors often feel like arbitrary cleanup. The student might view division by sqrt(d_k) as optional fine-tuning. | Without scaling, a model with d_k=512 produces dot products averaging ~512 in magnitude. Softmax of [512, 0, 0, 0] is [1.0, 0.0, 0.0, 0.0] — completely one-hot. The model can't learn because gradients are zero everywhere except one position. With scaling, dot products average ~1.0, and softmax produces a useful distribution. Training collapses without the scaling factor. | Dedicated section on scaling. Concrete numerical comparison: d_k=3 (fine without scaling) vs d_k=512 (catastrophic without scaling). Connect to vanishing gradients callback. |
| "Q and K operate in different spaces (like keys and locks are made of different material)" | The "key-and-lock" metaphor suggests Q and K are fundamentally different types of objects. | Q and K are produced by the SAME type of operation (matrix multiplication) on the SAME input (the embedding vector), into the SAME dimensional space (d_k). They can be dot-producted because they're in the same space. The difference is the learned matrix, not the space. If they were in different spaces, dot products between them would be meaningless. | After showing the formulas. Explicitly state: "Q and K live in the same d_k-dimensional space. That's why the dot product between them is meaningful." |
| "The attention score matrix QK^T should still be symmetric" | Student learned that XX^T is symmetric. They might assume adding projection matrices doesn't change this. | Take the same 4-token example from Lesson 1. Compute Q = W_Q @ X and K = W_K @ X with small concrete matrices. Show that QK^T has entry (i,j) = q_i dot k_j and entry (j,i) = q_j dot k_i. Since q_i != k_i (different projection matrices), these are different values. Print the matrix and visually confirm it's not symmetric. This is the resolution of the Lesson 1 cliffhanger. | Immediately after computing QK^T in the worked example. Call back to the transfer question: "Remember asking whether two separate vectors would break symmetry? Here's the answer." |
| "Larger d_k is always better (more dimensions = more information)" | Intuition from embeddings: larger dimension captures more nuance. | Larger d_k means larger dot products, which means sharper softmax (closer to one-hot), which means less information flows and gradients vanish. There's a direct tradeoff: more dimensions capture finer-grained relevance, but the scaling factor must compensate or the model can't train. This is also preparation for multi-head attention (Lesson 4) where d_k = d_model / n_heads — you trade per-head dimension for number of heads. | In the scaling section, as motivation for WHY we divide by sqrt(d_k). Sets up multi-head dimension tradeoff. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| 4-token sentence, 3-dim embeddings with concrete W_Q and W_K matrices: full hand trace of Q, K, QK^T, scaling, softmax | Positive | Make the entire Q/K computation concrete and traceable. Student computes every number. Proves asymmetry of QK^T. | Directly extends the Lesson 1 worked example (same 4 tokens, same embeddings). Student sees the SAME input processed differently. The "before and after" effect is powerful: same embeddings, now with projections, different attention pattern. |
| "The cat chased the mouse" — asymmetric attention resolved | Positive | Show that Q/K projections CAN produce asymmetric scores, resolving the Lesson 1 limitation. "cat" and "chased" now get different scores depending on who is looking at whom. | Callbacks to the specific example where the symmetry limitation was demonstrated. The student felt this problem; now they see it fixed. Closes the loop. |
| Same embeddings, DIFFERENT W_Q/W_K matrices produce different attention patterns | Positive (stretch) | Demonstrate that Q and K are properties of the learned matrices, not the tokens. Same input, different projections, different behavior. | Directly disproves the "Q/K are token properties" misconception. Shows the matrices are the interesting learned component, not some inherent property of words. |
| d_k=3 vs d_k=512: dot product magnitudes and softmax behavior | Negative | Show that without scaling, high-dimensional dot products produce softmax saturation. Gradients die. Training fails. | Motivates the scaling factor as essential rather than cosmetic. Concrete numbers (avg dot product ~3 vs ~512) make the problem visceral. Callbacks to vanishing gradients. |
| Raw XX^T vs scaled QK^T/sqrt(d_k) on the SAME input | Negative | Contrast raw attention (Lesson 1) with Q/K attention. Show that raw attention treats similarity as relevance; Q/K attention computes learned relevance. | The "bank" and "steep" example from Lesson 1 showed similarity != relevance. This example shows Q/K can learn to score "steep" highly for "bank" even though their embeddings aren't similar — the projections LEARN to map them into a space where the dot product reflects relevance. |

---

## Phase 3: Design

### Narrative Arc

The previous lesson ended with a crack in the wall. Raw dot-product attention works — tokens can create context-dependent representations by computing similarities and blending neighbors. But there's a structural flaw: each token has ONE embedding vector that must serve double duty. When token A is the focus, that vector determines what it's looking for (it wants context about what the cat DID). When token A is in the background, the SAME vector determines what it offers to other tokens (it advertises that it's a cat, a noun, an agent). These are fundamentally different questions, but one vector can only answer one way. The attention matrix is symmetric — A's score for B equals B's score for A — and real language relationships are not symmetric.

The student felt this in the transfer question: "If you had two separate vectors — one for seeking, one for advertising — would the matrix still be symmetric?" They answered no. This lesson delivers the mechanism. Two learned linear projection matrices, W_Q and W_K, transform the same embedding into two different vectors: a query (what am I looking for?) and a key (what do I have to offer?). The dot product between a query and a key computes relevance — not similarity between raw embeddings, but a learned function that the model trains to distinguish useful from useless context. The attention matrix QK^T is no longer symmetric, because q_i dot k_j is not the same as q_j dot k_i.

But there's a subtle trap: as the dimension of Q and K grows (and it must grow — real models use 64 or 128 dimensions per head), the dot products grow too. Large dot products push softmax toward one-hot distributions, which kills gradients. The fix is simple but essential: divide by sqrt(d_k). The student already has the pieces — they saw softmax sharpen with low temperature, and they learned why vanishing gradients are catastrophic. This lesson connects those pieces into a specific, practical mechanism.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | Same 4 tokens from Lesson 1 (3-dim embeddings), but now with explicit 3x3 W_Q and W_K matrices. Compute Q = W_Q @ X, K = W_K @ X, QK^T, scale, softmax — every number visible. Compare the resulting attention weights to the raw XX^T weights from Lesson 1. | The abstract concept "learned projections compute relevance instead of similarity" becomes "multiply these matrices and watch the numbers change." Using the same tokens from Lesson 1 creates a powerful before/after comparison. |
| Visual | Side-by-side attention heatmaps: left = raw XX^T (symmetric, from Lesson 1), right = QK^T (asymmetric, from this lesson). Same tokens, different patterns. Color intensity = attention weight. The asymmetry should be visually striking. | The symmetry breaking is the core insight. Seeing it as two heatmaps side by side makes the abstract concept ("projections break symmetry") immediately visceral. The student should look at the right heatmap and see that row i, column j is different from row j, column i. |
| Symbolic | Q = W_Q X, K = W_K X, scores = QK^T / sqrt(d_k), weights = softmax(scores, dim=-1). Build up piece by piece, each line connected to its purpose. | The formula is the anchor. Building it in four labeled steps (project, project, score, normalize) prevents the monolithic "softmax(QK^T/sqrt(d))V" from appearing without context. Each step has a name and a reason. |
| Verbal/Analogy | Job fair analogy: each person writes two cards — "What I'm looking for in a job" (query) and "What I bring to the table" (key). A recruiter matches job-seeker cards to job-provider cards. The same person's "looking for" card is different from their "bring to the table" card. | Extends the cocktail party analogy from Lesson 1 into something more concrete and naturally asymmetric. A job fair makes the asymmetry intuitive: what you want and what you offer are obviously different. |
| Intuitive | "The projection matrix is a learned lens." Same embedding, viewed through W_Q lens, reveals what the token is seeking. Viewed through W_K lens, reveals what the token offers. The lens is what the model learns during training. Different layers learn different lenses. | Addresses the "Q/K are token properties" misconception by making the projection matrix the interesting object, not the token. "Same person, different lens, different view" is a quick intuitive bridge. |
| Geometric/Spatial | 2D projection diagram: embedding in original space projected through two different matrices into Q-space and K-space. Same point, different projections, different coordinates. The dot product in the projected space measures relevance, not raw similarity. | The geometric interpretation of "projection" makes it concrete. The student sees that the same vector lands at different locations when projected through different matrices. This prevents the misconception that Q and K are in different spaces — they're in the same space but arrived there via different transformations. |

### Cognitive Load Assessment

- **New concepts in this lesson:** (1) Q and K as learned linear projections, (2) Scaling by sqrt(d_k) to prevent softmax saturation. That's 2 new concepts — well within the limit.
- **Previous lesson load:** STRETCH (the-problem-attention-solves). Three new concepts, first encounter with attention, new paradigm.
- **Assessment:** BUILD is appropriate. The module plan designates this as BUILD, and the analysis confirms it. The student already has dot-product attention at DEVELOPED depth. Q and K are the targeted fix for a limitation they already feel. The scaling factor bridges two existing concepts (softmax behavior + vanishing gradients). There's no new paradigm here — just a precise, motivated extension of the existing paradigm. The lesson should be tighter than Lesson 1.

### Connections to Prior Concepts

- **Dual-role limitation** (the-problem-attention-solves, INTRODUCED): Direct resolution. "You felt the problem. Here's the fix."
- **Attention matrix symmetry** (the-problem-attention-solves, INTRODUCED): Broken by Q/K. Side-by-side comparison makes this explicit.
- **Transfer question from Lesson 1** ("two separate vectors -> asymmetric matrix?"): The student already reasoned through the answer. This lesson confirms and develops it.
- **Dot-product attention formula** (the-problem-attention-solves, DEVELOPED): Extended from softmax(XX^T)X to softmax(QK^T/sqrt(d_k)). Same structure, two additions (projections + scaling).
- **nn.Linear** (Series 2, DEVELOPED): Q and K are just nn.Linear layers. "You've been using these since Module 2.1."
- **Vanishing gradients** (training-dynamics 1.3, DEVELOPED): Callback for the scaling section. "You already know what happens when gradients vanish — the telephone game, the frozen layers."
- **Temperature slider** (what-is-a-language-model, INTRODUCED): Callback for softmax saturation. "You already saw what happens when softmax inputs get extreme — the distribution collapses to one-hot."
- **Similarity vs relevance** (the-problem-attention-solves, INTRODUCED): Q/K projections learn relevance, not similarity. Extends this distinction from observation to mechanism.

**Analogies that could be misleading:**
- "Cocktail party" from Lesson 1 — this analogy established seeking vs offering, which is good. But it might suggest that Q and K are about the content of what the person says/hears. In reality, Q and K are learned features — the model decides WHAT aspect to seek/offer through training. The job fair analogy is more precise because the cards are explicitly written (learned) rather than being inherent properties of the person.
- "Database query and key lookup" — the classic analogy from transformers literature. This lesson should NOT use this analogy because it implies exact matching (database lookups are binary: match or no match) while attention computes graded similarity. It also suggests Q and K serve fundamentally different structural roles, when they're the same operation (linear projection) with different learned weights.

### Scope Boundaries

**This lesson IS about:**
- Why Q and K exist (resolving the dual-role limitation from Lesson 1)
- What Q and K compute (learned linear projections of the same embedding)
- How QK^T produces an asymmetric relevance matrix
- Why scaling by sqrt(d_k) is essential (softmax saturation, vanishing gradients)
- Tracing the full computation by hand with 4 tokens and concrete numbers
- Target depths: Q/K projections at DEVELOPED, scaling factor at DEVELOPED

**This lesson is NOT about:**
- V projection (what a token contributes when attended to) — that's Lesson 3
- The full attention output (the weighted sum using V) — Lesson 3
- Multi-head attention — Lesson 4
- The transformer block — Lesson 5
- Causal masking — Lesson 6
- Implementing complete attention in PyTorch (only Q, K, and the score matrix — the output step requires V)
- Cross-attention (all attention in this module is self-attention)
- Why Q and K might have different dimensions than the embedding (d_k != d_model is a multi-head concern, Lesson 4)
- Attention as a complete layer (no residual connection, no output projection yet)

### Lesson Outline

1. **Context + Constraints**
   - "Last lesson, you built dot-product attention from scratch and felt its limitation: one embedding per token for both seeking and offering. This lesson fixes that with two learned projection matrices."
   - Explicit scope: We add Q and K projections and the scaling factor. We do NOT yet have V (the output projection) — that's next lesson. By the end, you'll compute the full attention WEIGHT matrix with projections, but not the final output.
   - Has notebook: `4-2-2-queries-and-keys.ipynb`

2. **Hook: The Cliffhanger Resolution**
   - Callback to the transfer question from Lesson 1: "If every token had two vectors — one for seeking, one for advertising — would the attention matrix still be symmetric?"
   - "You answered no. Let's prove it and build the mechanism."
   - Quick visual reminder: the symmetric attention heatmap from Lesson 1, the "cat chased mouse" asymmetry problem. 1-2 sentences, not a full recap — the student just did this lesson.

3. **Explain: Two Projections from One Embedding**
   - The conceptual move: one embedding -> two different vectors via two different learned matrices.
   - Job fair analogy: every attendee writes two cards. "What I'm looking for" (query) and "What I bring to the table" (key). A recruiter scores each seeker-card against each provider-card. One person's seeker-card is different from their provider-card.
   - Formula introduction: Q = W_Q @ X, K = W_K @ X. Two separate matrices, applied to the same input.
   - "Learned lens" framing: W_Q is a lens that reveals what the token is seeking. W_K is a lens that reveals what the token offers. Same embedding, different lens, different view. The lens is what the model learns during training.
   - Connection to nn.Linear: "In PyTorch, W_Q is just nn.Linear(d_model, d_k, bias=False). You've been using these since Module 2.1."
   - WarningBlock: "Q and K are NOT properties of the token. They're properties of the learned projection matrices. Change the matrices (by training), and the same token's Q and K change."
   - Geometric/spatial: brief 2D projection diagram showing same embedding vector projected through two different matrices, landing at two different points in the projected space.

4. **Explain: The Relevance Matrix QK^T**
   - QK^T replaces XX^T. Entry (i,j) is q_i dot k_j — "how much does token i's query match token j's key?"
   - Key distinction: this is not similarity between embeddings. It's a learned relevance function. The projections transform embeddings into a space where dot products measure "is this token's offering what that token is seeking?"
   - Connection to similarity vs relevance from Lesson 1: "Remember 'bank' and 'steep' — low similarity but high relevance. W_Q and W_K can learn to project them so that in Q/K space, their dot product IS large."
   - Why QK^T is asymmetric: entry (i,j) = q_i dot k_j; entry (j,i) = q_j dot k_i. Since q_i = W_Q @ x_i and k_i = W_K @ x_i, and W_Q != W_K, these are different. "The seeking vector of token A dotted with the offering vector of token B is NOT the same as B's seeking dotted with A's offering."

5. **Check: Predict Asymmetry**
   - Give the student "The cat chased the mouse" with Q/K projections. Ask: should "cat"'s attention to "chased" equal "chased"'s attention to "cat"? Why not?
   - Expected reasoning: "cat" seeks what-action-did-I-do, "chased" seeks who-did-the-chasing. Different queries, same keys. Different scores.

6. **Explain: Worked Example with Concrete Numbers**
   - Same 4 tokens from Lesson 1 ("The", "cat", "sat", "here"), same 3-dim embeddings.
   - Provide concrete 3x3 W_Q and W_K matrices (small integers for hand-traceability).
   - Step 1: Compute Q = W_Q @ X^T (each column is a query vector). Show all 4 query vectors.
   - Step 2: Compute K = W_K @ X^T (each column is a key vector). Show all 4 key vectors.
   - Aside: "Same embeddings, different matrices, different vectors. This is the 'learned lens' in action."
   - Step 3: Compute QK^T (4x4 matrix). Show all 16 entries.
   - Step 4: Point out: this matrix is NOT symmetric. Compare row (1,2) to row (2,1). Callback: "Remember XX^T from last lesson was symmetric. Not anymore."
   - Step 5: Apply softmax to each row. Show the attention weights.
   - Side-by-side comparison: the raw XX^T weights from Lesson 1 (symmetric) vs the QK^T weights (asymmetric). Same tokens, different attention patterns.

7. **Explore: Interactive Widget or Notebook Moment**
   - Direct the student to the notebook. Key exercises:
     - Implement Q, K projection from scratch (matrix multiplication, no nn.Linear yet)
     - Compute QK^T and verify asymmetry
     - Visualize as heatmap, compare to raw XX^T heatmap from Lesson 1 notebook
     - Experiment: what happens if W_Q == W_K? (Symmetry returns! This proves the asymmetry comes from the matrices being different.)
   - TryThisBlock: try setting W_Q = W_K and observe the symmetry. Then make them different and watch it break.

8. **Elaborate: Why Scaling by sqrt(d_k) Matters**
   - Problem setup: "We've been using d_k=3. Real models use d_k=64 or d_k=128. What changes?"
   - Concrete calculation: if each element of q and k is drawn from a distribution with mean 0 and variance 1, then q dot k has mean 0 and variance d_k. At d_k=3, typical dot products are around +/-1.7. At d_k=512, typical dot products are around +/-22.6.
   - Callback to temperature: "Remember the temperature slider? T=0.1 made softmax nearly one-hot. Large dot products have exactly the same effect — they act like dividing by a tiny temperature."
   - Show concrete softmax: softmax([22, -15, 3, 8]) -> [~1.0, ~0.0, ~0.0, ~0.0]. Nearly all mass on one token.
   - Callback to vanishing gradients: "You already know what happens when products of local derivatives are tiny — the telephone game from Module 1.3. Softmax near 0 or 1 has near-zero gradient. The model stops learning."
   - The fix: divide by sqrt(d_k) before softmax. scores = QK^T / sqrt(d_k). At d_k=512, this brings dot products back to variance ~1.0, keeping softmax in a useful range.
   - Negative example: d_k=3 vs d_k=512 side by side. Without scaling: d_k=3 works fine, d_k=512 collapses. With scaling: both produce useful distributions.
   - Why sqrt(d_k) specifically? Variance of dot product = d_k. Dividing by sqrt(d_k) gives variance = 1. "It's not arbitrary — it's the exact normalization that keeps the variance of the scores constant regardless of dimension."

9. **Check: Transfer Question**
   - "A colleague says: 'I removed the sqrt(d_k) scaling and my small d_k=8 model still trains fine, so scaling doesn't matter.' What would you tell them?"
   - Expected: scaling matters more as d_k grows. At d_k=8, dot products aren't extreme enough to saturate softmax. At d_k=64 or d_k=128, they would be. The colleague's test doesn't generalize.

10. **Practice: Notebook** (`4-2-2-queries-and-keys.ipynb`)
    - Scaffolding: supported (less scaffolding than Lesson 1 — student already implemented raw attention)
    - Exercise 1: Implement Q and K projections. Given embeddings X (4 tokens, d=64) and weight matrices W_Q, W_K (64x64), compute Q, K, and the score matrix QK^T. Verify it's not symmetric.
    - Exercise 2: Implement scaled attention weights. Divide scores by sqrt(d_k), apply softmax. Compare the attention weight distribution with and without scaling.
    - Exercise 3: Experiment with dimension. Set d_k=8, d_k=64, d_k=512. For each, compute QK^T WITHOUT scaling and look at the softmax output. At what dimension does the distribution become effectively one-hot?
    - Stretch exercise: Use pretrained GPT-2 embeddings. Extract one attention head's W_Q and W_K matrices. Compute attention weights for a sentence with asymmetric relationships (like "The cat chased the mouse"). Is the attention pattern asymmetric? Compare to the raw XX^T pattern from the Lesson 1 notebook.

11. **Summarize**
    - Three key takeaways:
      1. Q and K are learned linear projections that separate the "seeking" and "offering" roles, breaking the symmetry of raw attention.
      2. QK^T computes a learned relevance function, not raw embedding similarity.
      3. Scaling by sqrt(d_k) prevents softmax saturation as dimensions grow, keeping gradients alive.
    - The complete formula so far: weights = softmax(QK^T / sqrt(d_k)). But we're not done — we have attention WEIGHTS but no attention OUTPUT. The weights tell us how much each token should attend to each other token, but when we compute the final weighted average, what vectors do we average? Right now, we'd use the raw embeddings X. But there's a problem...
    - Seed the next lesson's limitation: "The key that makes a token relevant (K) and the information it should provide when attended to are different things. 'Chased' is relevant to 'cat' because of its action semantics (captured by K), but what 'chased' CONTRIBUTES to 'cat's representation should be different — maybe verb tense, transitivity, or past-action features. Should we use the same embedding for matching AND contributing? We already know the answer to that kind of question."

12. **Next Step**
    - "Next: Values and the Attention Output — the third projection that separates 'what makes me relevant' from 'what information I provide.'"
    - The student should feel: "Q and K fixed matching. There's one more role separation needed."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth (11 concepts in prereq table)
- [x] Each traced via the records (the-problem-attention-solves record, Module 4.1 record, Module 1.3 record, Series 2 records)
- [x] Depth match verified for each (9 OK, 2 small GAPs)
- [x] No untaught concepts remain (both GAPs have resolution plans)
- [x] No multi-concept jumps in exercises (notebook builds on Lesson 1 notebook, same style, less scaffolding)
- [x] All gaps have explicit resolution plans (softmax saturation bridged in scaling section, linear projection framed in Q/K introduction)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (the cliffhanger resolution)
- [x] At least 3 modalities planned: concrete example, visual (side-by-side heatmaps), symbolic (formulas), verbal/analogy (job fair), intuitive (learned lens), geometric/spatial (2D projection diagram) — 6 modalities
- [x] At least 2 positive examples + 1 negative example: 3 positive (hand trace, asymmetric resolution, different W_Q/W_K) + 2 negative (d_k scaling, raw vs projected)
- [x] At least 3 misconceptions: 5 identified with negative examples (Q/K as token properties, scaling as cosmetic, Q/K in different spaces, QK^T still symmetric, larger d_k always better)
- [x] Cognitive load = 2 new concepts (well within limit)
- [x] Every new concept connected to existing concepts (Q/K -> dual-role limitation + nn.Linear; scaling -> temperature + vanishing gradients)
- [x] Scope boundaries explicitly stated (no V, no multi-head, no output computation)

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 5
- Polish: 3

### Verdict: NEEDS REVISION

No critical conceptual failures, but one factually incorrect numerical example that would confuse any student who checks the numbers. Five improvement-level findings that collectively weaken the lesson's effectiveness in important ways.

### Findings

### [CRITICAL] — Softmax example values are factually incorrect

**Location:** Section 8 (Why Scaling by sqrt(d_k) Matters), the "What softmax does with large inputs" box
**Issue:** The lesson shows:
```
softmax([1.7, -0.3, 0.5, 0.8]) -> [0.38, 0.05, 0.11, 0.15]
```
The actual softmax values are `[0.543, 0.073, 0.163, 0.221]`. The lesson's numbers sum to only 0.69, not 1.0. A student who has softmax at DEVELOPED depth (which this student does) will immediately notice these don't sum to 1 and either lose trust in the lesson or think they misunderstand softmax. In a lesson that relies heavily on numerical tracing as a modality, any factual error in the numbers is corrosive.
**Student impact:** Confusion and loss of trust. The student has been trained to trace numbers carefully. Incorrect numbers in a pedagogical example undermine that habit.
**Suggested fix:** Replace with correct values: `[0.543, 0.073, 0.163, 0.221]` or choose a different input vector whose softmax produces rounder numbers that are easier to read at a glance.

### [IMPROVEMENT] — Misconception #5 ("larger d_k is always better") not addressed

**Location:** Missing from the lesson entirely
**Issue:** The planning document identifies 5 misconceptions. Misconception #5 ("Larger d_k is always better — more dimensions = more information") is not addressed anywhere in the built lesson. The scaling section explains why large d_k causes problems and how scaling fixes them, but it never explicitly addresses the intuition that "more dimensions should be better." The plan specified this should appear in the scaling section as motivation and as preparation for multi-head attention (where d_k = d_model / n_heads).
**Student impact:** The student may leave thinking d_k should be as large as possible (just scale it and you're fine), missing the tradeoff that sets up multi-head attention in Lesson 4.
**Suggested fix:** Add 1-2 sentences in the scaling section or in an aside noting that larger d_k captures finer-grained relevance but requires more parameters, and that in practice d_k is intentionally kept small (64) because models use multiple heads that split the dimension. Frame it as a teaser: "Why not just use d_k = d_model? We'll see in the multi-head attention lesson."

### [IMPROVEMENT] — Misconception #3 ("Q and K are in different spaces") addressed too subtly

**Location:** Line 382, a caption below the geometric/spatial SVG diagram
**Issue:** The "Q and K live in the same d_k-dimensional space" statement is buried in a small-text caption below an SVG. The planning document calls for this to be explicitly stated after the formulas, making it clear that the dot product between Q and K is meaningful precisely because they are in the same space. As currently placed, it reads as an incidental technical note rather than a deliberate misconception correction.
**Student impact:** A student with the "keys and locks are different material" intuition would likely skim past this caption. The misconception is common enough that it deserves more prominent placement.
**Suggested fix:** Move or duplicate this point into the main prose in Section 4 (The Relevance Matrix QK^T), where the lesson explains why entry (i,j) is q_i dot k_j. Something like: "This dot product is meaningful because Q and K live in the same d_k-dimensional space — they arrived there via different matrices, but they share the same coordinate system."

### [IMPROVEMENT] — Planned positive example "different W_Q/W_K produce different patterns" is only in the notebook, not in the lesson body

**Location:** Missing from the lesson prose; only appears as a notebook experiment (Section 7, "Explore")
**Issue:** The planning document lists "Same embeddings, DIFFERENT W_Q/W_K matrices produce different attention patterns" as a positive example that demonstrates Q and K are properties of the learned matrices, not the tokens. In the built lesson, this is relegated to a notebook experiment ("set W_Q = W_K and watch symmetry return"). While the notebook coverage is good, the lesson body itself should contain at least a brief mention of this insight as prose, not just as an experiment the student may or may not do.
**Student impact:** A student who reads the lesson but doesn't immediately do the notebook misses this key insight. The lesson addresses the misconception in the WarningBlock (line 389-396), but the concrete demonstration that changing W_Q/W_K changes the pattern only exists in the notebook.
**Suggested fix:** Add 1-2 sentences in the worked example section or in an aside: "If we used different W_Q and W_K matrices, the attention pattern would be completely different — same tokens, same embeddings, different behavior. The matrices are what the model learns, and they determine the attention pattern."

### [IMPROVEMENT] — Two notebook link sections create redundancy and blur the lesson's pacing

**Location:** Section 7 ("Build It: Q, K, and the Score Matrix") and Section 10 ("Practice: Full Q/K Implementation")
**Issue:** The lesson has two separate notebook sections that both link to the same Colab notebook. Section 7 covers: implement Q/K from scratch, compute QK^T, verify asymmetry, experiment with W_Q = W_K. Section 10 covers: implement Q/K with d=64, implement scaled attention, experiment with dimensions, stretch with GPT-2. The plan intended one notebook moment (Section 7 in the outline, "Explore") and one practice section (Section 10, "Practice"). But in the built lesson, Section 7 already has implementation exercises, making Section 10 feel repetitive. A student arriving at Section 10 would wonder: "Didn't I already do this?"
**Student impact:** The two notebook sections break the lesson's flow. The scaling section (8-9) is sandwiched between two notebook prompts, which means the student is asked to stop and code, then read more theory, then stop and code again. This is disruptive, especially for an ADHD-friendly design.
**Suggested fix:** Consolidate into a single notebook section after the scaling material. Section 7 could become a brief "you'll implement this in the notebook at the end" teaser (1-2 sentences), and Section 10 could be the single notebook prompt that covers all exercises (Q/K implementation, asymmetry verification, W_Q = W_K experiment, scaling experiment, dimension experiment, GPT-2 stretch).

### [IMPROVEMENT] — The "bank"/"steep" callback in Section 4 makes a claim the student can't verify

**Location:** Section 4, paragraph about similarity vs relevance
**Issue:** The lesson states: "W_Q and W_K can learn to project them so that in Q/K space, their dot product IS large — because the projections learn to map vectors into a space where the dot product reflects relevance, not raw similarity." This is conceptually correct but not demonstrated anywhere. The worked example uses 4 tokens ("The", "cat", "sat", "here") and hand-picked matrices — it doesn't show the "bank"/"steep" case with learned matrices. The claim is asserted, not proven.
**Student impact:** The student might accept this on faith, but the lesson's pedagogical approach is "show, don't assert." This is an assertion without evidence. It connects to Lesson 1 well but doesn't close the loop with a concrete demonstration.
**Suggested fix:** Either (a) acknowledge explicitly that this is what training WOULD achieve ("In a trained model, W_Q and W_K would learn projections that make 'steep' and 'bank' produce high q dot k scores"), or (b) add this as a notebook experiment using the GPT-2 stretch exercise. Option (a) is simpler and sufficient — the student just needs to understand the mechanism's capability, not see a trained example.

### [POLISH] — Em-dash spacing violations

**Location:** Lines 310, 314, 737, 769, 954
**Issue:** The writing style rule requires em dashes with no spaces: `word—word`. Several instances use `&ensp;&mdash;&ensp;` (lines 310, 314) or ` &mdash; ` (lines 737, 769, 954). The `&ensp;` usage on lines 310 and 314 is borderline — it may be intentional formatting for the formula display. The spaces on lines 737, 769, and 954 are in labels/badges, which is a very minor context.
**Student impact:** None — purely a style consistency issue.
**Suggested fix:** Remove spaces around em dashes in the label text. For the formula display lines (310, 314), consider whether the `&ensp;` spacing is a deliberate design choice for readability after an InlineMath component; if so, document the exception.

### [POLISH] — The "here" token in the worked example is semantically weak

**Location:** Section 6 (Worked Example), the 4-token set
**Issue:** The tokens "The", "cat", "sat", "here" are carried over from Lesson 1 for continuity. However, when discussing asymmetric attention (e.g., "how much 'cat' attends to 'sat'"), "here" is an awkward token — it's an adverb without a clear semantic relationship to the others that would make the asymmetry feel meaningful. The worked example shows asymmetric numbers, but the student can't reason about WHY "cat" should attend to "here" differently than "here" attends to "cat" because these tokens don't have a linguistically interesting asymmetric relationship (unlike "cat" and "chased").
**Student impact:** Minor. The student can still see the asymmetry in the numbers. But the lesson would be stronger if the asymmetry felt semantically motivated, not just numerically demonstrated. Since changing the tokens would break continuity with Lesson 1, this is a tradeoff.
**Suggested fix:** No change needed — the continuity benefit outweighs the semantic weakness. The "cat chased mouse" example in the prediction exercise (Section 5) provides the semantically motivated case. Optionally, add a brief note: "These matrices are random, not trained — in a real model, the attention pattern would reflect learned linguistic relationships."

### [POLISH] — Summary block description uses plain text for formulas

**Location:** Section 11 (Summary), SummaryBlock items
**Issue:** The summary descriptions use plain text like "q_i · k_j" and "\u221Ad_k" instead of InlineMath/KaTeX rendering. This is inconsistent with the rest of the lesson, which uses KaTeX throughout. The SummaryBlock component may not support InlineMath in description strings, which would make this a component limitation rather than a lesson issue.
**Student impact:** Very minor visual inconsistency.
**Suggested fix:** If SummaryBlock supports JSX in descriptions, use InlineMath. If it only accepts strings, this is fine as-is.

### Review Notes

**What works well:**
- The narrative arc is excellent. The cliffhanger resolution from Lesson 1 creates genuine momentum. The student is primed to receive Q and K because they already predicted asymmetry would result from separate vectors.
- The worked example with the same 4 tokens from Lesson 1 is a strong pedagogical choice — the "before and after" effect is vivid.
- The scaling section is well-motivated. Connecting to temperature (from what-is-a-language-model) and vanishing gradients (from training-dynamics) is exactly right. The student has both pieces; this lesson connects them.
- The job fair analogy is effective and avoids the "database key/query" trap that the planning document warned against.
- The geometric SVG diagram is a good addition of spatial modality.
- The side-by-side heatmap comparison (raw vs projected) is the strongest visual moment in the lesson.
- The forward reference to V is well-seeded — it presents the same "should one vector serve two roles?" question, which the student has now seen resolved once and can predict the pattern.

**Patterns to watch:**
- The two-notebook-section pattern should be avoided in future lessons. One notebook moment per lesson is cleaner.
- The "assert a capability of the mechanism without demonstrating it" pattern (the "bank"/"steep" claim) should be caught during building. If a claim can't be demonstrated in the lesson, soften it to "the mechanism could learn to..." rather than stating it as fact.

**Modality check:** 6 modalities planned, all 6 present in the built lesson: verbal/analogy (job fair), visual (heatmaps, SVG), symbolic (KaTeX formulas throughout), geometric/spatial (projection SVG), intuitive ("learned lens"), concrete example (4-token worked trace). This exceeds the minimum of 3.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All iteration 1 findings (1 critical, 5 improvements) have been correctly fixed. No new critical or improvement-level issues found. Two minor polish findings remain.

### Iteration 1 Fix Verification

| Finding | Status | Verification |
|---------|--------|-------------|
| CRITICAL: Softmax values incorrect (summed to 0.69) | FIXED | Line 922 now shows `[0.543, 0.073, 0.163, 0.221]`, verified to sum to 1.000. |
| IMPROVEMENT: Misconception #5 (larger d_k always better) not addressed | FIXED | Lines 997-1004 now address this directly: "why not just make d_k as large as possible?" with the parameter tradeoff and multi-head teaser. |
| IMPROVEMENT: Misconception #3 (Q/K in different spaces) too subtle | FIXED | Lines 449-452 now explicitly state in the main prose of Section 4: "This dot product is meaningful because Q and K live in the same d_k-dimensional space — they arrived there via different matrices, but they share the same coordinate system." This is in addition to the SVG caption (line 383). |
| IMPROVEMENT: "Different W_Q/W_K produce different patterns" only in notebook | FIXED | Lines 607-610 now include this in the lesson body: "If we used entirely different W_Q and W_K matrices, the attention pattern would be completely different — same tokens, same embeddings, different behavior." |
| IMPROVEMENT: Two notebook sections creating redundancy | FIXED | Section 7 is now a brief 2-sentence teaser (lines 834-842) pointing forward to the consolidated notebook section (Section 10, lines 1105-1176). Single notebook moment. |
| IMPROVEMENT: "bank"/"steep" callback makes unverifiable claim | FIXED | Line 425 now uses "In a trained model, W_Q and W_K would learn projections that make..." — properly softened from assertion to mechanism description. |

### Findings

### [POLISH] — Awkward possessive after closing quotation marks

**Location:** Section 5 (Check: Predict Asymmetry), lines 481-482
**Issue:** The text reads `"cat"'s attention to "chased"` and `"chased"'s attention to "cat"`. The closing double quote immediately followed by a possessive apostrophe (`"'s`) creates a visually awkward character collision. This is typographically correct but reads poorly.
**Student impact:** Momentary visual stumble. The student can parse it, but it interrupts the reading flow of a prediction exercise where smooth comprehension matters.
**Suggested fix:** Rephrase to avoid the possessive-after-quote pattern. For example: "should the attention from 'cat' to 'chased' equal the attention from 'chased' to 'cat'?" or "should 'cat' attend to 'chased' as much as 'chased' attends to 'cat'?"

### [POLISH] — `&ensp;&mdash;&ensp;` spacing in Q/K formula block

**Location:** Lines 310, 314 (the Q and K formula definitions)
**Issue:** These two lines use `&ensp;&mdash;&ensp;` (thin spaces around the em dash) while every other em dash in the lesson uses `&mdash;` with no spaces. The iteration 1 review noted this as borderline intentional for readability after an InlineMath component.
**Student impact:** None. The visual spacing actually helps readability in the formula block context.
**Suggested fix:** No change needed. This is a deliberate formatting choice for formula-to-description separators. If consistency is desired, a brief comment in the code noting the intentional spacing would prevent future reviewers from flagging it.

### Review Notes

**All iteration 1 fixes were applied correctly.** The lesson is now clean.

**What works well (reinforced observations from iteration 1):**
- The narrative arc from cliffhanger to resolution is the strongest structural element. The student arrives primed from the transfer question and gets the exact mechanism they predicted.
- The worked example with the same 4 tokens creates a vivid before/after comparison. The QK^T values are all verified correct. The side-by-side heatmap comparison is the visual centerpiece.
- The scaling section is well-motivated with correct numerical examples. The temperature callback and vanishing gradient callback bridge existing concepts cleanly.
- The consolidated notebook section (single practice moment after all theory) is cleaner than the original two-section version.
- All 5 misconceptions from the planning document are addressed at appropriate locations.
- All 6 planned modalities are present and genuinely distinct.
- The forward reference to V naturally seeds the next lesson's limitation using the same "one vector serving two roles" pattern the student has now seen resolved once.

**Numerical verification:** All computed values in the lesson (Q vectors, K vectors, QK^T scores, softmax weights, raw XX^T scores, raw softmax weights, dot product magnitudes by dimension, softmax example values) were independently verified and are correct.

**Cognitive load assessment:** 2 new concepts (Q/K projections, sqrt(d_k) scaling), both connected to prior concepts. Well within the 2-3 limit. The lesson is appropriately scoped as a BUILD lesson.

**The lesson is ready to ship.**
