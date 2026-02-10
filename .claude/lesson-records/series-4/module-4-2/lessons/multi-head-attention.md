# Lesson: Multi-Head Attention

**Module:** 4.2 — Attention & the Transformer
**Position:** Lesson 4 of 6
**Slug:** `multi-head-attention`
**Type:** Hands-on (notebook: `4-2-4-multi-head-attention.ipynb`)
**Cognitive load:** STRETCH

---

## Phase 1: Orient — Student State

The student has built single-head attention piece by piece over three lessons and has a strong foundation in the complete input pipeline from Module 4.1.

### Relevant Concepts

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Full single-head attention formula: output = softmax(QK^T / sqrt(d_k)) V | DEVELOPED | values-and-attention-output | Student can trace the entire computation by hand with 4 tokens and 3-dim embeddings. Built incrementally across three lessons. |
| Q, K, V as learned linear projections (three nn.Linear layers applied to the same embedding) | DEVELOPED | queries-and-keys, values-and-attention-output | "Three lenses, one embedding" mental model. Student understands each is nn.Linear(d_model, d_k/d_v, bias=False). |
| QK^T as learned asymmetric relevance (not raw similarity) | DEVELOPED | queries-and-keys | Student saw symmetric XX^T replaced by asymmetric QK^T with concrete worked example. |
| Scaling by sqrt(d_k) to prevent softmax saturation | DEVELOPED | queries-and-keys | Student understands the variance argument: dot product variance = d_k, division normalizes to 1. Connected to temperature slider and vanishing gradients. |
| Residual stream (attention output added to input, not replaced) | INTRODUCED | values-and-attention-output | "Editing, not writing." Callback to ResNet F(x)+x. Student knows the concept but hasn't seen it across stacked layers. |
| Attention weight matrix (square matrix, row i = token i's attention distribution, rows sum to 1) | DEVELOPED | the-problem-attention-solves | Explored interactively via AttentionMatrixWidget. Hand-traced 4x4 matrix. |
| nn.Linear as a learned transformation | DEVELOPED | queries-and-keys | "Learned lens" framing. Connected to nn.Linear from Series 2 Module 2.1. |
| d_k as a dimension choice with tradeoffs | INTRODUCED | queries-and-keys | Misconception #5: "larger d_k is always better." Student saw that models keep d_k small (~64) and use multiple heads. Teaser for this lesson. |
| Token embeddings as d_model-dimensional vectors | DEVELOPED | embeddings-and-position | Student understands nn.Embedding(vocab_size, d_model) and the complete input pipeline. |
| Concatenation as a tensor operation | DEVELOPED | Series 2 (PyTorch Core) | Student has used torch.cat in PyTorch. Familiar operation. |

### Mental Models Already Established
- "Three lenses, one embedding" — W_Q, W_K, W_V extract three views of the same token
- "K gets you noticed; V is what you actually deliver" — job fair resume analogy
- "QK^T computes learned relevance, not raw similarity"
- "Scaling by sqrt(d_k) is not cosmetic — it's the difference between a model that learns and one that doesn't"
- "Attention edits the embedding, it doesn't replace it" — residual stream
- "The input decides what matters" — data-dependent weights as the paradigm shift

### What Was Explicitly NOT Covered
- Multi-head attention (multiple Q/K/V sets in parallel) — this lesson
- Output projection W_O — this lesson
- The transformer block architecture — Lesson 5
- Layer normalization — Lesson 5
- Causal masking — Lesson 6
- Cross-attention — out of scope for this module
- nn.MultiheadAttention as a PyTorch module — deferred

### Readiness Assessment
The student is well-prepared. Single-head attention is at DEVELOPED depth with hand-traced examples. The d_k tradeoff teaser from queries-and-keys explicitly set up this lesson ("models keep d_k small and use multiple heads"). The seed from values-and-attention-output ("one head captures one kind of relationship") provides the motivating limitation. The student has all the pieces; this lesson combines them into parallel operation and teaches the dimension management.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to understand why a single attention head captures only one type of relationship, and how multi-head attention runs multiple smaller heads in parallel and combines their outputs to capture diverse linguistic patterns simultaneously.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Full single-head attention formula | DEVELOPED | DEVELOPED | values-and-attention-output | OK | Student must be able to compute single-head attention to understand running multiple in parallel. Hand-traced worked example confirms depth. |
| Q, K, V as nn.Linear projections | DEVELOPED | DEVELOPED | queries-and-keys, values-and-attention-output | OK | Student must understand that each head gets its own W_Q, W_K, W_V. The "three lenses" model is well established. |
| Scaling by sqrt(d_k) | INTRODUCED | DEVELOPED | queries-and-keys | OK | Student needs to know scaling exists but doesn't need to re-derive it. Actual depth exceeds requirement. |
| Attention weight matrix interpretation | DEVELOPED | DEVELOPED | the-problem-attention-solves | OK | Student must read attention heatmaps to understand what different heads attend to. AttentionMatrixWidget established this. |
| d_k as a dimension choice | INTRODUCED | INTRODUCED | queries-and-keys | OK | Student heard that models keep d_k small (~64) and use multiple heads. This lesson develops that teaser. |
| nn.Linear (learned linear transformation) | DEVELOPED | DEVELOPED | queries-and-keys, Series 2 | OK | W_O is another nn.Linear. Student is comfortable with this building block. |
| Concatenation (torch.cat) | APPLIED | DEVELOPED | Series 2 | OK | Concatenating head outputs is a standard tensor operation. Student has used torch.cat. Slight depth gap but concatenation is a simple operation that doesn't need prior application-level practice. |
| Residual stream concept | INTRODUCED | INTRODUCED | values-and-attention-output | OK | Multi-head output goes into the residual stream. Student has the concept at the right depth; full development deferred to Lesson 5. |
| Matrix shapes and dimension tracking | DEVELOPED | DEVELOPED | Series 2, queries-and-keys | OK | This lesson requires careful dimension reasoning (d_model splits into h heads of d_k each). Student has traced matrix shapes through Q/K/V computations. |

No gaps or missing prerequisites. All concepts are at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "More heads is always better (just add more)" | Intuitive pattern: more = better, like more neurons or more layers. The student hasn't yet encountered the dimension tradeoff. | d_model=768 with 768 heads = d_k=1 per head. A 1-dimensional dot product is a single scalar multiplication — the head can only capture one feature of similarity. Trivially unable to represent complex relationships. At the other extreme, 1 head = d_k=768 captures rich relationships but only one type. The tradeoff is concrete. | Section 4 (Explain) — immediately after introducing dimension splitting. This is the module-level misconception; it must be addressed directly with the negative example. |
| "Each head specializes in a specific linguistic role (head 1 = syntax, head 2 = semantics, etc.)" | Natural extension of "different heads for different relationships." Implies clean, interpretable specialization like departments in a company. | In practice, head specialization is messy and emergent. Some heads in trained models attend to the previous token (positional pattern), some track subject-verb agreement (syntactic), some are nearly uniform (apparently redundant), and many don't map cleanly to any linguistic category. Pruning studies show you can remove 20-40% of heads with minimal performance loss. This is not a department org chart. | Section 7 (Elaborate) — after the student has implemented multi-head attention and can visualize what real heads do. Address with concrete examples from trained models. |
| "Multi-head attention is computationally more expensive than single-head (because you run multiple attention operations)" | Running h separate attention operations sounds like h times the compute. | If d_k = d_model/h, then each head's QK^T is (n x d_k) @ (d_k x n) instead of (n x d_model) @ (d_model x n). Total FLOPs across all heads: h * n^2 * d_k = n^2 * d_model — identical to a single head with d_k = d_model. The computation is split, not multiplied. (The output projection W_O adds some cost, but the attention itself is compute-equivalent.) | Section 4 (Explain) — dimension splitting subsection. Important to address early so the student doesn't resist multi-head on cost grounds. |
| "The output projection W_O is just reshaping (gluing heads back together)" | Concatenation already combines the heads, so W_O might seem like a formatting step. | W_O is a learned d_model x d_model matrix with d_model^2 parameters. It mixes information across heads — head 3's output can influence the final representation in dimensions that head 1 wrote to. Without W_O, each head's contribution is isolated to its d_k-dimensional slice of the output. With W_O, the model can learn to combine signals from different heads. It's a learned mixing layer, not a reshape. | Section 4 (Explain) — output projection subsection. Concrete contrast: without W_O, output is just concatenated slices (no cross-head interaction). With W_O, the model learns how to blend. |
| "Heads share weights and just look at different parts of the input" | CNN analogy: multiple filters share the convolution operation but have different learned weights. Student might think heads differ only in which tokens they attend to, not in their projection weights. | Each head has its OWN W_Q^(i), W_K^(i), W_V^(i) — completely separate learned parameters. Two heads processing the same input can produce entirely different Q, K, V vectors and entirely different attention patterns. The independence is in the weights, not just the attention distribution. | Section 4 (Explain) — when introducing "each head has its own projections." Brief but explicit. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| "The cat sat on the mat because it was soft" — dual-relationship problem | Positive (motivating) | Shows that "it" needs to attend to "mat" for coreference AND to "soft" for meaning — two different relationships a single head can't capture simultaneously. | Directly continues the seed from values-and-attention-output. Same sentence, familiar context. The two relationships are linguistically distinct (coreference vs. adjective modification) and require different Q/K projections to capture. |
| Two-head worked example with d_model=6, h=2, d_k=3 | Positive (core) | Traces multi-head attention end-to-end with concrete numbers. Head 1 and Head 2 get different W_Q, W_K, W_V matrices, produce different attention patterns, get concatenated, and pass through W_O. | Continues the 4-token running example from Lessons 1-3 but increases d_model from 3 to 6 to allow splitting into 2 heads of d_k=3. Keeps numbers hand-traceable while demonstrating the splitting/concatenation mechanics. Shows that different heads produce genuinely different attention weight distributions. |
| d_model=768 with extreme head counts (1 head vs. 768 heads) | Negative | Demonstrates the tradeoff: d_k=1 per head is degenerate (single scalar dot product, can't capture nuanced relationships); 1 head captures rich relationships but only one type. Neither extreme is good. | Concretizes the module-level misconception "more heads = better" with specific, graspable numbers. The d_k=1 case is absurd enough to be memorable. Connected to GPT-2's actual choice of 12 heads with d_k=64. |
| Real GPT-2 head attention patterns (visualization) | Positive (stretch) | Shows what trained heads actually learn — positional patterns, syntactic tracking, near-uniform heads. Grounds the theory in real model behavior. | Bridges from toy examples to reality. Student sees that specialization is messy and emergent, not clean and designed. Addresses misconception #2. Can be explored in the notebook with pretrained weights. |

### Gap Resolution

No gaps identified. All prerequisites are at sufficient depth.

---

## Phase 3: Design

### Narrative Arc

Single-head attention is complete, and the student can trace the entire formula by hand. But there's a problem they already felt at the end of the last lesson: one attention head computes one set of weights, capturing one notion of "what's relevant to what." In the sentence "The cat sat on the mat because it was soft," the pronoun "it" needs to attend to "mat" for coreference (what does "it" refer to?) and to "soft" for meaning (what property is being described?). These require fundamentally different Q/K projections — one set of W_Q, W_K weights can't produce high attention scores for both relationships simultaneously. The solution is not to make a single head bigger or smarter. It's to run multiple smaller heads in parallel, each free to learn its own notion of relevance, then combine their findings. This is multi-head attention: the same elegant move of "split the problem into parallel subproblems" that appears throughout deep learning. The twist is that splitting comes free — the total computation is the same as a single large head, just partitioned differently.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Verbal/Analogy | "Research team" analogy: one researcher reads the whole paper but can only track one thread; a team of specialists each reads through their own lens, then they pool findings. No single specialist sees everything, but the team covers more ground. | Multi-head attention is inherently about parallelism with specialization. The research team grounds the abstract "multiple subspaces" in a concrete collaborative scenario the student can visualize. The tradeoff (each specialist has a narrower view) maps directly to the d_k reduction. |
| Visual/Diagram | Side-by-side attention heatmaps for 2+ heads on the same sentence, showing different attention patterns. Plus an architectural diagram: input -> split into h streams -> each stream has its own Q/K/V + attention -> concatenate -> W_O -> output. | The visual of different heads attending to different things is the most immediate way to grasp WHY multi-head matters. The architectural diagram provides the structural overview that holds the pieces together. |
| Symbolic/Formula | MultiHead(X) = Concat(head_1, ..., head_h) W_O, where head_i = Attention(X W_Q^i, X W_K^i, X W_V^i). Shape annotations: X is (n, d_model), each head projects to (n, d_k) where d_k = d_model/h, concat produces (n, d_model), W_O is (d_model, d_model). | The formula is the concise statement of the mechanism. Shape annotations are essential because dimension management is the main source of confusion in multi-head attention. Without them, the formula is opaque. |
| Concrete example | Full worked example: 4 tokens, d_model=6, h=2, d_k=3. Two heads with different small-integer projection matrices. All computations traced. Different attention weight patterns emerge from different heads. Concatenation and W_O projection shown with actual numbers. | Continuing the hand-traced running example from Lessons 1-3 (same 4 tokens, upgrading from d_model=3 to d_model=6) maintains continuity and makes the extension to multi-head concrete rather than abstract. Specific numbers prove the heads really do attend differently. |
| Geometric/Spatial | d_model-dimensional space partitioned into h lower-dimensional subspaces. Visualized as a single wide rectangle (d_model=768) split into 12 colored strips (d_k=64 each). Each head operates in its own strip. Concatenation reassembles the full-width rectangle. W_O mixes across strips. | Dimension splitting is the concept that causes the most confusion. A geometric representation of "the same total space, partitioned differently" makes the "split, not multiply" insight tangible. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3
  1. Multiple heads operating in parallel on lower-dimensional subspaces (the core concept)
  2. Dimension splitting: d_k = d_model / h as a budget allocation (the tradeoff)
  3. Output projection W_O as learned cross-head mixing (the combination mechanism)
- **Previous lesson load:** BUILD (values-and-attention-output)
- **This lesson's load:** STRETCH — appropriate. The student had two BUILD lessons in a row (queries-and-keys, values-and-attention-output). STRETCH is warranted because the parallel-operation and dimension-management reasoning is conceptually demanding even though each individual piece (single-head attention, nn.Linear, concatenation) is familiar. The module plan designates this as STRETCH.
- **Assessment:** 3 new concepts is at the limit but acceptable because all three are tightly connected (they're aspects of the same mechanism, not independent topics). The building blocks (single-head attention, nn.Linear, torch.cat) are all well established, so the novelty is in how they compose, not in the components themselves.

### Connections to Prior Concepts

| Existing Concept | Connection | How |
|-----------------|------------|-----|
| Single-head attention formula | Extension | Multi-head IS single-head, just run h times in parallel with smaller dimensions. The formula inside each head is identical. |
| "Three lenses, one embedding" | Extension | Now it's "three lenses per head, h heads." Each head has its own set of lenses (W_Q^i, W_K^i, W_V^i). |
| d_k tradeoff (from queries-and-keys) | Development | The teaser "models keep d_k small (~64) and use multiple heads" is now fully developed. d_k = d_model / h is the specific mechanism. |
| Job fair analogy | Extension | Each head is a different job fair with different job-seeker cards and offering cards. One fair matches on technical skills, another on location, another on salary expectations. The hiring company (W_O) reviews all the resumes (V) from all the fairs and makes a combined decision. |
| nn.Linear as learned transformation | Reuse | W_O is another nn.Linear. Same building block. Nothing new in the operation, just a new purpose (cross-head mixing). |
| CNN parallel filters | Callback | CNNs use multiple filters to detect different features (edges, textures, patterns). Multi-head attention uses multiple heads to capture different relationships. Same principle: parallel specialization. |
| Residual stream | Forward reference | Multi-head output (after W_O) gets added to the input via the residual stream. Same mechanism as Lesson 3, now applied to the combined multi-head output rather than single-head output. |

### Analogies That Could Be Misleading

- **"Each head has a specific job"** — This implies clean, designed specialization (head 1 = syntax, head 2 = coreference). In reality, specialization is messy and emergent. The analogy is useful for motivation but must be softened during the Elaborate section with real head visualization data.
- **"Splitting d_model is like splitting a pizza"** — This could imply that the pieces don't interact. The W_O projection is specifically designed to mix across heads. The "split" is only for the attention computation; the output projection recombines.

### Scope Boundaries

**This lesson IS about:**
- Why one head isn't enough (different relationship types in one sentence)
- Each head has its own independent W_Q, W_K, W_V (separate learned parameters)
- Dimension splitting: d_k = d_model / h (budget allocation, not computation multiplication)
- Concatenation of head outputs into a single (n, d_model) tensor
- Output projection W_O as learned cross-head mixing
- The multi-head attention formula: MultiHead(X) = Concat(head_1, ..., head_h) W_O
- Visualizing what different heads attend to in practice (messy, emergent specialization)
- Total compute equivalence between multi-head and single-head
- Target depth: DEVELOPED for multi-head mechanism, INTRODUCED for head specialization patterns

**This lesson is NOT about:**
- The transformer block architecture (attention + FFN + residual + layer norm) — Lesson 5
- Causal masking — Lesson 6
- Cross-attention (encoder-decoder attention) — out of scope for this module
- nn.MultiheadAttention as a PyTorch module — student builds from scratch in notebook
- How heads are trained or how specialization emerges during training — Module 4.3
- Attention head pruning or efficiency optimizations — Module 4.3 (scaling lesson)
- The residual stream across stacked layers — Lesson 5

### Lesson Outline

1. **Context + Constraints**
   - "You have complete single-head attention. This lesson extends it to multi-head. By the end, you'll understand why models use 12, 16, or 32 heads instead of one, how the dimensions work, and what different heads actually learn."
   - Scope boundaries stated: not the transformer block, not causal masking, not how specialization emerges during training.

2. **Recap** (brief — prerequisites are solid)
   - One-paragraph refresher of the single-head formula with the three-lesson evolution displayed.
   - Emphasis on "one set of W_Q, W_K, W_V produces one attention pattern."

3. **Hook** (type: limitation reveal + prediction exercise)
   - Callback to the seed from values-and-attention-output: "Single-head attention computes one notion of relevance."
   - "The cat sat on the mat because it was soft" — "it" needs to attend to "mat" for coreference AND to "soft" for meaning.
   - Prediction exercise: Can one Q vector for "it" produce high attention on both "mat" and "soft" simultaneously? The student should reason: Q dot K_mat and Q dot K_soft both need to be large, but "mat" and "soft" have very different semantics, so their K vectors point in different directions. One Q can't be simultaneously aligned with both.
   - Reveal: this isn't a scaling problem. Making d_k bigger doesn't help — a single dot product computes a single scalar score per pair. The issue is that one Q/K projection extracts one notion of relevance.

4. **Explain** — Core concept with planned modalities

   4a. **The solution: multiple heads**
   - Research team analogy: instead of one researcher tracking everything, assign a team of specialists. Each reads through their own lens. Pool the findings at the end.
   - Each head i has its own W_Q^i, W_K^i, W_V^i — completely independent learned parameters.
   - Side-by-side attention heatmaps (visual): two heads on "The cat sat on the mat because it was soft." Head 1 might concentrate "it" -> "mat" (coreference). Head 2 might concentrate "it" -> "soft" (property attribution). Different weights because different projection matrices.
   - Explicit: heads share the input X but nothing else. Different matrices produce different Q, K, V, producing different attention patterns.

   4b. **Dimension splitting: the key insight**
   - The question: "If we run h separate attention heads, do we need h times the parameters?" Answer: no. We split the existing d_model budget.
   - d_k = d_model / h. For GPT-2: d_model=768, h=12 -> d_k=64. Each head's W_Q is (d_model, 64) instead of (d_model, 768).
   - Geometric visualization: wide rectangle (768) split into 12 colored strips (64 each). Each head operates in its strip.
   - Compute equivalence: h heads, each doing (n, d_k) operations = same total as one head doing (n, d_model). Split, not multiplied.
   - Address misconception #1 (more heads = better): d_model=768 with 768 heads = d_k=1. A 1-dimensional dot product is a single scalar multiplication. Absurdly limited. With 1 head: d_k=768, rich relationships but only one type. The tradeoff is real.
   - GPT-2 uses 12 heads (d_k=64). GPT-3 uses 96 heads (d_k=128 with d_model=12288). These are design choices, not "more is better."
   - Address misconception #3 (multi-head is more expensive): same total FLOPs, shown explicitly.

   4c. **Concatenation + output projection W_O**
   - Each head produces output of shape (n, d_k). Concatenate h of them: (n, h * d_k) = (n, d_model).
   - W_O: a (d_model, d_model) matrix applied after concatenation. Another nn.Linear.
   - Address misconception #4: W_O is NOT just reshaping. It's a learned mixing layer. Without W_O, each head's contribution stays in its d_k slice of the output — isolated. With W_O, head 3's findings can influence any dimension of the final output. W_O lets the model learn how to combine signals from different heads.
   - The full formula: MultiHead(X) = Concat(head_1, ..., head_h) W_O. Shape annotations on each piece.

5. **Check** — Prediction exercise
   - "You have d_model=512 and h=8. What is d_k? How many parameters does W_O have? If you add a 9th head, what changes?"
   - d_k = 64. W_O has 512 x 512 = 262,144 parameters. Adding a 9th head: either d_k drops to 512/9 (not a clean split — this is why h typically divides d_model evenly) or d_model increases. The constraint is d_model = h * d_k.

6. **Explore** — Worked example
   - Full hand-traced multi-head attention with 4 tokens, d_model=6, h=2, d_k=3.
   - Same 4 tokens ("The", "cat", "sat", "here") from Lessons 1-3, but now with 6-dim embeddings (extended from 3-dim).
   - Head 1: W_Q^1, W_K^1, W_V^1 are each (6, 3). Compute Q^1, K^1, V^1. Compute attention weights. Compute output^1: shape (4, 3).
   - Head 2: W_Q^2, W_K^2, W_V^2 are each (6, 3). Same computation, different matrices, different attention patterns.
   - Show side-by-side: Head 1's attention weights vs. Head 2's attention weights. Highlight that they are genuinely different (different rows highlight different tokens).
   - Concatenate: output = [output^1 | output^2], shape (4, 6).
   - Apply W_O (6 x 6): final output shape (4, 6). Same shape as input — ready for the residual stream.
   - Add residual: final = multi_head_output + input_embedding.

7. **Elaborate** — What heads actually learn
   - Address misconception #2: "each head has a specific linguistic role."
   - Real findings from trained models (Voita et al., Clark et al.):
     - Some heads attend to the previous token (positional, not semantic)
     - Some track syntactic relationships (subject-verb agreement)
     - Some are nearly uniform (apparently redundant — pruning studies show 20-40% of heads can be removed with minimal loss)
     - Most heads don't map cleanly to any single linguistic function
   - Key insight: specialization is emergent and messy, not designed and clean. The model discovers what's useful through training.
   - InsightBlock: "Multi-head attention gives the model the capacity to capture multiple relationship types. What it actually learns to capture is determined by the training data and objective, not by the architecture."
   - Callback to CNN filters: early layers learn edges and textures (somewhat interpretable), deeper layers learn abstract features (less interpretable). Same phenomenon: architecture provides capacity, training determines specialization.

8. **Check** — Transfer question
   - "A colleague builds a model with d_model=256 and h=1 (single head, d_k=256). Another colleague builds d_model=256 and h=4 (d_k=64). Which model has more parameters in the attention projections? Which captures more diverse relationships?"
   - Answer: The 4-head model has slightly more parameters (adds W_O: 256x256 = 65K extra params). But each head's Q/K/V params total the same (3 * d_model * d_k * h = 3 * 256 * 64 * 4 = 3 * 256 * 256 = same as single head). The 4-head model captures more diverse relationships because 4 independent attention patterns, but each in a 64-dim subspace instead of 256. The tradeoff is diversity vs. per-head expressiveness.

9. **Practice** — Notebook exercises (scaffolded, moving toward independent)
   - Exercise 1 (guided): Implement a single attention head as a function: `single_head_attention(X, W_Q, W_K, W_V)`. Verify it matches the Lesson 3 formula. This is recap but in code.
   - Exercise 2 (supported): Implement multi-head attention from scratch: split into h heads, run attention on each, concatenate, apply W_O. Use d_model=6, h=2 on the 4-token example. Verify output shape.
   - Exercise 3 (supported): Verify compute equivalence: time single-head (d_k=d_model) vs. multi-head (h heads, d_k=d_model/h) on a larger example (e.g., 32 tokens, d_model=64). Compare FLOPs and wall-clock time.
   - Exercise 4 (independent): Implement multi-head attention as a PyTorch nn.Module with proper nn.Linear layers and nn.Parameter. Forward pass should handle batched input.
   - Exercise 5 (stretch): Load GPT-2 pretrained weights. Extract attention weights for all 12 heads on a sample sentence. Visualize the 12 attention heatmaps side by side. Identify which heads show positional patterns vs. semantic patterns. Compare with the toy example.

10. **Summarize** — Key takeaways
    - Multi-head attention runs h independent attention functions in parallel, each in a d_k = d_model/h dimensional subspace.
    - Each head has its own W_Q, W_K, W_V — different learned projections capture different types of relationships.
    - Concatenation reassembles the outputs; W_O mixes information across heads.
    - Total computation is equivalent to single-head — the budget is split, not multiplied.
    - Head specialization is emergent and messy, not designed and clean.
    - Echo mental model: "Multiple lenses, pooled findings" — each head looks through its own set of three lenses, and W_O synthesizes what they all saw.

11. **Next step** — What comes after
    - "Multi-head attention is the core computational unit. But it doesn't stand alone. In the full transformer, multi-head attention feeds into a feed-forward network, with residual connections and layer normalization around each. The next lesson assembles these pieces into the transformer block — the repeating unit that stacks to form GPT."
    - Seed: "Attention reads from the residual stream; the FFN writes new information into it. Different roles, same stream."

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 2
- Improvement: 4
- Polish: 2

### Verdict: MAJOR REVISION

Two critical findings require fixes before this lesson is usable. The worked example section is too dense and lacks the guided interpretation that would help the student extract meaning from the numbers. The CNN callback in the "What Heads Actually Learn" section references concepts the student may not recall at the depth assumed, creating an ungrounded analogy. The improvement findings target weak areas where the lesson functions but significantly underperforms what it could be.

### Findings

#### [CRITICAL] — Worked example lacks guided interpretation of the two attention weight matrices

**Location:** Section 9 ("Worked Example: Two Heads in Action") — Head 1 and Head 2 weight tables
**Issue:** The lesson displays two 4x4 attention weight tables side by side and says "Compare the two weight matrices. Different projection matrices produce genuinely different attention patterns." But it never tells the student WHAT to compare or WHAT the differences mean. The tables are raw numbers with no narrative walkthrough. In Lessons 1-3, every worked example had a step-by-step trace with explicit interpretation ("cat attends most strongly to sat with weight 0.38, which means..."). Here, the student is handed two tables of 16 numbers each and told "they're different." The aside repeats the same point ("Both heads see the same 4 tokens, but their attention patterns differ") without adding any interpretive content.

**Student impact:** The student would look at two matrices of numbers, maybe notice some are highlighted green, and think "okay, they're different." But they wouldn't know WHY they're different in the specific way they are, or what kind of relationship each head is capturing. The worked example is supposed to be the concrete grounding for multi-head attention — the moment where the abstract concept becomes tangible. Without guided interpretation, it becomes an exercise in number-staring. The student leaves without the "aha" that different heads genuinely attend to different things for different reasons.

**Suggested fix:** After each head's weight table, add 2-3 sentences interpreting the most notable pattern. For example: "Head 1 assigns the highest weight for 'cat' to 'sat' (0.XX) — this head seems to be capturing verb-subject proximity. Head 2, using different projections, has 'cat' attending most strongly to 'here' (0.XX) — a different relationship entirely." Then after both tables, a brief comparison paragraph: "Notice that 'sat' attends to different tokens in each head. Head 1: ... Head 2: ... This is multi-head attention at work: the same token, two different views of what matters." The numbers are already computed; the lesson just needs to READ them for the student.

---

#### [CRITICAL] — CNN filters callback in "What Heads Actually Learn" assumes recall depth the records don't support

**Location:** Section 10 ("What Heads Actually Learn"), final paragraph
**Issue:** The lesson states: "This is the same phenomenon you saw with CNN filters: early-layer filters learn interpretable features (edges, textures), but deeper-layer filters become increasingly abstract." This callback assumes the student recalls that CNN filters exhibit depth-dependent interpretability (early = edges, deep = abstract). The module record for Series 3 Module 3.1 shows that convolution filters and locality were INTRODUCED in "what-convolutions-compute," but there is no record entry for "depth-dependent interpretability of filters" or "early layers = edges, deeper layers = abstract." The student may have encountered this concept informally, but the lesson records do not confirm it was explicitly taught at any depth. Additionally, the planning document listed this as a "callback to CNN filters" but the specific claim about early-vs-deep layer interpretability goes beyond what the records show was taught.

**Student impact:** If the student doesn't recall the early-vs-deep interpretability pattern for CNNs, this sentence feels like an unsupported assertion rather than a connecting callback. The analogy breaks because the anchor point is missing. The student might nod along without genuinely connecting the ideas, or worse, might feel confused about a CNN concept they were supposed to know.

**Suggested fix:** Either (a) soften the callback to something the records confirm was taught: "This is similar to what you saw with convolution — the architecture provides the capacity (filters, heads), and training determines what each one learns. The difference between a learned pattern and a designed pattern." Or (b) add a brief 1-sentence reminder: "In CNNs, early-layer filters tend to learn edges and textures while deeper filters learn abstract features — a pattern researchers discovered by visualizing trained filters. Attention heads show a similar phenomenon: the architecture provides slots, and training fills them in ways that are sometimes interpretable, sometimes not."

---

#### [IMPROVEMENT] — Concatenation + W_O section jumps to formula before concrete motivation

**Location:** Section 7 ("Combining Heads: Concatenation + W_O")
**Issue:** The section starts by stating the shapes ("Each head produces an output of shape (n, d_k). Concatenate all h of them to recover the original width"), immediately gives the formula, and then explains WHY concatenation alone isn't enough. This violates the "problem before solution" ordering rule. The student learns WHAT W_O does before understanding WHY it's needed. The problem (cross-head isolation) should come first, then W_O as the solution.

**Student impact:** The student encounters W_O as a mechanical step in the pipeline rather than as a solution to a problem they feel. The lesson DOES explain the problem (heads are isolated in their d_k slices), but the explanation comes AFTER the formula. The student reads the formula first, then learns why it matters — a weaker pedagogical sequence than problem-then-solution.

**Suggested fix:** Reorder the section: (1) State what concatenation does (shape recovery). (2) Immediately state the problem: "But concatenation keeps each head's contribution isolated to its own d_k-dimensional slice. Head 1's findings stay in dimensions 1-64, Head 2's in 65-128. There's no cross-head communication." (3) THEN introduce W_O as the solution: "The output projection W_O fixes this." The misconception block about W_O can stay where it is.

---

#### [IMPROVEMENT] — The "it" problem hook uses fabricated attention weights without grounding

**Location:** Section 4 ("One Head, Two Relationships") — side-by-side attention heatmaps
**Issue:** The hook presents two attention heatmaps showing Head 1 (coreference: "it" -> "mat" at 0.61) and Head 2 (property attribution: "it" -> "soft" at 0.52). These are fabricated numbers presented as if they come from a real or computed model. The student might wonder: where do these numbers come from? Are these from a trained model? From the hand-traced example? They appear before the student has learned multi-head attention, so they can't be from the lesson's worked example. This creates an awkward gap between the illustrative numbers and anything the student can verify.

**Student impact:** Minor confusion about the provenance of the numbers. The student might try to reconcile them with the later worked example (which uses "The cat sat here," not "The cat sat on the mat because it was soft") and find they don't connect. The illustration is pedagogically useful as motivation, but the fabricated precision (0.61, 0.52) could trigger the student's "where did this come from?" instinct.

**Suggested fix:** Add a brief qualifier like "In a trained model, the attention weights might look something like this:" or "Hypothetical weights from a trained model:" to signal that these are illustrative, not computed. Alternatively, round the numbers to simpler values (0.6, 0.5) to signal they are approximate/illustrative rather than exact.

---

#### [IMPROVEMENT] — Head 1 and Head 2 weight tables lack row-by-row narrative for even one token

**Location:** Sections following the worked example heading (Head 1 and Head 2 weight tables)
**Issue:** This is related to the first critical finding but distinct: even setting aside the cross-head comparison, neither individual head's table has ANY row-level interpretation. In Lessons 1-3, when the student saw an attention weight table, the lesson traced through at least one row in detail ("For 'cat': highest attention to 'sat' at 0.38, then 'The' at 0.27..."). Here, neither table has a single row trace. The tables are just displayed.

**Student impact:** The student has been trained by three prior lessons to expect row-level walkthroughs of attention weights. The sudden absence of narrative interpretation makes this section feel like a data dump rather than a learning experience. The student may skim the tables without engaging, missing the core point that different heads produce different patterns.

**Suggested fix:** Pick one token (e.g., "cat") and trace its attention weights in Head 1 and Head 2. Show that "cat" attends most strongly to different tokens in each head. This takes 3-4 sentences and connects the abstract tables to a concrete per-token story.

---

#### [IMPROVEMENT] — Misconception #5 (heads share weights) is addressed only in an aside, not in the main content

**Location:** Section 5 ("The Solution: Multiple Heads") — aside WarningBlock titled "Heads Share Input, Nothing Else"
**Issue:** The planning document identified misconception #5 ("Heads share weights and just look at different parts of the input") and specified it should be addressed "when introducing 'each head has its own projections' — brief but explicit." The lesson does address it in a WarningBlock aside, but not in the main content flow. The main content says "each with its own learned projection matrices" which is correct but doesn't explicitly confront the misconception. The aside is the only place that says "completely independent learned weights." A student who skips asides (common with ADHD) would miss the explicit misconception correction entirely.

**Student impact:** A student who reads only the main content might still carry the misconception that heads share some weights and only differ in what parts of the input they process (similar to how CNN filters share the convolution operation). The aside corrects this, but asides are supplementary by design — critical misconception corrections belong in the main flow.

**Suggested fix:** Add one sentence to the main content after "each with its own learned projection matrices": "These are completely independent sets of parameters — Head 1's W_Q has no connection to Head 2's W_Q. They start as different random matrices and learn to specialize in different ways."

---

#### [POLISH] — Residual section in worked example appears without motivation

**Location:** Section 9 — "Add residual (MHA output + original embedding)" subsection
**Issue:** The residual addition appears at the end of the worked example as a mechanical step without reminding the student WHY it's there. Lesson 3 introduced the residual stream concept with a full motivation ("token in uninformative context would lose identity"), but that was two lessons ago. A one-sentence reminder would help: "Just like single-head attention, multi-head output is added to the original embedding — preserving the token's identity when attention doesn't find useful context."

**Student impact:** Minor. The student has seen the residual concept before and will likely remember it. But a brief reminder connects the worked example to the conceptual framework and reinforces a concept that was only INTRODUCED (not yet DEVELOPED).

**Suggested fix:** Add one sentence before the residual computation: "Same principle from Lesson 3: add, don't replace. The token keeps its original meaning while attention enriches it."

---

#### [POLISH] — Hook prediction exercise answer is dense and could benefit from whitespace

**Location:** Section 4 — prediction exercise `<details>` reveal
**Issue:** The reveal contains two dense paragraphs of reasoning. The first explains why one Q vector can't align with both K vectors. The second adds a nuance about d_k not helping. Both are good content, but the formatting is wall-of-text within a small GradientCard. The second paragraph (about d_k) is in muted color, which helps, but the first paragraph could benefit from breaking after the key sentence about K vectors pointing in different directions.

**Student impact:** Minor readability issue. The content is correct but slightly harder to parse than it needs to be.

**Suggested fix:** Break the first paragraph after "One Q vector can't be simultaneously aligned with both." into a separate statement, or add a visual separator before the d_k nuance.

---

### Review Notes

**What works well:**
- The motivation is excellent. The "it" problem hook is compelling and directly continues the seed planted in Lesson 3. The student feels the limitation before getting the solution.
- Dimension splitting section is strong. The SVG visualization of d_model=768 split into 12 colored strips is effective. The compute equivalence derivation is clear and well-paced. The "more heads = better" misconception is well addressed with the d_k=1 extreme.
- The W_O explanation is conceptually solid. The misconception correction ("W_O is NOT just reshaping") is well-placed and clear.
- The summary and mental model sections are strong. "Multiple lenses, pooled findings" is a good extension of the established "three lenses" model.
- The scope boundaries are well respected. The lesson stays focused on multi-head attention and doesn't creep into transformer block territory.
- The notebook exercise scaffolding is well-structured (guided -> supported -> independent -> stretch).
- The seed for Lesson 5 is natural and specific ("Attention reads from the residual stream; the FFN writes new information into it").

**Systemic pattern:**
The two critical findings share a root cause: the worked example section prioritizes showing the computation over interpreting it. In Lessons 1-3, each worked example had a narrative thread ("let's trace what happens to 'cat'..."). This lesson computes everything correctly but doesn't narrate the results. The fix is to add interpretation, not to change the computation.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All critical and improvement findings from Iteration 1 were fixed correctly. The two remaining findings are polish-level issues that do not affect the lesson's pedagogical effectiveness.

### Iteration 1 Fix Verification

1. **CRITICAL — No guided interpretation of attention weight matrices:** FIXED. Head 1 now has a narrative trace for "here" -> "cat" (0.586) and "cat" -> "sat" (0.441). Head 2 has a narrative trace for "here" -> itself (0.628) and "sat" -> itself (0.322). A cross-head comparison paragraph explicitly contrasts "here"'s attention in both heads. All claimed values verified correct against the computed weights.

2. **CRITICAL — CNN callback assumes unrecorded depth:** FIXED. The callback is now softened to "the architecture provides capacity (filters, heads), and training determines what each one learns. The difference between a learned pattern and a designed pattern applies in both cases." This no longer assumes the student recalls depth-dependent interpretability of CNN filters.

3. **IMPROVEMENT — W_O section ordering (problem before solution):** FIXED. Section 7 now flows: concatenation recovers shape -> but each head's contribution stays isolated -> W_O fixes this -> formula. Correct problem-before-solution ordering.

4. **IMPROVEMENT — Fabricated attention weights without grounding:** FIXED. The hypothetical weights now carry the qualifier "Hypothetical weights from a trained model on..." which signals they are illustrative, not computed.

5. **IMPROVEMENT — No row-by-row narrative:** FIXED. "here" is traced through both heads with specific weight values and interpretive labels ("cross-token relationship" vs. "self-reinforcing pattern").

6. **IMPROVEMENT — Shared weights misconception only in aside:** FIXED. Main content now includes: "These are completely independent sets of parameters — Head 1's W_Q has no connection to Head 2's W_Q. They start as different random matrices and learn to specialize in different ways." The misconception correction is in the main flow, not just the aside.

### Findings

#### [POLISH] — Residual section in worked example still lacks motivating reminder

**Location:** Section 9 — "Add residual (MHA output + original embedding)" subsection (line ~1153)
**Issue:** This was identified as Polish #1 in Iteration 1 and was not addressed. The residual addition appears at the end of the worked example as a mechanical computation with only a shape note ("Same shape as input — (4, 6). Ready for the residual stream, just like single-head attention"). There is no brief reminder of WHY the residual exists. The residual concept was INTRODUCED one lesson ago and a one-sentence reminder would reinforce it.
**Student impact:** Minor. The student will likely remember the concept from one lesson ago. But a brief callback strengthens the connection.
**Suggested fix:** Add one sentence before the residual computation table: "Same principle from Lesson 3: add, don't replace. The token keeps its original meaning while multi-head attention enriches it with context."

---

#### [POLISH] — Head 2 "sat" self-attention claim has thin margin

**Location:** Section 9 — Head 2 interpretation paragraph (line ~1072)
**Issue:** The lesson says "'sat' also attends most to itself (0.322)." While technically correct, the margin is thin: "The" gets 0.291, "here" gets 0.280, "sat" (self) gets 0.322. The near-uniform distribution on this row makes the "attends most to itself" claim less convincing than "here" attending to itself at 0.628. A student who examines the table closely might question why a 0.031 margin counts as a clear pattern.
**Student impact:** Negligible. The student's attention is drawn to "here" (the primary example), and "sat" is a secondary observation. Most students won't scrutinize the exact margins.
**Suggested fix:** Optionally soften to "'sat' also leans toward itself (0.322), though the distribution is more spread out." Or simply remove the "sat" observation and keep only the stronger "here" example which has a clear 0.628 vs. 0.173 contrast.

---

### Review Notes

**What works well after fixes:**
- The guided interpretation of the worked example is a significant improvement. The cross-head comparison paragraph for "here" is the standout moment: "Same token, same input, but the two heads extract completely different views of what matters." This is exactly the narrative thread that was missing in Iteration 1.
- Moving the shared-weights misconception correction into the main content (not just the aside) ensures ADHD-friendly students who skip asides still get the key correction.
- The W_O section reordering follows problem-before-solution cleanly. The student now feels the isolation problem before learning W_O as the solution.
- The CNN callback softening is well-calibrated: it preserves the analogy's value (architecture provides capacity, training fills it) without assuming unrecorded depth on filter interpretability.
- All numerical claims verified against the actual computation. The lesson's computed values are correct and consistent.

**Overall assessment:**
This lesson is ready to ship. The two remaining polish items are genuinely minor and won't affect the student's learning experience. The lesson effectively teaches multi-head attention with strong motivation, clear worked examples with guided interpretation, well-addressed misconceptions, and appropriate scope boundaries. The narrative arc from "one head can't capture two relationships" through "split the budget into parallel heads" to "W_O synthesizes their findings" is coherent and well-paced.

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 3 new concepts (at the limit, acceptable given tight coupling)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated
