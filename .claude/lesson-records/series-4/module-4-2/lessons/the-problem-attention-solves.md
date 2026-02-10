# Lesson: The Problem Attention Solves

## Phase 1: Orient — Student State

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Token embeddings as learned lookup (nn.Embedding maps integer to dense vector) | DEVELOPED | embeddings-and-position | Core mechanism understood. Student verified embedding.weight[i] == embedding(tensor([i])). Knows parameter count (50K x 768 = 38.4M). |
| Embedding space clustering (similar tokens have nearby vectors after training) | DEVELOPED | embeddings-and-position | Interactive widget explored. Understands that training moves similar tokens together in embedding space. |
| Polysemy limitation of static embeddings ("bank" gets one vector regardless of context) | INTRODUCED | embeddings-and-position | Student was warned about this. Amber box mentioned that attention (Module 4.2) resolves it. Has recognition but no mechanism yet. |
| Bag-of-words problem (embeddings without position lose sequence order) | DEVELOPED | embeddings-and-position | "Dog bites man" vs "Man bites dog" example. GradientCard showing identical sets. Student understands the problem viscerally. |
| Positional encoding (sinusoidal and learned) | DEVELOPED | embeddings-and-position | Student implemented sinusoidal PE from formula. Understands addition to embeddings. Clock analogy. |
| Complete input pipeline: text -> tokens -> IDs -> embeddings + PE -> tensor | DEVELOPED | embeddings-and-position | Full pipeline traced. Student knows what goes INTO the model. Does not yet know what the model DOES with it. |
| Softmax as probability distribution | DEVELOPED | what-is-a-language-model, Series 1 | Used repeatedly: MNIST classification, temperature slider, LM output over vocabulary. Student is comfortable with softmax mechanics. |
| Dot product as similarity measure | INTRODUCED | embeddings-and-position | Cosine similarity used in notebook to compare embeddings. Dot product is related but the student hasn't formalized it as a "similarity score" between arbitrary vectors. |
| Matrix multiplication | APPLIED | Series 1-2 | Used throughout training loops, weight updates, forward passes. Student is fluent with matmul. |
| Residual connections (skip connections in ResNets) | DEVELOPED | Series 3 (modern-architectures) | Understood as "add input to output so gradients can flow." Will be reused for the residual stream concept. |

**Mental models and analogies already established:**
- "Without position, embeddings are a bag of words" — directly relevant, will be extended
- "Embeddings are a learned dictionary" — each token has one definition; this lesson shows why one definition isn't enough
- "A language model approximates P(next token | context)" — the task attention serves
- CNN spatial locality contrast — position implicit in CNNs, explicit in transformers. Attention is the mechanism that actually uses position information to create context-dependent representations.

**What was explicitly NOT covered:**
- How the model processes embeddings after they're created (the entire forward pass through a transformer)
- Any form of attention mechanism
- Q, K, V projections (not even mentioned by name)
- How context-dependent representations are created (polysemy warning pointed here but gave no mechanism)

**Readiness assessment:** The student is well-prepared. They understand the input pipeline completely, have felt the polysemy limitation, understand the bag-of-words problem, and have the mathematical prerequisites (dot products, softmax, matrix multiplication). The key forward reference ("attention resolves polysemy") was planted in the previous lesson. The student should be expecting this lesson to explain how.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand how dot-product attention allows tokens to create context-dependent representations by computing weighted averages where the weights are determined by the input itself — and to feel the specific limitation of using raw embeddings (one representation per token for both seeking and offering) that motivates Q/K projections.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Token embeddings (dense vectors for each token) | DEVELOPED | DEVELOPED | embeddings-and-position | OK | Student needs to manipulate embeddings as vectors, compute with them. Has done this in notebook. |
| Bag-of-words problem (order lost without position) | DEVELOPED | DEVELOPED | embeddings-and-position | OK | This is the launching point for the lesson. Student already feels this problem. |
| Polysemy limitation (one vector per token regardless of context) | INTRODUCED | INTRODUCED | embeddings-and-position | OK | Student needs to recognize the problem, not solve it. INTRODUCED is sufficient — the lesson will develop it further. |
| Dot product as similarity | DEVELOPED | INTRODUCED | embeddings-and-position | GAP | Student used cosine similarity in the notebook but hasn't formalized dot product as "how much two vectors point in the same direction" or computed raw dot products to score similarity. |
| Softmax (converting scores to probabilities) | DEVELOPED | DEVELOPED | what-is-a-language-model, Series 1 | OK | Will apply softmax to attention scores. Student has used softmax many times. |
| Matrix multiplication | APPLIED | APPLIED | Series 1-2 | OK | Attention is fundamentally matrix multiplication. Student is fluent. |
| Weighted average (sum of values weighted by coefficients that sum to 1) | INTRODUCED | Not explicitly taught | — | GAP | Implicit in many contexts (expected value, loss averaging) but never formalized as "weighted average" or connected to attention. |

### Gap Resolution

| Concept | Gap Size | Resolution |
|---------|----------|------------|
| Dot product as similarity | Small (has cosine similarity, needs to see raw dot product as similarity score) | Brief recap section (2-3 paragraphs). Show that dot product between two vectors is large when they point the same direction, small/zero when orthogonal, negative when opposite. Connect to cosine similarity: dot product = cosine similarity x magnitudes. Use embedding vectors the student already knows. |
| Weighted average | Small (has the ingredients, needs the framing) | Build into the main explanation. When moving from "average all embeddings" to "weighted average," explicitly define weighted average: each element multiplied by a weight, weights sum to 1, result is a blend biased toward high-weight elements. Use concrete 3-token numerical example. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Attention is just looking at nearby words (like a convolution filter)" | CNNs use local receptive fields. The student might assume attention is another locality mechanism. | "The cat sat on the mat because **it** was soft." "it" attends to "mat" 6 positions away, skipping all intervening tokens. Attention has no distance preference — it computes all-pairs relevance. | Hook section — show long-range dependency that convolutions can't capture in one layer. |
| "The dot product between two embeddings tells you their semantic similarity (and that's all you need)" | Student just learned embedding similarity in previous lesson. Might think raw embedding dot product is sufficient for attention. | "The **bank** was steep" — "bank" and "steep" have LOW embedding similarity (different semantic clusters) but "steep" is highly relevant to understanding "bank" in this context. Relevance is not the same as similarity. | Elaborate section — after showing raw dot-product attention works partially, demonstrate where similarity != relevance. |
| "Attention produces a completely new representation, replacing the original embedding" | Natural assumption: input goes in, something new comes out. | If attention replaced the embedding, a token in an uninformative context would lose its original meaning. In practice, attention output is ADDED to the original (residual stream). This lesson won't implement residuals yet but should plant the seed. | Summarize section — forward reference: "In the full transformer, attention output is added to the original embedding, not substituted for it." |
| "Each token 'decides' what to attend to (like conscious choice)" | Anthropomorphic language is tempting: "the token looks at..." | Attention weights are just dot products + softmax. Change the embedding of a distant token and ALL attention weights shift. There's no "decision" — it's a deterministic function of all embeddings simultaneously. | Explain section — use "mechanical" language: "the weights are computed from..." not "the token chooses to..." |
| "Symmetric attention (token A attends to B the same as B attends to A) is fine" | Raw dot-product attention IS symmetric. The student might not see this as a problem. | "The cat chased the mouse." "cat" should attend strongly to "chased" (cat is the agent of chasing). "chased" should attend to both "cat" (who chased) and "mouse" (what was chased). The relationship is asymmetric: "cat" needs "chased" for one reason, "chased" needs "cat" for a different reason. Symmetric scores can't express this. | Elaborate section — this is the cliffhanger limitation that motivates Q/K in Lesson 2. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| "The bank was steep" vs "The bank was closed" | Positive | Show context-dependent meaning. Same word "bank" should have different effective representations depending on surrounding words. | Classic polysemy example the student already knows from the embeddings lesson (amber warning box). Completing the promise. |
| 4-token sentence with 3-dim embeddings: compute attention weights by hand | Positive | Make attention concrete and traceable. Student computes dot products, applies softmax, computes weighted average — sees every number. | Tiny dimensions mean every step fits on screen. Demystifies "attention" as just arithmetic the student can do with pencil and paper. |
| Averaging all embeddings (bag of words) → "Dog bites man" same as "Man bites dog" | Negative | Show that uniform averaging loses word order and meaning. Motivate weighted averaging. | Directly callbacks to the bag-of-words problem from embeddings-and-position. Student already felt this — now they see the formal consequence. |
| "The cat chased the mouse" — asymmetric attention need | Positive (stretch) | Reveal the symmetry limitation of raw dot-product attention. "cat" and "chased" should attend to each other for different reasons, but raw dot products give symmetric scores. | Sets up the Q/K motivation for Lesson 2. The example is simple enough to trace but rich enough to show the asymmetry problem. |

---

## Phase 3: Design

### Narrative Arc

Every word in a sentence means something different depending on what's around it. "Bank" near "river" evokes a muddy slope; "bank" near "money" evokes a financial institution. But the embedding table the student just built gives "bank" exactly ONE vector, regardless of context. The student already felt this limitation — the amber warning box in the embeddings lesson said "attention resolves this." Now it's time to deliver on that promise.

The path from problem to mechanism follows three escalating attempts. First, the naive approach: average all the embeddings in a sentence. This is the bag-of-words problem the student already knows — "dog bites man" and "man bites dog" become identical. Second, a weighted average: some words should matter more than others for understanding a given word. But who decides the weights? A fixed formula? That can't adapt to new sentences. The breakthrough insight: let the tokens themselves decide, by computing how similar they are to each other. A token with high similarity to the current token gets a high weight. This is dot-product attention — and it works surprisingly well for something so simple. But it has a crack: each token has one embedding that must simultaneously serve as "what I'm searching for" (when it's the query) and "what I have to offer" (when other tokens look at it). This dual-role tension is the limitation the student should feel by the end — and it's exactly what Q and K will resolve in the next lesson.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | 4 tokens with 3-dimensional embeddings, every dot product and softmax value computed explicitly | Attention feels abstract until you trace real numbers. 3-dim embeddings mean the student can verify every multiplication. Turns "attention mechanism" into "arithmetic I can do by hand." |
| Visual | Attention heatmap — tokens on both axes, color intensity = attention weight. Interactive: student types a sentence, sees the attention matrix. | The attention weight matrix IS the core data structure. Seeing it as a heatmap makes patterns visible: which tokens attend to which. Symmetry becomes visible (the matrix is symmetric for raw dot-product attention). |
| Verbal/Analogy | "Cocktail party" analogy — in a noisy room, you selectively focus on relevant conversations. But you can only focus based on what you already know about yourself (one representation). You can't simultaneously be "person looking for directions" and "person who knows about restaurants." | Maps attention to familiar experience. The limitation (one representation for both roles) maps to the intuitive impossibility of being two things at once. |
| Symbolic | Attention(X) = softmax(XX^T) @ X — the full formula for raw dot-product attention without projections. Build up piece by piece: XX^T = similarity matrix, softmax = normalize rows, multiply by X = weighted average. | The formula is the anchor students return to. Building it incrementally from pieces (similarity -> normalize -> weight) prevents it from feeling monolithic. |
| Intuitive | "The input decides what matters" — this is the paradigm shift from CNNs (fixed filters) to attention (data-dependent weights). The weights aren't parameters learned once; they're freshly computed from every new input. | The conceptual revolution of attention is content-dependence. Without grasping this, the student will treat attention as "just another layer." |

### Cognitive Load Assessment

- **New concepts in this lesson:** (1) Context-dependent representations as a goal, (2) Weighted average as a mechanism, (3) Dot-product attention (data-dependent weight computation). That's 3 new concepts — at the limit.
- **Previous lesson load:** BUILD (embeddings-and-position). Two related concepts, notebook-heavy.
- **Assessment:** STRETCH is appropriate. First encounter with attention is inherently high-novelty. The student is entering a new paradigm (tokens communicating with each other). However, the mathematical prerequisites are solid (dot products, softmax, matrix multiplication all at DEVELOPED or APPLIED depth), so the stretch is conceptual, not mechanical. The notebook grounds it in computation.

### Connections to Prior Concepts

- **Bag-of-words problem** (embeddings-and-position, DEVELOPED): Direct callback. "You already know what happens when you average all embeddings. Now let's fix it."
- **Polysemy limitation** (embeddings-and-position, INTRODUCED): The promise. "We showed you 'bank' gets one vector. Now we show how context creates different effective representations."
- **Softmax** (what-is-a-language-model, Series 1, DEVELOPED): Applied to attention scores. "Same softmax you used for temperature and classification — now it normalizes similarity scores into weights."
- **Embedding similarity / dot product** (embeddings-and-position, INTRODUCED): Extended from cosine similarity in the embedding explorer. Now dot product is the mechanism for computing relevance.
- **CNN filters as fixed patterns** (Series 3, DEVELOPED): Contrast. "CNN filters are the same for every input. Attention weights change with every sentence."

**Analogies that could be misleading:**
- "Embeddings are a learned dictionary" — dictionaries have one definition per word. Attention creates context-dependent definitions, so this analogy should be explicitly updated: "The embedding table gives each word one dictionary entry. Attention is like reading the surrounding entries and rewriting your definition based on what's nearby."

### Scope Boundaries

**This lesson IS about:**
- Why static embeddings are insufficient (context-dependent meaning)
- How dot-product attention works using RAW embeddings (no projections)
- The mechanics: similarity scores, softmax normalization, weighted averaging
- Feeling the limitation: one representation per token for both roles
- Target depth: DEVELOPED for dot-product attention concept, INTRODUCED for the dual-role limitation

**This lesson is NOT about:**
- Q, K, V projections (Lesson 2 and 3) — explicitly deferred. Do NOT even name them.
- Multi-head attention (Lesson 4)
- The transformer architecture (Lessons 5-6)
- Causal masking (Lesson 6)
- Implementation of a complete attention module (just raw dot-product attention in the notebook)
- Scaled dot-product attention (scaling by sqrt(d) is Lesson 2, tied to Q/K)
- Self-attention vs cross-attention distinction (not relevant until encoder-decoder)

### Lesson Outline

1. **Context + Constraints**
   - "This lesson: how tokens can create context-dependent representations using only dot products and softmax — tools you already have."
   - Explicit scope: no Q, K, V. No projections. Raw embeddings only. We'll see why this isn't enough, and the next lesson will fix it.
   - Has notebook: `4-2-1-the-problem-attention-solves.ipynb`

2. **Recap: Dot Product as Similarity** (Gap resolution)
   - Brief section: dot product between two vectors measures how much they point in the same direction.
   - Connect to cosine similarity from the embedding notebook.
   - Tiny numerical example: two 3-dim vectors, compute dot product, interpret the sign and magnitude.
   - Visual: two arrows in 2D, angle between them, dot product value.

3. **Hook: The Polysemy Promise** (Before/after + callback)
   - Callback to the amber polysemy warning from embeddings-and-position: "We told you 'bank' gets one vector regardless of context. We promised attention would fix this. Let's deliver."
   - Show two sentences: "The bank was steep and muddy" / "The bank raised interest rates." Same embedding for "bank" in both.
   - Goal: by the end of this lesson, you'll see how surrounding words can create different effective representations for the same token.

4. **Explain: From Bag of Words to Weighted Average**
   - Start with the bag-of-words average (direct callback to embeddings-and-position). Average all embeddings: every token gets the same "summary."
   - Negative example: "Dog bites man" vs "Man bites dog" produce the same average embedding.
   - Improvement: weighted average. Each token should get a DIFFERENT weighted combination of the other tokens, emphasizing the ones most relevant to it.
   - Key question: who decides the weights? Not a person. Not learned-once parameters (that would be input-independent). The input itself.

5. **Explain: Dot-Product Attention**
   - Core insight: use dot products between embeddings to compute relevance scores. Similar embeddings get high scores.
   - Step-by-step with 4 tokens, 3-dim embeddings:
     - Compute all pairwise dot products (4x4 matrix)
     - Apply softmax to each row (now rows sum to 1 — these are weights)
     - For each token, compute weighted average of all embeddings using that token's weight row
     - The result: each token now has a context-dependent representation
   - Formula build-up: scores = XX^T, weights = softmax(scores), output = weights @ X
   - Full formula: Attention(X) = softmax(XX^T) X

6. **Check: Predict the Attention Pattern**
   - Give the student a 4-word sentence and ask: before computing anything, which tokens do you think will attend most to which? Then show the actual heatmap. Compare intuition to computation.

7. **Explore: Interactive Attention Heatmap Widget**
   - Student types a sentence, sees the raw dot-product attention weight matrix as a heatmap.
   - Color intensity = attention weight. Hover shows exact value.
   - TryThisBlock suggestions: "The bank was steep" vs "The bank was closed" — does "bank" attend differently? Try a sentence with repeated words. Try a very short sentence.
   - Key observation prompt: "Look at the matrix. Is it symmetric? Why or why not?" (It IS symmetric because dot(a,b) = dot(b,a). This becomes the limitation.)

8. **Elaborate: The Dual-Role Limitation**
   - Guided observation: the attention matrix is symmetric. Token A attends to Token B with the same weight as B attends to A.
   - Why this is a problem: "The cat chased the mouse." For "cat," "chased" is relevant because it tells you what the cat DID. For "chased," "cat" is relevant because it tells you WHO chased. Different reasons, but the score is identical.
   - Deeper: each token has ONE embedding vector. When it's the "focus" token, that vector determines what it looks for. When it's a "context" token, the SAME vector determines what it offers. These are fundamentally different roles.
   - Cocktail party analogy: you're at a party. When you're listening, you want to find people talking about AI. When someone else is listening, they see you as "software engineer who likes cooking." Your "what I seek" and "what I offer" should be different representations — but with one embedding, they're forced to be the same.
   - Negative example confirming the limitation: construct a case where raw dot-product attention gives wrong/misleading weights because the similarity score doesn't match the relevance score.

9. **Check: Transfer Question**
   - "If every token had TWO different vectors — one for 'what I'm looking for' and one for 'what I'm advertising' — would the attention matrix still be symmetric? Why not?" (Answer: No, because dot(query_A, key_B) != dot(query_B, key_A) in general. The student should be able to reason this through.)

10. **Practice: Notebook** (`4-2-1-the-problem-attention-solves.ipynb`)
    - Scaffolding: guided (first attention implementation, heavy scaffolding)
    - Exercise 1: Implement raw dot-product attention on a 4-token, 8-dim example. Compute XX^T, apply softmax, multiply by X.
    - Exercise 2: Visualize the attention weight matrix as a heatmap. Confirm it's symmetric.
    - Exercise 3: Use pretrained GPT-2 embeddings. Encode two sentences with "bank" in different contexts. Compute raw dot-product attention. Does "bank" get different attention patterns? (Partially — but limited by the single-representation constraint.)
    - Stretch exercise: Compute the attention output for "bank" in both sentences. How different are the resulting vectors? (They'll be somewhat different because the context embeddings differ, but the limitation is that "bank" can't seek different information in different contexts.)

11. **Summarize**
    - Key insight: attention is a weighted average where the input itself determines the weights via dot products.
    - What we achieved: context-dependent representations from static embeddings.
    - What's still broken: each token has one vector for both seeking and offering. The attention matrix is symmetric. Different roles require different representations.
    - Forward reference: "The next lesson introduces two separate projection matrices that let each token create a 'what I'm looking for' vector AND a 'what I'm advertising' vector. The asymmetry problem disappears."
    - Do NOT name Q or K yet. Just the concept: separate vectors for separate roles.

12. **Next Step**
    - "Next: Queries, Keys, and the Relevance Function — giving each token the ability to seek and advertise differently."

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings that would leave the student fundamentally lost, but one critical-severity omission (planned misconception completely missing) and three improvement findings that would significantly strengthen the lesson. The lesson is well-structured and the core pedagogical arc (three attempts, worked example, limitation reveal, cliffhanger) works. The main issues are gaps between plan and implementation.

### Findings

#### [CRITICAL] — Locality misconception (Misconception #1) completely absent

**Location:** The entire lesson (should have been addressed in the Hook or Explain sections per the plan)
**Issue:** The planning document identified a key misconception: "Attention is just looking at nearby words (like a convolution filter)." The plan called for a long-range dependency example ("The cat sat on the mat because **it** was soft" -- "it" attends to "mat" 6 positions away, skipping all intervening tokens) and the explicit statement that "attention has no distance preference -- it computes all-pairs relevance." Neither the example nor the statement appears anywhere in the built lesson.
**Student impact:** The student has extensive CNN experience from Series 3 where locality is a defining feature. Without explicitly contrasting attention's global computation to CNN locality, the student may carry over the "nearby things matter most" assumption and fail to appreciate that attention's all-pairs nature is a paradigm shift. The aside about CNN filters being "fixed" addresses the data-dependence contrast but not the locality contrast.
**Suggested fix:** Add 1-2 paragraphs in Section 4 (the three-attempts section, near the CNN aside) or as a standalone observation after the worked example. Use the planned "it" -> "mat" long-range example: "Notice that attention computes a score between EVERY pair of tokens, regardless of distance. 'It' can attend just as strongly to 'mat' (6 positions away) as to 'was' (next to it). This is fundamentally different from a CNN filter that only sees a fixed local window." This also reinforces the worked example's 4x4 matrix showing ALL pairwise computations.

#### [IMPROVEMENT] — Dot product recap missing the planned geometric/spatial visual

**Location:** Section 2 (Quick Recap: Dot Product as Similarity)
**Issue:** The planning document specified: "Visual: two arrows in 2D, angle between them, dot product value." The built lesson uses only numerical examples (two 3-dim vectors with explicit arithmetic) and the symbolic formula. There is no geometric/spatial representation of the dot product. The section presents the verbal explanation ("point in the same direction") and the numerical example, but without an arrow diagram the student must take "point in the same direction" on faith rather than seeing it.
**Student impact:** The student has the dot product at INTRODUCED depth. The gap resolution is designed to bring it to DEVELOPED. Without the geometric modality (arrows and angle), the student gets arithmetic and words but not the spatial intuition. "Two vectors pointing the same way" is less visceral without actually seeing the arrows. This is particularly important because the "similarity = how much they point the same way" intuition is the foundation for all the attention weight reasoning that follows.
**Suggested fix:** Add a small inline SVG or diagram showing two 2D arrows: one case where they point similarly (large positive dot product), one where they're perpendicular (zero), one where they point opposite (negative). This can be compact -- a simple 3-panel figure. Alternatively, a Mafs component with two draggable vectors and a real-time dot product readout (but that may be over-engineering for a recap section).

#### [IMPROVEMENT] — Q, K, V terms appear in the scope boundary list despite plan saying "do NOT even name them"

**Location:** Section 1 (ConstraintBlock, scope items 6-8)
**Issue:** The scope block lists "NOT: Q, K, V projections," "NOT: multi-head attention, transformer architecture, or causal masking," and "NOT: scaled dot-product attention (scaling by sqrt(d) is tied to Q/K, next lesson)." The planning document Phase 3 Scope Boundaries says: "Q, K, V projections (Lesson 2 and 3) -- explicitly deferred. Do NOT even name them." Naming them in the scope list contradicts this directive. Additionally, the NextStepBlock at the bottom links to "Queries, Keys, and the Relevance Function" which names Q and K, and the transfer question's reveal answer references "two separate projection matrices" (which is fine -- it describes the concept without naming Q/K).
**Student impact:** The student sees "Q, K, V projections" in the scope boundary. This plants terminology they don't yet have context for. When the next lesson introduces these as Q and K, the "aha" moment is slightly blunted because they already saw the labels. The pedagogical strategy is to build the NEED for two separate vectors from felt experience, then introduce Q and K as the names for those vectors. Showing the names prematurely short-circuits this.
**Suggested fix:** Rewrite the scope items to avoid the Q/K/V terminology: "NOT: Learned projection matrices that fix the limitation you'll feel here -- that's the next lesson's answer" or simply "NOT: The fix for the limitation you'll discover (next lesson)." The NextStepBlock title showing "Queries, Keys" is acceptable since it's the last element and the student has already felt the limitation -- seeing the next lesson's title after the cliffhanger is fine. But the scope block at the TOP shows the names before the limitation is felt.

#### [IMPROVEMENT] — Positional encoding status unclear for the widget and worked example

**Location:** Section 6 (Check: Predict the Pattern) and Section 7 (Interactive Widget)
**Issue:** The prediction exercise asks about two instances of "the" and whether they have the same attention pattern. The reveal says: "only positional encoding (which we're ignoring for clarity) could" distinguish them. This is the only place the lesson states that positional encoding is being ignored. The widget title says "Raw Dot-Product Attention Matrix" but doesn't explain what "raw" means. The student learned in Module 4.1 that the model's input is embedding + PE. They might wonder: "Are we computing attention on embeddings-with-position or embeddings-without-position?" This ambiguity isn't resolved until the student reaches the prediction exercise reveal.
**Student impact:** The student might be confused during the worked example (Section 5) about whether position is included. In Module 4.1, positional encoding was presented as an essential part of the input ("without position, embeddings are a bag of words"). Now they see attention on embeddings without any mention of position. They might wonder if they're supposed to include it. The parenthetical "(which we're ignoring for clarity)" in the reveal comes too late and is buried inside a spoiler.
**Suggested fix:** Add a brief note in Section 5 (before the worked example) or even in the scope block: "For this lesson, we compute attention on raw token embeddings WITHOUT positional encoding. This isolates the attention mechanism itself. In the full transformer, positional encoding is included -- but that doesn't change how the attention computation works." This sets expectations clearly before the student encounters the examples.

#### [POLISH] — Anthropomorphic "let the tokens decide" language contradicts planned misconception handling

**Location:** Section 4 (Attempt 3 header: "Let the tokens decide") and nearby text
**Issue:** The planning document's Misconception #4 says to use "mechanical" language and avoid phrasing like "the token chooses to..." The lesson uses "let the tokens decide" as the header for Attempt 3 and "the input itself decides what matters" in the following text. Later sections use more mechanical language ("the weights are computed from..."), but the introduction of the concept uses decision/agency language.
**Student impact:** Minor. The student may form a slightly anthropomorphic mental model initially, though the subsequent mechanical language in the formula section corrects this. The "decides" framing is engaging and motivating (which is why it was used), but it does plant the agency misconception.
**Suggested fix:** Consider rephrasing the header to "Attempt 3: Let the input determine the weights" and the following text to "the input itself determines what matters." This maintains the sense of data-dependence without implying conscious choice. Or keep the current language but add a brief note: "When we say 'the tokens decide,' we mean the dot products and softmax mechanically produce weights from the input vectors -- there's no decision-making, just linear algebra."

#### [POLISH] — Widget preset buttons lack explicit cursor-pointer class

**Location:** AttentionMatrixWidget.tsx, line 237-242 (preset sentence buttons) and lines 249-267 (toggle buttons)
**Issue:** The preset sentence buttons and the weights/scores toggle buttons rely on default browser cursor behavior for buttons. While most browsers do show cursor-pointer for buttons by default, the lesson-review skill's Interaction Design Rule states that interactive elements should have appropriate cursor styles explicitly set. The `<details><summary>` elements in the lesson DO have explicit `cursor-pointer`, showing awareness of this pattern, but the widget buttons don't.
**Student impact:** Minimal. Most students will see the correct cursor. But consistent explicit cursor styling is a best practice.
**Suggested fix:** Add `cursor-pointer` to the className of preset buttons and toggle buttons in the widget.

### Review Notes

**What works well:**
- The three-attempt escalation (average -> weighted average -> data-dependent weights) is the strongest structural choice in the lesson. It creates genuine intellectual momentum -- each attempt feels like a natural improvement on the last, and the "who decides the weights?" question creates real tension before the dot-product answer lands.
- The cliffhanger is compelling. The symmetry limitation is surfaced clearly, the transfer question asks the student to reason about the fix, and the forward reference avoids naming Q/K while making the next lesson feel like a natural resolution. This is exactly the "feel the limitation, then get the fix" pattern the module plan calls for.
- The worked example with 4 tokens and 3-dim embeddings is well-calibrated. Small enough to trace, large enough to show interesting patterns (diagonal dominance, off-diagonal variation).
- The widget is pedagogically well-designed. The symmetry indicator, the mirror-cell dashed outline on hover, and the toggle between raw scores and softmax weights all serve the lesson's goals. The preset sentences are well-chosen to let the student explore the planned comparisons.
- The polysemy hook callbacks the amber warning from Module 4.1 effectively. The student should feel a sense of "finally, we're getting to this."
- Q/K/V terminology does NOT leak into the lesson content (only in the scope boundary -- flagged as an improvement). The transfer question uses s_A/a_B notation, and the forward reference says "two separate projection matrices" without naming them. The pedagogical boundary is well-maintained where it matters most.

**Pattern to watch:**
- The lesson is long (~1080 lines of JSX). While it's well-structured with clear section breaks, future lessons in this module should aim tighter if possible. The STRETCH load type justifies the length here, but BUILD lessons (queries-and-keys, values-and-attention-output) should be shorter.

**Priority for revision:**
1. Add the locality misconception content (CRITICAL) -- 1-2 paragraphs plus the planned "it" -> "mat" example
2. Add positional encoding status clarification (IMPROVEMENT) -- 1 sentence in the worked example section
3. Remove Q/K/V from the scope boundary list (IMPROVEMENT) -- simple text edit
4. The dot product visual (IMPROVEMENT) is desirable but lower priority than the above three

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

All 6 findings from iteration 1 have been properly addressed. No new issues found.

### Iteration 1 Fix Verification

1. **CRITICAL -- Locality misconception:** Fixed. Two paragraphs added after the formula build-up (lines 742-773). The "it" -> "mat" long-range dependency example is present and well-constructed. The WarningBlock aside explicitly contrasts attention's all-pairs computation with CNN local windows. The placement after the worked example is effective -- the student has just seen the 4x4 score matrix where every pair has a score, so "every token computes a score with every other token, regardless of distance" lands naturally.

2. **IMPROVEMENT -- Q/K/V terms removed from scope boundary:** Fixed. Scope items now read "NOT: Learned projection matrices that fix the limitation you'll discover -- that's the next lesson's answer." Comprehensive search confirms Q/K/V terms appear only in developer-facing JSDoc comments (lesson component and widget), never in student-facing rendered text. The NextStepBlock title "Queries, Keys, and the Relevance Function" appears at the very end after the cliffhanger, which the iteration 1 review explicitly approved.

3. **IMPROVEMENT -- PE status clarification:** Fixed. Lines 515-519 state clearly before the worked example: "For now, we're working with raw token embeddings -- no positional encoding. This isolates the attention mechanism itself. In the full transformer, positional encoding is included, but that doesn't change how the attention computation works." Sets expectations early and prevents confusion.

4. **IMPROVEMENT -- 3-panel geometric SVG for dot products:** Fixed. Three inline SVGs (lines 206-264) show similar direction (small angle, emerald label), perpendicular (right angle indicator, amber label), and opposite (rose label). Color-coded vectors (sky-blue for a, violet for b) are consistent. Labels below each panel state the dot product interpretation. Completes the geometric/spatial modality for the recap section.

5. **POLISH -- Anthropomorphic language:** Fixed. "Let the tokens decide" header changed to "Let the input determine the weights" (line 483). "The input itself decides what matters" changed to "the input itself determines what matters" (line 493). Text now explicitly uses "operating mechanically on whatever vectors come in" (line 495). The established mechanical framing is maintained throughout.

6. **POLISH -- cursor-pointer on widget buttons:** Fixed. Preset sentence buttons (line 239), weights toggle button (line 251), and scores toggle button (line 261) all have explicit `cursor-pointer` in their className.

### Full Re-Review Checks

**Plan alignment:** All 12 outline items from Phase 3 are present and match the planned structure. All 5 planned misconceptions are addressed. All 4 planned examples are included. No undocumented deviations.

**Pedagogical principles:** 5 modalities present (verbal/analogy, visual, symbolic, concrete, intuitive). At least 2 positive and 2 negative examples. Concrete before abstract ordering maintained. Load at 3 concepts (at the limit for STRETCH). All new concepts connected to prior knowledge. Em-dashes properly formatted (no spaces in rendered text). Interactive elements have appropriate cursor styles.

**Q/K/V scope boundary:** Verified clean. grep for Q/K/V terminology across both the lesson component and widget returns only JSDoc developer comments, never rendered student-facing text.

### Review Notes

This lesson is ready to ship. The iteration 1 fixes were all cleanly implemented without introducing new issues. The locality misconception addition (the most significant fix) integrates naturally into the lesson flow and strengthens the CNN contrast that was already partially present in the aside about data-dependent vs fixed weights. The geometric SVG visual completes a modality gap that makes the dot-product recap section genuinely multi-modal. The lesson is long (~1180 lines) but well-structured with clear section breaks, appropriate for a STRETCH lesson that introduces the entire attention paradigm.
