# Lesson: Values and the Attention Output

## Phase 1: Orient — Student State

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Q and K as learned linear projections (W_Q and W_K transform the same embedding into separate "seeking" and "offering" vectors) | DEVELOPED | queries-and-keys | Core concept of Lesson 2. Job fair analogy, geometric SVG, "learned lens" framing, nn.Linear connection, full hand-traced worked example. WarningBlock: Q and K are properties of the learned matrices, not the token. |
| QK^T as a learned asymmetric relevance matrix (entry (i,j) = q_i dot k_j) | DEVELOPED | queries-and-keys | Replaces XX^T from Lesson 1. Explicitly shown why it's asymmetric. Side-by-side heatmap comparison (raw symmetric vs projected asymmetric) is the visual centerpiece. |
| Scaled dot-product attention weights: softmax(QK^T / sqrt(d_k)) | DEVELOPED | queries-and-keys | Built incrementally in four steps: project Q, project K, score QK^T/sqrt(d_k), normalize with softmax. Compared to Lesson 1's softmax(XX^T). Formula is the attention WEIGHTS only — output computation deferred to this lesson. |
| Scaling by sqrt(d_k) to prevent softmax saturation | DEVELOPED | queries-and-keys | Motivated by dimension growth. Callbacks to temperature slider and vanishing gradients. Concrete numerical examples at d_k=3 vs d_k=512. Essential for training, not cosmetic. |
| Dot-product attention on raw embeddings: Attention(X) = softmax(XX^T) X | DEVELOPED | the-problem-attention-solves | Full step-by-step worked example: 4 tokens, 3-dim embeddings. Student traced every number. Formula built incrementally. |
| Weighted average as a mechanism for context (blend embeddings with weights that sum to 1) | DEVELOPED | the-problem-attention-solves | Built through three escalating attempts. Explicit formula and interpretation. |
| Dual-role limitation resolved — seeking vs offering separated by Q/K | DEVELOPED | queries-and-keys | Resolved from INTRODUCED in Lesson 1. Q = seeking, K = offering. Asymmetry confirmed. |
| Similarity vs relevance distinction (dot product measures similarity; Q/K projections compute learned relevance) | DEVELOPED | queries-and-keys | Extended from INTRODUCED in Lesson 1 into a concrete mechanism. "W_Q and W_K would learn projections that make the dot product large" for relevant pairs. |
| Data-dependent weights (attention weights freshly computed from each input, not fixed parameters) | DEVELOPED | the-problem-attention-solves | Explicitly contrasted with CNN fixed filters. InsightBlock. |
| Residual connections / skip connections (F(x) + x formulation, "editing not writing") | DEVELOPED | resnets (3.2) | Core concept of ResNet. "A residual block starts from identity and learns to deviate." Gradient highway perspective. "Editing a document, not writing from scratch" analogy. Student implemented ResidualBlock class. |
| nn.Linear as learned matrix multiplication | DEVELOPED | Series 2 (nn-module, 2.1.3) | Student has built models with nn.Linear layers, knows weight shape, can predict parameter counts. Used for Q and K in Lesson 2 — "just nn.Linear(d_model, d_k, bias=False)." |
| Softmax (converting scores to probabilities) | DEVELOPED | what-is-a-language-model, Series 1 | Used repeatedly: classification, temperature slider, LM output, attention scores. Student is fluent with softmax mechanics. |
| Matrix multiplication | APPLIED | Series 1-2 | Used in training loops, forward passes, embedding lookup, Q/K projections, QK^T. Student is fluent. |
| Token embeddings as learned lookup (nn.Embedding maps integer ID to dense vector) | DEVELOPED | embeddings-and-position | nn.Embedding(50000, 768). Embeddings are learned parameters, not preprocessing. |
| Complete input pipeline: text -> tokens -> IDs -> embeddings + PE -> tensor | DEVELOPED | embeddings-and-position | Student knows what the transformer receives as input. |
| Linear projection as a learned transformation ("the matrix is the lens") | INTRODUCED | queries-and-keys | Gap resolved in Lesson 2. Framed as "same input, different lens, different view." Connected to nn.Linear. Not a separate section — integrated into Q/K explanation. |

**Mental models and analogies already established:**
- "Attention is a weighted average where the input determines the weights" — the defining insight (Lesson 1)
- "The input decides what matters" (data-dependent weights vs fixed parameters) — paradigm shift from CNNs (Lesson 1)
- "Similarity is not the same as relevance" — extended from observation (Lesson 1) to mechanism (Lesson 2, via Q/K)
- "Q and K are learned lenses — same embedding, different lens, different view" — the projection matrix is the interesting learned component (Lesson 2)
- "QK^T computes learned relevance, not raw similarity" — extends similarity-vs-relevance into concrete mechanism (Lesson 2)
- "Scaling by sqrt(d_k) is not cosmetic — it's the difference between a model that learns and one that doesn't" (Lesson 2)
- "Editing a document, not writing from scratch" — residual connections (Series 3, resnets)
- "Skip connection = direct phone line bypassing the telephone chain" (Series 3, resnets)
- Job fair analogy: two cards per person (seeking = query, offering = key) (Lesson 2)
- Cocktail party analogy: what you search for vs what you advertise (Lesson 1)

**What was explicitly NOT covered (and is relevant to this lesson):**
- V projection (what a token contributes when attended to) — this lesson's core concept
- The full attention output (the weighted sum using V) — this lesson
- Residual stream concept in the transformer context — seeded in Lesson 1 (muted aside), needs full development here
- Multi-head attention — Lesson 4
- The transformer block architecture — Lesson 5
- Causal masking — Lesson 6
- Attention as a complete layer (no output projection yet)

**Seeds planted for this lesson:**
1. From Lesson 1 (the-problem-attention-solves): "In the full transformer, attention output is ADDED to the original embedding, not substituted. Callback to skip connections from ResNets." (muted aside at the end)
2. From Lesson 2 (queries-and-keys), the forward reference: "The key that makes a token relevant (K) and the information it should provide when attended to are different things. 'Chased' is relevant to 'cat' because of its action semantics (captured by K), but what 'chased' CONTRIBUTES to 'cat's representation should be different — maybe verb tense, transitivity, or past-action features. Should we use the same embedding for matching AND contributing? We already know the answer to that kind of question."

**Readiness assessment:** The student is well-prepared. They have the full attention weight computation at DEVELOPED depth (Q, K, scaling, softmax) and have experienced two iterations of the "one vector serving two roles" pattern — first raw embeddings (seeking and offering), now embeddings again (matching and contributing). The Lesson 2 seed explicitly asks the student to recognize the same pattern. The mathematical prerequisites (matrix multiplication, softmax, linear projections) are at DEVELOPED or APPLIED. Residual connections are at DEVELOPED from Series 3 (resnets) — the student built ResidualBlock with F(x) + x and understands "editing not writing." The residual stream concept in attention extends this directly.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand that V (a third learned projection) separates "what makes a token relevant" (K) from "what information it contributes when attended to" (V), completing single-head attention as output = softmax(QK^T/sqrt(d_k))V, and that the attention output is ADDED to the input via the residual stream rather than replacing it.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Scaled attention weights: softmax(QK^T / sqrt(d_k)) | DEVELOPED | DEVELOPED | queries-and-keys | OK | Student must compute the full attention weight matrix to apply it to V. Has built this incrementally across Lessons 1-2 and computed by hand. |
| Q and K as learned linear projections | DEVELOPED | DEVELOPED | queries-and-keys | OK | Student must understand how projections work (same embedding, different matrix, different vector) so that V as a third projection feels like a natural extension. |
| Weighted average as a mechanism for context | DEVELOPED | DEVELOPED | the-problem-attention-solves | OK | The attention output IS a weighted average of V vectors using the attention weights. Student built this concept through three escalating attempts in Lesson 1. |
| Dual-role limitation pattern (one vector serving two functions) | DEVELOPED | DEVELOPED | queries-and-keys (resolved from INTRODUCED in Lesson 1) | OK | The student must recognize the SAME pattern: K serves matching, but the information contributed should be different. They've seen this pattern resolved once (Q/K for seeking/offering). |
| nn.Linear as learned matrix multiplication | DEVELOPED | DEVELOPED | Series 2, queries-and-keys | OK | V is another nn.Linear projection. Student already knows Q and K are nn.Linear layers. V is "the third one." |
| Residual connections (F(x) + x, "editing not writing") | DEVELOPED | DEVELOPED | resnets (3.2) | OK | The residual stream concept is a direct application of skip connections from ResNets. Student implemented ResidualBlock and understands F(x) + x. |
| Matrix multiplication | APPLIED | APPLIED | Series 1-2 | OK | Output = weights @ V is a matrix multiplication. Student is fluent. |
| Softmax | DEVELOPED | DEVELOPED | Series 1, what-is-a-language-model | OK | Already used in attention weights computation. No gap. |
| Dot-product attention on raw embeddings (Lesson 1 formula) | DEVELOPED | DEVELOPED | the-problem-attention-solves | OK | Reference point for comparison. In raw attention, the output is softmax(XX^T) X — the same X is used for scoring AND contributing. V replaces the second X with a separate projection. |

**Gap resolution:** No gaps. All prerequisites are at or above required depth. The student has seen the "two-roles-one-vector" pattern resolved once, making the V projection a natural pattern extension. Residual connections are at DEVELOPED from ResNets, requiring only a contextual bridge (same concept, new setting).

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "V is just the raw embedding — why do we need a third projection?" | The student saw raw embeddings used as the output in Lesson 1's formula (softmax(XX^T) X). If Q/K fixed the matching problem, maybe the raw embedding is fine for contributing. Also, three projections feels like unnecessary complexity. | Consider "chased" in "The cat chased the mouse." Its key (K) encodes "I'm an action verb" — this is what makes it relevant to "cat." But when "chased" contributes to "cat's" representation, what "cat" needs is not a duplicate of the action-verb signal. It needs information like "past tense," "transitive," "implies directed motion." These are different features of "chased" than what made it match. Without V, the contribution IS the key — the matching signal and the contribution signal are locked together. With V, the model learns to extract a different set of features for contribution than for matching. | In the motivation section (Section 3), immediately after re-establishing the "one vector, two roles" pattern. This is the core motivation for the entire lesson. |
| "V, K, and the embedding are all basically the same thing (just slightly different projections)" | All three are linear projections of the same embedding. The student might think they're redundant or interchangeable. | Set W_V = W_K in the worked example. The attention output would be a weighted average of the KEY vectors — meaning the contribution of each token to the output is the same signal used for matching. Now set W_V to project entirely different features. The output changes qualitatively. The contribution vector encodes different information than the matching vector. In a trained GPT-2 model, W_Q, W_K, and W_V learn very different projection patterns — they are not interchangeable. | In the worked example section, after computing the output. Brief aside: "What if W_V were the same as W_K?" Show that this collapses two distinct roles. |
| "The attention output replaces the original embedding (the token's meaning changes entirely)" | The attention mechanism produces a new vector for each token. If you're computing a "context-dependent representation," it feels like the old representation is gone. The student hasn't seen the residual stream yet (only a brief seed in Lesson 1). | If attention completely replaced the embedding, a token in an uninformative context (e.g., padding tokens, or a content word surrounded by stop words) would lose its original meaning entirely — the weighted average of uninformative neighbors would erase the token's identity. With the residual stream (output = attention(x) + x), the original embedding is always preserved. The attention output is a CORRECTION, not a replacement. A token that learns nothing useful from its neighbors still retains its own representation. | In the residual stream section (Section 7). This is the second core concept of the lesson. |
| "The residual stream in a transformer is just a skip connection (same as ResNet, nothing new)" | The student has residual connections at DEVELOPED from ResNets. They might view the transformer's residual stream as identical to what they've already learned, missing the specific architectural significance. | In ResNet, the skip connection is a local optimization trick — it helps gradients flow and makes identity easy to learn. In a transformer, the residual stream is the backbone of the entire architecture: every layer (attention AND feed-forward) reads from and writes to the same stream. It's not just "add the shortcut." It's the central communication channel that accumulates information across layers. The original embedding survives through the entire network, progressively enriched. This is a conceptual extension, not just a repeat. | After introducing the residual stream concept. InsightBlock or aside: "In ResNets, skip connections are local optimization tools. In transformers, the residual stream is the architecture's central highway." |
| "Attention produces a single output vector for the whole sequence" | The student might confuse the per-token output with a global sequence representation. Since attention combines information from all tokens, it might seem like the mechanism produces one summary. | The attention output has the SAME shape as the input: one vector per token. Token "cat" gets its own context-enriched vector (a weighted average of all V vectors, weighted by how much "cat" attends to each). Token "sat" gets a DIFFERENT context-enriched vector (same V vectors, different weights from "sat's" row of the attention matrix). Each token's output is different because each token has different attention weights (different row of the weight matrix). | In the worked example, when computing the output. Emphasize: "Each token computes its OWN weighted average of the V vectors. The output has the same shape as the input — one vector per token." |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Same 4 tokens ("The", "cat", "sat", "here"), same embeddings, same W_Q, W_K from Lesson 2, PLUS new W_V: full trace of V = W_V @ X, output = softmax(QK^T/sqrt(d_k)) @ V. Compute the final output vector for "cat." | Positive | Make the complete single-head attention operation concrete and traceable. Student computes every number from embedding to final output. The "before and after" across all three lessons: same input, progressively richer computation. | Continuity with Lessons 1-2 creates a powerful three-lesson arc. Same tokens, same embeddings — the student has traced raw attention and Q/K attention on these. Now V completes the picture. The output vector for "cat" is the culmination of the three-lesson build-up. |
| "The cat chased the mouse" — what does "chased" contribute to "cat" vs what made "chased" relevant to "cat"? K vs V distinction with semantic reasoning. | Positive | Demonstrate that K and V serve genuinely different functions. The matching signal (K: "I'm an action verb, relevant to a noun seeking its verb") is different from the contribution signal (V: "past tense, transitive, implies motion"). | This example has been used across all three lessons with evolving purpose: symmetry problem (Lesson 1), asymmetry resolution (Lesson 2), and now K-vs-V separation (this lesson). The student has deep familiarity with it. Using it again for V makes the three-role separation vivid across a single well-understood example. |
| W_V = identity matrix (V = raw embedding): show that this reduces to Lesson 1's formula (minus Q/K improvements). Compare the output to what the student would get from softmax(QK^T/sqrt(d_k)) @ X. | Positive (stretch) | Demonstrate that V generalizes the raw-embedding output from Lesson 1. When V = X (identity projection), the formula collapses to what they had before. V is not replacing something — it's generalizing it. | Connects the three lessons into one coherent story: Lesson 1 used X for everything (matching + contributing), Lesson 2 replaced X with Q and K for matching, this lesson replaces X with V for contributing. Identity matrix makes the connection mathematically explicit. |
| A token in an uninformative context without the residual stream: attention produces a meaningless average that erases the token's identity. With the residual stream: the original embedding is preserved. | Negative | Demonstrate that attention without the residual stream is destructive — it can erase token identity when context isn't informative. Motivates why the residual stream is necessary, not optional. | This directly addresses misconception #3 (attention replaces the embedding). The student must understand that attention's output is a CORRECTION added to the original, not a replacement. Without the residual stream, the mechanism is fragile. |
| W_V = W_K (contribution signal locked to matching signal): show that the output conflates what made tokens relevant with what they contribute. The output is qualitatively different from when W_V is independent. | Negative | Demonstrate that V and K are not interchangeable. When V = K, the contribution is the matching signal itself, which is the wrong information. | Directly addresses misconception #2 (all projections are basically the same). If W_V = W_K, the weighted average blends key vectors — the output tells "cat" about what made each token relevant, not what each token has to say. The distinction is concrete and traceable. |

---

## Phase 3: Design

### Narrative Arc

The student has spent two lessons building the attention weight matrix from scratch. Lesson 1 started with raw dot-product attention and felt its limitations — one embedding for both seeking and offering, symmetric scores for asymmetric relationships. Lesson 2 introduced Q and K, separate "lenses" that let each token create distinct seeking and offering vectors, producing asymmetric relevance scores properly scaled for training. The attention WEIGHTS are now complete: softmax(QK^T / sqrt(d_k)) tells us exactly how much each token should attend to every other token.

But weights need something to weight. Right now, when the student computes the attention output, the weighted average blends the raw embeddings (or at best, the key vectors). The forward reference from Lesson 2 planted the question directly: "The key that makes a token relevant and the information it should provide when attended to are different things. Should we use the same embedding for matching AND contributing?" The student has already resolved this kind of question once — they know that one vector serving two distinct roles is a design flaw. The answer is a third projection: V = W_V @ embedding, a "contribution lens" that lets the model learn WHAT information each token should provide when it's attended to, independently from what made it relevant in the first place.

This is the final piece that completes single-head attention: output = softmax(QK^T / sqrt(d_k)) V. But there's one more insight the student needs before the transformer block: the attention output is not a replacement for the original embedding. It's ADDED to the original via the residual stream — the same skip-connection concept from ResNets, now serving as the transformer's central communication channel. The student already understands F(x) + x ("editing not writing") from Series 3. Here, the attention output is the "edit" — a context-dependent correction that enriches the original embedding without erasing it.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | Same 4 tokens from Lessons 1-2 (3-dim embeddings), same W_Q, W_K, plus a new W_V. Full trace: compute V vectors, reuse attention weights from Lesson 2, compute output = weights @ V for "cat." Show the actual output numbers. | The three-lesson worked example is the pedagogical backbone. The student has traced raw attention and Q/K attention on these exact tokens. Adding V completes the picture with the same familiar numbers. The "same input, progressively richer computation" arc across three lessons is powerful. |
| Visual | Three-column comparison diagram: K vectors (what makes each token relevant), V vectors (what each token contributes), and the weighted output for "cat." Color-coded to show which V vectors contribute most (based on attention weights). | V is most intuitive when you can SEE that the key vector and the value vector contain different information. A side-by-side of K and V for the same token shows "this is what got it selected" vs "this is what it provides." The weighted output shows the blending in action. |
| Symbolic | Full formula: output = softmax(QK^T / sqrt(d_k)) V, with V = W_V X. Built as the final addition to the formula from Lesson 2. Residual: final_output = attention_output + X. Both in KaTeX. | The formula is the anchor. The student has seen it grow across three lessons: softmax(XX^T) X -> softmax(QK^T/sqrt(d_k)) X -> softmax(QK^T/sqrt(d_k)) V. Each step is a targeted replacement. The residual formula connects directly to F(x) + x from ResNets. |
| Verbal/Analogy | Extend the job fair analogy: the third card. At the job fair, you already have a "what I'm looking for" card (Q) and a "what I bring to the table" card (K). But when you're actually matched with a team, you don't hand them the "what I bring" card. You hand them your RESUME — a detailed description of your actual skills and experience (V). The "what I bring" card got you the match; the resume is what you actually contribute. | The job fair analogy has been the primary analogy across Lessons 1-2. Extending it with a third card (the resume) maintains continuity while making the K-vs-V distinction concrete. "The offering card got you noticed; the resume is what you deliver" is a natural, instantly understood distinction. |
| Intuitive | "Three lenses, one embedding." W_Q reveals what the token seeks (query lens). W_K reveals what the token advertises for matching (key lens). W_V reveals what the token actually has to say (value lens). Same embedding, three different views. The model learns all three lenses from data. | Extends the "learned lens" mental model from Lesson 2. Adding a third lens to the existing two feels like a natural completion rather than a new concept. The framing emphasizes that V is the same kind of operation as Q and K — no conceptual leap required. |
| Geometric/Spatial | Inline SVG: one embedding point projected through THREE matrices to three different destinations (Q in sky blue, K in amber, V in emerald). Same origin, three different arrows, three different endpoints. Caption: "same embedding, three lenses, three views." | Extends the two-projection SVG from Lesson 2 (which showed Q and K as two destinations). Adding V as the third makes the visual complete. The student sees that all three are the same operation — linear projection — just with different learned matrices. |

### Cognitive Load Assessment

- **New concepts in this lesson:** (1) V as a third learned projection that separates matching from contributing, (2) The residual stream concept (attention output added to input, not replacing it). That's 2 new concepts.
- **Previous lesson load:** BUILD (queries-and-keys). Two new concepts: Q/K projections and sqrt(d_k) scaling.
- **Assessment:** BUILD is appropriate. The module plan designates this as BUILD, and the analysis confirms it. The student already has the attention weight matrix at DEVELOPED depth. V is the same kind of operation as Q and K (a learned linear projection), so there's no new mechanism — just a third application of a familiar operation. The residual stream is a direct transfer of the skip connection concept from ResNets (DEVELOPED), applied to a new context. Both concepts complete an existing picture rather than introducing a new paradigm.

### Connections to Prior Concepts

- **Q and K projections** (queries-and-keys, DEVELOPED): V is "the third projection." Same operation, same framing ("learned lens"), different purpose. The student has seen two projections resolve the dual-role limitation; V resolves the last remaining dual role (matching vs contributing).
- **Weighted average** (the-problem-attention-solves, DEVELOPED): The attention output IS a weighted average — of V vectors using attention weights. "Same concept you built in Lesson 1, now applied to V instead of raw embeddings."
- **Raw attention formula: softmax(XX^T) X** (the-problem-attention-solves, DEVELOPED): The second X is what V replaces. Making this substitution explicit shows V as a targeted improvement, not a new concept.
- **Residual connections / F(x) + x** (resnets, DEVELOPED): "Remember the residual block? Attention uses the same pattern. The attention output is the 'edit'; the original embedding is the 'document.'"
- **"Editing not writing" analogy** (resnets, DEVELOPED): Directly applicable. Attention edits the embedding; it doesn't write a new one from scratch.
- **Job fair analogy** (queries-and-keys, DEVELOPED): Extended with the third card (resume/V).
- **"Learned lens" mental model** (queries-and-keys): Extended from two lenses to three.
- **"The cat chased the mouse"** example (Lessons 1-2): Used again for K-vs-V distinction. Same familiar example, new insight.
- **nn.Linear** (Series 2, DEVELOPED): V is yet another nn.Linear layer. "You now have three nn.Linear layers. That's the entire Q/K/V mechanism."

**Analogies that could be misleading:**
- The "database query/key/value" analogy from the broader ML literature. This lesson should NOT use it. In a database, the value is the record stored at a key — it's a static lookup. In attention, V is a learned projection of the input, not a stored record. The database analogy implies V is passively retrieved rather than actively computed.
- The ResNet analogy could be misleading if over-extended. In ResNet, skip connections are a training optimization (making identity easy to learn). In transformers, the residual stream is an architectural backbone — the central communication channel. The lesson should acknowledge and extend the analogy while noting the conceptual upgrade.

### Scope Boundaries

**This lesson IS about:**
- Why V exists (separating matching from contributing, the third role separation)
- What V computes (V = W_V @ embedding, a learned projection like Q and K)
- The full single-head attention formula: output = softmax(QK^T / sqrt(d_k)) V
- The residual stream concept: final_output = attention_output + x
- Tracing the full computation by hand with the same 4 tokens from Lessons 1-2
- Target depths: V projection at DEVELOPED, full single-head attention at DEVELOPED, residual stream at INTRODUCED (deeper development in Lesson 5 with the transformer block)

**This lesson is NOT about:**
- Multi-head attention (multiple Q/K/V sets in parallel) — Lesson 4
- The output projection W_O after concatenating heads — Lesson 4
- Feed-forward networks or the full transformer block — Lesson 5
- Layer normalization — Lesson 5
- Causal masking — Lesson 6
- Training attention or how W_Q, W_K, W_V are learned — Module 4.3
- Cross-attention (all attention in this module is self-attention)
- Attention as a standalone PyTorch module (nn.MultiheadAttention) — this is raw implementation
- The residual stream across multiple layers (stacking) — Lesson 5

### Lesson Outline

1. **Context + Constraints**
   - "Lessons 1 and 2 built the attention weight matrix from scratch. You know how much each token attends to every other token. This lesson answers: when a token IS attended to, what information does it contribute? And what happens to the output?"
   - Explicit scope: We add the V projection and the residual stream. We do NOT cover multi-head attention (Lesson 4) or the full transformer block (Lesson 5). By the end, you'll have implemented complete single-head attention from scratch.
   - Has notebook: `4-2-3-values-and-attention-output.ipynb`

2. **Hook: The Pattern You've Seen Before**
   - Type: Pattern recognition / callback
   - Callback to the Lesson 2 forward reference: "The key that makes a token relevant and the information it should provide when attended to are different things. Should we use the same embedding for matching AND contributing?"
   - "You've resolved this kind of question before. In Lesson 1, one embedding served as both seeking and offering — you felt the limitation, and Q/K fixed it. The same pattern appears again: K (what makes me relevant for matching) is a different role than 'what information I should provide when selected.' One vector, two roles. You know the fix."
   - Brief visual reminder: the formula from Lesson 2 — softmax(QK^T / sqrt(d_k)). "The weights are complete. But weights need something to weight."

3. **Explain: The Third Projection — V**
   - Problem before solution: In the current formula, what do we weight? If we multiply the attention weights by the raw embeddings (X), the contribution of each token to the output is its full embedding — the same vector used to compute everything else. But "what made 'chased' relevant to 'cat'" (K: action verb, matches a noun seeking its verb) is NOT the same as "what 'chased' should contribute to 'cat's representation" (V: past tense, transitivity, directed motion).
   - The fix: V = W_V @ embedding. A third learned lens that extracts the contribution signal — what each token has to say when it's attended to.
   - Job fair extension: you had two cards (seeking card = Q, offering card = K). But when matched with a team, you don't hand them the offering card. You hand them your RESUME (V). The offering card got you the match; the resume is what you actually deliver.
   - "Three lenses, one embedding:" W_Q (what am I seeking?), W_K (what do I advertise for matching?), W_V (what do I actually have to say?). Same embedding, three different views.
   - Geometric/spatial SVG: one embedding projected to three destinations (Q sky blue, K amber, V emerald). Extends the two-projection SVG from Lesson 2.
   - Connection to nn.Linear: "V is just another nn.Linear(d_model, d_v, bias=False). You now have three nn.Linear layers. That's the entire Q/K/V mechanism. No new operation — just a third application of the same building block."
   - WarningBlock: "V, like Q and K, is a property of the learned projection matrix, not the token. Different layers learn different W_V matrices, extracting different contribution signals from the same embedding."

4. **Check: Predict the K vs V Distinction**
   - "Consider 'The cat chased the mouse.' When 'chased' is attended to by 'cat,' what should K encode (for matching) vs what should V encode (for contributing)?"
   - Expected reasoning: K encodes "I'm an action verb, relevant to a noun seeking its action." V encodes different information — tense, transitivity, direction of motion. The matching signal and the contribution signal serve different purposes.

5. **Explain: The Full Formula**
   - Build incrementally, showing the evolution across all three lessons:
     - Lesson 1: output = softmax(XX^T) X — raw embeddings for everything
     - Lesson 2: weights = softmax(QK^T / sqrt(d_k)) — replaced X with Q and K for scoring
     - This lesson: output = softmax(QK^T / sqrt(d_k)) V — replaced X with V for contributing
   - "Each step is a targeted replacement. The formula's structure hasn't changed — it's still 'compute weights, then take a weighted average.' What changed is WHAT we take a weighted average of."
   - The output for token i = sum over j of (attention_weight_ij * v_j). Each token's output is its own weighted average of the V vectors, using its own attention weights.
   - Shape check: if input has n tokens with d-dim embeddings, attention weights are n x n, V is n x d_v, output is n x d_v. One output vector per token.

6. **Explain: Worked Example with Concrete Numbers**
   - Same 4 tokens ("The", "cat", "sat", "here"), same embeddings, same W_Q and W_K from Lesson 2.
   - New: concrete 3x3 W_V matrix (small values for hand-traceability).
   - Step 1: Compute V = W_V @ X. Show all 4 value vectors. Aside: "Compare these to the K vectors from Lesson 2. Different matrix, different vectors — same embeddings."
   - Step 2: Reuse the attention weights from Lesson 2 (already computed: softmax(QK^T / sqrt(3))). "We already have the weights. Now we apply them."
   - Step 3: Compute the output for "cat" — explicitly show the weighted sum: w_1 * v_The + w_2 * v_cat + w_3 * v_sat + w_4 * v_here = [output vector]. Show the actual numbers.
   - Emphasize: "Each token gets its OWN output vector. 'Cat' and 'sat' have different attention weights (different rows), so they compute different weighted averages of the SAME V vectors."
   - Brief aside: "What if W_V = W_K? Then V and K would be the same vectors, and the contribution signal would be locked to the matching signal. The model loses a degree of freedom." (Addresses misconception #2.)

7. **Elaborate: The Residual Stream**
   - Problem before solution: "We have the attention output — a context-dependent representation for each token. But does this REPLACE the original embedding? Consider a token surrounded by uninformative context (stop words, padding). Its attention output would be a bland average of uninformative V vectors. If this replaces the original embedding, the token loses its identity."
   - The fix: attention output is ADDED to the input, not substituted. final_output = attention(x) + x.
   - Callback to ResNets: "This is the same pattern you implemented in Series 3. The residual block: output = F(x) + x. 'Editing, not writing.' The attention output is the 'edit' — a context-dependent correction that enriches the original embedding."
   - Formula: residual_output_i = attention_output_i + embedding_i.
   - InsightBlock (the upgrade from ResNet): "In ResNets, skip connections are local optimization tools — they help gradients flow and make identity easy to learn. In transformers, the residual stream is the architecture's central highway. Every layer — attention and feed-forward — reads from it and writes back to it. The original embedding survives through the entire network, progressively enriched by each layer. This isn't just a training trick; it's the backbone of the architecture."
   - Concrete: "If attention learns nothing useful for token i (attention_output_i is near-zero), then residual_output_i is just embedding_i. The token keeps its original meaning. The residual stream makes 'do nothing' safe."
   - Depth note: the residual stream becomes much more important in Lesson 5 (transformer block) where we stack layers. Here, INTRODUCE the concept and connect it to ResNets. Full development later.

8. **Check: Transfer Question**
   - "A colleague removes the residual connection from their attention layer: they use output = attention(x) instead of output = attention(x) + x. What breaks?"
   - Expected reasoning: Without the residual connection, a token surrounded by uninformative context loses its identity. The original embedding information is gone. Also, in a deep model, gradients must flow entirely through the attention computation — no direct path (callback to gradient highway from ResNets). The residual connection provides both information preservation and gradient flow.

9. **Practice: Notebook** (`4-2-3-values-and-attention-output.ipynb`)
   - Scaffolding: supported (less scaffolding than Lessons 1-2 — student has implemented Q, K, and raw attention)
   - Exercise 1: Implement V projection. Given embeddings X (4 tokens, d=64) and weight matrix W_V (64x64), compute V = W_V @ X. Verify that V vectors differ from K vectors computed in the Lesson 2 notebook.
   - Exercise 2: Implement complete single-head attention. Given X, W_Q, W_K, W_V, d_k: compute Q, K, V, scores = QK^T / sqrt(d_k), weights = softmax(scores), output = weights @ V. Wrap in a function `single_head_attention(X, W_Q, W_K, W_V)`.
   - Exercise 3: Add the residual connection. Compute final_output = attention_output + X. Verify shape matches input shape.
   - Exercise 4: Experiment — set W_V = identity matrix. Compare the output to softmax(QK^T/sqrt(d_k)) @ X (no V projection). They should be identical. This proves V generalizes the raw-embedding output from Lesson 1.
   - Exercise 5: Experiment — set W_V = W_K. Compare the output to when W_V is independent. Observe that with W_V = W_K, the contribution signal is locked to the matching signal.
   - Stretch exercise: Load GPT-2 pretrained weights. Extract one attention head's W_Q, W_K, W_V. Run a sentence through and visualize the output. Compare the V vectors to the K vectors for the same tokens — are they similar or different?

10. **Summarize**
    - Three key takeaways:
      1. V is a third learned projection that separates "what makes me relevant" (K) from "what information I contribute when attended to" (V). Three lenses, one embedding, three different views.
      2. The full single-head attention formula: output = softmax(QK^T / sqrt(d_k)) V. Each step in this formula exists because the version without it was insufficient.
      3. The attention output is ADDED to the original embedding via the residual stream — "editing, not writing." The original information is always preserved.
    - Echo the three-lesson arc: "You built this formula piece by piece. Lesson 1: raw dot-product attention (felt the limitation). Lesson 2: Q and K for asymmetric matching (felt the scaling problem, fixed it). This lesson: V for contribution, residual stream for preservation. Single-head attention is complete."
    - But single-head attention computes ONE notion of relevance. "The cat sat on the mat because it was soft" — "it" needs to attend to "mat" for coreference AND to "soft" for meaning. One set of Q/K/V weights can only capture one type of relationship at a time.

11. **Next Step**
    - "Next: Multi-Head Attention — running multiple attention operations in parallel, each with its own Q/K/V, each capturing a different type of relationship."
    - The student should feel: "Single-head attention is complete. But one head isn't enough."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth (9 concepts in prereq table)
- [x] Each traced via the records (queries-and-keys record, the-problem-attention-solves record, Module 4.1 record, Module 3.2 record)
- [x] Depth match verified for each (all OK)
- [x] No untaught concepts remain (no gaps)
- [x] No multi-concept jumps in exercises (notebook builds incrementally on Lessons 1-2 notebooks, same style, less scaffolding)
- [x] All gaps have explicit resolution plans (N/A — no gaps)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (the "one vector, two roles" pattern recognized for the third time; residual stream as the completion of the attention layer)
- [x] At least 3 modalities planned: concrete example (4-token trace), visual (K/V comparison, 3-projection SVG), symbolic (formula evolution across 3 lessons), verbal/analogy (job fair resume card), intuitive (three lenses), geometric/spatial (three-destination SVG) — 6 modalities
- [x] At least 2 positive examples + 1 negative example: 3 positive (full trace, K-vs-V semantic, identity W_V) + 2 negative (uninformative context without residual, W_V = W_K)
- [x] At least 3 misconceptions: 5 identified with negative examples (V unnecessary, V/K interchangeable, attention replaces embedding, residual stream = just skip connection, attention produces single vector)
- [x] Cognitive load = 2 new concepts (V projection + residual stream) — within limit
- [x] Every new concept connected to existing concepts (V -> Q/K projections + nn.Linear + "learned lens"; residual stream -> ResNet skip connections + "editing not writing")
- [x] Scope boundaries explicitly stated (no multi-head, no transformer block, no layer norm, no causal masking)

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues found. The lesson is well-structured, pedagogically sound, and faithfully implements the planning document. The three-lesson arc continuity (same tokens, same matrices) is excellent. The narrative motivates V before introducing it, the residual stream section is well-motivated with the "uninformative context" negative example, and the formula evolution across three lessons is clear and powerful. However, three improvement-level findings would make the lesson significantly stronger if addressed.

### Findings

#### [IMPROVEMENT] — Identity-matrix positive example missing from lesson body

**Location:** Worked example section (Section 6) and the lesson body generally
**Issue:** The planning document includes a positive example: "W_V = identity matrix: show that this reduces to Lesson 1's formula." This is described as a positive example that "proves V generalizes the raw-embedding output from Lesson 1" and is listed as the third positive example in the examples table. In the built lesson, this example appears ONLY in the notebook exercise list (Exercise 4, line 1024) but is never shown or discussed in the lesson prose itself. The W_V = W_K negative example IS included in the lesson body (the amber callout box at lines 839-861), but the identity-matrix generalization is not.
**Student impact:** The student misses a key conceptual connection: V is not adding something new on top of Q/K, it is *generalizing* the raw-embedding output from Lesson 1. When W_V = I, the formula collapses to what they had before. This is the "aha" that ties all three lessons together into a single coherent story. Without it in the lesson body, the student encounters it for the first time in the notebook, where it arrives as a procedural exercise rather than a conceptual insight.
**Suggested fix:** Add a brief paragraph or callout in the worked example section (after showing the W_V = W_K negative example, or as a second callout) that states: "What if W_V were the identity matrix? Then V = I * embedding = the raw embedding itself. The output becomes softmax(QK^T / sqrt(d_k)) * X, which is exactly Lesson 1's formula with better scoring. V generalizes the raw-embedding output; identity is a special case." This can be concise; the notebook exercise then provides hands-on verification.

#### [IMPROVEMENT] — Misconception #5 (attention produces single output vector) addressed weakly

**Location:** Worked example section (Step 3, lines 808-812) and aside (lines 829-835)
**Issue:** The planning document identifies misconception #5: "Attention produces a single output vector for the whole sequence." The lesson addresses this primarily through a single sentence ("Each token gets its own output vector. 'Cat' and 'sat' have different attention weights...") and an InsightBlock aside ("The output has the same shape as the input: one vector per token"). This is correct but passive. The planning document says to "Emphasize" this point at the worked example, but there is no explicit negative framing that disproves the misconception. The student who holds this misconception never sees why their mental model is wrong; they only see the correct model stated.
**Student impact:** A student who initially pictures attention as "combine everything into one summary" (a reasonable initial intuition) gets a corrective statement but not a moment of "oh, I was wrong because..." The correction may not stick because it is stated as a fact rather than demonstrated by contrast.
**Suggested fix:** Add 1-2 sentences that directly name and disprove the misconception: "You might picture attention as producing a single summary vector for the whole sequence. It does not. Each token gets its own output, because each token has its own row of attention weights. The output has the same shape as the input: 4 tokens in, 4 vectors out." The "all 4 output vectors" display (lines 815-826) is already there and does the right thing; just add the explicit misconception framing before it.

#### [IMPROVEMENT] — Residual stream aside overreaches for INTRODUCED depth

**Location:** InsightBlock aside in the Residual Stream section (lines 932-941)
**Issue:** The planning document targets the residual stream at INTRODUCED depth, with "full development later" in Lesson 5. However, the InsightBlock aside makes strong architectural claims: "the residual stream is the architecture's central highway," "every layer reads from it and writes back to it," "the original embedding survives through the entire network, progressively enriched by each layer." These statements describe multi-layer stacking behavior that the student has not yet encountered. The student knows about single-head attention on one layer. The scope boundary explicitly says "NOT: the residual stream across multiple layers (stacking)."
**Student impact:** The student reads about layers reading and writing to a central highway, but has no context for what "layers" means in the transformer sense (Lesson 5 topic). The InsightBlock goes beyond what the student can ground. It is not *wrong*, but it plants ideas the student cannot yet verify, which weakens the "INTRODUCED" depth claim (it reads more like a preview of DEVELOPED content from Lesson 5).
**Suggested fix:** Scale the InsightBlock back to what the student can ground: "In ResNets, skip connections help gradients flow and make identity easy to learn. In the transformer, the residual stream does the same, but it also serves a deeper role: it preserves the original embedding while attention enriches it with context. You will see in Lesson 5 how this becomes the backbone of the entire architecture when layers stack." This preserves the seed for Lesson 5 without asserting multi-layer behavior as if the student already understands it.

#### [POLISH] — Typo in summary block: "Unsacled"

**Location:** SummaryBlock, second item description (line 1092)
**Issue:** "Unsacled scores" should be "Unscaled scores."
**Student impact:** Minor distraction. The summary is a key reference point the student may revisit; a typo in the formula evolution narrative slightly undermines confidence.
**Suggested fix:** Change "Unsacled" to "Unscaled."

#### [POLISH] — Shape check uses d_v without introduction

**Location:** Shape check block (lines 594-601)
**Issue:** The shape check introduces `d_v` as a dimension notation: "V: n x d_v" and "Output: n x d_v." However, d_v has not been defined or discussed in the lesson. The PyTorch code snippet (line 427) uses `d_v` as the output dimension of W_V, but it appears without explanation. The worked example uses 3x3 matrices (d_k = d_v = 3), so the student has no reason to distinguish d_v from d_k.
**Student impact:** The student may wonder whether d_v is always equal to d_k, or if it can differ, or what it even refers to. This is a minor loose end since the lesson does not discuss the distinction. The student can likely infer that d_v is the output dimension of W_V, but it is never stated.
**Suggested fix:** Either (a) add a brief parenthetical: "V: n x d_v (where d_v is the output dimension of W_V; often d_v = d_k)" or (b) use d_k consistently since the worked example uses the same dimension for all projections, and note that d_v can differ in multi-head attention (Lesson 4).

#### [POLISH] — Em-dash with thin spaces in shape check line

**Location:** Shape check block, line 599
**Issue:** The output line uses `&ensp;&mdash;&ensp;` (en-space, em-dash, en-space), which is technically spaces around the em-dash. The Writing Style Rule says em dashes must have no spaces. This is in a monospace formatting block rather than prose, so it is borderline, but it is inconsistent with the rest of the lesson.
**Student impact:** None. Purely a consistency issue.
**Suggested fix:** Change to `&mdash;` without the surrounding `&ensp;` entities, or accept this as a formatting exception for the monospace block.

### Review Notes

**What works well:**
- The three-lesson continuity (same 4 tokens, same matrices, adding W_V) is excellent pedagogy. The student sees the computation grow across three lessons with familiar numbers.
- The hook (Section 2) perfectly resolves the cliffhanger from Lesson 2. The callback to the forward reference is natural and effective.
- The "one vector, two roles, you know the fix" framing leverages the student's experience with Q/K resolution. The pattern-recognition aside captures this nicely.
- The formula evolution display (Section 5) with three stacked formulas showing Lesson 1 -> Lesson 2 -> This Lesson is the clearest possible way to show V as a targeted replacement.
- The job fair resume extension is a natural, instantly understandable extension of the existing analogy.
- The residual stream motivation (uninformative context erasing token identity) is well-motivated with the problem-before-solution pattern.
- The "What if W_V = W_K?" negative example (lines 839-861) is well-placed and concise.
- The two check exercises (K-vs-V prediction and residual stream transfer question) are well-designed with appropriately detailed reveals.
- Modality count is strong: concrete example (4-token trace), visual (three-projection SVG, K-vs-V side-by-side), symbolic (formula evolution, KaTeX), verbal/analogy (job fair resume), intuitive (three lenses framing), geometric (SVG). Six modalities present.
- Scope boundaries are clean; the lesson does not drift into multi-head territory.

**Patterns observed:**
- The lesson has a slight back-heavy tendency: the most conceptually dense section (residual stream) comes near the end after a lengthy worked example. The student's cognitive energy may be lower by that point. This is inherent to the topic order (V must come before residual stream) but worth noting.
- The notebook exercise list (6 exercises plus a stretch) is substantial. This is appropriate given that this is the "complete single-head attention" milestone, but the student should feel that Exercises 1-3 are the core and 4-5 are experiments.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All 6 findings from Iteration 1 (3 improvement, 3 polish) have been correctly applied:

1. **Identity-matrix positive example** — Now present as an emerald callout box (lines 865-878) in the worked example section, right after the W_V = W_K negative example. The callout explains that V = I * embedding = raw embedding, reducing to Lesson 1's formula. The notebook verification is mentioned. Well-placed and concise.
2. **Misconception #5 (single output vector)** — Now explicitly addressed with direct misconception framing at lines 808-815: "You might picture attention as producing a single summary vector for the whole sequence — combining all tokens into one representation. It does not." The "four tokens in, four vectors out" phrasing is clear. Well-integrated.
3. **Residual stream aside overreaches** — InsightBlock now scoped to what the student can ground: mentions ResNet benefits (gradients, identity), adds that the residual stream also preserves the original embedding, and defers multi-layer architecture to "the next lessons." No longer asserts multi-layer behavior the student hasn't encountered.
4. **Typo "Unsacled"** — Fixed to "Unscaled" in the summary block.
5. **Shape check d_v** — Now includes parenthetical: "V: n x d_v (output dimension of W_V; often d_v = d_k)."
6. **Em-dash spacing in shape check** — Fixed; no surrounding spaces.

### Findings

#### [POLISH] — Possessive after quoted token names creates visual clutter

**Location:** Lines 338, 512, 518, 782, 783 (five instances throughout the lesson)
**Issue:** The pattern `&ldquo;cat&rdquo;&rsquo;s` renders as `"cat"'s` — a right double-quote immediately followed by a right single-quote (apostrophe). While typographically correct, the two curly marks collide visually, creating a cluttered look at each occurrence.
**Student impact:** Minor visual friction when reading. Does not affect comprehension.
**Suggested fix:** Rephrase to avoid the possessive after quoted tokens. For example, `"cat"'s output` could become `the output for "cat"` or `the "cat" output`. Apply selectively — some instances may read fine as-is.

### Review Notes

**What works well (iteration 2 perspective):**
- The iteration 1 fixes are well-integrated and do not feel bolted on. The identity-matrix callout reads naturally alongside the W_V = W_K callout. The misconception #5 framing flows into the existing "all 4 output vectors" display without disruption. The residual stream InsightBlock revision maintains the conceptual seed while respecting the INTRODUCED depth target.
- The lesson as a whole is strong. The three-lesson arc (same 4 tokens, same matrices, progressively richer computation) is the standout pedagogical achievement. The narrative is well-motivated, the examples are concrete and traceable, and the scope boundaries are clean.
- The two check exercises (K-vs-V prediction and residual stream transfer) are well-designed with clear thinking prompts and detailed reveals.
- Six modalities present: concrete example, visual (SVG + side-by-side K/V comparison), symbolic (formula evolution, KaTeX), verbal/analogy (job fair resume), intuitive (three lenses), geometric/spatial (three-projection SVG). Strong modality coverage.
- All 5 planned misconceptions are addressed at appropriate locations with concrete examples or framing.
- The typecheck and lint pass cleanly.

**Overall assessment:** The lesson is ready to ship. The single remaining finding is a cosmetic polish item that does not affect pedagogical effectiveness. No critical or improvement-level issues remain.
