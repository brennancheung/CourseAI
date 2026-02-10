# Module 4.2: Attention & the Transformer — Record

**Goal:** The student understands how attention works (from raw dot products through full multi-head Q/K/V attention), why each piece exists by feeling the limitation of the version without it, and how the transformer block assembles attention with feed-forward networks, residual connections, and layer norm into the architecture behind modern LLMs.
**Status:** Complete (6 of 6 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Context-dependent representations as a goal (static embeddings can't distinguish "bank" (river) from "bank" (money); the model needs representations that change based on surrounding tokens) | DEVELOPED | the-problem-attention-solves | Motivated via polysemy callback from embeddings-and-position (amber warning box). ComparisonRow: "bank was steep and muddy" vs "bank raised interest rates." Student understands WHY context-dependent representations are needed. |
| Weighted average as a mechanism for context (blend all embeddings with weights that sum to 1; different weights per token = different context summaries) | DEVELOPED | the-problem-attention-solves | Built through three escalating attempts: (1) uniform average (bag-of-words, fails), (2) weighted average (who decides the weights?), (3) data-dependent weights via dot products. Explicit definition of weighted average with formula and concrete interpretation ("if w_3 is large, the output is mostly embed_3"). |
| Dot-product attention on raw embeddings (Attention(X) = softmax(XX^T) X — similarity scores, softmax normalization, weighted averaging) | DEVELOPED | the-problem-attention-solves | Full step-by-step worked example: 4 tokens with 3-dim embeddings. All pairwise dot products computed explicitly (4x4 matrix), softmax applied row-wise, weighted average computed for "cat." Formula built incrementally: XX^T -> softmax -> multiply by X. Student traces every number. |
| Data-dependent weights as the paradigm shift (attention weights are freshly computed from each input, not fixed parameters like CNN filters) | DEVELOPED | the-problem-attention-solves | Explicitly contrasted with CNN fixed filters (InsightBlock). "The weight matrix W is not a learned parameter. It's freshly computed from every new input X." This is framed as the conceptual revolution of attention. |
| Dot product as similarity measure (a dot b = sum of element-wise products; large positive when vectors point same direction, zero when perpendicular, negative when opposite) | DEVELOPED | the-problem-attention-solves | Gap resolution from INTRODUCED depth. Three-panel geometric SVG (similar direction, perpendicular, opposite) plus numerical examples with 3-dim vectors. Connected to cosine similarity: dot product = cosine similarity x magnitudes. Formula: a dot b = \|a\| \|b\| cos(theta). |
| Attention weight matrix (square matrix where row i contains token i's attention weights over all tokens; each row sums to 1) | DEVELOPED | the-problem-attention-solves | Computed explicitly in worked example. Interactive AttentionMatrixWidget: student types sentences and sees heatmap in real time. Hover shows exact values. Toggle between raw scores and softmax weights. Preset sentences for guided exploration. |
| All-pairs computation in attention (every token computes a score with every other token regardless of distance — no locality window) | DEVELOPED | the-problem-attention-solves | Explicit contrast with CNN local receptive fields. "it" -> "mat" long-range dependency example (6 positions apart). WarningBlock: "Attention computes all-pairs scores in a single step — a token can attend just as strongly to a word 20 positions away as to its immediate neighbor." |
| Dual-role limitation of raw embeddings (one embedding vector must serve as both "what I'm seeking" and "what I'm offering") | INTRODUCED | the-problem-attention-solves | Cocktail party analogy (what you search for vs what you advertise). ComparisonRow: "When This Token Is Looking" vs "When Other Tokens Look At It." The limitation is clearly stated but not resolved — resolution deferred to queries-and-keys. |
| Attention matrix symmetry as a design flaw (raw dot-product scores are always symmetric because a dot b = b dot a; real linguistic relationships are asymmetric) | INTRODUCED | the-problem-attention-solves | "The cat chased the mouse" example: "cat" needs "chased" for what it DID, "chased" needs "cat" for WHO did it. Same pair, different reasons, same score. WarningBlock clarifies: raw scores symmetric, but softmax weights not perfectly symmetric (different row denominators). |
| Similarity vs relevance distinction (dot products measure how similar two embeddings are, but relevance — "does this word help me understand the other?" — is a different question) | INTRODUCED | the-problem-attention-solves | Negative example: "The bank was steep" — "bank" and "steep" have low embedding similarity (different semantic clusters) but "steep" is highly relevant to disambiguating "bank." "Interest" is more similar to "bank" than "steep" is, even when "steep" is the word that disambiguates. |
| Q and K as learned linear projections (W_Q and W_K transform the same embedding into separate "seeking" and "offering" vectors, breaking the dual-role limitation) | DEVELOPED | queries-and-keys | Core concept. Job fair analogy: two cards per person ("what I'm looking for" and "what I bring to the table"). Geometric SVG: same embedding projected through two different matrices to two different points. Connected to nn.Linear from Series 2 ("just nn.Linear(d_model, d_k, bias=False)"). "Learned lens" framing: same input, different matrix, different view. Full hand-traced worked example with 4 tokens, 3x3 matrices. WarningBlock: Q and K are properties of the learned matrices, not the token. |
| QK^T as a learned asymmetric relevance matrix (entry (i,j) = q_i dot k_j measures "how much does token i's query match token j's key?") | DEVELOPED | queries-and-keys | Replaces XX^T from Lesson 1. Not similarity between embeddings — a learned relevance function. Explicitly shown why it's asymmetric: q_i = W_Q x_i and k_j = W_K x_j, with W_Q != W_K, so q_i dot k_j != q_j dot k_i. Side-by-side heatmap comparison (raw symmetric vs projected asymmetric) is the visual centerpiece. "Bank"/"steep" callback: softened to "in a trained model, W_Q and W_K would learn projections that make the dot product large." |
| Scaling by sqrt(d_k) to prevent softmax saturation (dividing QK^T scores by sqrt(d_k) normalizes variance from d_k back to 1, keeping softmax in a trainable range) | DEVELOPED | queries-and-keys | Motivated by dimension growth: d_k=3 gives dot products ~1.7 (fine), d_k=512 gives ~22.6 (catastrophic). Callback to temperature slider from what-is-a-language-model (large inputs = low temperature = one-hot softmax). Callback to vanishing gradients from training-dynamics (softmax near 0/1 = zero gradient = no learning). Concrete comparison: without scaling, softmax([22,-15,3,8]) -> [1.0, 0.0, 0.0, 0.0]. With scaling, variance normalized to 1. Not cosmetic — essential for training. |
| Scaled dot-product attention formula: weights = softmax(QK^T / sqrt(d_k)) | DEVELOPED | queries-and-keys | Built incrementally: (1) project Q = W_Q X, (2) project K = W_K X, (3) score S = QK^T / sqrt(d_k), (4) normalize W = softmax(S). Compared to Lesson 1's softmax(XX^T): same structure, two additions (projections + scaling). Formula is the attention WEIGHTS only — output computation (with V) deferred to Lesson 3. |
| Linear projection as a learned transformation (Wx as a "lens" that emphasizes certain features of the input) | INTRODUCED | queries-and-keys | Gap resolution from implicit nn.Linear knowledge. Framed as "the matrix is the lens; same input, different lens, different view." Geometric SVG showing same point projected to two different locations. Connected to nn.Linear: "same building block you've been using since Module 2.1." Not a separate section — integrated into Q/K explanation. |
| Softmax saturation (extreme inputs push softmax toward one-hot, killing gradients) | DEVELOPED | queries-and-keys | Gap resolution from INTRODUCED depth (temperature slider in what-is-a-language-model). Bridge: "Remember how T=0.1 made softmax concentrate nearly all mass on one token? Large dot products have the same effect." Concrete numerical example showing softmax outputs with small vs large inputs. Connected to vanishing gradients: near-0/1 outputs = near-zero gradient = frozen model. Elevated from INTRODUCED to DEVELOPED with quantitative treatment. |
| V as a third learned projection separating matching from contributing (V = W_V x embedding extracts what a token contributes when attended to, distinct from what makes it relevant for matching via K) | DEVELOPED | values-and-attention-output | Core concept. "One vector, two roles" pattern recognized for the third time (Lesson 1: seeking/offering, Lesson 2: Q/K, this lesson: matching/contributing). Job fair analogy extended: K = offering card that gets you the match, V = resume you hand over when matched. Three-lenses framing: W_Q (seeking), W_K (advertising), W_V (contributing). Geometric SVG with three projection arrows from one embedding point. PyTorch: "just another nn.Linear(d_model, d_v, bias=False)." WarningBlock: V is a property of the learned projection matrix, not the token. K-vs-V side-by-side comparison with concrete numbers. |
| Full single-head attention formula: output = softmax(QK^T / sqrt(d_k)) V (complete single-head attention as a weighted average of V vectors using learned relevance weights) | DEVELOPED | values-and-attention-output | Built as the culmination of a three-lesson arc. Formula evolution shown explicitly: Lesson 1 softmax(XX^T)X -> Lesson 2 softmax(QK^T/sqrt(d_k)) for weights -> this lesson softmax(QK^T/sqrt(d_k))V for output. "Each step is a targeted replacement. The structure hasn't changed — it's still compute weights, then average. What changed is what we score with (Q/K) and what we average (V)." Per-token output emphasized: each token gets its own output vector. Shape check: n tokens in, n vectors out. Misconception #5 explicitly addressed: attention does NOT produce a single summary vector. |
| Attention output as a per-token weighted average of V vectors (output_i = sum_j weight_ij * v_j; each token computes its own weighted average using its own row of attention weights) | DEVELOPED | values-and-attention-output | Full hand-traced worked example for "cat" using same 4 tokens from Lessons 1-2. All 4 output vectors shown. Explicit misconception correction: "You might picture attention as producing a single summary vector. It does not. Each token gets its own output because each token has its own row of attention weights." |
| V generalizes raw-embedding output (when W_V = identity, V = raw embedding and the formula reduces to softmax(QK^T/sqrt(d_k))X from Lesson 1 with better scoring) | INTRODUCED | values-and-attention-output | Emerald callout box in worked example section. "V is not replacing something — it's generalizing it. Identity is a special case." Notebook exercise 4 provides hands-on verification. |
| Matching signal vs contribution signal locked when W_V = W_K (if V and K use the same matrix, the weighted average blends what made tokens relevant rather than what they have to say) | INTRODUCED | values-and-attention-output | Amber callout box after worked example. "The model loses a degree of freedom: the matching signal and the contribution signal are locked together." Notebook exercise 5 provides hands-on verification. |
| Residual stream in attention (attention output is ADDED to the input embedding, not substituted: final_output = attention_output + embedding) | INTRODUCED | values-and-attention-output | Motivated by information loss: token in uninformative context would lose identity if attention output replaces embedding. The fix: add, don't replace. Direct callback to ResNet residual blocks: F(x) + x, "editing not writing." Concrete worked example: residual output for "cat" = attention_output + embedding. "If attention learns nothing useful, the token keeps its original meaning." InsightBlock notes this extends ResNet skip connections but defers multi-layer architecture role to Lesson 5. Transfer question: what breaks without the residual connection? (information loss + gradient flow). Depth is INTRODUCED — full development of residual stream across stacked layers deferred to the-transformer-block (Lesson 5). |
| Multiple attention heads operating in parallel on lower-dimensional subspaces (h independent heads, each with its own W_Q^i, W_K^i, W_V^i, running the same single-head formula in a d_k-dimensional subspace) | DEVELOPED | multi-head-attention | Core concept. Motivated by the "it" pronoun problem: one head can only capture one notion of relevance, but "it" needs coreference ("mat") and property attribution ("soft") simultaneously. Research team analogy: specialists reading through their own lens, pooling findings. Side-by-side hypothetical attention patterns on "The cat sat on the mat because it was soft" (Head 1: coreference, Head 2: property attribution). Explicit: heads share input X but have completely independent learned weights. Full worked example with 4 tokens, d_model=6, h=2, d_k=3 — two heads produce genuinely different attention patterns on the same input. Guided interpretation: "here" attends to "cat" (0.586) in Head 1 but to itself (0.628) in Head 2. |
| Dimension splitting in multi-head attention (d_k = d_model / h as budget allocation, not compute multiplication; total FLOPs across all heads = single-head FLOPs) | DEVELOPED | multi-head-attention | The question "do h heads need h times the compute?" answered with no. Each head's W_Q^i is (d_model, d_k) instead of (d_model, d_model). GPT-2 example: d_model=768, h=12, d_k=64. SVG visualization: wide rectangle split into 12 colored strips (same total width). Compute equivalence derived: h * n^2 * d_k = n^2 * d_model. Misconception "more heads = better" addressed with extreme case: d_k=1 per head is a single scalar multiplication, absurdly limited. Tradeoff: more heads = more diverse perspectives but less expressive power per head. Prediction exercise: d_model=512, h=8 — compute d_k, W_O parameter count, effect of adding a 9th head (512 doesn't divide evenly by 9). |
| Output projection W_O as learned cross-head mixing (d_model x d_model matrix applied after concatenation; mixes information across heads, not just reshaping) | DEVELOPED | multi-head-attention | Concatenation recovers shape (n, h*d_k) = (n, d_model), but each head's contribution stays isolated in its d_k slice. W_O fixes this: learned mixing layer that lets head 3's findings influence dimensions head 1 wrote to. Full formula: MultiHead(X) = Concat(head_1, ..., head_h) W_O. Misconception "W_O is just reshaping" explicitly addressed: W_O has d_model^2 parameters and is a learned nn.Linear. Without W_O, heads file reports but never meet to discuss; with W_O, findings are synthesized. Shape walkthrough: input (n, d_model) -> per head (n, d_k) -> concat (n, d_model) -> W_O (n, d_model). "The team meeting" analogy. |
| Head specialization patterns in trained models (messy and emergent, not designed and clean; some heads attend to previous token, some track syntax, some are nearly uniform; 20-40% can be pruned) | INTRODUCED | multi-head-attention | Addresses misconception that each head has a clean linguistic role (head 1 = syntax, head 2 = coreference). Research findings: positional patterns, syntactic tracking, near-uniform heads, most don't map to any single function. Callback to CNN filters: architecture provides capacity, training determines what each learns. "The difference between a learned pattern and a designed pattern." InsightBlock: "Capacity, not assignment." Notebook stretch exercise: load GPT-2 weights, extract attention weights for all 12 heads, visualize heatmaps, identify positional vs semantic patterns. |
| Multi-head attention formula: MultiHead(X) = Concat(head_1, ..., head_h) W_O where head_i = Attention(X W_Q^i, X W_K^i, X W_V^i) | DEVELOPED | multi-head-attention | Complete formula with full shape annotations. Built incrementally: each head runs single-head attention in its d_k subspace, outputs concatenated, W_O applied. Full worked example: 4 tokens, d_model=6, h=2, d_k=3. All values computed at load time and displayed inline. Residual connection shown: final = MHA_output + input_embedding. Transfer question: compare single-head (d_model=256, h=1) vs multi-head (d_model=256, h=4) on parameter count and relationship diversity. |
| Layer normalization (normalize across features within a single token; contrast with batch norm's cross-example normalization) | INTRODUCED | the-transformer-block | Motivated by batch norm's failure for variable-length sequences (5th token of different sentences has no shared statistics). Formula identical to batch norm but different axis: per-token mean/variance across d_model features, learned gamma/beta. Key differences from batch norm: independent of batch size, no train/eval distinction, each token normalized independently. ComparisonRow: batch norm (column-wise, batch-dependent, train/eval split) vs layer norm (row-wise, independent, same behavior always). |
| Pre-norm vs post-norm ordering (LayerNorm placement inside vs outside the residual branch) | INTRODUCED | the-transformer-block | Post-norm (original 2017 paper): x' = LayerNorm(x + MHA(x)), norm sits on the residual stream. Pre-norm (modern standard): x' = x + MHA(LayerNorm(x)), norm inside the branch, stream stays clean. ComparisonRow comparison. Gradient argument: post-norm forces gradients through 24 layer norms in a 12-block model; pre-norm gives a clean additive path. GPT-2, GPT-3, LLaMA all use pre-norm. |
| FFN structure and role in transformer block (two-layer network with 4x expansion: d_model -> 4*d_model -> d_model, GELU activation, "writes" to the residual stream) | DEVELOPED | the-transformer-block | Formula: FFN(x) = W_2 * GELU(W_1 x + b_1) + b_2. GPT-2 concrete dimensions: 768 -> 3072 -> 768. Parameter count: ~4.7M per block (2x attention's ~2.4M). 4x expansion creates higher-dimensional workspace for complex feature computations. GELU callback to activation-functions-deep-dive decision guide. Research reference: Geva et al. FFN neurons as key-value memories storing factual knowledge. |
| "Attention reads, FFN writes" mental model (attention gathers context from other tokens, FFN processes and transforms what attention found; complementary roles in the transformer block) | DEVELOPED | the-transformer-block | Core mental model of the lesson. Attention reads from the residual stream (weighted average of V vectors = linear blending, inside convex hull). FFN writes back (nonlinear transformation via GELU breaks the convex hull constraint, enables genuinely new representations). Without FFN: attention can only route and blend, not transform. Without attention: FFN processes each token independently, no cross-token communication. Both needed. |
| Residual stream as cross-layer backbone (the residual stream flows from embedding to output through every sub-layer in every block; central highway of the entire transformer) | DEVELOPED | the-transformer-block | Upgraded from INTRODUCED (values-and-attention-output). Two residual connections per block (one around MHA, one around FFN) = 24 in a 12-block GPT-2. "Shared document" analogy extended: starts as raw embedding, each of 24 sub-layers reads and annotates it. Concrete negative example: without residual connections, early-training near-uniform attention destroys token identity (near-uniform average replaces original embedding). With residual: original embedding flows through untouched, attention/FFN contributions are additive deltas. Gradient flow: 24 residual additions provide direct gradient highway from output to input. |
| Transformer block as the repeating unit (MHA + FFN + 2 residual connections + 2 layer norms; shape-preserving: (n, d_model) in, (n, d_model) out; stacks identically N times) | DEVELOPED | the-transformer-block | Complete block formula: x' = x + MHA(LayerNorm(x)), output = x' + FFN(LayerNorm(x')). TransformerBlockDiagram SVG: vertical flow with color-coded components (violet residual stream, sky blue MHA, amber FFN, emerald LayerNorm), dimension annotations, two branch-and-merge residual paths, pre-norm placement visible. StackedBlocksDiagram SVG: zoom-out showing Block 1 -> Block 2 -> ... -> Block N with residual stream backbone. Shape preservation enables stacking. GPT-2: 12 blocks, GPT-3: 96 blocks. |
| FFN as the source of nonlinearity in the transformer block (without FFN, attention output is a weighted average of linearly projected vectors — always inside the convex hull of inputs; GELU breaks this constraint) | DEVELOPED | the-transformer-block | Concrete negative example: three tokens at positions A, B, C in 768-dim space. Attention can only produce points inside the triangle (convex hull). FFN's GELU pushes representations to entirely new regions. Without nonlinearity, stacking blocks adds no expressive power. This is why ~2/3 of parameters are in FFN layers — they store the model's learned knowledge. |
| Parameter distribution in transformer models (~1/3 attention, ~2/3 FFN per block; GPT-2 124M: ~28M attention, ~57M FFN, ~38M embeddings) | DEVELOPED | the-transformer-block | Hook puzzle: student expected most parameters in attention after 4 attention-focused lessons. Reveal: FFN has 2x the parameters of attention per block. Per-block breakdown: FFN = 2 * d_model * d_ff = 4,718,592; Attention (Q+K+V+O) = 4 * d_model^2 = 2,359,296. Reframes the transformer from "a bunch of attention" to "a read-process cycle where most capacity is in the processing (FFN)." |
| Causal masking (set attention scores for future positions to negative infinity before softmax; lower-triangular attention matrix; prevents data leakage during parallel training) | DEVELOPED | decoder-only-transformers | Motivated by "the cheating problem": bidirectional attention lets position 3 copy the answer from position 4 instead of predicting it. Exam analogy: cardboard sleeve covers future answers. Mechanism: set j > i entries to -infinity, softmax produces 0, remaining entries renormalize automatically. Worked example for "sat" (row 3). Not a training trick like dropout -- mirrors inference reality where future tokens do not exist. |
| Full GPT architecture end-to-end (token embedding + positional encoding -> N transformer blocks with causal masking -> final layer norm -> output projection -> softmax -> next-token probabilities) | DEVELOPED | decoder-only-transformers | Assembly of all components from Modules 4.1 and 4.2. GptArchitectureDiagram SVG: vertical flow with color-coded components (violet residual stream, sky blue blocks, amber output projection, emerald layer norm, purple embedding+PE), dimension annotations at every stage, weight tying annotation. GPT-2 configuration box: vocab=50257, d_model=768, layers=12, heads=12, d_ff=3072, context=1024. |
| Output projection / unembedding (nn.Linear mapping d_model to vocabulary size, producing logits for next-token prediction) | INTRODUCED | decoder-only-transformers | Gap resolution from MENTIONED (embeddings-and-position aside). The reverse of embedding: 768-dim hidden state -> 50,257-dim logits -> softmax -> probabilities. Connection to MNIST output layer: "same idea, 10 digit classes -> 50K token classes." Weight tying: embedding matrix (V, d_model) shared with output projection (d_model, V) transposed, saving ~38M parameters. |
| GPT-2 total parameter counting (embedding + position embedding + 12 blocks + final layer norm = ~124.4M, verified against known figure) | DEVELOPED | decoder-only-transformers | Per-component breakdown: token embeddings 38.6M, position embeddings 786K, 12 blocks at ~7.1M each = 85.0M, final layer norm 1.5K, output projection weight-tied = 0 additional. Total matches known GPT-2 124M figure. Distribution: embeddings ~31%, attention ~23%, FFN ~46%. Bias terms (~83K) omitted as negligible. |
| Encoder-decoder vs decoder-only architectural distinction (original 2017 Transformer had encoder + decoder; GPT keeps only the decoder stack with causal masking) | INTRODUCED | decoder-only-transformers | Three variants presented: encoder-only (BERT, bidirectional, understanding tasks), encoder-decoder (T5, two stacks + cross-attention, sequence-to-sequence), decoder-only (GPT, causal masking, generation AND understanding). "Decoder" is a historical name meaning "uses causal masking," not "can only decode." GPT models understand through the same mechanism they generate with. |
| Why decoder-only won for LLMs (simplicity, scaling, generality) | INTRODUCED | decoder-only-transformers | One stack, one attention type, one training objective (next-token prediction). Scaling laws showed decoder-only models improve predictably with scale. One model handles generation AND understanding. "The simplest architecture that works is the one that scales." GPT-2 (124M) vs GPT-3 (175B) comparison table: same architecture, different scale (1400x more parameters). |
| BERT as an encoder-only transformer | MENTIONED | decoder-only-transformers | Name-dropped in three-variant comparison. Bidirectional attention (no causal mask). Great for understanding tasks (classification, NER). Cannot generate text autoregressively. Not explained in detail. |
| Training vs inference asymmetry in autoregressive models (training is parallel with causal masking; inference is sequential because future tokens do not exist) | DEVELOPED | decoder-only-transformers | ComparisonRow: training (entire sequence in parallel, all positions predict simultaneously, N training examples per sequence) vs inference (tokens generated one at a time, each appended to context, no mask needed). Causal masking makes parallel training safe -- each position sees only its past even though the full sequence is in the tensor. |

## Per-Lesson Summaries

### the-problem-attention-solves
**Status:** Built
**Cognitive load type:** STRETCH
**Type:** Hands-on (notebook: `4-2-1-the-problem-attention-solves.ipynb`)
**Widgets:** AttentionMatrixWidget — interactive heatmap of raw dot-product attention. Student types a sentence, sees the full attention weight matrix. Hover shows exact weight values plus dashed outline of the mirror cell (for observing symmetry). Toggle between raw scores and softmax weights. Preset sentences for guided exploration ("The bank was steep", "The cat chased the mouse", "The cat sat on the mat").

**What was taught:**
- Static embeddings are insufficient for language because the same token needs different representations in different contexts (polysemy problem)
- Dot-product attention creates context-dependent representations using only three matrix operations: similarity scores (XX^T), softmax normalization, and weighted averaging
- The attention weights are data-dependent — freshly computed from each new input, not fixed parameters
- Every token computes a score with every other token regardless of distance (no locality constraint)
- Raw dot-product attention has a fundamental limitation: one embedding per token for both "seeking" and "offering" roles

**How concepts were taught:**
- **Polysemy hook (callback to Module 4.1):** Recalled the amber warning from embeddings-and-position. ComparisonRow: "The bank was steep and muddy" vs "The bank raised interest rates." Same embedding for "bank" in both. Promise: "By the end of this lesson, you'll see how."
- **Dot product recap (gap resolution):** Three-panel inline SVG showing similar-direction (large positive), perpendicular (zero), and opposite (negative) dot products. Color-coded vectors. Numerical examples with 3-dim vectors. Connected to cosine similarity formula.
- **Three escalating attempts:** (1) Average all embeddings — callback to bag-of-words problem, "Dog bites man" = "Man bites dog." (2) Weighted average — different weights per token, but who decides? (3) Let the input determine the weights via dot products + softmax. The "who decides the weights?" tension drives the narrative.
- **Worked example with tiny numbers:** 4 tokens ("The", "cat", "sat", "here"), 3-dim embeddings. All pairwise dot products computed explicitly in a 4x4 table. Softmax applied row-wise. Weighted average computed for "cat" showing the formula with exact weights. Aside: XX^T notation, one matrix multiplication gives all pairwise dot products.
- **Formula build-up:** Three steps collapsed into one expression: Attention(X) = softmax(XX^T) X. Each step labeled (similarity scores, attention weights, context-dependent output). InsightBlock: the weight matrix W is not a learned parameter.
- **All-pairs contrast with CNNs:** Two paragraphs after the formula. "The cat sat on the mat because it was soft" — "it" attends to "mat" 6 positions away. WarningBlock: attention has no distance preference, unlike CNN local windows.
- **Prediction exercise:** Student predicts which tokens "cat" attends to most before computing. Reveals that repeated "the" tokens have identical patterns (no positional encoding in raw attention).
- **Interactive AttentionMatrixWidget:** Student types sentences and sees attention heatmaps. TryThisBlock: try "bank" in different contexts, try repeated words, try short sentences, observe symmetry via mirror-cell indicator.
- **Dual-role limitation:** Symmetry observation from widget. "The cat chased the mouse" — different reasons, same score. Cocktail party analogy. ComparisonRow: "When This Token Is Looking" vs "When Other Tokens Look At It." The same embedding vector serves both roles.
- **Similarity vs relevance negative example:** "The bank was steep" — "steep" disambiguates "bank" but has low embedding similarity. "Interest" is more similar to "bank" but less relevant in this context.
- **Transfer question (cliffhanger):** "If every token had TWO vectors — one for seeking, one for advertising — would the matrix still be symmetric?" Reveal: No, because s_A dot a_B != s_B dot a_A in general. Forward reference to next lesson (two projection matrices) without naming Q or K.
- **Residual stream seed:** Muted aside at the end noting that in the full transformer, attention output is ADDED to the original embedding, not substituted. Callback to skip connections from ResNets.

**Mental models established:**
- "Attention is a weighted average where the input determines the weights" — the defining insight of the attention mechanism
- "The input decides what matters" (data-dependent weights vs fixed parameters) — paradigm shift from CNNs
- "Similarity is not the same as relevance" — raw dot products measure embedding similarity, but relevance requires more
- "One embedding can't serve two roles" (seeking vs offering) — the limitation that motivates Q/K projections

**Analogies used:**
- Cocktail party: what you're searching for vs what you advertise are different (dual-role limitation)
- CNN contrast: fixed local filters vs data-dependent all-pairs computation (two different paradigms)
- Callback to bag-of-words from embeddings-and-position ("you already know this doesn't work")
- "Embeddings are a learned dictionary with one definition per word; attention rewrites the definition based on context"

**What was NOT covered (scope boundaries):**
- Q, K, V projections (not even named in student-facing text) — Lesson 2 and 3
- Scaled dot-product attention (division by sqrt(d_k)) — Lesson 2
- Multi-head attention — Lesson 4
- The transformer block architecture — Lesson 5
- Causal masking — Lesson 6
- Self-attention vs cross-attention distinction
- Positional encoding in the attention computation (deliberately isolated for clarity)

**Misconceptions addressed:**
1. "Attention is just looking at nearby words (like a convolution filter)" — "it" attends to "mat" 6 positions away. Attention computes all-pairs scores with no distance preference. Explicit WarningBlock contrasting CNN local windows.
2. "The dot product between two embeddings tells you their semantic similarity (and that's all you need)" — "bank" and "steep" have low similarity but "steep" is highly relevant. Similarity and relevance are related but different questions.
3. "Attention produces a completely new representation, replacing the original embedding" — Residual stream seed: in the full transformer, attention output is ADDED to the original. Token in uninformative context keeps its original meaning.
4. "Each token 'decides' what to attend to (like conscious choice)" — Mechanical language used throughout: "the weights are computed from..." "operating mechanically on whatever vectors come in." No anthropomorphic framing.
5. "Symmetric attention (token A attends to B the same as B attends to A) is fine" — "The cat chased the mouse": "cat" needs "chased" for a different reason than "chased" needs "cat." Transfer question reveals that two separate vectors would break the symmetry.

### queries-and-keys
**Status:** Built
**Cognitive load type:** BUILD
**Type:** Hands-on (notebook: `4-2-2-queries-and-keys.ipynb`)
**Widgets:** None (lesson uses inline computed tables and SVG diagrams; no interactive widget)

**What was taught:**
- Two learned projection matrices (W_Q, W_K) transform the same embedding into separate "seeking" (query) and "offering" (key) vectors, breaking the dual-role limitation from Lesson 1
- QK^T replaces XX^T as the attention score matrix, computing learned relevance rather than raw embedding similarity
- QK^T is asymmetric because q_i dot k_j != q_j dot k_i (different projection matrices produce different vectors)
- Scaling by sqrt(d_k) is essential to prevent softmax saturation as dimensions grow — not a cosmetic cleanup but a training necessity
- The complete attention weight formula: weights = softmax(QK^T / sqrt(d_k))

**How concepts were taught:**
- **Cliffhanger resolution hook:** Callback to the transfer question from Lesson 1 ("If every token had two vectors — one for seeking, one for advertising — would the matrix still be symmetric?"). Student already answered no. This lesson delivers the mechanism they predicted.
- **Job fair analogy:** Every attendee writes two cards — "what I'm looking for" (query) and "what I bring to the table" (key). A recruiter matches seeker-cards against provider-cards. One person's seeking card is different from their offering card. Extends the cocktail party analogy from Lesson 1 with more concrete, naturally asymmetric framing.
- **"Learned lens" framing:** W_Q is a lens that reveals what the token is seeking; W_K reveals what it offers. Same embedding, different lens, different view. The lens is what the model learns during training.
- **nn.Linear connection:** Q and K are just nn.Linear(d_model, d_k, bias=False) — "the same building block you've been using since Module 2.1." Nothing conceptually new; the novelty is applying two different matrices to the same input.
- **Geometric/spatial SVG:** Inline SVG showing one embedding point projected through two different matrices to two different destinations. Labels: embedding (violet), Q (sky blue via W_Q), K (amber via W_K). Caption: "same input, different matrices, different destinations."
- **WarningBlock on misconception:** "Q and K are NOT properties of the token. They're properties of the learned projection matrices. Change the matrices (by training longer), and the same token's Q and K change. Different layers produce different Q and K from the same embedding."
- **Relevance matrix explanation:** QK^T explained as a learned relevance function, not raw similarity. "Bank"/"steep" callback softened to "in a trained model, W_Q and W_K would learn projections that make the dot product large." Explicit asymmetry proof: q_i = W_Q x_i, k_i = W_K x_i, W_Q != W_K, so q_i dot k_j != q_j dot k_i. Explicit statement that Q and K live in the same d_k-dimensional space (misconception #3 correction).
- **Prediction exercise:** "The cat chased the mouse" — should "cat"'s attention to "chased" equal "chased"'s attention to "cat"? No: "cat" seeks what-action-did-I-do, "chased" seeks who-did-the-chasing. Different queries, same keys, different scores. Resolves the asymmetry problem from Lesson 1.
- **Full worked example with same 4 tokens:** Same embeddings from Lesson 1 ("The", "cat", "sat", "here", 3-dim). Two 3x3 matrices (W_Q, W_K) with small integer-ish values for hand-traceability. Step 1: compute all Q vectors. Step 2: compute all K vectors. Step 3: compute QK^T (4x4 matrix, verified asymmetric). Step 4: apply softmax row-wise. All values computed at load time and displayed inline.
- **Side-by-side comparison:** Raw XX^T weights (symmetric, from Lesson 1) next to QK^T weights (asymmetric, this lesson). Same tokens, same embeddings. Visual centerpiece of the lesson. Color-coded by weight magnitude.
- **Scaling section:** Dimension growth analysis: d_k=3 -> dot products ~1.7 (fine), d_k=64 -> ~8.0 (getting large), d_k=512 -> ~22.6 (catastrophic). Temperature callback: "Remember the temperature slider? Large dot products act like dividing by tiny temperature." Concrete softmax: softmax([1.7, -0.3, 0.5, 0.8]) -> useful distribution; softmax([22, -15, 3, 8]) -> one-hot. Vanishing gradients callback: near-0/1 softmax outputs = near-zero gradient = frozen model. The fix: divide by sqrt(d_k). Variance of dot product = d_k, so dividing by sqrt(d_k) normalizes to variance 1. Side-by-side GradientCards: without scaling (collapses) vs with scaling (learns normally).
- **d_k tradeoff (misconception #5):** "Why not just make d_k as large as possible?" Larger d_k captures finer-grained relevance but each projection matrix has d_model x d_k parameters. Models keep d_k small (typically 64) and use multiple attention heads. Teaser for multi-head lesson.
- **Transfer question on scaling:** Colleague says d_k=8 model trains fine without scaling. What would you tell them? At d_k=8, dot products ~2.8, not extreme enough to saturate softmax. At d_k=64 or d_k=128, they would be. The test doesn't generalize.
- **Consolidated notebook section:** Five exercises covering full Q/K pipeline — implement projections from scratch, verify asymmetry with heatmap, set W_Q=W_K and watch symmetry return, implement scaled attention weights, experiment with d_k=8/64/512, stretch with GPT-2 pretrained weights. Less scaffolding than Lesson 1 notebook.
- **Formula summary:** Three steps: (1) project Q=W_Q X, K=W_K X, (2) score S=QK^T/sqrt(d_k), (3) normalize W=softmax(S). Compared to Lesson 1's softmax(XX^T): same structure, two additions.
- **Forward reference to V:** Attention weights tell us how much to attend, but weights need something to weight. The key that makes a token relevant (K) and the information it should provide when attended to are different things. "Should we use the same embedding for matching AND contributing? We already know the answer to that kind of question."

**Mental models established:**
- "Q and K are learned lenses — same embedding, different lens, different view" — the projection matrix is the interesting learned component, not the token
- "QK^T computes learned relevance, not raw similarity" — extends the similarity-vs-relevance distinction from Lesson 1 into a concrete mechanism
- "Scaling by sqrt(d_k) is not cosmetic — it's the difference between a model that learns and one that doesn't" — connects temperature behavior and vanishing gradients into the attention context

**Analogies used:**
- Job fair: two cards per person (seeking card = query, offering card = key). Extends cocktail party from Lesson 1 with more concrete, naturally asymmetric framing.
- Learned lens: W_Q and W_K are different lenses applied to the same embedding. Change the lens (train longer) and the view changes.
- Temperature callback: large dot products act like low temperature (from what-is-a-language-model TemperatureExplorer)
- Telephone game callback: vanishing gradients from softmax saturation (from training-dynamics)

**What was NOT covered (scope boundaries):**
- V projection (what a token contributes when attended to) — Lesson 3
- The full attention output (the weighted sum using V) — Lesson 3
- Multi-head attention — Lesson 4
- The transformer block — Lesson 5
- Causal masking — Lesson 6
- Cross-attention (all attention in this module is self-attention)
- Why d_k might differ from d_model (a multi-head concern, Lesson 4)
- Attention as a complete layer (no residual connection, no output projection yet)

**Misconceptions addressed:**
1. "Q and K are properties of the token itself (like part of speech)" — WarningBlock: Q and K are properties of the learned projection matrices. Change the weights and the same token's Q and K change. Different layers produce different Q and K from the same embedding.
2. "Scaling by sqrt(d_k) is just a normalization trick (cosmetic, optional)" — Without scaling at d_k=512, softmax outputs are one-hot, gradients vanish, training collapses. Concrete numerical comparison. The scaling factor is the difference between a model that learns and one that doesn't.
3. "Q and K are in different spaces (like keys and locks are made of different material)" — Explicitly stated in Section 4 prose: "Q and K live in the same d_k-dimensional space — they arrived there via different matrices, but they share the same coordinate system. If they were in different spaces, the dot product between them would be meaningless."
4. "QK^T should still be symmetric (like XX^T was)" — Worked example computes QK^T and shows it's not symmetric. Callback to transfer question: "Remember XX^T was symmetric. Not anymore." Formal proof: q_i dot k_j != q_j dot k_i because W_Q != W_K.
5. "Larger d_k is always better (more dimensions = more information)" — Addressed in scaling section: larger d_k captures finer relevance but requires more parameters. Models keep d_k small (~64) and use multiple heads. Teaser for multi-head lesson.

### values-and-attention-output
**Status:** Built
**Cognitive load type:** BUILD
**Type:** Hands-on (notebook: `4-2-3-values-and-attention-output.ipynb`)
**Widgets:** None (lesson uses inline computed tables, SVG diagrams, and KaTeX formulas; no interactive widget)

**What was taught:**
- V = W_V x embedding is a third learned projection that separates "what makes me relevant for matching" (K) from "what I contribute when attended to" (V)
- The full single-head attention formula: output = softmax(QK^T / sqrt(d_k)) V, completing the three-lesson formula evolution
- Each token gets its own output vector (same shape as input: n tokens in, n vectors out)
- The residual stream: attention output is ADDED to the original embedding (final_output = attention_output + embedding), not substituted
- V generalizes the raw-embedding output (W_V = identity reduces to Lesson 1's formula)
- Locking W_V = W_K collapses matching and contributing into one signal

**How concepts were taught:**
- **Hook (pattern recognition callback):** Direct callback to the Lesson 2 forward reference: "The key that makes a token relevant and the information it should provide when attended to are different things. Should we use the same embedding for matching AND contributing?" Framed as "one vector, two roles. You know the fix." — the student has resolved this pattern twice before.
- **Three-lenses framing:** W_Q (what am I seeking?), W_K (what do I advertise for matching?), W_V (what do I actually have to say?). Same embedding, three different matrices, three different views. All learned from data.
- **Job fair analogy extended:** Third card added. Q = seeking card, K = offering card (gets you the match), V = resume (what you actually deliver when matched). "The offering card got you noticed; the resume is what you deliver."
- **Geometric SVG:** Same embedding point projected through three matrices to three different destinations. Q (sky blue), K (amber), V (emerald). Extends the two-projection SVG from Lesson 2.
- **nn.Linear connection:** V is just another nn.Linear(d_model, d_v, bias=False). "You now have three nn.Linear layers. That's the entire Q/K/V mechanism. No new operation."
- **K-vs-V prediction exercise:** "The cat chased the mouse" — what should K encode for matching vs what should V encode for contributing? K: "I'm an action verb." V: past tense, transitivity, directed motion. Different purposes, different information.
- **Formula evolution display:** Three stacked formulas showing the progression across three lessons: Lesson 1 (softmax(XX^T)X), Lesson 2 (weights = softmax(QK^T/sqrt(d_k))), this lesson (output = softmax(QK^T/sqrt(d_k))V). "Each step is a targeted replacement."
- **Full worked example (three-lesson continuity):** Same 4 tokens ("The", "cat", "sat", "here"), same embeddings, same W_Q and W_K from Lessons 1-2, plus new 3x3 W_V. Step 1: compute all V vectors. Step 2: reuse attention weights from Lesson 2. Step 3: compute weighted average of V vectors for "cat" with explicit formula and all numbers shown. All 4 output vectors displayed. K-vs-V side-by-side comparison showing different vectors from same embeddings.
- **Misconception #5 explicit correction:** "You might picture attention as producing a single summary vector for the whole sequence. It does not. Each token gets its own output because each token has its own row of attention weights. Four tokens in, four vectors out."
- **W_V = W_K negative example (amber callout):** If V and K used the same matrix, the contribution of each token would be its key vector — the matching signal. "The model loses a degree of freedom: matching and contributing are locked together."
- **W_V = identity positive example (emerald callout):** V = I x embedding = raw embedding. Output becomes softmax(QK^T/sqrt(d_k)) x X — exactly Lesson 1's formula with better scoring. "V generalizes the raw-embedding output; identity is a special case."
- **Residual stream (problem before solution):** Token in uninformative context loses identity if attention output replaces embedding. The fix: add, don't substitute. Callback to ResNet F(x) + x: "editing, not writing." Concrete: residual output for "cat" = attention_output + embedding, with actual numbers.
- **InsightBlock (residual stream):** Scaled to INTRODUCED depth: mentions ResNet benefits (gradients, identity), adds embedding preservation, defers multi-layer backbone role to Lesson 5.
- **Transfer question:** Colleague removes the residual connection. What breaks? (1) Information loss — tokens in uninformative context lose identity. (2) Gradient flow — no direct path from output to input (callback to gradient highway from ResNets).
- **Notebook exercises (6 + stretch):** (1) Implement V projection, verify V differs from K. (2) Implement complete single_head_attention function. (3) Add residual connection, verify shape. (4) Set W_V = identity, recover Lesson 1's formula. (5) Set W_V = W_K, observe contribution lock. (Stretch) Load GPT-2 pretrained weights, compare K and V vectors for same tokens.
- **Three-lesson arc echo:** "You built this formula piece by piece. Lesson 1: raw dot-product attention. Lesson 2: Q and K for asymmetric matching. This lesson: V for contribution, residual stream for preservation. Single-head attention is complete."
- **Seed for multi-head:** "Single-head attention computes one notion of relevance. 'The cat sat on the mat because it was soft' — 'it' needs to attend to 'mat' for coreference AND to 'soft' for meaning. One set of Q/K/V weights can only capture one type of relationship at a time."

**Mental models established:**
- "Three lenses, one embedding" — W_Q, W_K, and W_V extract three different views of the same token. The model learns all three from data.
- "K gets you noticed; V is what you actually deliver" (resume extension of job fair analogy) — separates the matching function from the contribution function
- "Attention edits the embedding, it doesn't replace it" — the residual stream preserves the original while enriching it with context (extends "editing not writing" from ResNets)

**Analogies used:**
- Job fair resume card: K = offering card (gets the match), V = resume (what you deliver). Third card extending the two-card analogy from Lesson 2.
- Three lenses: W_Q, W_K, W_V as three different lenses applied to the same embedding. Extends the "learned lens" framing from Lesson 2.
- "Editing, not writing" (callback to ResNets): attention output is the edit, original embedding is the document. Direct F(x) + x parallel.
- "The cat chased the mouse" (three-lesson recurring example): used for K-vs-V distinction. K encodes action-verb semantics, V encodes tense/transitivity/motion.

**What was NOT covered (scope boundaries):**
- Multi-head attention (multiple Q/K/V sets in parallel) — Lesson 4
- The output projection W_O after concatenating heads — Lesson 4
- The full transformer block architecture — Lesson 5
- Layer normalization — Lesson 5
- Causal masking — Lesson 6
- The residual stream across multiple stacked layers — Lesson 5
- Training attention or how W_Q, W_K, W_V are learned — Module 4.3
- Cross-attention (all attention in this module is self-attention)
- Attention as a standalone PyTorch module (nn.MultiheadAttention)

**Misconceptions addressed:**
1. "V is just the raw embedding — why do we need a third projection?" — "Chased" is relevant to "cat" because of action-verb semantics (K), but what "chased" contributes should be different information (tense, transitivity, motion). Without V, contribution IS the key — matching and contributing are locked. With V, the model extracts different features for each role. K-vs-V side-by-side with concrete numbers proves the vectors are different.
2. "V, K, and the embedding are all basically the same thing" — W_V = W_K amber callout shows that locking them collapses two distinct roles. K and V for "cat" are shown as different vectors from the same embedding. W_V = identity emerald callout shows identity is a special case, not the general case.
3. "The attention output replaces the original embedding" — Token in uninformative context loses identity if output replaces embedding. Residual stream preserves the original: final_output = attention_output + embedding. "If attention learns nothing useful, the token keeps its original meaning."
4. "The residual stream in a transformer is just a skip connection (same as ResNet, nothing new)" — InsightBlock acknowledges the parallel but notes that in transformers, the residual stream also preserves the original embedding while attention enriches it with context. Full architectural significance (central highway across layers) deferred to Lesson 5 to match INTRODUCED depth.
5. "Attention produces a single output vector for the whole sequence" — Explicitly named and disproved: "You might picture attention as producing a single summary vector. It does not." Each token gets its own output because each has its own row of attention weights. All 4 output vectors shown. Same shape as input: n in, n out.

### multi-head-attention
**Status:** Built
**Cognitive load type:** STRETCH
**Type:** Hands-on (notebook: `4-2-4-multi-head-attention.ipynb`)
**Widgets:** None (lesson uses inline computed tables, SVG diagrams, and KaTeX formulas; no interactive widget)

**What was taught:**
- A single attention head captures only one notion of relevance; multiple heads in parallel capture diverse relationship types simultaneously
- Dimension splitting: d_k = d_model / h partitions the existing budget rather than multiplying compute (same total FLOPs)
- Output projection W_O is a learned d_model x d_model mixing layer that lets heads synthesize their findings across the d_k slices
- Head specialization in trained models is messy and emergent, not cleanly assigned linguistic roles
- The complete multi-head attention formula: MultiHead(X) = Concat(head_1, ..., head_h) W_O

**How concepts were taught:**
- **Hook (limitation reveal + prediction exercise):** "The cat sat on the mat because it was soft" — "it" needs to attend to "mat" for coreference AND to "soft" for meaning. Prediction exercise: can one Q vector produce high attention on both? Reveal: Q must align with both K vectors, but "mat" and "soft" have different semantics, so their K vectors point in different directions. One dot product computes one scalar score — the issue is structural, not dimensional.
- **Research team analogy:** Instead of one researcher tracking every thread, assign a team of specialists. Each reads through their own lens. Pool findings at the end. No single specialist sees everything, but the team covers more ground.
- **Side-by-side hypothetical attention patterns:** Head 1 (coreference: "it" -> "mat" at 0.61) vs Head 2 (property attribution: "it" -> "soft" at 0.52). Qualified as "Hypothetical weights from a trained model" to signal they are illustrative.
- **Misconception #5 in main flow (not just aside):** "These are completely independent sets of parameters — Head 1's W_Q has no connection to Head 2's W_Q. They start as different random matrices and learn to specialize in different ways." Heads share input, nothing else.
- **Dimension splitting with SVG:** Wide rectangle (d_model=768) split into 12 colored strips (d_k=64 each). GPT-2 as concrete example. Compute equivalence derived: h * n^2 * d_k = n^2 * d_model. "Split, not multiplied" InsightBlock.
- **More heads = better misconception:** d_model=768 with 768 heads = d_k=1, a single scalar multiplication per head. Absurdly limited. 1 head = d_k=768, rich but only one type. GPT-2 (12 heads, d_k=64) and GPT-3 (96 heads, d_k=128) as design choices with tradeoffs.
- **Concatenation + W_O (problem before solution):** Each head's output is (n, d_k). Concatenate to (n, d_model). Problem: each head's contribution stays isolated in its d_k slice. W_O fixes this — learned mixing, not reshaping. d_model^2 parameters. "The team meeting" analogy: specialists pool findings.
- **Prediction exercise (dimension reasoning):** d_model=512, h=8. Compute d_k=64, W_O params=262,144, effect of adding a 9th head (512 doesn't divide evenly by 9).
- **Full worked example (4 tokens, d_model=6, h=2, d_k=3):** Same tokens from Lessons 1-3, extended to 6-dim embeddings. Head 1 and Head 2 with different 6x3 projection matrices. Attention weight tables for both heads with guided interpretation. Head 1: "here" attends to "cat" (0.586) — cross-token relationship. Head 2: "here" attends to itself (0.628) — self-reinforcing pattern. Cross-head comparison paragraph: "Same token, same input, but the two heads extract completely different views of what matters." Concatenation + W_O traced with actual numbers. Residual addition shown.
- **What heads actually learn (Elaborate):** Research findings on BERT and GPT-2 — previous-token attention, syntactic tracking, near-uniform heads, 20-40% prunable. Callback to CNN filters softened: "architecture provides capacity, training determines what each learns. The difference between a learned pattern and a designed pattern." InsightBlock: "Capacity, not assignment."
- **Transfer question:** Compare d_model=256 with h=1 (d_k=256) vs h=4 (d_k=64). Both have same Q/K/V parameters; 4-head model adds W_O (65K extra). 4-head captures more diverse relationships but each head less expressive.
- **Notebook exercises (5 exercises):** (1) Guided: implement single_head_attention function. (2) Supported: multi-head attention from scratch (d_model=6, h=2), verify shape. (3) Supported: verify compute equivalence (time comparison). (4) Independent: multi-head as nn.Module with batched forward pass. (5) Stretch: load GPT-2 pretrained weights, extract attention weights for all 12 heads, visualize heatmaps.
- **Mental model echo:** "Multiple lenses, pooled findings" — extends "three lenses, one embedding" to "three lenses per head, h heads, one synthesis step."
- **Seed for transformer block:** "Attention reads from the residual stream; the FFN writes new information into it. Different roles, same stream."

**Mental models established:**
- "Multiple lenses, pooled findings" — each head has its own set of three lenses (W_Q^i, W_K^i, W_V^i), and W_O synthesizes what they all saw. Extension of "three lenses, one embedding."
- "Split, not multiplied" — dimension splitting is budget allocation, not compute multiplication. Same total FLOPs, partitioned differently.
- "Capacity, not assignment" — multi-head attention provides capacity for diverse attention patterns; what each head actually learns is determined by training, not architecture.

**Analogies used:**
- Research team: specialists reading through their own lens, pooling findings. Maps to: each head operates independently, W_O synthesizes.
- Budget allocation: splitting d_model into h pieces of d_k each. Same total, partitioned differently.
- Team meeting: concatenation is each specialist filing their report; W_O is the meeting where they synthesize findings.
- CNN filters callback (softened): architecture provides capacity (filters, heads), training fills the slots.

**What was NOT covered (scope boundaries):**
- The transformer block (attention + FFN + residual + layer norm) — Lesson 5
- Causal masking — Lesson 6
- Cross-attention — out of scope for this module
- nn.MultiheadAttention as a PyTorch module — student builds from scratch
- How specialization emerges during training — Module 4.3
- Attention head pruning or efficiency optimizations — Module 4.3

**Misconceptions addressed:**
1. "More heads is always better" — d_k=1 per head (d_model=768, h=768) is a single scalar multiplication per head, absurdly limited. 1 head gives d_k=768, rich but only one type. The tradeoff is real. GPT-2 (12 heads) and GPT-3 (96 heads) are design choices, not "more is better."
2. "Each head specializes in a specific linguistic role (head 1 = syntax, head 2 = semantics)" — Research on trained models shows messy, emergent patterns. Some heads attend to previous token, some are near-uniform, 20-40% can be pruned. Not a department org chart.
3. "Multi-head attention is computationally more expensive than single-head" — Compute equivalence derived: h * n^2 * d_k = n^2 * d_model. Identical to single-head. The computation is split, not multiplied.
4. "W_O is just reshaping (gluing heads back together)" — W_O is a learned d_model x d_model matrix with d_model^2 parameters. It mixes information across heads. Without W_O, each head's contribution is isolated. With W_O, head 3's findings can influence dimensions head 1 wrote to. Learned mixing layer, not reshape.
5. "Heads share weights and just look at different parts of the input" — Each head has completely independent W_Q^i, W_K^i, W_V^i. Two heads processing the same input produce entirely different Q, K, V vectors and attention patterns. Addressed in main content flow (not just aside): "These are completely independent sets of parameters."

### the-transformer-block
**Status:** Built
**Cognitive load type:** BUILD
**Type:** Conceptual (no notebook)
**Widgets:** None (lesson uses two inline SVG diagrams: TransformerBlockDiagram and StackedBlocksDiagram)

**What was taught:**
- The transformer block assembles MHA, FFN, residual connections, and layer normalization into the single repeating unit that stacks to form GPT-2/GPT-3/modern LLMs
- "Attention reads, FFN writes" as the organizing mental model for why both sub-layers are needed
- Layer normalization: same formula as batch norm but normalizes across features within a single token instead of across examples in a batch
- Pre-norm (modern standard) vs post-norm (original 2017 paper): pre-norm keeps the residual stream clean for gradient flow
- The FFN's 4x expansion factor (768 -> 3072 -> 768 in GPT-2) and its role as the model's knowledge store
- The residual stream as the central backbone of the entire model (upgraded from INTRODUCED to DEVELOPED)
- Parameter distribution: ~1/3 attention, ~2/3 FFN per block
- The block is shape-preserving ((n, d_model) in, (n, d_model) out), enabling identical blocks to stack

**How concepts were taught:**
- **Hook (parameter puzzle):** GPT-2 parameter breakdown: ~28M attention (~23%), ~57M FFN (~46%), ~38M embeddings (~31%). Challenges the "transformer = attention" misconception that built over four lessons. "Where do 2/3 of the parameters live? What are they doing?"
- **Layer norm by contrast with batch norm:** Motivation: batch norm normalizes across examples in a batch, but for variable-length sequences this is nonsensical (5th token of different sentences has no shared statistics). Layer norm normalizes across features within a single token. ComparisonRow: batch norm (column-wise, batch-dependent, train/eval split) vs layer norm (row-wise, independent, same behavior always). Formula shown: identical to batch norm, different axis. Comprehension check: why not batch norm in a transformer? Does layer norm need train/eval distinction?
- **FFN structure:** Formula FFN(x) = W_2 * GELU(W_1 x + b_1) + b_2. GELU callback to activation-functions-deep-dive decision guide ("GELU for transformers -- this is where it lives"). 4x expansion as workspace analogy: model needs more room to think than to communicate. Parameter count derivation: FFN = 2 * 768 * 3072 = 4,718,592 vs attention = 4 * 768^2 = 2,359,296. FFN has 2x the parameters -- answers the hook.
- **"Attention reads, FFN writes" framing:** Attention gathers context from other tokens (reads the residual stream). FFN processes what attention found and updates the representation (writes to the stream). Different operations, complementary roles. InsightBlock aside reinforces the distinction.
- **Complete block diagram (visual centerpiece):** TransformerBlockDiagram SVG: vertical flow bottom-to-top. Color-coded: violet residual stream (dashed backbone), sky blue MHA, amber FFN, emerald LayerNorm. Two branch-and-merge residual paths with "skip" labels. Dimension annotations at every stage (n, 768). Pre-norm placement visible (LN before each sub-layer). Legend included.
- **Pre-norm vs post-norm (brief):** ComparisonRow with formulas. Post-norm: norm on the residual stream itself, requires learning rate warmup. Pre-norm: norm inside the branch, stream stays clean. Concrete gradient argument: post-norm forces gradients through 24 layer norms in GPT-2; pre-norm provides clean additive path. All lesson diagrams and formulas use pre-norm.
- **Residual stream development (INTRODUCED -> DEVELOPED):** Explicit misconception address: "You know residual connections from ResNets... It is tempting to think the transformer uses them the same way. The mechanism IS identical. But the role is fundamentally larger." Two residual connections per block = 24 in GPT-2. Shared document analogy extended: starts as raw embedding, 24 sub-layers each read and annotate. Concrete negative example: without residual connections, early-training near-uniform attention destroys token identity (original embedding lost, replaced by near-uniform average). With residual: original flows through untouched, contributions are additive deltas.
- **FFN importance (addressing "just plumbing" misconception):** Concrete negative example via convex hull argument: three tokens at A, B, C. Attention can only produce weighted averages inside the triangle. FFN's GELU breaks this constraint, enabling genuinely new representations. Research reference: Geva et al. FFN neurons as key-value memories, specific neurons for specific concepts. "2/3 of parameters store the model's learned knowledge."
- **Stacking visualization:** StackedBlocksDiagram SVG showing Block 1 -> Block 2 -> ... -> Block N with residual stream backbone. Earlier blocks capture simpler patterns, later blocks capture more complex ones (callback to CNN hierarchical features).
- **Transfer question:** Colleague proposes reducing FFN expansion from 4x to 1x. Expected: cuts FFN params by 75%, dramatically reduces knowledge storage capacity. The expansion is not wasteful -- it is where the work happens.

**Mental models established:**
- "Attention reads, FFN writes" -- attention gathers context (reads the residual stream), FFN processes and transforms (writes to the stream). Complementary, both essential.
- "The residual stream is a shared document" -- starts as raw embedding, each of 24 sub-layers annotates it. By block 12, the document has been enriched by all prior processing.
- "The model needs more room to think than to communicate" -- the 4x FFN expansion creates workspace for complex computations that gets compressed back to d_model.

**Analogies used:**
- "Editing a document, not writing from scratch" (extended from ResNets): MHA proposes a context-enrichment edit, FFN proposes a processing edit, both added via residual connections.
- Shared document analogy for the residual stream: each sub-layer reads the current version and adds annotations. Block 1's attention reads the raw document; Block 12's FFN reads a version enriched by 23 prior sub-layers.
- Workspace analogy for 4x expansion: the model needs more room to think (3072 dimensions) than to communicate (768 dimensions). Compression keeps only what is useful.
- Convex hull analogy for FFN nonlinearity: attention can only produce points inside the triangle formed by input token positions; FFN's GELU enables points outside it.
- Gradient highway from ResNets (extended): 24 residual additions in GPT-2 provide direct gradient path from output to any layer.

**What was NOT covered (scope boundaries):**
- Implementing the transformer block in PyTorch -- Module 4.3 (building-nanogpt)
- Causal masking -- Lesson 6 (decoder-only-transformers)
- The full decoder-only architecture -- Lesson 6
- Training transformers -- Module 4.3
- How many blocks to use / scaling -- Module 4.3
- Cross-attention, encoder-decoder architecture
- RMSNorm or other layer norm variants
- Mixture of experts or other FFN variations

**Misconceptions addressed:**
1. "The transformer is just attention (attention = transformer)" -- GPT-2 parameter count: ~23% attention, ~46% FFN. FFN has 2x the parameters. If the transformer were just attention, where do 2/3 of the parameters live? The FFN stores the model's learned knowledge.
2. "The FFN is just boring plumbing between attention layers" -- Concrete convex hull argument: attention produces weighted averages inside the convex hull of input vectors. FFN's GELU breaks this constraint. Research shows FFN neurons activate for specific concepts (key-value memories). The FFN is where the model does its thinking.
3. "Layer norm and batch norm are the same thing" -- Batch norm normalizes across examples (depends on batch composition, has train/eval distinction, fails for variable-length sequences). Layer norm normalizes across features within one token (independent, no train/eval distinction, works for any sequence length). Same formula, different axis, fundamentally different behavior.
4. "Pre-norm and post-norm are just implementation details" -- Post-norm forces gradients through a layer norm at every block on the residual stream (24 in GPT-2). Pre-norm gives a clean additive gradient path bypassing all norms. The difference compounds with depth, making post-norm unstable in deep models. Nearly all modern LLMs use pre-norm.
5. "Residual connections in transformers are just like ResNet skip connections (same purpose, same story)" -- Same F(x) + x mechanism, but fundamentally larger role. Two per block (not one). The residual stream is the central backbone of the entire model, not just a training aid for individual blocks. Every sub-layer reads from and writes to it. Concrete failure: without residual connections, early-training uniform attention destroys token identity.

### decoder-only-transformers
**Status:** Built
**Cognitive load type:** CONSOLIDATE
**Type:** Conceptual (no notebook)
**Widgets:** None (lesson uses two inline SVG diagrams: CausalMaskDiagram and GptArchitectureDiagram)

**What was taught:**
- Causal masking: why it exists (prevent data leakage during parallel next-token prediction training), how it works (set upper-triangle entries to negative infinity before softmax), and why it cannot be removed at inference (future tokens do not exist)
- The complete GPT architecture assembled end-to-end: token embedding + positional encoding -> N transformer blocks with causal masking -> final layer norm -> output projection -> softmax -> next-token probabilities
- Output projection: nn.Linear(d_model, vocab_size) mapping hidden states to vocabulary logits, weight-tied with token embeddings
- Total parameter counting for GPT-2: ~124.4M verified against the known figure
- Training vs inference asymmetry: training is fully parallel (causal mask prevents leakage), inference is sequential (future tokens do not exist)
- Encoder-decoder vs decoder-only distinction: the name is historical, decoder-only means causal masking
- Why decoder-only won: simplicity, scaling, generality

**How concepts were taught:**
- **Cheating problem hook:** "The cat sat on the mat" with full bidirectional attention. Position 3 ("sat") has high attention on position 4 ("on") -- the very token it is trying to predict. The model copies the answer instead of learning to predict. CausalMaskDiagram SVG shows before-and-after with "leak!" annotation on the cheating cell.
- **Exam analogy:** Answer key printed next to questions. Causal masking is the cardboard sleeve covering future answers. After the test (inference), there IS no answer key -- you generate the answers yourself. Naturally addresses the "masking is a training trick" misconception.
- **Causal mask mechanism:** Before softmax, set j > i entries to negative infinity. e^(-infinity) = 0, remaining entries renormalize automatically. Formula: same scaled dot-product attention, one additional step (masking before softmax). Lower-triangular matrix is the visual signature.
- **Worked example:** Row 3 ("sat") raw scores [2.1, 3.5, 1.8, 4.2, 0.9, 1.1]. Position 4 had the highest raw score (4.2) but is eliminated completely. After masking: [2.1, 3.5, 1.8, -inf, -inf, -inf]. After softmax: [0.27, 0.55, 0.18, 0, 0, 0]. Renormalization is free (softmax naturally sums to 1 over finite inputs).
- **Misconception #1 (not like dropout):** Causal masking is not removed at inference. Future tokens literally do not exist during generation. The mask simulates the inference constraint during training. Explicit ComparisonRow: dropout (on during training, off during inference) vs causal masking (mirrors inference reality).
- **Misconception #2 (training is parallel):** All positions predict their next token simultaneously in one forward pass. N training examples from one sequence. ComparisonRow: training (parallel, masked) vs inference (sequential, no future tokens).
- **Output projection gap resolution:** Recalled the symmetry aside from Module 4.1. Final block output (n, 768) multiplied by W_out (768, 50257) produces logits. Softmax gives P(next token | context). Connection to MNIST output layer. Weight tying: embedding and output projection share the same matrix (transposed), saving ~38M parameters.
- **Full architecture assembly:** Seven-step forward pass walkthrough. GptArchitectureDiagram SVG: vertical flow from token IDs to probability distribution. Color-coded consistently with Lesson 5 (violet residual stream, sky blue blocks, amber FFN/output, emerald layer norm, purple embedding). Dimension annotations at every stage. Weight tying annotation with dashed line connecting embedding and output projection.
- **Parameter counting closure:** Token embeddings 38.6M + position embeddings 786K + 12 blocks at (2.36M attention + 4.72M FFN + 3K layer norm) = 85.0M + final layer norm 1.5K + output projection (weight-tied, 0 additional) = 124,356,864 (~124.4M). Matches known GPT-2 figure. Distribution: embeddings ~31%, attention ~23%, FFN ~46%.
- **Encoder-decoder contrast:** Three GradientCards: encoder-only (BERT, bidirectional), encoder-decoder (T5, two stacks + cross-attention), decoder-only (GPT, causal only). "Decoder" is a historical name from the original 2017 Transformer architecture, not a description of capability.
- **Why decoder-only won:** Simplicity (one stack, one attention type, one objective), scaling (next-token prediction scales with data and compute), generality (handles generation AND understanding). GPT-2 vs GPT-3 comparison table showing same architecture at 1400x scale difference.
- **Module completion echo:** Traced the 6-lesson arc: raw attention (feel the limitation) -> Q/K (fix matching) -> V/residual (fix contribution) -> multi-head (capture diversity) -> transformer block (assemble repeating unit) -> causal masking + complete architecture.

**Mental models established:**
- "Causal masking simulates the inference constraint during training" -- future tokens do not exist at inference; the mask ensures the model practices under the same condition
- "The full GPT architecture is assembly, not invention" -- every piece is familiar from prior lessons; this lesson adds one new mechanism and connects everything
- "Decoder-only means causal masking, not 'can only decode'" -- historical name from the original encoder-decoder Transformer
- "Scale, not architecture" -- GPT-2 and GPT-3 are the same architecture with different hyperparameters

**Analogies used:**
- Exam with cardboard sleeve: answer key visible = cheating; sleeve covers future answers; after the exam (inference), no answer key exists
- Data leakage callback from Series 1: model sees the labels during training, validation metrics are meaningless
- MNIST output layer callback: same idea, 10 classes -> 50K classes
- "The simplest architecture that works is the one that scales" -- why decoder-only won

**What was NOT covered (scope boundaries):**
- Implementing causal masking or GPT in PyTorch -- Module 4.3 (building-nanogpt)
- Training, loss curves, learning rate scheduling -- Module 4.3
- KV caching, flash attention, efficient inference -- Module 4.3
- Cross-attention mechanics in detail -- Series 6
- BERT architecture in detail -- mentioned for contrast only
- Finetuning, instruction tuning, RLHF -- Module 4.4
- Mixture of experts, sparse attention

**Misconceptions addressed:**
1. "Causal masking is a training trick that can be removed at inference (like dropout)" -- At inference, future tokens literally do not exist. The mask is not removed; the constraint it simulates becomes reality. Explicit contrast with dropout (on during training, off during inference) and batch norm (uses running averages).
2. "The model processes tokens one at a time during training (like during generation)" -- Training processes the entire sequence in parallel. All positions predict simultaneously. Causal masking makes parallel training safe. ComparisonRow: training (parallel, masked) vs inference (sequential, no future tokens to mask).
3. "'Decoder-only' means the model can only decode (can't understand input)" -- GPT models answer questions, summarize text, follow instructions. The input IS the context processed through all N blocks. The name is historical, referring to the use of causal masking, not a lack of comprehension ability.
4. "The full QK^T matrix is computed and then masked, wasting compute on the upper triangle" -- Acknowledged as valid concern. In practice, implementations fuse masking with attention computation (flash attention). For understanding the architecture, the conceptual picture is correct. Compute optimization deferred to Module 4.3.
5. "GPT-2, GPT-3, and GPT-4 are fundamentally different architectures" -- GPT-2 (124M) and GPT-3 (175B) use the same decoder-only transformer. Differences are scale: layers, d_model, heads, training data. Configuration comparison table proves same blueprint, different numbers.
