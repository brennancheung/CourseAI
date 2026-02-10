# Module 4.2: Attention & the Transformer — Plan

**Module goal:** The student understands how attention works (from raw dot products through full multi-head Q/K/V attention), why each piece exists by feeling the limitation of the version without it, and how the transformer block assembles attention with feed-forward networks, residual connections, and layer norm into the architecture behind modern LLMs.

## Narrative Arc

The student arrives knowing the complete input pipeline: text becomes tokens, tokens become IDs, IDs become embedding vectors, and positional encoding injects order. But those embeddings are static — "bank" always gets the same vector whether it's near "river" or "money." The student already felt this limitation (the polysemy warning from embeddings-and-position). This module is the answer.

The arc follows a deliberate build-up-through-failure pattern:

1. **Feel the bag-of-words problem viscerally** (Lesson 1) — Averaging embeddings throws away word order. Can the tokens themselves decide what matters? Introduce dot-product attention using raw embeddings. It kinda works, but each token has ONE vector that must serve as both "what I'm looking for" and "what I offer." This dual-role tension is the cliffhanger.
2. **Resolve the dual-role problem** (Lesson 2) — Q and K are learned projections that let each token ask different questions than what it advertises. The QK^T matrix becomes a relevance score. Scaling by sqrt(d) prevents softmax saturation.
3. **Separate matching from contributing** (Lesson 3) — Even with Q/K for matching, using raw embeddings as the output conflates "what makes me relevant" with "what information I provide." V is the third projection: what a token contributes when attended to. Full single-head attention is complete. Residual stream introduced.
4. **Multiple notions of relevance** (Lesson 4) — One head captures one kind of relationship. "The cat sat on the mat because it was soft" — "it" needs to attend to "mat" for coreference and to "soft" for meaning. Multi-head attention runs several attention functions in parallel.
5. **The transformer block** (Lesson 5) — Attention reads from the residual stream, FFN writes new information. Residual connections (callback to ResNets) and layer norm stabilize deep stacking. The block is the repeating unit.
6. **The full decoder-only architecture** (Lesson 6) — Causal masking ensures tokens can only attend to past positions (no cheating at next-token prediction). Stack N blocks. This is GPT. Brief encoder-decoder contrast for future reference.

The key pedagogical move: lessons 1-3 are NOT "here are Q, K, V." They are "here's attention without projections (broken) -> here's Q and K fixing the matching problem -> here's V fixing the contribution problem." Each piece arrives because the student felt the limitation of the version without it.

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| the-problem-attention-solves | Dot-product attention on raw embeddings (no Q/K/V) | STRETCH | Must come first — introduces the entire attention mechanism concept. High novelty: first time seeing tokens communicate. The limitation felt here motivates everything that follows. |
| queries-and-keys | Q and K as learned projections, QK^T relevance matrix | BUILD | Directly resolves the dual-role limitation from Lesson 1. Student already has dot-product attention; Q and K are the targeted fix. |
| values-and-attention-output | V projection, full single-head attention, residual stream | BUILD | Completes the attention operation. Depends on Q/K from Lesson 2. Adds one new projection (V) plus the residual stream concept. |
| multi-head-attention | Multiple heads in parallel, concatenation + output projection | STRETCH | Extends single-head to multi-head. Conceptually simple once single-head is solid, but requires thinking about parallel operations and dimension management. |
| the-transformer-block | MHA + FFN + residual connections + layer norm | BUILD | Assembles pieces. Callback to ResNets for residual connections. "Attention reads, FFN writes" is the key new insight. |
| decoder-only-transformers | Causal masking, full GPT architecture | CONSOLIDATE | Wraps everything into the complete architecture. Causal masking is the one genuinely new concept; the rest is assembly and understanding. |

## Rough Topic Allocation

- **Lesson 1 (the-problem-attention-solves):** Context-dependent meaning ("bank" near "river" vs "money"), why bag-of-words (averaging embeddings) fails, weighted average as an improvement (but who decides the weights?), the key insight that the input itself can determine weights via dot products, dot-product attention using raw embedding vectors, attention weight heatmap visualization, the fundamental limitation: one representation per token must serve as both "what I'm seeking" and "what I offer." Notebook: compute raw dot-product attention on tiny examples.

- **Lesson 2 (queries-and-keys):** Recap the dual-role limitation from Lesson 1, the "seeking vs advertising" asymmetry, Q = W_Q @ embedding (what I'm looking for), K = W_K @ embedding (what I'm offering), QK^T as the relevance matrix, why scaling by sqrt(d_k) matters (softmax saturation / vanishing gradients callback), softmax rows to get attention weights, trace the full computation by hand with 4 tokens and 3-dim embeddings. Notebook: implement Q, K projections and compute attention weights.

- **Lesson 3 (values-and-attention-output):** Recap that Q/K solved matching but output still uses raw embeddings, the distinction between "what makes me relevant" (K) and "what information I provide" (V), V = W_V @ embedding as the contribution vector, full formula: softmax(QK^T/sqrt(d))V, residual stream concept (attention output is added to input, not replaced), implement complete single-head attention from scratch. Notebook: full single-head attention implementation with trace-through.

- **Lesson 4 (multi-head-attention):** Why one head isn't enough (different types of relationships in one sentence), each head has its own W_Q, W_K, W_V, heads operate in parallel on lower-dimensional subspaces (d_model / n_heads per head), concatenate outputs + linear projection, visualizing what different heads attend to in practice. Notebook: multi-head attention implementation.

- **Lesson 5 (the-transformer-block):** The "attention reads, FFN writes" framing, residual connections around both MHA and FFN (callback to ResNet skip connections), layer norm (pre-norm vs post-norm), the FFN as a two-layer network with expansion factor, the block as the repeating unit that stacks. Conceptual lesson, no notebook.

- **Lesson 6 (decoder-only-transformers):** Why autoregressive models can't look ahead, causal masking as a triangular matrix applied before softmax, the full GPT architecture: embedding + PE -> N transformer blocks -> output projection, parameter counting, why decoder-only won for LLMs (simplicity, scaling), brief encoder-decoder contrast (original transformer, BERT vs GPT) for context and future Series 6 reference. Conceptual lesson, no notebook.

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| the-problem-attention-solves | STRETCH | First encounter with attention. Three new concepts: context-dependent representations, weighted averaging as computation, dot-product as similarity. New domain of thinking (tokens communicating). |
| queries-and-keys | BUILD | One core concept (learned projections for Q/K) that directly resolves the limitation from Lesson 1. The student already has dot-product attention; this adds one targeted mechanism. |
| values-and-attention-output | BUILD | One new projection (V) plus the residual stream idea. Completes a known-incomplete picture rather than introducing something entirely new. |
| multi-head-attention | STRETCH | Requires thinking about parallel operations, dimension splitting, and why one head isn't enough. Conceptual complexity is high even though each piece is simple. |
| the-transformer-block | BUILD | Assembly of known pieces. Residual connections callback to ResNets (familiar). Layer norm is new but a close cousin of batch norm (from Series 1 and 3). FFN is just a two-layer network (very familiar). |
| decoder-only-transformers | CONSOLIDATE | Causal masking is the one new concept. The rest is putting it all together and stepping back to see the whole architecture. |

## Module-Level Misconceptions

- **"Attention is just a weighted average."** It IS a weighted average, but the weights are computed from the data itself — that's the revolution. The inputs determine what to attend to, making it content-dependent rather than position-dependent (unlike convolutions).

- **"Q, K, V are properties of the token (like part of speech)."** They're learned linear projections of the same embedding. The same token produces different Q, K, V vectors depending on the learned projection matrices. This is the core insight the three-lesson build-up is designed to establish.

- **"Attention replaces the embedding."** Attention output is ADDED to the input via the residual stream. The original information is preserved and enriched, not overwritten.

- **"More heads = better."** Each head operates in a lower-dimensional subspace (d_model / n_heads). More heads means more types of relationships captured but each in a smaller space. It's a tradeoff, not a monotonic improvement.

- **"The transformer is just attention."** The FFN layers contain roughly 2/3 of the parameters and are believed to store factual knowledge. Attention routes information; FFN processes it. Both are essential.

- **"Causal masking is a training trick."** It's fundamental to the autoregressive task. Without it, the model can cheat by looking at the answer. It also makes the model usable for generation — at inference time, future tokens literally don't exist yet.
