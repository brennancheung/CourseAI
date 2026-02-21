# Lesson 2: Long Context & Efficient Attention (long-context-and-efficient-attention) -- Planning Document

**Module:** 5.3 Scaling Architecture
**Position:** Lesson 2 of 3
**Type:** BUILD
**Slug:** long-context-and-efficient-attention

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Full attention formula: output = softmax(QK^T / sqrt(d_k)) V | DEVELOPED | values-and-attention-output (4.2.3) | Student built this formula across three lessons. Can trace every step: project Q, K, V via learned matrices, compute QK^T scores, scale by sqrt(d_k), apply softmax row-wise, weighted average of V vectors. This is the formula that becomes quadratic in sequence length. |
| Multi-head attention: h independent heads in parallel, each with own W_Q^i, W_K^i, W_V^i, outputs concatenated + W_O | DEVELOPED | multi-head-attention (4.2.4) | Student understands dimension splitting (d_k = d_model / h), head specialization is emergent, W_O mixes across heads. GPT-2: 12 heads, d_k=64. Relevant because GQA modifies the K/V head count while preserving Q heads. |
| All-pairs computation in attention (every token computes scores with every other token, no locality window) | DEVELOPED | the-problem-attention-solves (4.2.1) | Student knows attention has no distance preference, explicitly contrasted with CNN local windows. The all-pairs property is what makes attention quadratic. |
| Causal masking (lower-triangular attention matrix, future positions set to -inf before softmax) | DEVELOPED | decoder-only-transformers (4.2.6) | Student knows mechanism, purpose (prevent data leakage), and the training-inference asymmetry. Relevant because causal masking means each token attends to all prior tokens -- grows linearly in the number of tokens that must be attended to. |
| KV caching for autoregressive inference | DEVELOPED | scaling-and-efficiency (4.3.3) | Cache K and V from previous generation steps. O(n) vs O(n^2) without cache. Student understands the compute savings (55x at 100 tokens, 500x at 1000) and the memory tradeoff (37.7 MB per sequence for GPT-2 at 1024). This is the memory cost this lesson will address with GQA. |
| Flash attention (tiled computation, O(n) memory, same math) | INTRODUCED | scaling-and-efficiency (4.3.3) | Student knows flash attention reduces memory from O(n^2) to O(n) but does NOT reduce compute (still O(n^2) FLOPs). Numerically identical to standard attention. The student's understanding that flash attention fixes memory but not compute is a critical prerequisite -- it motivates sparse/linear attention. |
| Sinusoidal positional encoding (multi-frequency waves, unique per position, any-length, deterministic) | DEVELOPED | embeddings-and-position (4.1.3) | Student implemented sinusoidal PE from formula, understands the four requirements (unique, smooth, any-length, deterministic). Clock analogy: second/minute/hour hands at different frequencies. |
| Learned positional encoding (nn.Embedding(max_seq_len, embed_dim), can't generalize beyond training length) | INTRODUCED | embeddings-and-position (4.1.3) | Student knows learned PE is simpler (another embedding table), used by GPT-2, but the key tradeoff: it cannot generalize to unseen sequence lengths. The DNA transfer question established this limitation concretely. |
| RoPE (Rotary Position Embeddings) | MENTIONED | embeddings-and-position (4.1.3) | Single sentence: "Modern alternative encoding relative position between tokens rather than absolute position. Used by LLaMA." Student has name recognition only. This lesson develops it fully. |
| Positional encoding addition to token embeddings (input = embedding + PE) | DEVELOPED | embeddings-and-position (4.1.3) | Student knows position is added (not concatenated) to token embeddings. Same dimension d. This is the additive approach that RoPE replaces with a multiplicative (rotation) approach. |
| Scaling laws (Chinchilla) | INTRODUCED | scaling-and-efficiency (4.3.3) | N_opt ~ sqrt(C), D_opt ~ sqrt(C). Student understands compute-optimal training. Relevant as context for why longer context is desirable (more data per forward pass). |
| Conditional computation (MoE: not every parameter activates for every token) | DEVELOPED | mixture-of-experts (5.3.1) | Previous lesson in this module. Student has the paradigm that not all computation needs to happen for all tokens. This mindset transfers to sparse attention: not all tokens need to attend to all other tokens. |
| Parameter-compute decoupling (total parameters >> active parameters) | DEVELOPED | mixture-of-experts (5.3.1) | The MoE insight of decoupling two quantities the student previously assumed were linked. This lesson makes a parallel decoupling: context length and compute cost do not have to scale quadratically together. |
| Compute-bound vs memory-bound operations | INTRODUCED | scaling-and-efficiency (4.3.3) | GPU compute >> memory bandwidth. Arithmetic intensity determines bottleneck. Relevant because attention at long sequences transitions from memory-bound to compute-bound as n grows. |
| KV cache memory cost | INTRODUCED | scaling-and-efficiency (4.3.3) | GPT-2 KV cache at 1024: ~37.7 MB per sequence, ~1.2 GB for batch of 32. Student understands KV cache grows with sequence length. GQA reduces this proportionally. |

**Mental models and analogies already established:**
- "Attention is a weighted average where the input determines the weights" -- the core attention mechanism
- "Q and K are learned lenses" -- projection matrices, not token properties
- "Multiple lenses, pooled findings" -- multi-head attention as parallel specialists
- "Split, not multiplied" -- dimension splitting in multi-head attention
- "The bottleneck is the delivery truck, not the chefs" -- compute-bound vs memory-bound
- "Flash attention fixes memory, not compute" -- implicit from scaling-and-efficiency
- "Token embedding + positional encoding = the model's input" -- additive PE
- "Clock with many hands" -- sinusoidal PE multi-frequency intuition
- "Without position, embeddings are a bag of words" -- position must be injected
- "The dense transformer activates all knowledge for every token. MoE activates only the relevant knowledge." -- conditional computation from previous lesson

**What was explicitly NOT covered in prior lessons (relevant here):**
- RoPE mechanism (only name-dropped in 4.1.3)
- How any positional encoding interacts with the Q/K dot product (PE was always treated as "add to embedding before attention")
- Context extension beyond training length (the learned PE length limit was identified as a tradeoff but no solution was presented)
- The quadratic compute cost of attention as a bottleneck for long sequences (flash attention's compute limitation was stated but not developed as a problem to solve)
- Sparse attention patterns (sliding window, dilated, block-sparse)
- Linear attention variants
- Grouped Query Attention (GQA) -- sharing K/V across head groups
- The relationship between position encoding method and context window limits

**Readiness assessment:** The student is well-prepared for this lesson. The critical prerequisites are all at sufficient depth: full attention formula (DEVELOPED), multi-head attention (DEVELOPED), KV caching (DEVELOPED), flash attention (INTRODUCED -- the key fact that it fixes memory but not compute is explicitly established), sinusoidal and learned PE (DEVELOPED/INTRODUCED with the length-generalization tradeoff already identified). The BUILD designation is appropriate: RoPE extends existing positional encoding knowledge into a more principled framework, and sparse/linear attention extend the student's understanding of the attention mechanism itself. No genuinely new paradigm is introduced -- unlike MoE's conditional computation, everything here is a modification or optimization of the attention mechanism the student already knows deeply. The MoE lesson (STRETCH) precedes this, so the student gets progressive recovery.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how positional encoding (RoPE), attention pattern restrictions (sparse attention), and KV cache sharing (GQA) address the three bottlenecks that prevent standard attention from scaling to long contexts: position extrapolation failure, quadratic compute cost, and linear KV cache memory growth.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Full attention formula (softmax(QK^T / sqrt(d_k)) V) | DEVELOPED | DEVELOPED | values-and-attention-output (4.2.3) | OK | The lesson modifies how positions are encoded in Q/K and how the attention pattern is restricted. The student must understand the base formula to see what changes. |
| Multi-head attention (h heads, d_k = d_model / h, W_O projection) | DEVELOPED | DEVELOPED | multi-head-attention (4.2.4) | OK | GQA changes the K/V head count relative to Q heads. The student must understand multi-head attention to see what GQA modifies and what it preserves. |
| QK^T as learned asymmetric relevance matrix | DEVELOPED | DEVELOPED | queries-and-keys (4.2.2) | OK | RoPE modifies how position information enters the QK^T dot product. The student must understand QK^T deeply to see how RoPE makes position part of the scoring function rather than part of the input. |
| Scaling by sqrt(d_k) | DEVELOPED | DEVELOPED | queries-and-keys (4.2.2) | OK | The scaled attention formula is the starting point. Sparse attention modifies the softmax step; RoPE modifies the QK^T step. Both build on the existing formula. |
| KV caching mechanism and memory cost | DEVELOPED | DEVELOPED | scaling-and-efficiency (4.3.3) | OK | GQA is directly motivated by KV cache memory cost. The student must understand that each head stores separate K and V caches to see how sharing K/V across head groups reduces memory. |
| Flash attention (O(n) memory, O(n^2) compute) | INTRODUCED | INTRODUCED | scaling-and-efficiency (4.3.3) | OK | INTRODUCED is sufficient. The key fact needed is that flash attention does NOT reduce compute. This motivates why algorithmic alternatives (sparse, linear attention) are needed. No deeper understanding of flash attention's tiling mechanism is required. |
| Sinusoidal positional encoding (multi-frequency waves) | DEVELOPED | DEVELOPED | embeddings-and-position (4.1.3) | OK | RoPE extends the multi-frequency insight from sinusoidal PE. The "clock with many hands" analogy transfers directly. The student's understanding of sinusoidal PE's properties is the foundation for understanding what RoPE improves. |
| Learned PE length limitation (can't generalize beyond max_seq_len) | INTRODUCED | INTRODUCED | embeddings-and-position (4.1.3) | OK | INTRODUCED is sufficient. The key fact is the limitation: learned PE cannot extrapolate. This motivates RoPE's ability to represent relative positions. |
| Dot product as similarity/relevance measure | DEVELOPED | DEVELOPED | the-problem-attention-solves (4.2.1) | OK | RoPE encodes position directly into the dot product. The student must understand what the dot product computes to see how rotating Q and K vectors makes position influence the relevance score. |
| Causal masking (lower-triangular pattern) | DEVELOPED | DEVELOPED | decoder-only-transformers (4.2.6) | OK | Sparse attention patterns (sliding window) are a further restriction beyond causal masking. The student already knows masking restricts which tokens can attend to which. Sparse attention extends this concept. |

All prerequisites are at sufficient depth. No gaps.

### Gap Resolution

No gaps identified. All prerequisites are at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Longer context is just a matter of training with longer sequences" | The student has seen that transformer architecture itself is sequence-length-agnostic (the attention formula works for any n). They might think the only barrier is having long-enough training data. This ignores the positional encoding problem: learned PE has no embedding for unseen positions, and even sinusoidal PE degrades on lengths far beyond training. It also ignores the quadratic compute cost that makes naive long-context prohibitively expensive. | A GPT-2 model trained with max_seq_len=1024 using learned PE cannot process position 1025 -- there is no embedding for it. Even with sinusoidal PE, a model trained on 2K-token sequences has never seen the attention patterns that emerge at 100K tokens. The Q/K dot products at positions far apart have different statistical properties than the model learned during training. Context extension requires both a position encoding that generalizes AND techniques to manage the computational cost. | Hook section -- this is the motivating misconception. Present the three barriers (PE limits, quadratic compute, KV cache memory) before presenting solutions. |
| "Flash attention and KV caching solve the long-context problem" | The student learned both optimizations in scaling-and-efficiency (4.3.3) and may think the long-context problem is already solved. Flash attention reduces attention memory from O(n^2) to O(n). KV caching eliminates redundant computation during generation. Together, they seem to handle the problem. | Flash attention produces the exact same result as standard attention -- numerically identical (the student verified this with torch.allclose). The O(n^2) compute remains. At 100K tokens with 96 heads, the attention computation alone requires ~3.8 * 10^14 FLOPs per layer. Flash attention makes this fit in memory; it does not reduce the computation. KV caching helps generation (sequential) but does not help the prefill step or training, where the full context must be processed. The memory cost of the KV cache itself grows linearly with sequence length: at 128K context for a 70B model, the KV cache alone can consume 40+ GB. | After the hook, when establishing the quadratic bottleneck. Explicitly name flash attention and KV caching as "necessary but insufficient" and state what they do and do not fix. |
| "RoPE is just another positional encoding scheme (interchangeable with sinusoidal or learned PE)" | The student has seen sinusoidal and learned PE as alternatives that are added to embeddings. RoPE might seem like just another option in the same category -- a different formula for the same purpose. This misses the fundamental difference: sinusoidal/learned PE add position to the input before attention, while RoPE encodes position directly into the Q/K dot product, making the attention score itself position-aware. | Replace RoPE with sinusoidal PE in a model and try to extend context from 4K to 32K: performance degrades rapidly because the additive PE was not trained for those positions. RoPE encodes the relative distance between tokens into the Q/K dot product itself. The score between tokens at positions i and j depends on (i - j), not on the absolute positions. This means a model trained at 4K can potentially attend across longer distances because the relative position patterns it learned still apply -- the dot product between positions 1000 and 1005 uses the same rotation as between positions 50000 and 50005. Sinusoidal PE cannot do this because the position information is mixed into the embedding before projection. | During the RoPE explanation, after establishing the mechanism. Explicitly contrast the additive (sinusoidal/learned) vs multiplicative (RoPE) approaches and what this difference means for context extension. |
| "Sparse attention loses too much information to be useful (you need full attention for good performance)" | The student has spent 6 lessons building up full attention as the core mechanism. Every attention lesson emphasized the all-pairs computation -- "a token can attend just as strongly to a word 20 positions away as to its immediate neighbor" (4.2.1). They may resist the idea of restricting which tokens can attend to which, assuming this would cripple the model's ability to capture long-range dependencies. | Sliding window attention in Mistral-7B (window size 4096) achieves performance competitive with models using full attention. The key insight: most attention weight is local. In trained transformer models, the attention distribution is heavily concentrated on nearby tokens for most heads -- tokens rarely need to attend strongly to tokens thousands of positions away. A few heads specialize in longer-range patterns, but even those often attend to specific structural positions (beginning of document, paragraph boundaries) rather than requiring true all-pairs access. Stacked layers of local attention can propagate information across the full context: layer 1 sees positions [i-4K, i], layer 2 sees positions [i-8K, i] through the residual stream. | After the quadratic bottleneck section, when introducing sparse attention as a solution. Frame it as "trading expressiveness for efficiency" with concrete evidence that the tradeoff is favorable. |
| "GQA is just using fewer attention heads (reducing h)" | The student knows dimension splitting: d_k = d_model / h. Reducing heads means increasing d_k, changing the quality-diversity tradeoff. GQA might seem like the same thing with a different name. The critical distinction is that GQA reduces K/V heads while keeping Q heads unchanged -- the query diversity is preserved, only the KV cache memory is reduced. | With h=32 Q heads and GQA groups of 8, there are 32 Q heads but only 4 K/V heads (each shared across 8 Q heads). Each Q head still has its own W_Q^i, producing different queries that capture diverse relevance patterns. The K/V reduction saves 8x on KV cache memory (4 K/V pairs vs 32). If you simply reduced to 4 heads, you would have 4 Q heads with d_k = d_model/4 -- far fewer queries, less diversity, different quality tradeoffs. GQA preserves query diversity while reducing KV memory. It is not fewer heads; it is selective sharing of the KV component. | During the GQA section. Show the architecture difference explicitly: MHA (32 Q, 32 KV) vs GQA (32 Q, 4 KV) vs MQA (32 Q, 1 KV), with the same total Q capacity but different KV memory costs. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Quadratic attention cost at increasing sequence lengths: concrete FLOPs calculation for 1K, 4K, 32K, 128K tokens | Positive | Makes the O(n^2) bottleneck visceral with real numbers. At 1K tokens, attention is cheap. At 128K, it is 16,384x more expensive. Shows why the problem becomes critical at the sequence lengths modern applications demand. | Abstract "O(n^2)" does not convey urgency. Concrete numbers showing the 128x factor from 1K to 128K transforms the problem from theoretical to practical. The student can see exactly where the wall is. |
| RoPE rotation of Q and K vectors: two tokens at positions 3 and 7 vs positions 100 and 104 producing the same dot product | Positive | Demonstrates the relative position property -- the core insight of RoPE. Same relative distance (4 positions apart) produces the same rotation-adjusted dot product regardless of absolute position. This is what enables context extension: patterns learned at training length transfer to longer sequences. | Makes the abstract "relative position" concrete. The student can trace the math: rotate Q by position i, rotate K by position j, the dot product depends on (i-j). The worked example with two position pairs shows this is not just a claim but a mathematical property. |
| Sliding window attention on a long document: a token at position 50,000 cannot directly attend to position 1, but information can flow through intermediate tokens across layers | Negative (turns positive) | Shows the tradeoff of sparse attention honestly -- direct long-range attention is lost -- but then shows how stacked layers recover it. Layer 1 propagates information from position 1 to position 4096 (via residual stream), layer 2 propagates from 4096 to 8192, and so on. After ~13 layers with window 4096, information from position 1 has reached position 50,000. Disproves "sparse attention cannot capture long-range dependencies" while honestly acknowledging the indirection. | The student needs to see both the cost and the recovery mechanism. An example that only shows the restriction would reinforce the misconception that sparse attention is crippling. An example that only shows the recovery would be dishonest about the tradeoff. This example presents both, giving the student an accurate mental model. |
| GQA KV cache memory comparison: MHA vs GQA vs MQA for a 70B model at 128K context | Positive (stretch) | Extends the student's KV cache memory understanding (GPT-2 at 1024) to frontier model scale. Concrete numbers: MHA with 64 heads at 128K context = X GB of KV cache. GQA with 8 KV groups = X/8 GB. MQA with 1 KV head = X/64 GB. Makes the practical impact tangible for production serving. | Connects GQA to the real-world deployment constraint the student already understands (KV cache memory from 4.3.3). The 8x memory reduction is not abstract -- it determines how many concurrent users a serving system can handle. LLaMA 2 70B uses GQA with 8 KV groups; this is not a theoretical technique. |

---

## Phase 3: Design

### Narrative Arc

The previous lesson solved the parameter-compute problem: MoE lets the model store more knowledge without proportionally increasing per-token cost. But there is a second scaling bottleneck the student has been accumulating hints about without a direct confrontation. The attention formula they built across four lessons in Module 4.2 -- softmax(QK^T / sqrt(d_k)) V -- has a property they noticed early: every token computes a score with every other token. At 1024 tokens (GPT-2's context), this is manageable. At 128K tokens (what modern applications demand for processing long documents, codebases, or multi-turn conversations), the attention computation is 16,384 times more expensive. Flash attention, which the student already knows, reduces memory but not compute -- it still performs every one of those n^2 dot products. KV caching, which the student also knows, helps generation but does not help the initial processing of a long context. And the positional encoding methods the student learned (sinusoidal and learned PE) have their own limits: learned PE cannot represent positions beyond training length at all, and even sinusoidal PE degrades on out-of-distribution lengths. The student is facing three interrelated barriers to long context -- position encoding that fails to generalize, compute that grows quadratically, and KV cache memory that grows linearly per head -- and none of the tools they have solve all three. This lesson addresses each barrier with a targeted architectural innovation: RoPE for position generalization, sparse attention for compute reduction, and GQA for KV cache compression.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Verbal/Analogy** | RoPE as "encoding position in the handshake, not the nametag." Sinusoidal/learned PE add position to the embedding (the nametag -- what the token carries everywhere). RoPE rotates Q and K vectors so that position is encoded in the dot product itself (the handshake -- the interaction between two tokens). The handshake between two people 3 seats apart feels the same regardless of which seats they are in. | Maps to a concrete social experience. The nametag-vs-handshake distinction captures the additive-vs-multiplicative difference between traditional PE and RoPE. The "3 seats apart" detail maps to relative position invariance. |
| **Visual (inline SVG)** | Three-barrier diagram showing the three bottlenecks side by side: (1) Position encoding wall (learned PE stops at max_seq_len, sinusoidal degrades), (2) Quadratic compute wall (O(n^2) FLOPs curve), (3) KV cache memory growth (linear per head, multiplied by num_heads). Each barrier labeled with the solution: RoPE, sparse/linear attention, GQA. | The lesson addresses three related but distinct problems. A visual overview at the start gives the student a map of where the lesson is going and how the three solutions connect. Prevents the "random bag of techniques" feeling. |
| **Visual (inline SVG)** | RoPE rotation diagram: two 2D vectors (q, k) in the embedding subspace, rotated by angles proportional to their positions. The dot product between the rotated vectors depends on the angular difference (relative position), not the absolute angles. Show two pairs: (position 3, position 7) and (position 100, position 104), both with the same relative rotation and therefore the same dot product contribution from position. | RoPE's mechanism is inherently geometric -- rotation in 2D subspaces. A visual showing the rotation makes the math intuitive before the formula. The student can see that the angle between two rotated vectors depends on the difference in rotation angles, which is the relative position. |
| **Visual (inline SVG)** | Attention pattern comparison: full causal attention (lower-triangular, all dark), sliding window (diagonal band), dilated (periodic gaps in the band), and their combination. Color intensity = attention weight. Each pattern labeled with compute cost: O(n^2), O(n*w), O(n*w) where w << n. | Sparse attention patterns are spatial -- they define which token pairs can interact. Side-by-side comparison makes the restriction visible and the compute savings intuitive: the dark area (computed dot products) is dramatically smaller for sparse patterns. |
| **Symbolic/Code** | RoPE pseudocode showing the rotation applied to Q and K vectors in 2D subspaces. Pair consecutive dimensions, apply rotation matrix [[cos(m*theta), -sin(m*theta)], [sin(m*theta), cos(m*theta)]] where m is the position index and theta varies by dimension pair (high frequency for early pairs, low for later). The dot product q_rot^T k_rot = q^T R(m-n) k shows the relative position property. | Connects to the student's code skills and makes the mechanism concrete. The rotation matrix is a familiar linear algebra concept. The pseudocode shows that RoPE is applied per dimension pair, not to the whole vector -- connecting to the sinusoidal PE's "clock with many hands" at different frequencies. |
| **Concrete example** | GQA architecture comparison with concrete numbers: LLaMA 2 70B uses 64 Q heads and 8 KV groups. At 128K context length with d_k=128, calculate KV cache per head, total MHA cache (64 KV heads), total GQA cache (8 KV groups). Show the 8x reduction. | Makes the KV cache savings tangible with a real architecture. The student has already computed KV cache costs for GPT-2 at 1024 in scaling-and-efficiency. This extends that calculation to a production model at production context lengths, showing why GQA is not optional at scale. |
| **Intuitive** | The "of course" beat for sparse attention: "You already knew from Module 4.2 that trained attention heads develop different specialization patterns. Some attend to the previous token. Some track syntax. Some are near-uniform. Very few heads attend strongly to tokens thousands of positions away. If most attention weight is local, of course you should not compute scores for distant pairs that will receive near-zero weight anyway. Of course you should let the model focus its compute budget where attention actually concentrates." | Three established facts combine: head specialization is messy (4.2.4), most weight is local in practice (implicit from attention matrix observations), and compute resources are finite (4.3.3). The "of course" moment makes sparse attention feel obvious rather than lossy. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3
  1. RoPE (rotary position embeddings) -- encoding relative position into the Q/K dot product via rotation, enabling context extension beyond training length
  2. Sparse attention patterns (sliding window, dilated) -- restricting which token pairs compute attention scores, trading full expressiveness for subquadratic compute
  3. Grouped Query Attention (GQA) -- sharing K/V heads across groups of Q heads, reducing KV cache memory while preserving query diversity
- **Previous lesson load:** STRETCH (mixture-of-experts introduced the conditional computation paradigm)
- **This lesson's load:** BUILD -- appropriate. All three concepts extend existing knowledge rather than introducing a new paradigm:
  - RoPE extends sinusoidal PE (same multi-frequency idea, different mechanism for injecting position)
  - Sparse attention extends the attention matrix concept (restricting which entries are computed)
  - GQA extends multi-head attention (modifying the head sharing pattern)
  The student is not changing their mental model of what attention is -- they are learning modifications that make it work at scale. No single concept requires the kind of mental model overhaul that conditional computation required in MoE.
- **Load trajectory (cross-module):** STRETCH (mixture-of-experts) -> BUILD (this lesson). Progressive recovery after the STRETCH lesson. Appropriate spacing.
- **Risk assessment:** Three new concepts is at the upper limit. Mitigation: RoPE and sparse attention are the primary concepts with full development. GQA is the third concept and will be developed but more concisely, leveraging the multi-head attention knowledge already at DEVELOPED depth. The three concepts also address three distinct bottlenecks, giving them a natural organizational structure (not three competing ideas but three solutions to three problems).

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| Sinusoidal PE with multi-frequency waves (4.1.3) | RoPE uses the same multi-frequency insight -- different dimension pairs rotate at different frequencies, like the "clock with many hands." But sinusoidal PE adds position to the embedding; RoPE rotates Q and K directly. The student's "clock with many hands" analogy extends: the hands now rotate the query and key vectors instead of adding to the embedding. |
| Learned PE length limitation (4.1.3) | RoPE's relative position encoding solves the extrapolation problem the student identified in 4.1.3. The DNA transfer question established that learned PE cannot handle unseen positions. RoPE can, because the rotation for relative distance 5 is the same regardless of absolute position. |
| QK^T as learned relevance matrix (4.2.2) | RoPE modifies how position enters the QK^T computation. Instead of position being baked into the embedding before Q/K projection, position is applied after projection, directly inside the dot product. The student's understanding that QK^T computes "how relevant is token j to token i" extends: with RoPE, this relevance naturally incorporates "how far apart are they." |
| Multi-head attention with dimension splitting (4.2.4) | GQA modifies the head structure: Q heads remain independent but K/V heads are shared across groups. The student's understanding of d_k = d_model / h and independent head weights is the foundation. GQA says: "Keep the query diversity (32 Q heads), reduce the KV redundancy (group into 4-8 KV sets)." |
| Flash attention reduces memory but not compute (4.3.3) | This is the explicit bridge. Flash attention was presented as an optimization for attention. This lesson starts where flash attention stops: "Memory is fixed. Compute is not. What fixes compute?" The answer: sparse and linear attention. |
| KV caching and its memory cost (4.3.3) | GQA is directly motivated by the KV cache cost the student computed. At GPT-2 scale (12 heads, 1024 context), KV cache was manageable (~37.7 MB). At LLaMA-70B scale (64 heads, 128K context), it is untenable without sharing. |
| All-pairs computation contrasted with CNN locality (4.2.1) | Sparse attention reintroduces a locality constraint -- not the fixed, hard-coded locality of CNN filters, but a learned or designed locality pattern. This is a callback to the CNN contrast, now with a twist: locality is not always bad, especially when most attention weight is local anyway. |
| Causal masking as attention pattern restriction (4.2.6) | Sparse attention is a further restriction on the attention pattern beyond causal masking. The student already knows how to restrict attention (set entries to -inf before softmax). Sparse attention uses the same mechanism with a different mask pattern (sliding window instead of lower-triangular). |
| Conditional computation from MoE (5.3.1) | Sparse attention applies the same principle to attention that MoE applied to FFN: do not compute everything for every token. MoE says "activate only the relevant experts." Sparse attention says "compute scores only for potentially relevant token pairs." The "not all computation is necessary" insight transfers. |

**Analogies from prior lessons that can be extended:**
- "Clock with many hands" (sinusoidal PE) -> RoPE uses the same multi-frequency structure, but the hands rotate the Q/K vectors instead of adding to the embedding
- "Split, not multiplied" (multi-head) -> GQA takes this further: split Q into many heads for diversity, but share K/V across groups for memory efficiency
- "The bottleneck is the delivery truck" (compute-bound vs memory-bound) -> KV cache memory is the delivery truck for long-context serving; GQA makes the truck bigger by reducing per-head storage
- "Flash attention fixes memory, not compute" -> this lesson picks up exactly where that statement left off

**Analogies from prior lessons that could be misleading:**
- "Attention computes all-pairs scores" (4.2.1) -- the student spent multiple lessons building the mental model that all-pairs is a defining feature of attention. Sparse attention restricts this. Important to frame sparse attention as a pragmatic optimization with empirical validation, not as "breaking" attention.
- "Every head has its own W_Q, W_K, W_V" (4.2.4) -- GQA changes this for K and V. Important to be explicit that Q heads remain independent; only K/V sharing changes.

### Scope Boundaries

**This lesson IS about:**
- The three bottlenecks preventing standard attention from scaling to long contexts (PE limits, quadratic compute, KV cache memory)
- RoPE: how it encodes relative position into the Q/K dot product via rotation in 2D subspaces, why this enables context extension, and how it differs from additive PE
- Context extension via RoPE (the principle: relative position patterns learned at training length transfer to longer sequences)
- The quadratic attention bottleneck with concrete cost calculations
- Sparse attention patterns: sliding window and dilated, with compute cost analysis
- Linear attention as a concept (kernel trick replacing softmax(QK^T)V with phi(Q)(phi(K)^T V), reducing to O(n) compute)
- GQA: sharing K/V across groups of Q heads, KV cache memory reduction, the MHA-GQA-MQA spectrum
- Real-world usage: LLaMA 2/3 (GQA + RoPE), Mistral (sliding window + GQA + RoPE)

**This lesson is NOT about:**
- Implementing RoPE, sparse attention, or GQA in code (conceptual lesson; notebook demonstrates concepts on small proxies)
- The exact NTK-aware scaling or YaRN context extension formulas (MENTIONED at most for context extension techniques beyond basic RoPE)
- ALiBi (Attention with Linear Biases) in depth (named as alternative, not developed)
- Ring attention or sequence parallelism (deferred to Lesson 3: training-and-serving-at-scale)
- Flash attention implementation details (already INTRODUCED in 4.3.3; referenced but not re-taught)
- State space models (SSMs) or Mamba as attention alternatives (out of scope for this module)
- Multi-Head Latent Attention (MLA) as used in DeepSeek (MENTIONED at most)
- Benchmarking or comparing specific model performances on long-context tasks
- Training with very long sequences (curriculum, data preparation)

**Target depths:**
- RoPE mechanism (rotation in 2D subspaces, relative position encoding in dot product): DEVELOPED (can explain the mechanism, trace the rotation, articulate why it enables context extension)
- Context extension via RoPE (relative position patterns transfer to longer sequences): INTRODUCED (knows the principle and why it works, but not the specific extension techniques like NTK-aware scaling)
- Quadratic attention bottleneck: DEVELOPED (can compute the cost at different sequence lengths, explain why flash attention does not fix it)
- Sparse attention patterns (sliding window, dilated): DEVELOPED (can explain the mechanism, articulate the expressiveness-efficiency tradeoff, explain how stacked layers recover long-range information)
- Linear attention: INTRODUCED (knows the concept of kernel-based reformulation for O(n) compute, understands the tradeoff with full attention, but not the mathematical details of kernel functions)
- GQA (sharing K/V across Q head groups): DEVELOPED (can explain the mechanism, compute KV cache savings, articulate the MHA-GQA-MQA spectrum)

### Lesson Outline

**1. Context + Constraints**
What this lesson is about: the three barriers that prevent the attention mechanism from scaling to long contexts, and the targeted architectural innovations that address each one. What we are NOT doing: implementing these techniques in production code, covering flash attention internals (already taught), or replacing attention entirely with alternative architectures. This is a BUILD lesson -- extending the attention mechanism the student knows deeply with modifications that make it work at 100K+ token scale.

**2. Recap**
Brief reconnection to three facts from prior lessons:
1. The attention formula: softmax(QK^T / sqrt(d_k)) V -- "every token computes a score with every other token" (4.2.1)
2. Positional encoding: sinusoidal adds position to embeddings, learned PE cannot generalize beyond training length (4.1.3). "Remember the DNA transfer question? Learned PE has no embedding for position 2049."
3. Flash attention reduces memory from O(n^2) to O(n) but compute stays O(n^2). KV caching helps generation but costs memory per head per token (4.3.3). "Flash attention and KV caching are optimizations within the existing framework. They do not change the algorithmic cost."

Connect: "What happens when we push the attention mechanism to 128K tokens? Three things break."

**3. Hook (three walls)**
Present the three barriers concretely, one at a time.

**Wall 1 -- Position:** A model trained with learned PE at max_seq_len=4096. Token at position 4097 has no positional encoding. The model literally cannot represent this position. Even sinusoidal PE, which can generate any position, has never been trained on the attention patterns that emerge at 100K -- the model's Q/K projections were trained on much shorter distances.

**Wall 2 -- Compute:** Concrete FLOPs table. Attention FLOPs ~ 2 * n^2 * d_model per layer. At n=1K: ~1.5 billion. At n=4K: ~24 billion. At n=32K: ~1.5 trillion. At n=128K: ~24 trillion. 16,384x increase from 1K to 128K. "Flash attention makes this fit in memory. It does not make this computation go away."

**Wall 3 -- Memory:** KV cache for a 70B model with 64 heads, d_k=128, at 128K context. Per-head KV cache: 2 * 128K * 128 * 2 bytes (bf16) = ~65 MB. Total: 64 heads * 80 layers * ~65 MB = ~332 GB. More than the model weights. "The cache is now larger than the model itself."

GradientCard: "Three walls. Three different problems. Three different solutions. Position encoding that generalizes. Attention that scales subquadratically. KV cache that compresses."

**4. Explain Part 1 -- RoPE (Rotary Position Embeddings)**
Problem-first: existing PE approaches add position to the embedding *before* Q/K projection. This means position information gets mixed with semantic information during projection. What if position could be injected *directly* into the Q/K dot product?

The key insight: rotate Q and K vectors by an angle proportional to their position. When computing q_i^T k_j (the attention score), the rotation causes the result to depend on the *relative* position (i - j), not the absolute positions.

Analogy: "Position encoded in the handshake, not the nametag." Sinusoidal/learned PE puts position on the nametag (the embedding). RoPE encodes position in the handshake (the Q/K dot product). Two people 3 seats apart feel the same handshake regardless of which seats they are in.

RoPE rotation diagram (inline SVG): show 2D subspace with vectors q and k. Position i rotates q by angle i*theta. Position j rotates k by angle j*theta. The dot product between the rotated vectors depends on the angle difference (i-j)*theta.

Mechanism walkthrough: pair consecutive dimensions of Q and K vectors into 2D subspaces. Apply rotation matrix R(m*theta_d) to each 2D pair, where m is the position index and theta_d varies by dimension pair (high frequency for early dimensions, low frequency for later dimensions). Connect to "clock with many hands": "Same multi-frequency structure as sinusoidal PE. Different dimension pairs capture position at different scales -- some rotate fast (local position), others rotate slow (global position)."

Pseudocode for RoPE rotation showing the dimension pairing and rotation application.

Address misconception 3: "RoPE is not just another encoding scheme. Sinusoidal and learned PE add position to the embedding -- they modify the input to attention. RoPE rotates Q and K after projection -- it modifies the dot product itself. This difference is what enables context extension."

**Why RoPE enables context extension:** The rotation for relative distance d is always the same: R(d*theta). A model trained at 4K context learns what the rotation for distance 5, distance 100, distance 2000 looks like. At inference with 32K context, distances 5, 100, 2000 still use the same rotations. Distances beyond 4K use rotations the model has not seen -- but interpolation and extension techniques (MENTIONED: NTK-aware scaling, YaRN) adjust the frequency base to smooth the transition. The key point is that RoPE gives something to extend; learned PE gives nothing.

**5. Check 1 (predict-and-verify)**
Two models trained on 4K-token sequences. Model A uses learned PE. Model B uses RoPE. Both are asked to process an 8K-token document.

Questions:
- What happens to Model A at position 4097? (No PE for that position -- embedding lookup fails or falls back to the last position)
- What happens to Model B at position 4097? (RoPE rotation is applied normally -- theta * 4097 is computable. The model has seen relative distances up to 4K but not absolute position 4097. The rotation is novel but the relative patterns for nearby tokens are familiar.)
- Which model has a better chance of handling the 8K document? Why?
- Does Model B work perfectly at 8K with no further modification? (Not necessarily -- positions beyond training length may still degrade, but RoPE provides a foundation for extension techniques that learned PE cannot.)

Details/summary reveal.

**6. Explain Part 2 -- The Quadratic Bottleneck and Sparse Attention**
Address misconception 2 explicitly: "Flash attention and KV caching are essential optimizations. But they do not change the fundamental algorithm. Flash attention computes the same O(n^2) dot products -- it just does so without materializing the full attention matrix. To reduce the actual computation, we need to compute fewer dot products."

Transition from MoE: "MoE said: not every FFN parameter needs to activate for every token. Sparse attention says something similar: not every token pair needs to compute an attention score."

Present the quadratic cost visually. Attention pattern comparison (inline SVG): full causal attention (lower-triangular, all dark) vs sliding window (diagonal band) vs dilated (periodic gaps). Color intensity = attention weight.

**Sliding window attention:** Each token attends to at most w preceding tokens (plus itself). Cost: O(n * w) instead of O(n^2). For w=4096 and n=128K, this is 32x cheaper. Used by Mistral-7B.

"Of course" beat: "You already knew from Module 4.2 that trained attention heads develop messy specialization patterns. Some track previous tokens. Some attend to nearby structure. Very few heads attend strongly to tokens thousands of positions away. If most weight is local, of course you should skip the distant computations."

Address misconception 4: "But what about long-range dependencies?" The negative-turns-positive example: token at position 50K cannot directly attend to position 1 through a sliding window of 4096. But through the residual stream across multiple layers: layer 1 propagates information from position 1 to ~4096, layer 2 from ~4096 to ~8192, and by layer 13 (13 * 4096 = ~53K), information from position 1 has reached position 50K. Stacked local attention approximates global attention. "The same residual stream that carried information through your GPT-2 carries it across layers of sparse attention."

**Dilated attention (brief):** Like dilated convolutions, skip positions at regular intervals. Captures longer-range patterns without the compute of full attention. Often combined with sliding window in different heads or layers.

**Linear attention (brief, INTRODUCED depth):** The kernel trick: replace softmax(QK^T) with phi(Q) * phi(K)^T, where phi is a feature map. Exploit associativity: instead of computing n^2 scores and then multiplying by V, compute phi(K)^T V first (d_k x d_v, independent of n) and then multiply by phi(Q). Cost drops from O(n^2 * d_k) to O(n * d_k^2). Tradeoff: phi approximates the softmax kernel, introducing approximation error. Standard softmax attention remains more expressive for tasks requiring sharp, position-dependent attention patterns. Mentioned but not deeply developed.

**7. Explain Part 3 -- Grouped Query Attention (GQA)**
Transition from compute to memory. "Sparse attention addresses the compute wall. What about the memory wall -- the KV cache?"

Recall multi-head attention: each head has its own W_Q^i, W_K^i, W_V^i. During generation, each head stores its own K and V cache. At 64 heads, 128K context, d_k=128 -- the KV cache consumes hundreds of GB.

GQA insight: multiple Q heads can share a single K/V pair. Queries still need diversity (different W_Q^i per head) but keys and values have significant redundancy across heads. Share K/V across groups of Q heads.

Address misconception 5: "GQA is not just using fewer heads. All 32 (or 64) Q heads remain, each with their own projection. Only the K/V heads are shared across groups."

Three-way comparison (architecture diagram or table):
- **MHA (Multi-Head Attention):** h_Q = h_KV = 32. Full independence. Largest KV cache.
- **GQA (Grouped Query Attention):** h_Q = 32, h_KV = 8 (4 Q heads per KV group). 4x KV cache reduction.
- **MQA (Multi-Query Attention):** h_Q = 32, h_KV = 1 (all Q heads share one KV). 32x KV cache reduction but more quality degradation.

Concrete KV cache calculation for LLaMA 2 70B (GQA with 8 KV groups): compare with the full MHA calculation from the hook. 8x reduction in KV cache memory.

Connection to multi-head attention: "In Module 4.2, you learned that heads share input but have completely independent weights. GQA says: keep the query diversity (each head's W_Q^i is unique) but acknowledge that K/V across heads are often similar enough to share."

**8. Check 2 (transfer question)**
A team is building a document analysis system that processes 64K-token legal documents. They have a 7B model with:
- Learned positional encoding (max 4096)
- Full multi-head attention (32 heads)
- No special efficiency techniques beyond flash attention

They want to handle 64K tokens. What needs to change?

Student should identify:
1. Replace learned PE with RoPE (or apply context extension) -- the model cannot represent positions beyond 4096 with current PE
2. Add sparse attention (sliding window) -- at 64K, quadratic compute is 256x more expensive than at 4K; flash attention handles memory but not compute
3. Add GQA -- at 64K, KV cache with 32 heads would be very large; sharing KV across groups reduces serving memory
4. These are three independent modifications addressing three independent bottlenecks -- they combine, not replace each other

Details/summary reveal with the three-barrier framework from the hook.

**9. Practice -- Notebook Exercises (Colab)**
`notebooks/5-3-2-long-context-and-efficient-attention.ipynb` (4 exercises)

- **Exercise 1 (Guided): RoPE rotation on 2D vectors.** Implement the 2D rotation matrix. Rotate pairs of vectors at different absolute positions but the same relative distance. Compute dot products and verify that the relative position property holds: dot(rotate(q, pos_i), rotate(k, pos_j)) depends only on (i-j). Visualize the rotation in 2D for 3-4 different relative distances. Predict-before-run: "Will the dot product change if we shift both positions by 1000?" First pair fully worked with visualization code. Insight: position enters the dot product through rotation, and the result depends on relative distance.

- **Exercise 2 (Supported): Sparse attention mask patterns.** Implement full causal attention, sliding window attention (window size w), and dilated attention masks for a 64-token sequence. Visualize all three as heatmaps. Compare the number of computed entries (dark cells) for each pattern. Extend to 256 tokens and plot compute scaling. First causal mask and sliding window mask implementations provided. Insight: sparse patterns dramatically reduce the number of computed attention scores while preserving local structure.

- **Exercise 3 (Supported): GQA forward pass.** Implement a GQA layer with h_Q=8, h_KV=2 (4 Q heads per KV group). Run a forward pass on a batch of 16 tokens with d_model=64. Compare output shape to standard MHA. Count KV cache parameters: MHA (8 KV heads) vs GQA (2 KV groups). Router provided, student implements the grouped key/value expansion and attention computation. Insight: GQA produces the same output shape as MHA with 4x fewer KV parameters.

- **Exercise 4 (Independent): Attention cost calculator.** Build a function that computes the total attention FLOPs and KV cache memory for a given model configuration (num_layers, d_model, num_heads, num_kv_heads, seq_len, dtype). Compute costs for: (a) GPT-2 at 1K, (b) LLaMA 2 70B with MHA at 4K, (c) LLaMA 2 70B with GQA at 4K, (d) LLaMA 2 70B with GQA at 128K, (e) same with sliding window (w=4096) at 128K. Present results in a table. No skeleton provided. Insight: the three optimizations (RoPE for position, sparse for compute, GQA for memory) address different bottlenecks and their benefits compound.

Exercises are independent in code. Progression: RoPE mechanics -> sparse attention patterns -> GQA architecture -> cost analysis integrating all three.

**10. Summarize**
Key takeaways:
1. Three barriers prevent standard attention from scaling to long contexts: position encoding limits, quadratic compute cost, and KV cache memory growth
2. RoPE encodes position in the Q/K dot product via rotation, making attention scores depend on relative position -- this enables context extension beyond training length
3. Sparse attention (sliding window, dilated) computes scores for only a subset of token pairs, reducing O(n^2) to O(n*w). Information propagates across the full context through stacked layers
4. GQA shares K/V heads across groups of Q heads, reducing KV cache memory while preserving query diversity -- the MHA-GQA-MQA spectrum trades memory for quality
5. These three innovations address three independent bottlenecks and combine in practice: LLaMA and Mistral use RoPE + GQA + sparse attention together

Echo the mental model: "Position in the handshake, not the nametag. Compute where attention concentrates, not everywhere. Cache what's needed, not everything. Three barriers, three targeted solutions."

**11. Next Step**
"We have now addressed two scaling bottlenecks: MoE decouples parameters from per-token compute, and long-context techniques decouple context length from quadratic cost. But building and serving a model with trillions of tokens of training data, hundreds of billions of parameters distributed across hundreds of GPUs -- that is an engineering challenge beyond any single machine. Next: how parallelism strategies distribute training across devices, and how speculative decoding and continuous batching make serving practical."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (7 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (2 positive + 1 negative-turns-positive + 1 stretch)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (3: RoPE, sparse attention, GQA)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 5
- Polish: 4

### Verdict: NEEDS REVISION

No critical findings — the lesson works and the student will not be lost or form wrong mental models. However, five improvement-level findings would meaningfully strengthen the lesson. Another pass is needed after addressing them.

### Findings

#### [IMPROVEMENT] — Linear attention section drops the student with a cold formula

**Location:** Sparse Attention section, final three paragraphs (lines ~1511-1535)
**Issue:** The linear attention explanation leaps directly into symbolic notation (`phi(Q)(phi(K)^T V)`, associativity, `d_k x d_v` matrix) without any concrete motivation, analogy, or example. The preceding sections (sliding window, dilated) use the established pedagogical pattern: problem statement, intuitive explanation, concrete example, then formalism. Linear attention skips straight to the kernel trick reformulation. The planning document specifies INTRODUCED depth with "knows the concept of kernel-based reformulation for O(n) compute," but the current text assumes familiarity with kernel methods and associativity of matrix multiplication in a way the student has not practiced.
**Student impact:** The student follows the sliding window explanation well (grounded in the masking mechanism they know), then hits a wall of symbolic manipulation. They likely skim the linear attention paragraph and take away only "there is an O(n) variant with tradeoffs" — which is fine for INTRODUCED depth, but the journey to get there feels like hitting a brick wall rather than a gentle mention.
**Suggested fix:** Restructure as a brief two-step introduction: (1) State the idea in plain language first: "What if we could reorder the computation so we never form the n x n matrix? Instead of computing all pairwise scores first and then averaging V vectors, compute a summary of K and V first and then let each Q token query the summary." (2) Then optionally show the symbolic reformulation. The "compute a summary first" framing connects to the student's existing mental model of attention as "weighted average of V" — they can understand why a compressed summary of K^T V would be useful. The formula can follow as confirmation, not as the primary explanation.

#### [IMPROVEMENT] — Notebook Exercise 4 does not test the planned concept

**Location:** Notebook Exercise 4 (planning doc: "Attention Cost Calculator"; built notebook: "Context Extension Experiment")
**Issue:** The planning document specifies Exercise 4 as an Independent exercise where the student builds a cost calculator function computing FLOPs and KV cache memory for various model configurations (GPT-2, LLaMA 2 70B with MHA, GQA, sliding window). The built notebook instead implements a "Context Extension Experiment" testing attention entropy at varying sequence lengths with/without RoPE and position interpolation. These are entirely different exercises. The cost calculator would have integrated all three barriers and their solutions into a single quantitative comparison — the planned "insight: the three optimizations address different bottlenecks and their benefits compound." The context extension experiment only exercises RoPE concepts and does not address sparse attention or GQA at all.
**Student impact:** The student misses the integrative exercise that ties all three techniques together. The planned Exercise 4 was the capstone that connected the dots: "here are the concrete numbers showing how RoPE, sparse attention, and GQA each address a different cost and how their savings compound." The current Exercise 4 is a good exercise for RoPE understanding, but it leaves sparse attention and GQA without quantitative practice beyond Exercise 2's mask counting and Exercise 3's cache comparison.
**Suggested fix:** Replace the current Exercise 4 with the planned cost calculator exercise. The context extension experiment, while interesting, could be an optional stretch exercise or deferred. The integrative cost analysis is more valuable as the capstone because it reinforces the "three barriers, three solutions" framework with concrete numbers the student computes themselves.

#### [IMPROVEMENT] — Notebook Exercise 4 solution references Exercise 1 functions but uses different implementation

**Location:** Notebook Exercise 4 solution (cell-21)
**Issue:** The Exercise 4 requirements say "Use the `rotation_matrix_2d` and `rotate_vector` functions from Exercise 1." However, the solution defines a new `apply_rope_nd` function from scratch that does not use either of those functions. This is a minor inconsistency in the requirements vs. solution, but for an Independent exercise where the student is designing their own experiment, the contradictory guidance is confusing. The solution also does not reference `rotation_matrix_2d` or `rotate_vector` at all — it implements rotation via element-wise cos/sin operations, which is the more efficient approach for multi-dimensional RoPE but breaks the stated requirement.
**Student impact:** A student following the requirements literally would try to build multi-dimensional RoPE using the 2D `rotate_vector` function in a loop over dimension pairs. This is possible but awkward. The solution ignores this requirement entirely. The disconnect between requirements and solution creates confusion about what the "right" approach is.
**Suggested fix:** If keeping this exercise, either update the requirements to say "extend the rotation concept from Exercise 1 to multiple dimensions" (removing the specific function references) or update the solution to actually call `rotate_vector` in a loop. Given the recommendation to replace Exercise 4 entirely, this becomes moot.

#### [IMPROVEMENT] — Dilated attention gets minimal development compared to planning doc

**Location:** Sparse Attention section, single paragraph on dilated attention (lines ~1501-1509)
**Issue:** The planning document includes dilated attention as a co-equal sparse pattern alongside sliding window, with the attention pattern comparison SVG showing all three patterns side by side. The SVG diagram does show full causal, sliding window, and dilated patterns — good. But the prose explanation of dilated attention is a single paragraph that says "like dilated convolutions, skip positions at regular intervals" without grounding it in the student's experience. The student has not encountered dilated convolutions in this curriculum (the CNN coverage is in Series 3, and the records do not show dilated convolutions as a taught concept). The analogy to "dilated convolutions" assumes knowledge the student may not have.
**Student impact:** The student sees "dilated" in the diagram, reads a single paragraph referencing an unfamiliar analogy ("like dilated convolutions"), and gets the gist but not the understanding. The diagram helps — the visual pattern is clear — but the prose relies on a connection the student cannot make. This is a softer version of using an untaught concept.
**Suggested fix:** Drop the "like dilated convolutions" framing. Instead, describe dilated attention on its own terms: "Instead of attending to the w nearest consecutive tokens, attend to every 2nd (or 4th) token within a wider range. This gives the same compute budget as a window of size w but covers a 2w (or 4w) span." The visual diagram already does this well — the prose should match.

#### [IMPROVEMENT] — RoPE formula presented before concrete walkthrough

**Location:** RoPE section, rotation matrix formula (lines ~1226-1232) appearing before the code block (lines ~1244-1277)
**Issue:** The ordering rule says "concrete before abstract." The RoPE section presents the analogy (handshake vs nametag) and the diagram well, but then shows the rotation matrix formula R(m, theta_d) before the code walkthrough. The code block is actually more concrete and readable for this student (a software engineer) than the matrix notation. The formula with its 2x2 matrix, subscripted theta_d, and cos/sin entries is the abstract version; the Python code with comments is the concrete version.
**Student impact:** Minor — the student can likely parse both — but the ordering could be tighter. The student would benefit from seeing the code first ("here is what RoPE does to each dimension pair") and then the formula as confirmation ("this is the rotation matrix that the code applies").
**Suggested fix:** Move the code block to appear before the formal rotation matrix. Present it as "here is how RoPE works in code" followed by "the rotation matrix this implements is..." This matches the ordering rule and aligns with the student's strengths (software engineer who reads code fluently).

#### [POLISH] — Spaced em dashes in lesson TSX SVG text and string literals

**Location:** Multiple locations in the TSX file: SVG text elements (lines 818, 840, 862, 916), string literals in ComparisonRow items (line 1670), and checkpoint reveals (lines 1726, 1735, 1744)
**Issue:** The writing style rule requires no-space em dashes (`word—word`). Several locations use spaced em dashes with Unicode characters (` — `) rather than `&mdash;`. These appear in SVG `<text>` elements (e.g., "55 scores — O(n^2)"), in JS string literals (e.g., "'8x reduction — fits on fewer GPUs'"), and in the checkpoint reveal section (e.g., "Wall 1 — Position:"). The student-facing prose sections correctly use `&mdash;` HTML entities without surrounding spaces.
**Student impact:** Minimal visual impact — the SVG text and reveal sections render with slightly wider em dashes than the main prose. Inconsistent but not disruptive.
**Suggested fix:** Replace ` — ` with `—` (no spaces) in SVG text elements and string literals. For the SVG elements, this is purely cosmetic. For the reveal section strong tags ("Wall 1 — Position:"), use `&mdash;` for consistency with the rest of the lesson.

#### [POLISH] — Spaced em dashes throughout notebook markdown and code comments

**Location:** Throughout the notebook: markdown cells and Python code comments use ` — ` (spaced Unicode em dash) consistently
**Issue:** The writing style rule requires `word—word` not `word — word`. The notebook uses spaced em dashes in approximately 40+ locations across markdown text, code comments, and print statements. This was flagged in the MoE lesson review as well.
**Student impact:** Negligible — the notebook is a supplementary resource, and the spacing is consistent within the notebook even if it differs from the lesson TSX convention.
**Suggested fix:** Batch find-and-replace ` — ` with `—` in the notebook. Low priority but would bring it in line with the style guide.

#### [POLISH] — Code comment block in TSX uses spaced em dashes

**Location:** Component header comment block (lines 27-55)
**Issue:** The multi-line comment at the top of the component uses spaced em dashes (` — `) in several places: "BUILD lesson — extends," "RoPE (rotary position embeddings) — encoding," etc. These are developer-facing comments, not student-facing prose, but the style guide applies to the codebase.
**Student impact:** None — these comments are not rendered.
**Suggested fix:** Replace with unspaced em dashes for consistency. Very low priority.

#### [POLISH] — Notebook introduction lists four exercises but the fourth does not match

**Location:** Notebook cell-0, fourth bullet point
**Issue:** The notebook introduction says "Design a context extension experiment: generate sequences of varying length, measure attention score quality with and without RoPE, and test position interpolation for lengths beyond training." This matches the built Exercise 4 but not the planned Exercise 4 (cost calculator). If Exercise 4 is replaced per the IMPROVEMENT finding above, this introduction will need updating.
**Student impact:** Currently none (intro matches the built exercise). Would become stale if Exercise 4 is replaced.
**Suggested fix:** Update the notebook introduction bullet if/when Exercise 4 is replaced with the planned cost calculator.

### Review Notes

**What works well:**
- The three-barrier organizational framework is excellent. It gives the lesson a clear map and prevents the "random bag of techniques" feeling. The student always knows which barrier is being addressed and why.
- The "nametag vs handshake" analogy for RoPE is strong and well-extended. It captures the additive-vs-multiplicative distinction in a way the student can remember and reason with.
- The visual diagrams are well-designed. The ThreeBarrierDiagram provides the roadmap, the RoPERotationDiagram makes the relative-position property geometric, and the AttentionPatternDiagram makes sparse vs full visually obvious. All three serve genuine pedagogical purposes.
- The RoPE section is the strongest part of the lesson. Multiple modalities (analogy, diagram, code, formula, worked example in the checkpoint), misconception directly addressed, and the "clock with many hands" callback to sinusoidal PE is well-placed.
- The connection to MoE ("conditional computation again") is a good cross-lesson callback that reinforces the previous lesson's paradigm.
- The GQA section's MHA-GQA-MQA comparison table with concrete numbers is very effective. The student can trace the tradeoff across the spectrum.
- Both checkpoints (predict-and-verify for RoPE, transfer question for the three barriers) are well-designed and test genuine understanding rather than recall.

**Patterns to watch:**
- The linear attention section stands out as the one place where the lesson breaks its own pedagogical pattern. Every other concept follows problem-intuition-example-formula. Linear attention goes straight to formula. This is likely because it is at INTRODUCED depth and the builder wanted to be brief — but brief does not have to mean abstract-first.
- The notebook Exercise 4 mismatch with the plan is the most significant structural issue. The planned integrative exercise would have been the strongest capstone; the built alternative, while valid, only exercises one of the three techniques.
- Spaced em dashes are a recurring pattern across lessons and notebooks. A project-wide convention fix would be more efficient than per-lesson corrections.

---

## Review — 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All five improvement findings from iteration 1 have been fully resolved. No new critical or improvement issues were introduced by the fixes. Two minor polish findings remain, neither of which affects the student experience meaningfully. The lesson is ready to ship.

### Verification of Iteration 1 Fixes

**1. Linear attention section (IMPROVEMENT) — RESOLVED.** The linear attention explanation now follows a clear two-step structure: (1) plain-language motivation ("What if you could skip [the n x n matrix]? Instead of computing all pairwise scores first and then averaging V, compute a summary of K and V first, then let each Q token query that summary directly"), followed by (2) the symbolic reformulation with phi(Q), phi(K)^T V. The "compute a summary first" framing connects to the student's existing mental model of attention as "weighted average of V." The formula now confirms the intuition rather than being the primary explanation. The student will follow this.

**2. Notebook Exercise 4 replaced (IMPROVEMENT) — RESOLVED.** The "Context Extension Experiment" has been replaced with the planned "Attention Cost Calculator (Independent)." The new Exercise 4 defines five configurations (GPT-2 at 1K, LLaMA 2 70B MHA at 4K, GQA at 4K, GQA at 128K, GQA+sliding window at 128K) and asks the student to compute FLOPs and KV cache memory for each. The solution includes a formatted comparison section that explicitly calls out what each optimization does and does not affect. This is the integrative capstone the plan specified, tying all three barriers and solutions together quantitatively.

**3. Notebook Exercise 4 solution function references (IMPROVEMENT) — RESOLVED.** This finding was about the old Exercise 4 referencing `rotation_matrix_2d` and `rotate_vector` but the solution not using them. The replacement Exercise 4 (cost calculator) has no dependency on Exercise 1 functions, so this inconsistency no longer exists.

**4. Dilated attention "dilated convolutions" reference (IMPROVEMENT) — RESOLVED.** The prose now describes dilated attention on its own terms: "Instead of attending to the w nearest consecutive tokens, attend to every 2nd (or 4th) token within a wider range. This gives the same compute budget as a window of size w but covers a 2w (or 4w) span." No reference to dilated convolutions. The student can follow this without any untaught concept.

**5. RoPE formula-before-code ordering (IMPROVEMENT) — RESOLVED.** The code block (apply_rope function) now appears at line 1227, followed by the rotation matrix formula at line 1267. The student sees the concrete Python implementation first ("here is what RoPE does to each dimension pair"), then the formal rotation matrix as confirmation. This matches the ordering rule (concrete before abstract) and aligns with the student's strengths as a software engineer.

**Iteration 1 Polish fixes:**
- Spaced em dashes in TSX SVG text, string literals, and component header comments — ALL RESOLVED. Every em dash in the TSX file is now unspaced.
- Spaced em dashes in notebook markdown and code comments — ALL RESOLVED. Every em dash in the notebook is now unspaced.
- Notebook introduction fourth bullet mismatch — RESOLVED. The introduction now describes the cost calculator exercise, matching the rebuilt Exercise 4.

### Findings

#### [POLISH] — Notebook Exercise 4 solution is inside a markdown details block, not a code cell

**Location:** Notebook cell-21, the `<details>` block containing the Exercise 4 solution
**Issue:** The Exercise 4 solution is presented entirely as a markdown code block inside a `<details><summary>Solution</summary>` block (cell-21). Unlike Exercises 2 and 3, which have both a markdown solution reveal AND a helper code cell that the student can run, Exercise 4 has no runnable solution cell. The student must copy the solution from the markdown block into cell-20 to run it. Exercises 2 and 3 include helper cells (cell-10 for Exercise 2, cell-17 for Exercise 3) that provide reference implementations the student can execute directly, ensuring the visualization and analysis code works regardless of the student's implementation.
**Student impact:** Minor friction. An independent exercise should require more self-reliance, so not providing a helper cell is defensible. However, the inconsistency with Exercises 2 and 3 might confuse the student who expects a "run this cell to continue" pattern.
**Suggested fix:** Add a helper code cell after cell-21 containing the solution code, matching the pattern of Exercises 2 and 3. Prefix it with a markdown cell: "### Helper: Working Cost Calculator / Run the cell below to see the complete cost analysis." This maintains the independent exercise's challenge (the student should try first) while ensuring the comparison analysis runs cleanly.

#### [POLISH] — Notebook Exercise 4 solution `<details>` block includes a "Design choices explained" section that duplicates the markdown takeaways

**Location:** Notebook cell-21, the bottom of the `<details>` solution block
**Issue:** The solution block ends with a "Design choices explained" section that restates why FLOPs and KV cache are separated, what each formula counts, and why the comparison section exists. This overlaps significantly with the "Key Takeaways" in cell-22, which makes the same points. The duplication is not harmful but adds length to an already long solution block.
**Student impact:** Negligible. The student who reads both gets reinforcement, which is fine. The student who reads only one gets the key points.
**Suggested fix:** Trim the "Design choices explained" section to 1-2 sentences focusing on implementation decisions (why the function is structured this way), and let the Key Takeaways cell handle the conceptual summary. Low priority.

### Review Notes

**Verification of all iteration 1 fixes:** All 5 improvement findings and all 4 polish findings from iteration 1 have been fully addressed. The fixes are clean and do not introduce new issues.

**What works well (fresh-eyes assessment):**

- **Three-barrier framework is the lesson's backbone.** The ThreeBarrierDiagram at the start, the three-section explain arc (RoPE, sparse, GQA), the three-barrier transfer question, and the "three targeted solutions" mental model echo create a coherent structure. The student always knows where they are in the lesson and why each concept matters.

- **RoPE section is excellent.** The progression is: problem statement (PE fails at long context) -> analogy (nametag vs handshake) -> visual (RoPERotationDiagram showing same angular difference at different positions) -> concrete code (apply_rope function) -> formal math (rotation matrix) -> misconception correction -> context extension explanation. Six modalities, tightly integrated. The student builds the concept layer by layer.

- **Sparse attention section connects deeply to prior knowledge.** The transition from MoE ("not every FFN parameter needs to activate") to sparse attention ("not every token pair needs an attention score") is clean and makes sparse attention feel like a natural extension rather than a new idea. The "of course" beat grounds it in head specialization patterns from Module 4.2. The stacked-layers recovery mechanism (information flowing through residual stream across layers) directly answers the student's likely objection.

- **GQA section properly distinguishes GQA from "fewer heads."** The misconception card, the four-row comparison table (MHA, GQA, MQA, "fewer heads"), and the ComparisonRow with LLaMA 2 70B concrete numbers make the mechanism and its practical impact clear. The connection back to "split, not multiplied" from multi-head attention is well-placed.

- **Both checkpoints test genuine understanding.** Checkpoint 1 (RoPE vs learned PE at 8K) requires the student to reason about position encoding failure modes. Checkpoint 2 (design architecture changes for 64K legal documents) requires the student to apply all three solutions to a concrete scenario. Neither is answerable by recall alone.

- **The notebook is well-structured after the Exercise 4 replacement.** The progression (RoPE rotation -> sparse masks -> GQA forward pass -> integrative cost calculator) mirrors the lesson's three-barrier structure. Exercise 4 is now the capstone that ties everything together quantitatively, matching the plan.

- **Modality count is strong.** For the core concept (three barriers to long context and their solutions): verbal/analogy (nametag vs handshake, conditional computation callback), visual (ThreeBarrierDiagram, RoPERotationDiagram, AttentionPatternDiagram), symbolic (rotation matrix, O(n^2) vs O(n*w) formulas), concrete code (apply_rope function), concrete numbers (FLOPs table, KV cache calculations), intuitive ("of course" beat for sparse attention). Well above the 3-modality minimum.

- **Misconception coverage is thorough.** All 5 planned misconceptions are addressed at the locations specified in the plan: "just train longer" (hook), "flash attention solves it" (recap aside), "RoPE is just another PE" (RoPE section), "sparse loses information" (sparse section), "GQA is fewer heads" (GQA section). Each has a concrete negative example.

**No systemic issues identified.** The lesson is pedagogically sound, well-structured, and matches the plan.
