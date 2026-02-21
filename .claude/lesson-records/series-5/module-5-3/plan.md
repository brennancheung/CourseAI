# Module 5.3: Scaling Architecture -- Plan

**Status:** In progress
**Prerequisites:** Module 5.2 (Reasoning & In-Context Learning) complete, Module 4.3 (Building & Training GPT) complete, Series 4.2 (Attention & the Transformer) complete

## Module Goal

The student understands how architectural innovations change the scaling equation for large language models -- conditional computation via mixture of experts, long-context techniques that extend the memory-compute tradeoff, and the infrastructure that makes training and serving at scale possible -- and can explain why each innovation addresses a specific bottleneck in the standard dense transformer.

## Narrative Arc

Module 5.2 asked "what can models do at inference time?" and revealed capabilities (ICL, CoT, reasoning) that emerge from the basic transformer. This module asks a different question: "how do you make the transformer itself bigger, longer, and faster?" The student knows the dense transformer: every token flows through every parameter in every layer. That design is elegant but wasteful -- most parameters are irrelevant to any given token. The module progresses through three layers of scaling innovation:

1. **Mixture of Experts** (Lesson 1): The fundamental insight that not every parameter needs to activate for every token. A router network selects a subset of FFN "experts" per token, decoupling total parameters from per-token compute. The model gets "bigger without getting slower." This directly extends the student's deep understanding of FFN as the transformer's knowledge store ("attention reads, FFN writes") and challenges the implicit assumption that all parameters participate in every forward pass.

2. **Long Context & Efficient Attention** (Lesson 2): The quadratic cost of attention limits context length. RoPE provides a principled positional encoding that can be extended beyond training length. Sparse and linear attention variants trade expressiveness for subquadratic cost. The student already knows flash attention (memory optimization) and KV caching (inference optimization) -- this lesson addresses the algorithmic limits they leave untouched.

3. **Training & Serving at Scale** (Lesson 3): The engineering that turns a research model into a product. Data parallelism, tensor parallelism, and pipeline parallelism distribute training across GPUs. Speculative decoding and continuous batching accelerate inference. The student has MENTIONED-level knowledge of these from scaling-and-efficiency -- this lesson develops the concepts enough to understand how frontier models are actually built and deployed.

The connecting thread: the dense transformer has three scaling bottlenecks (parameter count vs compute, context length vs memory, single-device limits vs model size). Each lesson addresses one bottleneck with a specific architectural or engineering innovation. By the end, the student understands the design space that frontier models (GPT-4, Mixtral, DeepSeek-V3) navigate.

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| mixture-of-experts | Conditional computation: router selects subset of FFN experts per token, decoupling parameters from per-token compute | STRETCH | First lesson introduces the most architecturally novel concept. MoE changes how the student thinks about the transformer block itself -- the FFN is no longer monolithic. Must come before training-at-scale (which covers parallelism strategies partly motivated by MoE's communication overhead). |
| long-context-and-efficient-attention | RoPE, context extension, sparse/linear attention as alternatives to full quadratic attention | BUILD | Builds on attention knowledge from Series 4.2. Less architecturally radical than MoE (modifies attention, does not introduce new routing mechanisms). Requires understanding standard attention at DEVELOPED depth, which the student has. |
| training-and-serving-at-scale | Data/tensor/pipeline parallelism for training; speculative decoding, continuous batching for inference | CONSOLIDATE | Capstone that synthesizes the module. MoE and long-context create new parallelism challenges (expert routing across devices, KV cache at long sequences). This lesson develops the engineering solutions. Lower conceptual novelty -- the ideas are straightforward once the architectural challenges are understood. |

## Rough Topic Allocation

- **Lesson 1 (mixture-of-experts):** Conditional computation as a concept, router networks (softmax gating over expert indices), top-k expert selection, load balancing (auxiliary loss), expert specialization patterns, the "bigger but not slower" insight, connection to FFN as knowledge store, connection to Chinchilla (MoE changes the parameter-compute relationship)
- **Lesson 2 (long-context-and-efficient-attention):** RoPE (rotary position embeddings) as a principled replacement for sinusoidal/learned PE, how RoPE enables context extension beyond training length, the quadratic attention bottleneck at long sequences, sparse attention patterns (sliding window, dilated), linear attention variants and their tradeoffs, Grouped Query Attention (GQA) for KV cache reduction
- **Lesson 3 (training-and-serving-at-scale):** Data parallelism (simplest: replicate model, split data), tensor parallelism (split layers across devices), pipeline parallelism (split blocks across devices), communication overhead as the key constraint, speculative decoding (small model drafts, large model verifies), continuous batching (fill completed slots with new requests), ZeRO optimizer states

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| mixture-of-experts | STRETCH | Genuinely new architectural concept: conditional computation, routing, expert selection. Changes the mental model of what a forward pass looks like. Follows BUILD (reasoning-models was the last lesson in 5.2). |
| long-context-and-efficient-attention | BUILD | Extends existing attention knowledge (DEVELOPED from 4.2) with new encoding and efficiency techniques. No new paradigm -- more sophisticated versions of familiar mechanisms. |
| training-and-serving-at-scale | CONSOLIDATE | Develops MENTIONED concepts from scaling-and-efficiency (4.3.3). Engineering solutions to known problems. Lower conceptual novelty. Module capstone. |

No two STRETCH lessons are adjacent. STRETCH-BUILD-CONSOLIDATE gives the student progressive recovery.

## Module-Level Misconceptions

- **"MoE models are just ensembles of smaller models"** -- Experts are not independent models. They are FFN sub-networks within a single transformer block, sharing the same attention layers, residual stream, and embeddings. The router selects which FFN to use, not which model.

- **"More parameters always means proportionally more compute"** -- MoE decouples parameters from per-token compute. A 47B-parameter MoE model with top-2 routing uses roughly the same per-token compute as a 13B dense model, because only 2 of 8 experts (plus shared attention) activate per token.

- **"Longer context is just a matter of training with longer sequences"** -- Positional encoding schemes have length limits. Sinusoidal/learned PE cannot generalize beyond training length. RoPE provides a principled solution, but even RoPE requires extension techniques for lengths far beyond training. The quadratic cost of attention makes naive long-context prohibitively expensive.

- **"Parallelism is just splitting the data across GPUs"** -- Data parallelism is the simplest strategy but hits limits when the model does not fit on one GPU. Tensor parallelism and pipeline parallelism address different bottlenecks (memory vs compute vs communication) with different tradeoffs.

- **"Flash attention and KV caching solve the long-context problem"** -- Flash attention reduces memory from O(n^2) to O(n) but does NOT reduce compute (still O(n^2) FLOPs). KV caching eliminates redundant computation during generation but does not help with the attention computation itself at long sequence lengths. Algorithmic alternatives (sparse, linear attention) are needed to change the computational complexity.

## Connections to Prior Modules

| Module 5.3 Concept | Earlier Concept | Source | Connection |
|--------------------|----------------|--------|------------|
| MoE expert selection via softmax | Softmax for classification and gating | Series 1, 4.2 | Router uses softmax over expert scores to produce selection probabilities |
| MoE conditional FFN | FFN as knowledge store, "attention reads, FFN writes" | 4.2 (transformer-block) | MoE replaces the single monolithic FFN with multiple specialized FFNs, selected per token |
| MoE parameter-compute decoupling | Chinchilla scaling laws (scale both, not just one) | 4.3 (scaling-and-efficiency) | MoE adds a third axis: total parameters vs active parameters. Changes the Chinchilla equation. |
| RoPE | Sinusoidal positional encoding, learned PE | 4.1 (embeddings-and-position) | RoPE is a principled replacement that embeds position into the Q/K dot product itself |
| GQA | Multi-head attention, KV caching | 4.2, 4.3 | GQA shares K/V across groups of heads, reducing KV cache size proportionally |
| Sparse/linear attention | Quadratic attention cost, flash attention | 4.2, 4.3 | Flash attention fixes memory but not compute. Sparse/linear attention fixes compute at cost of expressiveness. |
| Parallelism strategies | Compute-bound vs memory-bound | 4.3 (scaling-and-efficiency) | The choice of parallelism strategy depends on whether the bottleneck is compute, memory, or communication |
| Speculative decoding | Autoregressive generation, KV caching | 4.3 (building-nanogpt, scaling-and-efficiency) | Uses the same generate() loop the student built, but with a draft model to amortize the sequential bottleneck |
| Continuous batching | Batched inference | 4.3 | Extends static batching to dynamic slot filling during inference serving |
