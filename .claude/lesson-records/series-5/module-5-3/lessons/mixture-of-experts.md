# Lesson 1: Mixture of Experts (mixture-of-experts) -- Planning Document

**Module:** 5.3 Scaling Architecture
**Position:** Lesson 1 of 3
**Type:** STRETCH
**Slug:** mixture-of-experts

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| FFN structure and role in transformer block (two-layer network with 4x expansion, GELU, "writes" to the residual stream) | DEVELOPED | the-transformer-block (4.2.5) | Student knows FFN(x) = W_2 * GELU(W_1 x + b_1) + b_2 with GPT-2 dimensions (768 -> 3072 -> 768). Knows FFN has 2x the parameters of attention per block. Understands the 4x expansion as "workspace for complex computations." |
| "Attention reads, FFN writes" mental model | DEVELOPED | the-transformer-block (4.2.5) | Core mental model: attention gathers context from other tokens (reads the residual stream), FFN processes what attention found and updates the representation (writes to the stream). Complementary, both essential. |
| FFN as source of nonlinearity (GELU breaks convex hull constraint, FFN stores learned knowledge) | DEVELOPED | the-transformer-block (4.2.5) | Without FFN, attention output is a weighted average inside the convex hull of inputs. GELU enables genuinely new representations. Research reference: Geva et al. FFN neurons as key-value memories. "2/3 of parameters store the model's learned knowledge." |
| Parameter distribution in transformer models (~1/3 attention, ~2/3 FFN per block) | DEVELOPED | the-transformer-block (4.2.5) | GPT-2: ~28M attention (~23%), ~57M FFN (~46%), ~38M embeddings (~31%). Student was surprised by this in the lesson hook -- expected attention to dominate. |
| Softmax for probability distributions / gating | DEVELOPED | Multiple (1.x, 4.2) | Student has used softmax extensively: attention weights (softmax(QK^T/sqrt(d_k))), temperature-controlled generation, classification outputs. Softmax converts arbitrary real-valued scores into a probability distribution that sums to 1. |
| Scaling laws (Chinchilla compute-optimal training) | INTRODUCED | scaling-and-efficiency (4.3.3) | N_opt ~ sqrt(C), D_opt ~ sqrt(C). Scale model size and data together. Power law: L ~ C^(-0.05). The student understands that bigger models need more data and the tradeoff is compute-optimal, but has not worked through the implications deeply. |
| Compute-bound vs memory-bound operations (arithmetic intensity) | INTRODUCED | scaling-and-efficiency (4.3.3) | GPU compute throughput vastly exceeds memory bandwidth. Matrix multiplication is compute-bound, most other ops are memory-bound. Kitchen analogy: fast chefs but slow delivery truck. |
| Mixture of experts (MoE) | MENTIONED | scaling-and-efficiency (4.3.3) | Name-drop only: "Router activates subset of parameters per token. More total parameters, same compute per token." Student has recognition of the term but no understanding of the mechanism. |
| Transformer block as repeating unit (shape-preserving, stacks identically) | DEVELOPED | the-transformer-block (4.2.5) | Complete block formula: x' = x + MHA(LayerNorm(x)), output = x' + FFN(LayerNorm(x')). Shape-preserving: (n, d_model) in, (n, d_model) out. GPT-2: 12 blocks. GPT-3: 96 blocks. Student can trace a forward pass through a block. |
| Residual stream as cross-layer backbone | DEVELOPED | the-transformer-block (4.2.5) | Two residual connections per block. "Shared document" analogy: starts as raw embedding, each sub-layer reads and annotates. Central highway of the entire transformer. |
| Test-time compute scaling (inference compute as new scaling axis) | DEVELOPED | reasoning-models (5.2.4) | Paradigm shift from "bigger model" to "more thinking time." Two independent axes of scaling (model size and inference compute). Student understands this deeply from the reasoning models lesson. |

**Mental models and analogies already established:**
- "Attention reads, FFN writes" -- the defining mental model for the transformer block's two sub-layers
- "The model needs more room to think than to communicate" -- the 4x FFN expansion as workspace
- "Scale both, not just one" -- Chinchilla: model size and data should scale together
- "The bottleneck is the delivery truck, not the chefs" -- compute-bound vs memory-bound
- Convex hull argument for FFN nonlinearity (attention can only blend, FFN can transform)
- "2/3 of parameters store the model's learned knowledge" -- FFN dominates the parameter budget
- "Bigger brain vs more thinking time" -- model size vs inference compute (from reasoning-models)

**What was explicitly NOT covered in prior lessons (relevant here):**
- How MoE actually works beyond the name-drop (router mechanism, expert selection, load balancing)
- Conditional computation as a concept (all forward passes so far activate all parameters)
- The distinction between total parameters and active parameters
- Expert specialization patterns
- How MoE changes the Chinchilla scaling equation
- Why not every parameter needs to fire for every token

**Readiness assessment:** The student is well-prepared for the core MoE concept. The deep understanding of FFN as the transformer's knowledge store (DEVELOPED) is the critical prerequisite -- MoE is a direct modification of the FFN. The student's surprise at the parameter distribution ("2/3 FFN") sets up the MoE insight naturally: if most parameters are in the FFN and most FFN knowledge is irrelevant to any given token, why activate all of it? Softmax for gating is at DEVELOPED depth from extensive use in attention. Scaling laws at INTRODUCED depth provides sufficient grounding for the "MoE changes the scaling equation" discussion without requiring deep quantitative analysis. The STRETCH designation is appropriate: conditional computation is a genuinely new paradigm for this student.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how mixture-of-experts replaces the monolithic FFN with multiple specialized sub-networks and a learned router, decoupling total model parameters from per-token computation cost.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| FFN structure and role (two-layer, 4x expansion, GELU, writes to residual stream) | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | MoE replaces the FFN with multiple FFNs. The student must understand the single FFN thoroughly to grasp what changes and what stays the same. |
| "Attention reads, FFN writes" mental model | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | MoE modifies only the "writes" half. Attention is unchanged. This distinction is critical for understanding that MoE is a targeted modification, not a redesign. |
| Parameter distribution (~2/3 FFN) | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | The fact that FFN dominates the parameter budget motivates WHY conditional computation targets the FFN specifically. |
| Softmax for gating/probability distributions | DEVELOPED | DEVELOPED | Multiple | OK | The router uses softmax to produce expert selection probabilities. Student has extensive softmax experience from attention and generation. |
| Scaling laws (Chinchilla) | INTRODUCED | INTRODUCED | scaling-and-efficiency (4.3.3) | OK | INTRODUCED is sufficient. This lesson uses scaling laws as context ("MoE changes the equation") without requiring the student to derive or apply the formulas. |
| Transformer block as repeating unit | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | The student needs to understand where MoE fits in the block: it replaces the FFN sub-layer while attention remains unchanged. |
| Residual stream | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | Expert outputs are added to the residual stream the same way the dense FFN output was. The stream is unchanged; what writes to it changes. |

All prerequisites are at sufficient depth. No gaps.

### Gap Resolution

No gaps identified. All prerequisites are at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "MoE models are just ensembles of smaller models" | Ensembles are the most familiar "multiple models" concept. The word "experts" reinforces this -- it sounds like a committee of independent specialists. The student may picture 8 complete transformer models voting on an answer. | In an ensemble, each model processes the entire input independently and outputs are aggregated. In MoE, experts are FFN sub-networks within a single transformer block. They share the same attention layers, residual stream, and embeddings. Only the FFN varies per token. Show the architecture: one attention layer feeding into a router that selects among 8 FFNs. The experts cannot function independently -- they depend on the shared attention output. | Explain section, immediately after introducing the MoE architecture. Address head-on because this is the most common first impression. |
| "Every expert sees every token (experts specialize by topic, not by token)" | The word "specialist" evokes a topic expert -- a person who knows about one subject and handles all queries in that domain. The student might think Expert 1 handles "science tokens" and Expert 2 handles "literature tokens," with all tokens being routed to their topic expert. | The router decides per-token, not per-sequence or per-topic. In the same sentence, "The" might go to Expert 3 while "mitochondria" goes to Expert 7. Different tokens in the same sentence activate different experts. Expert specialization is emergent and often surprising -- some experts specialize in syntax (function words), others in domain vocabulary, others in positional patterns. It is not a clean topic partition. | After explaining the routing mechanism. The per-token granularity is a key architectural detail that differentiates MoE from ensembles and from the student's likely mental model. |
| "More total parameters always means proportionally more compute and memory during inference" | The student's entire experience (GPT-2 124M, GPT-3 175B, the scaling laws lesson) has established a direct relationship between parameter count and compute. "Bigger model = more computation per forward pass" has been true for every model the student has seen. | Mixtral 8x7B has ~47B total parameters but uses roughly the same per-token compute as a 13B dense model. Only 2 of 8 experts activate per token. The router + 2 experts + shared attention ≈ 13B active parameters out of 47B total. The remaining 34B parameters exist but do not contribute to this token's forward pass. Per-token FLOPs scale with active parameters, not total parameters. | Core insight section, right after the architecture. This is the central "aha" of the lesson -- decoupling parameters from compute. Reinforce with a concrete parameter count comparison. |
| "The router is a complex, learned component that 'understands' which expert is best" | The student has seen attention (complex Q/K/V mechanism) and RL (reward-driven optimization). They might assume the router is similarly complex -- perhaps a small neural network that analyzes the token and makes a sophisticated decision. | The router is typically a single linear layer: router_logits = W_router @ x, where W_router is (num_experts, d_model). That is one matrix multiplication followed by softmax -- simpler than a single attention head. The routing "decision" is a dot product between the token's hidden state and learned expert embeddings, selected via top-k. The same mechanism as attention scores (dot product + softmax), but over expert indices instead of token positions. | After the routing mechanism explanation. Connecting to the familiar dot-product-softmax pattern demystifies the router. |
| "Load imbalance is just an efficiency problem, not a training problem" | The student might see load imbalance (some experts getting most tokens, others starving) as a GPU utilization issue -- some GPUs idle while others are overloaded. They would be right that it is an efficiency problem, but might miss the deeper issue. | If one expert receives 80% of tokens, that expert's parameters get 80% of the gradient updates. Other experts get too few examples to learn effectively. The imbalance creates a positive feedback loop: the popular expert gets better (more gradient updates), which makes the router send even more tokens to it, which gives it even more updates. Eventually, only 1-2 experts are used and the rest are dead weights. This is a training collapse problem, not just an efficiency problem. The auxiliary loss exists to prevent this collapse. | Elaborate section, when introducing load balancing. Frame as a training failure mode, not just an engineering concern. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Dense FFN vs MoE FFN forward pass on the same token | Positive | Show the structural difference: same input, same attention output, but the FFN step changes from "one FFN processes the token" to "router selects 2 of 8 FFNs, their outputs are weighted and summed." Concrete parameter counts make the compute savings tangible. | Directly extends the transformer block the student already knows. Uses the same GPT-2-scale dimensions for continuity. The comparison makes the "what changes, what stays the same" crystal clear. |
| Mixtral 8x7B as a real-world MoE architecture | Positive | Show that MoE is not theoretical -- it is the architecture behind frontier models. Concrete numbers: 8 experts per layer, top-2 routing, ~47B total parameters, ~13B active. Performance competitive with models 3-4x its active parameter count. | Named architecture gives the student vocabulary for the real world. The 47B-total-13B-active numbers make the parameter-compute decoupling concrete and memorable. Mixtral is well-documented and the student can look it up. |
| Router collapse: all tokens go to one expert | Negative | Disproves "the router naturally distributes tokens evenly." Shows what happens without load balancing: one expert dominates, others atrophy, the model degenerates to a dense model with wasted parameters. The training loss stalls because most experts never learn. | Makes the load balancing problem visceral. Without this negative example, the auxiliary loss feels like an arbitrary addition. With it, the student understands it as a necessary correction for a predictable failure mode. Also connects to reward hacking: the system optimizes for the proxy (send to the expert with highest score) at the expense of the true objective (use all experts effectively). |
| A sentence where different tokens route to different experts | Stretch | Show per-token routing in action on a concrete sentence. "The mitochondria is the powerhouse of the cell." -- the function words ("The," "is," "the," "of," "the") might route to one expert, while "mitochondria," "powerhouse," and "cell" route to different domain-specific experts. Illustrates that expert specialization is per-token, emergent, and does not follow clean topic boundaries. | Extends the positive examples into a concrete, traceable instance. The student can reason about why each token might route differently. The mixed routing within a single sentence disproves the "experts = topic specialists" misconception. Uses a sentence the student will find memorable. |

---

## Phase 3: Design

### Narrative Arc

The student has spent 18 lessons with the dense transformer. Every forward pass activates every parameter. When the model needs to know about mitochondria, the same FFN layers that also "know" about French cooking, JavaScript syntax, and medieval history all fire. The student already learned that the FFN stores the model's knowledge -- 2/3 of the parameters, key-value memories, the "writes" half of "attention reads, FFN writes." But here is the wasteful part: for any given token, the vast majority of that stored knowledge is irrelevant. "Mitochondria" does not need the parameters that encode cooking recipes. If 2/3 of the model's capacity is knowledge storage, and most knowledge is irrelevant to any given token, then the dense transformer is doing something deeply inefficient: activating all knowledge for every token, regardless of what that token actually needs. What if, instead of one massive FFN, you had multiple smaller FFNs -- each specializing in different aspects of the model's knowledge -- and a lightweight router that decides which ones to activate for each token? This is mixture of experts. The model can have vastly more total parameters (more total knowledge) while keeping per-token computation constant (only the relevant knowledge activates). The result: models that are "bigger without being slower."

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Verbal/Analogy** | "Library with specialist librarians" -- a dense FFN is one librarian who has read every book and answers every question. An MoE FFN is a library with 8 specialist librarians (biology, history, syntax, etc.) plus a front desk clerk (router) who directs your question to the 2 most relevant librarians. You get answers from 2 people instead of 1, but the library can collectively know far more because each librarian only needs deep expertise in their specialty. | Maps to a familiar experience. The "front desk clerk" maps directly to the router (simple lookup, not deep analysis). The "2 out of 8 librarians" maps to top-k routing. The insight that the library's total knowledge exceeds any individual librarian's knowledge maps to total parameters vs active parameters. |
| **Visual (inline SVG)** | MoE block architecture diagram: side-by-side comparison of a standard transformer block (attention -> FFN -> residual) and an MoE block (attention -> router -> top-k experts -> weighted sum -> residual). Color-coded: shared components (attention, residual stream) in violet, router in amber, selected experts in emerald, unselected experts in gray. | The structural change is spatial -- what replaces the FFN in the block. A diagram makes the modification visible and makes clear what stays the same (attention, residual) vs what changes (FFN -> router + experts). |
| **Symbolic/Code** | Router pseudocode: `router_logits = W_router @ x` (one linear layer), `probs = softmax(router_logits)` (familiar operation), `top_k_experts = top_k(probs, k=2)`, weighted output = sum of expert outputs scaled by router probabilities. Annotated with attention parallels: "This is dot-product-softmax-select, the same pattern as attention, but selecting experts instead of tokens." | Connects to the student's strongest skill (code) and their deepest conceptual anchor (dot-product + softmax from attention). The pseudocode is concise enough to fit in a code block and makes the router demystifiable. |
| **Concrete example** | Parameter count walkthrough for Mixtral 8x7B: total parameters (~47B), active parameters per token (~13B), breakdown by component (shared attention layers + 2 selected experts + router). Compared to a hypothetical 47B dense model (all parameters active) and a 13B dense model (same active compute). | Makes the abstract "decoupling" concrete with real numbers. The student can verify: 47B total, ~13B active, competitive with 40B+ dense models on benchmarks. The three-way comparison (47B dense, 47B MoE, 13B dense) crystallizes the tradeoff. |
| **Intuitive** | The "of course" beat: "You already knew that 2/3 of parameters are in the FFN. You already knew FFN neurons store specific knowledge (key-value memories). If most knowledge is irrelevant to any given token, of course you should only activate the relevant knowledge. Of course the model should be able to have more total knowledge while keeping per-token compute constant." | Three established facts combine into the MoE insight. The "of course" moment collapses the conceptual distance. Each premise is something the student has at DEVELOPED depth. |
| **Concrete example** | Per-token routing visualization: "The mitochondria is the powerhouse of the cell" with hypothetical expert assignments. Function words -> Expert 2 (syntax specialist), "mitochondria"/"powerhouse"/"cell" -> Expert 5 (science domain). Different tokens, same sentence, different experts. | Makes per-token routing tangible. The student can reason about why different tokens route differently. Disproves the "experts = topic categories" misconception by showing mixed routing within one sentence. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. Conditional computation (the paradigm shift: not every parameter activates for every token, decoupling total parameters from per-token compute)
  2. Router mechanism (learned gating network that selects which experts to activate, using familiar dot-product + softmax + top-k)
  3. Load balancing / auxiliary loss (preventing router collapse where one expert dominates -- this is a consequence of the design, not a fully independent concept, so it is borderline whether it counts as a third new concept)
- **Previous lesson load:** BUILD (reasoning-models was the last lesson, a BUILD following the STRETCH chain-of-thought)
- **This lesson's load:** STRETCH -- appropriate. Conditional computation is a genuinely new paradigm for this student. Every model they have studied so far activates all parameters for every input. MoE changes that assumption. The router mechanism is demystified by connecting to attention (dot-product + softmax), but the architectural implication (decoupling parameters from compute) requires a mental model update.
- **Load trajectory (cross-module):** BUILD (reasoning-models) -> STRETCH (mixture-of-experts). One recovery lesson (reasoning-models was BUILD) between the last STRETCH (chain-of-thought) and this one. Appropriate spacing.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| FFN as knowledge store, "attention reads, FFN writes" (4.2.5) | MoE replaces the "writes" half with multiple specialized writers. Attention (the "reads" half) is completely unchanged. This is the single most important connection in the lesson. |
| Parameter distribution ~2/3 FFN (4.2.5) | If most parameters are in the FFN and most FFN knowledge is irrelevant to any given token, conditional computation is the natural optimization. The student's own surprise at the parameter distribution ("I expected attention to dominate") is the motivation for MoE. |
| Softmax for gating (Series 1, 4.2) | The router uses softmax over expert scores -- the same operation the student has used dozens of times for attention weights and generation probabilities. Demystifies the router. |
| Dot product as similarity/relevance (4.2.1, 4.2.2) | Router logits are dot products between the token's hidden state and learned expert embeddings. Same mechanism as Q*K attention scoring but over expert indices instead of token positions. |
| Chinchilla scaling laws (4.3.3) | MoE changes the scaling equation by decoupling total parameters from per-token compute. The Chinchilla "scale both" insight now has a third dimension: you can scale total parameters without proportionally scaling compute. |
| Geva et al. FFN neurons as key-value memories (4.2.5) | If individual FFN neurons store specific knowledge (facts, patterns), then grouping related neurons into "experts" and routing tokens to relevant experts is a natural extension. The neuron-level insight motivates the expert-level architecture. |
| Test-time compute scaling (5.2.4) | MoE is a different axis of scaling efficiency: model-architecture scaling (more parameters, same compute) vs inference-time scaling (same model, more compute). Both challenge the "bigger always means slower" assumption, through different mechanisms. |

**Analogies from prior lessons that can be extended:**
- "Attention reads, FFN writes" -> "Attention reads, the right experts write" (MoE only changes who writes)
- "Scale both, not just one" (Chinchilla) -> "Scale parameters without scaling compute" (MoE adds a new dimension)
- Dot-product + softmax as a selection mechanism (attention) -> Dot-product + softmax as a routing mechanism (MoE router)
- "The model needs more room to think than to communicate" (4x FFN expansion) -> "The model needs more knowledge than it activates" (MoE: more total FFN capacity, subset activated)

**Analogies from prior lessons that could be misleading:**
- "Attention reads, FFN writes" could suggest MoE changes the attention mechanism. It does not. Only the FFN is modified. Important to state this explicitly.
- The "ensemble" mental model (if the student has encountered ensembles informally) could lead to thinking experts are independent models. Need to address this head-on.

### Scope Boundaries

**This lesson IS about:**
- Conditional computation: the concept of activating only a subset of parameters per token
- The MoE architecture: replacing the monolithic FFN with router + multiple expert FFNs
- The router mechanism: how tokens are assigned to experts (linear layer + softmax + top-k)
- Top-k expert selection and weighted output combination
- Load balancing: why it is necessary (collapse failure mode) and how it works (auxiliary loss)
- The parameter-compute decoupling: total parameters vs active parameters
- Expert specialization patterns (emergent, not designed)
- Mixtral 8x7B as a concrete real-world example

**This lesson is NOT about:**
- Implementing MoE in code (conceptual lesson; notebook demonstrates routing on a small proxy)
- Training an MoE model from scratch
- Specific MoE training recipes or hyperparameters
- Token dropping or capacity factors in detail (MENTIONED at most)
- Switch Transformer or other historical MoE variants in depth (named for context, not developed)
- Communication overhead of MoE across devices (deferred to Lesson 3: training-and-serving-at-scale)
- Comparing specific MoE model benchmark results
- DeepSeek-V3 MoE architecture in detail (named for context)

**Target depths:**
- Conditional computation (not every parameter activates per token): DEVELOPED (can explain the concept, articulate why it helps, connect to FFN knowledge storage)
- Router mechanism (linear layer + softmax + top-k): DEVELOPED (can explain the mechanism, connect to attention's dot-product-softmax pattern)
- Load balancing / auxiliary loss: INTRODUCED (knows the problem and the solution concept, but not the mathematical details of the auxiliary loss)
- Expert specialization patterns: INTRODUCED (knows specialization is emergent and per-token, not designed and per-topic)
- Mixtral / DeepSeek-V3 as MoE examples: INTRODUCED (name recognition, key numbers, high-level differentiation)

### Lesson Outline

**1. Context + Constraints**
What this lesson is about: how mixture of experts replaces the monolithic FFN with specialized sub-networks and a learned router, decoupling total model parameters from per-token computation. What we are NOT doing: implementing MoE, training from scratch, or covering the communication challenges of distributing experts across GPUs (that is Lesson 3). This is a new architectural paradigm -- the first time in this course that not every parameter participates in every forward pass.

**2. Recap**
Brief reconnection to three facts from the transformer block lesson:
1. "Attention reads, FFN writes" -- the two complementary sub-layers
2. The FFN has 2/3 of the parameters (student's own surprise from the hook)
3. FFN neurons as key-value memories (Geva et al.) -- specific neurons store specific knowledge

Connect these: "If 2/3 of the model's capacity is stored knowledge, and most stored knowledge is irrelevant to any given token, what happens during a forward pass?"

Brief reconnection to Chinchilla: "Scale both, not just one. But what if you could scale parameters without proportionally scaling compute?"

**3. Hook (problem reveal + cost calculation)**
Present the waste problem concretely. GPT-3: 175B parameters, ~117B in FFN layers. Every token -- whether "the" or "mitochondria" -- activates all 117B FFN parameters. A simple calculation: if only ~10-20% of FFN knowledge is relevant to any given token, the model is doing 5-10x more FFN computation than necessary for each token. GradientCard: "The dense transformer activates all knowledge for every token. Most of that knowledge is irrelevant. What if the model could activate only the knowledge it needs?"

**4. Explain Part 1 -- The MoE Architecture**
The solution: replace the single monolithic FFN with N smaller FFN "experts" and a router that selects which ones to activate.

Walk through the architecture:
- Same attention layer (unchanged -- "attention reads" is untouched)
- Same residual stream (unchanged)
- The FFN sub-layer changes: instead of one FFN(x), there are N expert FFNs (Expert_1(x), Expert_2(x), ..., Expert_N(x)), each with the same structure (two-layer, expansion, GELU) but their own learned weights
- A router network that takes the token's hidden state and produces a probability distribution over experts
- Top-k selection: only k experts (typically k=1 or k=2) activate for each token
- Weighted output: the outputs of the selected experts are weighted by the router probabilities and summed

MoE block architecture diagram (inline SVG): side-by-side standard block vs MoE block. Color-coded shared components (violet) vs router (amber) vs selected experts (emerald) vs inactive experts (gray).

Address misconception 1 immediately: "Experts are NOT independent models. They are FFN sub-networks within a single transformer block, sharing the same attention layers, residual stream, and embeddings. Think of them as different FFN writers working from the same attention reader's notes."

Library analogy introduced here: dense FFN = one librarian who read every book, MoE = library with specialist librarians + front desk clerk.

**5. Explain Part 2 -- The Router Mechanism**
Demystify the router. Show it is remarkably simple:

Pseudocode:
```
router_logits = W_router @ hidden_state  # (num_experts,) -- one score per expert
router_probs = softmax(router_logits)     # probability distribution over experts
top_k_indices, top_k_probs = top_k(router_probs, k=2)  # select top-2 experts
output = sum(top_k_probs[i] * Expert_i(hidden_state) for i in top_k_indices)
```

Connect to attention: "This is dot-product + softmax + weighted sum. The same pattern you have seen in attention, but selecting experts instead of tokens. W_router rows are learned 'expert embeddings'; the dot product measures relevance of the token to each expert."

Address misconception 4: "The router is a single linear layer. One matrix multiplication and a softmax. Simpler than a single attention head."

"Of course" beat: "You already knew that 2/3 of parameters are in the FFN. You already knew FFN neurons store specific knowledge. If most knowledge is irrelevant to any given token, of course you should only activate the relevant knowledge. Of course the model should be able to have more total knowledge while keeping per-token compute constant."

**6. Explain Part 3 -- The Parameter-Compute Decoupling (Core Insight)**
This is the central "aha" of the lesson.

Address misconception 3: "You have been trained to think that more parameters = proportionally more compute. For dense models, that is exactly true. MoE breaks this relationship."

Concrete parameter walkthrough (Mixtral 8x7B):
- 8 experts per MoE layer, top-2 routing
- Each expert ≈ 7B-class FFN parameters
- Shared attention layers ≈ same as 7B model
- Total parameters: ~47B
- Active parameters per token: shared attention (~few B) + 2 selected experts (~14B-equivalent FFN) ≈ ~13B
- Per-token compute: comparable to a 13B dense model
- Total knowledge stored: comparable to a 47B dense model

Three-way comparison table: 13B dense (13B active, 13B total, baseline compute), Mixtral 8x7B (13B active, 47B total, same compute), 47B dense (47B active, 47B total, ~3.5x compute). Mixtral gets 47B worth of knowledge at 13B worth of compute.

Connection to Chinchilla: "Chinchilla said 'scale both.' MoE adds a third option: scale total parameters without scaling per-token compute. A new dimension in the scaling equation."

**7. Check 1 (predict-and-verify)**
Present a scenario: a model has 8 experts per layer, top-1 routing (only 1 expert per token instead of 2). A user sends the prompt "Translate this French sentence to English: Le chat est sur le tapis."

Questions:
- How many experts activate for each token?
- Do all tokens in the sentence activate the same expert?
- What happens to the other 7 experts for each token?
- How does the total compute compare to an 8x dense model?

Reveal: 1 expert per token. Different tokens likely activate different experts ("Le" might route differently than "chat" or "tapis"). The other 7 experts are skipped entirely -- their parameters exist but do not contribute compute or gradients for this token. Total compute ≈ 1/8th of the 8x dense model (for the FFN portion) because only 1 of 8 experts activates per token. The compute savings are proportional to 1/N, where N is the number of experts.

**8. Explain Part 4 -- Expert Specialization**
Per-token routing visualization: "The mitochondria is the powerhouse of the cell" with hypothetical expert assignments. Show that within a single sentence, different tokens route to different experts. Function words ("The," "is," "the," "of," "the") cluster toward one expert; content words route to others; "mitochondria" and "cell" might share an expert while "powerhouse" goes elsewhere.

Address misconception 2: "Expert specialization is emergent and per-token, not designed and per-topic. There is no 'biology expert' and 'literature expert.' The router learns what groupings are useful from the training data. Some experts end up specializing in syntax (function words), others in domain vocabulary, others in positional patterns. The boundaries are often surprising and do not map to human categories."

Research findings (briefly): studies on Mixtral show experts develop preferences but not clean topic boundaries. Expert utilization varies by layer -- early layers show more syntactic specialization, later layers show more semantic patterns.

**9. Elaborate -- Load Balancing**
Now the problem: what happens if the router consistently prefers one expert?

The collapse scenario: Expert 3 starts slightly better (random initialization). The router sends more tokens to Expert 3. Expert 3 gets more gradient updates and improves further. The router sends even more tokens to it. Positive feedback loop. Eventually, Expert 3 handles 80%+ of tokens. The other 7 experts are undertrained and effectively dead weight. The model degenerates to approximately a dense model -- all the extra parameters provide no benefit.

Address misconception 5: "Load imbalance is not just an efficiency problem. It is a training collapse problem. The positive feedback loop kills diversity."

The solution: an auxiliary loss that penalizes uneven expert utilization. During training, measure how evenly tokens are distributed across experts. Add a small penalty to the main loss when the distribution is too skewed. This gently pushes the router toward balanced utilization without overriding its learned preferences.

Connection to reward hacking: "The router's proxy objective (send to the highest-scoring expert) conflicts with the system objective (use all experts effectively). The auxiliary loss is a constraint on the proxy, similar to how the KL penalty constrains the reward model in RLHF."

Load balancing depth is INTRODUCED: the student knows the problem and the concept of the solution, but not the mathematical details of the auxiliary loss.

**10. Check 2 (transfer question)**
A company is training a large language model. They have a fixed compute budget (GPU-hours) and want to maximize model quality. They are deciding between:
- Option A: Train a 13B dense model for 1T tokens (compute-optimal per Chinchilla)
- Option B: Train a 47B MoE model (8 experts, top-2) for 1T tokens (same per-token compute as 13B dense)

Questions: Which option uses approximately the same total compute? Which has more stored knowledge? Which is likely to perform better on diverse benchmarks? What is the tradeoff?

Reveal: Both use approximately the same per-token compute (same FLOPs per token, same number of tokens), so total training compute is roughly similar. The MoE model stores more knowledge (47B parameters vs 13B) but uses the same compute to process each token. The MoE model is likely to perform better on diverse benchmarks because it can store more specialized knowledge. The tradeoff: the MoE model uses ~3.5x more memory (all 47B parameters must be loaded even though only 13B activate per token), and distributing experts across multiple GPUs introduces communication overhead. More knowledge, same speed, more memory.

**11. Practice -- Notebook Exercises (Colab)**
`notebooks/5-3-1-mixture-of-experts.ipynb` (4 exercises)

- **Exercise 1 (Guided): Implement a simple router.** Build a router for 4 experts on a toy d_model=64 hidden state. Apply softmax, select top-2, observe how different random input vectors route to different experts. Visualize router probabilities as a bar chart for 5 different inputs. Predict-before-run: "Will all inputs route to the same experts?" First 2 inputs fully worked with visualization code. Insight: the router naturally produces different selections for different inputs because the dot products depend on the input.

- **Exercise 2 (Supported): MoE forward pass on a toy model.** Build a complete MoE layer: 4 small expert FFNs (d_model=64, d_ff=256) + router. Run a forward pass for a batch of 8 tokens. Compare output shapes to a single dense FFN. Count active parameters per token vs total parameters. First expert FFN and the router are provided; student builds remaining experts and the weighted combination. Insight: same output shape as a dense FFN, but only 2/4 experts contributed to each token.

- **Exercise 3 (Supported): Visualize expert routing on real text.** Use a small pretrained MoE model (or simulate routing with a trained router on embeddings from a small model). Tokenize 3-5 sentences and visualize which expert each token is routed to. Color-code tokens by expert assignment. Look for patterns: do function words cluster? Do domain words cluster? Sentence: "The mitochondria is the powerhouse of the cell." vs "The stock market crashed yesterday." First sentence visualization fully set up. Insight: expert specialization is per-token and emergent, not a clean topic partition.

- **Exercise 4 (Independent): Router collapse experiment.** Train a toy MoE model (4 experts, tiny dataset) with and without an auxiliary load-balancing loss. Track expert utilization over training steps. Plot the distribution of tokens across experts at step 0, 100, 500, 1000. Without auxiliary loss: observe collapse toward 1-2 experts. With auxiliary loss: observe more uniform distribution. No skeleton provided. Insight: load balancing is necessary to prevent expert collapse during training.

Exercises are cumulative in concept but independent in code -- each can be run standalone. Progression: router mechanism -> full forward pass -> real routing patterns -> training dynamics.

**12. Summarize**
Key takeaways:
1. Mixture of experts replaces the monolithic FFN with multiple specialized FFN sub-networks and a learned router -- "attention reads, the right experts write"
2. The router is a single linear layer with softmax -- the same dot-product-softmax pattern as attention, applied to expert selection
3. MoE decouples total parameters from per-token compute: more stored knowledge, same inference cost per token
4. Expert specialization is emergent and per-token, not designed and per-topic
5. Load balancing (auxiliary loss) prevents router collapse, where one expert dominates and others atrophy

Echo the mental model: "The dense transformer activates all knowledge for every token. MoE activates only the relevant knowledge. More total knowledge, same compute per token. The library got bigger, but you still only talk to two librarians."

**13. Next Step**
"MoE solved the parameter-compute problem: the model can store more knowledge without proportionally increasing per-token cost. But there is another scaling bottleneck we have not addressed: attention is quadratic in sequence length. As context windows grow to 100K+ tokens, the attention computation itself becomes the bottleneck -- and flash attention reduces memory but not compute. Next: how positional encodings enable long context, and how attention can be made more efficient."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (6 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (2 positive + 1 negative + 1 stretch)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2-3: conditional computation, router mechanism, load balancing)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: MAJOR REVISION

Critical finding exists regarding the notebook. Must fix before this lesson is usable.

### Findings

#### [CRITICAL] — Notebook Exercise 2 MoELayer has unfinished TODOs that break all downstream cells

**Location:** Notebook cell-6 (Exercise 2: MoE Forward Pass)
**Issue:** The `MoELayer` class has two `TODO` blocks with placeholder values (`self.experts = None` and `output = None`). The solution is provided only in a separate markdown cell (cell-7) as a `<details>` block, but the actual code cell never gets filled in. Exercises 3 and 4 both depend on a working `MoELayer` class. Exercise 3 does not import or reuse `MoELayer` so it works independently, but Exercise 4's solution explicitly uses `MoELayer` via `ToyMoEModel`. If the student does not correctly complete Exercise 2 (or copy-pastes the solution incorrectly), Exercise 4 will fail with `TypeError: 'NoneType' object is not iterable` from `self.experts` being `None`. This is a scaffolding design issue: a Supported exercise whose incomplete implementation is load-bearing for a later Independent exercise. The Independent exercise should be fully self-contained, or the MoELayer should be provided as a complete utility after Exercise 2.
**Student impact:** The student working on Exercise 4 (the Independent exercise, no skeleton provided) must either have perfectly completed Exercise 2 or copy-paste the solution. If they made a subtle bug in Exercise 2, it will silently propagate to Exercise 4 with confusing errors. For an Independent exercise, this dependency is inappropriate. The student should be focused on the *new* challenge (auxiliary loss and collapse dynamics), not debugging their Exercise 2 implementation.
**Suggested fix:** After Exercise 2's solution reveal, add a "helper cell" that provides the complete, working `MoELayer` and `ExpertFFN` classes so Exercise 4 can reference them cleanly. Alternatively, have Exercise 4's solution include its own complete `MoELayer` definition rather than assuming the class from Exercise 2 is available and correct.

#### [IMPROVEMENT] — Objective block repeats the lesson header description nearly verbatim

**Location:** ObjectiveBlock (section 2, "Context + Constraints")
**Issue:** The `ObjectiveBlock` text reads: "This lesson teaches you how mixture of experts replaces the monolithic FFN with multiple specialized sub-networks and a learned router, decoupling total model parameters from per-token computation cost." The `LessonHeader` description reads: "...Mixture of experts replaces the monolithic FFN with specialized sub-networks and a learned router, decoupling total parameters from per-token computation." These are nearly identical sentences. The student reads the same idea twice in immediate succession with no new information in the second instance.
**Student impact:** The student skims the objective because they just read the same thing. The objective block loses its pedagogical value as a clear, distinct statement of what the student will be able to do after the lesson.
**Suggested fix:** Rewrite the ObjectiveBlock to be a distinct "by the end of this lesson you will be able to..." statement that differs from the header description. For example: "By the end of this lesson, you will be able to explain why conditional computation targets the FFN specifically, trace how a token flows through a router to selected experts, and articulate the parameter-compute tradeoff using Mixtral's concrete numbers."

#### [IMPROVEMENT] — Auxiliary loss mathematical detail is missing for an INTRODUCED-depth concept

**Location:** Load Balancing section (section 10)
**Issue:** The planning document specifies load balancing at INTRODUCED depth: "knows the problem and the concept of the solution, but not the mathematical details of the auxiliary loss." The lesson prose says "an auxiliary loss that penalizes uneven expert utilization" and "add a small penalty to the main loss when the distribution is too skewed." However, it gives no formula, no pseudocode, and no concrete illustration of what "penalize uneven utilization" means mechanically. Meanwhile, the notebook (Exercise 4 solution) provides the actual formula: `L_balance = num_experts * sum(f_i * P_i)`. The lesson teaches the *why* of load balancing well (the collapse diagram is excellent), but the *what* of the solution is left entirely abstract. For INTRODUCED depth ("can explain in own words"), the student should have at least a conceptual sketch of the mechanism, not just "add a penalty."
**Student impact:** The student understands the problem (collapse) but has only a vague sense of the solution ("some penalty"). When they reach Exercise 4 in the notebook, the auxiliary loss formula appears with no lesson grounding. The student must learn the formula from the notebook rather than the lesson, violating the principle that notebooks practice concepts from the lesson rather than introducing new ones.
**Suggested fix:** Add a brief conceptual explanation (not necessarily a formula, but at least a concrete description): "The auxiliary loss measures two things per expert: (1) what fraction of tokens were routed to it, and (2) how much probability the router assigned to it on average. If both are high for one expert, the penalty is large. If both are spread evenly, the penalty is small. This product-based penalty gently pushes the router toward balance without forcing uniform routing." This grounds the concept enough for INTRODUCED depth and prepares the student for the notebook's formula.

#### [IMPROVEMENT] — The "10-20% of FFN knowledge is relevant" claim in the hook is unsourced and potentially misleading

**Location:** Section 4, "The Waste in Every Forward Pass"
**Issue:** The lesson states: "A rough calculation: if only ~10-20% of FFN knowledge is relevant to any given token, the model is doing 5-10x more FFN computation than necessary for each token." This is presented as a factual claim but is not sourced. It is actually a rough analogy/estimate, but the lesson presents it as a calculation. The student has no way to evaluate this number. If the student takes it literally, they might conclude that MoE with top-1 of 8 experts (12.5% utilization) is the theoretically optimal configuration, which is not well-supported.
**Student impact:** The student might anchor on "10-20%" as a precise number and reason from it as established fact. When they later see Mixtral uses top-2 of 8 (25% utilization), they might wonder why it "over-activates" relative to the 10-20% claim.
**Suggested fix:** Soften the framing: "Consider how much of the FFN's stored knowledge could be relevant to any given token. 'The' does not need the parameters that encode French cooking or JavaScript syntax. For most tokens, the vast majority of FFN knowledge is irrelevant." This conveys the same motivating insight without a specific number that could become a misleading anchor.

#### [IMPROVEMENT] — Connection to reward hacking in load balancing section assumes more depth than the student has

**Location:** Load Balancing section, final paragraph
**Issue:** The lesson says: "The connection to reward hacking: the router's proxy objective ('send to the highest-scoring expert') conflicts with the system objective ('use all experts effectively'). The auxiliary loss is a constraint on the proxy, similar to how the KL penalty constrains the reward model in RLHF." The student's depth on reward hacking is DEVELOPED from the RLHF/alignment lessons (4.4.3), but the analogy drawn here is a stretch. The router is not optimizing a "proxy objective" in the RL sense. The router's behavior is a consequence of standard gradient descent on a single loss, not a separate optimization process with a proxy reward. Calling the router's preference a "proxy objective" and comparing it to the KL penalty in RLHF conflates two different mechanisms: one is a training instability (positive feedback loop in gradient updates) and the other is a misalignment between a learned reward model and a true objective.
**Student impact:** The student might form the misconception that the router is doing RL-style optimization, or that auxiliary losses are always "KL constraints on proxies." The analogy is loose enough to confuse rather than clarify.
**Suggested fix:** Either remove the reward hacking connection or make the analogy more precise: "There is a pattern here you have seen before: an unconstrained optimization process finds a degenerate solution. In RLHF, the policy exploits the reward model without a KL constraint. In MoE, gradient descent concentrates tokens on one expert without a balancing constraint. Different mechanisms, same pattern: unconstrained optimization of a local signal leads to collapse. The auxiliary loss constrains the optimization, keeping the system in a useful regime."

#### [POLISH] — Notebook uses spaced em dashes throughout

**Location:** Notebook cells 0, 1, 2, 5, 9, 15, 17, 18 (multiple markdown cells and comments)
**Issue:** The notebook consistently uses ` — ` (space-em dash-space) where the project convention is `—` (no spaces). Examples: "No pretrained models or GPUs needed — everything runs in seconds on CPU", "Setup — self-contained for Google Colab", "The router — a single matrix multiply + softmax — naturally produces different routing decisions."
**Student impact:** No pedagogical impact. Cosmetic inconsistency with the lesson's em dash style.
**Suggested fix:** Replace all ` — ` with `—` in notebook markdown cells and code comments.

#### [POLISH] — PerTokenRoutingDiagram expert labels are "E2", "E5", "E7" but the text discusses 8 experts without establishing which number maps to which specialty

**Location:** Section 9, PerTokenRoutingDiagram component and surrounding text
**Issue:** The diagram uses E2, E5, E7 as expert labels with a legend saying "Expert 2 (syntax)", "Expert 5 (science)", "Expert 7 (domain)." These specific numbers and labels are arbitrary and appear only in the diagram. The surrounding prose does not mention these expert numbers or specialties. The labeling (syntax, science, domain) is also somewhat imprecise: "domain" is vague and does not distinguish itself from "science" clearly. The misconception text below the diagram correctly says specialization is emergent and not cleanly mappable to human categories, which somewhat contradicts the clean labels in the diagram.
**Student impact:** Minor confusion. The student may wonder what makes "powerhouse" a "domain" word vs a "science" word. The clean labels in the diagram could subtly reinforce the very misconception the section is trying to dispel.
**Suggested fix:** Either remove the parenthetical specialty labels from the legend (just "Expert 2", "Expert 5", "Expert 7") and let the observation boxes below the diagram carry the analysis, or add a note that the labels are illustrative and that real expert specialization does not have clean boundaries. The observation boxes below already do a better job of this.

#### [POLISH] — Two details/summary checkpoints both use identical styling and phrasing for the reveal prompt

**Location:** Check 1 (section 8) and Check 2 (section 11)
**Issue:** Both checkpoints use `<summary className="font-medium cursor-pointer text-primary">Think about it, then reveal</summary>`. This is functional and consistent, but both checkpoints could benefit from slightly differentiated reveal prompts that hint at the nature of the exercise. Check 1 is "predict the numbers" (concrete, quantitative) and Check 2 is "apply the concepts" (analytical, comparative). The identical phrasing flattens this distinction.
**Student impact:** Very minor. The student knows to predict before revealing in both cases.
**Suggested fix:** Optionally differentiate: Check 1 could say "Make your predictions, then reveal" and Check 2 could say "Reason through it, then reveal." Low priority.

### Review Notes

**What works well:**
- The narrative arc is strong. The lesson builds from established mental models ("attention reads, FFN writes," "2/3 of parameters") to the MoE insight in a natural, motivated way. The problem-before-solution structure is excellent.
- The SVG diagrams are high quality. The MoE Architecture Diagram is clear, the color coding is consistent, and the side-by-side comparison makes the "what changes, what stays the same" immediately visible. The Router Collapse Diagram effectively visualizes the positive feedback loop.
- All 5 planned misconceptions are addressed with concrete negative examples. The misconception GradientCards are well-placed: each appears at the point where the student would form the misconception.
- The Mixtral 8x7B three-way comparison table is an excellent concrete example. The numbers make the decoupling tangible.
- The connection to attention's dot-product + softmax is the best demystifying move in the lesson. The router goes from "mysterious new component" to "something I already know" in one code block.
- The notebook exercises have strong progression (router -> forward pass -> routing patterns -> collapse) and the guided/supported/independent scaffolding is appropriate.
- The library analogy is effective and well-connected to the architecture.

**Systemic observation:**
- The lesson is dense but well-paced. The STRETCH designation is appropriate. The two core new concepts (conditional computation, router mechanism) are well-supported. Load balancing as a borderline third concept is handled at the right level of depth, though the solution mechanism needs slightly more grounding (see IMPROVEMENT finding).
- The notebook's Exercise 4 dependency on Exercise 2 is the most actionable issue. Independent exercises should be self-contained by definition.

---

## Review — 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All iteration 1 findings have been addressed. No new critical or improvement-level issues found. Two minor polish items remain.

### Iteration 1 Fix Verification

All 8 findings from iteration 1 have been verified:

1. **[CRITICAL] Notebook Exercise 2 dependency** -- FIXED. Cell-8 and cell-9 now provide a "Helper: Complete MoELayer for Remaining Exercises" section that gives a complete, working `MoELayer` and `ExpertFFN` implementation after Exercise 2's solution reveal. The helper cell includes a clear explanation: "Run the cell below to get a working MoELayer and ExpertFFN for Exercises 3 and 4. This ensures the remaining exercises work correctly regardless of whether your Exercise 2 implementation has bugs." Exercise 4 is now self-contained.

2. **[IMPROVEMENT] Objective block repetition** -- FIXED. The ObjectiveBlock now reads: "By the end of this lesson, you will be able to explain why conditional computation targets the FFN specifically, trace how a token flows through a router to selected experts, and articulate the parameter-compute tradeoff using Mixtral's concrete numbers." This is a distinct "able to..." statement, clearly different from the header description.

3. **[IMPROVEMENT] Auxiliary loss detail missing** -- FIXED. The load balancing section now includes a concrete conceptual description: "the loss measures two things: (1) what fraction of tokens in the batch were routed to it, and (2) how much probability the router assigned to it on average. If both are high for one expert, the penalty is large. If both are spread evenly, the penalty is small. This product-based penalty gently pushes the router toward balanced utilization without forcing uniform routing or overriding the router's learned preferences." This grounds the concept at INTRODUCED depth and prepares the student for the notebook's formula.

4. **[IMPROVEMENT] Unsourced "10-20%" claim** -- FIXED. The hook now uses softened framing: "Consider how much of the FFN's stored knowledge could be relevant to any given token. 'The' does not need the parameters that encode French cooking or JavaScript syntax. For most tokens, the vast majority of FFN knowledge is irrelevant." No specific percentages that could become misleading anchors.

5. **[IMPROVEMENT] Reward hacking connection too loose** -- FIXED. The text now reads: "There is a pattern here you have seen before: an unconstrained optimization process finds a degenerate solution. In RLHF, the policy exploits the reward model without a KL constraint. In MoE, gradient descent concentrates tokens on one expert without a balancing constraint. Different mechanisms, same pattern: unconstrained optimization of a local signal leads to collapse." This properly distinguishes the mechanisms while preserving the useful pattern connection.

6. **[POLISH] Notebook spaced em dashes** -- PARTIALLY FIXED. All markdown cells now use unspaced em dashes. Four code comments in Exercise 1 and Exercise 2 cells still use spaced em dashes (e.g., "Router logits — one score per expert"). See Polish finding below.

7. **[POLISH] PerTokenRoutingDiagram expert labels** -- NOT FIXED. The observation boxes still label experts with parenthetical specialties: "Expert 2 (syntax patterns)", "Expert 5 (science)", "Expert 7 (domain)." However, the diagram's bottom box states "Specialization is emergent and per-token, not designed and per-topic" and the misconception GradientCard immediately below the diagram addresses this directly. The context mitigates the issue. See Polish finding below.

8. **[POLISH] Identical reveal prompts** -- FIXED. Check 1 now says "Make your predictions, then reveal" and Check 2 says "Reason through it, then reveal."

### Findings

#### [POLISH] — Notebook code comments still use spaced em dashes

**Location:** Notebook cells 3 and 6 (Exercise 1 and Exercise 2 code cells)
**Issue:** Four code comments use ` — ` (spaced em dash): "Router logits — one score per expert", "Softmax — convert to probability distribution", "Top-k selection — pick the top-2 experts", and "Replace this — follow the approach above." The markdown cells were fixed in iteration 1 but these code comments were not updated.
**Student impact:** No pedagogical impact. Cosmetic inconsistency between code comments and lesson prose em dash style.
**Suggested fix:** Replace ` — ` with `—` in the four code comments, or replace with `: ` which is more natural for code comments (e.g., "Step 1: Router logits: one score per expert").

#### [POLISH] — PerTokenRoutingDiagram observation boxes label experts with clean specialty names that subtly contradict the "emergent, not designed" message

**Location:** Section 9, PerTokenRoutingDiagram SVG component, observation boxes at y=150
**Issue:** The observation boxes label "Expert 2 (syntax patterns)" and "Expert 5 (science)" and "Expert 7 (domain)." The label "domain" remains vague and does not clearly distinguish from "science." Clean specialty labels could subtly reinforce the misconception that expert specialization maps to human-interpretable categories, which the misconception GradientCard immediately below works to dispel.
**Student impact:** Minimal. The misconception text and diagram's bottom annotation ("Specialization is emergent and per-token, not designed and per-topic") provide adequate counterbalance. The visual and the text work together, even if the labels could be more precise.
**Suggested fix:** Low priority. Either remove parenthetical labels from observation boxes (keeping the routing pattern analysis only) or add a brief qualifier like "(approximate grouping)" to signal that these labels are human-imposed post-hoc descriptions, not designed categories.

### Review Notes

**What works well (reaffirmed from iteration 1, plus new observations):**
- All iteration 1 fixes are well-implemented. The ObjectiveBlock rewrite is particularly good -- it gives the student three concrete "able to" goals that serve as a self-check throughout the lesson.
- The auxiliary loss conceptual description added to the load balancing section is at exactly the right level for INTRODUCED depth: concrete enough to ground the concept, abstract enough not to overwhelm. The student will recognize the formula in the notebook without needing to learn it there.
- The notebook's helper cell (cell-8/cell-9) is a clean solution to the Exercise 2 dependency. The explanatory text ("this ensures the remaining exercises work correctly regardless of whether your Exercise 2 implementation has bugs") is honest and avoids making the student feel bad about needing it.
- The lesson's narrative arc, SVG diagrams, misconception coverage, and notebook progression all remain strong from iteration 1. No regression detected.

**Overall assessment:** The lesson is ready. The two remaining polish items are genuinely cosmetic and have no pedagogical impact. The lesson teaches conditional computation and the router mechanism at DEVELOPED depth with strong modality coverage (6 modalities), addresses all 5 planned misconceptions with concrete negative examples, and the notebook exercises progress cleanly from guided through independent with the dependency issue resolved.
