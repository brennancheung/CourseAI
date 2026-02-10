# Lesson Planning: the-transformer-block

**Module:** 4.2 — Attention & the Transformer
**Position:** Lesson 5 of 6
**Type:** Conceptual (no notebook)
**Cognitive load type:** BUILD

---

## Phase 1: Orient — Student State

The student arrives having built multi-head attention from scratch across four lessons. They understand every component of the MHA layer: Q/K/V projections, scaled dot-product scoring, per-head subspace operation, concatenation, and W_O mixing. They have a working residual stream concept (INTRODUCED in lesson 3, touched in lesson 4), and they know residual connections from ResNets (DEVELOPED in Series 3). They have batch normalization at DEVELOPED depth from the ResNets lesson but have never seen layer normalization. The FFN inside a transformer is just a two-layer network with ReLU/GELU, and two-layer networks are thoroughly familiar from Series 1-2.

### Relevant Concepts with Depths

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Multi-head attention (complete formula + implementation) | DEVELOPED | multi-head-attention (4.2.4) | Student can trace MHA from input through h parallel heads to W_O output. Built in notebook. |
| Residual connection / skip connection (F(x) + x) | DEVELOPED | resnets (3.2) | Core ResNet concept. "Editing a document, not writing from scratch." Learned the formulation, gradient highway, identity-as-default. |
| Residual stream in attention (attention_output + embedding) | INTRODUCED | values-and-attention-output (4.2.3) | Applied the residual idea to attention output. Student knows attention output is ADDED to input, not substituted. Transfer question answered. Touched again in lesson 4 (MHA output + input). Full architectural role across stacked layers explicitly deferred to THIS lesson. |
| Batch normalization (concept + CNN practice) | DEVELOPED | training-dynamics (1.3) + resnets (3.2) | Normalize activations between layers. Learned gamma/beta. Conv-BN-ReLU pattern. train() vs eval() distinction. Per-batch statistics during training, running averages at eval. |
| Two-layer neural network (nn.Linear -> activation -> nn.Linear) | DEVELOPED | Across Series 1-2 | Student has built and trained many two-layer networks. Understands hidden layers, activation functions, weight matrices. |
| nn.Linear as learned linear transformation | DEVELOPED | nn-module (2.1.3) | Core building block. Used extensively. W_Q, W_K, W_V, W_O all framed as nn.Linear. |
| GELU activation function | INTRODUCED | activation-functions-deep-dive (1.2) | Shape, formula (x * Phi(x)), "smooth ReLU." Decision guide: "GELU for transformers." Has not used GELU in practice. |
| ReLU activation function | DEVELOPED | activation-functions (1.2) + extensive use | Default activation. max(0,x). Used in every network since Module 1.2. |
| Vanishing/exploding gradients | DEVELOPED | training-dynamics (1.3) | Products of local derivatives. Telephone game analogy. Flatline = vanishing, NaN = exploding. |
| Gradient highway through skip connections | INTRODUCED | resnets (3.2) | Derivative = 1.0 on skip path. Presented as partial explanation alongside easier optimization. |

### Mental Models Already Established

- **"Editing a document, not writing from scratch"** (residual learning) -- from resnets (3.2)
- **"Skip connection = direct phone line"** (gradient highway) -- from resnets (3.2), extends telephone game from training-dynamics (1.3)
- **"Three lenses, one embedding"** (W_Q, W_K, W_V) -- from values-and-attention-output (4.2.3)
- **"Multiple lenses, pooled findings"** (multi-head attention) -- from multi-head-attention (4.2.4)
- **"Attention reads from the residual stream; the FFN writes new information into it"** -- seeded at end of multi-head-attention (4.2.4), not yet developed
- **"Each layer should preserve the signal"** (initialization principle) -- from training-dynamics (1.3)
- **"Capacity, not assignment"** (heads learn emergently) -- from multi-head-attention (4.2.4)

### What Was Explicitly NOT Covered in Prior Lessons

- Layer normalization (never introduced; batch norm is the only normalization the student knows)
- The FFN component of a transformer block (never discussed; all focus has been on the attention mechanism)
- Pre-norm vs post-norm ordering (never mentioned)
- The residual stream as the central highway across multiple stacked layers (explicitly deferred from lessons 3 and 4 to this lesson)
- How blocks repeat/stack (the concept of the "block" as a repeating unit)
- The parameter split between attention and FFN layers (~1/3 vs ~2/3)

### Readiness Assessment

The student is well-prepared. Every component of the transformer block is either already taught (MHA, residual connections) or a close cousin of something taught (layer norm ~ batch norm, FFN = two-layer network with familiar activation). The key new insight -- "attention reads, FFN writes" -- is genuinely new but builds directly on the residual stream concept they already have. The cognitive load is appropriate for a BUILD lesson following a STRETCH lesson (multi-head-attention).

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how the transformer block assembles multi-head attention, feed-forward networks, residual connections, and layer normalization into the repeating unit that stacks to form a transformer.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Multi-head attention (formula + computation) | DEVELOPED | DEVELOPED | multi-head-attention (4.2.4) | OK | Student must understand MHA as a complete component. They built it from scratch across 4 lessons. |
| Residual connection / skip connection (F(x) + x) | DEVELOPED | DEVELOPED | resnets (3.2) | OK | Residual connections wrap both MHA and FFN in the transformer block. Student has this at DEVELOPED depth including the "editing not writing" mental model. |
| Residual stream in attention | INTRODUCED | INTRODUCED | values-and-attention-output (4.2.3) | GAP (small) | Student knows attention output is added to input. This lesson needs to develop the concept into the residual stream as a backbone across stacked layers. Brief recap + extension, not re-teaching. |
| Batch normalization | DEVELOPED | DEVELOPED | training-dynamics (1.3) + resnets (3.2) | OK | Layer norm is the cousin of batch norm. Student has batch norm at DEVELOPED depth including formula, learned parameters, and train/eval distinction. This gives a strong foundation for introducing layer norm by contrast. |
| Two-layer neural network | DEVELOPED | DEVELOPED | Series 1-2 (multiple lessons) | OK | The FFN inside a transformer block is just a two-layer network. Student has built many. |
| nn.Linear | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | FFN is two nn.Linear layers. Thoroughly familiar. |
| GELU activation | INTRODUCED | INTRODUCED | activation-functions-deep-dive (1.2) | OK | Student knows GELU shape and "for transformers" decision. Does not need DEVELOPED depth here -- the point is "smooth ReLU alternative, not the interesting part of the FFN." |
| Vanishing/exploding gradients | DEVELOPED | DEVELOPED | training-dynamics (1.3) | OK | Needed to understand why layer norm and residual connections matter for deep stacking. |
| Gradient highway through skip connections | INTRODUCED | INTRODUCED | resnets (3.2) | OK | Needed for understanding why residual connections enable deep stacking. INTRODUCED is sufficient -- we are extending the concept, not developing the gradient math. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Residual stream (INTRODUCED -> needs DEVELOPING as cross-layer backbone) | Small | The student already knows attention output + input = residual output. This lesson extends that to: the residual stream is the central highway that flows through every sub-layer in every block. 2-3 paragraphs + a vertical block diagram showing the stream flowing through MHA -> Add&Norm -> FFN -> Add&Norm. Not re-teaching the concept, just showing its full architectural significance. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The transformer is just attention" | The entire module so far has focused on attention. Four lessons on attention, zero on anything else. Natural conclusion: attention IS the transformer. | GPT-2 parameter count: ~1/3 in attention, ~2/3 in FFN layers. If "transformer = attention," where do 2/3 of the parameters live? The FFN layers contain the majority of parameters and are believed to store factual knowledge. Removing FFN layers devastates model performance. | Main explain section, right after introducing the FFN. This is the central misconception of the lesson and the main reason the "attention reads, FFN writes" framing matters. |
| "The FFN is just a boring connector between attention layers (not doing real work)" | The FFN is architecturally simple (two linear layers + activation). Compared to the elaborate multi-head attention mechanism, it looks like plumbing. | The FFN has an expansion factor of 4x (768 -> 3072 -> 768 in GPT-2). That 4x expansion is where the model "thinks" -- the higher-dimensional space lets it compute complex feature combinations that can't be represented in the smaller d_model space. Research (Geva et al.) shows FFN neurons activate for specific concepts ("the Eiffel Tower neuron"). | Elaborate section, after the "attention reads, FFN writes" framing is established. |
| "Layer norm and batch norm are the same thing (just normalizing activations)" | Student knows batch norm well. Layer norm sounds like a variant. The names are similar. Both normalize. | Batch norm normalizes across the batch dimension (different examples, same feature). Layer norm normalizes across the feature dimension (same example, different features). Critical difference: batch norm's behavior changes between training and eval (running statistics), and it depends on other examples in the batch. Layer norm is independent of batch size and has no train/eval distinction. For sequence data with variable lengths, batch norm would normalize across different sequence positions in different examples -- nonsensical. | Dedicated section on layer norm, right after the first residual connection. Compare to batch norm with a side-by-side diagram. |
| "Pre-norm and post-norm are just implementation details (doesn't matter which)" | The student has only seen post-norm (original transformer paper). Pre-norm looks like a minor reordering. Both normalize. | Pre-norm is now the standard because it makes training more stable at depth. With post-norm, the residual stream passes through layer norm, which can distort the gradient path. With pre-norm, the residual stream has a clean identity path from input to output. GPT-2 and virtually all modern LLMs use pre-norm. Post-norm requires careful learning rate warmup; pre-norm is more forgiving. | Brief elaboration after the main block diagram is established. Not a major section -- the student should know the distinction exists and that pre-norm is standard, but this is not the lesson's core. |
| "Residual connections in transformers are just like ResNet skip connections (same purpose, same story)" | Student learned residual connections from ResNets and has seen them applied to attention output. Natural assumption: same thing. | In ResNets, the residual connection wraps 2-3 conv layers (one type of computation). In a transformer block, there are TWO residual connections: one around MHA, one around FFN. Furthermore, the residual stream is the central backbone of the entire model -- information flows through it from embedding to output, with each sub-layer "reading from and writing to" the stream. This is a richer role than skip connections in CNNs. | When introducing the block diagram. Explicitly acknowledge the callback, then extend: "same mechanism, bigger role." |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| GPT-2 transformer block with concrete dimensions (d_model=768, d_ff=3072, h=12) | Positive | Ground the abstract block diagram in real numbers. Student computes parameter counts, sees the 4x expansion, connects to models they know about. | GPT-2 is the running example from the module. Concrete numbers make the architecture tangible. The 4x expansion ratio is visible: 768 -> 3072 -> 768. Total FFN params (768*3072 + 3072*768 = ~4.7M) vs attention params (~2.4M) proves FFN is not "just plumbing." |
| Tracing a token through the complete block (input -> MHA -> Add&Norm -> FFN -> Add&Norm -> output) | Positive | Shows the data flow through the entire block. Student follows a vector's journey: same dimension in, same dimension out, but enriched by context (MHA) and processed (FFN). | Demonstrates that the block is shape-preserving (input dim = output dim), which is why blocks can stack. Also shows the residual additions concretely: the original information is never lost. |
| Removing the FFN (attention-only transformer) | Negative | Disproves "the transformer is just attention." Shows that without FFN, the model can route information (attention) but cannot process it. | Concrete: attention computes weighted averages of V vectors, which are linear projections of embeddings. Without FFN, the model is limited to linear combinations. The FFN's nonlinearity is what enables complex feature computation. Without it: attention can copy/blend information but cannot transform it. |
| Removing residual connections (what breaks) | Negative | Demonstrates why residual connections are essential, not optional. | Callback to ResNet resnets lesson: without skip connections, deep networks degrade. Same principle but worse: in a transformer, each sub-layer would need to learn to pass through ALL information plus its own contribution. The residual stream means each sub-layer only needs to learn the DELTA. Transfer question from values-and-attention-output reinforced. |
| Stacking N blocks (the repeating unit concept) | Positive (stretch) | Shows why the block is designed as a repeating unit. Same block, N times. GPT-2: 12 blocks. GPT-3: 96 blocks. | The block's shape-preserving property (d_model in, d_model out) means identical blocks can stack. Each block reads from and writes to the same residual stream. Earlier blocks capture simpler patterns, later blocks capture more complex ones (callback to hierarchical features in CNNs). |

---

## Phase 3: Design

### Narrative Arc

The student has spent four lessons building the most elaborate component of the transformer: multi-head attention. They understand every piece of it. But MHA alone is not a transformer. The last lesson ended with a seed: "Attention reads from the residual stream; the FFN writes new information into it." This lesson delivers on that promise. The motivating question is: what wraps around MHA to form the actual repeating unit of a transformer? The answer is surprisingly simple -- two residual connections, two layer norms, and a two-layer feed-forward network. The student already knows all of these building blocks (residual connections from ResNets, normalization from batch norm, two-layer networks from Series 1). The new insight is not any individual component but how they compose: attention reads context from the residual stream, the FFN processes what attention found, and the residual stream carries everything forward to the next block. This is the "attention reads, FFN writes" mental model that reframes the transformer from "a bunch of attention" to "a repeating read-process cycle."

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual (block diagram) | Vertical flow diagram: input at bottom, residual stream as a vertical highway with branch-and-merge at each sub-layer. MHA box on the left, FFN box on the right, LayerNorm before each, + nodes for residual additions. Color-coded: residual stream in violet, MHA in sky blue, FFN in amber, LayerNorm in emerald. Dimensions annotated at every stage. | The transformer block IS a data flow diagram. The visual is the concept. Student needs to see the two residual connections, the branching pattern, and the shape preservation. This is the central artifact of the lesson. |
| Concrete example (GPT-2 dimensions) | d_model=768 flowing through the block. MHA: 768 -> 12 heads x 64 dims -> concat 768 -> W_O 768. FFN: 768 -> 3072 -> 768. Layer norm: 768 -> 768. Every stage annotated with parameter counts. | Abstract diagrams are insufficient. Concrete dimensions from a real model make the architecture tangible and allow parameter counting that reveals the FFN's dominance. |
| Symbolic (formulas) | Pre-norm block equations: (1) x' = x + MHA(LayerNorm(x)), (2) output = x' + FFN(LayerNorm(x')). FFN formula: FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2. | The formulas are compact and the student can verify shape preservation. The pre-norm placement (inside the residual branch) is visible in the formula. |
| Analogy (reading and writing) | "Attention reads from the residual stream -- it gathers relevant context from other tokens. The FFN writes back -- it processes what attention found and updates the representation." Extended: "If the residual stream is a shared document, attention is reading other people's contributions and the FFN is writing your response." | This is the core mental model of the lesson. It reframes the two sub-layers from "two things that happen" to "two complementary operations with distinct roles." It also naturally explains why both are needed. |
| Verbal/comparison (layer norm vs batch norm) | Side-by-side comparison. Batch norm: normalize across examples in a batch (column-wise in a batch x features matrix). Layer norm: normalize across features in a single example (row-wise). Diagram showing which axis each normalizes along. Key difference: layer norm is independent of batch size, no train/eval mode distinction. | Student knows batch norm well. Layer norm is best taught by contrast with what they know. The comparison makes both the similarity (normalization with learned scale/shift) and the critical difference (which axis, batch dependence) explicit. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 genuinely new
  1. Layer normalization (new, but close cousin of batch norm which is DEVELOPED)
  2. The FFN's role and structure within the transformer block (new framing, but the FFN itself is just a two-layer network)
  3. "Attention reads, FFN writes" as a mental model for understanding the two sub-layers (new framing for known components)
- **Previous lesson's load:** STRETCH (multi-head-attention -- parallel heads, dimension splitting, W_O)
- **Assessment:** BUILD is appropriate. No single component is genuinely novel -- layer norm is a cousin of batch norm, the FFN is a two-layer network, residual connections are from ResNets. The novelty is in the assembly and the "reads/writes" framing. This is classic BUILD: combining known pieces into a larger structure. Coming after a STRETCH lesson, this provides appropriate cognitive relief.

### Connections to Prior Concepts

| Existing Concept | Connection | How |
|-----------------|------------|-----|
| Residual connections from ResNets (3.2) | Direct callback | "Same mechanism you learned in ResNets. F(x) + x. Same 'editing a document' analogy. But in the transformer, there are TWO residual connections per block, and the residual stream plays a bigger role -- it's the central highway across all layers." |
| Batch normalization (1.3, 3.2) | Contrast bridge | Layer norm introduced by contrast with batch norm. "You know batch norm normalizes across examples. Layer norm normalizes across features within a single example. Same idea -- normalize, then apply learned scale and shift -- but a different axis." |
| Two-layer networks (Series 1-2) | Direct callback | "The FFN inside a transformer block is just a two-layer network: nn.Linear(768, 3072) -> GELU -> nn.Linear(3072, 768). You have built dozens of these." |
| GELU activation (1.2) | Callback | "Remember GELU from the activation functions lesson? The decision guide said 'GELU for transformers.' Here's why -- this is where it lives, inside the FFN." |
| "Editing a document, not writing from scratch" (resnets 3.2) | Extended | The "editing" analogy extends naturally: MHA reads context and proposes an edit (context-enrichment). The FFN reads the edited document and proposes another edit (feature processing). Each edit is added to the original via the residual stream. |
| Hierarchical features in CNNs (3.1-3.2) | Parallel | "Earlier blocks capture simpler patterns, later blocks capture more complex ones. Same principle as CNN layer hierarchy, but now the 'features' are contextual relationships between tokens rather than spatial patterns." |
| Gradient highway (resnets 3.2) | Extended | "The gradient highway from ResNets applies here too, but now there are TWO skip connections per block, creating an even more direct path from output to input. In a 12-block GPT-2, gradients can flow through 24 residual additions." |

### Analogies from Prior Lessons -- Risks

- **"Editing a document" (residual learning)** -- extends cleanly. No risk of misleading.
- **"Skip connection = direct phone line"** -- extends cleanly to "two phone lines per block."
- **"Three lenses, one embedding"** -- this analogy is about Q/K/V, not about the block. No need to extend it here. Risk: student might try to force-fit it. Mitigation: don't reference it in this lesson.

### Scope Boundaries

**This lesson IS about:**
- The transformer block as the repeating unit: MHA + FFN + residual connections + layer norm
- The "attention reads, FFN writes" mental model
- Layer normalization (introduced by contrast with batch norm, INTRODUCED depth)
- Pre-norm vs post-norm (INTRODUCED depth -- know the distinction and that pre-norm is standard)
- The FFN as a two-layer network with 4x expansion factor (DEVELOPED depth)
- The residual stream as the central backbone across stacked layers (DEVELOPED depth, upgraded from INTRODUCED)
- Why this block can stack (shape preservation)
- Parameter split between attention and FFN (~1/3 vs ~2/3)

**This lesson is NOT about:**
- Implementing the transformer block in PyTorch (that's Module 4.3, building-nanogpt)
- Causal masking (Lesson 6)
- The full decoder-only architecture (Lesson 6)
- Training transformers (Module 4.3)
- How many blocks to use / scaling (Module 4.3)
- Cross-attention, encoder-decoder architecture
- Specific layer norm variants (RMSNorm)
- Positional encoding interaction with the block (already taught in 4.1)
- Mixture of experts or other FFN variations

**Depth targets:**
- Layer normalization: INTRODUCED (know what it does, how it differs from batch norm, where it goes)
- FFN structure and role: DEVELOPED (understand the 4x expansion, GELU, parameter count, "writes" role)
- Residual stream as cross-layer backbone: DEVELOPED (upgraded from INTRODUCED in lesson 3)
- Pre-norm vs post-norm: INTRODUCED (know the distinction and standard choice)
- Transformer block as repeating unit: DEVELOPED (can explain all components and their roles)
- "Attention reads, FFN writes" mental model: DEVELOPED (can apply to explain why both sub-layers are needed)

### Lesson Outline

1. **Context + Constraints**
   - What: "This lesson assembles the transformer block -- the repeating unit that stacks to form a transformer."
   - NOT: implementing in code, causal masking, the full architecture. "We are zooming in on one block."
   - Callback to lesson 4 seed: "Attention reads from the residual stream; the FFN writes new information into it."

2. **Hook (the missing 2/3)**
   - Type: Misconception reveal + puzzle
   - "You've spent four lessons on attention. It's natural to think the transformer IS attention. Here's a number that might surprise you." GPT-2 parameter count: ~124M total. Attention parameters: ~28M. That's about 1/4. "Where are the other 3/4?" Reveal: ~57M in FFN layers, ~38M in embeddings. The FFN layers contain more parameters than attention. "What are they doing?"
   - Why this hook: Directly challenges the "transformer = attention" misconception that has been building for four lessons. Creates a genuine puzzle: if attention is so important, why does the FFN have more parameters?

3. **Explain: The Block Diagram (core concept)**
   - Start with MHA as a known black box. "You know what goes in and what comes out."
   - First addition: residual connection around MHA. Callback: "Same F(x) + x from ResNets. The attention output is the edit, the input is the document." Student already knows this from lesson 3.
   - Second addition: layer normalization. "Before the input goes into MHA, we normalize it." Transition to dedicated layer norm section.

4. **Layer Norm Section**
   - Motivation: "Why normalize at all?" Callback to batch norm from training-dynamics and resnets. "You know that normalizing activations helps training. But batch norm has a problem for sequences."
   - The problem with batch norm for sequences: batch norm normalizes across examples in a batch. For sequences of variable length, this means normalizing across different positions in different examples -- position 5 of a 10-token sentence gets averaged with position 5 of a 100-token sentence. Nonsensical.
   - Layer norm solution: normalize across features within a single example. Each token's d_model-dimensional vector is independently normalized to mean 0, variance 1, then scaled/shifted by learned gamma and beta.
   - Comparison (batch norm vs layer norm): side-by-side diagram showing which axis each normalizes along. Batch norm: column-wise (across batch). Layer norm: row-wise (across features). Key differences: (1) layer norm independent of batch size, (2) no train/eval mode distinction, (3) each example normalized independently.
   - Formula: LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta, where mu and sigma are per-example, per-position statistics.
   - Depth target: INTRODUCED. Student should understand what it does and why it replaces batch norm, not derive or implement it.

5. **Check 1: Predict-and-verify**
   - "Why can't we use batch norm in a transformer?" Student should articulate: variable-length sequences make cross-example normalization at each position meaningless; layer norm normalizes each token independently.
   - Follow-up: "Does layer norm need different behavior at train vs eval time?" No -- it computes per-example statistics, not running averages.

6. **Explain: The FFN (the other half)**
   - Reveal the FFN structure: two nn.Linear layers with GELU activation between them.
   - Formula: FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2
   - The expansion factor: W_1 is (d_model, 4*d_model), W_2 is (4*d_model, d_model). GPT-2: 768 -> 3072 -> 768.
   - "Attention reads, FFN writes" framing: attention gathers context from other tokens (reads the stream). The FFN processes what attention found and updates the representation (writes to the stream). Different operations, complementary roles.
   - Why the expansion? The 4x wider hidden layer creates a higher-dimensional space where the model can compute complex feature combinations that don't fit in d_model dimensions. Then it projects back down, keeping only what's useful.
   - Parameter count: FFN per block = 2 * d_model * d_ff = 2 * 768 * 3072 = 4,718,592. Attention per block (Q+K+V+O) = 4 * d_model^2 = 4 * 768^2 = 2,359,296. FFN has 2x the parameters of attention. This answers the hook's puzzle.
   - GELU callback: "The activation functions lesson said 'GELU for transformers.' This is where it lives."
   - Second residual connection: FFN output is also added to its input via a residual connection. Same pattern as MHA.

7. **The Complete Block**
   - Assemble the full pre-norm block with formulas:
     - x' = x + MHA(LayerNorm(x))
     - output = x' + FFN(LayerNorm(x'))
   - Central block diagram (the visual centerpiece): vertical flow with residual stream as the backbone. Color-coded components. Dimensions annotated.
   - Pre-norm vs post-norm: brief comparison. Original paper (2017) used post-norm: x' = LayerNorm(x + MHA(x)). Modern standard is pre-norm: x' = x + MHA(LayerNorm(x)). The difference: with pre-norm, the residual stream has a clean identity path (no layer norm on the main highway). This makes training more stable at depth. GPT-2 and virtually all modern LLMs use pre-norm.
   - Shape preservation: input to block is (n, d_model), output is (n, d_model). This is why blocks can stack.

8. **Check 2: Apply the mental model**
   - "A token just went through block 5 of a 12-block GPT-2. Its representation now contains information from attention (context from other tokens) and FFN processing. What happens next?" Expected: it enters block 6 as input. Block 6's MHA reads from the updated residual stream (which now includes contributions from blocks 1-5). Block 6's FFN processes what block 6's attention found.
   - "Why does stacking more blocks help?" Expected: each block refines the representation. Earlier blocks capture simpler patterns (adjacent tokens, basic syntax). Later blocks capture more complex patterns (long-range dependencies, semantic relationships). Callback to CNN hierarchical features.

9. **Elaborate: The Residual Stream as Backbone**
   - Develop the residual stream from INTRODUCED to DEVELOPED.
   - "In ResNets, the skip connection helps one block. In a transformer, the residual stream is the backbone of the entire model."
   - The stream flows from embedding input to final output. Every sub-layer (every MHA, every FFN, in every block) reads from and writes to this stream.
   - Analogy extension: "The residual stream is a shared document that starts as the raw embedding. Block 1's attention reads it and adds context notes. Block 1's FFN reads the annotated version and adds processed insights. Block 2 reads the version enriched by block 1. By block 12, the document has been annotated by 24 sub-layers."
   - The negative example: remove residual connections. Each sub-layer must pass through ALL information (identity + its contribution). Callback to ResNet degradation problem: learning identity in a plain layer is hard. With residual connections, each sub-layer only learns the delta.
   - Gradient flow: with residual connections, gradients have a direct path from output to any layer's input. In a 12-block GPT-2 with 2 sub-layers per block, there are 24 residual additions. The gradient highway from ResNets, but deeper and more critical.

10. **Elaborate: Why the FFN Matters (addressing "just plumbing" misconception)**
    - The attention-only negative example: without FFN, each block is just weighted averaging of linearly projected vectors. The model can route and blend information but cannot transform it nonlinearly. The FFN's GELU is the source of nonlinear computation within each block.
    - Research insight (kept light): work by Geva et al. suggests FFN layers function as key-value memories where the first layer's rows are "keys" matching input patterns and the second layer's rows are "values" storing associated information. "The Eiffel Tower neuron" -- specific FFN neurons activate for specific concepts.
    - Parameter count echo: "2/3 of the model's parameters live in FFN layers. Those parameters store the knowledge the model has learned."
    - InsightBlock: "Attention decides which tokens are relevant to each other. The FFN decides what to DO with that information. The transformer needs both."

11. **Check 3: Transfer question**
    - "A colleague proposes making the transformer 'more efficient' by reducing the FFN expansion factor from 4x to 1x (no expansion -- just d_model -> d_model -> d_model). What would you tell them?"
    - Expected: The expansion gives the FFN a higher-dimensional space to compute in. Reducing it to 1x dramatically cuts the FFN's capacity (from ~4.7M to ~1.2M parameters per block in GPT-2). The model would lose most of its ability to store and process knowledge. The expansion factor is not wasteful -- it's where the work happens.

12. **Summarize**
    - The transformer block: MHA + FFN + 2 residual connections + 2 layer norms.
    - "Attention reads, FFN writes" -- attention gathers context from other tokens, FFN processes and transforms.
    - The block is shape-preserving (d_model in, d_model out), so identical blocks stack.
    - The residual stream is the central backbone -- every sub-layer reads from and writes to it.
    - Layer norm stabilizes training; pre-norm is the modern standard.
    - ~1/3 of parameters in attention, ~2/3 in FFN. The FFN is not plumbing -- it's where knowledge lives.

13. **Next Step**
    - "We have the transformer block -- the repeating unit. But there is a critical constraint we have not addressed. In next-token prediction, the model should NOT be able to look at future tokens. How do we prevent a token from attending to positions that come after it? That's causal masking, and it's what makes this architecture a decoder -- the architecture behind GPT."

### Widget

No interactive widget for this lesson. The lesson is conceptual and the primary visual artifact is the block diagram (inline SVG). The block diagram should be carefully designed with:
- Vertical flow (bottom to top, matching the "stream flows upward" framing)
- Color-coded components (violet for residual stream, sky blue for MHA, amber for FFN, emerald for LayerNorm)
- Dimension annotations at every stage
- Two clear branch-and-merge points for the residual connections
- Optional: a "zoom out" view showing 3 blocks stacked, demonstrating the repeating pattern

Consider a static but well-annotated SVG rather than an interactive widget. The lesson's value is in the framing and mental model, not in parameter manipulation.

### Concept Depths After This Lesson

| Concept | Depth | Notes |
|---------|-------|-------|
| Layer normalization | INTRODUCED | Knows what it does, how it differs from batch norm, where it goes in the block. Does not need to implement or derive. |
| FFN structure and role in transformer | DEVELOPED | Understands 4x expansion, GELU, parameter count, and the "writes" role. Can explain why FFN matters. |
| Residual stream as cross-layer backbone | DEVELOPED | Upgraded from INTRODUCED (lesson 3). Understands the stream as the central highway across all blocks, not just a single skip connection. |
| Pre-norm vs post-norm | INTRODUCED | Knows the distinction, knows pre-norm is standard, knows why (cleaner gradient path). |
| Transformer block as repeating unit | DEVELOPED | Can name all components, explain their roles, trace data flow, explain why blocks stack. |
| "Attention reads, FFN writes" | DEVELOPED | Can use this mental model to explain why both sub-layers are needed and what each contributes. |

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues found -- the lesson is structurally sound, follows the plan closely, and the student would not be lost or form fundamentally wrong mental models. However, there are improvement-level findings that would make the lesson significantly more effective.

### Findings

#### [IMPROVEMENT] — Missing negative example: attention-only transformer is told, not shown

**Location:** Section 12 ("Why the FFN Matters"), paragraph 2
**Issue:** The plan calls for a concrete negative example showing what breaks when you remove the FFN (attention-only transformer). The lesson states "without FFN, each block is just weighted averaging of linearly projected vectors" and says the model "could not transform it." This is assertion, not demonstration. There are no concrete numbers, no worked example, no side-by-side comparison showing output with vs. without FFN. The student is asked to take this on faith.
**Student impact:** The student might accept the claim intellectually without truly feeling why it matters. The whole point of the "attention reads, FFN writes" mental model is that BOTH are necessary. Without a concrete negative example, the FFN's importance remains abstract rather than visceral.
**Suggested fix:** Add a small concrete example or thought experiment. For instance: "Consider the attention output for 'bank' -- it's a weighted average of V vectors. Weighted averages can only produce outputs that lie within the convex hull of the input vectors. The FFN's nonlinearity breaks this constraint, allowing the model to compute representations that no single input token could provide." Alternatively, show a simple 2D example where weighted averaging can only produce points on a line segment, while FFN+GELU can reach points outside it.

#### [IMPROVEMENT] — Missing negative example: removing residual connections is asserted but not demonstrated

**Location:** Section 11 ("The Residual Stream"), second-to-last paragraph
**Issue:** The plan calls for a negative example showing what breaks without residual connections. The lesson says "each sub-layer would need to pass through ALL information (identity plus its own contribution)" and callbacks to ResNets. But there is no concrete demonstration -- no side-by-side, no worked example, no specific description of what degradation looks like. The student who already understood this from ResNets gets a reminder. The student who is shaky on it gets another abstract statement.
**Student impact:** The residual stream is being upgraded from INTRODUCED to DEVELOPED in this lesson. Relying only on assertion and callback for the negative case undercuts that depth target. The "without residual connections" argument is one of the strongest motivators for why the architecture is designed this way.
**Suggested fix:** Add a brief concrete scenario: "Without the residual connection around MHA, Block 1's FFN receives only the attention output -- the original embedding is gone. If attention produced a near-uniform average (as it might early in training), the token's identity is destroyed. With the residual connection, the original embedding flows through untouched, and attention's contribution is additive -- it can only help, never erase." This makes the failure mode tangible.

#### [IMPROVEMENT] — Misconception #4 (pre-norm vs post-norm is "just an implementation detail") is addressed weakly

**Location:** Section 9 (pre-norm vs post-norm)
**Issue:** The plan identifies this as a misconception to address with a concrete negative example. The lesson explains the difference clearly and states that pre-norm keeps the residual stream "clean" with no norm on the main highway. However, it does not concretely demonstrate WHY post-norm is problematic. The statement "post-norm requires careful learning rate warmup" is mentioned in the ComparisonRow but never explained. The student is told pre-norm is better but does not feel the problem with post-norm.
**Student impact:** The student will correctly remember "pre-norm is standard" but won't have a mental model for WHY. This is acceptable at INTRODUCED depth, but the misconception specifically calls for addressing the false belief that "it doesn't matter which," and the lesson doesn't quite land that punch.
**Suggested fix:** Add one sentence after the ComparisonRow making the gradient argument more concrete: "With post-norm, every gradient flowing through the residual stream must pass through a layer norm operation at every block -- 12 layer norms in GPT-2. With pre-norm, the gradient has a clean additive path that bypasses all norms. The difference compounds with depth, which is why post-norm becomes unstable in very deep models."

#### [IMPROVEMENT] — Misconception #5 ("residual connections in transformers are the same as ResNet skip connections") is not explicitly addressed as a misconception

**Location:** Throughout sections 4, 11
**Issue:** The plan identifies this as a misconception and calls for explicitly acknowledging the callback then extending: "same mechanism, bigger role." The lesson does this implicitly -- the "Residual Stream" section (11) describes the broader role. But it never directly names the misconception. The student who thinks "oh, same as ResNets, I already know this" might skim Section 11 without registering the key distinction. The InsightBlock aside says "Same Mechanism, Bigger Role" but the main content doesn't explicitly say "you might think this is just the same thing you learned in ResNets -- it's not."
**Student impact:** A student with strong ResNet understanding might not engage deeply with Section 11, missing the upgrade from INTRODUCED to DEVELOPED for the residual stream concept.
**Suggested fix:** Add an explicit sentence at the start of Section 11: "You know residual connections from ResNets, where they help individual blocks learn useful residuals. It is tempting to think the transformer uses them the same way. The mechanism is identical -- same F(x) + x. But the role is fundamentally larger." This is nearly what the plan says, and the lesson is close but slightly too implicit.

#### [POLISH] — Hook parameter numbers don't perfectly match the plan

**Location:** Section 3 ("The Missing Two-Thirds"), the parameter breakdown
**Issue:** The plan says "~124M total. Attention parameters: ~28M. That's about 1/4." and "'Where are the other 3/4?'" The lesson says ~28M attention (~23%), ~57M FFN (~46%), ~38M embeddings (~31%). These add to ~123M, close enough. But the plan's framing is "1/4 attention, where are the other 3/4?" while the lesson frames it as "attention vs FFN" (the 2:1 ratio). The lesson's framing is arguably better (more focused on the FFN puzzle), but it deviates slightly from the plan's "3/4" framing.
**Student impact:** Negligible -- the lesson's version is actually more precise and more useful, since it separates embeddings from FFN parameters.
**Suggested fix:** No change needed. The lesson's framing is an improvement over the plan. Document this as an intentional deviation.

#### [POLISH] — Layer norm section leads with batch norm's problems rather than the problem layer norm solves

**Location:** Section 5 ("Layer Normalization"), first two paragraphs
**Issue:** The section opens with "Batch normalization normalizes each feature across all examples in a batch" and then explains why it fails for sequences. This follows the "problem before solution" ordering rule, which is good. However, the section subtitle ("Same idea as batch norm, different axis") frames layer norm as a variant of batch norm rather than as a solution to a training stability problem. A student unfamiliar with the broader context might wonder "why are we talking about normalization at all?" before the connection to training stability is made.
**Student impact:** Minor -- the student has batch norm at DEVELOPED depth and the transition is natural. But the motivation could be slightly stronger if the section opened with "the transformer block needs normalization between sub-layers to train stably at depth" before diving into "but batch norm won't work."
**Suggested fix:** Consider adding one sentence before the batch norm comparison: "As you learned in the ResNets and training dynamics lessons, normalizing activations between layers helps deep networks train. The transformer block normalizes before each sub-layer. But the normalization method you know -- batch norm -- won't work here."

#### [POLISH] — The "shared document analogy" box in Section 11 repeats the aside content

**Location:** Section 11 ("The Residual Stream"), violet callout box + InsightBlock aside
**Issue:** The violet callout box ("The shared document analogy (extended)") and the InsightBlock aside ("Same Mechanism, Bigger Role") both describe the residual stream as the central highway. The aside says "central highway through the entire model -- from embedding to output. Every sub-layer in every block reads from and writes to this shared stream." The callout box says "shared document that starts as the raw embedding... annotated by 24 sub-layers." These are the same point in two modalities (analogy vs direct statement), which is fine, but they appear very close together and partially overlap in phrasing.
**Student impact:** Minor redundancy. The student gets the point but might feel the lesson is repeating itself.
**Suggested fix:** Differentiate the aside content more -- perhaps the aside could focus on the gradient flow benefit while the main content focuses on the document analogy. Or simply accept the redundancy as reinforcement.

### Review Notes

**What works well:**
- The hook is excellent. The parameter count puzzle genuinely challenges the "transformer = attention" misconception and creates real curiosity about what the FFN does.
- The "attention reads, FFN writes" mental model is established clearly and reinforced in the summary and echo sections.
- The block diagram SVG is well-designed with clear color coding, dimension annotations, and pre-norm placement visible.
- The stacked blocks diagram effectively shows the repeating pattern.
- Layer norm is taught well by contrast with batch norm -- this is exactly the right approach given the student's existing knowledge.
- The three comprehension checks are well-placed and test genuine understanding, not recall. The transfer question (reducing FFN expansion to 1x) is particularly good.
- Scope boundaries are clearly stated and respected -- the lesson does not drift into causal masking, PyTorch implementation, or other out-of-scope topics.
- Em dashes are properly formatted throughout (no spaces).
- All interactive elements (reveal summaries) have appropriate cursor styles.
- The Row layout is used consistently throughout.

**Pattern observation:**
The lesson's main weakness is a tendency toward assertion over demonstration for negative examples. The plan identifies five misconceptions with corresponding negative examples, but in the built lesson, two of the negative examples (attention-only transformer, removing residual connections) are stated as facts rather than concretely shown. The positive examples (GPT-2 parameter count, tracing through blocks, stacking) are well-executed. The lesson would benefit from making the negative cases more tangible.

**Modality check (Step 3):**
- Verbal/Analogy: Present (shared document analogy, attention reads/FFN writes, editing not writing)
- Visual: Present (TransformerBlockDiagram SVG, StackedBlocksDiagram SVG)
- Symbolic: Present (block formulas, FFN formula, LayerNorm formula, parameter count derivations)
- Concrete example: Present (GPT-2 dimensions throughout, parameter counts)
- Intuitive: Present (workspace analogy for 4x expansion, "attention can only route, not transform")

Five modalities for the core concept -- exceeds the minimum of 3. The modalities are genuinely different perspectives, not rephrasing.

**Example count:**
- Positive examples: GPT-2 block with concrete dimensions (strong), tracing through blocks (in check 2, adequate), stacking N blocks (via diagram, adequate)
- Negative examples: Attention-only / removing FFN (asserted, not demonstrated -- flagged above), removing residual connections (asserted, not demonstrated -- flagged above)

The negative examples exist in the text but lack concrete demonstration. This is the main area for improvement.

**Load check:** New concepts: layer normalization (new but cousin of batch norm), FFN role (new framing, familiar architecture), "attention reads, FFN writes" (new mental model), pre-norm vs post-norm (brief). This is 2-3 genuinely new concepts depending on how you count. Within the guideline of 2-3.

**Connection check:** Every new concept is explicitly connected to prior knowledge: layer norm to batch norm, FFN to two-layer networks and GELU decision guide, residual connections to ResNets, gradient flow to training dynamics. Strong connections throughout.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All four improvement findings from iteration 1 have been correctly applied. The lesson is structurally sound, pedagogically effective, and ready to ship. The two polish items below are minor and do not require a re-review.

### Findings

#### [POLISH] — Em dashes with spaces in FFN formula annotations

**Location:** Section 7 ("The Feed-Forward Network"), FFN formula annotations (lines 993, 998)
**Issue:** Two em dashes (`&mdash;`) have spaces on both sides, in the formula annotation lines for W_1 ("&mdash; expands from 768 to 3072") and W_2 ("&mdash; compresses from 3072 back to 768"). The writing style rule requires em dashes with no spaces: `word&mdash;word` not `word &mdash; word`.
**Student impact:** None. Purely a formatting consistency issue.
**Suggested fix:** Remove the spaces around both em dashes, or restructure as parenthetical descriptions (e.g., "(expands from 768 to 3072)").

#### [POLISH] — Convex hull claim slightly imprecise with residual connections present

**Location:** Section 12 ("Why the FFN Matters"), paragraph 3 (line 1359)
**Issue:** The paragraph states "Without it [FFN], stacking more blocks changes nothing — a stack of linear operations is still linear, still stuck inside the same convex hull." The convex hull framing applies cleanly to a single attention operation (attention output is a convex combination of V vectors). However, with residual connections, x + attention(x) moves outside the convex hull of the V vectors since the original embedding is added back. The deeper point (without nonlinearity, the entire stack is affine and limited in expressiveness) is correct, but the specific "still stuck inside the same convex hull" claim is imprecise when residual additions are factored in.
**Student impact:** Minimal. The core insight (FFN provides essential nonlinearity) is correct and well-motivated. A very careful student familiar with the additive nature of residual connections might notice the tension, but this is unlikely to cause confusion or a wrong mental model. The pedagogical value of the convex hull framing outweighs the imprecision.
**Suggested fix:** Either accept as-is (the pedagogical framing works), or adjust the sentence to: "Without it, each attention sub-layer can only blend existing representations — the nonlinearity that enables genuinely new computations is missing." This avoids the convex hull claim for the stacked case while keeping the core insight.

### Review Notes

**Fix verification (iteration 1 findings):**

All four improvement fixes from iteration 1 were applied correctly and effectively:

1. **Attention-only / convex hull negative example (IMPROVEMENT):** The convex hull argument with the A, B, C triangle example is concrete and accessible. The student can picture the geometric constraint and understand why nonlinearity is needed. Well-executed.

2. **Removing residual connections negative example (IMPROVEMENT):** The early-training uniform attention scenario is vivid and specific. The contrast between "near-uniform average is all that passes to the FFN" (without residual) vs "added to the original embedding" (with residual) makes the failure mode tangible. The extension to "24 sub-layers each learning near-perfect identity" ties it back to ResNets. Well-executed.

3. **Pre-norm vs post-norm gradient argument (IMPROVEMENT):** The added paragraph makes the gradient path difference concrete with specific numbers (24 layer norms vs clean path). The "compounds with depth" phrasing connects the scaling concern. Well-executed.

4. **Explicit misconception addressing for residual connections (IMPROVEMENT):** The opening sentence of the Residual Stream section now directly names the misconception ("It is tempting to think the transformer uses them the same way") and immediately pivots to "the role is fundamentally larger." This prevents the student from skimming. Well-executed.

Both polish fixes from iteration 1 were also applied:

5. **Layer norm section motivation (POLISH):** The added opening sentence provides the "why normalize at all?" context before diving into batch norm's problems. Smooth transition.

6. **Aside differentiation (POLISH):** The InsightBlock in the Residual Stream section now focuses on gradient flow ("24 residual additions") rather than repeating the backbone metaphor, which stays in the main text's document analogy. Eliminates redundancy.

**What works well (unchanged from iteration 1):**
- The hook remains excellent — genuine surprise at FFN parameter dominance
- "Attention reads, FFN writes" mental model is clearly established and reinforced
- Block diagram SVGs are well-designed with consistent color coding
- Layer norm taught effectively by contrast with batch norm
- Three comprehension checks test genuine understanding
- Scope boundaries clearly stated and respected
- All Row layouts used correctly throughout
- All interactive elements have appropriate cursor styles

**Overall assessment:** The lesson is pedagogically strong, follows the plan faithfully, addresses all five planned misconceptions with concrete examples, provides five modalities for the core concept, stays within cognitive load limits (2-3 new concepts), and connects every new concept to prior knowledge. The two remaining polish items are minor formatting and precision issues that do not affect student understanding. The lesson is ready to ship.
