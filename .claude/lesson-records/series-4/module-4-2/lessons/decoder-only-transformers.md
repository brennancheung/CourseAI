# Lesson Planning: decoder-only-transformers

**Module:** 4.2 -- Attention & the Transformer
**Position:** Lesson 6 of 6 (final lesson in module)
**Type:** Conceptual (no notebook)
**Cognitive load type:** CONSOLIDATE

---

## Phase 1: Orient -- Student State

The student arrives having built the complete transformer block across five lessons. They understand every piece: raw dot-product attention, Q/K/V projections, scaled dot-product scoring, multi-head attention with dimension splitting and W_O, the FFN with 4x expansion, residual connections, layer normalization, and the "attention reads, FFN writes" mental model. The transformer block is at DEVELOPED depth -- the student can trace data flow, explain component roles, and state why the block is shape-preserving.

Critically, every attention example so far has used full (bidirectional) attention. The student has never seen masking. The attention matrix widget from Lesson 1 shows every token attending to every other token. All worked examples compute pairwise scores for all token positions. The student implicitly assumes attention is bidirectional -- this assumption has never been challenged because it was pedagogically useful to defer causal masking to this lesson.

The student also knows the complete input pipeline from Module 4.1: text becomes tokens (BPE), tokens become IDs, IDs become embedding vectors, and positional encoding injects order. They know autoregressive generation (sample, append, repeat) at DEVELOPED depth from what-is-a-language-model. They know next-token prediction as the training task. These concepts have been established but not directly connected to the transformer architecture -- this lesson makes that connection.

### Relevant Concepts with Depths

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Transformer block (MHA + FFN + residual + layer norm) | DEVELOPED | the-transformer-block (4.2.5) | Student can name all components, explain roles, trace data flow, explain stacking. Shape-preserving: (n, d_model) in and out. |
| Multi-head attention (complete formula + computation) | DEVELOPED | multi-head-attention (4.2.4) | Student can trace MHA from input through h parallel heads to W_O output. Built in notebook. |
| Attention weight matrix (row i = token i's weights over all tokens) | DEVELOPED | the-problem-attention-solves (4.2.1) | Student has computed and visualized attention matrices. Interactive widget. Each row sums to 1. |
| Softmax normalization in attention | DEVELOPED | the-problem-attention-solves (4.2.1) | Softmax applied row-wise to score matrix. Temperature callback from what-is-a-language-model. |
| Residual stream as cross-layer backbone | DEVELOPED | the-transformer-block (4.2.5) | Upgraded from INTRODUCED in 4.2.3. Shared document analogy. 24 sub-layers each read/write. Gradient highway with 24 additions. |
| "Attention reads, FFN writes" mental model | DEVELOPED | the-transformer-block (4.2.5) | Attention gathers context from other tokens; FFN processes and transforms. Complementary roles. |
| Stacking N identical blocks | DEVELOPED | the-transformer-block (4.2.5) | Shape preservation enables stacking. GPT-2: 12 blocks. StackedBlocksDiagram SVG. Earlier blocks = simpler patterns, later blocks = more complex. |
| Parameter distribution (~1/3 attention, ~2/3 FFN) | DEVELOPED | the-transformer-block (4.2.5) | GPT-2: ~28M attention, ~57M FFN, ~38M embeddings. FFN stores knowledge. |
| Autoregressive generation (sample, append, repeat) | DEVELOPED | what-is-a-language-model (4.1.1) | The feedback loop. Five-step walkthrough. Mermaid diagram. |
| Next-token prediction as training task | DEVELOPED | what-is-a-language-model (4.1.1) | P(x_t given x_1,...,x_{t-1}). Self-supervised labels. Universal training signal. |
| Token embeddings as learned lookup (nn.Embedding) | DEVELOPED | embeddings-and-position (4.1.3) | nn.Embedding(50000, 768) = 38.4M params. Learned parameters, not preprocessing. |
| Positional encoding (sinusoidal + learned) | DEVELOPED | embeddings-and-position (4.1.3) | Added to embeddings. GPT-2 uses learned PE. input_i = embedding(token_i) + PE(i). |
| Temperature as distribution reshaping | INTRODUCED | what-is-a-language-model (4.1.1) | Interactive TemperatureExplorer. softmax(logits/T). Sampling, not knowledge. |
| Output layer / unembedding (d -> V logits) | MENTIONED | embeddings-and-position (4.1.3) | Violet aside noting embedding maps V->d, output maps d->V. Everything interesting happens in d-dimensional space. |
| Softmax saturation / vanishing gradients from large inputs | DEVELOPED | queries-and-keys (4.2.2) | Scaling by sqrt(d_k) prevents saturation. Concrete numerical examples. |

### Mental Models Already Established

- **"Attention reads, FFN writes"** -- attention gathers context; FFN processes and transforms (the-transformer-block)
- **"The residual stream is a shared document"** -- each sub-layer reads and annotates it; 24 sub-layers in GPT-2 (the-transformer-block)
- **"Multiple lenses, pooled findings"** -- multi-head attention with W_O synthesis (multi-head-attention)
- **"Three lenses, one embedding"** -- W_Q, W_K, W_V extract different views (values-and-attention-output)
- **"Editing a document, not writing from scratch"** -- residual learning (resnets, extended in transformer block)
- **"A language model approximates P(next token | context)"** -- the defining objective (what-is-a-language-model)
- **"Autoregressive generation is a feedback loop"** -- outputs become inputs (what-is-a-language-model)
- **"Token embedding + positional encoding = the model's input"** -- complete input pipeline (embeddings-and-position)
- **"Capacity, not assignment"** -- heads learn emergently, not by design (multi-head-attention)

### What Was Explicitly NOT Covered in Prior Lessons

- Causal masking (explicitly deferred from EVERY prior lesson in this module -- scope boundaries in all five lesson plans say "causal masking -- Lesson 6")
- How the transformer is used for next-token prediction specifically (the training task was introduced in Module 4.1 but never connected to the transformer architecture)
- The output projection layer that maps d_model to vocabulary logits (MENTIONED as a symmetry aside in 4.1.3 but never developed)
- Why "decoder-only" is the name (the encoder/decoder distinction from the original 2017 paper)
- The original Transformer paper's encoder-decoder architecture
- BERT vs GPT as different architectural choices
- How the full GPT architecture is assembled end-to-end (embedding -> blocks -> output)
- Parameter counting for a complete model (per-block counts done, but total model with embeddings + output projection not assembled)
- Why decoder-only won over encoder-decoder for LLMs

### Readiness Assessment

The student is very well-prepared. This is a CONSOLIDATE lesson by design -- the student has all the building blocks at DEVELOPED depth. The transformer block is a known, understood unit. Autoregressive generation is a known, understood concept. The one genuinely new concept (causal masking) is mechanically simple: set certain attention scores to negative infinity before softmax. The cognitive challenge is understanding WHY (connecting the training task to the architectural constraint) and WHERE (causal masking operates within the attention computation the student already knows). The rest of the lesson is assembly: connecting the input pipeline (4.1) through stacked transformer blocks (4.2.5) to an output projection that produces next-token logits. This is the right time for a CONSOLIDATE lesson -- the student needs to see the whole picture after five lessons of building pieces.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how the complete GPT architecture assembles the components from this module (causal masking + N stacked transformer blocks) with the input pipeline from Module 4.1 (embedding + positional encoding) and an output projection into a decoder-only transformer that performs next-token prediction.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Transformer block (complete) | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | The block is the repeating unit we stack. Student can trace data flow through all components. |
| Attention weight matrix (n x n, row-wise softmax) | DEVELOPED | DEVELOPED | the-problem-attention-solves (4.2.1) | OK | Causal masking modifies this matrix. Student must understand the matrix structure to see how masking works. |
| Softmax normalization | DEVELOPED | DEVELOPED | the-problem-attention-solves (4.2.1) | OK | Masking happens BEFORE softmax. The student must understand that softmax renormalizes after masked positions are zeroed out. |
| Stacking N identical blocks | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | GPT architecture = embedding + N blocks + output projection. Stacking already understood. |
| Autoregressive generation (sample, append, repeat) | DEVELOPED | DEVELOPED | what-is-a-language-model (4.1.1) | OK | Causal masking exists BECAUSE of the autoregressive task. Student must understand the generation process to see why future tokens can't be accessed. |
| Next-token prediction as training task | DEVELOPED | DEVELOPED | what-is-a-language-model (4.1.1) | OK | The training objective dictates the architectural constraint (masking). Student must understand what the model is trying to do. |
| Token embedding + positional encoding | DEVELOPED | DEVELOPED | embeddings-and-position (4.1.3) | OK | The input pipeline feeds into the first block. Already well-established. |
| Output layer / unembedding (d -> V) | MENTIONED | MENTIONED | embeddings-and-position (4.1.3) | GAP (small) | The student has only seen a one-sentence aside noting the symmetry. This lesson needs to develop the output projection as the final step: d_model -> vocabulary logits -> softmax -> probabilities. Brief section, not re-teaching nn.Linear. |
| Parameter distribution per block | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | Per-block breakdown done. This lesson extends to total model parameter count. |
| Residual stream as backbone | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | The stream flows from embedding through N blocks to output. Already established at DEVELOPED. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Output projection (MENTIONED -> needs INTRODUCED) | Small | The student already knows nn.Linear, knows embedding maps V->d, and saw the one-line symmetry aside. This lesson adds 2-3 paragraphs showing the output projection: the final block's output (n, d_model) is multiplied by a weight matrix (d_model, V) to produce logits (n, V). Softmax produces probabilities. This is the reverse of embedding. Connection to nn.Linear: "just another nn.Linear(d_model, vocab_size)." Connection to MNIST output layer: "same idea, 10 classes -> 50K classes." Weight tying mentioned: many models share embedding and output projection weights. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Causal masking is a training trick that can be removed at inference" | The student knows regularization techniques (dropout) that behave differently at train vs eval time. Masking might seem like another regularization technique. Batch norm has train/eval distinction. | At inference time, future tokens literally do not exist yet. When generating token 6, tokens 7, 8, 9, ... have not been produced. Causal masking during training mirrors the inference reality: the model learns to predict the next token using only past context because that is all it will ever have during generation. Removing the mask at inference is not an option -- there is nothing to unmask. | Core explain section, immediately after introducing causal masking. This is the single most important misconception to address because it determines whether the student understands causal masking as fundamental or cosmetic. |
| "The model processes tokens one at a time (like during generation)" | The student has seen autoregressive generation, where tokens ARE produced sequentially. The causal masking discussion emphasizes "only see past tokens." Natural conclusion: the model handles one token at a time. | During training, the model processes the entire sequence simultaneously in one forward pass. Every position predicts its next token in parallel. Causal masking is what makes this possible: position 3 only attends to positions 1-3 even though positions 4, 5, 6 are right there in the same tensor. Without masking, parallel training with teacher forcing would leak future information. At inference, generation IS sequential (only because future tokens literally do not exist), but training is fully parallel. | After explaining causal masking mechanics. Explicitly contrast training (parallel, masked) vs inference (sequential, no future tokens to mask). |
| "Decoder-only means the model only decodes (can't understand input, only generates)" | The name "decoder" suggests output-only, like a translator that can speak but not listen. The original Transformer had a separate encoder for understanding and a decoder for generating. Without an encoder, the decoder-only model seems to lack the understanding component. | GPT models demonstrably "understand" their input -- they answer questions, summarize text, follow instructions. The input IS the context that is processed through all N blocks. The "decoder" name is historical: it refers to the use of causal (autoregressive) masking, not to a lack of comprehension ability. The original encoder-decoder split (separate stacks for understanding vs generating) turns out to be unnecessary -- a single causal stack handles both. | Encoder-decoder contrast section. Explicitly address that "decoder" is a naming convention from the original architecture, not a description of capability. |
| "The full attention matrix is computed and then masked (wasting compute on values that get thrown away)" | The student has seen the attention matrix computed as QK^T, then softmax applied. Adding masking after QK^T seems to waste compute on the upper triangle. | In practice, implementations are optimized. But conceptually, the masking happens to the scores before softmax, not to the final weights. Setting scores to -infinity before softmax makes those positions contribute exactly 0 after softmax. The mask is a fixed triangular matrix that does not depend on the input -- it can be precomputed once and applied cheaply. For this module, the conceptual picture (mask then softmax) is what matters; the compute optimization (flash attention, fused kernels) belongs to Module 4.3 (scaling-and-efficiency). | Brief aside after the masking mechanics section. Acknowledge the valid intuition but redirect: the mask is computationally cheap, and real implementations fuse the operations. |
| "GPT-2, GPT-3, and GPT-4 are fundamentally different architectures" | Each GPT version is discussed as a distinct, more powerful model. Marketing emphasizes leaps in capability. Natural assumption: the architecture must have changed dramatically. | GPT-2 (124M) and GPT-3 (175B) use the same decoder-only transformer architecture. The differences are primarily scale: more layers, wider d_model, more heads, more training data, longer training. GPT-2: 12 layers, d_model=768, 12 heads. GPT-3: 96 layers, d_model=12288, 96 heads. The architecture the student just learned IS GPT. What changes between versions is mainly the numbers, not the blueprint. | Architecture assembly section, after presenting GPT-2 and GPT-3 configurations. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| "The cat sat on the" with causal masking -- step-by-step attention computation for position 3 ("sat") | Positive | Shows exactly how causal masking works: "sat" can attend to "The", "cat", "sat" but NOT "on" or "the." The attention weight row has zeros for positions 4 and 5. Connects to the attention matrix the student already knows from Lesson 1. | Uses the same running example from all five prior lessons (4 tokens: "The", "cat", "sat", "here"). Extending it to 6 tokens and applying a mask makes the mechanism concrete and familiar. The student has computed attention weights for these tokens before -- now they see what changes with the mask. |
| Removing the causal mask during training (cheating at next-token prediction) | Negative | Disproves "masking is optional." Without the mask, position 3 can attend to position 4 (the answer it is trying to predict). The model achieves perfect training loss by copying the answer instead of learning to predict. At inference, the answer is not available -- the model has learned nothing useful. | The "cheating on a test" framing makes the failure mode intuitive. The student can immediately see why this is broken: training performance means nothing if the model fails at the actual task (generation). Connects to data leakage concepts from Series 1. |
| Full GPT-2 architecture assembly with parameter count | Positive | Assembles every piece the student has learned into one complete picture: embedding (38.4M) + PE + 12 transformer blocks (12 x ~7.1M = ~85.2M) + output projection. Shows the architecture end-to-end. Verifiable: adds up to ~124M (matching the known GPT-2 size). | GPT-2 has been the running concrete example throughout Module 4.2 (d_model=768, h=12, d_ff=3072). The student already has per-block parameter counts from Lesson 5. This lesson connects all the pieces and the total should match the ~124M figure from the Lesson 5 hook. Satisfying closure. |
| Encoder-decoder original Transformer (for contrast) | Negative (scope-bounding) | Shows what GPT is NOT. The original 2017 Transformer has an encoder stack (bidirectional attention) and a decoder stack (causal attention + cross-attention). GPT removes the encoder entirely and keeps only the decoder stack. This contrast makes "decoder-only" meaningful rather than abstract. | Brief, not deep. The student needs to know the name "decoder-only" has a reason. Also sets up Series 6 (Stable Diffusion uses U-Net with cross-attention, which is an encoder-decoder concept). BERT as an "encoder-only" model provides a second contrast point. |
| GPT-2 vs GPT-3 configuration comparison | Positive (stretch) | Shows that the architecture is the SAME, only the numbers change. Directly disproves the "fundamentally different architectures" misconception. Makes the student realize the architecture they just learned IS the architecture behind the most capable LLMs. | The comparison is a simple table: layers, d_model, heads, d_k, parameters. The student can verify that d_k = d_model / h in both cases. The scale difference (1400x more parameters) with the same blueprint is the key insight about why this architecture won. |

---

## Phase 3: Design

### Narrative Arc

The student has spent five lessons building the transformer block piece by piece -- from raw dot-product attention through Q/K/V, multi-head attention, and the full block with FFN and residual connections. The previous lesson ended with: "Stack N blocks. Add causal masking. That's GPT." This lesson delivers on that promise. But first, there is a gap that the student may not have noticed: every attention computation so far has been bidirectional. Every token could attend to every other token. For next-token prediction, this is cheating. If position 3 is trying to predict what comes at position 4, and the model can look at position 4, it just copies the answer. The motivating question is: how do we train a next-token predictor on entire sequences efficiently (in parallel) without letting the model see the answers? The answer is causal masking -- a triangular mask that prevents each position from attending to future positions. This is the one genuinely new mechanism. Once the student has it, the rest of the lesson is assembly and perspective: connecting the input pipeline (Module 4.1) through N masked transformer blocks to an output projection that produces next-token probabilities, counting parameters, and stepping back to see that this single architecture -- the decoder-only transformer -- is what powers GPT-2, GPT-3, and essentially all modern LLMs. The lesson closes with a brief encoder-decoder contrast that explains why the architecture is called "decoder-only" and plants a seed for cross-attention in Series 6.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual (masked attention matrix) | The 6x6 attention matrix for "The cat sat on the mat" with the upper triangle grayed out or crossed out. Each row shows which positions that token can attend to. Row 1 (position 1) has one valid entry. Row 6 (position 6) has all six valid. The triangular pattern is immediately visible. Before-and-after: full matrix (bidirectional) vs masked matrix (causal). | The causal mask IS a triangular matrix. The visual makes the constraint immediately obvious: everything above the diagonal is blocked. The student has seen attention matrices in the widget from Lesson 1 -- this is the same matrix with a mask applied. The before-and-after directly shows what changes. |
| Concrete example (worked masking computation) | Same running example tokens. Compute raw scores for one row (e.g., position 3: "sat"). Show scores for all 6 positions. Apply mask: set positions 4-6 to -infinity. Apply softmax: positions 4-6 become exactly 0, positions 1-3 renormalize. Show the weight vector before and after masking. | The student has computed attention weights by hand in Lessons 1-3. Adding the masking step is a small, concrete extension to a familiar procedure. Seeing -infinity become 0 after softmax is the key mechanical insight. The renormalization (remaining weights sum to 1) shows that masked positions truly contribute nothing. |
| Symbolic (formulas) | Causal mask formula: mask_{ij} = 0 if j > i, 1 if j <= i. Application: scores_{ij} = (QK^T)_{ij} / sqrt(d_k) + (mask_{ij} == 0 ? -infinity : 0). Or equivalently: set upper-triangle entries to -infinity before softmax. Full GPT formula chain: input = embed(tokens) + PE, for each block: x = block(x), logits = x @ W_output, probs = softmax(logits). | The masking operation is simple enough to state in one line. The formula connects to the attention formula the student already knows -- it is one additional step inserted before softmax. The full GPT chain from input to output probabilities shows the complete architecture in compact notation. |
| Analogy (exam with answer key) | "Imagine taking a test where the answer key is printed next to each question. You'd get a perfect score, but you'd learn nothing. Causal masking is the cardboard sleeve that covers the answers below the question you're currently working on. During the test (training), the sleeve ensures you can only see what came before. After the test (inference), there IS no answer key -- you are generating the answers yourself." | This analogy makes "why masking?" viscerally obvious. The student immediately grasps that seeing the answer is cheating, not helpful. The inference connection is natural: after the test, the answers do not exist yet. It also addresses misconception #1 (masking is a training trick) because the sleeve is not optional -- it mirrors reality. |
| Visual (full GPT architecture diagram) | End-to-end vertical diagram: token IDs at the bottom -> embedding lookup -> + positional encoding -> Block 1 -> Block 2 -> ... -> Block N -> Layer Norm -> output projection (d_model -> vocab) -> softmax -> probability distribution at the top. Color-coded consistently with the transformer block diagram from Lesson 5 (violet residual stream, sky blue blocks, amber output projection). Dimension annotations at each stage. GPT-2 concrete dimensions annotated. | This is the culminating visual of the entire module. Every piece the student has learned over 6 lessons (tokenization, embeddings, PE, attention, FFN, residual stream, layer norm, stacking) appears in one diagram. The student should look at this and recognize every component. It is the architectural payoff of the module. |
| Intuitive (why decoder-only won) | Brief comparison: encoder-decoder requires designing two separate stacks plus cross-attention. Decoder-only has one stack, one attention type, one training objective. Simpler to scale, simpler to implement, simpler to reason about. "The simplest architecture that works is the one that scales." | The "why did this win?" question builds strategic understanding beyond mechanics. The student should leave understanding that architectural simplicity is a feature, not a limitation -- it is what enabled scaling from 124M to 175B+ parameters. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 1 genuinely new (causal masking), plus assembly of known components
  1. Causal masking (genuinely new mechanism, but mechanically simple: set upper triangle to -infinity before softmax)
  2. Output projection layer (small gap resolution, not a new concept -- it is nn.Linear applied to produce vocabulary logits)
  3. The full architecture as an assembled whole (not a new concept but a new perspective -- seeing all pieces together)
- **Previous lesson's load:** BUILD (the-transformer-block -- assembling MHA, FFN, residual, layer norm)
- **Assessment:** CONSOLIDATE is appropriate. Only one genuinely new mechanism (causal masking), and it is mechanically simple. The rest is connecting pieces the student already has. This follows a BUILD lesson, maintaining the trajectory of decreasing novelty: STRETCH -> BUILD -> BUILD -> STRETCH -> BUILD -> CONSOLIDATE. The module ends with integration and perspective rather than new complexity.

### Connections to Prior Concepts

| Existing Concept | Connection | How |
|-----------------|------------|-----|
| Attention weight matrix (4.2.1) | Direct extension | "You've been computing the full attention matrix for five lessons. Now we modify it: set everything above the diagonal to -infinity before softmax. That's causal masking." |
| Autoregressive generation (4.1.1) | Motivation | "Remember the generation loop: predict the next token, append it, repeat. During generation, future tokens literally do not exist. Causal masking during training mirrors this constraint." |
| Next-token prediction (4.1.1) | Motivation | "The training task is P(x_t given x_1,...,x_{t-1}). The 'given everything before' is the key: the model should never see x_{t+1} when predicting x_t. Causal masking enforces this." |
| Token embedding + PE (4.1.3) | Assembly | "The input pipeline you learned in Module 4.1 feeds directly into the first transformer block. Token IDs -> embedding lookup -> add positional encoding -> into Block 1." |
| Transformer block (4.2.5) | Assembly | "Stack N of the blocks you just learned. Each block is identical: MHA + FFN + residual + layer norm. GPT-2 stacks 12. GPT-3 stacks 96." |
| Output layer / unembedding aside (4.1.3) | Completion | "Remember the aside about symmetry: embedding maps V -> d, output maps d -> V? Now we make that concrete: the final block's output is projected to vocabulary size to produce logits." |
| Softmax over vocabulary (4.1.1) | Assembly | "The output logits go through softmax to produce the probability distribution over tokens. Same as the MNIST output layer, same as the Lesson 1 probability bar chart -- just 50K classes instead of 10." |
| Data leakage (Series 1) | Parallel | "Causal masking prevents data leakage. You've seen this concept before: if the model sees the test labels during training, validation metrics are meaningless. Here, the 'label' for position 3 is the token at position 4. Looking at position 4 IS leaking the label." |
| Parameter counting per block (4.2.5) | Extension | "You computed per-block parameters in the last lesson. Now add it up: 12 blocks + embedding + output projection = the full GPT-2 model at ~124M." |
| "The simplest architecture that works" | Callback to architectural design philosophy | Resonates with the student's experience that simpler designs often win in practice. Connected to the observation that decoder-only scaled better than encoder-decoder despite seeming "less capable." |

### Analogies from Prior Lessons -- Risks

- **"Attention is a weighted average where the input determines the weights"** -- extends cleanly. Causal masking simply restricts which inputs participate in the average. No risk.
- **"The residual stream is a shared document"** -- extends cleanly. In a causal model, when sub-layer k writes to the document for token i, it only had access to tokens 1..i for the attention portion. No risk.
- **"Autoregressive generation is a feedback loop"** -- critical connection. The student needs to see that causal masking during training mirrors the sequential reality of generation. No risk.
- **"Attention reads, FFN writes"** -- no modification needed. Causal masking constrains what attention can read, but the FFN role is unchanged. No risk.

### Scope Boundaries

**This lesson IS about:**
- Causal masking: why it exists (autoregressive training), how it works (triangular mask, -infinity before softmax), where it operates (inside scaled dot-product attention)
- The full GPT architecture assembled end-to-end: embedding + PE -> N transformer blocks (with causal masking) -> output projection -> softmax
- Total parameter counting for GPT-2 (embedding + blocks + output, verifiable to ~124M)
- Why decoder-only won for LLMs (simplicity, scaling, one architecture does everything)
- Brief encoder-decoder contrast (original Transformer, BERT vs GPT) for naming context and future reference
- GPT-2 vs GPT-3 as same architecture at different scales

**This lesson is NOT about:**
- Implementing causal masking or the GPT architecture in PyTorch (Module 4.3, building-nanogpt)
- Training the model (pretraining, loss curves, learning rate scheduling) -- Module 4.3
- KV caching or efficient inference -- Module 4.3 (scaling-and-efficiency)
- Flash attention or fused attention kernels -- Module 4.3
- Cross-attention mechanics in detail -- Series 6 (Stable Diffusion)
- BERT architecture in detail -- out of scope, mentioned for contrast only
- The original Transformer paper in detail (positional FFN, label smoothing, etc.)
- Finetuning, instruction tuning, RLHF -- Module 4.4
- Mixture of experts, sparse attention, or other architectural variants
- Weight tying strategies in depth (mentioned briefly)

**Depth targets:**
- Causal masking: DEVELOPED (can explain why it exists, how it works mechanically, and why it cannot be removed at inference)
- Full GPT architecture end-to-end: DEVELOPED (can trace the complete data flow from text to probability distribution, name every component)
- Output projection: INTRODUCED (knows it maps d_model -> vocabulary, produces logits, is nn.Linear)
- Encoder-decoder vs decoder-only: INTRODUCED (knows the distinction, knows why it is called "decoder-only," knows decoder-only won for LLMs)
- BERT as encoder-only: MENTIONED (name-dropped for contrast, not explained)

### Lesson Outline

1. **Context + Constraints**
   - What: "This lesson assembles the complete GPT architecture -- every piece you've built across this module, plus one critical new mechanism."
   - NOT: implementing in code (Module 4.3), training (Module 4.3), finetuning (Module 4.4).
   - Callback to Lesson 5 seed: "Stack N blocks. Add causal masking. That's GPT."
   - This is the final lesson in Module 4.2. After this, the student knows the complete conceptual architecture.

2. **Hook (the cheating problem)**
   - Type: Puzzle / reveal
   - Setup: "You've been training a language model. The task is next-token prediction: given tokens 1 through t, predict token t+1. Here's the attention matrix from the model. Look at row 3." Show a full (unmasked) attention matrix for "The cat sat on the mat." Row 3 ("sat") has high attention weight on position 4 ("on") -- the very token it is trying to predict.
   - The question: "Is this model actually learning to predict the next token?"
   - Reveal: No. It is cheating. Position 3 can look at position 4, which IS the answer. The model achieves low training loss by copying, not predicting. At inference, position 4 does not exist when generating position 3. The model has learned nothing useful.
   - "How do we train on full sequences efficiently while preventing this cheating?"
   - Why this hook: Creates the need for causal masking through a concrete failure. The student sees the problem before the solution. The "cheating" framing connects to data leakage from Series 1.

3. **Explain: Causal Masking (core new concept)**
   - The constraint: position i can only attend to positions 1 through i. Never to positions i+1 and beyond.
   - **Exam analogy:** answer key printed next to questions, cardboard sleeve covering future answers.
   - Mechanism: before applying softmax to the attention scores, set all entries where j > i to negative infinity. After softmax, these positions become exactly 0 (e^{-inf} = 0).
   - **Visual: masked attention matrix.** 6x6 matrix for "The cat sat on the mat." Upper triangle grayed out. Each row shows increasing context: row 1 sees 1 token, row 2 sees 2, row 6 sees all 6. The triangular shape is the defining visual.
   - **Worked example:** Row 3 ("sat"). Raw scores: [s_31, s_32, s_33, s_34, s_35, s_36]. After masking: [s_31, s_32, s_33, -inf, -inf, -inf]. After softmax: [w_31, w_32, w_33, 0, 0, 0] where w_31 + w_32 + w_33 = 1. Renormalization is automatic -- softmax naturally redistributes weight to the unmasked positions.
   - **Key formula:** Same scaled dot-product attention, one additional step: scores = QK^T / sqrt(d_k), masked_scores = scores + mask (where mask has 0 for allowed, -infinity for blocked), weights = softmax(masked_scores).
   - Connection to autoregressive generation: "During generation, future tokens do not exist. Causal masking during training simulates this. The model practices under the same constraint it will face at inference."
   - **Misconception #1 addressed directly:** "Causal masking is not a training trick that gets removed at test time. At test time, there IS no future to mask -- the model is generating token by token. The mask during training ensures the model learns to predict without information it will never have."
   - **Misconception #2 addressed:** "During training, the entire sequence is processed in one forward pass. All positions predict their next token simultaneously. This is efficient because we get N training examples from one sequence. Causal masking is what makes parallel training safe -- each position only sees its own past, even though the full sequence is in the tensor."
   - Comprehension check: "Position 5 in a 10-token sequence. How many tokens can it attend to? What happens to positions 6-10 in the attention computation?"

4. **Check 1: The mask in action**
   - Predict-and-verify: "Draw the causal mask for a 4-token sequence. Which entries are -infinity? After softmax, what is the sum of weights in row 1? In row 4?"
   - Expected: 4x4 lower-triangular matrix. Row 1 sums to 1 (only one entry, which must be 1.0). Row 4 sums to 1 (four entries, flexible distribution). Upper triangle all -infinity / all zeros after softmax.

5. **Explain: The Output Projection (gap resolution)**
   - Recall the symmetry aside from Module 4.1: embedding maps vocab -> d_model, output projection maps d_model -> vocab.
   - The last transformer block outputs (n, d_model). To predict the next token, we need a probability distribution over the vocabulary.
   - Output projection: nn.Linear(d_model, vocab_size). Produces (n, vocab_size) logits.
   - Softmax converts logits to probabilities: P(next token | context) for each position.
   - Connection to MNIST: "Same idea as the MNIST output layer. Instead of 10 digit classes, 50K token classes."
   - Weight tying (brief): many models share the embedding weight matrix with the output projection (transposed). If embedding is (V, d) then output is (d, V). The model learns a single mapping between token space and embedding space. GPT-2 uses weight tying.
   - Depth: INTRODUCED. The student knows what it does, not how to implement or train it.

6. **Explain: Assembling the Full Architecture**
   - Walk through the complete forward pass, top to bottom:
     1. Tokenize input text (BPE from Module 4.1)
     2. Look up token embeddings: nn.Embedding(vocab_size, d_model)
     3. Add positional encoding: embedding + PE
     4. Pass through N transformer blocks, each with causal masking in the attention
     5. Apply final layer norm (pre-norm convention: one extra LN after the last block)
     6. Output projection: nn.Linear(d_model, vocab_size)
     7. Softmax: probability distribution over next token
   - **Full architecture diagram (visual centerpiece):** Vertical flow from tokens at bottom to probabilities at top. Every component labeled with GPT-2 dimensions. Color-coded consistently with Lesson 5's block diagram.
   - GPT-2 configuration box: vocab_size=50257, d_model=768, n_layers=12, n_heads=12, d_ff=3072, context_length=1024.
   - **Parameter counting exercise (closure):**
     - Token embeddings: 50257 x 768 = 38,597,376
     - Position embeddings: 1024 x 768 = 786,432
     - Per block attention (Q+K+V+O): 4 x 768^2 = 2,359,296
     - Per block FFN: 2 x 768 x 3072 = 4,718,592
     - Per block layer norm (x2): 2 x 2 x 768 = 3,072
     - 12 blocks total: 12 x (2,359,296 + 4,718,592 + 3,072) = 84,971,520
     - Final layer norm: 2 x 768 = 1,536
     - Output projection: shared with token embeddings (weight tying), so 0 additional
     - Total: ~124.4M -- matches the known GPT-2 124M parameter count
   - The callback to Lesson 5's hook: "In Lesson 5, you learned that ~2/3 of per-block parameters are in FFN. Now you see the full model: embeddings (~31%), attention blocks (~23%), FFN blocks (~46%). The distribution holds."

7. **Check 2: Architecture comprehension**
   - "A colleague says they want to make GPT-2 better by doubling d_model from 768 to 1536 while keeping everything else the same. How does this affect: (a) per-block parameters? (b) total parameters? (c) context length?"
   - Expected: (a) Attention params quadruple (4 x d_model^2). FFN params also scale (2 x d_model x d_ff, but d_ff also typically scales as 4x d_model, so it would also quadruple). (b) Roughly 4x total model parameters. (c) Context length is unchanged -- it's a separate hyperparameter (1024) controlled by positional encoding, not d_model.

8. **Elaborate: Why Decoder-Only Won**
   - Brief history: the original 2017 Transformer had an encoder (bidirectional attention, processes input) and a decoder (causal attention + cross-attention from encoder, generates output). Designed for machine translation.
   - Encoder-only variant: BERT. Bidirectional attention (every token sees every token). Great for understanding tasks (classification, NER). Cannot generate text autoregressively.
   - Decoder-only variant: GPT. Causal attention only. Can generate text AND "understand" its input (the input IS the context processed through all N blocks).
   - Why decoder-only scaled better:
     - Simplicity: one stack, one attention type, one training objective (next-token prediction)
     - Scaling: next-token prediction on vast text corpora is a simple, universal training signal that scales with data and compute
     - Generality: one model handles generation AND understanding -- no need for separate architectures per task
     - The scaling laws (Kaplan et al., Chinchilla) showed that decoder-only models improve predictably with scale. Simplicity enabled the massive scaling that produced GPT-3 and beyond.
   - **Misconception #3 addressed:** "'Decoder' does not mean 'can only decode.' It means 'uses causal masking.' GPT models understand their input -- they process it through all N blocks with the same attention + FFN machinery. The input IS the context. The name is historical, inherited from the original encoder-decoder Transformer."
   - **Misconception #5 addressed:** GPT-2 vs GPT-3 configuration table. Same architecture. Different numbers. 12 layers vs 96 layers. 768 vs 12288 d_model. 124M vs 175B parameters. "The architecture you just learned IS the architecture behind GPT-3. What changed is the scale, not the blueprint."
   - This section is brief (INTRODUCED depth). The student should know the landscape, not deeply understand encoder-decoder mechanics. Cross-attention details are deferred to Series 6.

9. **Elaborate: Misconception #4 (brief aside on efficiency)**
   - Brief aside: "You might wonder whether computing the full QK^T matrix and then zeroing out the upper triangle wastes compute. Valid concern. In practice, implementations can avoid computing the masked entries entirely (flash attention, fused kernels). For understanding the architecture, the conceptual picture -- compute scores, mask, softmax -- is correct and complete. The compute optimization is a Module 4.3 topic."
   - Depth: MENTIONED. The student knows the concern exists and that solutions exist, but does not need to understand flash attention here.

10. **Summarize**
    - Causal masking: set future positions to -infinity before softmax. Each position can only attend to past and present.
    - Why: the model learns to predict without future information, matching the inference constraint where future tokens do not exist.
    - The full GPT architecture: token embedding + positional encoding -> N transformer blocks (with causal masking) -> layer norm -> output projection -> softmax.
    - GPT-2 at ~124M parameters. GPT-3 at ~175B. Same architecture. Different scale.
    - "Decoder-only" means causal masking, not "can only decode." The simplicity of one stack, one objective, one attention type is what enabled scaling.
    - **Module completion echo:** "Over six lessons, you built the entire architecture behind modern LLMs. Lesson 1: raw attention (feel the limitation). Lesson 2: Q and K (fix the matching problem). Lesson 3: V and residual stream (fix the contribution problem). Lesson 4: multi-head attention (capture diverse relationships). Lesson 5: the transformer block (assemble the repeating unit). This lesson: causal masking and the complete architecture. Every piece exists because the previous version was insufficient."

11. **Next Step (Module 4.3 seed)**
    - "You understand the complete architecture. In Module 4.3, you will build it. nanoGPT: assemble every component in PyTorch, train it on real text, watch the loss drop and the generated text improve from random noise to coherent English. Then you'll load GPT-2's actual pretrained weights into your architecture and generate text with a real model."
    - "The architecture is the blueprint. Next, you build the house."

### Widget

No interactive widget for this lesson. The lesson is conceptual and the primary visual artifacts are:
1. **Masked attention matrix diagram** (inline SVG) -- 6x6 matrix with upper triangle grayed out, dimension annotations, before-and-after comparison showing full vs causal attention.
2. **Full GPT architecture diagram** (inline SVG) -- end-to-end vertical flow from token IDs to probability distribution. Color-coded consistently with Lesson 5's block diagram (violet residual stream, sky blue blocks, amber output/FFN, emerald layer norm). GPT-2 dimensions annotated.

Both should be static SVGs, not interactive widgets. The lesson's value is in seeing the complete picture and understanding the causal masking constraint, not in parameter manipulation.

### Concept Depths After This Lesson

| Concept | Depth | Notes |
|---------|-------|-------|
| Causal masking | DEVELOPED | Can explain why it exists (training matches inference), how it works (-infinity before softmax), where it goes (inside attention), and why it cannot be removed. |
| Full GPT architecture (end-to-end) | DEVELOPED | Can trace the complete forward pass from text input to probability output. Can name every component and state its role. |
| Output projection (d_model -> vocab logits) | INTRODUCED | Knows it maps final hidden states to vocabulary logits. Knows weight tying exists. Does not need to implement. |
| Encoder-decoder vs decoder-only distinction | INTRODUCED | Knows decoder-only uses causal masking, encoder-decoder has two stacks + cross-attention. Knows decoder-only won for LLMs due to simplicity and scaling. |
| BERT as encoder-only | MENTIONED | Name-dropped for contrast. Bidirectional attention. Cannot generate autoregressively. |
| GPT parameter counting | DEVELOPED | Can break down total model parameters by component. Verified GPT-2 at ~124M. |
| Why decoder-only won | INTRODUCED | Simplicity, scaling, generality. Not deeply analyzed but the student can articulate the key arguments. |

---

## Review -- 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings -- the lesson is pedagogically sound, well-structured, and faithfully executes the plan. The student would learn what they need to learn. However, two improvement findings would meaningfully strengthen the lesson if addressed.

### Findings

#### [IMPROVEMENT] -- "Teacher forcing" used without introduction

**Location:** Section 7 (Misconception #2: Parallel training), line 972
**Issue:** The sentence "Without the mask, parallel training with teacher forcing would leak future information" uses the term "teacher forcing" which has never been introduced in any student-facing lesson. It appeared in the planning notes for what-is-a-language-model but was never taught. The student has no concept at MENTIONED or higher for this term.
**Student impact:** The student hits an unfamiliar term in a parenthetical that they cannot look up from prior lessons. This breaks the flow of an otherwise clear explanation. The student might think they missed something, or might form an incorrect understanding of what "teacher forcing" means.
**Suggested fix:** Either (a) remove the term entirely -- the sentence works fine as "Without the mask, parallel training would leak future information" since the preceding paragraphs already explain WHY it leaks, or (b) briefly define it inline: "Without the mask, parallel training (where the model sees the full ground-truth sequence during training) would leak future information." Option (a) is simpler and more appropriate for a CONSOLIDATE lesson.

#### [IMPROVEMENT] -- Check 2 answer contradicts its own premise

**Location:** Section 12 (Check 2: Architecture Comprehension), lines 1309-1349
**Issue:** The question states "doubling d_model from 768 to 1536 while keeping everything else the same." The answer for part (a) says "FFN parameters also quadruple if d_ff scales as 4 x d_model." But the question premise says "keeping everything else the same," which means d_ff stays at 3072. Under that premise, FFN parameters would double (2 x 1536 x 3072 = 2 x original), not quadruple. The conditional "if" in the answer creates ambiguity -- the student who carefully reads the question would get a different answer than the one provided.
**Student impact:** A careful student would compute FFN params as 2 x 1536 x 3072 = 9,437,184 (2x the original), see the answer says "quadruple," and feel confused about whether they got it wrong. This undermines confidence at a point where the lesson should be building it (this is a CONSOLIDATE lesson).
**Suggested fix:** Either (a) change the question to "doubling d_model and scaling d_ff proportionally" and then the 4x answer is correct, or (b) keep the question as-is and fix the answer: "Attention parameters quadruple. FFN parameters double (d_ff stays at 3072, but d_model doubled). Note: in practice, d_ff is usually set to 4 x d_model, which would make FFN also quadruple -- but the question says 'everything else the same.'" Option (b) is pedagogically richer because it teaches the student about the convention while respecting the literal question.

#### [POLISH] -- Parameter counting omits bias terms

**Location:** Section 11 (Counting Every Parameter), lines 1228-1260
**Issue:** The attention count `4 x 768^2 = 2,359,296` and FFN count `2 x 768 x 3072 = 4,718,592` omit bias terms. GPT-2 uses biases in Q/K/V/O projections and FFN layers. The total bias omission across 12 blocks is approximately 83K parameters -- a negligible fraction of 124M. The final total still matches the known GPT-2 figure.
**Student impact:** Minimal. The student gets the right mental model (parameter distribution, total magnitude) and the approximation is close enough. A student who later implements GPT-2 in Module 4.3 would discover the biases naturally.
**Suggested fix:** No fix needed for the numbers, but could optionally add a small text note below the total: "These counts omit bias terms (~83K total), which is why the number is slightly approximate." This is entirely optional -- the lesson already says "~124.4M" rather than claiming exact precision.

#### [POLISH] -- Row numbering in hook uses 1-indexed language but causal mask explanation could be clearer about indexing

**Location:** Sections 3-4 (The Cheating Problem and Causal Masking)
**Issue:** The hook says "position 3 ('sat')" and "position 4 ('on')" for the sequence "The cat sat on the mat." This uses 1-based indexing. The causal mask formula uses j <= i (consistent with 1-based). The CausalMaskDiagram labels rows/columns with token names rather than indices, which avoids the issue. However, the Insight aside says "Row i has exactly i non-zero entries" -- this only holds for 1-based indexing. The rest of the lesson is consistent, but a student used to 0-based Python indexing might momentarily wonder. Since the lesson is conceptual (not implementing in code), this is fine as-is.
**Student impact:** Negligible. The token names in the diagram ground everything concretely, and the formulas use mathematical convention (1-based). Any confusion would resolve immediately in Module 4.3 when implementing.
**Suggested fix:** None needed. The lesson is consistent within its own conventions. If anything, one could add "(1-based indexing)" after the first use of "position 3" in the hook, but this is unnecessary given the concrete token names.

### Review Notes

**What works well:**
- The cheating problem hook is the strongest motivational opening in the module. It creates genuine need for causal masking through a concrete, visceral failure mode.
- The exam/cardboard sleeve analogy is excellent and directly addresses the most important misconception (masking as a training trick).
- The CausalMaskDiagram with the "leak!" annotation is a clever visual that ties the hook to the mechanism.
- The GptArchitectureDiagram is well-designed: color-coded consistently with Lesson 5, dimension annotations at every stage, weight tying annotation with the dashed line connecting embedding and output projection. This is the culminating visual of the module and it delivers.
- The module completion echo and the 6-lesson narrative recap are effective closure. The student can trace the entire arc: limitation -> fix -> limitation -> fix -> assembly -> complete architecture.
- Cognitive load is appropriate for a CONSOLIDATE lesson: one genuinely new mechanism, everything else is assembly and perspective.
- All five planned misconceptions are addressed at appropriate locations in the lesson.
- The parameter counting section provides satisfying closure by connecting back to Lesson 5's per-block counts and arriving at the known GPT-2 figure.
- Scope boundaries are rigorously maintained -- the lesson does not stray into implementation, training, or optimization topics.

**Pattern observation:** The two improvement findings are both about precision in technical details -- an undefined term and an inconsistent check question. The lesson's pedagogical structure, narrative arc, and conceptual coverage are all strong. This suggests the builder was focused on the right things (motivation, flow, visual design, misconception handling) and the technical details are the area that benefits from a review pass.

---

## Review -- 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All Iteration 1 findings have been addressed correctly. No new critical or improvement findings. The lesson is ready to ship.

### Iteration 1 Fix Verification

| Finding | Status | Verification |
|---------|--------|-------------|
| [IMPROVEMENT] "Teacher forcing" used without introduction | FIXED | The term has been completely removed. Line 971-972 now reads "Without the mask, parallel training would leak future information." Clean and clear. |
| [IMPROVEMENT] Check 2 answer contradicts its own premise | FIXED | The answer now correctly states FFN parameters **double** (not quadruple) under the literal "everything else the same" premise, then notes the convention that d_ff usually scales with d_model. Both the literal answer and the practical convention are taught. Pedagogically richer than before. |
| [POLISH] Parameter counting omits bias terms | FIXED | A note at line 1280 now reads "These counts omit bias terms, which add ~83K -- negligible at this scale." Clear, honest, and does not distract. |
| [POLISH] Row numbering / 1-based indexing | NO FIX NEEDED | Correctly left as-is. The lesson uses 1-based indexing consistently with mathematical convention, and the diagram uses token names to ground everything concretely. |

### Findings

#### [POLISH] -- CausalMaskDiagram has no explicit dimension annotation or axis labels

**Location:** CausalMaskDiagram component (lines 60-225), rendered in Section 4
**Issue:** The two matrices show token labels on rows and columns, which is good. However, unlike the GptArchitectureDiagram (which has dimension annotations at every stage), the causal mask diagram does not label the axes with "Query token (row)" and "Key token (column)." The student has seen attention matrices before and knows that rows correspond to the token doing the attending and columns correspond to the token being attended to, so this is not confusing -- but adding axis labels would reinforce the Q/K relationship one more time and make the diagram fully self-explanatory.
**Student impact:** Minimal. The student already has strong familiarity with attention matrix conventions from five prior lessons. The token labels on rows and columns are sufficient for comprehension. This is a polish item, not a learning obstacle.
**Suggested fix:** Optionally add small axis labels ("query" on the left, "key" on top) to the CausalMaskDiagram. This is entirely optional and the lesson works well without it.

### Review Notes

**Iteration 1 fixes were applied cleanly.** The "teacher forcing" removal was the cleanest option (option a) and the sentence reads naturally without it. The Check 2 fix chose the pedagogically richer option (option b) and the answer now teaches both the literal answer and the practical convention -- better than the original. The bias terms note is appropriately brief.

**What works exceptionally well in the final version:**

- **The cheating problem hook** remains the strongest motivational opening in the module. The "leak!" annotation on the CausalMaskDiagram directly connects the hook to the mechanism.
- **The exam/cardboard sleeve analogy** is intuitive and directly addresses the most important misconception. The analogy naturally extends to inference ("after the test, there IS no answer key") which prevents the student from forming the "masking is a training trick" misconception.
- **The narrative arc** is complete and satisfying. Problem (cheating) -> solution (causal masking) -> verification (worked example + check) -> gap fill (output projection) -> assembly (full architecture) -> closure (parameter counting + module echo) -> perspective (decoder-only context) -> forward reference (Module 4.3).
- **Cognitive load** is perfectly calibrated for a CONSOLIDATE lesson. One new mechanism (causal masking), one small gap resolution (output projection), everything else is assembly and perspective.
- **All five misconceptions** are addressed at appropriate locations with concrete corrections.
- **The GptArchitectureDiagram** with weight tying annotation is the culminating visual of the module.
- **The Check 2 answer** is now pedagogically stronger after the fix -- it teaches the student about the d_ff convention while respecting the literal question.
- **The module completion echo** provides effective closure by tracing the 6-lesson arc of limitation -> fix -> limitation -> fix -> assembly -> architecture.
- **Scope boundaries** are rigorously maintained throughout.

**The lesson passes review and is ready for the Record phase.**

---

## What Was Actually Built

The implementation follows the planning document faithfully. No significant deviations from the design.

**Matched the plan:**
- The cheating problem hook with the "leak!" annotation on the causal mask SVG
- Exam/cardboard sleeve analogy for causal masking
- Worked example for row 3 ("sat") with concrete numbers
- All five misconceptions addressed at their planned locations
- CausalMaskDiagram: before-and-after comparison (full vs causal attention) as inline SVG
- GptArchitectureDiagram: full end-to-end vertical flow with color-coded components, dimension annotations, weight tying annotation
- Output projection gap resolution (MENTIONED -> INTRODUCED)
- GPT-2 parameter counting with verified ~124.4M total
- Encoder-decoder vs decoder-only contrast with three GradientCards (encoder-only, encoder-decoder, decoder-only)
- GPT-2 vs GPT-3 comparison table
- Module completion echo tracing the 6-lesson arc
- ModuleCompleteBlock with all six achievements listed
- Module 4.3 seed ("The architecture is the blueprint. Next, you build the house.")

**Minor differences from plan:**
- The "teacher forcing" term was removed per review finding (used in planning notes but never taught to student)
- Check 2 answer was refined per review finding to correctly state FFN params double (not quadruple) under "everything else the same" premise, then notes the d_ff convention
- Bias term note added to parameter counting section (~83K omitted, negligible at this scale)
- No interactive widget (as planned) -- both diagrams are static inline SVGs
