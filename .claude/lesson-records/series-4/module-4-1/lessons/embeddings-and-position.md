# Lesson Plan: Embeddings & Positional Encoding

**Module:** 4.1 — Language Modeling Fundamentals
**Position:** Lesson 3 of 3 (final)
**Slug:** embeddings-and-position
**Type:** Hands-on (notebook: `4-1-3-embeddings-and-position.ipynb`)

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Relevance |
|---------|-------|--------|-----------|
| Tokenization produces integer IDs from text (BPE) | APPLIED | tokenization (4.1.2) | Direct prerequisite. The student built BPE from scratch and knows tokens are integer indices into a vocabulary. This lesson starts exactly where that left off: "you have integers, now what?" |
| Vocabulary as a finite set of token types (~50K) | DEVELOPED | tokenization (4.1.2) | The vocabulary size determines the number of rows in the embedding matrix. Student knows vocab sizes for GPT-2 (50K), LLaMA (32K), etc. |
| Language modeling as next-token prediction | DEVELOPED | what-is-a-language-model (4.1.1) | The task embeddings serve. Student knows the model takes context and predicts a distribution over vocabulary. Embeddings are the first transformation in that pipeline. |
| Softmax over vocabulary to produce probability distribution | INTRODUCED | what-is-a-language-model (4.1.1) | Connected via MNIST softmax (10 classes -> 50K classes). Relevant because the output layer mirrors the embedding dimension. |
| nn.Linear as w*x + b (weight matrix times input plus bias) | DEVELOPED | nn-module (2.1.3) | Direct connection: an embedding lookup is mathematically equivalent to multiplying a one-hot vector by a weight matrix. Student can verify this equivalence. |
| Matrix multiplication with @ operator | DEVELOPED | tensors (2.1.1) | Student needs this to see that one-hot @ weight_matrix = row selection. The embedding "lookup" IS matrix multiplication with a shortcut. |
| Parameters as learnable knobs updated by backprop | DEVELOPED | gradient-descent (1.1.4) | Embeddings are learnable parameters, not fixed preprocessing. They update during training just like weights in nn.Linear. |
| Convolution exploits spatial locality (filters slide over local neighborhoods) | DEVELOPED | what-convolutions-compute (3.1.1) | Critical contrast: CNNs handle position implicitly through sliding filters and weight sharing. Transformers have no such mechanism — motivates why positional encoding is needed. |
| Weight sharing in convolutions (same filter at every position) | INTRODUCED | what-convolutions-compute (3.1.1) | The CNN's spatial inductive bias. The transformer lacks this — it processes all positions identically, with no built-in notion of "nearby." |
| Sinusoidal functions (sin, cos) | Assumed | Math background | Student is a software engineer comfortable with basic trigonometry. Not taught in the course, but safe to assume recognition. |

### Established Mental Models and Analogies

- **"A language model approximates P(next token | context)"** — the task this input pipeline feeds
- **"Tokenization defines what the model can see"** — tokens are atomic units; embeddings give them meaning
- **"nn.Linear IS w*x + b"** — makes the one-hot-times-matrix equivalence followable
- **"Architecture encodes assumptions about data"** — extended from CNNs to tokenization; now extends again to embeddings and position
- **"MNIST softmax (10 classes) -> LM softmax (50K classes)"** — vocabulary size scaling
- **"Convolution is a sliding pattern detector"** — the spatial mechanism transformers lack

### What Was Explicitly NOT Covered in Prior Lessons

- The word "embedding" was carefully avoided in the tokenization lesson. SolidGoldMagikarp was rewritten to use "learned representation" / "internal representation." Vocabulary size discussion replaced "embedding matrix" with "more entries = more parameters." The student has NOT encountered embeddings at all.
- One-hot encoding was NOT explicitly taught. The student knows about feature vectors from Series 1 (house prices, pixel values) and integer token IDs from tokenization. One-hot will be introduced here as the naive approach.
- Positional encoding — entirely new concept. The closest foundation is the CNN spatial locality concept.
- How the model processes token vectors internally (attention, etc.) — Module 4.2.

### Readiness Assessment

The student is well-prepared. They have a clear gap to fill: "I have integer token IDs but neural networks multiply by weight matrices — how do integers become vectors?" They know what the tokens represent (from tokenization), what task the model performs (from language modeling), and how weight matrices work (from nn.Linear). The main genuinely new idea is positional encoding, which is well-motivated by contrast with CNNs.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how token IDs are transformed into rich vector representations via learned embedding lookup and how positional encoding injects sequence order information that the model would otherwise lack.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Tokenization produces integer IDs | DEVELOPED | APPLIED | tokenization (4.1.2) | OK | Student must understand that we start with integers. They built BPE from scratch — exceeds requirement. |
| Vocabulary as a finite set of token types | INTRODUCED | DEVELOPED | tokenization (4.1.2) | OK | Need to know vocab size determines embedding table rows. Student knows vocab sizes for major models. |
| nn.Linear weight matrix (w*x + b) | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | Embedding = matrix row selection. Student verified nn.Linear IS w*x+b. |
| Matrix multiplication (@) | DEVELOPED | DEVELOPED | tensors (2.1.1) | OK | Need to show one-hot @ matrix = row selection. Student has used @ extensively. |
| Parameters are learnable (updated by backprop) | DEVELOPED | DEVELOPED | gradient-descent (1.1.4) | OK | Embeddings are learned, not fixed. This is a core mental model. |
| CNN spatial locality (filters slide over neighborhoods) | INTRODUCED | DEVELOPED | what-convolutions-compute (3.1.1) | OK | Contrast: CNNs handle position implicitly, transformers don't. Student has strong spatial intuition. |
| One-hot encoding | INTRODUCED | MISSING | — | MISSING | Needed as the bridge from integer to vector, and to motivate dense embeddings. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| One-hot encoding | Small (student knows feature vectors and integer indexing, just hasn't seen this specific representation) | Brief section (2-3 paragraphs + visual) introducing one-hot as "the naive approach." The student knows vectors and knows token IDs are integers 0..V-1. One-hot is just "put a 1 at the token's position." Show it, show why it's wasteful (50K-dimensional sparse vector), show it doesn't capture similarity — then immediately use it to derive embeddings as the better alternative. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Embeddings are a preprocessing step, like normalization — you compute them once and they're fixed" | Tokenization IS a fixed preprocessing step (deterministic once the merge table is trained). Student may assume embeddings work the same way. | Show that embedding vectors change during training: the embedding for token "the" at initialization is random, but after training it's near other determiners ("a", "an"). If embeddings were fixed, this clustering couldn't emerge. Alternatively: show that nn.Embedding has requires_grad=True — it's in model.parameters(). | During the "embeddings are learned" section, immediately after showing nn.Embedding. Address head-on with a WarningBlock. |
| "Similar tokens should have similar integer IDs" (i.e., the integer itself carries meaning) | Token ID 464 for "the" and 465 for "." seem arbitrary. Maybe the student thinks they should be organized somehow. In the BPE lesson, earlier merges produce lower IDs — maybe that ordering matters? | The integer assignment is arbitrary. "cat" might be token 2364 and "dog" might be token 8976 — their IDs are determined by BPE merge order, not semantic meaning. The embedding layer's job is precisely to learn the meaningful representation that the integer ID doesn't encode. | In the one-hot section, when motivating why one-hot is insufficient: IDs are arbitrary indices, not meaningful coordinates. |
| "One-hot encoding is fine for small vocabularies, so embeddings are just an efficiency trick for large vocabularies" | With 10 classes (MNIST), one-hot works. With 50K tokens, it seems like embeddings are just compression for performance. | Embeddings capture semantic relationships (king-queen, cat-dog). One-hot vectors are all equally distant from each other — one_hot("cat") is exactly as far from one_hot("dog") as from one_hot("the"). Dimensionality is one problem, but the deeper issue is that one-hot cannot represent similarity. A 10-word vocabulary would still benefit from embeddings. | In the "why one-hot fails" section. Show the equal-distance property explicitly, then contrast with embedding space where similar tokens cluster. |
| "Word order doesn't matter much — the model mostly looks at which words are present (bag-of-words)" | Bag-of-words models are historically successful (spam filters, sentiment analysis). The student's intuition from Series 3 is that spatial structure matters for images but may not transfer to language. | "Dog bites man" vs "Man bites dog" contain the exact same tokens with the exact same embeddings. Without positional encoding, the model sees IDENTICAL inputs for both sentences. A set of embeddings, without position, is literally a bag of words. | As the hook for the positional encoding section. This is the core motivating example. |
| "Positional encoding is like adding a channel dimension — the model gets position as an extra feature" | In CNNs, you can add coordinate channels (CoordConv). The student might think positional encoding works similarly — extra information appended. | Positional encoding is ADDED to the token embedding, not concatenated. This is important: the final vector has the same dimensionality as the embedding alone. Position and meaning are blended into a single representation, not kept as separate features. (Concatenation would double the dimension and require different downstream architecture.) | When showing the addition operation: token_embedding + positional_encoding. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| One-hot vector for "cat" (token ID 2364 in a 50K vocabulary) → 50,000-dimensional vector with a single 1 | Positive | Show one-hot encoding concretely. The 50K dimension makes the waste visceral. | Uses a real-ish token ID from BPE. The student immediately feels the absurdity of a 50K-sparse vector. |
| One-hot @ embedding_matrix = row selection (worked with 3 tokens, 4-dim embeddings) | Positive | The critical mathematical insight: one-hot times a matrix just selects a row. This IS what nn.Embedding does, but without actually creating the sparse vector. | Tiny enough to compute by hand (3 tokens, 4 dimensions). Student can verify that multiplying a one-hot vector by a matrix gives back exactly one row. This makes embedding lookup feel like a mathematical operation, not magic. |
| "Dog bites man" vs "Man bites dog" — same embeddings, no positional encoding | Negative | Shows that embeddings without position produce a bag of words. The model literally cannot distinguish these two sentences. | Classic linguistics example. Both sentences have the same tokens. The meaning is entirely determined by order. If the model can't see order, it can't do language. |
| Sinusoidal positional encoding heatmap for a 20-token sequence, 64-dim | Positive | Shows the visual pattern: each position gets a unique pattern, nearby positions have similar patterns, the encoding is smooth and periodic. | Small enough to visualize but large enough to show the multi-frequency structure. The heatmap makes the abstract formula tangible. |
| "cat" embedding at initialization vs after training (random noise vs clustered with "dog", "bird") | Positive (stretch) | Makes the "learned" nature of embeddings visceral. Training transforms random vectors into a meaningful space. | Connects to the "parameters are knobs" mental model. The student sees that backprop shapes the embedding space — it's the same training process they know, applied to a lookup table. |

### Gate Check (Phase 2)
- [x] Every prerequisite has a status with reasoning
- [x] One-hot encoding gap has explicit resolution plan
- [x] 4 misconceptions identified, each with negative example
- [x] 5 examples with stated rationale (3 positive, 1 negative, 1 stretch)

---

## Phase 3: Design

### Narrative Arc

The student has just built a BPE tokenizer from scratch. They can convert "The cat sat on the mat" into a sequence of integers like [464, 3797, 3332, 319, 262, 2603]. But here's the problem: a neural network multiplies inputs by weight matrices and passes them through activation functions. You can't multiply an integer by a weight matrix in any meaningful way. Token 464 and token 465 aren't "close" to each other — the numbers are arbitrary indices assigned by BPE merge order. So before the model can do anything with these tokens, it needs to convert each integer ID into a vector that the network can actually compute with. The naive approach — one-hot encoding — creates a 50,000-dimensional vector with a single 1. That's wasteful (most dimensions are zero) and uninformative (every token is equidistant from every other token). Embeddings solve both problems: they're a learned lookup table that maps each token ID to a dense, low-dimensional vector where similar tokens end up near each other. But there's a second problem. Once you have embedding vectors for each token, you've essentially created a bag of words — the set {"dog", "bites", "man"} is the same regardless of order. Unlike CNNs, where convolution filters naturally handle spatial locality by sliding over neighboring pixels, a transformer processes all positions identically. If you want the model to know that "dog" is at position 0 and "man" is at position 2, you have to explicitly tell it. That's positional encoding. By the end of this lesson, the student can trace the complete input pipeline: text -> BPE tokens -> integer IDs -> embedding vectors + positional encoding -> the tensor that enters the transformer.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example (worked)** | One-hot vector times embedding matrix for 3 tokens with 4-dim embeddings. Student traces the matrix multiplication by hand and sees that the result is row selection. | Embeddings feel like magic ("the model just knows what words mean"). Showing the actual math — one-hot times matrix = select a row — strips the mystery. The student already knows matrix multiplication from tensors (2.1). |
| **Visual (widget)** | Embedding space explorer: 2D projection (PCA or t-SNE) of pretrained token embeddings. Student can search for tokens and see clusters: numbers cluster together, animals cluster together, "king" is near "queen." Interactive — hover for token labels, search to highlight. | The whole point of embeddings is that similar tokens end up near each other. A formula or code snippet can't show this — you need to see the clustering. Interactivity lets the student test their own hypotheses ("are colors near each other?"). |
| **Visual (static)** | Sinusoidal positional encoding heatmap. Rows = positions (0-19), columns = encoding dimensions (0-63). Color intensity shows the value. Student sees the characteristic pattern: low-frequency waves on the left, high-frequency on the right, each row (position) has a unique pattern. | The sinusoidal formula is abstract. The heatmap makes it geometric — you can SEE that nearby positions have similar patterns (similar colors) and distant positions differ. This visual maps directly to "nearby positions should have similar encodings." |
| **Symbolic (code)** | PyTorch nn.Embedding implementation and usage. Create an embedding layer, index into it, verify that embedding.weight[token_id] equals embedding(token_id_tensor). Then implement sinusoidal positional encoding from the formula. | This is a hands-on lesson with a notebook. The student needs to see that nn.Embedding is just a matrix with a fancy indexing shortcut, and positional encoding is a few lines of sin/cos. Code grounds the concepts in something they can run and modify. |
| **Intuitive / Analogy** | Embedding as a dictionary where the definitions are learned, not written. You start with a dictionary where every word's definition is random noise. During training, the definitions get refined until words with similar meanings have similar definitions. | Connects to everyday experience. Dictionaries define words. Embeddings define tokens. But unlike a real dictionary (fixed, written by humans), this dictionary is learned by seeing how tokens are used in context. |
| **Geometric / Spatial** | The equal-distance property of one-hot: in a one-hot space, every pair of tokens is at Euclidean distance sqrt(2). Show three tokens in 3D one-hot space: they sit at the tips of an equilateral triangle on the axes. In embedding space, similar tokens cluster. | Makes the one-hot limitation visceral: you cannot encode similarity when every point is equidistant. The geometric view shows that embeddings literally move related tokens closer together in space. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 core (token embeddings, positional encoding) + 1 supporting (one-hot encoding as a bridge concept)
- **Previous lesson load:** BUILD (tokenization — one core algorithm, hands-on)
- **Load trajectory:** STRETCH -> BUILD -> BUILD. Two BUILD lessons in a row is acceptable because the second BUILD (this lesson) has stronger novelty — positional encoding is genuinely new with no direct predecessor. However, the strong connections to existing knowledge (learnable parameters, matrix multiplication, CNN spatial contrast) keep it manageable.
- **Assessment:** BUILD is appropriate. The embedding concept connects heavily to existing knowledge (learnable parameters, matrix multiplication). Positional encoding is new but well-motivated by the CNN contrast. The notebook keeps it concrete.

### Connections to Prior Concepts

| Prior Concept | Connection | How |
|---------------|-----------|-----|
| nn.Linear weight matrix (2.1.3) | Embedding IS a weight matrix indexed by token ID. If you one-hot encode the token and multiply by the embedding matrix, you get the same result as nn.Embedding. | Show the equivalence: `one_hot @ embedding.weight` == `embedding(token_id)`. Same numbers. |
| Parameters are learnable knobs (1.1.4) | Embedding vectors are parameters that update during training, just like weights in nn.Linear. | Show `embedding.weight.requires_grad` is True. Show embedding in `model.parameters()`. The "knobs" metaphor extends naturally. |
| CNN spatial locality and weight sharing (3.1.1) | CNNs get position information for free: the filter slides across spatial positions, so output(i,j) corresponds to input position (i,j). Transformers don't slide — they process all tokens in parallel with no inherent position. Positional encoding is the fix. | Explicit ComparisonRow: "How CNNs see position" vs "How transformers see position (without positional encoding)." The CNN's inductive bias is spatial; the transformer has no spatial bias. |
| "Architecture encodes assumptions about data" (3.1, tokenization) | Extended again: the embedding dimension is a design choice (how rich a representation each token gets), and the choice to ADD positional encoding (vs concatenate, vs nothing) encodes an assumption about how position information should interact with token identity. | Brief note in the positional encoding section. |
| MNIST softmax (10 classes -> 50K classes) analogy (4.1.1, 4.1.2) | The output layer maps FROM embedding dimension back TO vocabulary size. So there's a symmetry: embedding maps V -> d, output layer maps d -> V. The embedding dimension is the "bottleneck" where the model does all its thinking. | Brief aside: "Notice the symmetry. The embedding table maps 50K token IDs to d-dimensional vectors. The output layer maps d-dimensional hidden states back to 50K logits. Everything interesting happens in that d-dimensional space." |

### Analogies That Could Be Misleading

- **"Embeddings are like a dictionary"** — Partially misleading because dictionaries have one definition per word. An embedding is a fixed vector per token, not a context-dependent meaning. Context-dependent meaning comes from attention (Module 4.2). Make this explicit: "The embedding gives a starting-point representation. It's the same vector for 'bank' whether you mean riverbank or financial bank. The model's later layers (attention) figure out which meaning applies." Forward reference, but necessary to prevent a misconception that embeddings solve polysemy.

### Scope Boundaries

**This lesson IS about:**
- Token embeddings as learnable lookup tables (nn.Embedding)
- One-hot encoding as a bridge concept (wasteful, no similarity, leads to embeddings)
- Why position matters for sequences (bag-of-words problem, contrast with CNN locality)
- Sinusoidal positional encoding (the original transformer approach)
- Learned positional encoding (the simpler, now more common approach)
- Combining token embeddings + positional encoding via addition
- Implementing both in PyTorch in the notebook
- Visualizing embedding space and positional encoding patterns

**This lesson is NOT about:**
- Rotary positional encodings (RoPE) — MENTIONED at most as a modern alternative
- ALiBi or other relative position methods — out of scope
- How the model uses the embeddings downstream (attention, Q/K/V) — Module 4.2
- Training embeddings from scratch on a dataset — Module 4.3 (pretraining)
- Word2Vec, GloVe, or other standalone embedding methods — different paradigm (these are trained separately, not jointly with the model). MENTION only if useful for context.
- Context-dependent representations (the output of attention layers) — Module 4.2
- The output layer / unembedding — MENTIONED briefly for symmetry but not developed

**Target depths:**
- Token embeddings (learned lookup table): DEVELOPED
- One-hot encoding: INTRODUCED (bridge concept, not a skill to practice)
- Positional encoding (sinusoidal): DEVELOPED
- Learned positional encoding: INTRODUCED
- Embedding + position addition: DEVELOPED (implemented in notebook)

### Lesson Outline

#### 1. Context + Constraints
What: "This lesson completes the input pipeline: how integer token IDs become the vectors that feed into the model."
Not: "We're not yet looking at what happens to these vectors inside the model (attention, transformer blocks). That's Module 4.2."
Callback: "In Lesson 1, you learned that a language model predicts the next token. In Lesson 2, you built the tokenizer that converts text to integer IDs. Now: how do those integers become something a neural network can process?"

#### 2. Recap (brief)
Quick visual: the pipeline so far. Text -> BPE -> integer IDs. "You can now produce [464, 3797, 3332, 319, 262, 2603]. But 464 is just an index. The model needs vectors. How do we get there?" (1-2 paragraphs + a pipeline diagram, not a full re-teach.)

#### 3. Hook — The "Integers Aren't Meaningful" Problem
Type: Puzzle/problem reveal.
Show the student: token IDs for "cat" (2364) and "dog" (8976) and "the" (464). Ask: "Is 'cat' closer to 'dog' than to 'the'? Not according to the integers — 2364 is actually closer to 464 than to 8976. The IDs are arbitrary. The model needs a representation where similarity is meaningful."
Why this hook: Creates the specific problem that embeddings solve. The student already knows IDs are arbitrary (from BPE merge order), but hasn't felt the consequence for the model.

#### 4. Explain Part 1 — One-Hot Encoding (the naive approach)
- What it is: V-dimensional vector, all zeros except a 1 at the token's index
- Show concretely: token "cat" (ID 2) in a 5-word vocabulary -> [0, 0, 1, 0, 0]
- The waste problem: for a real 50K vocabulary, each token is a 50,000-dimensional vector with one nonzero entry
- The similarity problem (geometric): every pair of one-hot vectors is at distance sqrt(2). Show 3 tokens in 3D space sitting at axis tips — they form an equilateral triangle. "cat" is equally far from "dog" and from "the". One-hot CANNOT encode similarity.
- The saving grace: one-hot @ matrix = row selection. Work through the tiny example (3 tokens, 4-dim embeddings). This IS embedding lookup.

#### 5. Explain Part 2 — Embeddings as Learned Lookup
- The insight: instead of creating a 50K-sparse one-hot vector and multiplying by a matrix, just look up the row directly. nn.Embedding does exactly this.
- Code: `embedding = nn.Embedding(vocab_size, embed_dim)`. Show that `embedding(torch.tensor([2]))` returns `embedding.weight[2]`.
- Learned, not fixed: `embedding.weight.requires_grad` is True. These vectors update during training. At initialization, they're random. After training, similar tokens cluster.
- The analogy: a dictionary where definitions are learned, not written. Start with random noise for every definition. Training refines them.
- Address misconception: "Embeddings are preprocessing." No — they're the FIRST layer of the model. They have gradients. They learn.
- Address misconception: "IDs carry meaning." No — the embedding layer's entire job is to assign meaning to arbitrary IDs.
- WarningBlock: "The embedding gives each token ONE vector regardless of context. 'bank' (river) and 'bank' (money) get the same embedding. Context-dependent meaning comes later, from attention (Module 4.2)."

#### 6. Check 1 — Predict and Verify
"You have nn.Embedding(50000, 768). How many learnable parameters does this layer have?" Answer: 50000 * 768 = 38,400,000. "That's 38.4 million parameters — just for the embedding table. Now you understand the parameter count discussion from the vocabulary size section in Lesson 2."
Follow-up: "If you double the vocabulary size, what happens to the embedding parameter count?" (Doubles.) "If you halve the embedding dimension?" (Halves.)
This callback resolves the opacity from the tokenization lesson where "more entries = more parameters" was stated without the student understanding the mechanism.

#### 7. Explore — Embedding Space Widget
Interactive 2D projection (PCA or t-SNE) of pretrained token embeddings (e.g., from GPT-2).
- Student can search for tokens and see where they land
- Pre-loaded interesting clusters: numbers (1,2,3...), animals (cat,dog,bird), countries, programming keywords
- TryThisBlock: "Search for 'king' and 'queen'. Are they near each other? Now search for 'man' and 'woman'. What pattern do you see?"
- Observation: similar tokens cluster. This is what training produces — not what initialization looks like.

#### 8. Explain Part 3 — The Bag-of-Words Problem (Position Motivation)
- Transition: "We now have a vector for each token. But we have a problem."
- The key negative example: "Dog bites man" and "Man bites dog" have the SAME set of tokens with the SAME embeddings. If we just sum or concatenate these embeddings, the model sees identical inputs for both sentences. This is a bag of words.
- ComparisonRow: How CNNs handle position vs how embeddings (without position) handle position.
  - CNN: filter slides over spatial positions. Output at position (i,j) comes from input at position (i,j). Position is implicit in the architecture.
  - Embeddings alone: all positions are interchangeable. No inherent notion of first, second, third.
- "If you care about order — and language IS order — you have to inject position information explicitly."

#### 9. Explain Part 4 — Sinusoidal Positional Encoding
- The formula: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
- Don't start with the formula. Start with the requirements: (1) each position gets a unique encoding, (2) nearby positions should have similar encodings, (3) the encoding should work for any sequence length (not just lengths seen during training), (4) it should be deterministic (no learning needed)
- Sinusoidal waves at different frequencies satisfy all four: low-frequency waves change slowly (nearby positions similar), high-frequency waves change fast (distant positions differ), waves are periodic (any position has a value), and the pattern is fixed.
- Static visual: heatmap of positional encoding. Rows = positions (0-19), columns = dimensions (0-63). Left columns (low freq) change slowly, right columns (high freq) change rapidly.
- The addition: token_embedding + positional_encoding. Same dimension (d). Position information is blended into the representation, not appended.
- Address misconception: "Why not concatenate?" Concatenation would double the dimension and require different downstream architecture. Addition keeps the dimension the same and lets the model learn to use both signals from the same representation.

#### 10. Explain Part 5 — Learned Positional Encoding
- The simpler alternative: just create another nn.Embedding(max_seq_len, embed_dim) and LEARN the position vectors.
- This is what GPT-2 and most modern models actually use (with the exception of RoPE-based models).
- Tradeoff: can't generalize to sequence lengths not seen during training (the learned embedding for position 2049 doesn't exist if max_seq_len was 2048). Sinusoidal can extrapolate.
- In practice, learned positional encoding works just as well or better for the sequence lengths models are trained on.
- Brief MENTION of RoPE: "Modern models like LLaMA use Rotary Position Embeddings (RoPE), which encode relative position between tokens rather than absolute position. We'll encounter this in Module 4.3."

#### 11. Check 2 — Transfer Question
"You're building a language model for DNA sequences (A, C, G, T — a 4-token vocabulary). Would you use sinusoidal or learned positional encoding? Does it change if your training data has sequences of length 1000 but you want to process sequences of length 5000 at inference?"
Expected reasoning: With a 4-token vocabulary, embedding dimension might be small. For length extrapolation (1000 -> 5000), sinusoidal can generalize; learned can't. This tests whether the student understood the tradeoff, not just the mechanism.

#### 12. Practice — Notebook Implementation
Notebook: `4-1-3-embeddings-and-position.ipynb`
Scaffolding level: Supported (structure provided, student fills in key parts)

Exercises:
1. **Create an embedding layer and verify the lookup** — Create nn.Embedding(10, 4), manually index embedding.weight[3], call embedding(tensor([3])), verify they match. Then show the one-hot equivalence: one_hot @ embedding.weight vs embedding(token_id).
2. **Implement sinusoidal positional encoding from scratch** — Given the formula, implement a function that returns the positional encoding matrix for a given sequence length and embedding dimension. Visualize as a heatmap.
3. **Combine embedding + positional encoding** — Take a sentence, tokenize it (using their BPE from Lesson 2 or a pretrained tokenizer), look up embeddings, add positional encoding, print the final representation shape.
4. **Explore pretrained embeddings (stretch)** — Load GPT-2 token embeddings, find nearest neighbors for a few tokens, compute cosine similarity between token pairs. See that semantically similar tokens cluster.

#### 13. Summarize
Key takeaways:
- Token IDs are arbitrary integers. Embeddings map them to dense vectors where similarity is meaningful.
- nn.Embedding is a learnable weight matrix indexed by integer. Mathematically equivalent to one-hot @ matrix, but without creating the sparse vector.
- Without positional encoding, embeddings create a bag of words — the model can't tell "dog bites man" from "man bites dog."
- Sinusoidal positional encoding uses multi-frequency waves to give each position a unique, smooth encoding. Learned positional encoding just learns a vector per position.
- Token embedding + positional encoding = the input to the transformer.

Mental model to echo: "The input pipeline is now complete: text -> tokens -> IDs -> embedding vectors + position -> the tensor the model processes. Everything from here (attention, transformer blocks, the whole model) operates on these vectors."

#### 14. Next Step
"These embedding vectors are what the model processes. But right now, each token's vector is independent — the embedding for 'cat' is the same whether the context is 'the cat sat' or 'the cat died.' The next module introduces the mechanism that makes tokens context-aware: attention. In Module 4.2, you'll see how each token looks at all other tokens to build a context-dependent representation. The embedding vectors you just built are exactly what the queries, keys, and values will be projected FROM."

---

## Gate Check (Phase 3)

### Prerequisite Audit
- [x] Every assumed concept listed with required depth (10 concepts in Phase 1)
- [x] Each traced via the records — nn.Linear from nn-module (2.1.3), matrix multiply from tensors (2.1.1), etc.
- [x] Depth match verified: all OK except one-hot (MISSING, resolved with in-lesson section)
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises (embedding space widget uses pretrained embeddings, no attention knowledge needed)
- [x] One-hot gap has explicit resolution plan (small gap, 2-3 paragraphs + visual)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] 5 modalities planned: concrete example, visual (widget), visual (static heatmap), symbolic (code), intuitive/analogy, geometric/spatial
- [x] 5 examples: 3 positive (one-hot matrix mult, sinusoidal heatmap, init vs trained embeddings) + 1 negative (dog bites man) + 1 stretch (pretrained embedding exploration)
- [x] 4 misconceptions with negative examples
- [x] Cognitive load = 2 core concepts (embeddings, positional encoding) + 1 bridge (one-hot)
- [x] Every new concept connected: embeddings -> nn.Linear/learnable params; positional encoding -> CNN spatial locality
- [x] Scope boundaries explicitly stated

---

## Widget Specification

### EmbeddingSpaceExplorer

**Purpose:** Make the "similar tokens cluster in embedding space" claim visceral and explorable.

**Data:** Pretrained token embeddings (GPT-2 or similar), reduced to 2D via PCA or t-SNE. Pre-computed and bundled as a static JSON file (no runtime ML computation).

**Interactions:**
- Search bar: type a token, it highlights in the scatter plot
- Pre-loaded clusters: buttons for "Numbers", "Animals", "Countries", "Emotions" that zoom to those regions
- Hover: shows token label and nearest neighbors
- Click: selects a token, shows its 5 nearest neighbors with cosine similarity scores

**Visual design:** Scatter plot with ~2000 points (most common tokens). Light gray dots for background, violet for highlighted/selected tokens. Clean, minimal axes (these are projections, exact coordinates don't matter).

**TryThisBlock suggestions:**
- "Search for 'king' and 'queen.' How close are they?"
- "Find the cluster of numbers. Are they in order?"
- "Search for 'Python' (the language). What's near it?"
- "Compare 'happy' and 'sad' — close or far?"

### PositionalEncodingHeatmap (static visual, not interactive widget)

**Purpose:** Show the sinusoidal pattern visually.

**Design:** Heatmap with positions (0-19) on y-axis and dimensions (0-63) on x-axis. Diverging color scale (blue for negative, white for zero, red for positive). Left columns (low-frequency) change slowly across rows. Right columns (high-frequency) oscillate rapidly.

**Annotation:** Label a few dimensions with their wavelength. Arrow pointing to the left: "Changes slowly (captures coarse position)." Arrow pointing to the right: "Changes fast (captures fine position)."

This could be a Recharts-based or Visx-based static component rather than a full interactive widget.

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings that would leave the student fundamentally lost, but one critical consistency error in the heatmap that directly contradicts the lesson's own formula explanation, plus several improvement-level findings that would meaningfully strengthen the lesson.

### Findings

---

### [CRITICAL] — Positional encoding heatmap annotations are backwards

**Location:** PositionalEncodingHeatmap widget (annotations at lines 177-186) AND lesson aside "What to Notice" (lines 735-739)
**Issue:** The heatmap widget's directional annotations say "Changes slowly (coarse position)" on the left and "Changes rapidly (fine position)" on the right. The "What to Notice" aside says "Left columns change slowly across rows" and "Right columns oscillate rapidly." Both are backwards. In the actual computation (and in the standard PE formula), dimension index i=0 (leftmost column) has denominator 10000^0 = 1, so sin(pos/1) oscillates rapidly across positions. High-index dimensions (rightmost) have enormous denominators, so values barely change across 20 positions. The lesson's own formula explanation (lines 707-708) correctly states "the first dimensions oscillate rapidly (like a second hand), later dimensions oscillate slowly (like an hour hand)" -- but then the visualization says the opposite.
**Student impact:** The student reads the correct explanation ("first dimensions oscillate rapidly"), then looks at the heatmap and sees arrows claiming the opposite. Either they believe the arrows (wrong mental model) or they're confused by the contradiction. Either outcome is harmful. This is especially bad because the whole point of the heatmap is to make the abstract formula tangible -- if the tangible visualization contradicts the formula, neither sticks.
**Suggested fix:** Swap the heatmap annotations: left arrow should say "Changes rapidly (fine position)" and right arrow should say "Changes slowly (coarse position)." Update the "What to Notice" aside to match: "Left columns oscillate rapidly across rows -- these are high-frequency waves" and "Right columns change slowly -- low-frequency waves." Verify visually after the fix that the heatmap pattern matches the corrected labels. Note: the planning document's widget spec (line 311) also has this backwards and should be corrected for the record.

---

### [IMPROVEMENT] — Embedding space explorer uses fabricated data without clear disclosure

**Location:** Section 7 "Explore: Embedding Space" and EmbeddingSpaceExplorer widget
**Issue:** The lesson says "This is a 2D projection of token embeddings from a trained model" (line 537) and the widget's doc comment says "Pre-computed embedding data (PCA-reduced from GPT-2 embeddings)" with a note "Coordinates are approximate 2D projections that preserve cluster structure." But the data is hand-placed points with neatly separated clusters. Real PCA-reduced GPT-2 embeddings do not produce clusters this clean. The widget has ~120 hand-curated tokens in fabricated positions. This is defensible as a pedagogical simplification, but the framing "from a trained model" implies real data.
**Student impact:** The student believes they are seeing real embedding geometry. When they later explore actual pretrained embeddings (in the notebook stretch exercise), the messier reality may undermine their trust in what the lesson taught. The lesson should be honest about the simplification rather than claiming the data comes from a trained model.
**Suggested fix:** Change the introductory text to something like: "This shows the general structure of token embeddings. The positions are simplified from real GPT-2 embeddings to make cluster structure visible -- in practice the clusters are less cleanly separated, but the overall pattern holds: semantically similar tokens group together." This maintains the pedagogical value while being transparent.

---

### [IMPROVEMENT] — CNN contrast paragraph references lesson title that may not match route

**Location:** Lines 627-633, the CNN contrast paragraph
**Issue:** The paragraph says "In What Convolutions Compute, you saw that CNNs handle position naturally." This references a lesson by its title, but the student may not remember this exact title. More importantly, the claim is presented without any recap of why CNNs handle position -- the student is expected to recall the sliding-filter mechanism from a lesson that may have been several sessions ago (it's in Series 3, Module 3.1). The Reinforcement Rule says concepts from 3+ lessons ago should be reinforced before building on them.
**Student impact:** If the student doesn't immediately recall how CNN filters encode spatial position, the contrast feels like an unsupported assertion rather than a genuine "aha" moment. The ComparisonRow below helps, but the transition paragraph that sets it up assumes more recall than is safe.
**Suggested fix:** Add one sentence of recap before the contrast: "In What Convolutions Compute, you saw that a CNN filter slides over spatial positions -- the output at position (i, j) comes directly from the input at and around position (i, j), so position is baked into the architecture." Then the contrast ("Transformers don't slide") has its foundation refreshed. The ComparisonRow already contains this information, but the lead-in paragraph needs it too so the student isn't confused during the transition.

---

### [IMPROVEMENT] — Missing concrete negative example for the "embeddings are preprocessing" misconception

**Location:** Lines 425-463, the "Embeddings are learned, not fixed" section
**Issue:** The planning document specifies a misconception "Embeddings are a preprocessing step" with a concrete negative example: "Show that embedding vectors change during training: the embedding for 'the' at initialization is random, but after training it's near other determiners." The built lesson addresses this with a WarningBlock ("Embeddings are the first layer of the model, not a preprocessing step") and code showing requires_grad=True. But there is no concrete before/after comparison. The WarningBlock asserts the point; it doesn't demonstrate it with a negative example that the student can see.
**Student impact:** The student reads "embeddings are not preprocessing" as a declarative statement they must accept. Without a concrete before/after example (random noise at init -> meaningful clusters after training), the claim rests on authority rather than evidence. The embedding space widget partially fills this gap (the paragraph after it says "At initialization, every token is random noise"), but that observation comes much later and isn't framed as disproving the misconception.
**Suggested fix:** In the "Embeddings are learned, not fixed" section, add a brief concrete illustration. Even pseudo-code or a conceptual example: "At initialization, embedding('the') = [0.42, -0.18, 0.73, ...] (random noise). embedding('a') = [-0.55, 0.91, 0.02, ...] (also random noise, nowhere near 'the'). After training on billions of tokens: embedding('the') = [0.91, 0.12, -0.21, ...] and embedding('a') = [0.89, 0.15, -0.19, ...] (nearly identical, because they appear in similar contexts)." This makes the "learned" claim tangible.

---

### [IMPROVEMENT] — The one-hot encoding section has a small but real modality gap

**Location:** Section 4, "One-Hot Encoding"
**Issue:** The planning document specifies a geometric/spatial modality: "Show three tokens in 3D one-hot space: they sit at the tips of an equilateral triangle on the coordinate axes." The built lesson has a text reference to this idea (line 283-286: "Think of three tokens in 3D one-hot space: they sit at the tips of an equilateral triangle on the coordinate axes") but no actual visual. The equilateral triangle on the axes is described verbally, not shown. The Modality Rule requires that modalities be genuinely different perspectives, not "the same explanation rephrased." A verbal description of a spatial concept is still verbal.
**Student impact:** The student has to mentally construct a 3D coordinate system with points at (1,0,0), (0,1,0), (0,0,1) from text alone. Some students will; many won't. The insight that all one-hot vectors are equidistant would land much harder with even a simple static 3D diagram showing three dots at the axis tips.
**Suggested fix:** Add a small static visual (even an SVG) showing three points labeled "cat", "dog", "the" at the tips of three coordinate axes. Annotate the equal distances. This doesn't need to be a full interactive widget -- a styled inline SVG or a simple diagram would suffice.

---

### [POLISH] — "Articles" cluster label in widget is misleading

**Location:** EmbeddingSpaceExplorer widget, CLUSTER_LABELS mapping (line 63)
**Issue:** The cluster is called "articles" internally but labeled "Function Words" in the UI. The tokens in this cluster include "the", "a", "an" (articles), but also "is", "was", "are", "were", "been", "have", "has" (copulas and auxiliaries). "Function Words" is linguistically correct. However, the "Try This" aside says "Click 'Function Words.' Where are 'the', 'a', 'is'?" -- the student may not know what "function words" means.
**Student impact:** Minor confusion for students unfamiliar with the linguistics term. The context makes it clear enough, but a brief parenthetical would help.
**Suggested fix:** Either add a parenthetical in the "Try This" aside: "Click 'Function Words' (the, a, is, was...)" or rename the button to "Common Words" which is more intuitive for non-linguists.

---

### [POLISH] — Next step text mentions Q/K/V which has not been taught

**Location:** Lines 1020-1024, NextStepBlock description
**Issue:** The NextStepBlock says "the mechanism that makes tokens context-aware." This is fine. But the planning document's Section 14 outline includes: "The embedding vectors you just built are exactly what the queries, keys, and values will be projected FROM." This forward reference to Q/K/V was wisely omitted from the built lesson (good judgment by the builder). However, the built text could be slightly stronger in its forward tease.
**Student impact:** Negligible. This is a note that the builder made a good scope decision. The current forward reference is well-calibrated.
**Suggested fix:** No action needed. This is a positive observation, not something to fix.

---

### [POLISH] — The module completion block lists "BPE tokenization (built from scratch)" but the student also learned about broader tokenization concepts

**Location:** Lines 1002-1014, ModuleCompleteBlock
**Issue:** The achievement "BPE tokenization (built from scratch)" accurately reflects what the student did, but the tokenization lesson also covered subword tokenization as a concept, vocabulary size tradeoffs, and tokenization artifacts. These don't need individual bullet points, but the current wording might slightly undersell the tokenization lesson's breadth.
**Student impact:** Negligible. The student knows what they learned. The module completion block is a moment of satisfaction, not a comprehensive record.
**Suggested fix:** Consider changing to "Subword tokenization and BPE (built from scratch)" to nod at the conceptual coverage. Low priority.

---

### Review Notes

**What works well:**
- The narrative arc is strong. The "integers aren't meaningful" hook creates genuine tension that the lesson resolves systematically. The pipeline-so-far visual is a clean callback to the tokenization lesson.
- The one-hot to embedding bridge is well-executed. The progression from "naive approach" -> "two problems" -> "saving grace (row selection)" -> "nn.Embedding is the efficient version" is logical and well-paced.
- The bag-of-words motivation for positional encoding is excellent. "Dog bites man" vs "Man bites dog" is a classic example used perfectly -- concrete, memorable, and the formalization (sets are identical) is rigorous.
- The CNN contrast via ComparisonRow is well-structured and draws on a genuine strength of the curriculum (the student actually built CNN intuition in Series 3).
- Scope boundaries are well-maintained. The lesson correctly avoids going deep into RoPE, Word2Vec, or attention. Forward references are brief and clearly deferred.
- The polysemy warning (line 467-481) is well-placed and correctly scoped. It addresses a misconception without digressing.
- Both knowledge checks (parameter count, DNA transfer question) test understanding rather than recall.
- The module completion block provides genuine closure for Module 4.1.

**Systemic pattern:**
The heatmap direction error likely originated in the planning document's widget spec (which also says "left = slow, right = fast"), and propagated into both the widget implementation and the lesson aside. This is a good argument for verifying widget behavior against the mathematical definitions during building, not just implementing what the plan says.

**Forward-reference handling:**
The review brief asked about whether the forward-reference handling from the tokenization lesson is properly resolved (the "embedding" references that were removed). This is handled well: the lesson starts from "you have integers, now what?" and never assumes the student has heard the word "embedding" before. The tokenization lesson's careful avoidance of the term pays off -- this lesson gets to introduce it fresh.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All critical and improvement findings from iteration 1 have been properly addressed. The lesson is pedagogically sound, well-paced, and consistent throughout. The two remaining findings are minor polish items that do not affect learning outcomes.

### Findings

---

### [POLISH] — Spaced em dash in EmbeddingSpaceExplorer search result

**Location:** EmbeddingSpaceExplorer widget, line 364 (`<span className="text-muted-foreground"> — nearest: </span>`)
**Issue:** The em dash has spaces on both sides: ` — nearest: `. The project's writing style rule requires no spaces around em dashes: `word—word`.
**Student impact:** Negligible. A minor visual inconsistency with the rest of the lesson's punctuation style.
**Suggested fix:** Change ` — nearest: ` to `—nearest: ` or restructure to avoid the em dash (e.g., use a colon or pipe character instead).

---

### [POLISH] — Cluster buttons in EmbeddingSpaceExplorer lack explicit cursor-pointer

**Location:** EmbeddingSpaceExplorer widget, line 344-356 (cluster `<button>` elements)
**Issue:** The cluster filter buttons are `<button>` elements but do not have `cursor-pointer` in their className. Browser default for `<button>` is `cursor: default`, not pointer. Other interactive widgets in the codebase (e.g., AutogradExplorer) explicitly add `cursor-pointer` to buttons. The clear-search button (line 334) also lacks it.
**Student impact:** Minor. Most users still understand buttons are clickable, but the cursor doesn't change to a pointer on hover, which is slightly less polished than other widgets.
**Suggested fix:** Add `cursor-pointer` to both the cluster buttons and the clear-search button className.

---

### Review Notes

**Iteration 1 fixes — all properly resolved:**

1. **CRITICAL (heatmap annotations):** Fixed correctly. The heatmap widget annotations, the lesson aside "What to Notice," and the formula explanation text all now consistently say: left columns = rapid oscillation / high frequency / small denominator, right columns = slow change / low frequency / huge denominator. Verified against the actual computation in `computePositionalEncoding` — dimension 0 has denominator 10000^0 = 1 (fast oscillation), dimension 63 has denominator ~10000^0.97 (barely changes). The visual pattern matches the corrected labels.

2. **IMPROVEMENT (embedding explorer transparency):** The introductory text now says "The positions are simplified to make cluster structure visible — in real models, clusters are messier but the pattern holds." Honest and pedagogically appropriate.

3. **IMPROVEMENT (CNN convolution recap):** The paragraph before the CNN contrast ComparisonRow now includes a full recap sentence: "a small filter slides over spatial neighborhoods, comparing each pixel to its immediate neighbors. Because the filter moves one step at a time, the output at position (i, j) comes directly from the input at and around position (i, j)." This refreshes the concept adequately before building the contrast with transformers.

4. **IMPROVEMENT (before/after embedding comparison):** Added a concrete side-by-side in the "Embeddings are learned, not fixed" section showing random init vectors vs trained vectors for "the" and "a". Includes explicit labels ("At initialization (random noise)" and "After training on billions of tokens") with color-coded notes (rose for dissimilar, emerald for similar). Makes the "learned" claim tangible rather than assertive.

5. **IMPROVEMENT (SVG triangle for one-hot equidistance):** Added an inline SVG showing three points at coordinate axis tips (1,0,0), (0,1,0), (0,0,1) with dashed distance lines and sqrt(2) labels. Includes a caption. This fills the geometric/spatial modality gap that existed when the equidistance property was described only verbally.

6. **POLISH (function words parenthetical):** The "Try This" aside now reads "Click 'Function Words' (the, a, is, was...)." Clear enough for non-linguists.

7. **POLISH (module completion wording):** Changed to "Subword tokenization and BPE (built from scratch)" — acknowledges the broader tokenization concepts while still highlighting the hands-on BPE work.

**What works particularly well after the fixes:**
- The heatmap is now a reliable visual aid. The annotations, aside, and formula text form a consistent trio that reinforces the same understanding from three angles.
- The before/after embedding comparison in the "learned, not fixed" section is well-placed and well-formatted. It comes right after the requires_grad code, so the student gets both the technical evidence (it has gradients) and the conceptual evidence (random noise becomes meaningful clusters).
- The SVG triangle is simple but effective — it adds a genuinely different modality (geometric/spatial) rather than just another verbal description.
- The CNN recap paragraph gives the student just enough refresher to make the contrast land without being patronizing.

**Overall assessment:**
This lesson is ready to ship. The narrative arc is strong, the modality coverage is excellent (6+ modalities for the core concepts), misconceptions are proactively addressed with concrete negative examples, scope boundaries are well-maintained, and the two knowledge checks test genuine understanding. The remaining two polish items are minor enough to fix without re-review.
