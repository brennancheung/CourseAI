# Module 4.1: Language Modeling Fundamentals — Record

**Goal:** The student understands what language models are (probability distributions over token sequences), how text is represented for neural networks (tokenization and embeddings), and why position information requires special handling — giving them the complete input pipeline before attention is introduced in Module 4.2.
**Status:** Complete (3 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Language modeling as next-token prediction (predict probability distribution over vocabulary given context) | DEVELOPED | what-is-a-language-model | Core concept. Connected to supervised learning: input = context tokens, target = next token, loss = cross-entropy. Comparison table mapping house prices/MNIST/LM into the same framework. |
| Autoregressive generation (sample token, append to context, repeat) | DEVELOPED | what-is-a-language-model | The feedback loop that turns a next-token predictor into a text generator. Five-step walkthrough generating from "The cat sat on the" with violet highlighting showing context growth. Mermaid diagram of the loop. |
| Probability distribution over vocabulary (softmax over 50,000+ tokens) | INTRODUCED | what-is-a-language-model | Grounded via connection to MNIST softmax (10 classes -> 50,000 tokens). Recharts bar chart showing distribution for "The cat sat on the ___". Emphasized that the right prediction is the entire distribution, not just the top token. |
| Conditional probability P(x_t \| x_1, ..., x_{t-1}) | INTRODUCED | what-is-a-language-model | Notation introduced with plain-English gloss: "the probability of the next token, given everything that came before." Connected to "predict the output given the input" — what the student has been doing all along. |
| Temperature as distribution reshaping (softmax(logits / T)) | INTRODUCED | what-is-a-language-model | Interactive TemperatureExplorer widget with slider (0.1 to 3.0). Low T = winner-take-all, T=1 = standard softmax, high T = nearly uniform. Explicitly addressed misconception that temperature changes model knowledge. Entropy stat shown. |
| Self-supervised labels (text provides its own training labels) | INTRODUCED | what-is-a-language-model | Every position in a text is simultaneously a training example — the next token IS the label. No human labeling required. This is why LMs can train on trillions of tokens. |
| Sampling vs greedy decoding | INTRODUCED | what-is-a-language-model | WarningBlock: model samples from distribution, doesn't always pick highest-probability token. Greedy decoding (always pick top token) produces repetitive text. |
| Top-k and top-p (nucleus) sampling | MENTIONED | what-is-a-language-model | Single paragraph noting these exist as additional sampling strategy knobs. Not developed. |
| Base model vs chat model distinction | MENTIONED | what-is-a-language-model | Base models trained on raw text learn to continue text, not answer questions. Chat models have additional finetuning (SFT + RLHF). Connected to transfer learning pattern from Series 3. Forward reference to Module 4.4. |
| Training vs generation difference (parallel vs sequential) | MENTIONED | what-is-a-language-model | Brief note: during training the model sees entire text at once (efficient). During generation, one token at a time (sequential). Mentioned, not belabored. |
| Subword tokenization (tokens are pieces of words, not characters or whole words) | DEVELOPED | tokenization | Core concept. Character-level has tiny vocab but sequences explode; word-level is compact but novel words are invisible ([UNK]). Subword is the middle ground: common words stay whole, rare words split into recognizable pieces. Three-way comparison on "The cat sat on the mat" (character=22 tokens, word=6, BPE=6 for common text). |
| BPE algorithm (start from characters, iteratively merge most frequent pair) | APPLIED | tokenization | Implemented from scratch in notebook. Training phase: count adjacent pairs, merge most frequent, repeat until target vocab size. Encoding phase: apply learned merges in priority order to new text. Step-by-step walkthrough with "low lower newest" corpus. Interactive BpeVisualizer widget. Five functions: get_pair_counts, merge_pair, train_bpe, encode, decode. |
| Character vs word vs subword tokenization tradeoffs | DEVELOPED | tokenization | Character-level: ~100 vocab items, 22 tokens for "The cat sat", no OOV, but sequences too long and model must learn spelling. Word-level: compact sequences, each token meaningful, but OOV problem (ChatGPT -> [UNK]). Subword: sweet spot. ComparisonRow and three-way example. |
| Vocabulary size as a design tradeoff | INTRODUCED | tokenization | Bigger vocab = shorter sequences but more output classes, more parameters, rare tokens poorly learned. Smaller vocab = longer sequences but each token better learned. Connected to MNIST 10 -> LM 50K scaling. Examples: GPT-2 50K, GPT-4 ~100K, LLaMA 32K, LLaMA 3 128K. |
| Tokenization artifacts and failure modes (strawberry problem, arithmetic, multilingual inequality, glitch tokens) | INTRODUCED | tokenization | "strawberry" -> ["straw", "berry"] explains why LLMs fail at letter counting. Numbers tokenize inconsistently (42 vs [42, 7]). BPE trained on English fragments other languages (2-5x more tokens). SolidGoldMagikarp: token in vocabulary but absent from training data = untrained representation = bizarre behavior. Key takeaway: tokenization is not neutral preprocessing. |
| Out-of-vocabulary (OOV) problem with word-level tokenization | DEVELOPED | tokenization | "ChatGPT" -> [UNK] with word-level. All information lost for any word not in training vocabulary. Motivates why subword tokenization is necessary. GradientCard with concrete example. |
| BPE merge table as the tokenizer (deterministic encoding once trained) | DEVELOPED | tokenization | The ordered list of merges IS the tokenizer. Same input + same merge table = same tokens, every time. Explicitly addressed misconception that BPE is stochastic. Training uses frequency statistics; encoding is deterministic lookup. |
| WordPiece tokenization (BERT) | MENTIONED | tokenization | Similar to BPE but selects merges by maximizing likelihood, not just frequency. Single bullet point. |
| SentencePiece / Unigram tokenization (LLaMA, T5) | MENTIONED | tokenization | Starts with large vocabulary and prunes down — reverse of BPE's bottom-up approach. Single bullet point. |
| Byte-level BPE (GPT-2) | MENTIONED | tokenization | BPE applied to raw bytes instead of characters. Eliminates Unicode handling, guarantees all inputs tokenizable. Single sentence. |
| Token embeddings as learned lookup table (nn.Embedding maps integer ID to dense vector) | DEVELOPED | embeddings-and-position | Core concept. Motivated by "integers aren't meaningful" problem. Showed one-hot encoding as naive approach (two problems: sparse/wasteful, every pair equidistant at sqrt(2)). Key insight: one-hot @ matrix = row selection, so nn.Embedding just skips the sparse vector and indexes the row directly. Code: nn.Embedding(50000, 768). Verified embedding.weight[i] == embedding(tensor([i])). Parameter count exercise: 50K x 768 = 38.4M. |
| One-hot encoding (V-dimensional sparse vector with single 1 at token index) | INTRODUCED | embeddings-and-position | Bridge concept motivating dense embeddings. Two problems: (1) enormous sparse vectors (50K dims, 99.998% zeros), (2) every pair equidistant (cannot represent similarity). Inline SVG showing 3 tokens at axis tips forming equilateral triangle, all at distance sqrt(2). The "saving grace": one-hot times a matrix selects a row, which IS embedding lookup. |
| Embedding space clustering (similar tokens have nearby vectors after training) | DEVELOPED | embeddings-and-position | Interactive EmbeddingSpaceExplorer widget showing 2D projection with ~120 tokens in semantic clusters (animals, numbers, emotions, countries, function words, programming). Hover for nearest neighbors, search, cluster filter buttons. Explicitly noted positions are simplified from real GPT-2 to make clusters visible. Before/after training comparison: random noise at init vs meaningful clusters after training. Connected to "parameters are learnable knobs" mental model. |
| Embeddings are learned parameters, not preprocessing (requires_grad=True, updated by backprop) | DEVELOPED | embeddings-and-position | Explicitly contrasted with tokenization (fixed, deterministic). Showed embedding.weight.requires_grad is True and appears in model.parameters(). Before/after comparison: embed("the") and embed("a") random at init, nearly identical after training. WarningBlock: "Embeddings are the first layer of the model, not a preprocessing step." |
| Polysemy limitation of static embeddings (one vector per token regardless of context) | INTRODUCED | embeddings-and-position | "Bank" (river) and "bank" (money) get the same embedding. Context-dependent meaning comes from attention (Module 4.2). Amber warning box. Forward reference to attention as the mechanism that resolves polysemy. |
| Bag-of-words problem (embeddings without position lose sequence order) | DEVELOPED | embeddings-and-position | "Dog bites man" vs "Man bites dog" have identical embeddings as a set. Without position, model sees identical inputs. GradientCard with concrete formalization showing same set of embeddings. Motivated positional encoding. ComparisonRow: CNN spatial locality (filter slides, position implicit) vs embeddings without position (all positions interchangeable). |
| Sinusoidal positional encoding (sin/cos at exponentially increasing wavelengths) | DEVELOPED | embeddings-and-position | Formula: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d)). Four requirements derived before formula: unique per position, nearby positions similar, works for any length, deterministic. Clock analogy: second hand (fine), minute hand (coarse), hour hand (broadest). Interactive PositionalEncodingHeatmap widget: 20 positions x 64 dimensions, diverging color scale, hover for values. Left columns oscillate rapidly (high freq), right columns change slowly (low freq). |
| Positional encoding addition to token embeddings (input = embedding + PE, same dimension) | DEVELOPED | embeddings-and-position | Added, not concatenated. Both have dimension d, result is d-dimensional. Position and meaning blended into single representation. WarningBlock explaining why not concatenate (would double dimension, require different architecture). Formula: input_i = embedding(token_i) + PE(i). |
| Learned positional encoding (nn.Embedding(max_seq_len, embed_dim) for position) | INTRODUCED | embeddings-and-position | Simpler alternative: another embedding table indexed by position 0, 1, 2, ... Used by GPT-2 and most modern models. Tradeoff: can't generalize to unseen sequence lengths (no embedding for position 2049 if max_seq_len=2048). Sinusoidal can extrapolate. Code shown alongside token embedding. |
| Rotary Position Embeddings (RoPE) | MENTIONED | embeddings-and-position | Modern alternative encoding relative position between tokens rather than absolute position. Used by LLaMA. Single sentence, forward reference to Module 4.3. |
| Output layer / unembedding symmetry (embedding maps V->d, output maps d->V) | MENTIONED | embeddings-and-position | Violet-bordered aside noting the symmetry: embedding table maps 50K IDs to d-dimensional vectors, output layer maps d-dimensional hidden states back to 50K logits. Everything interesting happens in d-dimensional space. |

## Per-Lesson Summaries

### what-is-a-language-model
**Status:** Built
**Cognitive load type:** STRETCH
**Type:** Conceptual (no notebook)
**Widgets:** TemperatureExplorer — interactive bar chart with temperature slider (0.1 to 3.0) showing probability distribution over ~15 candidate tokens. Shows entropy stat, top token probability, and effective tokens count. Student drags slider and watches distribution flatten/sharpen.

**What was taught:**
- Language modeling is next-token prediction: given context, output a probability distribution over the vocabulary
- Autoregressive generation: the sample-append-repeat loop that turns a single-token predictor into a text generator
- Temperature reshapes the probability distribution before sampling (divides logits by T before softmax)
- Next-token prediction is simple to state but requires compressing enormous knowledge to do well
- This is supervised learning with self-generated labels — same training loop the student already knows

**How concepts were taught:**
- **Phone autocomplete hook:** Schematic mockup of phone keyboard with message bubble ("I'll be there in" with blinking cursor) and three suggestion buttons ("5", "a", "10"). Reframes daily experience as "you already use a language model."
- **Probability distribution grounding:** Connected to MNIST softmax (10 digit classes -> 50,000 tokens). Recharts bar chart showing probability distribution for "The cat sat on the ___" with violet gradient fills and custom tooltip.
- **Formal notation bridge:** P(x_t | x_1, ..., x_{t-1}) introduced with plain-English gloss. Conditional probability bar "|" explained as "given." Connected to "you've been doing this all along."
- **Supervised learning comparison:** ComparisonRow mapping house prices, MNIST, and language modeling into the same input-target-loss-training-loop framework. Key insight: training data generates its own labels.
- **Autoregressive walkthrough:** Five-step generation from "The cat sat on the" with GenerationStepCard components. Violet highlighting distinguishes generated tokens from prompt tokens (ContextWithHighlights component). Top probabilities shown at each step with sampled token highlighted.
- **Autoregressive loop diagram:** Mermaid diagram showing Context -> Model -> Distribution -> Sample -> Append -> back to Context.
- **Temperature widget (TemperatureExplorer):** Interactive slider with real-time bar chart. Formula shown: softmax(logits / T). TryThisBlock with specific experiments at T=0.1, T=1.0, T=3.0. Observation prompt: ranking stays the same at all temperatures.
- **"Why this is powerful" elaboration:** Chess analogy — "predict the next move" sounds simple but requires understanding all of chess. The task is a universal training signal.
- **"Does the model understand?" section:** Negative example — small LM asked to continue "2+3=" might say "5" or "23." Mechanistic framing: it predicts tokens based on learned patterns.

**Mental models established:**
- "A language model approximates P(next token | context)" — extension of "ML is function approximation"
- "Autoregressive generation is a feedback loop: outputs become inputs" — why same prompt can produce different text
- "Temperature changes sampling, not knowledge" — the model's parameters are fixed; only the distribution shape changes
- "Next-token prediction is a universal training signal" — grammar, facts, reasoning, style all compressed into one objective

**Analogies used:**
- Phone autocomplete as a tiny language model (demystifies the concept via universal daily experience)
- MNIST softmax (10 classes) -> language model softmax (50,000 classes): same operation, different scale
- Chess analogy: "predict the next move" is simple to state but requires understanding all of chess
- Transfer learning pattern: base model = pretrained backbone, chat behavior = finetuned head

**What was NOT covered (scope boundaries):**
- How the model works internally (attention, transformers) — Module 4.2
- Tokenization (what a "token" actually is, subword tokenization) — Lesson 2
- Embeddings (how token IDs become vectors) — Lesson 3
- How the model is trained at scale (pretraining data, compute) — Module 4.3
- Chat models vs base models in depth (finetuning, RLHF) — Module 4.4
- Any implementation or code — this is conceptual only

**Misconceptions addressed:**
1. "Language models understand language / think / reason" — Mechanistic framing: it predicts tokens based on learned text patterns. "2+3=" might produce "5" or "23." For engineering purposes: it predicts tokens.
2. "Language models generate text word by word" — Used "token" from the start. Noted tokens can be words, pieces of words, or punctuation. Full treatment deferred to Lesson 2.
3. "Temperature makes the model more creative / smarter" — Interactive widget makes it visceral: temperature reshapes probabilities, nothing more. The model's knowledge (parameters) is unchanged.
4. "Next-token prediction is too simple to produce intelligent behavior" — Predicting well in diverse text requires learning grammar, facts, reasoning patterns, style, domain knowledge. Universal training signal.
5. "A language model is trained on conversations / question-answer pairs" — Base models trained on raw text learn to continue text. Chat behavior requires additional finetuning (SFT + RLHF).

### tokenization
**Status:** Built
**Cognitive load type:** BUILD
**Type:** Hands-on (notebook: `4-1-2-tokenization.ipynb`)
**Widgets:** BpeVisualizer — interactive step-through of BPE algorithm. Student types text, clicks "Step" to apply one merge at a time. Shows: current tokens with color-coded boundaries, which pair was merged, vocabulary size, compression percentage. Stops when no pair appears more than once.

**What was taught:**
- Subword tokenization is the middle ground between character-level (tiny vocab, long sequences) and word-level (compact but can't handle novel words)
- BPE algorithm: start from characters, count adjacent pairs, merge most frequent, repeat until target vocabulary size
- The merge table IS the tokenizer — encoding is deterministic once trained
- Tokenization is not neutral preprocessing — it determines what the model can and cannot see
- Vocabulary size is a design tradeoff, not a universal constant

**How concepts were taught:**
- **"The Model Can't Read" hook:** The model needs integers, not text. Mystery integers [464, 3797, 3332, 319, 262, 2603] for "The cat sat on the mat". By the end you'll build the algorithm that produces these.
- **Three-way comparison:** Same sentence tokenized as characters (22 tokens, sky-blue chips), words (6 tokens, amber chips), and BPE (6 tokens for common text, violet chips). Character and word versions shown first as "obvious solutions that fail," BPE introduced as the resolution.
- **OOV negative example:** "ChatGPT" -> [UNK] with word-level tokenization. All information lost. GradientCard with rose color. Makes OOV visceral with a word the student knows.
- **Compression/abbreviation analogy:** BPE is like learning abbreviations — "th" + "e" -> "the" is like "btw" meaning "by the way." You compress the most common thing until you hit your budget.
- **BPE step-by-step walkthrough:** StepCard components tracing merges on "low low low low low lower lower newest newest." Steps 0-4 shown with highlighted merged tokens. Tie-breaking note between steps 2 and 3 explains implementation-dependent choice when counts are equal.
- **Interactive BpeVisualizer widget:** Student types their own text and steps through merges. TryThisBlock suggests: "unhappiness" (suffixes merge first), "ChatGPT" (split into pieces, not [UNK]), repeated words (faster merges), code (`print("hello")`), watching compression percentage.
- **Notebook implementation:** Five scaffolded functions — get_pair_counts, merge_pair, train_bpe, encode, decode. Student builds complete BPE tokenizer from scratch.
- **Encoding phase explanation:** After walkthrough, explicit example encoding "lowest" with the learned merge table. Start with characters, apply merges in priority order. Closes the gap between training and encoding.
- **Tokenization artifacts:** Three GradientCards (orange): "strawberry" -> ["straw", "berry"] (letter counting fails), "42" vs ["42", "7"] (arithmetic difficulty), multilingual inequality (same sentence costs 2-5x more tokens in Korean/Arabic). SolidGoldMagikarp: token in vocabulary but absent from training data = untrained internal representation = bizarre behavior.
- **Vocabulary size section:** Connected to MNIST 10 -> LM 50K callback. Bigger = more classes + more parameters + rare tokens poorly learned. Smaller = longer sequences + each token better learned. Real examples: GPT-2 50K, GPT-4 ~100K, LLaMA 32K, LLaMA 3 128K.
- **Transfer question:** 32K vocab model failing at multiplication. Would character-level fix it? Partially (consistent digit access), but sequences explode. Real fix is architectural (chain-of-thought, tool use).

**Mental models established:**
- "Tokenization defines what the model can see" — tokens are the atomic units of perception; anything below token level is invisible
- "BPE is text compression repurposed for tokenization" — iteratively compress most frequent pairs into single symbols
- "The merge table IS the tokenizer" — deterministic: same input + same merges = same tokens
- "Architecture encodes assumptions about data — and the tokenizer is the first architectural decision" — callback to Series 3 mental model, extended to tokenization as a design choice

**Analogies used:**
- Text compression / abbreviation: BPE merges common pairs like learning "btw" = "by the way"
- MNIST 10 classes -> LM 50K classes (reused from lesson 1): vocabulary size IS the output layer size
- Architecture encodes assumptions about data (reused from Series 3): tokenization is a design decision encoding assumptions about language

**What was NOT covered (scope boundaries):**
- Other subword algorithms in detail (WordPiece, SentencePiece) — MENTIONED only
- Byte-level BPE — MENTIONED only
- How embeddings work (how token IDs become vectors) — Lesson 3
- Training a language model — Module 4.3
- Production tokenizer implementation — student builds minimal, correct BPE
- Unicode handling, normalization, or encoding details

**Misconceptions addressed:**
1. "Tokens are words" — Disproved by OOV problem: word-level can't handle novel words. Subword tokens are pieces of words.
2. "Character-level is simplest, so it should work fine" — "The cat sat" = 22 tokens. Model must learn spelling. Sequences too long for practical use.
3. "A bigger vocabulary is always better" — More entries = more parameters = rare tokens poorly learned. There's a sweet spot.
4. "Tokenization is just preprocessing that doesn't affect the model" — Strawberry problem, arithmetic difficulty, multilingual inequality, SolidGoldMagikarp. Tokenization determines what the model can see.
5. "BPE is stochastic" — Training uses frequency statistics, but encoding is fully deterministic. Same input + same merge table = same tokens, every time.

### embeddings-and-position
**Status:** Built
**Cognitive load type:** BUILD
**Type:** Hands-on (notebook: `4-1-3-embeddings-and-position.ipynb`)
**Widgets:** EmbeddingSpaceExplorer — interactive 2D scatter plot with ~120 tokens in semantic clusters. Search bar, cluster filter buttons (Animals, Numbers, Emotions, Countries, Function Words, Programming), hover for nearest neighbors. Positions simplified from real GPT-2 embeddings to make cluster structure visible. PositionalEncodingHeatmap — interactive heatmap showing sinusoidal PE for 20 positions x 64 dimensions with diverging color scale. Hover for exact values and position/dimension info.

**What was taught:**
- Token IDs are arbitrary integers; embeddings map them to dense vectors where similarity is meaningful
- nn.Embedding is a learnable weight matrix indexed by integer (mathematically equivalent to one-hot x matrix, but without creating the sparse vector)
- One-hot encoding fails for two reasons: enormous sparse vectors and inability to represent similarity (every pair equidistant at sqrt(2))
- Embeddings are learned parameters (first layer of the model, not preprocessing) that cluster similar tokens through training
- Without positional encoding, embeddings create a bag of words — model can't distinguish "dog bites man" from "man bites dog"
- Sinusoidal positional encoding uses multi-frequency waves satisfying four requirements: unique, smooth, any-length, deterministic
- Learned positional encoding is simpler (another embedding table) but can't extrapolate to unseen lengths
- Token embedding + positional encoding (added, not concatenated) = the tensor the transformer processes

**How concepts were taught:**
- **"Integers aren't meaningful" hook:** Token IDs for "cat" (2364), "dog" (8976), "the" (464). |2364-464| = 1900 but |2364-8976| = 6612 — integer says "cat" is closer to "the" than to "dog." IDs are arbitrary BPE merge-order indices, not meaningful coordinates.
- **Pipeline recap:** Visual showing text -> BPE -> integer IDs -> ??? -> model. "What fills the ??? box?"
- **One-hot as naive approach:** 5-word vocabulary example. Two GradientCards (rose) for the two problems. Problem 2 includes inline SVG showing 3 tokens at coordinate axis tips forming equilateral triangle, all at distance sqrt(2). Geometric/spatial modality.
- **One-hot saving grace (row selection):** Worked example with 3 tokens, 4-dim embeddings in a table. one_hot(i) x W = W[i,:]. BlockMath formula. Key insight: nn.Embedding skips the sparse vector and indexes directly.
- **nn.Embedding code:** Create embedding(50000, 768), index with tensor([2364]), show equivalence with embedding.weight[2364]. Dictionary analogy: definitions learned, not written by humans. Start with random noise, training refines until similar tokens have similar definitions.
- **Embeddings are learned, not fixed:** Code showing requires_grad=True and model.parameters(). Before/after comparison: embed("the") and embed("a") random at init (rose "not similar"), nearly identical after training (emerald "similar"). WarningBlock: "Embeddings are the first layer of the model, not a preprocessing step."
- **Polysemy warning:** Amber box. "Bank" (river) and "bank" (money) get the same embedding. Context comes from attention (Module 4.2).
- **Parameter count check:** nn.Embedding(50000, 768) = 38.4M parameters. Callback to tokenization lesson's "more entries = more parameters." Follow-up: double vocab = double params, halve embed_dim = halve params.
- **Embedding space explorer widget:** Interactive 2D scatter with ~120 tokens, cluster buttons, search, hover. TryThisBlock: king/queen, numbers, happy/sad, man/woman direction, function words cluster.
- **Bag-of-words problem:** "Dog bites man" vs "Man bites dog" — same embeddings, different meaning. GradientCard showing identical sets. ComparisonRow: CNN spatial locality (filter slides, position implicit) vs embeddings without position (all interchangeable). CNN recap paragraph refreshes sliding-filter mechanism from What Convolutions Compute (Series 3).
- **Sinusoidal PE:** Four requirements derived before formula. Formula with BlockMath. Clock analogy: second hand (fine), minute hand (coarse), hour hand (broadest). PositionalEncodingHeatmap widget: left columns rapid oscillation (high freq, small denominator), right columns slow change (low freq, huge denominator).
- **Addition, not concatenation:** Formula: input_i = embedding(token_i) + PE(i). Same dimension d. WarningBlock: concatenation would double dimension.
- **Learned PE:** Code: nn.Embedding(2048, 768) indexed by positions 0..seq_len-1. Simpler, used by GPT-2. Tradeoff: can't generalize to unseen lengths. RoPE mentioned for Module 4.3.
- **DNA transfer question:** 4-token vocabulary, train on length 1000, inference on length 5000. Sinusoidal can generalize; learned can't.
- **Notebook exercises:** Create nn.Embedding and verify lookup, prove one-hot equivalence, compute cosine similarity, implement sinusoidal PE from formula, combine embedding + PE into model input, explore pretrained GPT-2 embeddings (stretch).
- **Module completion block:** Lists all 5 achievements across Module 4.1. Points to Module 4.2 (Attention).
- **Symmetry aside:** Violet box noting embedding maps V->d, output layer maps d->V. Everything interesting happens in d-dimensional space.

**Mental models established:**
- "Token embedding + positional encoding = the model's input" — completes the full input pipeline: text -> tokens -> IDs -> embeddings + position -> tensor
- "Embeddings are a learned dictionary" — definitions start as random noise, training refines until similar tokens have similar definitions
- "One-hot times a matrix = row selection" — embedding lookup IS matrix multiplication, just without the sparse vector
- "Without position, embeddings are a bag of words" — order must be injected explicitly
- "Sinusoidal PE is like a clock with many hands" — different frequencies capture different granularities of position

**Analogies used:**
- Dictionary with learned definitions (embedding as a lookup table whose "definitions" are refined by training)
- Clock with many hands (sinusoidal PE: second/minute/hour hands capture different position granularities)
- CNN spatial locality contrast (filters slide and encode position implicitly; transformers don't, so position must be injected)

**What was NOT covered (scope boundaries):**
- How the model uses embeddings downstream (attention, Q/K/V) — Module 4.2
- Training embeddings from scratch on a dataset — Module 4.3
- Word2Vec, GloVe, or standalone embedding methods — different paradigm
- Rotary Position Embeddings (RoPE) in detail — MENTIONED, deferred to Module 4.3
- ALiBi or other relative position methods — not mentioned
- Context-dependent representations (attention output) — Module 4.2
- Output layer / unembedding in detail — MENTIONED for symmetry only

**Misconceptions addressed:**
1. "Embeddings are preprocessing (like normalization — compute once, fixed)" — Showed requires_grad=True, appears in model.parameters(). Before/after comparison: random at init, clustered after training. If fixed, clustering couldn't emerge.
2. "Token IDs carry meaning (similar tokens should have similar integers)" — IDs are BPE merge-order indices. |cat-the| < |cat-dog| in integer space. The embedding layer's job is to assign meaning to arbitrary IDs.
3. "One-hot is fine for small vocabularies; embeddings are just an efficiency trick" — Even 3 tokens in one-hot space are all at distance sqrt(2). SVG triangle proof. Dimensionality is one problem but similarity is the deeper issue.
4. "Word order doesn't matter much (bag-of-words is fine)" — "Dog bites man" vs "Man bites dog" with identical embedding sets. Order IS meaning in language.
5. "Positional encoding is like adding a channel (extra feature appended)" — It's ADDED, not concatenated. Same dimension. Position and meaning blended into single representation.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "A language model approximates P(next token \| context)" | what-is-a-language-model | |
| "Autoregressive generation is a feedback loop" | what-is-a-language-model | |
| "Temperature changes sampling, not knowledge" | what-is-a-language-model | |
| "Next-token prediction is a universal training signal" | what-is-a-language-model | |
| "Phone autocomplete is a tiny language model" | what-is-a-language-model | tokenization (phone autocomplete also has a vocabulary) |
| "MNIST softmax (10 classes) -> LM softmax (50K classes)" | what-is-a-language-model | tokenization (vocab size = output classes) |
| "Tokenization defines what the model can see" | tokenization | |
| "BPE is text compression repurposed for tokenization" | tokenization | |
| "The merge table IS the tokenizer" (deterministic encoding) | tokenization | |
| "Architecture encodes assumptions — tokenizer is the first architectural decision" | tokenization (extended from Series 3) | embeddings-and-position (embedding dimension, addition vs concatenation are design choices) |
| "Token embedding + positional encoding = the model's input" (complete pipeline: text -> tokens -> IDs -> embeddings + position -> tensor) | embeddings-and-position | |
| "Embeddings are a learned dictionary" (definitions start random, training refines) | embeddings-and-position | |
| "One-hot times a matrix = row selection" (nn.Embedding is the efficient version) | embeddings-and-position | |
| "Without position, embeddings are a bag of words" | embeddings-and-position | |
| "Sinusoidal PE is like a clock with many hands" (different frequencies for different granularities) | embeddings-and-position | |
| CNN spatial locality contrast (position implicit in CNNs, explicit in transformers) | embeddings-and-position (extended from Series 3) | |
