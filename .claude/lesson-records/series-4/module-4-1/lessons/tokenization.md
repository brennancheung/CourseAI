# Lesson: Tokenization (4.1.2)

**Module:** 4.1 — Language Modeling Fundamentals
**Position:** Lesson 2 of 3
**Type:** BUILD (hands-on notebook)
**Notebook:** `4-1-2-tokenization.ipynb`

---

## Phase 1: Orient — Student State

### Relevant Concepts

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Language modeling as next-token prediction | DEVELOPED | what-is-a-language-model (4.1.1) | Core framing. Student knows the model outputs a probability distribution over a vocabulary of tokens. Directly motivates "what IS a token?" |
| Probability distribution over vocabulary (softmax over 50K+ tokens) | INTRODUCED | what-is-a-language-model (4.1.1) | Student has seen the bar chart over ~15 candidate tokens. Knows the vocabulary can be 50,000+ items. Has NOT thought about what defines those items. |
| Autoregressive generation (sample, append, repeat) | DEVELOPED | what-is-a-language-model (4.1.1) | Student understands the generation loop. Tokenization defines what units enter and exit that loop. |
| "Tokens are not necessarily words" | MENTIONED | what-is-a-language-model (4.1.1) | Lesson 1 used "token" carefully and noted tokens can be words, pieces of words, or punctuation. But no explanation of HOW tokenization works. |
| Cross-entropy loss | INTRODUCED | datasets-and-dataloaders (2.2.1) | L = -log(p_correct). Relevant because vocabulary size directly affects the classification problem: more tokens = harder prediction task. |
| Softmax for classification | INTRODUCED | datasets-and-dataloaders (2.2.1) | Student understands softmax converts logits to probabilities for classification. The vocabulary IS the class set. |
| Python programming (comfortable) | APPLIED | Series 2 notebooks | Student has implemented training loops, custom datasets, model classes in Python/PyTorch. Can write algorithmic code. |
| Dictionary/hash table data structures | APPLIED | Prior programming experience | Student is a software engineer. Merge tables, vocabulary lookups, encoding/decoding are familiar programming patterns. |

### Mental Models Already Established

- "A language model approximates P(next token | context)" -- the framing that makes "what is a token?" urgent
- "MNIST softmax (10 classes) -> LM softmax (50K classes)" -- vocabulary size as class count; tokenization determines that count
- "Phone autocomplete is a tiny language model" -- grounded daily experience to connect back to
- "Architecture encodes assumptions about data" (Series 3) -- tokenization is a design choice that encodes assumptions about language

### What Was NOT Covered

- How text becomes integers (the actual conversion process)
- Why "token" is not the same as "word"
- Any tokenization algorithm (BPE, WordPiece, SentencePiece)
- Vocabulary construction as a design decision
- How tokenization affects model behavior (arithmetic, multilingual, code)

### Readiness Assessment

The student is well-prepared. They have the motivation ("I know the model predicts over a vocabulary -- but what defines that vocabulary?"), the programming skills for an implementation notebook, and the conceptual foundation (probability over discrete classes). The gap is clean: they have zero knowledge of tokenization algorithms, which is exactly what this lesson teaches.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to understand how text is converted to integer sequences via subword tokenization, and to implement BPE (Byte Pair Encoding) from scratch.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Language modeling as next-token prediction | INTRODUCED | DEVELOPED | what-is-a-language-model | OK | Student needs to know that the model operates on tokens and predicts the next one. Has this at DEVELOPED. |
| Vocabulary as the set of things the model can predict | INTRODUCED | INTRODUCED | what-is-a-language-model | OK | Student knows the model outputs a distribution over a vocabulary. Lesson 1 showed the bar chart. |
| Softmax over discrete classes | INTRODUCED | INTRODUCED | datasets-and-dataloaders (2.2.1) | OK | The link between vocabulary size and output layer size requires understanding softmax classification. |
| Python string manipulation | APPLIED | APPLIED | Programming background | OK | BPE implementation requires splitting strings, counting pairs, replacing substrings. Student is a software engineer. |
| Dictionaries / frequency counting | APPLIED | APPLIED | Programming background | OK | Core data structure for BPE merge table and vocabulary. |

**Gap resolution:** No gaps. All prerequisites are met.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Tokens are words" (split on spaces) | Daily experience: we think of language as words separated by spaces. Lesson 1 used the word "token" and showed examples that happened to be whole words. | The word "tokenization" itself: a word-level tokenizer needs it in the vocabulary as a single entry. If not seen in training, it's OOV (out-of-vocabulary) -- completely invisible to the model. Meanwhile, a BPE tokenizer splits it into ["token", "ization"] and handles it fine. | Hook section -- this is the motivating problem. Revisited during BPE explanation. |
| "Character-level tokenization is simplest, so it should work fine" | Characters are the atomic units of text. If you tokenize by character, you never have OOV problems. Seems like the clean solution. | The sentence "The cat sat on the mat" is 27 characters. The model must learn that "T-h-e" means "the" from scratch. Sequences become extremely long (a 500-word article becomes ~2500 tokens), making the model much slower and harder to train. The model wastes capacity learning spelling. | Section comparing character vs word vs subword -- character is presented first as the "obvious" solution, then its problems are felt. |
| "A bigger vocabulary is always better" | More vocabulary entries = more expressiveness = better, right? Like having a bigger dictionary. | A vocabulary of 1 million entries means the output layer has 1 million classes. The embedding matrix alone would be enormous (1M x 768 = 768M parameters just for embeddings). Each token is seen less often during training, so rare token embeddings are poorly learned. There's a sweet spot. | Elaboration section, after BPE is understood. Connect to "50K is not arbitrary." |
| "Tokenization is a simple preprocessing step that doesn't affect the model" | It happens before the model, so it's just formatting. Like converting uppercase to lowercase. | Ask GPT-4 "How many r's in strawberry?" -- it often gets it wrong because "strawberry" is tokenized as ["straw", "berry"], so the model never sees the individual letters. Arithmetic is hard because "42" might be one token but "427" is ["42", "7"]. The tokenizer determines what the model can "see." | Elaborate section with real-world tokenization artifacts. The SolidGoldMagikarp example. |
| "Every time you run a tokenizer, it could produce different results" | BPE has a learning/training phase, so maybe it's stochastic? | BPE tokenization is fully deterministic once the merge table is fixed. Given the same merge table and input, you always get the same tokens. The training phase (learning merges) uses frequency statistics, but the encoding phase is a fixed lookup. | During BPE implementation -- make the deterministic nature explicit. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Three ways to tokenize "The cat sat on the mat" (character / word / BPE) | Positive | Show the spectrum of granularity and introduce the tradeoffs concretely | Simple sentence everyone can parse. Character version is visibly long (27 tokens). Word version is clean (6 tokens) but fragile. BPE is the middle ground. Sets up the "why not just use words/characters?" question. |
| BPE on a tiny corpus: "low low low low low" / "lower lower newest newest" | Positive | Walk through the BPE merge algorithm step by step with real numbers | Small enough to trace by hand. Shows merge pairs being identified, counted, and applied iteratively. The student can verify each step. Classic example from the original Sennrich et al. paper. |
| OOV failure with word-level: "ChatGPT" not in vocabulary | Negative | Show why word-level tokenization breaks on novel words | "ChatGPT" wasn't in any training vocabulary in 2020. A word-level tokenizer maps it to [UNK], losing all information. BPE splits it into recognizable pieces ["Chat", "G", "PT"]. Makes the OOV problem visceral with a real, familiar word. |
| "strawberry" tokenized as ["straw", "berry"] -- letter counting fails | Positive (real-world) | Show how tokenization creates blind spots in model behavior | Connects to a real phenomenon the student has likely encountered (LLMs failing at letter counting). Makes tokenization feel consequential, not academic. |
| Tokenizing code: `print("hello")` vs natural language | Stretch | Show how tokenization varies across domains | Code has different statistical patterns than English prose. Whitespace matters. Parentheses and quotes need their own tokens. Shows why different tokenizers exist for different domains (and why code-specific models use code-trained tokenizers). |

---

## Phase 3: Design

### Narrative Arc

The student just learned that language models predict the next token -- but what IS a token? They saw a bar chart of candidate next tokens and a vocabulary of 50,000 items, but nobody explained where that vocabulary comes from. This lesson starts from a concrete problem: the model needs integers, not text. You could split on characters (too granular, sequences explode in length), or on words (too coarse, any word not in the vocabulary is invisible). Neither works well. Byte Pair Encoding is the elegant middle ground: start from characters and iteratively merge the most common pairs until you reach your target vocabulary size. The student builds this algorithm from scratch in the notebook, gaining hands-on understanding of how every modern LLM converts text into the integer sequences it actually processes. By the end, the student sees that tokenization is not a boring preprocessing step -- it's a design decision that determines what the model can see, what it struggles with (arithmetic, spelling, multilingual text), and even produces bizarre failure modes (the SolidGoldMagikarp problem).

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | Three-way comparison: character / word / BPE tokenization of the same sentence, showing token counts and boundaries | Tokenization is abstract until you see the same text split three different ways. The visual comparison makes the tradeoff immediately obvious -- character is too long, word is too fragile, BPE is the middle ground. |
| Visual | Interactive BPE widget where the student types text and sees it tokenized step by step, with merge operations highlighted | BPE is an iterative algorithm. Watching merges happen in sequence -- pair counts updating, most frequent pair getting merged, token boundaries shifting -- builds procedural understanding that reading the algorithm description alone cannot. |
| Symbolic / Code | BPE implementation in the notebook: count pairs, find most frequent, merge, repeat | The student is a programmer. Implementing the algorithm makes it concrete and permanent. Each function is small enough to understand, and the student can trace through their own code. |
| Verbal / Analogy | "Text compression" framing: BPE is like learning abbreviations for common phrases. "th" + "e" -> "the" is like learning that "btw" means "by the way" -- you compress frequent patterns into single symbols | Connects to familiar concept (abbreviations/compression). Makes the iterative merge process intuitive: you keep abbreviating the most common thing until you hit your budget. |
| Concrete example (real-world) | Tokenization artifacts: "strawberry" splitting, arithmetic difficulty, SolidGoldMagikarp | These make tokenization consequential. Without them, the student might think "ok, text becomes numbers, moving on." These examples create the "wait, THAT's why LLMs can't count letters?!" moment. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. Subword tokenization as a concept (the idea that tokens are neither characters nor words)
  2. BPE algorithm (the specific merge procedure)
  3. Vocabulary size as a design tradeoff (stretch concept, lighter treatment)
- **Previous lesson load:** STRETCH (what-is-a-language-model introduced language modeling, autoregressive generation, temperature, probability distributions over vocabulary -- high novelty, new domain)
- **This lesson's load:** BUILD -- appropriate. One core algorithm (BPE) with hands-on implementation. The student builds understanding through doing. The concepts are concrete and procedural, not abstract. Coming after a STRETCH lesson, this gives the student something tangible to build.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|------------|
| Vocabulary as the set of things the model predicts (4.1.1) | "Remember the bar chart of 50,000 candidate tokens? Tokenization is what defines those 50,000 items. The tokenizer builds the vocabulary." |
| MNIST softmax over 10 classes -> LM softmax over 50K classes (4.1.1) | "The vocabulary size IS the number of output classes. A vocabulary of 50,000 means the model's output layer has 50,000 neurons. Tokenization determines that number." |
| Cross-entropy loss over classes (2.2.1) | "The model computes cross-entropy loss over the vocabulary. Bigger vocabulary = more classes = harder classification problem at each position." |
| Architecture encodes assumptions about data (3.1) | "Just like choosing convolutions encodes the assumption that nearby pixels matter, choosing a tokenizer encodes assumptions about language -- what units are meaningful, how to handle novel words, what granularity of patterns the model should learn." |
| Transfer learning: base model trained on one dataset, adapted to another (3.2) | Forward reference: "The tokenizer is trained on a corpus BEFORE the model trains. It's a fixed preprocessing decision. You can't change the tokenizer without retraining the model." |

**Potentially misleading prior analogies:** None identified. The "phone autocomplete" analogy from lesson 1 is fine -- phone autocomplete also has a vocabulary (its suggestion list), and that vocabulary was built from some process (usage data).

### Scope Boundaries

**This lesson IS about:**
- Why text needs to be converted to integers
- Character-level, word-level, and subword tokenization tradeoffs
- The BPE algorithm: training (learning merges) and encoding (applying merges)
- Implementing BPE from scratch in the notebook
- How tokenization affects model behavior (artifacts, failure modes)
- Vocabulary size as a design choice

**This lesson is NOT about:**
- Other subword algorithms in detail (WordPiece, Unigram/SentencePiece) -- MENTIONED only, not implemented
- Byte-level BPE (GPT-2's variant) -- MENTIONED as "BPE applied to bytes instead of characters"
- How embeddings work (Lesson 3)
- Training a language model (Module 4.3)
- Implementing a production tokenizer (we build a minimal BPE, not a fast one)
- Unicode handling, normalization, or encoding details

**Target depths:**
- Subword tokenization concept: DEVELOPED
- BPE algorithm: APPLIED (implemented from scratch)
- Character vs word tradeoffs: DEVELOPED
- Vocabulary size effects: INTRODUCED
- Tokenization artifacts / failure modes: INTRODUCED
- WordPiece, SentencePiece: MENTIONED

### Lesson Outline

1. **Context + Constraints**
   - "This lesson: how text becomes integers. You'll implement BPE from scratch."
   - Scope: we build a minimal BPE tokenizer. Not production-grade, not fast, but correct.
   - By the end: you can tokenize any text and explain exactly what happened and why.

2. **Hook — The model can't read**
   - Type: Problem reveal
   - "Your language model from Lesson 1 predicts over a vocabulary of integers. But you type text. Something has to convert 'The cat sat on the mat' into [464, 3797, 3332, 319, 262, 2603]. What is that something, and why those numbers?"
   - Show a real tokenizer output (e.g., tiktoken on a sentence) -- mysterious integers.
   - "By the end of this lesson, you'll build the algorithm that produces these numbers."

3. **Explain: The Obvious Solutions (and why they fail)**
   - **Character-level tokenization**
     - Split text into individual characters. Vocabulary = {a, b, c, ..., z, A, B, ..., Z, 0-9, punctuation}. ~100 items.
     - Pro: tiny vocabulary, no OOV problem ever.
     - Con: "The cat sat on the mat" = 27 tokens. Sequences explode. Model must learn spelling from scratch. Slow (attention scales with sequence length).
     - Example: character-tokenized sentence with token count.
   - **Word-level tokenization**
     - Split on spaces/punctuation. Vocabulary = all words seen in training.
     - Pro: each token carries meaning. Short sequences.
     - Con: "ChatGPT" not in vocabulary -> [UNK]. Every misspelling, new word, foreign word is invisible. Vocabulary size explodes (English has ~170K words, plus inflections, compounds, names...).
     - Negative example: OOV with "ChatGPT" mapped to [UNK].
   - **The tradeoff:** Small vocabulary = long sequences. Large vocabulary = sparse coverage. We need a middle ground.

4. **Explain: Subword Tokenization — The Key Insight**
   - "What if tokens were PIECES of words? Common words stay whole. Rare words get split into recognizable pieces."
   - "tokenization" -> ["token", "ization"]. The model has never seen "tokenization" as a whole, but it knows "token" and "ization" (which appears in "organization", "civilization", "authorization"...).
   - This is the core insight: subword tokenization gives you compact sequences (like word-level) AND coverage of novel words (like character-level).
   - Compression analogy: "It's like learning abbreviations. Frequent patterns get their own symbol."

5. **Check: Predict-and-verify**
   - "If a BPE tokenizer was trained on English Wikipedia, which of these would likely be a SINGLE token: 'the', 'tokenization', 'xyzzy'? Why?"
   - Answer: "the" (extremely common), "tokenization" might be split, "xyzzy" almost certainly split to characters.

6. **Explain: BPE Algorithm — Training Phase**
   - Start with character-level vocabulary.
   - Count all adjacent pairs in the corpus.
   - Merge the most frequent pair into a new token. Add it to the vocabulary.
   - Repeat until you reach your target vocabulary size.
   - Step-by-step walkthrough with tiny corpus: "low low low low low lower lower newest newest"
   - Show each merge iteration with pair counts, selected merge, updated corpus.
   - Key insight: common subwords emerge naturally from frequency statistics. "th" + "e" merges early because "the" is frequent.

7. **Explore: Interactive BPE Widget**
   - Student types text into input field.
   - Widget shows the current tokenization with color-coded token boundaries.
   - "Step" button applies one BPE merge at a time, showing: which pair was merged, the updated token list, the current vocabulary size.
   - "Run all" shows the final tokenization.
   - Try this: tokenize "unhappiness", "ChatGPT", "print('hello')", a sentence in another language.

8. **Practice: Implement BPE in the Notebook**
   - Guided implementation (scaffolded):
     - `get_pair_counts(tokens)` -- count adjacent pairs
     - `merge_pair(tokens, pair, new_token)` -- replace all occurrences of a pair
     - `train_bpe(corpus, num_merges)` -- the training loop
     - `encode(text, merges)` -- apply learned merges to new text
     - `decode(token_ids, vocab)` -- convert back to text
   - Student runs BPE training on a small text corpus, examines the merge table, and tokenizes new sentences.

9. **Elaborate: Tokenization Artifacts — Why This Matters**
   - **"strawberry" problem:** tokenized as ["straw", "berry"]. The model never sees the individual letters r-r-r. This is why LLMs struggle with "How many r's in strawberry?"
   - **Arithmetic:** "42" might be one token, "427" might be ["42", "7"]. The model doesn't see digits consistently.
   - **SolidGoldMagikarp:** A Reddit username that appeared in training data but whose tokens were never seen in normal text during fine-tuning. Prompting GPT with this token caused bizarre behavior. Tokenization creates the vocabulary, and some vocabulary entries can become "glitch tokens."
   - **Multilingual inequality:** BPE trained primarily on English compresses English efficiently (1 token per common word) but fragments other languages (multiple tokens per word). Same sentence costs 2-5x more tokens in Korean or Arabic.
   - Key takeaway: "Tokenization is not neutral preprocessing. It's a design decision that shapes what the model can and cannot do."

10. **Check: Transfer question**
    - "A language model trained with a 32K vocabulary consistently fails at multi-digit multiplication. A researcher proposes switching to a character-level tokenizer to fix this. Would this help? What tradeoff would it introduce?"
    - Expected reasoning: Character-level would let the model see individual digits (helps arithmetic), but sequences become much longer (slower, harder to learn long-range patterns). The real solution is likely architectural (scratchpad, chain-of-thought), not tokenization alone.

11. **Summarize**
    - The model needs integers. Tokenization converts text to integer sequences.
    - Character-level: tiny vocabulary, very long sequences. Word-level: short sequences, can't handle novel words.
    - BPE: start from characters, iteratively merge the most frequent pair. Common words stay whole, rare words split into recognizable pieces. The sweet spot.
    - You built a BPE tokenizer from scratch. The merge table IS the tokenizer.
    - Tokenization defines the vocabulary. The vocabulary defines what the model can predict. This is not neutral preprocessing.

12. **Next Step**
    - "You can now convert text to integer sequences. But integer 464 is just an index into a table -- it doesn't carry any meaning yet. Next lesson: how those integers become rich vector representations the model can actually compute with. That's embeddings."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative + 1 stretch)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2-3 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings that would leave the student completely lost, but one finding borders on critical (forward-referencing an untaught concept in a way that could confuse) and three improvement findings that would meaningfully strengthen the lesson.

### Findings

#### [CRITICAL] — SolidGoldMagikarp section references "embedding" before it is taught

**Location:** Section 8, "Glitch tokens: SolidGoldMagikarp" paragraph
**Issue:** The text says "a token that exists in the vocabulary but whose embedding was never properly trained" and "its embedding is essentially random noise." The student has not learned what an embedding is. Embeddings are explicitly out of scope for this lesson and are the subject of Lesson 3. The scope boundaries section states: "NOT: how embeddings work -- that's the next lesson." Yet the SolidGoldMagikarp explanation depends on the student understanding that each vocabulary entry has an associated learned vector (embedding) that gets trained through exposure.
**Student impact:** The student encounters a term they have no mental model for. They cannot fully understand why a token being in the vocabulary but absent from training data is a problem, because the mechanism (poorly trained embedding vector) is opaque. The student either (a) glosses over the explanation without understanding, (b) forms an incorrect mental model of what "embedding" means, or (c) loses confidence because they feel they missed something.
**Suggested fix:** Reframe the explanation without using "embedding." Instead, describe the issue in terms the student already has: "The model needs to learn what each vocabulary entry means by seeing it used in many contexts during training. If a token appears in the vocabulary but barely appears in the training text, the model has no data to learn what it means. Its internal representation for that token is essentially random -- it was never trained. Prompting GPT with this token caused bizarre, erratic behavior because the model had no meaningful representation of it." This preserves the insight while staying within scope. Lesson 3 can then revisit this as "remember the glitch token problem? Now you know the mechanism: the embedding vector was never updated."

#### [IMPROVEMENT] — Vocabulary size section uses "embedding matrix" and "768-dimensional embeddings" without prerequisite knowledge

**Location:** Section 10, "50,000 tokens is not arbitrary"
**Issue:** The text says "the embedding matrix grows. A vocabulary of 100K with 768-dimensional embeddings is 76.8 million parameters just for the embedding table" and "rare tokens have poorly trained embeddings." This leans on embedding knowledge that comes in Lesson 3. While less central than the SolidGoldMagikarp case, it asks the student to reason about parameter counts in a representation they haven't been taught.
**Student impact:** The student can still follow the high-level argument (bigger vocabulary = some cost, smaller vocabulary = longer sequences). But the specific argument about parameter counts and embedding quality is opaque. They take it on faith rather than understanding.
**Suggested fix:** Reframe in terms the student has: "Bigger vocabulary: the model's output layer has more classes (remember MNIST 10 -> LM 50K?), and each vocabulary entry needs its own learned representation. A vocabulary of 100K means 100K representations to train. Rare tokens barely appear in training data, so their representations are poor. Smaller vocabulary: sequences are longer, but each token is seen more often and better learned." The specific 768-dimensional / 76.8M calculation can be deferred to Lesson 3, where it will land with full understanding.

#### [IMPROVEMENT] — BPE walkthrough does not explain tie-breaking

**Location:** Section 5, BPE step-by-step walkthrough, Steps 3-4
**Issue:** After merging "low" (step 2), multiple pairs have count 2: (e, r), (e, w), (w, e), (e, s), (s, t). The walkthrough shows (e, r) being merged first without explaining why it was chosen over (e, w) or (e, s). The student who is carefully tracing the algorithm (as intended) will notice that multiple pairs tie at count 2 and wonder how the choice is made.
**Student impact:** A careful student counts the pairs themselves, notices the tie, and wonders if they're missing something. This creates a small confusion that could undermine confidence in their understanding of the algorithm. An inattentive student misses it entirely and moves on.
**Suggested fix:** Add a brief note after step 2 or at step 3: "Multiple pairs now tie at count 2 (e+r, e+s, e+w...). When there's a tie, the algorithm just picks one -- the choice is implementation-dependent and doesn't affect the final result much. We'll go with (e, r)." This one sentence prevents the confusion and reinforces that BPE is about frequency, not about clever ordering.

#### [IMPROVEMENT] — Three-way comparison is incomplete: BPE version of "The cat sat on the mat" is never shown

**Location:** Section 3, "The Obvious Solutions"
**Issue:** The planning document calls for "Three ways to tokenize 'The cat sat on the mat' (character / word / BPE)." The lesson shows the character version (22 tokens) and the word version (6 tokens), but never shows the BPE version of the same sentence. The subword section (Section 4) uses "tokenization" as its example, not "The cat sat on the mat." The student sees two failed approaches on one sentence and then a different example for the working approach. The planned three-way comparison that makes the tradeoff "immediately obvious" is incomplete.
**Student impact:** The student doesn't get to see the same sentence tokenized all three ways side by side. The tradeoff is understood intellectually but the concrete "aha" of seeing one sentence at 22 / 6 / ~8 tokens is missing. The ComparisonRow partially compensates with summary bullets, but a direct BPE tokenization of the same sentence would close the loop more satisfyingly.
**Suggested fix:** After the subword explanation in Section 4, add a brief example showing "The cat sat on the mat" under BPE: something like ["The", " cat", " sat", " on", " the", " mat"] (6 tokens, similar to word-level for this common sentence). Then note: "For this simple sentence, BPE produces similar results to word-level. The difference shows up with rare words..." This completes the three-way comparison and sets up why subword's advantage is about coverage, not common text.

#### [POLISH] — Widget stopping condition differs from lesson's BPE algorithm description

**Location:** Section 5 (algorithm description) vs Section 6 (widget behavior)
**Issue:** The lesson describes the BPE stopping condition as "Repeat from step 2 until you reach your target vocabulary size." But the widget stops when no pair appears 2+ times (the `getMostFrequentPair` function returns null if `bestCount < 2`). These are different stopping conditions. The lesson never explains the widget's stopping condition.
**Student impact:** Minor. The widget shows "Done! No more pairs appear 2+ times" when it stops, which is self-explanatory. But a careful student might notice that the algorithm says "until target vocabulary size" while the widget stops at a different point. This creates a small inconsistency.
**Suggested fix:** Either (a) add a brief note below the widget: "The widget stops when no pair appears more than once -- there's nothing left to compress. In practice, BPE training stops at a target vocabulary size (e.g., 50,000 merges), which is reached long before running out of pairs on a large corpus." Or (b) add a vocab size target input to the widget. Option (a) is simpler and sufficient.

#### [POLISH] — The "Encoding" phase of BPE (applying merges to new text) is described in the notebook section but not in the lesson explanation

**Location:** Section 5 (BPE algorithm) and Section 7 (notebook exercise)
**Issue:** The lesson carefully explains BPE training (learning merges) but doesn't explicitly explain the encoding phase (given a merge table, apply merges to new text in priority order). The encoding phase appears only as a function signature in the notebook section (`encode(text, merges)`). The distinction between training and encoding is mentioned in the section header ("The algorithm behind modern tokenizers") and in the aside ("Once the merge table is learned, encoding is fully deterministic"), but the actual encoding procedure is never walked through.
**Student impact:** The student understands training well but might not immediately grasp how a trained BPE tokenizer is applied to new, unseen text. The notebook will fill this gap during implementation, so it's not blocking. But the lesson text itself leaves the encoding phase underexplained.
**Suggested fix:** After the training walkthrough, add a brief paragraph: "Once you have the merge table, encoding new text follows the same merges in priority order. Given the text 'lowest' and the merge table [l+o -> lo, lo+w -> low, ...], you start with characters [l, o, w, e, s, t], apply merge #1 (l+o -> lo), then merge #2 (lo+w -> low), then check for further merges. The merge table IS the tokenizer." This would take 3-4 sentences and close the gap.

### Review Notes

**What works well:**
- The hook is strong. "The Model Can't Read" creates genuine curiosity and the mystery integers are compelling.
- The character/word/subword progression follows concrete-before-abstract beautifully. The student feels the failure of the obvious approaches before being given the solution.
- The BPE walkthrough with the "low lower newest" corpus is effective. The step-by-step cards with highlighted tokens make the merge process visible.
- The interactive widget is a genuine asset. Letting the student type their own text and watch merges happen builds intuition that static examples cannot.
- The tokenization artifacts section (strawberry, arithmetic, multilingual, SolidGoldMagikarp) makes the lesson feel consequential rather than "just preprocessing." This is one of the lesson's strongest sections.
- The mental model echo at the end ("Architecture encodes assumptions about data") is an elegant callback that positions tokenization within the broader course narrative.
- The compression/abbreviation analogy ("btw" = "by the way") is accessible and accurate.
- All five planned modalities are present and genuinely different.

**Pattern to watch:**
- The lesson occasionally reaches forward to Lesson 3 concepts (embeddings). This happened in both the SolidGoldMagikarp section and the vocabulary size section. While forward references can be effective when clearly marked ("you'll see this in detail next lesson"), using untaught concepts in explanations is different -- it asks the student to reason with tools they don't have. The fix is straightforward: reframe those sections in terms the student currently has (vocabulary entries need to be learned from training data, bigger vocabulary = more things to learn) and let Lesson 3 connect the mechanism (embeddings) later.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All 6 findings from iteration 1 have been properly addressed. The lesson is pedagogically sound, well-scoped, and effective. Two minor polish items noted below, neither of which warrants a revision cycle.

### Findings

#### [POLISH] — BPE three-way comparison uses leading-space convention without explanation

**Location:** Section 4, "Subword Tokenization," BPE tokenization of "The cat sat on the mat"
**Issue:** The BPE tokens are shown as `[The] [␣cat] [␣sat] [␣on] [␣the] [␣mat]` — using the leading-space convention where whitespace is attached to the following token. The character-level example shows spaces as separate characters, and the word-level example strips spaces entirely. The BPE example introduces a third convention (leading-space) without noting it. A careful student might wonder why the space moved from being its own character to being glued to the next token.
**Student impact:** Very minor. The student sees the open-box glyph and can infer it represents a space. The explanatory note below says "BPE produces a similar result to word-level (6 tokens)" which is accurate. The leading-space convention is a real aspect of how BPE tokenizers work (GPT-2, tiktoken), so showing it is not wrong. But the student has no explanation for why BPE treats spaces this way.
**Suggested fix:** Optional. Could add a parenthetical like "(notice spaces attach to the following token — this is how most BPE tokenizers handle word boundaries)" or simply leave as-is. The notebook will reinforce this when the student implements their own tokenizer.

#### [POLISH] — BPE walkthrough uses underscore word-boundary convention while the widget uses actual spaces

**Location:** Section 5 (walkthrough StepCards) vs Section 6 (BpeVisualizer widget)
**Issue:** The walkthrough represents word boundaries as `_` (rendered as ␣), which is the convention from the original Sennrich et al. BPE paper. The widget, by contrast, operates on raw character sequences including actual space characters. These are two different conventions for the same concept. When the student steps through the widget with "low low low low low lower lower newest newest" (the same corpus as the walkthrough), the merge sequence will differ because the widget treats spaces as regular characters that participate in pair counting.
**Student impact:** Minimal. The walkthrough and widget serve different purposes (the walkthrough shows a clean algorithmic trace; the widget lets the student explore freely). Most students won't type the exact same corpus. But a very careful student who does type the walkthrough corpus into the widget would see different merge behavior and might be confused.
**Suggested fix:** Not worth fixing. The pedagogical value of both representations is high, and unifying them would either make the walkthrough harder to read (if using actual spaces) or make the widget less realistic (if using underscore boundaries). A footnote could be added, but would probably add more noise than clarity.

### Iteration 1 Fix Verification

| Iteration 1 Finding | Fix Applied | Verification |
|---------------------|-------------|-------------|
| CRITICAL: SolidGoldMagikarp references "embedding" | Rewrote to use "learned meaning" / "internal representation" / "the model has no data to learn what it means" | Verified. Zero mentions of "embedding" in the SolidGoldMagikarp section. Explanation is self-contained using only concepts the student has. |
| IMPROVEMENT: Vocab size section uses "embedding matrix" / "768-dimensional" | Rewrote to use "learned representation" / "output layer has more classes (remember MNIST 10 -> LM 50K?)" | Verified. No embedding terminology. The MNIST callback works well. Parameter count argument replaced with "more entries = more parameters" which is followable. |
| IMPROVEMENT: Tie-breaking not explained in BPE walkthrough | Added italic note between steps 2 and 3 explaining that multiple pairs tie at count 2 | Verified. The note says "Multiple pairs now tie at count 2 (e+r, e+s, e+w...). When there's a tie, the algorithm just picks one — the choice is implementation-dependent and doesn't affect the final result much." Clear and correctly placed. |
| IMPROVEMENT: Three-way comparison missing BPE version | Added BPE tokenization of "The cat sat on the mat" in Section 4 with explanatory note | Verified. Shows 6 BPE tokens with violet styling. Note explains that for common sentences BPE looks like word-level, and the advantage shows with rare words. Completes the three-way comparison. |
| POLISH: Widget stopping condition unexplained | Added note below widget explaining the stopping condition difference | Verified. Text says "The widget stops when no pair appears more than once — there's nothing left to compress. In practice, BPE training stops at a target vocabulary size..." Correctly explains the discrepancy. |
| POLISH: Encoding phase not explained in lesson text | Added paragraph after walkthrough explaining encoding with "lowest" example | Verified. Text walks through encoding "lowest" using the merge table: start with characters, apply merges in priority order. "The merge table IS the tokenizer" is reinforced. |

### Review Notes

**What works well (confirmed from iteration 1, reinforced by fixes):**
- The hook remains strong and the narrative arc is clean: problem -> failed approaches -> key insight -> algorithm -> implementation -> real-world consequences.
- The three-way comparison is now complete and effective. Seeing character (22 tokens), word (6 tokens), and BPE (6 tokens for common text, splits rare words) on the same sentence makes the tradeoff concrete.
- The SolidGoldMagikarp section is now cleanly explained without forward-referencing embeddings. The reasoning chain (token in vocabulary + absent from training data = untrained representation = bizarre behavior) is followable with the student's current knowledge.
- The vocabulary size section is now grounded in concepts the student has (MNIST classes, output layer size) rather than embedding dimensions they don't.
- The tie-breaking note in the walkthrough is well-placed and prevents a potential confidence-undermining confusion for careful students.
- The encoding phase explanation fills a real gap — the student now understands both halves of BPE (training and encoding) before reaching the notebook.
- All five modalities remain present and genuinely different.
- The interactive widget is a genuine asset for building procedural intuition about BPE.
- The tokenization artifacts section makes the lesson feel consequential rather than "just preprocessing."

**Overall assessment:** This lesson is ready to ship. The fixes from iteration 1 were well-executed — each addressed the specific issue without introducing new problems. The lesson teaches its target concepts at the planned depths, stays within scope, addresses all planned misconceptions, and follows pedagogical principles consistently.
