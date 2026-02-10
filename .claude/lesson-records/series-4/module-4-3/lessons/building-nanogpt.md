# Lesson: building-nanogpt

**Module:** 4.3 (Building & Training GPT)
**Position:** Lesson 1 of 4
**Type:** Hands-on (notebook: `4-3-1-building-nanogpt.ipynb`)

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Full GPT architecture end-to-end (token embedding + PE -> N blocks with causal masking -> final LN -> output projection -> softmax) | DEVELOPED | decoder-only-transformers (4.2.6) | The student can trace the complete forward pass on paper with dimension annotations at every stage. They have the GptArchitectureDiagram SVG internalized. This is the blueprint they will now implement. |
| Transformer block as repeating unit (MHA + FFN + 2 residual connections + 2 layer norms, shape-preserving) | DEVELOPED | the-transformer-block (4.2.5) | Student knows the complete block formula: x' = x + MHA(LayerNorm(x)), output = x' + FFN(LayerNorm(x')). They know pre-norm ordering. They know it stacks identically N times. |
| Multi-head attention formula (Concat(head_1,...,head_h) W_O, each head = Attention(XW_Q^i, XW_K^i, XW_V^i)) | DEVELOPED | multi-head-attention (4.2.4) | Full formula with shape annotations. Student did a worked example with d_model=6, h=2, d_k=3. Knows dimension splitting: d_k = d_model/h. |
| Scaled dot-product attention (softmax(QK^T/sqrt(d_k))V) | DEVELOPED | values-and-attention-output (4.2.3) | Complete single-head formula. Student traced by hand with 4 tokens, 3-dim embeddings across three lessons. |
| Causal masking (set upper-triangle to -inf before softmax) | DEVELOPED | decoder-only-transformers (4.2.6) | Student understands why (prevent data leakage), how (mask before softmax), and that it mirrors inference reality. |
| FFN structure (W_2 * GELU(W_1 x + b_1) + b_2, 4x expansion) | DEVELOPED | the-transformer-block (4.2.5) | GPT-2 dimensions: 768->3072->768. GELU activation. Student knows parameter count per block. |
| Layer normalization (per-token feature norm, pre-norm placement) | INTRODUCED | the-transformer-block (4.2.5) | Student knows what it does and where it goes (pre-norm). Has NOT implemented it or explored the formula deeply. |
| Output projection / weight tying | INTRODUCED | decoder-only-transformers (4.2.6) | Student knows output projection maps d_model -> vocab_size and that it shares weights with the embedding matrix (transposed). Has not implemented weight tying. |
| GPT-2 parameter counting (~124.4M) | DEVELOPED | decoder-only-transformers (4.2.6) | Per-component breakdown verified against known figure. Student knows where every parameter lives. |
| Token embeddings (nn.Embedding, 50K x 768) | DEVELOPED | embeddings-and-position (4.1.3) | Implemented in notebook. One-hot x matrix = row selection. |
| Positional encoding (learned, another nn.Embedding indexed by position) | INTRODUCED | embeddings-and-position (4.1.3) | Student knows GPT-2 uses learned positional embeddings. Has not implemented them specifically for GPT. |
| nn.Linear, nn.Embedding, nn.Module | APPLIED | Series 2 (PyTorch Core) | Student has built multiple models using these. Comfortable writing forward() methods, managing parameters. |
| Training loop (forward, loss, backward, step) | APPLIED | Series 2 (Real Data, Practical Patterns) | Student has trained CNNs and simple models. The loop pattern is deeply familiar. |
| Weight initialization (Xavier, He) | INTRODUCED | training-dynamics (1.3) | Student knows initialization matters and has seen Xavier/He. Has not applied it to transformers specifically. |

### Mental Models Already Established

- "The full GPT architecture is assembly, not invention" -- every component is familiar
- "Attention reads, FFN writes" -- complementary roles in the transformer block
- "The residual stream is a shared document" -- each sub-layer reads and annotates
- "Three lenses, one embedding" (W_Q, W_K, W_V) -- three projections from the same input
- "Split, not multiplied" -- multi-head dimension splitting as budget allocation
- nn.Module as the building block pattern (from Series 2)
- "Same training loop, different model" pattern from Series 2 CNN work

### What Was NOT Covered That Is Relevant Here

- How to translate architecture diagrams into class hierarchies (nn.Module composition)
- Weight initialization strategies specific to transformers (scaled initialization for residual projections)
- Config/hyperparameter management for models with many settings
- Generating text from a model (the autoregressive inference loop in code)
- How to organize a multi-file model (though we will keep it in one notebook)

### Readiness Assessment

The student is fully prepared. Every architectural component is at DEVELOPED depth. The PyTorch building blocks are at APPLIED depth. The only new skill is the translation from conceptual understanding to code -- which is precisely the point of this lesson. This is a BUILD lesson: low conceptual novelty, high implementation satisfaction.

---

## Phase 2: Analyze

### Target Concept

**This lesson teaches the student to translate the complete GPT architecture into working PyTorch code, verify each component's shapes, and generate text from the assembled model.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Full GPT architecture | DEVELOPED | DEVELOPED | decoder-only-transformers | OK | Student needs to know what to build. Architecture is the blueprint. |
| Transformer block structure | DEVELOPED | DEVELOPED | the-transformer-block | OK | Must implement MHA + FFN + residual + layer norm as a class. |
| Multi-head attention formula | DEVELOPED | DEVELOPED | multi-head-attention | OK | Must implement the formula with dimension splitting. |
| Scaled dot-product attention | DEVELOPED | DEVELOPED | values-and-attention-output | OK | Core computation inside each head. |
| Causal masking | DEVELOPED | DEVELOPED | decoder-only-transformers | OK | Must implement the mask in the attention computation. |
| FFN structure | DEVELOPED | DEVELOPED | the-transformer-block | OK | Must implement the two-layer network with GELU. |
| Layer normalization | INTRODUCED | INTRODUCED | the-transformer-block | OK | Student knows what it does and where it goes. nn.LayerNorm is a one-liner in PyTorch; no deeper understanding needed for implementation. |
| nn.Module, nn.Linear, nn.Embedding | APPLIED | APPLIED | Series 2 | OK | Core PyTorch building blocks. Student has used these extensively. |
| Weight initialization | INTRODUCED | INTRODUCED | training-dynamics | GAP (small) | Student knows initialization matters but hasn't seen transformer-specific strategies. Brief recap + concrete recipe needed. |
| Autoregressive generation loop | DEVELOPED (conceptual) | DEVELOPED (conceptual) | what-is-a-language-model | GAP (small) | Student understands the concept (sample, append, repeat) but has never implemented it in code. Brief implementation walkthrough needed. |

### Gap Resolution

| Gap | Size | Action |
|-----|------|--------|
| Transformer weight initialization | Small | Brief section (2-3 paragraphs) explaining why default init is problematic for deep transformers + a concrete recipe: Xavier for most linear layers, scaled init (1/sqrt(2N)) for residual projection layers. No separate section -- integrated into the model assembly. |
| Autoregressive generation in code | Small | The generate() method is the payoff section. Student already has the conceptual model (sample, append, repeat from 4.1.1). Translating to code is straightforward: forward pass with context, sample from logits, append token, repeat. 1-2 paragraphs + code. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Implementing a transformer requires advanced PyTorch features I haven't learned (custom autograd, CUDA kernels, etc.)" | Transformers power the most powerful AI systems; surely the code must be exotic. | The entire GPT model uses only nn.Linear, nn.Embedding, nn.LayerNorm, nn.GELU, and nn.Dropout. Count the unique PyTorch operations: 5. All familiar from Series 2. | Hook section -- reveal the "parts list" immediately. |
| "Multi-head attention requires a loop over heads" | The formula shows head_1, head_2, ..., head_h as separate computations. Natural to think of a for-loop. | The batched implementation reshapes (B, T, d_model) to (B, h, T, d_k) and computes all heads simultaneously with one matrix multiply. No loop. The tensor operations ARE the parallelism. | Multi-head attention implementation section. Show the loop version first (correct but slow), then the reshape trick. |
| "Weight initialization is just setting everything to zero or small random values" | In Series 2, default init worked fine for small models. Why would it matter here? | Initialize a 12-block transformer with default PyTorch init. Measure activation standard deviations at each block: they explode or collapse. 12 residual additions compound the variance issue. | Weight initialization section. Concrete: show activation stats with default vs proper init. |
| "The generate() function should use the training forward pass directly" | Forward pass computes next-token logits; generation just calls forward repeatedly. | During generation, you only need logits for the LAST position (the one being generated), not all positions. Using the full training forward pass wastes compute by computing logits for positions you already have. Also: no loss computation, no gradients, temperature/sampling needed. | Generate section. Show the difference between training mode (all positions, loss) and inference mode (last position, sampling). |
| "If I get the architecture right, the shapes will just work out" | The architecture diagram has dimension annotations; follow them and shapes match. | A single mistake in reshape ordering (e.g., swapping h and T dimensions in the multi-head attention view) produces output with correct final shape but completely wrong values. Shape correctness is necessary but not sufficient. | Throughout -- shape verification assertions at every step. Emphasize that correct shapes do not guarantee correct computation. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Building the model bottom-up: Head -> MHA -> FFN -> Block -> GPT | Positive | Show how nn.Module composition creates the full architecture from small, testable pieces. Each class is tiny (5-15 lines). | Matches the bottom-up learning from Module 4.2. The student built understanding bottom-up; now they build code bottom-up. Each class maps 1:1 to a concept they already know. |
| Shape verification at every layer boundary | Positive | Demonstrate that asserting shapes catches bugs before they propagate. Create a dummy input, pass through each component, verify output shape matches expectation. | Practical engineering habit. Transformers have many reshape operations; one wrong transpose produces valid-shape but wrong-value output. The assertion discipline is the skill being taught. |
| Default init vs proper init: activation statistics through 12 blocks | Negative | Show that default initialization causes activation explosion or collapse in a 12-block model. Measure std at each block. | Disproves "initialization doesn't matter" misconception. Concrete numbers make the problem visceral. Then the fix (scaled init) is motivated by the measured problem. |
| Generating text from the untrained model | Positive (payoff) | The "it runs!" moment. Random gibberish, but syntactically it is a working language model. The architecture is correct; it just needs training (Lesson 2). | The emotional reward for the entire lesson. Also demonstrates the generate() function and connects to the autoregressive loop from Module 4.1. |

---

## Phase 3: Design

### Narrative Arc

You have spent six lessons understanding every component of the GPT architecture. You can trace a forward pass on paper, name every projection matrix, count every parameter. But understanding and building are different things. This lesson crosses that gap. You will write a complete GPT model in PyTorch -- every component, from token embedding to output logits -- and by the end, you will type a prompt and watch your model generate text. It will be nonsensical gibberish (the model is untrained), but it will be YOUR gibberish, generated by YOUR architecture. The architecture diagram from Module 4.2 becomes a living, running model. And the surprising thing? The code is not exotic. It is nn.Linear, nn.Embedding, nn.LayerNorm -- the same building blocks you have been using since Series 2. The complexity is in the assembly, not the parts.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Symbolic (code)** | Complete PyTorch implementation: Head, MultiHeadAttention, FeedForward, TransformerBlock, GPT classes. Each with explicit shape comments. | This is an implementation lesson. Code IS the primary modality. Every concept from Module 4.2 is re-expressed as a class. |
| **Visual** | Architecture diagram with code class boundaries drawn on it. Each colored region of the GptArchitectureDiagram mapped to the class that implements it. | Bridges the conceptual diagram (familiar) to the code structure (new). The student sees which class corresponds to which box in the diagram they already know. |
| **Concrete example** | Shape trace with actual numbers: GPT-2 small config (vocab=50257, d_model=768, n_heads=12, n_layers=12, context=1024). Input (B=2, T=64) traced through every layer with explicit shape at each boundary. | Numbers make abstract shapes concrete. The student runs the shape trace in the notebook and sees every intermediate tensor. This is the "worked example" equivalent for implementation. |
| **Intuitive** | The "parts list" reveal: count the unique PyTorch operations used in the entire model (nn.Linear, nn.Embedding, nn.LayerNorm, nn.GELU, nn.Dropout = 5). "You already know all five." | Deflates the intimidation factor. The student realizes the model is assembled entirely from familiar pieces. This is the "of course" feeling applied to implementation. |

### Cognitive Load Assessment

- **New concepts:** 2 (transformer-specific weight initialization recipe, autoregressive generation in code)
- **Previous lesson load:** CONSOLIDATE (decoder-only-transformers was assembly and wrap-up)
- **This lesson's load:** BUILD -- appropriate. The student is translating well-understood concepts into code. Low conceptual novelty. The challenge is craft (getting the code right), not understanding (grasping new ideas). Coming after a CONSOLIDATE lesson, BUILD is the right next step.

### Connections to Prior Concepts

| Prior Concept | How It Connects |
|--------------|----------------|
| GptArchitectureDiagram from decoder-only-transformers | The diagram becomes the class hierarchy. Each colored box maps to a class. "You built this diagram; now you build this code." |
| nn.Module from Series 2 | Same pattern: __init__ defines layers, forward() defines computation. "Same building block, bigger Lego set." |
| Transformer block formula (x' = x + MHA(LN(x)), out = x' + FFN(LN(x'))) | Directly translates to the forward() method of the Block class. Every operator is a line of code. |
| Multi-head dimension splitting (d_k = d_model / h) | The reshape operation in the attention code. The "split, not multiplied" mental model explains the reshape. |
| Causal masking (set upper-triangle to -inf) | torch.tril() to create the mask, masked_fill_() to apply it. The concept is familiar; the PyTorch API is new. |
| Weight initialization from training-dynamics (1.3) | Extends Xavier/He to transformer-specific concerns. The new element: scaling residual projections by 1/sqrt(2N) to prevent activation growth across N blocks. |
| Autoregressive generation loop from what-is-a-language-model (4.1.1) | The generate() method implements the "sample, append, repeat" loop they learned conceptually. Temperature from the TemperatureExplorer widget becomes a parameter in the code. |

**Potentially misleading analogies:** None identified. The prior analogies (architecture diagram, nn.Module pattern, training loop) all transfer directly and accurately to this implementation context.

### Scope Boundaries

**This lesson IS about:**
- Translating the GPT architecture into PyTorch nn.Module classes
- Verifying shapes at every layer boundary
- Weight initialization for transformers (brief, practical recipe)
- Generating text from the assembled (untrained) model
- The GPT-2 small (124M) configuration

**This lesson is NOT about:**
- Training the model (Lesson 2 -- pretraining)
- Dataset preparation (Lesson 2)
- Optimization, learning rate scheduling (Lesson 2)
- GPU utilization, mixed precision, flash attention (Lesson 3)
- Loading pretrained weights (Lesson 4)
- nn.MultiheadAttention (student builds from scratch to cement understanding)
- Dropout (mentioned in code but not explained in depth -- callback to regularization from Series 1)
- Different model sizes (GPT-2 medium/large/XL)
- Batched KV caching for efficient generation (Lesson 3)

**Target depth:** The student reaches APPLIED depth for "implementing the GPT architecture in PyTorch" -- they write every component and verify it runs.

### Lesson Outline

1. **Context + Constraints** -- This is an implementation lesson. You are building the model, not learning the architecture (that was Module 4.2). We will use GPT-2 small config. We will NOT train it (that is Lesson 2). By the end, you will have a running model that generates (random) text.

2. **Hook (demo + parts list reveal)** -- Show the "parts list": nn.Linear, nn.Embedding, nn.LayerNorm, nn.GELU, nn.Dropout. Five operations. "You know all five. Let's assemble them." Then: show the architecture diagram with class boundaries drawn on it. Each region maps to a class you will write. Deflates intimidation while previewing the structure.

3. **Config** -- GPTConfig dataclass with all hyperparameters. Mirror GPT-2 settings. Brief: this is bookkeeping, not a concept. Show config for GPT-2 small AND a tiny debug config for fast iteration.

4. **Build: Single Attention Head** -- Implement Head class. Q, K, V projections as nn.Linear. Causal mask as a buffer (register_buffer). Scaled dot-product attention with masking. Shape comments on every line. **Callback:** "This is the formula from Lesson 3 (values-and-attention-output). Every line maps to a step you traced by hand."

5. **Build: Multi-Head Attention** -- Two approaches shown. First: the explicit loop-over-heads version (matches the formula, correct, readable). Second: the batched reshape version (efficient, used in practice). Show they produce the same output. Output projection W_O. **Callback:** "Split, not multiplied" -- the reshape IS the dimension splitting. **Shape verification:** assert output shape matches input shape.

6. **Build: Feed-Forward Network** -- FeedForward class. Two nn.Linear layers with GELU. 4x expansion. Dropout. "Two lines of code for a component that holds 2/3 of the block's parameters." **Callback:** "Attention reads, FFN writes" -- this is the writer.

7. **Build: Transformer Block** -- Block class combining MHA, FFN, two LayerNorm, two residual connections. The forward() method IS the formula: x = x + self.mha(self.ln1(x)), x = x + self.ffn(self.ln2(x)). **Callback to the-transformer-block:** "Remember the block formula? Here it is in code."

8. **Build: Full GPT Model** -- GPT class. Token embedding + positional embedding, N transformer blocks (nn.ModuleList), final layer norm, output projection (lm_head). Weight tying between embedding and lm_head. Forward pass returns logits with shape (B, T, vocab_size). **Shape trace:** Create dummy input, trace through every component, print shape at each boundary.

9. **Weight Initialization** -- Why default init fails for deep transformers (brief: activation stats through 12 blocks). The recipe: Xavier/normal for most layers, scaled by 1/sqrt(2N) for residual projections (attention output, FFN output). Apply with model.apply() custom function. Show before/after activation statistics. **This addresses the small GAP from Phase 2.**

10. **Check (parameter counting)** -- Count parameters programmatically. Compare to the GPT-2 figure from decoder-only-transformers (~124.4M). If the count matches, the architecture is correct. This is the first verification. **Prediction exercise:** Before running the count, student predicts based on Module 4.2 knowledge.

11. **Generate: Autoregressive Inference** -- Implement generate() method. Forward pass with current context, extract logits for last position only, apply temperature, sample from distribution, append token, repeat. torch.no_grad() context manager. Temperature parameter connects to TemperatureExplorer from Module 4.1. **This addresses the small GAP from Phase 2.** Show generated text: random gibberish. "The architecture works. It just needs training."

12. **Summarize** -- The architecture diagram is now a running model. Five PyTorch operations. Each class maps to a concept from Module 4.2. The parameter count matches. The model generates text (badly). Next: make it generate WELL (Lesson 2 -- pretraining).

13. **Next step** -- "The model generates random text because every weight is random. In the next lesson, we train it. You will watch the gibberish gradually become recognizable English."

### Notebook Structure (4-3-1-building-nanogpt.ipynb)

The notebook mirrors the lesson outline:
1. Config dataclass
2. Build Head, test with dummy input
3. Build MultiHeadAttention, test with dummy input
4. Build FeedForward, test with dummy input
5. Build Block, test with dummy input
6. Build GPT, test with dummy input
7. Weight initialization, measure activation stats before/after
8. Parameter counting verification
9. Generate method, produce random text
10. (Stretch) Try different configs: tiny (4 layers, d=128) vs GPT-2 small. Compare parameter counts and generation speed.

---

## Review -- 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 5
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings that would leave the student lost, but the one critical finding (missing planned visual modality) combined with several improvement findings warrant a revision pass. The lesson is structurally sound and well-connected to prior knowledge. The code is clean and well-annotated. The main issues are a missing modality, some narrative ordering problems, and missed opportunities for the negative examples that were explicitly planned.

### Findings

#### [CRITICAL] -- Missing planned visual modality: architecture diagram with class boundaries

**Location:** Hook section (Section 3, "The Parts List") and throughout
**Issue:** The planning document explicitly lists a visual modality: "Architecture diagram with code class boundaries drawn on it. Each colored region of the GptArchitectureDiagram mapped to the class that implements it." This was planned as one of four modalities and serves a critical bridging function between the conceptual diagram the student already knows (from decoder-only-transformers) and the code class hierarchy they are about to build. The built lesson has no visual/diagram of any kind. The only modalities present are: symbolic (code), concrete example (shape trace), and intuitive (parts list). This drops the lesson to 3 modalities, but the missing one is the most important for an implementation lesson because it bridges "I know the architecture" to "I know which class implements which box."
**Student impact:** The student reads five separate class implementations without a visual anchor mapping them to the architecture diagram they already have internalized. The PhaseCard summary at the end partially compensates (listing class names with their conceptual sources), but it comes after all the code -- too late to serve as a roadmap. The student must build the mapping mentally rather than seeing it drawn.
**Suggested fix:** Add an inline SVG (or reference to the GptArchitectureDiagram with class name annotations overlaid) near the hook section, before the Config. Draw colored boundaries on the architecture diagram showing which region each class implements: Head (sky blue), CausalSelfAttention (sky blue), FeedForward (amber), Block (the full block region), GPT (the entire stack). This becomes the roadmap the student follows as they build each class.

#### [IMPROVEMENT] -- Assembly Path section is misplaced after the payoff

**Location:** Section 14, "The Assembly Path" (PhaseCards)
**Issue:** The bottom-up build order summary (PhaseCards showing Head -> MHA -> FFN -> Block -> GPT) appears after the generation payoff and generated text sections. By this point, the student has already built everything. A roadmap shown after the journey is complete is retrospective rather than navigational. The planning document placed this as part of the hook: "show the architecture diagram with class boundaries drawn on it. Each region maps to a class you will write."
**Student impact:** The student lacks a clear roadmap at the start. They encounter each class sequentially without knowing the full build plan. The assembly path works as a summary but does not serve its planned navigational purpose.
**Suggested fix:** Move the PhaseCard sequence (or a condensed version of it) to immediately after the hook/parts-list section and before the Config. This gives the student a map of the journey before they start. The current location can be removed or replaced with a brief "you built all five" echo.

#### [IMPROVEMENT] -- Weight initialization section lacks the planned negative example with concrete numbers

**Location:** Section 10, "Weight Initialization"
**Issue:** The planning document specified: "Initialize a 12-block transformer with default PyTorch init. Measure activation standard deviations at each block: they explode or collapse. 12 residual additions compound the variance issue." The built lesson describes this in prose ("activation standard deviations grow with depth") and defers concrete measurement to the notebook ("The notebook shows before-and-after measurements"). But the lesson page itself shows no concrete numbers. The misconception table planned: "Concrete: show activation stats with default vs proper init." Without concrete numbers in the lesson, the weight initialization section is a recipe without motivation.
**Student impact:** The student reads "default init causes activations to grow or collapse" as a claim to accept on faith rather than a demonstrated fact. The weight initialization recipe becomes a "just do this" instruction rather than a motivated fix for a measured problem. The misconception "initialization is just setting everything to zero or small random values" is addressed only by assertion, not by the planned negative example.
**Suggested fix:** Add a small inline table or comparison showing activation std at blocks 1, 4, 8, 12 with default init (e.g., 0.8, 1.6, 3.2, 6.4) vs scaled init (e.g., 0.8, 0.9, 0.85, 0.82). The numbers do not need to be exact -- they need to make the problem visceral. Even a simple pair of numbers ("default: std grows from 0.8 to 6.4 across 12 blocks; scaled: std stays near 0.8") would be sufficient.

#### [IMPROVEMENT] -- Multi-head attention section does not explicitly show that both approaches produce the same output

**Location:** Section 6, "Build: Multi-Head Attention"
**Issue:** The planning document specified: "Show they produce the same output." The lesson states "Both produce the same output" in the concluding paragraph but does not include a code snippet or assertion demonstrating this equivalence. For a student learning the batched reshape approach for the first time, the reshape from (B, T, d_model) to (B, h, T, d_k) is a non-obvious transformation. Claiming equivalence without demonstrating it leaves the student trusting the builder rather than verifying themselves.
**Student impact:** The student must accept on faith that the loop version and the batched version produce identical results. This is one of the lesson's key implementation insights -- the reshape IS the dimension splitting -- and it would be stronger with a concrete verification.
**Suggested fix:** Add a brief code snippet (3-5 lines) or a "verify this in the notebook" callout that shows: create a random input, pass through both implementations, assert outputs are equal (within floating-point tolerance). This can be brief -- the notebook presumably covers it in detail.

#### [IMPROVEMENT] -- No inline shape verification assertions in the lesson code

**Location:** Sections 5-9 (all Build sections)
**Issue:** The planning document listed "Shape verification at every layer boundary" as a positive example and described it as: "Demonstrate that asserting shapes catches bugs before they propagate. Create a dummy input, pass through each component, verify output shape matches expectation." The built lesson includes shape comments on every line of code (good) and a full-model shape trace in the GPT section (good), but no actual assertion code. The TipBlock in the Assembly Path section says "After writing each class, create a dummy input and verify the output shape" but this advice comes at the end rather than being demonstrated inline with the code.
**Student impact:** The student sees shape comments as documentation but does not internalize the practice of writing assertions. The planned skill ("the assertion discipline") is described but not demonstrated. Shape comments tell you what the shape should be; assertions tell you what the shape actually is.
**Suggested fix:** After the Head class code, add a brief "verify it" snippet: `head = Head(debug_config, 32); x = torch.randn(2, 64, 128); assert head(x).shape == (2, 64, 32)`. One or two more at key boundaries (Block, GPT). This establishes the pattern the student should follow in the notebook.

#### [IMPROVEMENT] -- Training vs generation comparison does not explicitly address misconception #4

**Location:** Section 12, "Generate: Autoregressive Inference" (the amber comparison box)
**Issue:** Misconception #4 from the planning document: "The generate() function should use the training forward pass directly." The lesson includes a comparison box showing training mode vs generation mode differences, which partially addresses this. However, the lesson's generate() method actually DOES call the full training forward pass (`logits, _ = self(idx_cond)`) and then throws away all but the last position's logits. The planned distinction was: "Using the full training forward pass wastes compute by computing logits for positions you already have." The lesson does not call this out explicitly -- it computes all positions and silently discards most of them.
**Student impact:** A careful student would notice that generate() calls self() (the full forward pass) and then takes only `logits[:, -1, :]`. This is correct but wasteful, and the lesson does not acknowledge the waste or explain that production systems use KV caching to avoid it. The student might think this is the "right" way to do generation rather than a simplified version.
**Suggested fix:** Add 1-2 sentences after the generate() code noting: "Notice that we compute logits for ALL positions but only use the last one. This is wasteful -- the full forward pass recomputes attention for tokens we already processed. Production systems use KV caching to avoid this recomputation. We will cover that in Lesson 3." This both addresses the misconception and sets up a forward reference.

#### [POLISH] -- Em dash spacing inconsistency in code comments

**Location:** Head class code (line 260), CausalSelfAttention class code (line 389)
**Issue:** In the prose, em dashes correctly have no spaces (`word--word` rendered as em dashes). In the code comments, the style is: `three "lenses" on the same input` (line 260) -- no em dash issue. However, checking the built lesson's prose against the writing style rule: all em dashes in the HTML use `&mdash;` with no surrounding spaces, which is correct. No violations found in prose. This is a minor note that the code comments use standard Python comment style (fine).
**Student impact:** None.
**Suggested fix:** No action needed.

#### [POLISH] -- Lesson outline planned 13 sections; built lesson has 17 sections (some structural rearrangement)

**Location:** Overall structure
**Issue:** The built lesson splits the planned "Generate" section into two parts (the generate() method code and the "Moment of Truth" generated text output), adds the Assembly Path as a separate section, adds the Notebook Link section, and adds a separate Next Step section. This is fine structurally but represents an undocumented deviation from the planned outline.
**Student impact:** No negative impact. The splits are natural and improve readability. The Assembly Path placement is the only structural issue (addressed in the IMPROVEMENT finding above).
**Suggested fix:** No action needed beyond addressing the Assembly Path placement.

### Review Notes

**What works well:**
- The hook (parts list reveal) is genuinely effective at deflating intimidation. Five colored boxes with familiar PyTorch operations is a strong opening.
- Code quality is excellent: every line has shape comments, naming is consistent with the planning document and GPT-2 conventions (`c_attn`, `c_proj`, `c_fc`), and the code is clean enough to type from scratch.
- Callbacks to prior lessons are explicit and well-placed: "from Values and the Attention Output," "from The Transformer Block," "from What is a Language Model?" The student never encounters a concept without being told where they learned it.
- The transformer block section is particularly satisfying -- the formula literally becomes the code. This is the "of course" moment the lesson is designed to produce.
- The weight tying aside is clear and well-placed.
- The training vs generation comparison box is a clean, scannable summary of the key differences.
- The summary takeaways are well-chosen and capture the essential insights.

**Systemic patterns:**
- The lesson leans heavily on "the notebook will show this" for concrete demonstrations (shape assertions, init statistics, loop-vs-batch equivalence). This is acceptable for a hands-on lesson that is paired with a notebook, but the lesson page should contain enough concrete evidence to stand on its own for review reading. The planned negative example for weight initialization should not be entirely deferred to the notebook.
- The lesson is strong on symbolic (code) and intuitive (parts list, callbacks) modalities but weak on visual modality. For an implementation lesson where the student is translating a visual diagram into code, the visual bridge is the most important modality to include.

---

## Review -- 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All iteration 1 findings have been correctly addressed. The lesson now has four strong modalities (symbolic/code, visual/architecture SVG, concrete example/shape trace + activation stats, intuitive/parts list), a clear navigational roadmap before the code begins, inline shape assertions, concrete weight initialization evidence, MHA equivalence verification, and a KV caching forward reference. No critical or improvement findings remain. Two minor polish items noted below.

### Findings

#### [POLISH] -- Weight initialization concrete numbers could precede the recipe for stronger "problem before solution"

**Location:** Section 10, "Weight Initialization"
**Issue:** The weight initialization section presents the problem in prose ("activations grow with depth"), then gives the recipe (bullet list + code), then shows the concrete before/after numbers. For maximum "problem before solution" impact, the concrete numbers could appear before the recipe so the student sees the measured problem and then receives the fix. However, the current order works: the prose states the problem, the recipe is the fix, and the numbers serve as evidence. This is a minor ordering preference, not a pedagogical failure.
**Student impact:** Negligible. The student understands the problem from the prose and sees the evidence after the fix. The narrative is coherent.
**Suggested fix:** Optional. If desired, move the side-by-side activation stats grid above the recipe bullet list. But the current order is acceptable.

#### [POLISH] -- MHA equivalence verification comment "details in the notebook" is slightly hand-wavy

**Location:** Section 6, MHA equivalence verification code snippet
**Issue:** The verification code snippet includes a comment `# Copy weights so both use the same parameters` followed by `# (details in the notebook)`. For a student reading the lesson page, this leaves a gap: they see the assertion but do not see how to set up the test. This is acceptable because (a) the concept is clear -- same weights = same output, and (b) the notebook provides the full implementation. But a 1-line comment like `# (set loop_mha weights equal to batched_mha weights)` would be slightly more informative than "details in the notebook."
**Student impact:** Minimal. The student understands the point being made.
**Suggested fix:** Optional. Replace the comment with a slightly more specific hint, e.g., `# (copy loop_mha's per-head weights into batched_mha's combined projection)`.

### Iteration 1 Fix Verification

All 8 findings from iteration 1 have been verified as correctly addressed:

| Finding | Severity | Status |
|---------|----------|--------|
| Missing architecture diagram with class boundaries | CRITICAL | Fixed. Inline SVG added in Section 4 with color-coded class boundaries (GPT, Block, CausalSelfAttention, FeedForward), dimension annotations, weight tying, legend, and build order visible. |
| Assembly Path section misplaced after payoff | IMPROVEMENT | Fixed. Moved to Section 5, before Config. Student now has a roadmap before building. |
| Weight init lacks concrete numbers | IMPROVEMENT | Fixed. Side-by-side comparison grid shows default init (std 0.82 -> 6.55) vs scaled init (std 0.81 -> 0.80) across 12 blocks. |
| MHA equivalence not demonstrated | IMPROVEMENT | Fixed. Verification code snippet added with `torch.allclose` assertion. |
| No inline shape assertions | IMPROVEMENT | Fixed. Assertions added after Head class (verify_head.py) and Block class (verify_block.py). |
| KV caching not mentioned in generate section | IMPROVEMENT | Fixed. Paragraph added after generate() code explaining the waste and forward-referencing Lesson 3. |
| Em dash spacing in code comments | POLISH | Was already fine (code comments use standard Python convention). |
| Outline deviation (13 planned vs 17 built) | POLISH | Was already fine (natural splits improve readability). |

### Review Notes

**What works well (reinforced by fixes):**
- The architecture diagram SVG is well-designed: class boundaries are drawn with dashed lines and monospace labels, color coding matches the legend and is consistent with prior lessons (sky blue for attention, amber for FFN, purple for embeddings, emerald for layer norm, violet for residual stream). Dimension annotations at key stages. Weight tying annotation with dashed connecting line. This is now the visual anchor that bridges the conceptual diagram from Module 4.2 to the code class hierarchy.
- The assembly roadmap in its new position (Section 5, before Config) gives the student a clear 5-step plan before they write any code. The combination of the architecture diagram (which region corresponds to which class) and the roadmap (build order with line counts) is a strong navigational pair.
- The side-by-side activation statistics for weight initialization are visceral and concrete. The student can see the 8x growth with default init vs stable values with scaled init. This transforms the initialization recipe from "just do this" into a motivated fix.
- The shape assertion snippets after Head and Block establish a pattern the student can follow in the notebook. They are brief (3-4 lines each) and demonstrate the practice rather than just describing it.

**Overall assessment:**
The lesson is now a strong BUILD lesson. It has four modalities, a clear navigational structure, well-placed callbacks to prior lessons, concrete evidence for every claim, and an effective emotional arc (intimidation deflated by parts list -> steady build -> payoff with generated text). The two remaining Polish findings are genuinely minor and can be addressed at the builder's discretion without a re-review.

---

## What Was Actually Built

The built lesson follows the design closely. Key deviations addressed during review:

1. **Architecture diagram added** (was missing in iteration 1). Inline SVG with color-coded class boundaries (sky blue for attention, amber for FFN, purple for embeddings, emerald for layer norm). Weight tying annotation. Build order visible.
2. **Assembly roadmap moved** from after the payoff to Section 5 (before Config), where it serves as a navigational preview rather than a retrospective.
3. **Weight initialization section** gained concrete side-by-side activation statistics (default: std 0.82->6.55; scaled: std 0.81->0.80) rather than deferring entirely to the notebook.
4. **MHA equivalence verification** added with torch.allclose assertion snippet.
5. **Shape assertion snippets** added inline after Head and Block classes.
6. **KV caching forward reference** added after generate() code, noting the wasteful recomputation.
7. **Structural expansion:** 13 planned sections became 17 built sections (natural splits for readability: separate generated text section, separate notebook link, separate next step).

No conceptual changes from the design. The lesson teaches exactly what was planned. Two POLISH findings from iteration 2 were left at builder's discretion (weight init ordering, MHA equivalence comment wording).
