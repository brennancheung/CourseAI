# Module 4.3: Building & Training GPT -- Record

**Goal:** The student can implement a complete GPT model in PyTorch, train it from scratch on a text corpus, understand the engineering decisions that make training work at scale, and verify their implementation by loading real GPT-2 weights and generating coherent text.
**Status:** Complete (4 of 4 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| GPT architecture implemented in PyTorch (Head, CausalSelfAttention, FeedForward, Block, GPT classes) | APPLIED | building-nanogpt | Student writes every class, verifies shapes, runs the full model. Five nn.Module classes composing bottom-up. |
| GPTConfig dataclass for hyperparameter management | APPLIED | building-nanogpt | Debug config (tiny) vs GPT-2 config pattern. Student uses both during development. |
| Single attention head implementation (Q/K/V projections, scaled dot-product, causal mask) | APPLIED | building-nanogpt | Direct translation of the formula from Module 4.2. register_buffer for causal mask. Shape comments on every line. |
| Batched multi-head attention via reshape (no loop over heads) | APPLIED | building-nanogpt | Reshape from (B, T, d_model) to (B, h, T, d_k) IS the dimension split. Both loop and batched approaches shown and verified equivalent with torch.allclose. |
| Loop-based multi-head attention (explicit Head per head, concatenate) | INTRODUCED | building-nanogpt | Shown as Approach 1 for clarity. Correct but slow. Replaced by batched version in practice. |
| register_buffer for non-learnable tensors (causal mask) | DEVELOPED | building-nanogpt | Mask moves to GPU with model but excluded from parameters(). Explained in aside. |
| Weight tying (embedding and output projection share weights) | DEVELOPED | building-nanogpt | self.transformer.wte.weight = self.lm_head.weight. Saves ~38M parameters. Reverse mapping: token->vector and vector->token scores. |
| Transformer weight initialization (normal with sigma=0.02, scaled residual projections by 1/sqrt(2N)) | DEVELOPED | building-nanogpt | Side-by-side activation statistics: default init std grows 0.82->6.55 across 12 blocks; scaled init stays 0.80-0.85. model.apply() pattern. |
| Activation growth in deep residual networks from unscaled initialization | DEVELOPED | building-nanogpt | Concrete before/after numbers make the problem visceral. 24 residual additions compound variance. |
| Parameter counting as architecture verification | APPLIED | building-nanogpt | Programmatic count verified against hand-computed ~124.4M from Module 4.2. Prediction exercise before running code. |
| Autoregressive generation in code (generate method) | DEVELOPED | building-nanogpt | torch.no_grad(), crop to block_size, forward pass, take last position logits, apply temperature, sample, append. Connected to conceptual loop from 4.1.1. |
| Training mode vs generation mode differences | INTRODUCED | building-nanogpt | Comparison box: training uses all positions with gradients and loss; generation uses last position only with no_grad and sampling. |
| KV caching motivation (generation recomputes attention wastefully) | MENTIONED | building-nanogpt | generate() calls full forward pass but only uses last position. Forward reference to Lesson 3. |
| Shape verification via assertions at layer boundaries | APPLIED | building-nanogpt | Inline assertion snippets after Head and Block classes. Pattern for the student to follow in the notebook. |
| nn.ModuleDict and nn.ModuleList for organizing complex models | APPLIED | building-nanogpt | GPT class uses ModuleDict for named submodules, ModuleList for repeated blocks. |
| Pre-norm layer norm placement in transformer blocks | DEVELOPED | building-nanogpt | LN before sub-layer (MHA or FFN). Code directly implements the formula x' = x + MHA(LN(x)). Callback to The Transformer Block. |
| Complete GPT training loop (forward, loss, backward, step with LR scheduling and gradient clipping) | APPLIED | pretraining | Student assembles the full loop from familiar pieces plus three new additions. Side-by-side comparison with MNIST loop demonstrates structural identity. |
| Text dataset preparation for language modeling (tokenize, context windows, input/target offset) | DEVELOPED | pretraining | Raw text -> token IDs -> Dataset with input[i:i+T] / target[i+1:i+T+1]. Concrete trace-through with "The cat sat on the mat" showing all positions predict simultaneously. |
| Cross-entropy for next-token prediction over 50K vocabulary | DEVELOPED | pretraining | Same formula as 10-class classification, reshape logits from (B, T, V) to (B*T, V). Initial loss sanity check: -ln(1/50257) ~= 10.82 confirms correct initialization. |
| Learning rate scheduling (linear warmup + cosine decay) | DEVELOPED | pretraining | Motivating problem: constant LR fails at transformer scale (too high = divergence, too low = slow). Two-phase schedule: warmup first 5% of steps, cosine decay remainder. SVG visualization of schedule shape. Manual param_group update pattern. |
| Gradient clipping (clip_grad_norm_) | INTRODUCED | pretraining | One-line addition after backward(), before step(). Global norm computed across all parameters, scaled proportionally if exceeding threshold. Concrete example showing normal norms (~0.6-0.9) with occasional spike (12.3) being clipped. |
| AdamW optimizer (decoupled weight decay) | INTRODUCED | pretraining | One-line swap from Adam. Weight decay applied directly to parameters rather than added to loss gradient. Transformer default. Known-good hyperparameters from nanoGPT (lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1). |
| Loss curve interpretation for language models (noisy curves, nonlinear loss-to-quality mapping) | DEVELOPED | pretraining | Language model loss is noisier than MNIST due to variable batch difficulty. Diagnostic question format: "Is something wrong?" for jagged curves, "What happened?" for spikes. Loss-to-text-quality is logarithmic: biggest qualitative leaps happen early. |
| Periodic text generation during training as qualitative monitor | DEVELOPED | pretraining | Generate text every N steps to observe quality improvement. Five-stage progression from gibberish (loss ~10.8) to coherent Shakespeare (loss ~2.0). The emotional payoff of the module. |
| Validation loss evaluation during training (step-based, periodic) | DEVELOPED | pretraining | Same scissors pattern from Series 1 applied to language model training. Val iterator created fresh per evaluation window. TinyShakespeare + 124M params = inevitable overfitting acknowledged. |
| Checkpointing during language model training | APPLIED | pretraining | Same save dict pattern from Series 2 (model, optimizer, step, loss). Periodic saves every 2000 steps. Reinforcement of existing DEVELOPED concept. |
| Compute-bound vs memory-bound operations (arithmetic intensity) | INTRODUCED | scaling-and-efficiency | GPU compute throughput (312 TFLOPS) vastly exceeds memory bandwidth (2 TB/s). Low arithmetic intensity = memory-bound (layer norm, softmax). High = compute-bound (matmul). Kitchen analogy: fast chefs but slow delivery truck. Corrects the "more cores always means faster" misconception from Series 2. |
| Mixed precision with bfloat16 (same range as float32, less mantissa) | DEVELOPED | scaling-and-efficiency | Extends Series 2 float16/GradScaler INTRODUCED concept. bfloat16: 8-bit exponent (float32 range), 7-bit mantissa (less precision). No GradScaler needed (no underflow). But precision loss during addition (1.0 + 0.0001 = 1.0) means master weights in float32 still required. The "mixed" is the essential part. |
| Master weights pattern (float32 accumulation, bfloat16 forward/backward) | DEVELOPED | scaling-and-efficiency | Float32 copy of weights for the accumulation step where tiny gradients must survive addition to large parameters. Cast to bfloat16 for speed in forward and backward passes. Concrete weight update example: param=1.0, grad=0.0001 showing float32 preserves, both 16-bit formats lose. |
| KV caching for autoregressive inference | DEVELOPED | scaling-and-efficiency | Cache K and V from previous generation steps. Only compute Q, K, V for the new token. Concatenate new K,V with cache. Cost: O(n) vs O(n^2) without cache. Concrete: generating 100 tokens from 10-token prompt: 5550 vs 100 token positions processed (55x). At length 1000: 500x. Not optional at production scale. SVG diagram of cache growth. |
| KV cache memory cost (compute vs memory tradeoff) | INTRODUCED | scaling-and-efficiency | GPT-2 KV cache at seq_len 1024: ~37.7 MB per sequence, ~1.2 GB for batch of 32. Explains why long-context models need more GPU memory during inference. |
| Flash attention (tiled attention computation, same result, O(n) memory) | INTRODUCED | scaling-and-efficiency | Standard attention materializes full n x n matrix (4 memory trips). Flash attention tiles computation, never stores full matrix. Same math (numerically identical, torch.allclose returns True). O(n) memory vs O(n^2). 2-4x faster. Fuses causal mask into tiled computation. Built into PyTorch via scaled_dot_product_attention. Concrete: seq_len 4096, 12 heads: 384 MB vs 384 KB (~1000x reduction). |
| Scaling laws (Chinchilla compute-optimal training) | INTRODUCED | scaling-and-efficiency | N_opt ~ sqrt(C), D_opt ~ sqrt(C). Scale model size and data together. Chinchilla (70B, 1.4T tokens) outperforms Gopher (280B, 300B tokens). Most early LLMs were undertrained. Power law: L ~ C^(-0.05), doubling compute reduces loss by ~3.4%. Predictability enables planning multi-million dollar training runs. |
| torch.compile (operator fusion) | MENTIONED | scaling-and-efficiency | Fuses multiple operations into single GPU kernel, eliminating intermediate memory reads/writes. Name-drop only. |
| Continuous batching (inference serving) | MENTIONED | scaling-and-efficiency | Slot new requests into completed batch positions during serving. Name-drop only. |
| Speculative decoding | MENTIONED | scaling-and-efficiency | Small fast model drafts tokens, large model verifies in parallel. Name-drop only. |
| Mixture of experts (MoE) | MENTIONED | scaling-and-efficiency | Router activates subset of parameters per token. More total parameters, same compute per token. Name-drop only. |
| Weight name mapping between codebases (translating parameter names from HuggingFace to student model) | APPLIED | loading-real-weights | Student builds the complete mapping function: iterates over their state_dict keys, finds corresponding HuggingFace key, transposes where needed, copies data. The mapping doubles as per-component architecture verification. |
| Conv1D vs nn.Linear weight transposition (.t() for 2D weights in attention and FFN) | DEVELOPED | loading-real-weights | HuggingFace GPT-2 uses Conv1D storing weights as (in_features, out_features), transposed relative to nn.Linear (out_features, in_features). Pattern: all 2D weight matrices in attention (c_attn, c_proj) and FFN (c_fc, c_proj) need transposing. Embeddings, layer norms, and biases copy directly. ComparisonRow + WeightMappingDiagram SVG + table. |
| Logit comparison as model verification (torch.allclose on output logits) | DEVELOPED | loading-real-weights | Feed same input tokens to both models, compare output logits with torch.allclose(atol=1e-5). Stronger than parameter counting (verifies computation, not just structure) and stronger than text comparison (deterministic, not stochastic). Extends "parameter count = architecture verification" mental model. |
| Silent failure from incorrect weight transposition (model runs but produces garbage) | DEVELOPED | loading-real-weights | Negative example: skipping transposition for one layer produces incoherent text with no error message. Shape trace explains why: copy_() ignores semantics, forward pass dimensions still align, matmul succeeds with meaningless values. The most insidious bug type. |
| Weight tying behavior during cross-codebase loading (skip lm_head.weight, embedding handles both) | DEVELOPED | loading-real-weights | Reinforces building-nanogpt DEVELOPED concept. data_ptr() check confirms same memory. State dict key count difference (148 vs 147) is weight tying in action, not a bug. Load embedding once; tying handles the rest. |
| HuggingFace transformers library (GPT2LMHeadModel.from_pretrained for downloading weights) | INTRODUCED | loading-real-weights | Two lines of code: import + from_pretrained("gpt2"). Used only as weight source and reference implementation. No deep dive into the library. |
| Verification chain (parameter count + logit comparison + coherent generation) | DEVELOPED | loading-real-weights | Three levels of evidence: parameter count verifies structure, logit comparison verifies computation, coherent text generation verifies behavior. All three confirmed = implementation correct. Synthesizes the module's verification thread. |

## Per-Lesson Summaries

### building-nanogpt
**Status:** Built
**Cognitive load type:** BUILD
**Type:** Hands-on (notebook: `4-3-1-building-nanogpt.ipynb`)

**What was taught:**
- Translation of the complete GPT architecture (from Module 4.2) into five PyTorch nn.Module classes: Head, CausalSelfAttention (batched MHA), FeedForward, Block, GPT
- Bottom-up assembly order matching the learning order from Module 4.2
- Shape verification discipline with inline assertions after key components
- Weight initialization recipe for transformers with concrete activation statistics showing why it matters
- Parameter counting as architecture verification (programmatic count matching hand-computed ~124.4M)
- Autoregressive text generation via the generate() method
- Weight tying between embedding and output projection

**Mental models established:**
- "Five PyTorch operations build the entire GPT" -- nn.Linear, nn.Embedding, nn.LayerNorm, nn.GELU, nn.Dropout. Deflates intimidation.
- "The complexity is in the assembly, not the parts" -- every operation is familiar from Series 2.
- "The formula IS the code" -- transformer block forward() method directly implements the mathematical formula.
- "Parameter count = architecture verification" -- one number confirms the entire structure.
- "Untrained gibberish is a success" -- correct architecture producing random text means everything works; training is separate.

**Analogies used:**
- Architecture diagram with class boundaries (inline SVG with color-coded regions mapping to code classes)
- Assembly roadmap (5-step visual preview before building)
- "Mapping code to concepts" boxes connecting code lines to prior lesson concepts

**How key concepts were taught:**
- **Batched MHA:** Both loop (Approach 1, explicit) and batched (Approach 2, reshape) versions shown side-by-side. Equivalence verified with torch.allclose. The reshape from (B, T, d_model) to (B, h, T, d_k) is explained as the "split, not multiplied" operation from Module 4.2.
- **Weight initialization:** Prose motivates the problem (24 residual additions compound variance), recipe given, then concrete side-by-side grid shows default init (std 0.82->6.55 across 12 blocks) vs scaled init (std 0.81->0.80). Notebook reproduces measurements.
- **Generation:** Conceptual loop from What is a Language Model? (4.1.1) implemented as the generate() method. Training vs generation comparison box. KV caching waste noted with forward reference to Lesson 3. Temperature parameter connects to TemperatureExplorer widget from 4.1.1.

**What is NOT covered (explicitly deferred):**
- Training the model (Lesson 2)
- Dataset preparation (Lesson 2)
- Optimization, learning rate scheduling (Lesson 2)
- GPU utilization, mixed precision, flash attention (Lesson 3)
- Loading pretrained weights (Lesson 4)
- nn.MultiheadAttention (student builds from scratch)
- Different model sizes (GPT-2 medium/large/XL)
- Batched KV caching for efficient generation (Lesson 3)

**Modalities used:** Symbolic/code (primary -- every class with shape comments), Visual (architecture SVG with class boundaries), Concrete example (shape trace with GPT-2 dimensions, activation statistics), Intuitive (parts list reveal, "the formula is the code" moment)

### pretraining
**Status:** Built
**Cognitive load type:** STRETCH
**Type:** Hands-on (notebook: `4-3-2-pretraining.ipynb`)

**What was taught:**
- Text dataset preparation: tokenizing a corpus with tiktoken, slicing into context windows, the input/target one-position offset pattern, custom Dataset/DataLoader for language modeling
- Cross-entropy loss for next-token prediction: same formula as classification but with 50,257 classes, reshape trick (B, T, V) -> (B*T, V), initial loss sanity check (~10.82 confirms uniform predictions)
- AdamW optimizer: one-line swap from Adam, decoupled weight decay, known-good hyperparameters from nanoGPT
- Learning rate scheduling: linear warmup (first ~5% of steps) + cosine decay (remaining ~95%), motivated by constant LR failure at transformer scale
- Gradient clipping: clip_grad_norm_ after backward() and before step(), global norm threshold of 1.0, preserves gradient direction while bounding magnitude
- Complete training loop assembly: all pieces combined with periodic logging, validation evaluation, checkpointing, and text generation
- Loss curve interpretation: noisy curves are normal for language models, diagnostic reasoning for spikes and plateaus, nonlinear loss-to-quality mapping
- Watching generated text improve from gibberish to coherent Shakespeare across training

**Mental models established:**
- "Same heartbeat, new instruments" -- the GPT training loop is structurally identical to the MNIST loop with three additions (LR scheduling, gradient clipping, text dataset prep)
- "Dynamic Goldilocks" -- LR scheduling navigates changing optimal learning rates as training progresses, extending the static Goldilocks zone concept from Series 1
- "Cross-entropy doesn't care about vocabulary size" -- same formula, same PyTorch call, just more classes
- "The biggest leaps are early" -- loss-to-text-quality mapping is logarithmic, not linear; the dramatic qualitative improvement happens in the first few thousand steps

**Analogies used:**
- "Same heartbeat, new instruments" -- training loop as a heartbeat that never changes; LR scheduling and gradient clipping are new instruments added to the band
- "Dynamic Goldilocks" -- LR scheduling as a dynamic version of the Goldilocks zone (gentle warmup for fragile random weights, aggressive mid-training, careful convergence)
- Gradient clipping as a "safety net" (callback to the MENTIONED concept from 1.3.3, now deployed in practice)

**How key concepts were taught:**
- **Text dataset preparation:** Problem-first ("raw text to training data?"), three-step process (tokenize, slice, offset), concrete trace-through with "The cat sat on the mat" showing 6 tokens producing 5 training examples. WarningBlock addresses misconception that only the last token predicts during training.
- **LR scheduling:** Negative example first -- three-column comparison showing constant LR at 6e-4 (explodes at step 300), constant LR at 1e-5 (crawls to loss 4.3 at step 10000), warmup + cosine (reaches loss 2.0 at step 10000). SVG visualization of schedule shape. Code implementation with manual param_group update.
- **Gradient clipping:** Concrete example of gradient norms (0.83, 0.61, 0.92, 12.3, 0.74) with the spike highlighted. Three-step formula box. Placement clarified: after backward(), before step().
- **Loss-to-quality mapping:** Five color-coded boxes showing generated text at loss ~10.8 (random), ~4.0 (common words), ~3.0 (crude grammar), ~2.5 (grammatical sentences), ~2.0 (coherent Shakespeare). Nonlinear insight stated in main content flow, not just aside.
- **Loss curve interpretation:** Two GradientCards use question-first diagnostic format ("Is something wrong?" / "What happened?") to engage reasoning before presenting answers. Two additional cards cover normal patterns and warning signs.

**What is NOT covered (explicitly deferred):**
- GPU optimization, mixed precision, flash attention (Lesson 3: scaling-and-efficiency)
- Multi-GPU or distributed training
- Different tokenization strategies (Module 4.1 covered this)
- Hyperparameter search (uses known-good values from nanoGPT)
- Fine-tuning or transfer learning (Module 4.4)
- Evaluation beyond loss and qualitative text inspection

**Modalities used:** Symbolic/code (training loop, dataset, LR schedule function, gradient clipping), Visual (SVG LR schedule plot, color-coded text progression boxes, three-column constant LR comparison), Concrete example (input/target offset trace-through, gradient norm spike values, loss-to-text progression), Verbal/analogy ("same heartbeat," "dynamic Goldilocks," "safety net"), Intuitive ("cross-entropy doesn't care about vocabulary size")

### scaling-and-efficiency
**Status:** Built
**Cognitive load type:** BUILD
**Type:** Conceptual (no notebook)

**What was taught:**
- Compute-bound vs memory-bound operations: GPU compute throughput (312 TFLOPS) vastly exceeds memory bandwidth (2 TB/s). Arithmetic intensity (FLOPs per byte transferred) determines which regime. Most transformer operations (layer norm, softmax, dropout) are memory-bound; only matrix multiplication is compute-bound.
- Mixed precision with bfloat16: extends Series 2 float16/GradScaler knowledge. bfloat16 has same exponent range as float32 (8-bit exponent) but reduced mantissa (7 bits). No gradient underflow, no GradScaler needed. But precision loss during addition means master weights in float32 are still essential for accumulation.
- KV caching for autoregressive inference: cache K and V from previous generation steps, compute Q/K/V only for the new token. Transforms generation cost from O(n^2) to O(n). Concrete cost comparison: 55x at 100 tokens, 500x at 1000 tokens. SVG diagram of cache growth per step.
- Flash attention: tiled attention computation that avoids materializing the full n x n attention matrix. Produces numerically identical results to standard attention (torch.allclose returns True). O(n) memory instead of O(n^2). Fuses causal mask. Built into PyTorch. Concrete: 384 MB vs 384 KB at seq_len 4096.
- Scaling laws (Chinchilla): compute-optimal scaling requires matching model size to data quantity. N_opt ~ sqrt(C), D_opt ~ sqrt(C). Chinchilla (70B, 1.4T tokens) outperforms Gopher (280B, 300B tokens). Power law L ~ C^(-0.05): doubling compute reduces loss by ~3.4%.
- Broader efficiency landscape (MENTIONED only): torch.compile, continuous batching, speculative decoding, MoE.

**Mental models established:**
- "The bottleneck is the delivery truck, not the chefs" -- for memory-bound operations, faster GPU compute does nothing; the GPU is already done computing and is waiting for data from memory. Corrects the "more cores = faster" model from Series 2.
- "The math is elegant; the engineering makes it work" -- mixed precision, KV caching, flash attention, and scaling laws are not afterthoughts but what separates a research prototype from a real system.
- "Scale both, not just one" -- like building a house, the optimal allocation between model size and data depends on the ratio. Scale both together with sqrt(C).

**Analogies used:**
- Kitchen with fast chefs and slow delivery truck (compute-bound vs memory-bound)
- Vector addition vs matrix multiplication as extremes of arithmetic intensity spectrum
- bfloat16 as "a different kind of ruler -- same range as float32, coarser markings" (extends "micrometer -> ruler -> tape measure" from Series 2)
- Building a house (scaling laws resource allocation -- workers vs materials)
- Each model size plateaus (iso-parameter curves on scaling laws diagram)

**How key concepts were taught:**
- **Compute-bound vs memory-bound:** Problem-first (GPU is faster than its own memory). Kitchen analogy. ComparisonRow of two regimes. Concrete example: vector addition (N FLOPs, 3N transfers) vs matmul (O(n^3) FLOPs, O(n^2) transfers). Arithmetic intensity formula. Explicit misconception correction: "your mental model from Series 2 -- 'GPUs are fast because they have thousands of cores' -- is incomplete."
- **Mixed precision / bfloat16:** Callback to Series 2 autocast + GradScaler. Format comparison table (float32/float16/bfloat16 with bits, exponent, mantissa, range). Explicit distinction between underflow (can the value be represented?) and precision loss during addition (can a small value be added to a large value?). Concrete weight update example showing all three formats. Master weights pattern explained. Prediction checkpoint ("What breaks if you do everything in float16?").
- **KV caching:** Direct callback to generate() method from building-nanogpt. Step-by-step walkthrough showing K,V recomputation at each generation step. SVG diagram showing cached vs new tokens at each step (labeled "Q,K,V" for new, "K,V" for cached). Cost comparison formulas (with and without cache). Memory tradeoff checkpoint (GPT-2 KV cache at 1024: ~37.7 MB/sequence).
- **Flash attention:** Promise first ("numerically identical, not approximately"). Problem statement: standard attention materializes full n x n matrix, four memory trips. Insight: tiled computation, never store full matrix. SVG diagram (full matrix vs tiled blocks). Concrete memory comparison (384 MB vs 384 KB at seq_len 4096). Causal mask fusion callback to decoder-only-transformers.
- **Scaling laws:** Problem-first ("fixed compute budget, how big should the model be?"). Negative example: student's own TinyShakespeare overfitting (val loss diverging). Chinchilla result table (Gopher vs Chinchilla). Formulas (N_opt ~ sqrt(C), D_opt ~ sqrt(C)). SVG diagram with iso-parameter curves and compute-optimal frontier. Power law grounding: "doubling compute reduces loss by ~3.4%, halving loss requires ~1 million times more compute."

**What is NOT covered (explicitly deferred):**
- Implementing any of these optimizations (no notebook)
- Multi-GPU or distributed training
- Mixture of experts mechanics (MENTIONED only)
- Specific GPU hardware details beyond the basic insight
- Quantization (deferred to Module 4.4)
- Inference serving optimizations in depth

**Modalities used:** Verbal/analogy (kitchen/delivery truck, house building, ruler types), Visual (3 inline SVG diagrams: KV cache growth, flash attention memory comparison, scaling laws with iso-parameter curves), Symbolic (arithmetic intensity formula, KV cache cost formulas, Chinchilla scaling relationships, power law L ~ C^(-0.05)), Concrete example (weight update in float16/float32/bfloat16, generation cost counting 5550 vs 100, flash attention 384 MB vs 384 KB, Gopher vs Chinchilla), Intuitive ("of course generate() recomputes wastefully -- look at your code")

### loading-real-weights
**Status:** Built
**Cognitive load type:** CONSOLIDATE
**Type:** Hands-on (notebook: `4-3-3-loading-real-weights.ipynb`)

**What was taught:**
- Weight name mapping between HuggingFace's GPT-2 implementation and the student's own GPT architecture: iterating over state_dict keys, identifying correspondences, handling naming differences
- Conv1D vs nn.Linear weight transposition: HuggingFace stores (in_features, out_features) while nn.Linear stores (out_features, in_features). Pattern: all 2D weights in attention and FFN need `.t()`, everything else copies directly
- Weight tying handling during cross-codebase loading: skip lm_head.weight (same tensor as wte.weight), data_ptr() verification, state dict key count difference (148 vs 147)
- Logit comparison as the gold standard for model verification: torch.allclose on output logits is deterministic and definitive, stronger than parameter counting or text comparison
- Silent failure from incorrect transposition: model runs without error but produces incoherent text. Shape trace explaining why dimensions align but values are meaningless
- HuggingFace transformers library: minimal introduction (two lines) for downloading pretrained weights. Used only as weight source and reference model
- Coherent text generation from pretrained GPT-2 weights, completing the module arc from gibberish to real GPT-2

**Mental models established:**
- "The mapping IS the verification" -- every shape match is a component verified, every matching logit is a computation confirmed. The weight mapping is not bookkeeping but a per-component X-ray
- "The architecture is the vessel; the weights are the knowledge" -- same code, different weights, dramatically different behavior. Your code defines what the model can compute; pretrained weights encode what it has learned
- "Parameter count verifies structure; logit comparison verifies computation" -- escalation of the existing "parameter count = architecture verification" model to include functional correctness

**Analogies used:**
- Weight mapping as "per-component X-ray" (vs parameter counting as "aggregate checksum")
- The verification chain: three levels of evidence (structure, computation, behavior)
- Module arc summary: random weights -> gibberish -> Shakespeare -> coherent GPT-2 text

**How key concepts were taught:**
- **Weight mapping:** Problem-before-solution. Student tries naive `load_state_dict()` and sees it fail. Side-by-side state dict key comparison reveals name and shape mismatches. Component-by-component table (color-coded: green = direct copy, red = transpose needed) plus WeightMappingDiagram SVG showing both module hierarchies with connecting arrows. The mapping function iterates over student keys, transposes Conv1D weights, copies in-place with `copy_()`.
- **Conv1D transposition:** ComparisonRow showing nn.Linear (out, in) vs Conv1D (in, out) conventions. Concrete shape example: HuggingFace c_attn.weight [768, 2304] -> student model [2304, 768]. Prediction checkpoint: "What shape should the corresponding tensor have?" with the 2304 = 3 * 768 (combined Q/K/V) derivation.
- **Silent failure:** Negative example: load without transposing one layer, generate text. Side-by-side styled boxes: incoherent output (no error, no crash) vs coherent output (correct loading). Shape trace walks through exactly why the model runs: copy_() ignores semantics, x @ W.T + b dimensions align, matmul succeeds with wrong values. Features that should map to Q map to K instead.
- **Logit verification:** Same input to both models, torch.allclose(atol=1e-5). Connection to prior: "Parameter counting verified structure. Logit comparison verifies computation. Together: right structure AND right computation." Prediction checkpoint on what could cause small numerical differences (floating-point ordering, fused implementations, dtypes).
- **Weight tying:** Callback to building-nanogpt. data_ptr() comparison shows same memory address. Key count discrepancy (148 vs 147) as concrete puzzle. Prose: "The key count difference is not a bug -- it is weight tying in action."

**What is NOT covered (explicitly deferred):**
- Training or fine-tuning the loaded model (Module 4.4)
- Loading different GPT-2 sizes (medium, large, XL) -- stretch exercise only
- HuggingFace library in depth
- Quantization or model compression
- Serving or deploying the model
- KV caching implementation during generation

**Modalities used:** Symbolic/code (weight mapping function, logit comparison, generation code, data_ptr verification), Visual (WeightMappingDiagram inline SVG with color-coded arrows and hierarchy indentation, weight mapping table with green/red color coding, ComparisonRow for nn.Linear vs Conv1D, styled comparison boxes for coherent vs incoherent output), Concrete example (side-by-side state dict keys, specific shapes [768, 2304] vs [2304, 768], generated text before/after correct loading, 148 vs 147 key count), Intuitive ("the mapping IS the verification," "the architecture is the vessel, the weights are the knowledge")
