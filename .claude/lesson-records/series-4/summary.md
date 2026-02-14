# Series 4: LLMs & Transformers -- Summary

**Status:** Complete (Module 4.1: complete, Module 4.2: complete, Module 4.3: complete (4 of 4), Module 4.4: complete (5 of 5))

## Series Goal

Build deep, implementation-level understanding of transformers and LLMs. The student should be able to build a GPT from scratch, understand every line, and have the mental models to read real LLM papers. By the end: you've pretrained a small GPT, loaded real GPT-2 weights into your architecture, and understand the full pipeline from pretraining through instruction tuning and alignment.

## Rolled-Up Concept List

### From Module 4.1: Language Modeling Fundamentals (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Language modeling as next-token prediction | DEVELOPED | P(x_t given x_1,...,x_{t-1}). Connected to supervised learning framework: same input-target-loss pattern. Self-supervised labels (text provides its own training labels). |
| Autoregressive generation (sample, append, repeat) | DEVELOPED | The feedback loop. Five-step walkthrough with "The cat sat on the" example. Mermaid diagram. Outputs become inputs. |
| Temperature as distribution reshaping (softmax(logits/T)) | INTRODUCED | Interactive TemperatureExplorer widget. Low T = winner-take-all, T=1 = standard, high T = uniform. Does NOT change model knowledge. |
| Subword tokenization (BPE) | APPLIED | Built BPE tokenizer from scratch in notebook. Merge-pair algorithm. Character/word/subword tradeoffs. Interactive BpeVisualizer. Merge table IS the tokenizer (deterministic). |
| Tokenization artifacts and failure modes | INTRODUCED | Strawberry problem, arithmetic inconsistency, multilingual inequality, SolidGoldMagikarp. Tokenization is not neutral preprocessing. |
| Token embeddings as learned lookup (nn.Embedding) | DEVELOPED | 50K x 768 = 38.4M parameters. One-hot x matrix = row selection. Embeddings are learned parameters, not preprocessing. Cluster after training. |
| Embedding space clustering | DEVELOPED | Interactive EmbeddingSpaceExplorer. Similar tokens have nearby vectors after training. Before/after training comparison. |
| Bag-of-words problem (embeddings without position lose order) | DEVELOPED | "Dog bites man" = "Man bites dog" without position. CNN contrast: position implicit in filters, explicit in transformers. |
| Sinusoidal positional encoding | DEVELOPED | Multi-frequency waves. Clock analogy. Four requirements derived before formula. Interactive PositionalEncodingHeatmap widget. |
| Learned positional encoding (GPT-2) | INTRODUCED | Another nn.Embedding indexed by position. Simpler but can't extrapolate to unseen lengths. |
| Token embedding + positional encoding = model input | DEVELOPED | Added, not concatenated. Same dimension. Formula: input_i = embedding(token_i) + PE(i). |
| Polysemy limitation of static embeddings | INTRODUCED | "Bank" gets one vector regardless of context. Resolution comes from attention (Module 4.2). |
| Output layer / unembedding symmetry | MENTIONED | Embedding maps V->d, output maps d->V. Noted as aside, developed in Module 4.2. |

### From Module 4.2: Attention & the Transformer (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Context-dependent representations via attention | DEVELOPED | Polysemy callback. Weighted average with data-dependent weights. Three escalating attempts: uniform average, fixed weights, dot-product (data-dependent) weights. |
| Dot-product attention on raw embeddings (softmax(XX^T)X) | DEVELOPED | Full worked example with 4 tokens. All pairwise dot products, softmax row-wise, weighted average. Interactive AttentionMatrixWidget. |
| Data-dependent weights as paradigm shift | DEVELOPED | Weights freshly computed from each input, not fixed parameters like CNN filters. The conceptual revolution of attention. |
| Dual-role limitation and attention matrix symmetry | INTRODUCED | One embedding serves both "seeking" and "offering." Cocktail party analogy. Symmetry observation motivates Q/K projections. |
| Q and K as learned projections breaking the dual-role limitation | DEVELOPED | Job fair analogy (seeking card vs offering card). QK^T is asymmetric learned relevance, not raw similarity. Full worked example. Side-by-side symmetric vs asymmetric heatmaps. |
| Scaling by sqrt(d_k) preventing softmax saturation | DEVELOPED | Dimension growth analysis. Temperature callback. Vanishing gradient callback. Not cosmetic -- essential for training. Concrete comparison: softmax([22,-15,3,8]) -> one-hot collapse. |
| Scaled dot-product attention formula (weights = softmax(QK^T/sqrt(d_k))) | DEVELOPED | Built incrementally across Lessons 1-2. Two additions to Lesson 1: projections + scaling. |
| V as third projection separating matching from contributing | DEVELOPED | Three-lenses framing (W_Q, W_K, W_V). K gets you noticed; V is what you deliver. Full formula: output = softmax(QK^T/sqrt(d_k))V. Three-lesson formula evolution shown explicitly. |
| Attention output as per-token weighted average of V vectors | DEVELOPED | Each token gets its own output. n tokens in, n vectors out. Misconception "attention produces single summary" explicitly corrected. |
| Residual stream (attention output added to embedding, not substituted) | DEVELOPED | Initially INTRODUCED in Lesson 3, upgraded to DEVELOPED in Lesson 5. "Shared document" analogy. 24 sub-layers in GPT-2. Gradient highway. |
| Multi-head attention (h parallel heads in d_k subspaces) | DEVELOPED | "It" pronoun problem motivates multiple relationship types. Research team analogy. Dimension splitting: d_k = d_model/h, same total FLOPs. Full worked example. |
| Output projection W_O as learned cross-head mixing | DEVELOPED | Not reshaping -- learned d_model^2 mixing layer. "Team meeting" analogy. Concatenation isolates; W_O synthesizes. |
| Head specialization (messy and emergent, not cleanly assigned) | INTRODUCED | 20-40% of heads prunable. "Capacity, not assignment" mental model. |
| Layer normalization (per-token feature normalization) | INTRODUCED | Same formula as batch norm, different axis. No train/eval distinction. Batch norm fails for variable-length sequences. |
| Pre-norm vs post-norm | INTRODUCED | Pre-norm (modern standard) keeps residual stream clean. GPT-2/3/LLaMA use pre-norm. |
| FFN structure and role (4x expansion, GELU, "writes" to residual stream) | DEVELOPED | 768->3072->768. ~2/3 of block parameters. Convex hull argument: attention can only blend, FFN can transform. Geva et al. FFN as key-value memories. |
| "Attention reads, FFN writes" mental model | DEVELOPED | Attention gathers context (reads stream), FFN processes and transforms (writes to stream). Complementary roles. |
| Transformer block as repeating unit (MHA + FFN + residual + layer norm) | DEVELOPED | Shape-preserving: (n, d_model) in and out. Enables stacking. GPT-2: 12 blocks. Color-coded architecture diagram. |
| Parameter distribution (~1/3 attention, ~2/3 FFN) | DEVELOPED | GPT-2: ~23% attention, ~46% FFN, ~31% embeddings. Reframes transformer from "attention" to "read-process cycle." |
| Causal masking (lower-triangular attention preventing future token leakage) | DEVELOPED | "Cheating problem" hook. Exam analogy. Set j>i entries to -infinity before softmax. Not a training trick -- mirrors inference reality. Training is parallel with masking; inference is sequential. |
| Full GPT architecture end-to-end | DEVELOPED | Token embedding + PE -> N blocks with causal masking -> final layer norm -> output projection -> softmax. GptArchitectureDiagram SVG. GPT-2 configuration. |
| Output projection / unembedding | INTRODUCED | nn.Linear(d_model, vocab_size). Weight-tied with token embeddings. Reverse of embedding. |
| GPT-2 total parameter counting (~124.4M verified) | DEVELOPED | Per-component breakdown matching known figure. Distribution: embeddings ~31%, attention ~23%, FFN ~46%. |
| Encoder-decoder vs decoder-only distinction | INTRODUCED | Three variants: encoder-only (BERT), encoder-decoder (T5), decoder-only (GPT). "Decoder" means causal masking, not "can only decode." |
| Why decoder-only won for LLMs | INTRODUCED | Simplicity, scaling, generality. GPT-2 (124M) vs GPT-3 (175B) = same architecture, different scale. |

### From Module 4.3: Building & Training GPT (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| GPT architecture implemented in PyTorch (Head, CausalSelfAttention, FeedForward, Block, GPT) | APPLIED | Five nn.Module classes, bottom-up assembly. Shape verification via assertions. Five PyTorch operations build the entire GPT. |
| Weight initialization for transformers (normal sigma=0.02, scaled residual projections) | DEVELOPED | Concrete activation statistics: default init std grows 0.82->6.55; scaled init stays 0.80-0.85. model.apply() pattern. |
| Parameter counting as architecture verification | APPLIED | Programmatic count verified against hand-computed ~124.4M. Prediction exercise before running code. |
| Autoregressive generation in code (generate method) | DEVELOPED | torch.no_grad(), crop to block_size, forward pass, last position logits, temperature, sample, append. |
| Weight tying (embedding and output projection share weights) | DEVELOPED | self.transformer.wte.weight = self.lm_head.weight. Saves ~38M parameters. data_ptr() verification. |
| Complete GPT training loop (forward, loss, backward, step + LR scheduling + gradient clipping) | APPLIED | Same heartbeat as MNIST with three additions. Side-by-side comparison demonstrates structural identity. |
| Text dataset preparation for language modeling | DEVELOPED | Tokenize, slice into context windows, input/target one-position offset. Concrete trace-through. |
| Cross-entropy for next-token prediction over 50K vocabulary | DEVELOPED | Reshape (B, T, V) to (B*T, V). Initial loss sanity check: -ln(1/50257) ~= 10.82. |
| Learning rate scheduling (linear warmup + cosine decay) | DEVELOPED | Warmup first 5% of steps, cosine decay remainder. SVG visualization. Motivated by constant LR failure. |
| Gradient clipping (clip_grad_norm_) | INTRODUCED | Global norm threshold, preserves direction, bounds magnitude. After backward(), before step(). |
| Loss curve interpretation for language models | DEVELOPED | Noisy curves normal, nonlinear loss-to-quality mapping, diagnostic question format. |
| Compute-bound vs memory-bound operations (arithmetic intensity) | INTRODUCED | GPU compute (312 TFLOPS) exceeds memory bandwidth (2 TB/s). Kitchen analogy. |
| Mixed precision with bfloat16 | DEVELOPED | Same exponent range as float32, less mantissa. No GradScaler needed. Master weights in float32 for accumulation. |
| KV caching for autoregressive inference | DEVELOPED | Cache K,V from previous steps. O(n) vs O(n^2). Concrete: 55x at 100 tokens, 500x at 1000. SVG diagram. |
| Flash attention (tiled computation, O(n) memory, numerically identical) | INTRODUCED | Never stores full n x n matrix. Built into PyTorch. 384 MB vs 384 KB at seq_len 4096. |
| Scaling laws (Chinchilla compute-optimal training) | INTRODUCED | N_opt ~ sqrt(C), D_opt ~ sqrt(C). Chinchilla outperforms Gopher. Power law L ~ C^(-0.05). |
| Weight name mapping between codebases | APPLIED | Translate HuggingFace parameter names to student model. Mapping as per-component architecture verification. |
| Conv1D vs nn.Linear weight transposition | DEVELOPED | HuggingFace Conv1D (in, out) vs nn.Linear (out, in). Pattern: transpose all 2D weights in attention and FFN. |
| Logit comparison as model verification | DEVELOPED | torch.allclose on output logits. Deterministic, definitive. Stronger than parameter counting or text comparison. |
| HuggingFace transformers library | INTRODUCED | GPT2LMHeadModel.from_pretrained("gpt2"). Minimal introduction as weight source only. |
| Verification chain (parameter count + logit comparison + coherent generation) | DEVELOPED | Three levels of evidence: structure, computation, behavior. Module synthesis. |

### From Module 4.4: Beyond Pretraining (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Adding a classification head to a pretrained LM | DEVELOPED | Replace lm_head with nn.Linear(d_model, num_classes). Last-token hidden state as sequence representation. Same pattern as CNN transfer learning. |
| Frozen backbone finetuning for transformers | DEVELOPED | requires_grad=False on transformer parameters. Train only tiny classification head (1,536 params for binary). |
| Full finetuning vs frozen backbone tradeoffs | INTRODUCED | Frozen = fast/safe/low memory. Full = slow/risky/higher accuracy. Overfitting argument. Partial unfreezing as middle ground. |
| Catastrophic forgetting in finetuned LMs | INTRODUCED | Full finetuning on narrow task risks destroying general language abilities. Observable in notebook. |
| Instruction dataset format (instruction/response pairs) | DEVELOPED | Multiple task types (factual, creative, coding, summarization). Alpaca 52K and LIMA 1K references. |
| Chat templates and special tokens as structural delimiters | DEVELOPED | ChatML format. Special tokens acquire meaning from SFT. Wrong-template negative example. |
| Loss masking / prompt masking (label = -100) | DEVELOPED | Compute loss only on response tokens. PyTorch ignore_index=-100 convention. LossMaskingDiagram SVG. |
| SFT teaches format, not knowledge | DEVELOPED | Central insight. Expert-in-monologue analogy. "Capital of France" dual-prompt evidence. |
| SFT training mechanics (same loop, same loss, different data) | APPLIED | Side-by-side with pretraining loop. 5 steps, 4 identical. No new head, no new loss. |
| The alignment problem (SFT-only models can be harmful, sycophantic, confidently wrong) | DEVELOPED | Three concrete failure modes. Format without quality signal. |
| Human preference data format (comparison pairs) | DEVELOPED | Relative signal (A > B) more reliable than absolute scoring. |
| Reward model architecture (pretrained LM + scalar head) | INTRODUCED | Same pattern as classification finetuning. Trained on preference pairs. |
| PPO for language models (generate-score-update with KL penalty) | INTRODUCED | Three-step loop. Two models. Breaks "same heartbeat" pattern. |
| KL penalty as soft constraint preventing reward hacking | INTRODUCED | Continuous version of "freeze the backbone." |
| DPO (preference optimization without separate reward model) | INTRODUCED | Directly adjusts probabilities. One model. Partially restores familiar loop. |
| Training memory breakdown (weights + gradients + optimizer states, ~12 bytes/param) | DEVELOPED | bf16 weights + bf16 gradients + fp32 Adam states = ~84 GB for 7B. Optimizer states dominate (two-thirds). |
| The memory wall problem (full finetuning does not scale) | DEVELOPED | ~84 GB for 7B model. Consumer GPU (24 GB) cannot fit. "Two problems, two solutions" framing. |
| Low-rank decomposition / matrix factorization | INTRODUCED | Large matrix as product of two thin matrices. 768x768 -> rank-8: 48x reduction. No SVD. |
| LoRA: Low-Rank Adaptation | DEVELOPED | h = Wx + BAx*(alpha/r). Freeze base, add trainable detour. B initialized to zeros. Applied to W_Q, W_V. Merge at inference. |
| Why LoRA works (low-rank constraint as regularization) | DEVELOPED | Matches or exceeds full finetuning. Implicit regularization. |
| Quantization (float to int8/int4 for inference) | DEVELOPED | Absmax walkthrough with concrete numbers. Outlier problem as negative example. Zero-point quantization. |
| QLoRA (quantized base + LoRA adapters) | INTRODUCED | 4-bit base + bf16 adapters. ~4 GB total for 7B model. ~21x more efficient than full finetuning. |
| GPTQ, AWQ (post-training quantization methods) | MENTIONED | Name-drop level. INT4 with less than 1% perplexity degradation. |
| The complete LLM pipeline as sequential dependency chain | DEVELOPED | CONSOLIDATE synthesis. Pretraining -> SFT -> alignment -> LoRA -> quantization. Each stage depends on the previous. Capstone FullPipelineDiagram SVG. |
| Pipeline stages mapped to open-source model artifacts | DEVELOPED | CONSOLIDATE synthesis. Base models, instruct models, LoRA adapters, quantized variants. Llama 3 family as concrete example. |
| The adaptation spectrum (all methods as points on "how much to change") | DEVELOPED | CONSOLIDATE synthesis. Extends "frozen backbone -> KL penalty -> LoRA" to full spectrum: no adaptation, classification head, LoRA/QLoRA, SFT, RLHF/DPO, full finetuning. |

## Key Mental Models Carried Forward

1. **"A language model approximates P(next token | context)"** -- Extension of "ML is function approximation." The defining objective.
2. **"Autoregressive generation is a feedback loop"** -- Outputs become inputs. Why same prompt produces different text (sampling).
3. **"Temperature changes sampling, not knowledge"** -- Model parameters fixed; only distribution shape changes.
4. **"Tokenization defines what the model can see"** -- Tokens are atomic units of perception. Below token level is invisible.
5. **"The merge table IS the tokenizer"** -- BPE encoding is deterministic once trained.
6. **"Token embedding + positional encoding = the model's input"** -- Complete pipeline: text -> tokens -> IDs -> embeddings + position -> tensor.
7. **"Embeddings are a learned dictionary"** -- Definitions start random, training refines until similar tokens cluster.
8. **"Without position, embeddings are a bag of words"** -- Order must be injected explicitly.
9. **"Attention is a weighted average where the input determines the weights"** -- Data-dependent weights vs fixed parameters. The conceptual revolution.
10. **"Similarity is not the same as relevance"** -- Raw dot products measure similarity; Q/K projections compute learned relevance.
11. **"Q and K are learned lenses -- same embedding, different lens, different view"** -- The projection matrix is the interesting learned component.
12. **"Three lenses, one embedding"** -- W_Q (seeking), W_K (advertising), W_V (contributing). Three different views from one input.
13. **"K gets you noticed; V is what you deliver"** -- Separates matching function from contribution function.
14. **"Multiple lenses, pooled findings"** -- Multi-head attention: h parallel heads, W_O synthesizes.
15. **"Split, not multiplied"** -- Dimension splitting is budget allocation. Same total FLOPs.
16. **"Capacity, not assignment"** -- Heads learn emergently, not by design.
17. **"Attention reads, FFN writes"** -- Complementary roles in the transformer block.
18. **"The residual stream is a shared document"** -- Each of 24 sub-layers reads and annotates it.
19. **"Causal masking simulates the inference constraint during training"** -- Future tokens do not exist at inference; mask ensures training matches.
20. **"The full GPT architecture is assembly, not invention"** -- Every piece is familiar; the lesson adds one mechanism and connects everything.
21. **"Decoder-only means causal masking, not 'can only decode'"** -- Historical name, not a description of capability.
22. **"Scale, not architecture"** -- GPT-2 and GPT-3 are the same blueprint with different numbers.
23. **"The simplest architecture that works is the one that scales"** -- Why decoder-only won.
24. **"The bottleneck is the delivery truck, not the chefs"** -- For memory-bound operations, faster GPU compute does nothing. Corrects "more cores = faster."
25. **"The math is elegant; the engineering makes it work"** -- Mixed precision, KV caching, flash attention, scaling laws separate prototypes from real systems.
26. **"Scale both, not just one"** -- Chinchilla: model size and data should grow together with sqrt(compute).
27. **"The mapping IS the verification"** -- Every shape match during weight loading is a component verified. Weight mapping is a per-component X-ray, not bookkeeping.
28. **"The architecture is the vessel; the weights are the knowledge"** -- Same code, different weights, dramatically different behavior. Code defines capability; weights encode learning.
29. **"Parameter count verifies structure; logit comparison verifies computation"** -- Two-level verification. Together: right shapes AND right function.
30. **"A pretrained transformer is a text feature extractor"** -- Extension of CNN transfer learning. Add a head, freeze the backbone, train the head. Applies to classification finetuning but NOT to SFT (explicitly corrected in Lesson 2).
31. **"SFT teaches format, not knowledge"** -- The base model already has vast knowledge from pretraining. SFT on instruction-response pairs teaches it to express that knowledge in conversational, instruction-following format. Expert-in-monologue analogy.
32. **"Same heartbeat, third time"** -- The training loop (forward, loss, backward, step) is identical across pretraining, classification finetuning, and SFT. Only the data changes.
33. **"SFT gives the model a voice; alignment gives it judgment"** -- Mute (base) to speaking (SFT) to speaking wisely (aligned). Each stage adds something essential the previous stage could not provide.
34. **"For the first time, the training loop changes shape"** -- PPO breaks the "same heartbeat" pattern. Generate-score-update with two models at response level, not token level. DPO partially restores familiar loop shape.
35. **"The reward model is an experienced editor"** -- Learned judgment from human comparisons, not rules. Has blind spots that can be exploited (reward hacking).
36. **"KL penalty is the continuous version of 'freeze the backbone'"** -- Soft constraint preventing drift from SFT model. Same purpose as frozen backbone (catastrophic forgetting prevention), but gradient rather than binary.
37. **"LoRA is the surgical version of 'freeze the backbone'"** -- Same philosophy (preserve pretrained knowledge, adapt minimally) but applied inside the model with tiny trainable detours alongside frozen weights. Highway and detour metaphor.
38. **"Finetuning is a refinement, not a revolution"** -- Weight changes during finetuning are low-rank because you are adjusting, not rewriting. The adaptation lives in a small subspace. This is why LoRA works.
39. **"Frozen backbone -> KL penalty -> LoRA" spectrum** -- Three approaches to adapting without forgetting, at increasing surgical precision. Binary constraint, soft constraint, targeted bypass.
40. **"The precision spectrum continues"** -- float32 -> bfloat16 -> int8 -> int4. Each step trades precision for memory, and neural networks tolerate it because weight distributions are compressible.
41. **"The pipeline is assembly, not invention"** -- Extension of the GPT architecture principle to the entire LLM pipeline. Each stage is a known technique; the pipeline is their careful composition. No stage can be skipped or reordered.
42. **"Every adaptation method answers the same question: how much should I change?"** -- Extension of the "frozen backbone -> KL penalty -> LoRA" spectrum to all adaptation methods in the module. A continuum from "change nothing" to "change everything."

## What This Series Has NOT Covered

- ~~Implementing any component in PyTorch~~ (done: building-nanogpt, Module 4.3)
- ~~Training a model (pretraining, loss curves, learning rate scheduling)~~ (done: pretraining, Module 4.3)
- ~~KV caching, flash attention, efficient inference~~ (done: scaling-and-efficiency, Module 4.3. KV caching and flash attention INTRODUCED/DEVELOPED. MoE, speculative decoding, continuous batching MENTIONED only.)
- ~~Loading real pretrained weights~~ (done: loading-real-weights, Module 4.3. Weight name mapping APPLIED, Conv1D transposition DEVELOPED, logit verification DEVELOPED, HuggingFace INTRODUCED.)
- ~~Classification finetuning~~ (done: finetuning-for-classification, Module 4.4. Add head, freeze backbone, train. DEVELOPED.)
- ~~Instruction tuning / SFT~~ (done: instruction-tuning, Module 4.4. Instruction datasets, chat templates, loss masking, "format not knowledge." DEVELOPED/APPLIED.)
- ~~RLHF, DPO, alignment~~ (done: rlhf-and-alignment, Module 4.4. Alignment problem DEVELOPED, human preference data DEVELOPED, reward models INTRODUCED, PPO intuition INTRODUCED, DPO INTRODUCED, reward hacking INTRODUCED. Conceptual lesson, no notebook.)
- ~~LoRA, quantization~~ (done: lora-and-quantization, Module 4.4. LoRA DEVELOPED, quantization DEVELOPED, low-rank decomposition INTRODUCED, QLoRA INTRODUCED, GPTQ/AWQ MENTIONED. PEFT and bitsandbytes libraries INTRODUCED. Notebook with 5 exercises.)
- Cross-attention mechanics in detail (Series 6)
- BERT architecture beyond name-drop
- Mixture of experts, sparse attention
- Multi-GPU or distributed training
- RoPE in detail (mentioned)

## Module Completion Notes

### Module 4.1 (complete)
Three lessons covering the complete input pipeline: what language models are (next-token prediction), how text becomes tokens (BPE), and how tokens become the tensor the model processes (embeddings + positional encoding). All three lessons built, two with notebooks. Student can trace the full path from raw text to model-ready tensor.

### Module 4.2 (complete)
Six lessons building the complete transformer architecture from scratch. The "feel the limitation before seeing the solution" approach: raw dot-product attention (limitation: one embedding for two roles) -> Q/K projections (fix matching, add scaling) -> V projection + residual stream (fix contribution) -> multi-head attention (capture diverse relationships) -> transformer block (assemble repeating unit with FFN, residual, layer norm) -> causal masking + complete GPT architecture (final assembly). Four lessons have notebooks. Student can explain the complete GPT architecture end-to-end, name every component, count every parameter, and articulate why decoder-only won for LLMs.

The narrative arc across the six lessons is the strongest in the course so far. Every mechanism arrives because the previous version was insufficient, creating a causal chain that makes the architecture feel inevitable rather than arbitrary. The module ends with the student able to look at the full GPT architecture diagram and recognize every component from lessons they built it in.

### Module 4.3 (complete)
Four lessons taking the student from architecture implementation through training, engineering, and verification. The module arc: Building nanoGPT (architecture, gibberish generation), Pretraining (training loop, Shakespeare), Scaling & Efficiency (engineering that makes it practical), Loading Real Weights (verification with real GPT-2 weights, coherent text generation). The cognitive load progression is BUILD -> STRETCH -> BUILD -> CONSOLIDATE, ending on a high note. Three lessons have notebooks. The student can implement a complete GPT, train it, understand the engineering trade-offs, load real pretrained weights, verify correctness via logit comparison, and generate coherent text from their own code. The verification chain (parameter count + logit comparison + coherent generation) provides three levels of evidence that the implementation is correct. The final lesson is the emotional capstone: the student's own code running real GPT-2, the "I built GPT" moment.

### Module 4.4 (complete)
Five lessons covering the full post-pretraining pipeline: classification finetuning (frozen backbone + task head), instruction tuning (SFT teaches format, not knowledge), alignment (RLHF/DPO gives judgment beyond format), practical adaptation (LoRA and quantization make it accessible), and a final synthesis lesson connecting the entire 18-lesson series into one coherent pipeline. The cognitive load progression is BUILD -> STRETCH -> BUILD -> STRETCH -> CONSOLIDATE. Three lessons have notebooks (Lessons 1, 2, 4). Lesson 3 (RLHF & Alignment) is conceptual due to implementation complexity. Lesson 5 (Putting It All Together) is a CONSOLIDATE synthesis with no new concepts, no notebook, no exercises -- the student sees the complete LLM pipeline as a single picture and can trace the journey from raw text to deployed model on their laptop.

The module establishes a clear dependency chain through adaptation methods: frozen backbone (Lesson 1) -> KL penalty (Lesson 3) -> LoRA (Lesson 4), all unified in the final lesson as points on an adaptation spectrum. The series capstone (Lesson 5) extends "assembly, not invention" from the GPT architecture to the entire pipeline, and maps pipeline stages to real open-source model artifacts the student can download and use.

### Series 4 (complete)
Eighteen lessons across four modules. From "what is a language model?" (4.1.1) to "here is the complete pipeline from raw text to aligned model on your laptop" (4.4.5). The student has implementation-level understanding of: tokenization (BPE, built from scratch), embeddings and positional encoding, the complete transformer architecture (attention, Q/K/V, multi-head, blocks, causal masking), GPT implementation and training, engineering optimizations (mixed precision, KV caching, flash attention), classification finetuning, instruction tuning, RLHF/DPO alignment, LoRA, and quantization. 42 mental models accumulated. The student can read a HuggingFace model card like "Llama 3 70B Instruct, 4-bit GPTQ" and explain every word.
