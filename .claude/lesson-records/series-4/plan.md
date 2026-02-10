# Series 4: LLMs & Transformers — Plan

**Status:** Planned
**Prerequisites:** Series 2 (PyTorch), Series 3 (CNNs — specifically ResNets/skip connections)

## Series Goal

Build deep, implementation-level understanding of transformers and LLMs. The student should be able to build a GPT from scratch, understand every line, and have the mental models to read real LLM papers. By the end: you've pretrained a small GPT, loaded real GPT-2 weights into your architecture, and understand the full pipeline from pretraining through instruction tuning and alignment.

## Pedagogical Approach

Same "feel the problem before seeing the solution" pattern that worked in Series 1:
- Language modeling with naive approaches before introducing attention
- Raw dot-product attention before Q/K/V
- Single-head before multi-head
- Each mechanism arrives because the previous version was insufficient, not because "the paper says so"

### KQV Deep Dive — Design Notes

The student has persistent difficulty with Q, K, V. Standard analogies ("library lookup", "database query", "verb in a sentence") are lossy and don't build real understanding. The core problems:

1. **Analogies imply Q/K/V are inherent properties of tokens.** In reality, they're learned linear projections of the same embedding — the same token produces different vectors depending on whether it's asking (Q), advertising (K), or contributing (V).
2. **The formula feels monolithic.** `softmax(QK^T/√d)V` looks like one operation. It needs to be decomposed into stages the student can trace by hand with tiny examples (4 tokens, 3-dim embeddings).
3. **The asymmetry between Q and K isn't obvious.** Why can't one vector serve both roles? Lesson 4 should make the student feel this: raw dot-product attention (no projections) kinda works but every token has ONE representation for both "what I seek" and "what I offer." That tension motivates Q/K.
4. **V as a third projection is the least intuitive.** Why not just use the embedding directly? Because matching (Q/K) and contributing (V) are different tasks — what makes a token relevant isn't the same as what information it should provide when attended to.

**Solution:** Three dedicated lessons (4, 5, 6) that build attention from scratch. Lesson 4 shows attention WITHOUT Q/K/V so the student feels the limitation. Lesson 5 introduces Q and K as the fix. Lesson 6 adds V. Interactive widgets where you manipulate projection matrices and trace the actual numbers are critical here.

## Modules and Lessons

### Module 4.1: Language Modeling Fundamentals (3 lessons)

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 1 | what-is-a-language-model | What is a Language Model? | No | Probability over sequences, autoregressive generation, next-token prediction as the universal task, temperature and sampling |
| 2 | tokenization | Tokenization | Yes | Character vs word vs subword, BPE from scratch, how tokenization choices affect model behavior |
| 3 | embeddings-and-position | Embeddings & Positional Encoding | Yes | Token embeddings as lookup tables, why position matters for sequences (unlike images), sinusoidal vs learned positional encodings |

### Module 4.2: Attention & the Transformer (6 lessons)

This is the intellectual core. Three lessons (4-6) are dedicated to building attention intuition piece by piece, then three lessons (7-9) assemble the full architecture.

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 4 | the-problem-attention-solves | The Problem Attention Solves | Yes | Context-dependent meaning, bag-of-words limitations, dot-product attention WITHOUT Q/K/V — feel the limitation of one representation per token |
| 5 | queries-and-keys | Queries, Keys, and the Relevance Function | Yes | The asymmetry insight: seeking vs advertising, Q and K as learned projections, QK^T as a relevance matrix, scaling factor, trace by hand with tiny examples |
| 6 | values-and-attention-output | Values and the Attention Output | Yes | V as "what I contribute when attended to", the full single-head attention operation, residual stream concept, implement from scratch |
| 7 | multi-head-attention | Multi-Head Attention | Yes | Multiple notions of relevance in parallel, concatenation + output projection, visualizing what different heads attend to |
| 8 | the-transformer-block | The Transformer Block | No | MHA → add & norm → FFN → add & norm, residual connections (callback to ResNets), layer norm, "attention reads, FFN writes" |
| 9 | decoder-only-transformers | Decoder-Only Transformers (GPT Architecture) | No | Causal masking, why decoder-only won for LLMs, full architecture assembly, parameter counting, brief encoder-decoder contrast for Series 6 prep |

### Module 4.3: Building & Training GPT (4 lessons)

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 10 | building-nanogpt | Building nanoGPT | Yes | Assemble the full model in PyTorch, weight initialization, forward pass walkthrough, generating random text from an untrained model |
| 11 | pretraining | Pretraining: The Training Loop | Yes | Dataset preparation (TinyShakespeare or similar), training loop, loss curves, learning rate scheduling, watching text quality improve |
| 12 | scaling-and-efficiency | Scaling & Efficiency | No | What changes at scale: GPU utilization, mixed precision, KV caching, flash attention, scaling laws intuition |
| 13 | loading-real-weights | Loading Real Weights | Yes | Load GPT-2 weights into your architecture, verify against OpenAI's implementation, generate coherent text — the "aha" moment |

### Module 4.4: Beyond Pretraining (5 lessons)

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 14 | finetuning-for-classification | Finetuning for Classification | Yes | Pretrained model → task-specific adaptation, freezing layers, adding task heads (callback to transfer learning from CNNs) |
| 15 | instruction-tuning | Instruction Tuning (SFT) | Yes | How chat models differ from base models, instruction datasets, supervised finetuning, chat templates |
| 16 | rlhf-and-alignment | RLHF & Alignment | No | Reward models, PPO basics, DPO as the simpler alternative, why alignment matters |
| 17 | lora-and-quantization | LoRA, Quantization & Inference | Yes | Parameter-efficient finetuning, quantization for running big models on small hardware, KV caching for fast inference |
| 18 | putting-it-all-together | Putting It All Together | No | The full LLM pipeline: pretrain → SFT → alignment → serve. Mental model synthesis, what Series 5 will build on |

## Scope Boundaries

**In scope:**
- Decoder-only transformer architecture (GPT family)
- Full attention mechanism with deep KQV intuition
- Pretraining, supervised finetuning, basic RLHF/DPO
- Building and training a small GPT from scratch
- Loading and using real pretrained weights
- LoRA and quantization basics

**Out of scope:**
- Encoder-only models (BERT) — mentioned for contrast, not built
- Encoder-decoder details — brief mention for Series 6 prep (U-Net cross-attention)
- Advanced alignment techniques (constitutional AI, RLAIF) — Series 5
- Reasoning models, chain-of-thought training — Series 5
- Multimodal models — Series 5
- Mixture of experts — mentioned in scaling lesson, not implemented

## Key Resources & Inspiration

- **Sebastian Raschka, "Build a Large Language Model (From Scratch)"** — Closest match to our progression. Tokenization → attention → GPT → pretraining → finetuning. Implementation-heavy, runs on a laptop.
- **Andrej Karpathy, "Zero to Hero"** — Pedagogical moves: simple language models before transformers, "Let's build GPT" video, "Let's reproduce GPT-2 (124M)".
- **3Blue1Brown transformer videos** — Best visual intuitions for attention mechanics and how MLPs store facts. Reference for widget design.
- **Harvard NLP, "The Annotated Transformer"** — Literate programming walkthrough of the original paper. Reference for implementation correctness.
- **Jay Alammar, "The Illustrated Transformer"** — Visual reference for attention diagrams.

## Notebook Convention

Same as Series 2-3: `{series}-{module}-{lesson}-{slug}.ipynb`

Example: `4-2-4-the-problem-attention-solves.ipynb`

## Connections to Earlier Series

| LLM Concept | Earlier Concept | Source |
|------------|----------------|--------|
| Embeddings as learned lookup | Weights as learnable parameters (1.1) | Foundations |
| Attention weights via softmax | Softmax for classification (2.2) | PyTorch |
| Residual connections in transformer blocks | Skip connections in ResNets (3.2) | CNNs |
| Layer normalization | Batch normalization (1.3, 3.2) | Foundations/CNNs |
| Transfer learning / finetuning | Feature extraction vs fine-tuning (3.2, 3.3) | CNNs |
| Causal masking | Train/val split, data leakage concepts (1.1) | Foundations |
| Scaling factor √d_k | Vanishing gradients, softmax saturation (1.3) | Foundations |
