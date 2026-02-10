# Module 4.1: Language Modeling Fundamentals — Plan

**Module goal:** The student understands what language models are (probability distributions over token sequences), how text is represented for neural networks (tokenization and embeddings), and why position information requires special handling — giving them the complete input pipeline before attention is introduced in Module 4.2.

## Narrative Arc

The student has spent three series training neural networks on numbers and images. They know how to feed structured data (house prices) and spatial data (pixels) into models. But language is different: it's sequential, variable-length, and its meaning depends on order. This module builds the bridge from "I can train models on numeric data" to "I understand how text becomes the kind of input a neural network can process."

The arc follows three natural questions:
1. **What task are we even solving?** (Lesson 1) — Language modeling is next-token prediction. That single task, scaled up, produces ChatGPT. The student already understands predicting targets from inputs — this reframes language generation as the same pattern.
2. **How does text become numbers?** (Lesson 2) — You can't multiply words by weights. Tokenization converts text into integer sequences, and the choice of tokenization scheme profoundly affects what the model can learn.
3. **How do those numbers become vectors the model can use?** (Lesson 3) — Embeddings are learned lookup tables (just a matrix of weights), and unlike images, the model has no built-in notion of position — you have to inject it.

By the end, the student can trace the full path from raw text to the tensor that enters a transformer block: text -> tokens -> token IDs -> embedding vectors + positional encoding -> input representation. Module 4.2 picks up from there with attention.

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| what-is-a-language-model | Next-token prediction as the universal task | STRETCH | Must come first — establishes the entire framing for Series 4. New domain (language), new task formulation (autoregressive generation). High novelty but connects to familiar supervised learning. |
| tokenization | Subword tokenization (BPE) | BUILD | Depends on understanding what a "token" is from Lesson 1. Hands-on notebook implementing BPE. Bridges the abstract "predict the next token" to concrete "here's how text becomes integers." |
| embeddings-and-position | Token embeddings + positional encoding | BUILD | Depends on token IDs from Lesson 2. Completes the input pipeline. Embeddings connect to "learnable parameters" from Series 1. Positional encoding is genuinely new but the student has the conceptual foundation. |

## Rough Topic Allocation

- **Lesson 1 (what-is-a-language-model):** Probability distributions over sequences, conditional probability P(next | context), autoregressive generation (sample one token, append, repeat), next-token prediction as supervised learning, temperature and sampling strategies, why this single task produces useful models. Conceptual lesson, no notebook.
- **Lesson 2 (tokenization):** Why raw text needs conversion, character-level vs word-level tradeoffs, subword tokenization as the practical middle ground, BPE algorithm step-by-step, implement BPE from scratch in notebook, vocabulary size effects on model behavior.
- **Lesson 3 (embeddings-and-position):** Token IDs to vectors via embedding lookup (nn.Embedding is just a matrix indexed by integer), why one-hot encoding fails at scale, why position matters for sequences (bag-of-words loses "dog bit man" vs "man bit dog"), sinusoidal positional encoding, learned positional encoding, combining token + position embeddings. Notebook: implement embedding + positional encoding, visualize embedding space.

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| what-is-a-language-model | STRETCH | New domain, new task formulation (autoregressive), probability over sequences. Lots of reframing. No notebook keeps implementation load zero. |
| tokenization | BUILD | One core algorithm (BPE), hands-on implementation, builds directly on Lesson 1's "token" concept. |
| embeddings-and-position | BUILD | Two related concepts (embeddings, positional encoding) but embeddings connect strongly to existing knowledge (learnable parameters, weight matrices). Positional encoding is new but motivated by clear problem. |

## Module-Level Misconceptions

- **"Language models understand language / are intelligent."** The student uses ChatGPT daily and may attribute understanding to the model. This module should consistently frame LMs as probability machines — predict the next token, nothing more. Whether that constitutes "understanding" is a philosophical question, not an engineering one.
- **"Tokens are words."** The tokenization lesson will address this directly, but Lesson 1 should be careful not to reinforce this assumption. Use "token" from the start, define it, and note that what counts as a token depends on the tokenizer.
- **"Embeddings are a preprocessing step (like normalization)."** Embeddings are learned parameters — they update during training just like weights. This connects to the "parameters are knobs" mental model from Series 1.
- **"Word order doesn't matter much / the model just looks at which words are present."** The positional encoding lesson exists precisely because a naive embedding approach creates a bag-of-words. The student needs to feel why order matters before seeing the solution.
- **"Temperature controls how 'creative' the model is."** Temperature controls the entropy of the probability distribution. High temperature = more uniform = more surprising choices. "Creativity" is a human interpretation of sampling from a flatter distribution.
