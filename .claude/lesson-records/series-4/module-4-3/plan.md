# Module 4.3: Building & Training GPT -- Plan

**Module goal:** The student can implement a complete GPT model in PyTorch, train it from scratch on a text corpus, understand the engineering decisions that make training work at scale, and verify their implementation by loading real GPT-2 weights and generating coherent text.

## Narrative Arc

The student arrives with complete conceptual understanding of every component in the GPT architecture. They can explain Q/K/V projections, multi-head attention, the transformer block, causal masking, and the full end-to-end architecture. They know the parameter count. They can trace a forward pass on paper. But they have never written a single line of transformer code.

This module is the bridge from understanding to building. The arc follows a natural implementation progression:

1. **Assemble the model** (Lesson 1) -- Turn the architecture diagram from Module 4.2 into working PyTorch code. Every component the student learned conceptually becomes a class they write. The payoff: generating random text from an untrained model. It is gibberish, but it runs. The architecture works.

2. **Train it** (Lesson 2) -- The untrained model generates nonsense. Fix that. Prepare a dataset, build a training loop, watch loss curves, implement learning rate scheduling, and observe text quality improving from random garbage to recognizable patterns. The "aha": training is not magic, it is the same loop from Series 2 applied to a larger model.

3. **Think about scale** (Lesson 3) -- The training loop works but is slow. What changes when you scale up? GPU utilization, mixed precision, KV caching for inference, flash attention, and the scaling laws that predict performance from compute budget. No notebook -- this is understanding the engineering, not implementing it.

4. **Load real weights** (Lesson 4) -- The ultimate validation: load OpenAI's GPT-2 weights into YOUR architecture. If the shapes match and the outputs are correct, your implementation is right. Generate coherent text from a real pretrained model using code you wrote. This is the "I built GPT" moment.

The key pedagogical move: the student is NOT learning the architecture here (that was Module 4.2). They are learning what it takes to make the architecture work in practice. The conceptual understanding is already solid; this module turns it into craft.

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| building-nanogpt | Translating architecture knowledge into PyTorch code | BUILD | Must come first -- everything else depends on having a working model. Connects conceptual understanding from Module 4.2 to nn.Module classes. Low novelty (every component is familiar), high satisfaction (it runs!). |
| pretraining | Training loop, dataset prep, LR scheduling, watching quality improve | STRETCH | Requires the model from Lesson 1. Introduces genuinely new content: dataset preparation, training dynamics at this scale, LR scheduling strategies. Watching text improve from gibberish to coherent is the emotional arc. |
| scaling-and-efficiency | GPU utilization, mixed precision, KV caching, flash attention, scaling laws | BUILD | Requires understanding training (Lesson 2) to appreciate what needs to be optimized. Conceptual lesson -- understanding trade-offs, not implementing them. |
| loading-real-weights | Loading GPT-2 weights, verifying against OpenAI, generating coherent text | CONSOLIDATE | The capstone. Validates everything: if real weights work in your model, every component is correct. Deeply satisfying closure. Requires the model from Lesson 1 and understanding from Lessons 2-3. |

## Rough Topic Allocation

- **Lesson 1 (building-nanogpt):** Translate the architecture diagram into PyTorch classes: token + positional embedding, single attention head, multi-head attention, feed-forward network, transformer block, full GPT model. Weight initialization (why random isn't good enough). Forward pass walkthrough tracing shapes at every layer. Config dataclass mirroring GPT-2 hyperparameters. Generate random text from the untrained model (the fun payoff). Notebook: build and run the model.

- **Lesson 2 (pretraining):** Dataset preparation (character-level or BPE tokenized TinyShakespeare), creating training batches with context windows, the training loop (forward pass, loss computation, backward pass, optimizer step), cross-entropy loss for next-token prediction, learning rate scheduling (warmup + cosine decay), gradient clipping, tracking and plotting loss curves, periodically generating text to observe quality improvement. Notebook: full training pipeline.

- **Lesson 3 (scaling-and-efficiency):** What changes at scale: memory bottlenecks, GPU utilization and compute-bound vs memory-bound operations, mixed precision training (float16/bfloat16 with loss scaling), KV caching for autoregressive inference (why recomputing attention each step is wasteful), flash attention (the algorithmic insight, not the implementation), scaling laws (Chinchilla: how to allocate compute budget between model size and data). No notebook -- conceptual understanding of engineering trade-offs.

- **Lesson 4 (loading-real-weights):** Mapping your module names to OpenAI's weight names, handling weight tying (embedding and output projection share weights), loading and reshaping weights, verifying outputs match OpenAI's implementation for the same input, generating coherent text. The emotional closure: real GPT-2 runs on code you wrote. Notebook: load, verify, generate.

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| building-nanogpt | BUILD | Every component is conceptually familiar from Module 4.2. The new skill is translating understanding into nn.Module code. Low novelty, high craft satisfaction. |
| pretraining | STRETCH | Training dynamics, LR scheduling, and dataset preparation are genuinely new. The student has trained models before (Series 2) but not at this scale or complexity. Emotional payoff (watching text improve) sustains motivation. |
| scaling-and-efficiency | BUILD | Understanding trade-offs and engineering decisions. No implementation required. Connects to familiar concepts (memory, compute) but applies them to transformer-specific contexts. |
| loading-real-weights | CONSOLIDATE | The only new skill is weight mapping. Everything else is verification and celebration. Capstone moment that validates the entire module. |

## Module-Level Misconceptions

- **"Building a transformer in PyTorch requires advanced/unfamiliar operations."** Every component is nn.Linear, nn.Embedding, nn.LayerNorm, or nn.GELU -- building blocks the student already knows from Series 2. The architecture is complex but the code is not.

- **"Weight initialization doesn't matter."** With default PyTorch initialization, deep transformers can have exploding or vanishing activations. Proper initialization (Xavier/He for linear layers, small std for residual projections) is essential for training stability.

- **"Training a language model is fundamentally different from training the models in Series 2."** The loop is identical: forward pass, compute loss, backward pass, update weights. The differences are scale (more parameters, more data), the loss function (cross-entropy over vocabulary), and learning rate scheduling. The pattern is the same.

- **"KV caching is an optimization trick."** It is the standard way to do autoregressive inference. Without it, you recompute all previous tokens' attention at every generation step. With it, you cache and reuse. The "trick" is actually the baseline expectation.

- **"If my model architecture is correct, loading pretrained weights should 'just work.'"** Weight names, tensor shapes, transpositions, and weight tying all need to be handled carefully. A correct architecture with wrong weight mapping produces garbage. The mapping IS the verification.
