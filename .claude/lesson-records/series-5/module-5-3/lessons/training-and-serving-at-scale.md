# Lesson Plan: Training & Serving at Scale

**Module:** 5.3 (Scaling Architecture)
**Position:** Lesson 3 of 3 (module capstone)
**Slug:** `training-and-serving-at-scale`
**Cognitive Load:** CONSOLIDATE
**Previous Lesson Load:** BUILD (long-context-and-efficient-attention)

---

## Phase 1: Student State

### Relevant Concepts (with depths and sources)

| Concept | Depth | Source Lesson | How Established |
|---------|-------|---------------|-----------------|
| Compute-bound vs memory-bound operations (arithmetic intensity) | INTRODUCED | scaling-and-efficiency (4.3.3) | Kitchen analogy: fast chefs, slow delivery truck. GPU compute throughput (312 TFLOPS) vastly exceeds memory bandwidth (2 TB/s). Student can classify operations by arithmetic intensity. |
| Training memory breakdown (weights + gradients + optimizer states = ~12 bytes/param for mixed-precision Adam) | DEVELOPED | lora-and-quantization (4.4.4) | Four PhaseCards with concrete arithmetic: bf16 weights (14 GB) + bf16 gradients (14 GB) + fp32 Adam momentum (28 GB) + fp32 Adam variance (28 GB) = ~84 GB for 7B model. Optimizer states dominate at two-thirds of total. |
| Mixed precision with bfloat16 (master weights in fp32, forward/backward in bf16) | DEVELOPED | scaling-and-efficiency (4.3.3) | Same range as float32, less mantissa. Master weights pattern for accumulation. The "mixed" is the essential part. |
| KV caching for autoregressive inference | DEVELOPED | scaling-and-efficiency (4.3.3) | Cache K/V from previous steps, compute Q/K/V only for new token. O(n) vs O(n^2). SVG diagram of cache growth. Memory cost at GPT-2 scale: ~37.7 MB/sequence at 1024. |
| Autoregressive generation loop (generate() method) | DEVELOPED | building-nanogpt (4.3.1) | torch.no_grad(), crop to block_size, forward, take last position logits, sample, append. The student implemented this from scratch. |
| Speculative decoding | MENTIONED | scaling-and-efficiency (4.3.3) | "Small fast model drafts tokens, large model verifies in parallel." Name-drop only, no mechanism. |
| Continuous batching | MENTIONED | scaling-and-efficiency (4.3.3) | "Slot new requests into completed batch positions during serving." Name-drop only, no mechanism. |
| MoE memory tradeoff (all parameters must be loaded, only subset activates) | INTRODUCED | mixture-of-experts (5.3.1) | Mixtral uses ~3.5x more memory than its active parameter count suggests. MoE models need more memory to store inactive experts, motivating distribution across multiple GPUs. |
| Parameter-compute decoupling (total parameters >> active parameters) | DEVELOPED | mixture-of-experts (5.3.1) | Concretized with Mixtral 8x7B: ~47B total, ~13B active. Student understands that memory scales with total parameters while compute scales with active parameters. |
| GQA for KV cache reduction | DEVELOPED | long-context-and-efficient-attention (5.3.2) | Sharing K/V across Q head groups. LLaMA 2 70B: MHA ~332 GB vs GQA ~42 GB KV cache. Student can reason about cache memory at production scale. |
| Chinchilla scaling laws (compute-optimal training: scale both model and data) | INTRODUCED | scaling-and-efficiency (4.3.3) | N_opt ~ sqrt(C), D_opt ~ sqrt(C). Power law: doubling compute reduces loss by ~3.4%. Student can reason about compute budgets but has not applied this to multi-GPU settings. |

### Mental Models and Analogies Already Established

- **"The bottleneck is the delivery truck, not the chefs"** -- memory-bound vs compute-bound. Student already knows that faster compute does not help memory-bound operations.
- **"Attention reads, the right experts write"** -- MoE as targeted modification of the transformer block. Experts are distributed knowledge stores.
- **"Three barriers, three targeted solutions"** -- position (RoPE), compute (sparse attention), memory (GQA). Independent bottlenecks, independent solutions.
- **"The math is elegant; the engineering makes it work"** -- from scaling-and-efficiency. Mixed precision, KV caching, flash attention are not afterthoughts but what separates research prototypes from real systems. This lesson is the fullest expression of this mental model.
- **Training memory breakdown** -- the student has concrete intuition that optimizer states dominate training memory (2/3 of total). This is the direct setup for ZeRO.

### What Was Explicitly NOT Covered (Relevant Here)

- Multi-GPU or distributed training (explicitly deferred from scaling-and-efficiency AND pretraining lessons)
- Communication overhead between GPUs (deferred from mixture-of-experts to this lesson)
- Ring attention or sequence parallelism (deferred from long-context-and-efficient-attention to this lesson)
- Inference serving optimizations in depth (deferred from scaling-and-efficiency)

### Readiness Assessment

The student is well-prepared. This is a CONSOLIDATE lesson that develops concepts the student already has at MENTIONED depth (speculative decoding, continuous batching) and introduces engineering concepts (parallelism strategies, ZeRO) that connect directly to established knowledge. The training memory breakdown from lora-and-quantization provides the concrete arithmetic intuition that makes ZeRO immediately motivating ("optimizer states are 2/3 of training memory -- what if you could split them across GPUs?"). The generate() loop from building-nanogpt provides the foundation for understanding speculative decoding's draft-and-verify mechanism. No prerequisite gaps block this lesson.

---

## Phase 2: Analysis

### Target Concept

This lesson teaches the student to explain how training and inference workloads are distributed across multiple GPUs, identify which parallelism strategy addresses which bottleneck, and describe how speculative decoding and continuous batching transform autoregressive inference from a sequential bottleneck into a practical serving system.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Training memory breakdown (weights + gradients + optimizer states) | DEVELOPED | DEVELOPED | lora-and-quantization (4.4.4) | OK | ZeRO optimizer sharding is motivated by the dominance of optimizer states in training memory. Student has concrete 7B arithmetic (84 GB total, 56 GB optimizer states). |
| Compute-bound vs memory-bound operations | INTRODUCED | INTRODUCED | scaling-and-efficiency (4.3.3) | OK | Parallelism strategies are motivated by which bottleneck dominates. INTRODUCED sufficient -- lesson needs the concept but not application-level mastery. |
| Autoregressive generation loop (generate() method) | DEVELOPED | DEVELOPED | building-nanogpt (4.3.1) | OK | Speculative decoding modifies the generate() loop. Student implemented this and understands each step (forward pass, last logits, sample, append). |
| KV caching for inference | DEVELOPED | DEVELOPED | scaling-and-efficiency (4.3.3) | OK | Speculative decoding relies on KV caching for the verify step. Student understands cache growth, memory cost, and why caching is essential at scale. |
| Transformer block architecture (attention + FFN + residual) | DEVELOPED | APPLIED | building-nanogpt (4.3.1) | OK | Tensor parallelism splits within layers and pipeline parallelism splits across layers. Student built the full architecture in PyTorch. |
| MoE memory tradeoff (all params loaded, subset activates) | INTRODUCED | INTRODUCED | mixture-of-experts (5.3.1) | OK | MoE experts distributed across GPUs motivates tensor parallelism. Student knows Mixtral's memory footprint exceeds a single GPU. |
| Parameter-compute decoupling | DEVELOPED | DEVELOPED | mixture-of-experts (5.3.1) | OK | Understanding that memory scales with total params while compute scales with active params is needed to reason about parallelism for MoE models. |
| Speculative decoding (name recognition) | MENTIONED | MENTIONED | scaling-and-efficiency (4.3.3) | OK | Lesson develops this from MENTIONED to DEVELOPED. Name recognition is the expected starting point. |
| Continuous batching (name recognition) | MENTIONED | MENTIONED | scaling-and-efficiency (4.3.3) | OK | Lesson develops this from MENTIONED to DEVELOPED. Name recognition is the expected starting point. |
| Batched training (DataLoader, batch dimension) | APPLIED | APPLIED | pretraining (4.3.2) | OK | Static batching during inference is the baseline that continuous batching improves upon. Student ran batched training loops. |

**Gap resolution:** No gaps. All prerequisites are at or above required depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **1. "Parallelism is just data parallelism -- split the data, replicate the model"** | Data parallelism is the most intuitive form and is how most people first encounter multi-GPU training. The student has used DataLoader with batches in Series 2. The mental model "split the work across workers" naturally maps to "split the data." | A 70B model requires ~840 GB for training (weights + gradients + optimizer). A single A100 has 80 GB. Even the data cannot be processed because the model itself does not fit on one GPU. Data parallelism requires each GPU to hold the full model -- it cannot solve the memory problem. | Explain section, right after data parallelism is presented. Transition from "what data parallelism cannot do" to tensor and pipeline parallelism. |
| **2. "Tensor parallelism and pipeline parallelism are the same thing (just splitting the model)"** | Both are model parallelism strategies. Without careful distinction, "split the model across GPUs" sounds like one idea. The student may conflate splitting within a layer (tensor) with splitting across layers (pipeline). | Tensor parallelism: split a single matrix multiply across 2 GPUs. Both GPUs compute simultaneously on the same token. GPU-to-GPU communication happens WITHIN each layer (must synchronize partial results). Pipeline parallelism: GPU 1 runs layers 0-11, GPU 2 runs layers 12-23. Communication happens only BETWEEN stages. Different communication patterns, different bottlenecks, different use cases. | Explain section, immediately after introducing both. Side-by-side comparison with communication pattern diagrams. |
| **3. "Speculative decoding makes the model faster by using a smaller model"** | The student knows small models are faster. "Small model drafts, large model verifies" sounds like a speed hack -- just use the fast model. The misconception is that the speed comes from the small model being fast, rather than from parallelizing the sequential bottleneck. | If the small model alone were sufficient, you would just use the small model. The speed comes from the large model verifying multiple draft tokens IN PARALLEL in a single forward pass, rather than generating them one at a time. A single forward pass through the large model is the same cost whether processing 1 token or 5 tokens (compute-bound matmul scales with sequence length, not number of new tokens). The large model is doing the same total work -- it is just doing it in parallel. | Explain section on speculative decoding, after the mechanism is presented. Address by walking through the cost analysis. |
| **4. "Communication between GPUs is fast because they are in the same machine"** | Software engineers think of inter-process communication as fast. The student's experience is with single-GPU training where there is no communication overhead. PCIe or NVLink bandwidth sounds large in absolute terms. | An A100 GPU computes at ~312 TFLOPS (bf16). NVLink bandwidth is ~600 GB/s. Transferring even a modest tensor (1 GB) takes ~1.7 ms -- during which the GPU could have done ~530 billion floating-point operations. At scale, training a 175B model with naive tensor parallelism across 8 GPUs can spend 40-60% of time waiting for communication. Communication is the dominant constraint, not compute. | Introduce after data parallelism, as the central constraint that shapes all parallelism design. Make it the "pivot" -- the problem statement that motivates everything else. |
| **5. "Continuous batching is just using a bigger batch size"** | The student knows batched training from Series 2 and batched inference conceptually. "Fill completed slots with new requests" sounds like dynamic batch sizing. The misconception collapses continuous batching into static batching with variable size. | Static batching: 8 requests start together. Request 3 finishes at 20 tokens, but the batch waits until request 7 finishes at 200 tokens. GPU generates 180 padding tokens for request 3, doing work that produces nothing. Continuous batching: when request 3 finishes at 20 tokens, its slot is immediately filled with request 9 from the queue. GPU utilization stays near 100% instead of declining as short requests complete. The improvement is in slot utilization, not batch size. | Explain section on continuous batching, right after the static batching baseline. Visual showing wasted compute in static vs continuous. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Training 70B model: single GPU impossibility** | Positive | Motivates parallelism. Concrete numbers: 70B params x 12 bytes/param = 840 GB. A100 = 80 GB. The model does not fit, period. | Uses the student's existing training memory breakdown (from 4.4.4) at a scale they have not considered. The arithmetic is trivial but the conclusion is visceral. |
| **Data parallelism for a model that fits on one GPU** | Positive | Shows the simplest parallelism strategy. GPT-2 124M fits easily. 4 GPUs = 4x throughput. Each GPU processes different data, computes gradients, all-reduce averages them. | Starts with what the student's model CAN do. Same 124M model they built. Accessible, concrete, connects to their experience. |
| **Tensor parallelism: splitting a single linear layer across 2 GPUs** | Positive | Shows the core mechanism. A linear layer W of shape (d_model, 4*d_model) is split column-wise across 2 GPUs. Each GPU stores half the columns, computes half the output, then they share results. | A single linear layer is the simplest possible unit of tensor parallelism. Connects directly to the nn.Linear the student has built dozens of times. |
| **Pipeline parallelism: 4 GPUs, 12 layers each for a 48-layer model** | Positive | Shows the layer-splitting strategy. GPU 0 runs layers 0-11, GPU 1 runs layers 12-23, etc. One microbatch flows through the pipeline. | Natural extension of the Block class the student implemented. "Your model has 12 blocks. What if different GPUs ran different blocks?" |
| **Pipeline bubble (idle time when pipeline is partially filled)** | Negative | Shows pipeline parallelism's key inefficiency. With 4 stages and 1 microbatch, only 1 GPU is active at a time. 75% of GPUs are idle. Must fill the pipeline with microbatches to amortize. | Prevents the misconception that pipeline parallelism is free. The student needs to see that splitting across GPUs creates a new problem (bubbles) that requires a new solution (microbatching). |
| **Speculative decoding: 5 draft tokens verified in one large-model forward pass** | Positive | Develops from MENTIONED. Small model (7B) generates 5 candidate tokens. Large model (70B) runs one forward pass on all 5 to verify. Accepts 3, rejects 2. Net: 3 tokens from one large forward pass instead of 3 separate passes. | Connects directly to the generate() loop the student built. The mechanism is a modification of that exact loop. Concrete token counts make the speedup tangible. |
| **Static batching waste: batch of 8 requests with varying lengths** | Negative | Motivates continuous batching. Request lengths: [20, 45, 200, 15, 80, 150, 30, 200]. After 20 tokens, 2 requests are done but the batch runs until 200. Wasted compute shown. | Concrete numbers make the waste visceral. The student can compute the wasted GPU-token-seconds themselves. |

### Gap Resolution

No gaps identified. All prerequisites are at or above required depth.

---

## Phase 3: Design

### Narrative Arc

You have spent two lessons learning how to make the transformer itself more powerful -- MoE decoupled parameters from compute, and long-context techniques broke through the sequence length wall. But there is a gap between "here is a better architecture" and "here is a model serving millions of users." A Mixtral 8x7B model has 47 billion parameters. Even before training starts -- before a single gradient is computed -- those parameters do not fit on a single GPU. And once the model is trained, generating text one token at a time means a 70B model sits idle between forward passes while the next token is sampled. This lesson closes the gap: how does a model that is too large for one GPU get trained, and how does a model that generates one token at a time serve thousands of concurrent requests? The answers are engineering solutions to concrete bottlenecks, and every solution introduces a new tradeoff. The connecting thread: communication between devices is the constraint that shapes everything.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual (inline SVG diagrams)** | (1) ParallelismComparisonDiagram: three-panel showing data parallelism (full models, split data), tensor parallelism (split layers, split data), and pipeline parallelism (sequential stages, microbatch flow). (2) PipelineBubbleDiagram: timeline showing GPU idle time with 1 microbatch vs 4 microbatches. (3) SpeculativeDecodingDiagram: draft model producing 5 tokens, large model verifying in one pass, accept/reject visualization. (4) ContinuousBatchingDiagram: timeline comparing static batching (wasted slots) vs continuous batching (filled slots). | Parallelism strategies are spatial (how computation maps to devices). Speculative decoding and continuous batching are temporal (how work is scheduled over time). Both demand visual representation because the core insight is about arrangement, not formula. |
| **Concrete example (worked numbers)** | Training memory arithmetic for 70B model (weights + gradients + optimizer = 840 GB vs 80 GB GPU). ZeRO stage arithmetic (optimizer states alone = 560 GB; across 8 GPUs = 70 GB each). Speculative decoding token count (5 drafted, 3 accepted = 3 tokens per large forward pass vs 3 separate passes = 3x speedup for those tokens). Continuous batching utilization calculation (static: 8 requests, 200 max length, 2 finish at 20 = 22.5% wasted; continuous: near 0% wasted). | The student's strongest learning pattern is concrete-numbers-first. The training memory breakdown from lora-and-quantization was taught with specific arithmetic and became immediately intuitive. Same approach here. |
| **Symbolic (pseudocode)** | (1) Data parallelism all-reduce pseudocode. (2) Speculative decoding loop showing draft-then-verify. (3) Continuous batching slot management pseudocode. | The student is a software engineer. Pseudocode is their native representation. Seeing the mechanism as code makes it immediately graspable. |
| **Verbal/Analogy** | (1) "Assembly line" for pipeline parallelism (each station does one step; pipeline is inefficient for one item but fast for many). (2) "Rough draft and editor" for speculative decoding (fast writer produces draft, careful editor verifies in parallel). (3) "Restaurant with a waitlist" for continuous batching (as a table opens, the next party is seated immediately rather than waiting for all tables to clear). | Each analogy maps to the student's non-technical experience, providing intuitive access before the technical mechanism. |
| **Intuitive ("of course" moment)** | "Communication is the constraint. Of course it is. You already knew this. When you studied compute-bound vs memory-bound, you learned that moving data is slower than computing on it. Parallelism just makes this worse: now you are moving data between GPUs, not just between memory and compute on the same chip." | Connects the central lesson insight to the established mental model from scaling-and-efficiency. Reframes parallelism overhead as a familiar bottleneck at a larger scale. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new (parallelism strategies as a family; ZeRO optimizer sharding). Speculative decoding and continuous batching are upgrades from MENTIONED, not entirely new.
- **Previous lesson load:** BUILD (long-context-and-efficient-attention introduced RoPE, sparse attention, GQA -- moderate conceptual novelty)
- **Assessment:** CONSOLIDATE is appropriate. The core ideas (split work across devices, handle communication overhead) are engineering solutions to known problems, not new paradigms. The student has the conceptual vocabulary (compute-bound vs memory-bound, training memory breakdown, generate() loop) and just needs to see how these concepts operate at multi-GPU scale. No single concept here requires the kind of mental model shift that MoE demanded.

### Connections to Prior Concepts

| New Concept | Prior Concept | Connection |
|-------------|--------------|------------|
| Data parallelism (gradient all-reduce) | Gradient descent, backpropagation, training loop | Same gradients, same update rule. The only difference: gradients are averaged across GPUs before the optimizer step. |
| Tensor parallelism (split linear layers) | nn.Linear, matrix multiplication, transformer block | The student built these layers. Splitting columns of a weight matrix across GPUs is splitting the same nn.Linear they implemented. |
| Pipeline parallelism (split blocks across GPUs) | Block class from building-nanogpt (layers 0-11) | "Your model has 12 blocks. What if GPU 0 ran blocks 0-5 and GPU 1 ran blocks 6-11?" Directly references their implementation. |
| Communication overhead | Compute-bound vs memory-bound (delivery truck analogy) | "You already know that moving data is slower than computing on it. Parallelism puts this on a bigger stage -- now data moves between GPUs." |
| ZeRO optimizer sharding | Training memory breakdown (optimizer states = 2/3 of total) | "You computed this: 56 GB of 84 GB is optimizer states. What if each GPU stored only its share?" |
| Speculative decoding | generate() method from building-nanogpt, KV caching | "Remember your generate() loop? Each iteration is one forward pass for one token. Speculative decoding modifies that loop." |
| Continuous batching | Batched training from Series 2, DataLoader | "You used fixed batches for training. Inference serving has the same concept, but requests arrive continuously and finish at different times." |
| "Unconstrained optimization finds degenerate solutions" pattern | RLHF reward hacking (4.4.3), MoE load balancing (5.3.1) | Pipeline bubble is not an optimization failure, but the communication constraint is analogous: "naively distributing work without considering communication is like optimizing reward without considering KL penalty -- the system finds a degenerate solution." This is a light callback, not a core connection. |

### Prior Analogies -- Extension and Risk

- **"Delivery truck" analogy** (compute-bound vs memory-bound): extends naturally. Inter-GPU communication is an even slower delivery truck than GPU memory bandwidth. NVLink is fast but still orders of magnitude slower than on-chip compute.
- **"Assembly line" analogy** for pipeline parallelism: new, but maps to common experience. Risk: assembly lines suggest uniform processing time per station. In practice, transformer blocks take roughly equal time, so this analogy holds reasonably well. Caveat explicitly stated: if stages are unbalanced, some GPUs wait (pipeline bubble).
- **No misleading analogies identified.** The training memory breakdown analogy ("optimizer states dominate") extends directly to ZeRO without distortion.

### Scope Boundaries

**This lesson IS about:**
- The three parallelism strategies (data, tensor, pipeline) at DEVELOPED depth -- the student can explain when to use each and what each trades off
- Communication overhead as the central constraint at DEVELOPED depth
- ZeRO optimizer state sharding at INTRODUCED depth -- the student can explain the concept and motivation, but not the implementation or all three ZeRO stages
- Speculative decoding mechanism at DEVELOPED depth -- the student can trace the draft-verify loop and explain why it works
- Continuous batching at DEVELOPED depth -- the student can explain slot management and why it improves over static batching
- How these techniques combine in frontier model deployment (LLaMA, Mixtral)

**This lesson is NOT about:**
- Implementing any parallelism strategy in code (no PyTorch distributed, no FSDP, no DeepSpeed)
- NCCL, MPI, or communication primitives
- Specific hardware interconnect details beyond the basic insight (NVLink bandwidth as a number for context, not a deep dive)
- Ring attention or sequence parallelism (could be MENTIONED as further techniques but not developed)
- Expert parallelism for MoE models in detail (MENTIONED as a specific form of tensor parallelism)
- Model architecture search or optimal parallelism configuration
- Quantized inference serving (already INTRODUCED in lora-and-quantization)
- vLLM, TGI, or specific serving frameworks (MENTIONED at most)
- Gradient accumulation in depth (MENTIONED as related to microbatching)

**Target depth:**
- Data parallelism: DEVELOPED
- Tensor parallelism: DEVELOPED
- Pipeline parallelism: DEVELOPED
- Communication overhead: DEVELOPED
- ZeRO optimizer sharding: INTRODUCED
- Speculative decoding: DEVELOPED (upgrade from MENTIONED)
- Continuous batching: DEVELOPED (upgrade from MENTIONED)

### Lesson Outline

#### 1. Context + Constraints
What this lesson covers: the engineering that gets large models trained and served. What it does NOT cover: implementing these in code, specific libraries (DeepSpeed, FSDP, vLLM), hardware details beyond the essential insight. This is about understanding the design space, not configuring a cluster.

#### 2. Recap (Brief -- prerequisites are solid)
Three facts reconnected:
1. Training memory breakdown from lora-and-quantization: "You computed this -- 7B model needs ~84 GB. Optimizer states are 2/3."
2. generate() method from building-nanogpt: "Each token is one forward pass. Sequential by nature."
3. Compute-bound vs memory-bound from scaling-and-efficiency: "Moving data is slower than computing on it."

Connect: "What happens when the model has 70 billion parameters? 70B x 12 bytes = 840 GB. Your A100 has 80 GB. You cannot even start training."

#### 3. Hook (Scale Wall)
Concrete arithmetic: 70B model, training memory = 840 GB. Best single GPU = 80 GB. The model does not fit. Not "it's slow" -- it is physically impossible to begin. Second punch: even if you somehow train it, serving it means generating one token at a time in a sequential loop. A 70B model for 1000 concurrent users each generating 100 tokens = 100,000 sequential forward passes. The engineering problem is not "how to make it faster" but "how to make it possible at all."

GradientCard (orange): "Two walls. Training: the model does not fit on one GPU. Inference: the model generates one token at a time. Different problems, different solutions."

#### 4. Explain Part 1 -- Training Parallelism

**4a. Data Parallelism (the simple case)**
Start with what works: GPT-2 at 124M params fits easily on one GPU. Data parallelism: replicate the full model on each GPU, split the training data. Each GPU computes gradients on its batch. All-reduce averages gradients across GPUs. Each GPU applies the same averaged gradient update. Result: N GPUs = N times throughput.

Pseudocode: all-reduce operation.

Positive example: 4 GPUs training GPT-2. Each sees 1/4 of the data per step. Gradients are averaged. All weights stay synchronized.

GradientCard: "Data parallelism works when the model fits on one GPU. What when it does not?"

**4b. Communication is the Constraint (pivot)**
Before presenting solutions to "model too large," establish the central constraint. Any time GPUs share computation, they must communicate. Concrete numbers: A100 computes at ~312 TFLOPS (bf16). NVLink bandwidth is ~600 GB/s. Transferring 1 GB takes ~1.7 ms. In that time, the GPU could have done ~530 billion floating-point operations.

"Of course" callback: "You already knew this. Moving data is slower than computing on it -- the delivery truck analogy from scaling-and-efficiency. Parallelism puts this on a bigger stage: now the delivery truck drives between buildings, not between floors."

GradientCard (violet): "Communication overhead is the central constraint. Every parallelism strategy is a tradeoff between computation distribution and communication cost."

**4c. Tensor Parallelism (split within layers)**
Problem: a single linear layer in a 70B model might have a weight matrix of shape (8192, 32768). That is ~1 GB for one layer in bf16. Hundreds of these.

Mechanism: split the weight matrix column-wise across GPUs. GPU 0 stores columns 0-16383, GPU 1 stores columns 16384-32767. Each GPU computes its half of the output. GPUs exchange partial results (all-reduce) to reconstruct the full output.

Key insight: both GPUs compute simultaneously on the same input. Communication happens within each layer (after each split matmul). Fine-grained communication, high frequency.

Where it is used: within a single transformer block's attention and FFN layers. Connect to student's nn.Linear -- "the same layer you built, split across two chips."

MENTIONED: expert parallelism for MoE models -- placing different experts on different GPUs is a specialized form of tensor parallelism where the router determines which GPU does the work per token.

**4d. Pipeline Parallelism (split across layers)**
Problem: even with tensor parallelism, extremely large models need another dimension of splitting.

Mechanism: assign different layers to different GPUs. GPU 0 runs blocks 0-11, GPU 1 runs blocks 12-23, GPU 2 runs blocks 24-35, GPU 3 runs blocks 36-47. A microbatch flows through the pipeline: GPU 0 -> GPU 1 -> GPU 2 -> GPU 3.

Assembly line analogy: each station processes one step. Inefficient for a single item, but fast when the line is full.

**Negative example (pipeline bubble):** with 4 stages and 1 microbatch, only one GPU is active at a time. 75% idle. PipelineBubbleDiagram showing the timeline.

Solution: microbatching. Split the batch into microbatches. While GPU 1 processes microbatch 1, GPU 0 starts microbatch 2. The pipeline fills. With enough microbatches, utilization approaches 100%. MENTIONED: gradient accumulation is related -- accumulate gradients over microbatches before the optimizer step.

**4e. Side-by-side comparison**
ParallelismComparisonDiagram: three-panel visual.

ComparisonRow or table:

| Property | Data Parallelism | Tensor Parallelism | Pipeline Parallelism |
|----------|-----------------|-------------------|---------------------|
| What is replicated | Full model | Nothing (model is split) | Nothing (model is split) |
| What is split | Data (batches) | Layers (within) | Layers (across) |
| Communication pattern | After each backward pass (all-reduce gradients) | Within each layer (all-reduce partial outputs) | Between stages only (activations forwarded) |
| Communication frequency | Low (once per step) | High (every layer) | Medium (between stages) |
| When to use | Model fits on one GPU | Model is too wide for one GPU | Model is too deep for one GPU |

Address Misconception 2 here: tensor vs pipeline parallelism are not the same. Different granularity, different communication patterns.

#### 5. Check 1 (Predict-and-Verify)
Scenario: You have a 30B dense model and 4 A100 GPUs (80 GB each). Training memory for 30B is ~360 GB. Can you use data parallelism? (No -- 360 GB > 80 GB.) Can you use pipeline parallelism with 4 stages? (Each stage is ~7.5B params = ~90 GB -- still does not fit on one GPU with optimizer states.) What combination works? (Pipeline parallelism to split layers into stages small enough for one GPU, possibly combined with tensor parallelism within stages.)

Details/summary reveal.

#### 6. Explain Part 2 -- ZeRO Optimizer Sharding

Transition: "Data parallelism requires each GPU to hold the full model. But we showed that 70B doesn't fit. Is there a middle ground?"

ZeRO (Zero Redundancy Optimizer): each GPU stores the full model weights (for forward/backward), but only a fraction of the optimizer states. Recall: optimizer states are 2/3 of training memory. If 8 GPUs each store 1/8 of the optimizer states, that alone cuts total per-GPU memory from 840 GB to ~350 GB (still need tensor/pipeline parallelism, but the gap is much smaller).

Concrete arithmetic for the student:
- 70B model, 8 GPUs
- Without ZeRO: each GPU needs ~840 GB (impossible)
- ZeRO Stage 1 (shard optimizer states): each GPU needs weights (140 GB) + gradients (140 GB) + 1/8 optimizer states (70 GB) = ~350 GB
- Still does not fit on 80 GB, but combined with tensor parallelism the per-GPU requirements become manageable

INTRODUCED depth: the concept and motivation, not the implementation. MENTIONED: ZeRO has three stages (shard optimizer states, shard gradients, shard parameters). DeepSpeed and FSDP as frameworks that implement this.

#### 7. Explain Part 3 -- Inference: The Sequential Bottleneck

Transition: "Training solved. Now the model is deployed. New problem."

Restate the inference bottleneck: autoregressive generation is sequential. Each token requires a full forward pass. A 70B model can do ~30-50 tokens/second. 1000 concurrent users each wanting 100 tokens = impossible with naive generation.

Two independent solutions for two independent bottlenecks:
1. Speculative decoding: make each user's generation faster (latency)
2. Continuous batching: serve more users simultaneously (throughput)

**7a. Speculative Decoding (Develop from MENTIONED)**
Callback to generate() method: "Your loop generates one token, runs a forward pass, generates one token, runs a forward pass."

Key insight: the large model's forward pass costs roughly the same whether processing 1 token or 5 tokens (compute-bound matmul). The sequential bottleneck is not the compute per token but the number of serial forward passes.

Mechanism:
1. Draft model (small, fast -- e.g., 7B) generates K draft tokens autoregressively (fast, ~5x faster per token)
2. Large model (70B) runs ONE forward pass on all K draft tokens simultaneously (verifying all at once)
3. Compare: for each draft position, does the large model agree? Accept matching tokens, reject from first disagreement.
4. Result: 1 large forward pass produces up to K tokens instead of 1.

SpeculativeDecodingDiagram: visual showing draft (5 tokens), verify (1 forward pass), accept 3, reject 2.

Worked example: "The capital of France is Paris, and" -- draft model produces ["Paris", ",", "and", "the", "city"]. Large model verifies: agrees on ["Paris", ",", "and"], disagrees on "the" (prefers "it"). Accept first 3. One large forward pass instead of 3.

Misconception 3 addressed: "The speed does not come from the small model being fast. It comes from the large model verifying in parallel."

Pseudocode: modified generate() loop with draft-and-verify.

**7b. Continuous Batching (Develop from MENTIONED)**
Problem: static batching wastes compute. 8 requests start together. One finishes at 20 tokens, another at 200. The batch runs until the longest finishes. Completed requests sit idle, consuming GPU memory and producing nothing.

Negative example (static batching waste): concrete timeline with 8 requests, varying completion lengths. Compute wasted slots.

Mechanism: when a request completes, its slot is immediately filled with the next request from the queue. The batch size stays constant; the contents change dynamically.

ContinuousBatchingDiagram: timeline showing static vs continuous, with request arrivals and completions.

Restaurant waitlist analogy: "As a table opens, the next party is seated immediately."

Pseudocode: slot management with a queue.

#### 8. Check 2 (Transfer Question)
Scenario: A company deploys a Mixtral 8x7B model (47B total params, 13B active per token) for a customer service chatbot. They have 8 A100 GPUs. Questions:
1. Why does MoE make parallelism especially important? (All 47B params must be loaded even though only 13B activate per token -- memory requirement exceeds a single GPU.)
2. Would speculative decoding work well here? (Yes -- the 13B active compute per token is still expensive, and a small dense draft model could produce candidates quickly. But: the draft model must approximate the MoE model's output distribution, which is harder if MoE has specialized expert knowledge.)
3. What role does continuous batching play? (Customer service = many concurrent short requests. High throughput more important than single-request latency. Continuous batching keeps GPU utilization high as short requests complete.)

Details/summary reveal.

#### 9. Elaborate -- How Frontier Models Combine Everything

Brief synthesis: modern frontier models use combinations of all these techniques. LLaMA 2 70B training: tensor parallelism across 8 GPUs per node, pipeline parallelism across nodes, data parallelism across node groups, ZeRO for optimizer sharding. Mixtral serving: expert parallelism (different experts on different GPUs) + continuous batching + speculative decoding.

The key insight: each technique addresses a specific bottleneck. The art of distributed systems engineering is identifying which bottleneck dominates and applying the right combination. This echoes the module's connecting thread -- "three barriers, three targeted solutions" -- extended from architecture to engineering.

MENTIONED: vLLM, TGI, and other serving frameworks. Ring attention / sequence parallelism for long-context models. Gradient accumulation for effective larger batch sizes.

#### 10. Practice -- Notebook Exercises

**Notebook:** `notebooks/5-3-3-training-and-serving-at-scale.ipynb` (4 exercises)

Exercises are designed around calculation and analysis rather than distributed systems implementation (which requires multi-GPU hardware the student does not have in Colab).

- **Exercise 1 (Guided): Training memory calculator.** Build a function that computes training memory for a given model configuration (num_params, precision, optimizer). Apply to GPT-2 (124M), LLaMA 7B, LLaMA 70B. Compare single-GPU requirement to A100 capacity. Calculate ZeRO Stage 1 savings for each model across 1, 2, 4, 8 GPUs. Present as table. Predict-before-run: "Will ZeRO Stage 1 alone make 70B fit on 8 A100s?" Insight: optimizer sharding alone is not enough for very large models, but dramatically extends what data parallelism can handle.

- **Exercise 2 (Supported): Speculative decoding simulation.** Implement a simplified speculative decoding loop using two language models (small: GPT-2 124M, large: GPT-2 355M or a reasoning model via API). Draft K tokens with the small model, verify with the large model. Measure acceptance rate at K=1, 3, 5, 8. Plot acceptance rate vs K. Measure wall-clock time per accepted token. Insight: acceptance rate decreases with K, and there is an optimal draft length that balances the overhead of verification against the benefit of parallel acceptance.

- **Exercise 3 (Supported): Continuous batching simulation.** Simulate an inference server: queue of 50 requests with varying target lengths (sampled from realistic distribution: mean 80, std 60, min 10, max 300). Implement static batching (batch_size=8, wait for all to finish) and continuous batching (batch_size=8, fill completed slots). Measure: total time to serve all 50 requests, average latency per request, GPU utilization (fraction of batch slots doing useful work). Plot utilization over time for both strategies. Insight: continuous batching achieves near-100% utilization while static batching degrades as short requests complete.

- **Exercise 4 (Independent): Parallelism strategy advisor.** Given a model size, number of GPUs, and GPU memory, determine which parallelism strategy (or combination) is needed. Build a function that takes (num_params, num_gpus, gpu_memory_gb, precision) and outputs: can use data parallelism alone? If not, minimum tensor parallelism degree needed? Pipeline parallelism stages? ZeRO stage recommendation? Test on 5 configurations ranging from GPT-2 (124M, 1 GPU) to hypothetical 175B model (64 GPUs). No skeleton provided. Insight: the choice of parallelism strategy is determined by whether the bottleneck is compute, memory, or communication -- connecting back to the central lesson theme.

Exercises are independent (can be done in any order after Exercise 1, which establishes the memory calculator used by Exercise 4).

#### 11. Summarize
Key takeaways:
1. Three parallelism strategies: data (split data), tensor (split layers), pipeline (split blocks). Each addresses a different bottleneck.
2. Communication overhead is the constraint that shapes every parallelism decision.
3. ZeRO reduces redundancy by sharding optimizer states across GPUs.
4. Speculative decoding turns sequential generation into parallel verification.
5. Continuous batching eliminates wasted compute from static batching.

Mental model echo: "The math is elegant; the engineering makes it work. Every technique in this lesson is an answer to the same question: how do you distribute work across devices when communication is expensive? Data parallelism distributes data. Tensor parallelism distributes computation. Pipeline parallelism distributes layers. ZeRO distributes memory. Speculative decoding distributes generation steps. Continuous batching distributes serving capacity. The bottleneck determines the solution."

Module completion note: this concludes Module 5.3 (Scaling Architecture). The module traced the full arc from the dense transformer's bottlenecks through architectural innovations (MoE, efficient attention) to the engineering that makes frontier models possible.

#### 12. Next Step
Forward reference to the next module or series, depending on course state.

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
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (5 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 2 new concepts (parallelism family, ZeRO) + 2 upgrades from MENTIONED (speculative decoding, continuous batching) = within limit
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: MAJOR REVISION

### Findings

#### [CRITICAL] — Misconception 4 ("Communication between GPUs is fast") not addressed in the lesson

**Location:** Entire lesson; planned for the "Communication Is the Constraint" pivot section
**Issue:** The planning document identifies five misconceptions, and Misconception 4 ("Communication between GPUs is fast because they are in the same machine") is planned to be the "pivot" that motivates everything after data parallelism. The built lesson's "Communication Is the Constraint" section (lines 1316-1374) presents the numbers (312 TFLOPS compute, 600 GB/s NVLink, 530 billion ops wasted during 1 GB transfer) and connects to the delivery truck analogy, but it never explicitly names and disproves Misconception 4. The planning document specifies: "Transferring even a modest tensor (1 GB) takes ~1.7 ms — during which the GPU could have done ~530 billion floating-point operations. At scale, training a 175B model with naive tensor parallelism across 8 GPUs can spend 40-60% of time waiting for communication." The lesson includes the 1 GB / 1.7 ms / 530 billion ops numbers but omits the "40-60% time waiting" concrete consequence and never frames it as explicitly disproving the misconception that inter-GPU communication is fast. Without the explicit negative example (the student expects GPUs in the same machine to communicate quickly), the student may absorb the numbers without updating their prior belief. This is the planned "pivot" of the entire lesson and needs the misconception framing to land.
**Student impact:** The student sees the raw numbers but may not connect them to their prior belief that in-machine communication is fast. The numbers alone don't force the "oh, communication is actually the bottleneck" realization. The lesson needs the explicit misconception-then-disproof structure to create the conceptual shift.
**Suggested fix:** Add a rose GradientCard in the "Communication Is the Constraint" section that explicitly names the misconception ("Communication between GPUs is fast because they are in the same machine"), provides the concrete consequence (40-60% time waiting for communication in naive tensor parallelism at 175B scale), and connects it to the student's software engineering intuition (inter-process communication feels fast in their experience). Then the "of course" moment already present can land more powerfully.

---

#### [IMPROVEMENT] — Misconception 1 ("Parallelism is just data parallelism") not explicitly addressed with a negative example

**Location:** Data Parallelism section (lines 1252-1311) and the transition to tensor/pipeline
**Issue:** The planning document identifies Misconception 1: "Parallelism is just data parallelism — split the data, replicate the model." The plan calls for explicitly disproving this right after data parallelism is presented, using the negative example that a 70B model at 840 GB cannot use data parallelism because each GPU must hold the full model. The built lesson mentions this in the GradientCard ("For a 70B model needing 840 GB? Data parallelism alone cannot help") but does not frame it as a misconception the student would naturally hold. There is no rose GradientCard, no explicit "you might think parallelism means splitting the data — here's why that is incomplete." The misconception treatment is implicit rather than explicit.
**Student impact:** The student may move past data parallelism without fully internalizing that it requires the full model on every GPU. The "aha" moment of "wait, I need more than data parallelism" is weaker without the explicit misconception framing.
**Suggested fix:** Add a rose GradientCard after the data parallelism section that explicitly names the misconception ("The most common mental model of parallelism is data parallelism — split the data, replicate the model. But data parallelism requires each GPU to hold the full model. For a 70B model at 840 GB, this is impossible even before splitting data."). The emerald GradientCard already present covers the "when it works" case; the rose card covers "when it fails and why."

---

#### [IMPROVEMENT] — Speculative decoding claim about forward pass cost being independent of token count is stated but not grounded

**Location:** Speculative Decoding section, paragraph 2 (lines 1773-1780)
**Issue:** The lesson states: "the large model's forward pass costs roughly the same whether processing 1 token or 5 tokens. Compute-bound matrix multiplications scale with the dimensions of the weight matrices, not the number of input tokens (up to a point)." This is the key insight that makes speculative decoding work, but the "(up to a point)" hedging combined with no concrete grounding leaves this as an assertion the student must take on faith. The planning document's worked example (one forward pass costs the same for 1 or 5 tokens because matmul is compute-bound) is partially present but the lesson doesn't connect it to the student's existing knowledge of compute-bound vs memory-bound from scaling-and-efficiency. Specifically: during inference with KV caching, the forward pass for 5 new tokens involves a matmul of shape (batch, 5, d_model) @ (d_model, vocab_size), which is still compute-bound because d_model and vocab_size dominate. This is the same arithmetic intensity reasoning the student learned but it's not invoked.
**Student impact:** The student is told the insight but doesn't derive it from first principles they already have. They may accept it passively rather than truly understanding why. This weakens the "of course" moment.
**Suggested fix:** Add 1-2 sentences connecting to the compute-bound/memory-bound framework: "Remember from Scaling & Efficiency: matrix multiplications are compute-bound — the time depends on the matrix dimensions (d_model, vocab_size), not the batch dimension. Processing 5 tokens instead of 1 increases the batch dimension from 1 to 5, but the weight matrices are the same. The compute is dominated by the weights, not the inputs." This grounds the claim in the student's existing mental model.

---

#### [IMPROVEMENT] — Continuous batching pseudocode presents the mechanism before the conceptual explanation

**Location:** Continuous Batching section (lines 1877-1966)
**Issue:** The section follows this order: (1) static batching problem, (2) negative example (static waste GradientCard), (3) diagram, (4) restaurant analogy, (5) one-paragraph mechanism explanation, (6) misconception card, (7) pseudocode. The ordering principle "concrete before abstract" is followed for the static batching problem, but the mechanism paragraph (line 1916-1922) comes after the analogy and diagram. More importantly, the pseudocode arrives after the misconception card, which is the right placement, but the mechanism paragraph itself is quite brief (5 lines) compared to the detailed pseudocode. The student transitions from "the restaurant seats the next party" analogy to a 20-line class with slots and queues. The cognitive jump is moderate but could be smoothed.
**Student impact:** The transition from analogy to code is abrupt. The student gets the restaurant analogy, then must parse a class-based implementation. A brief verbal walkthrough of the mechanism between the analogy and the code would help.
**Suggested fix:** Expand the mechanism paragraph (lines 1916-1922) with one more sentence that explicitly maps the analogy to the implementation: "In code, each batch slot tracks one request. At each step, all active slots generate one token. When a slot's request completes, the server pulls the next request from the queue and assigns it to that slot. The batch size stays constant; the contents change dynamically." Then the pseudocode confirms what was already explained verbally.

---

#### [IMPROVEMENT] — Notebook Exercise 2 uses simulation not actual models, but planning doc specifies GPT-2 124M and GPT-2 355M

**Location:** Notebook Exercise 2 (cells 8-15)
**Issue:** The planning document specifies: "Implement a simplified speculative decoding loop using two language models (small: GPT-2 124M, large: GPT-2 355M or a reasoning model via API). Draft K tokens with the small model, verify with the large model." The built notebook uses a simulation with `match_prob` parameter instead of actual models. This is a reasonable design decision (avoids Colab GPU requirements, focuses on the mechanism), but it's an undocumented deviation from the plan. The exercise description in the TSX lesson (lines 2116-2123) also says "Implement a simplified speculative decoding loop with GPT-2 124M (draft) and GPT-2 355M (verifier)" — this matches the plan, not the notebook. The student reads the lesson, sees "GPT-2 124M (draft) and GPT-2 355M (verifier)," opens the notebook, and finds a probability-based simulation instead.
**Student impact:** Mismatch between what the lesson promises and what the notebook delivers. The student expects to work with actual language models and instead finds a mathematical simulation. This could be confusing ("where are the models?") or feel like a downgrade.
**Suggested fix:** Either (a) update the lesson's Exercise 2 description to accurately describe the simulation approach: "Simulate a speculative decoding loop: model the draft-and-verify mechanism with configurable match probability, measure how acceptance rate changes with draft length K, and find the optimal draft length balancing draft overhead against expected accepted tokens," or (b) add an optional bonus cell to the notebook that uses actual GPT-2 models (if a GPU is available) alongside the simulation.

---

#### [POLISH] — Spaced em dashes in SVG text elements and subtitle strings

**Location:** Lines 580, 635, 956, 1012, 1118, 1256, 1383, 1435, 2095, 2323
**Issue:** The writing style rule requires em dashes with no spaces: `word—word` not `word — word`. Multiple SVG text elements and subtitle strings use spaced em dashes: "1 Microbatch — 75% idle", "The simple case — when the model fits", "Split within layers — both GPUs compute simultaneously", etc. Line 1118 in the ConstraintBlock also uses spaced em dash.
**Student impact:** None functionally, but inconsistent with established style.
**Suggested fix:** Replace ` — ` with `—` in all SVG text elements and string props. Use `&mdash;` in JSX string literals where appropriate.

---

#### [POLISH] — Notebook uses `--` (double hyphen) instead of em dashes in markdown cells

**Location:** Multiple notebook markdown cells (cells 0, 4, 8, 16, 25, 28)
**Issue:** Notebook markdown uses `--` as a separator/dash (e.g., "No multi-GPU hardware needed -- everything runs as calculation and simulation on a single CPU"). While this is common in plain text and code comments, the lesson style convention is em dashes without spaces. Notebook markdown is student-facing prose.
**Student impact:** Minimal. The double hyphen is readable but inconsistent with the lesson component's use of proper em dashes.
**Suggested fix:** Replace ` -- ` with `—` in notebook markdown cells. Leave code comments and print statements as-is (they're code, not prose).

---

#### [POLISH] — Aside TipBlock uses internal lesson-design terminology "CONSOLIDATE Lesson"

**Location:** Lines 1132-1138 (TipBlock in the Constraints section aside)
**Issue:** The aside title is "CONSOLIDATE Lesson" — this is internal planning terminology (CONSOLIDATE vs BUILD vs STRETCH) that the student has not been exposed to and doesn't need. The body text is fine ("After the BUILD of Long Context & Efficient Attention" also uses internal terminology). Previous lesson reviews flagged similar issues (e.g., chain-of-thought review flagged "STRETCH Lesson").
**Student impact:** Minor confusion. The student might wonder what "CONSOLIDATE" and "BUILD" mean as pedagogical categories. These are implementation details, not student-facing concepts.
**Suggested fix:** Change the TipBlock title to something like "Building on What You Know" and remove the "BUILD" reference from the body text: "After the conceptual work of Long Context & Efficient Attention, this lesson connects concepts you already know..."

---

### Review Notes

**What works well:**
- The narrative arc is strong. The "two walls" hook is compelling and the lesson maintains momentum through both the training and inference sections. The student is never asked to care about a technique without first understanding the problem it solves.
- The visual diagrams (ParallelismComparisonDiagram, PipelineBubbleDiagram, SpeculativeDecodingDiagram, ContinuousBatchingDiagram) are well-designed with clear color coding and informative captions. The pipeline bubble diagram with microbatch coloring is particularly effective.
- The connections to prior concepts are consistently made: training memory breakdown from LoRA & Quantization, generate() loop from nanoGPT, compute-bound vs memory-bound from Scaling & Efficiency, MoE memory from this module's lesson 1. The lesson genuinely feels like a capstone that brings prior knowledge together.
- The notebook is well-structured with good scaffolding progression (Guided -> Supported -> Supported -> Independent). The reference implementation pattern (provide working code after the student attempts their own) is effective.
- The ComparisonRow for ZeRO (with/without) and the three-strategy comparison table provide clear structured comparisons.

**Patterns to watch:**
- The lesson addresses 3 of 5 planned misconceptions explicitly (Misconceptions 2, 3, 5 with rose GradientCards). Misconception 1 is implicitly addressed but not named. Misconception 4 is partially addressed with numbers but not framed as a misconception. The lesson should aim to explicitly address all planned misconceptions, even if briefly.
- The notebook exercise descriptions in the lesson TSX must match the actual notebook content. The speculative decoding mismatch (lesson says GPT-2 models, notebook uses simulation) is the specific instance, but this is worth checking generally.

**Modality count for core concept (parallelism strategies as a family):**
1. Visual: ParallelismComparisonDiagram (SVG), PipelineBubbleDiagram (SVG)
2. Verbal/Analogy: "assembly line" for pipeline, "delivery truck between buildings" for communication
3. Symbolic: pseudocode for data parallelism all-reduce
4. Concrete example: 70B model memory arithmetic, GPT-2 124M on 4 GPUs
5. Intuitive: "of course" callback to compute-bound vs memory-bound
Result: 5 modalities present. Exceeds minimum of 3. Good.

**Modality count for speculative decoding (upgrade from MENTIONED):**
1. Visual: SpeculativeDecodingDiagram (SVG)
2. Verbal/Analogy: "rough draft and editor"
3. Symbolic: pseudocode for draft-verify loop
4. Concrete example: "The capital of France is" worked example with token-by-token acceptance
5. Intuitive: connection to generate() loop the student built
Result: 5 modalities. Good.

**Modality count for continuous batching (upgrade from MENTIONED):**
1. Visual: ContinuousBatchingDiagram (SVG)
2. Verbal/Analogy: "restaurant waitlist"
3. Symbolic: pseudocode for InferenceServer class
4. Concrete example: 8 requests with varying lengths, waste calculation
Result: 4 modalities. Meets minimum.

**Cognitive load assessment:** 2 genuinely new concepts (parallelism family, ZeRO) + 2 upgrades from MENTIONED (speculative decoding, continuous batching). Within the 2-3 new concept limit. The lesson is long but each section is self-contained and the two-wall framing (training vs inference) creates natural breaks.

---

## Review — 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

### Iteration 1 Resolution Check

All 8 findings from iteration 1 were properly resolved:

1. **CRITICAL (Misconception 4 not addressed):** RESOLVED. Rose GradientCard at lines 1370-1387 now explicitly names the misconception ("Communication between GPUs is fast because they are in the same machine"), includes the "40-60% of time waiting for communication" concrete consequence, and connects to the student's software engineering intuition. The pivot lands.
2. **IMPROVEMENT (Misconception 1 not explicit):** RESOLVED. Rose GradientCard at lines 1302-1314 explicitly names the misconception and explains why data parallelism requires the full model on every GPU.
3. **IMPROVEMENT (Speculative decoding forward pass cost not grounded):** RESOLVED. Lines 1811-1820 now explicitly connect to the compute-bound/memory-bound framework from Scaling & Efficiency, naming d_model and d_vocab as the dimensions that dominate compute.
4. **IMPROVEMENT (Continuous batching mechanism before pseudocode):** RESOLVED. Lines 1957-1966 have an expanded mechanism paragraph with explicit slot management description before the pseudocode.
5. **IMPROVEMENT (Notebook Exercise 2 mismatch):** RESOLVED. Exercise 2 description in the TSX (lines 2162-2169) now accurately describes the simulation approach ("Simulate the draft-and-verify mechanism with configurable match probability").
6. **POLISH (Spaced em dashes):** RESOLVED. All SVG text elements and subtitle strings now use unspaced em dashes.
7. **POLISH (Notebook double hyphens):** RESOLVED. Markdown cells use proper em dashes. Remaining `--` are in Python code (print statements, code comments), which is appropriate for code.
8. **POLISH (CONSOLIDATE Lesson terminology):** RESOLVED. TipBlock title changed to "Building on What You Know" with no internal terminology.

### Findings

#### [POLISH] — Exercises aside still references "small GPT-2 models" for Exercise 2

**Location:** Line 2231 (TipBlock aside in the Practice section)
**Issue:** The aside says "The speculative decoding exercise uses small GPT-2 models." The exercise description in the main content (lines 2162-2169) was correctly updated to describe the simulation approach, and the notebook uses a probability-based simulation. But the aside was not updated to match.
**Student impact:** Minor. The aside contradicts the exercise description the student just read. If the student reads both, they may wonder which is correct. Not confusing enough to be an improvement-level finding since the main exercise description is accurate.
**Suggested fix:** Change line 2231 from "The speculative decoding exercise uses small GPT-2 models" to "The speculative decoding exercise is a probability-based simulation" or simply "The speculative decoding exercise simulates the draft-verify mechanism."

---

#### [POLISH] — Notebook code cells use `--` instead of em dashes in print statements

**Location:** Multiple code cells (cells 5, 6, 7, 9, 13, 14, 23, 24)
**Issue:** Python print statements use ` -- ` as a dash (e.g., `"it's slow" -- it is physically impossible to begin`). These are student-facing output strings, not code comments. However, since they appear inside Python string literals in code cells, this is a grey area between code convention and prose convention.
**Student impact:** Negligible. The double hyphen is perfectly readable in terminal output. The lesson's HTML content uses proper em dashes. The difference is between rendered HTML and terminal output, which naturally have different conventions.
**Suggested fix:** No action needed. This is within acceptable convention for Python print output. Flagged for completeness only.

---

### Review Notes

**All iteration 1 findings properly resolved.** The critical finding (Misconception 4) is now thoroughly addressed with a rose GradientCard that explicitly names the misconception, provides the concrete 40-60% communication overhead statistic, and connects to the student's software engineering intuition. This was the pivotal section of the lesson and it now lands correctly.

**What works well (unchanged from iteration 1, confirmed on re-read):**
- The narrative arc is strong. The "two walls" hook is compelling and the student is never asked to care about a technique without understanding the problem it solves.
- All five misconceptions are now explicitly addressed with rose GradientCards. The misconception-then-disproof structure is consistent throughout.
- The speculative decoding section now grounds the key insight (forward pass cost independent of token count) in the student's existing compute-bound/memory-bound framework, making it a derivation rather than an assertion.
- The continuous batching mechanism paragraph bridges the analogy-to-pseudocode gap smoothly.
- The notebook Exercise 2 description now matches the actual simulation content.
- The four inline SVG diagrams are well-designed with clear color coding and informative captions.
- The notebook is well-structured with proper scaffolding progression and good predict-before-run prompts.

**Modality counts confirmed:**
- Parallelism strategies (core concept): 5 modalities (visual, verbal/analogy, symbolic, concrete example, intuitive). Exceeds minimum.
- Speculative decoding: 5 modalities. Exceeds minimum.
- Continuous batching: 4 modalities. Meets minimum.

**Pedagogical principles confirmed:**
- Motivation Rule: Problem before solution throughout. Every section starts with the bottleneck.
- Modality Rule: 4-5 modalities per core concept. Met.
- Example Rules: 5+ positive, 2 negative examples. First examples are simplest (GPT-2 124M for data parallelism). Met.
- Misconception Rule: 5 misconceptions, all with rose GradientCards. Met.
- Ordering Rules: Concrete before abstract throughout. Met.
- Load Rule: 2 new + 2 upgrades. Within limit. Met.
- Connection Rule: Every concept connected to prior knowledge. Met.
- Reinforcement Rule: No fading concepts used without recap. Met.
- Interaction Design Rule: summary elements have cursor-pointer. No other interactive elements. Met.
- Writing Style Rule: Em dashes unspaced in HTML content. Met.

The two polish findings are minor and do not require a re-review. This lesson is ready to ship.
