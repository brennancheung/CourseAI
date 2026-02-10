# Lesson Plan: GPU Training

**Module:** 2.3 (Practical Patterns)
**Position:** Lesson 2 of 3
**Slug:** `gpu-training`
**Load Type:** STRETCH

---

## Phase 1: Orient --- Student State

The student has completed all of Series 1 (Foundations), Modules 2.1 (PyTorch Core) and 2.2 (Real Data), and the first lesson of Module 2.3 (Saving and Loading). They can build, train, evaluate, debug, save, load, and checkpoint models end-to-end on CPU. They have conceptual exposure to GPUs from the tensors lesson (2.1.1) but have never actually trained a model on GPU.

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| GPU as parallel processor | INTRODUCED | tensors (2.1.1) | CPU (8-16 fast cores) vs GPU (thousands of simple cores). ComparisonRow visual. Knows the concept but has not used it in training. |
| Device management (.to(), .cpu()) | INTRODUCED | tensors (2.1.1) | Standard pattern: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Used on individual tensors, never on a model or in a training loop. |
| GPU transfer overhead | INTRODUCED | tensors (2.1.1) | Small tensors faster on CPU; timing comparison showing 10-element (CPU 12x faster) vs 10M-element (GPU 350x faster). "GPU wins at scale" mental model. |
| Device mismatch error | INTRODUCED | tensors (2.1.1) | "All tensors in an operation must be on the same device." Fix with `.to(X.device)`. Seen once, not practiced in a multi-component context. |
| map_location for cross-device loading | INTRODUCED | saving-and-loading (2.3.1) | Brief preview: `map_location='cpu'` or `'cuda:0'` or device variable. Portable pattern shown. |
| Complete PyTorch training loop | DEVELOPED | training-loop (2.1.4) | forward-loss-backward-update heartbeat. Practiced on regression and classification. |
| Checkpoint pattern (model + optimizer + epoch + loss) | DEVELOPED | saving-and-loading (2.3.1) | Bundle all state into one dictionary; torch.save/load. Integrated into training loop with periodic saves + best-model save. |
| state_dict as persistence format | DEVELOPED | saving-and-loading (2.3.1) | "Snapshot of all the knobs." Save and load state_dicts for both model and optimizer. |
| nn.Module subclass pattern | DEVELOPED | nn-module (2.1.3) | Layers in __init__, data flow in forward(). model.parameters() collects all learnable tensors. |
| model.train() / model.eval() | DEVELOPED | mnist-project (2.2.2) | Practiced: train() enables dropout + batch-stat BN; eval() disables. |
| PyTorch dtype system | INTRODUCED | tensors (2.1.1) | float32, float64, float16 -- table with use cases. `.float()`, `.half()` conversion methods. |
| PyTorch float32 default (vs float64) | DEVELOPED | tensors (2.1.1) | "Measuring a rough sketch with a micrometer" analogy. Mini-batch gradients are approximate, float64 is wasted precision. |
| DataLoader (batch_size, shuffle, iteration) | DEVELOPED | datasets-and-dataloaders (2.2.1) | Wraps Dataset; handles batching, shuffling, iteration. Yields tensors with batch dimension. |
| TensorBoard for training monitoring | INTRODUCED | debugging-and-visualization (2.2.3) | SummaryWriter logs scalars per epoch. "Flight recorder" analogy. Run comparison. |
| Systematic debugging checklist | DEVELOPED | debugging-and-visualization (2.2.3) | torchinfo before, gradient check first iteration, TensorBoard during, diagnose by symptom. |

### Mental Models Already Established

- **"GPU wins at scale, CPU wins for small operations"** -- Transfer overhead is the deciding factor. Established in tensors (2.1.1) with timing comparison.
- **"Shape, dtype, device -- check these first"** -- The debugging trinity from tensors (2.1.1). Device is already part of this mental model.
- **"Same heartbeat, new instruments"** -- The training loop pattern is universal. New tools (GPU placement) slot in without changing the rhythm.
- **"Rough sketch with a micrometer"** -- float64 is wasted precision for approximate gradients. Sets up the idea that float16 might be "just right" for certain operations.
- **"Parameters are knobs" / "state_dict is a snapshot of all the knobs"** -- Model state as a collection of learned values that can be moved, saved, loaded.
- **"Not a black box"** -- From training-loop (2.1.4). The student expects to understand what abstractions do, not just call them.

### What Was Explicitly NOT Covered

- Training a model on GPU -- explicitly deferred from tensors (2.1.1), training-loop (2.1.4), mnist-project (2.2.2), and saving-and-loading (2.3.1).
- Moving a model to GPU (`.to(device)` on a model) -- only shown for individual tensors.
- Moving DataLoader outputs to GPU -- never shown.
- Mixed precision training (float16 for speed) -- never discussed.
- `torch.amp` or `GradScaler` -- never mentioned.
- `num_workers` or `pin_memory` in DataLoader -- never discussed.
- CUDA-specific debugging -- explicitly noted as not covered in debugging-and-visualization (2.2.3).
- Memory management, CUDA streams, pinned memory -- explicitly deferred in tensors (2.1.1).

### Readiness Assessment

The student is ready but this will be a stretch. They have all the conceptual prerequisites: GPU as parallel processor, device management on individual tensors, the "GPU wins at scale" mental model, device mismatch errors, and the complete training loop. The stretch comes from coordination complexity -- they must apply `.to(device)` consistently across model, data batches, and targets within the training loop, which is a higher-order pattern than any single API call they have seen. The float32-sufficiency analogy sets up mixed precision naturally: if float64 is too precise, maybe float16 is precise enough for some operations.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to **write device-aware training code that runs identically on CPU and GPU, and to use mixed precision to accelerate training when it matters**.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Device management (.to(), .cpu()) | INTRODUCED | INTRODUCED | tensors (2.1.1) | OK | Need to recognize .to(device) pattern. This lesson DEVELOPS it from tensor-level to training-loop-level. |
| GPU as parallel processor | INTRODUCED | INTRODUCED | tensors (2.1.1) | OK | Need the conceptual frame of why GPU helps. Not re-teaching the concept, extending it. |
| GPU transfer overhead | INTRODUCED | INTRODUCED | tensors (2.1.1) | OK | Need the "GPU wins at scale" intuition to understand when GPU training helps and when it does not. |
| Device mismatch error | INTRODUCED | INTRODUCED | tensors (2.1.1) | OK | Need to recognize the error. This lesson makes it visceral through a multi-component example. |
| Complete PyTorch training loop | DEVELOPED | DEVELOPED | training-loop (2.1.4) | OK | The GPU training loop is the existing loop with device placement added. Must be fluent with the base pattern. |
| DataLoader (batching and iteration) | DEVELOPED | DEVELOPED | datasets-and-dataloaders (2.2.1) | OK | Data comes from DataLoader in batches. Each batch must be moved to GPU inside the training loop. |
| Checkpoint pattern | DEVELOPED | DEVELOPED | saving-and-loading (2.3.1) | OK | GPU checkpoints need map_location for portability. The student already has the checkpoint pattern; this lesson adds device-awareness to it. |
| map_location for cross-device loading | INTRODUCED | INTRODUCED | saving-and-loading (2.3.1) | OK | Already previewed. This lesson develops it into a full pattern (save on GPU, load on CPU or vice versa). |
| PyTorch float32 default | DEVELOPED | DEVELOPED | tensors (2.1.1) | OK | Foundational for mixed precision -- float32 is the baseline, float16 is the acceleration. |
| PyTorch dtype system (float16 exists) | INTRODUCED | INTRODUCED | tensors (2.1.1) | OK | Student knows float16 exists and its use case column said "mixed precision training." This lesson develops that. |

**No gaps. All prerequisites met.** Every concept this lesson builds on has been taught at a sufficient depth. The lesson DEVELOPS device management from INTRODUCED (individual tensors) to DEVELOPED (coordinated across model + data + training loop), and INTRODUCES mixed precision as a new concept.

### Gap Resolution

No gaps found. All prerequisites are at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **"GPU is always faster"** | The student has the "GPU wins at scale" model but may overgeneralize. Tutorials and marketing always emphasize GPU speed. The student may assume any model on GPU is faster. | Time a small 2-layer model on a tiny dataset (100 samples): CPU is faster because transfer overhead dominates. Then time the same model on MNIST (60,000 samples, larger batches): GPU is 3-5x faster. The crossover point makes the tradeoff concrete. | Hook section -- establish this misconception immediately and disprove it with timing. |
| **"Moving to GPU is just model.to(device)"** | The student has seen `.to(device)` on tensors and may think moving the model is all that is needed. They may forget that every batch of data from the DataLoader also needs to be moved. | Move only the model to GPU, leave data on CPU. First training iteration hits `RuntimeError: Expected all tensors to be on the same device`. The error is familiar from tensors (2.1.1) but now in a real training context. | Core Explain section -- show the error first, then the fix. Problem before solution. |
| **"I need to rewrite my training loop for GPU"** | The student may expect GPU training to require a fundamentally different code structure -- special GPU-aware layers, different APIs, or a separate training function. | Show the CPU training loop and the GPU training loop side by side. Highlight the only differences: device variable declaration, `model.to(device)`, and `inputs.to(device), targets.to(device)` in the loop body. The heartbeat is identical. 3 lines of change, not a rewrite. | After the first device mismatch error is fixed -- the relief of "that is all?" is the pedagogical moment. |
| **"Mixed precision means using float16 everywhere"** | The word "mixed" is easy to miss. If float16 is faster, why not use it for everything? The student may think mixed precision = full float16. | Show that computing gradients in float16 causes underflow -- small gradients round to zero, and the model stops learning. Mixed precision uses float16 for forward/backward (speed) but float32 for gradient accumulation and weight updates (stability). It is mixed because different operations use different precisions. | Mixed precision section -- explicitly name and disprove this before introducing torch.amp. |
| **"Mixed precision is complicated and requires manual dtype management"** | The student has seen `.float()` and `.half()` as manual conversion methods. They may expect mixed precision to require manually casting every tensor to the right dtype at the right time. | Show the manual approach (3-4 lines of casting per operation, error-prone) then reveal `torch.amp.autocast` which does it automatically in 2 lines. The contrast demonstrates that PyTorch automates the complexity. | Mixed precision section -- manual-then-automated progression. Connects to the autograd lesson pattern: "Not magic -- automation." |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Timing: small model on tiny data (CPU faster) vs MNIST model on full data (GPU faster)** | Positive + Negative | Shows that GPU is not universally faster; establishes the crossover point where GPU wins. Two examples in one comparison. | Directly disproves the #1 misconception. Uses the student's own MNIST model (familiar) and a tiny model (shows the boundary). The timing numbers make the tradeoff concrete, not theoretical. |
| **Device mismatch error in a real training loop** | Negative | Shows what happens when you forget to move data to GPU -- the familiar device mismatch error in a new context. | The student saw this error on individual tensors in 2.1.1. Seeing it in a training loop is a higher-stakes version. It motivates the device-aware training pattern -- "this is why you need to move data too." |
| **CPU training loop vs GPU training loop side-by-side** | Positive | Shows that the GPU training loop is the CPU loop with exactly 3 lines changed. Disproves the "need to rewrite" misconception. | The relief of "that is all?" is the central pedagogical moment. Uses ComparisonRow to make the minimal diff visually obvious. Extends "same heartbeat, new instruments." |
| **float16 gradient underflow (why not full float16)** | Negative | Shows that using float16 everywhere causes gradients to underflow to zero, breaking training. | Directly disproves the "mixed precision = all float16" misconception. The student needs to understand WHY it is "mixed" before the autocast API makes sense. |
| **torch.amp.autocast: before and after** | Positive | Shows the autocast context manager wrapping the forward pass and the GradScaler wrapping the backward pass. Minimal code change, automatic precision management. | Demonstrates that mixed precision in modern PyTorch is nearly automatic. Connects to "not magic -- automation" from autograd. |

---

## Phase 3: Design

### Narrative Arc

You have been training models on CPU this entire series. Your MNIST model reached 97% accuracy in about a minute -- fast enough that you never felt the pain. But that is about to change. As models get larger and datasets get bigger (and they will -- CNNs, transformers, diffusion models are all ahead), CPU training goes from "a bit slow" to "physically impossible." A model that takes 2 minutes on GPU might take 2 hours on CPU, or never finish at all. This lesson answers two questions: how do you move your existing training code to GPU (it is easier than you think), and how do you squeeze even more speed out of the GPU with mixed precision. The good news: you already know almost everything. The training loop does not change. The "same heartbeat, new instruments" pattern holds -- GPU placement is just one more instrument.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example (timing)** | Wall-clock timing comparisons: small model/tiny data (CPU wins) vs MNIST model/full data (GPU wins). Actual seconds displayed. | "GPU wins at scale" was an assertion in 2.1.1. This lesson needs to make it empirical. The student should see real numbers, not just believe the claim. Timing is the ground truth. |
| **Symbolic (code)** | Side-by-side CPU vs GPU training loops with highlighted differences. Device-aware checkpoint save/load code. torch.amp.autocast + GradScaler code. | This is fundamentally a code lesson. The training loop IS the concept. The student needs to see exact code, type it, and run it. |
| **Visual (side-by-side diff)** | ComparisonRow showing CPU loop vs GPU loop, with the 3 changed lines highlighted. Before/after for mixed precision (standard loop vs autocast loop). | The "it is only 3 lines" insight is visual. A wall of code obscures it; a highlighted diff reveals it. The student should SEE the minimal change. |
| **Verbal/Analogy** | "Same assembly line, faster workers" -- extending the GPU-as-parallel-processor concept. The training loop is the assembly line. Moving to GPU swaps in faster workers for the compute-heavy steps. The assembly line itself (the loop structure) does not change. | Connects to "same heartbeat, new instruments" and to the CPU/GPU worker comparison from tensors (2.1.1). The analogy makes explicit that the loop structure is invariant. |
| **Intuitive** | The precision tradeoff: float64 is a micrometer (too precise), float32 is a ruler (right-sized), float16 is a tape measure (good enough for the forward pass, not for tiny gradient updates). Extends the "rough sketch with a micrometer" analogy. | The student already has the float64-is-overkill intuition. Extending the analogy to float16 makes mixed precision feel natural: "of course you would use a rougher tool for rough work and a finer tool for fine work." |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. Device-aware training loop pattern (model + data + targets all on the same device) -- this DEVELOPS the INTRODUCED device management concept
  2. Mixed precision training with torch.amp.autocast and GradScaler -- genuinely new
  3. (Borderline) The "when does GPU help?" timing-based decision -- this is more of a DEVELOPED extension of the existing "GPU wins at scale" mental model than a new concept
- **Previous lesson load:** BUILD (saving-and-loading)
- **This lesson's load:** STRETCH -- appropriate. The device-aware training loop requires coordinating device placement across model, data batches, and targets within the training loop, which is a coordination challenge beyond any single API call. Mixed precision adds a second genuinely new concept. Coming after a BUILD lesson means the student has had a lower-load session to recover.

The load trajectory BUILD -> STRETCH -> CONSOLIDATE mirrors Module 2.2 and is within bounds.

### Connections to Prior Concepts

- **"GPU wins at scale" (tensors, 2.1.1)** -- This lesson develops the INTRODUCED concept with empirical timing on real training runs, not just individual tensor operations. The mental model is confirmed and refined: the crossover depends on model size AND data size.
- **Device mismatch error (tensors, 2.1.1)** -- Seen once on individual tensors. Now encountered in a real training loop where the consequences are more complex (model on GPU, data on CPU). Same error, higher-stakes context.
- **"Same heartbeat, new instruments" (training-loop, 2.1.4)** -- The GPU training loop IS the same heartbeat. GPU placement is a new instrument that slots into the existing pattern.
- **"Rough sketch with a micrometer" (tensors, 2.1.1)** -- Extends to float16: if float64 is too precise, float16 might be precise enough for parts of the computation. The precision tradeoff is the same principle applied at a different scale.
- **"Not magic -- automation" (autograd, 2.1.2)** -- torch.amp.autocast is automation of manual precision management, just as autograd is automation of manual differentiation.
- **Checkpoint pattern (saving-and-loading, 2.3.1)** -- Device-aware checkpoints need map_location. The student already has the checkpoint pattern; this lesson adds the device portability layer.
- **map_location preview (saving-and-loading, 2.3.1)** -- Was briefly previewed. This lesson develops it into a complete pattern for portable checkpoints.

**Potentially misleading prior analogies:** None identified. The "GPU wins at scale" model is correct but incomplete -- this lesson refines it with empirical boundaries rather than contradicting it.

### Scope Boundaries

**This lesson IS about:**
- Device-aware training loop: `model.to(device)`, `inputs.to(device)`, `targets.to(device)`
- CUDA availability checking and the portable device pattern
- Timing CPU vs GPU training to establish when GPU helps
- Mixed precision basics: `torch.amp.autocast` and `GradScaler`
- Device-aware checkpoints (saving on GPU, loading on CPU and vice versa)
- Practical guidelines for when to use GPU and when mixed precision helps

**This lesson is NOT about:**
- Multi-GPU or distributed training (out of scope for Series 2)
- CUDA programming, kernels, or streams
- Memory management, `torch.cuda.empty_cache()`, or OOM debugging (mentioned briefly if relevant but not developed)
- `pin_memory` or `num_workers` in DataLoader (mentioned as optimization tips, not developed)
- Model parallelism or pipeline parallelism
- TensorCores or hardware-specific optimization
- bfloat16 (mentioned as alternative to float16, not developed)
- Custom CUDA extensions or torch.compile
- Profiling or benchmarking tools (torch.profiler)
- Specific GPU model recommendations or cloud provider guidance

**Target depths:**
- Device-aware training loop pattern: DEVELOPED (write it, understand each line, handle edge cases)
- "When does GPU help?" decision: DEVELOPED (can articulate the tradeoff, has seen empirical evidence)
- Mixed precision (autocast + GradScaler): INTRODUCED (can use the 2-line pattern, understand why it is "mixed," know when it helps)
- Device-aware checkpoints: DEVELOPED (extends existing checkpoint pattern with device portability)
- num_workers / pin_memory: MENTIONED (optimization tips for the future)

### Lesson Outline

1. **Context + Constraints** -- This lesson is about training on GPU: moving your existing training code to GPU and accelerating it with mixed precision. We are NOT covering multi-GPU, CUDA programming, or hardware optimization. By the end, you can write training code that runs on whatever hardware is available and use mixed precision when it helps.

2. **Hook** -- Type: *timing challenge / before-after*. Present the student's MNIST model: "Your MNIST model trains in about 60 seconds on CPU. That is fine. But your next project (Fashion-MNIST) has the same data size and you will want to experiment more. And after that, CNNs and larger models are coming. Let's see what happens." Show two timing numbers: a small model on tiny data (CPU faster) and MNIST on full data (GPU faster). The student should feel the inflection point where CPU stops being sufficient. Directly disproves "GPU is always faster" with the small-data example. Ask: "So when does GPU help, and how do you use it?"

3. **Recap: Device Fundamentals** (brief, 2-3 paragraphs) -- The student has device management at INTRODUCED depth from 6 lessons ago. Brief refresher: the device pattern (`torch.device('cuda' if ... else 'cpu')`), `.to(device)` moves tensors, all tensors in an operation must be on the same device. This is not re-teaching -- it is activation of existing knowledge before building on it.

4. **Explain: The Device Mismatch Problem** -- Start with the problem. Take the existing CPU training loop and add only `model.to(device)`. Run it. `RuntimeError: Expected all tensors to be on the same device`. The model is on GPU, the data is on CPU. The student saw this error on individual tensors in 2.1.1 -- now it appears in a real training context. The error motivates the solution: data must move too.

5. **Explain: The Device-Aware Training Loop** -- Show the fix: `inputs, targets = inputs.to(device), targets.to(device)` inside the loop. Present the full GPU training loop side-by-side with the CPU loop (ComparisonRow). Highlight the 3 changed lines. The moment: "That is all? Just 3 lines?" Reinforce: "Same heartbeat, new instruments." The loop structure is identical. Address the "need to rewrite" misconception explicitly.

6. **Check 1** -- Predict-and-verify: "You move your model to GPU with `model.to(device)` but forget to move the data. What error do you get, and on which line?" (Answer: RuntimeError on the forward pass line -- model expects GPU tensors, gets CPU tensors.) Follow-up: "Your colleague puts `inputs.to(device)` OUTSIDE the training loop, before it starts. Why does this not work?" (Answer: DataLoader yields new CPU tensors each iteration -- you must move each batch inside the loop.)

7. **Explain: When Does GPU Help?** -- Return to the timing hook. Develop the "GPU wins at scale" model with training-specific factors: (a) model size -- more parameters means more parallel computation, (b) batch size -- larger batches give the GPU more work to parallelize, (c) data transfer overhead -- each batch moves CPU-to-GPU every iteration. Present practical guidelines: "If your model trains in under 30 seconds on CPU, GPU probably does not help. If training takes minutes to hours, GPU will likely give 3-10x speedup." This is a heuristic, not a law.

8. **Explain: Device-Aware Checkpoints** -- Connect to the saving-and-loading lesson. "Your checkpoint from last lesson now includes tensors on GPU. What happens when you save on GPU and load on CPU?" Show the pattern: `torch.load('checkpoint.pth', map_location=device)`. Extend the portable device pattern to cover the complete save/load cycle. Show the robust pattern: always save with `model.cpu().state_dict()` or use `map_location` on load. Brief -- this connects two existing concepts (checkpoints + device management) rather than introducing new ones.

9. **Explore: GPU Training in Practice** -- A guided walkthrough code block (not a widget): the student's MNIST model from 2.2.2, now with device-aware training, checkpointing, and timing. Print device being used, training time per epoch, total time. Compare mentally to CPU time. This is the "do it for real" moment. Designed to be typed into Colab.

10. **Check 2** -- Transfer question: "A colleague says their GPU training is SLOWER than CPU. Their model has 500 parameters and their dataset has 200 samples with batch_size=32. What would you tell them?" (Answer: model and data are too small for GPU to overcome transfer overhead. Stick with CPU until model/data scale up.)

11. **Explain: Why Mixed Precision?** -- Transition: "GPU training is faster. Can we make it even faster?" Revisit the precision analogy: float64 is a micrometer, float32 is a ruler. Introduce float16 as a tape measure: "good enough for measuring the rough shape (forward pass), not precise enough for fine adjustments (gradient updates)." Show that float16 uses half the memory and modern GPUs have hardware (Tensor Cores) that process float16 2-4x faster. But: show the failure mode. Full float16 training causes gradient underflow -- small gradients round to zero, learning stops. This is why it is called "mixed" precision.

12. **Explain: torch.amp (Automatic Mixed Precision)** -- Problem before solution established. Now the solution: `torch.amp.autocast(device_type='cuda')` wraps the forward pass, automatically choosing float16 where safe and float32 where needed. `torch.amp.GradScaler()` scales the loss up before backward (so small gradients do not underflow to zero in float16) and scales gradients back down before optimizer.step(). Show the code pattern: autocast context manager around forward + loss, scaler.scale(loss).backward(), scaler.step(optimizer), scaler.update(). Compare to the standard training loop: 4 lines changed, same structure. Connect to "not magic -- automation" from autograd.

13. **Elaborate: When Mixed Precision Helps** -- Mixed precision helps most when: (a) using a GPU with Tensor Cores (most modern NVIDIA GPUs), (b) the model is large enough that memory/compute is the bottleneck, (c) the forward pass dominates training time. It helps less for small models or when data loading is the bottleneck. On CPU, autocast uses bfloat16 instead of float16 (mention briefly, do not develop). Practical guideline: "Try it. If training gets faster without accuracy loss, keep it. If accuracy drops, remove it."

14. **Check 3** -- Predict-and-verify: "Why does GradScaler multiply the loss by a large number before backward()?" (Answer: float16 has a smaller range. Small gradients that would be zero in float16 become non-zero after scaling. The scaler divides them back before the optimizer step.) This tests understanding of WHY mixed precision is "mixed," not just the API.

15. **Practice** -- Colab notebook exercises:
    - **Guided:** Move the MNIST model and training loop to GPU. Time it. Compare to CPU.
    - **Supported:** Add device-aware checkpointing. Save during GPU training, load and verify on CPU using map_location.
    - **Supported:** Add mixed precision (autocast + GradScaler) to the GPU training loop. Compare training speed with and without.
    - **Independent:** Write a complete, portable training script that: detects device, uses GPU if available, uses mixed precision if on GPU, checkpoints with device portability. This is the "production-ready" pattern the student carries forward.

16. **Summarize** -- Key takeaways:
    - GPU training is the same training loop with 3 lines of device placement added.
    - All tensors (model, inputs, targets) must be on the same device -- the device mismatch error tells you when they are not.
    - GPU wins at scale; small models on small data can be slower on GPU.
    - Mixed precision uses float16 for speed where safe, float32 for precision where needed. autocast + GradScaler handle this automatically.
    - Portable checkpoints use map_location to load across devices.

17. **Next step** -- "You now have the full practical toolkit: build, train, save, load, checkpoint, GPU, mixed precision. The next lesson puts it all together: an independent Fashion-MNIST project where you make your own design decisions."

### Widget Decision

**No custom interactive widget needed.** Like saving-and-loading, this is a practical code lesson where the interaction happens in Colab. The key pedagogical moments are:

1. The timing comparison (numbers in the lesson, reproduced in Colab) -- does not need interactivity beyond running cells.
2. The side-by-side CPU vs GPU loop -- best served by a ComparisonRow component, not a widget.
3. The mixed precision code -- an API pattern best learned by typing and running it.

A widget that simulates GPU timing or device placement would be artificial -- the real thing (actual GPU training in Colab) is better. Recharts could be used for a timing comparison bar chart (CPU time vs GPU time vs GPU+AMP time) but this is optional and simple enough to be a static visualization.

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
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2-3 new concepts: device-aware loop, mixed precision, when-GPU-helps refinement)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings. The lesson is well-structured, follows the plan faithfully, and successfully teaches the core concepts. However, four improvement-level findings would meaningfully strengthen the lesson if addressed: a missing misconception treatment, the lack of a concrete negative example for mixed precision, a checkpoint-section detail that may confuse the student, and the absence of a verbal/analogy modality for the device-aware training loop concept itself.

### Findings

#### [IMPROVEMENT] — Missing planned misconception: "Mixed precision is complicated and requires manual dtype management"

**Location:** Section 12 (torch.amp: Automatic Mixed Precision)
**Issue:** The planning document identifies misconception #5: "Mixed precision is complicated and requires manual dtype management." The plan calls for showing the manual approach (3-4 lines of casting per operation, error-prone) and then revealing torch.amp.autocast as the automated solution. The built lesson mentions this in passing ("You could manually cast every tensor to the right dtype at the right time... That would be error-prone and tedious. Instead, PyTorch automates it") but never actually shows the manual approach. The plan says "Show the manual approach then reveal autocast" as a contrast. The lesson skips the "show" part and only tells the student it would be tedious.
**Student impact:** The student misses the contrast that makes autocast feel like a relief. Without seeing even a sketch of manual dtype management, the claim "this would be error-prone and tedious" is an assertion rather than a demonstration. The "not magic--automation" connection to autograd is weaker because the autograd lesson actually showed the manual version first.
**Suggested fix:** Add a brief 3-4 line pseudo-code or code snippet showing what manual mixed precision would look like (cast to float16 before forward, cast back to float32 before backward, manage dtypes on loss, etc.). Does not need to be runnable -- just enough to make the student wince before seeing autocast. This mirrors the autograd lesson's pattern of showing manual work before the automation.

#### [IMPROVEMENT] — No concrete negative example for "full float16 breaks training"

**Location:** Section 11 (Why Mixed Precision?)
**Issue:** The lesson shows a code block demonstrating that very small float32 values round to zero in float16 (the underflow demonstration). This is a good standalone example. However, the planning document calls for showing that "full float16 training causes gradient underflow -- small gradients round to zero, learning stops." The built lesson demonstrates underflow on isolated tensors but never shows it in the context of training. The student sees that 1e-08 rounds to zero but does not see that this causes a training loop to stop learning. The connection "underflow -> model stops learning" is stated verbally but not demonstrated.
**Student impact:** The student understands the float16 underflow mechanism on individual values but must take on faith that this actually breaks training. A software engineer with a skeptical mindset might wonder: "How often do gradients actually get that small? Maybe it is fine in practice." The lesson does not answer this.
**Suggested fix:** Add a brief verbal bridge after the underflow code block: something like "In a deep network, gradients passing through many layers often reach magnitudes of 1e-6 to 1e-8. At float16 precision, these effectively disappear. The optimizer receives zeros and makes no update. The model appears to train (loss computation still works in float16) but the deepest layers stop learning." This grounds the underflow example in training reality without requiring a full training demonstration.

#### [IMPROVEMENT] — Checkpoint section uses `weights_only=False` without recapping security context

**Location:** Section 8 (Device-Aware Checkpoints), line 529
**Issue:** The code block shows `torch.load('checkpoint.pth', map_location=device, weights_only=False)`. The `weights_only` parameter was INTRODUCED in the saving-and-loading lesson (2.3.1) with security context. Here it appears in a code example without any explanation. A student who has partially forgotten the saving-and-loading details might wonder: "Why False? What does this mean?" More importantly, setting `weights_only=False` is the less-safe option, and the student should be reminded why it is needed here (the checkpoint dict contains non-tensor metadata like epoch and loss).
**Student impact:** Mild confusion or glossing over a security-relevant parameter. The student may copy the pattern without understanding why `weights_only=False` is needed, and may default to using it everywhere.
**Suggested fix:** Add a brief inline comment in the code block or a one-sentence note after it: "weights_only=False is needed because our checkpoint dictionary contains metadata (epoch, loss) alongside tensors. For state_dicts alone, use weights_only=True." This is a 15-word addition that reinforces the saving-and-loading lesson's concept.

#### [IMPROVEMENT] — Verbal/analogy modality underused for the device-aware training loop concept

**Location:** Section 5 (The Device-Aware Training Loop) and Section 7 (When Does GPU Help?)
**Issue:** The planning document identifies five modalities, including "Verbal/Analogy: Same assembly line, faster workers." The aside in Section 7 uses the assembly line analogy briefly ("Think of the training loop as an assembly line. Moving to GPU swaps in faster workers..."). However, this is in an aside, not in the main content flow. The core device-aware training loop section (Section 5) relies almost entirely on code (symbolic) and the side-by-side visual (ComparisonRow). The "same heartbeat, new instruments" phrase appears but is not developed as an analogy -- it is used as a tagline rather than a mapped metaphor. The plan calls for the assembly line analogy as a primary modality, not a sidebar.
**Student impact:** The student who learns best through analogy gets the concept mostly from code. The "same heartbeat, new instruments" tagline is memorable but does not map the parts: what is the heartbeat (the loop structure), what are the instruments (device placement calls), why do the instruments not change the heartbeat (because .to() moves data but does not alter the computation). The mapping is implicit.
**Suggested fix:** In Section 5, after the ComparisonRow and before or after the "Three lines changed" paragraph, add 2-3 sentences that explicitly map the analogy: "The assembly line has four stations: forward, loss, backward, update. Moving to GPU does not add or remove any station. It upgrades the workers at each station to faster ones (GPU cores instead of CPU cores). The three new lines (.to(device)) are the truck that transports parts to the faster factory -- they are logistics, not a new manufacturing step." This makes the analogy do real work rather than serving as a label.

#### [POLISH] — Section 8 checkpoint save pattern does not follow the plan's "robust pattern" suggestion

**Location:** Section 8 (Device-Aware Checkpoints)
**Issue:** The planning document mentions "Show the robust pattern: always save with model.cpu().state_dict() or use map_location on load." The built lesson only shows the map_location-on-load approach. This is the simpler and more common pattern, and is sufficient. The alternative (saving with model.cpu().state_dict()) is an unnecessary complication that might confuse the student. This is a planned element that was reasonably omitted during building.
**Student impact:** None negative. The omission is fine. Documenting it for the record.
**Suggested fix:** No fix needed. Acknowledge as a reasonable deviation from the plan.

#### [POLISH] — "about six lessons ago" in the recap section is imprecise

**Location:** Section 3 (Recap: Device Fundamentals), line 175
**Issue:** The text says "You learned device management in Tensors, about six lessons ago." Counting from tensors (2.1.1) to gpu-training (2.3.2): tensors -> autograd -> nn-module -> training-loop -> datasets-and-dataloaders -> mnist-project -> debugging-and-visualization -> saving-and-loading -> gpu-training. That is 8 lessons ago, not 6. The imprecision is minor but a student who actually counts could lose a small amount of trust.
**Student impact:** Negligible. Most students will not count.
**Suggested fix:** Change "about six lessons ago" to "several lessons ago" or count accurately ("eight lessons ago").

#### [POLISH] — Colab link points to a notebook that may not exist yet

**Location:** Section 15 (Build It Yourself), line 1015
**Issue:** The href points to `notebooks/2-3-2-gpu-training.ipynb`. If this notebook has not been created yet, the link will 404.
**Student impact:** If the notebook is not ready, the student clicks the link and gets an error page. Frustrating but not a lesson content issue.
**Suggested fix:** Verify the notebook exists. If not, create it before marking the lesson as complete.

### Review Notes

**What works well:**
- The narrative arc is strong. The lesson starts with a genuine problem (CPU training will not scale), makes it concrete with timing numbers, and builds toward a portable solution. The "same heartbeat, new instruments" thread runs through the entire lesson and gives it coherence.
- The ordering is excellent: problem before solution at every level. The device mismatch error comes before the fix. The float16 underflow comes before mixed precision. The student feels each problem before getting the tool.
- The ComparisonRow usage is effective: CPU vs GPU training loops side-by-side, small vs large model timing. These visual diffs are where the key insights land.
- The three checks are well-placed and test the right things: predict-the-error (device mismatch), transfer (advise a colleague about slow GPU training), and explain-the-mechanism (GradScaler purpose).
- Scope boundaries are respected. The lesson does not wander into multi-GPU, CUDA programming, or memory management.
- Connections to prior concepts are explicit and well-done: the "rough sketch with a micrometer" extension to float16, the "not magic -- automation" connection to autograd, the checkpoint pattern extension.
- The cognitive load is well-managed for a STRETCH lesson. Device-aware training and mixed precision are the two genuine new concepts. The "when does GPU help?" section is a refinement of an existing mental model, not a new concept.

**Patterns to watch:**
- The lesson is quite long (1089 lines of JSX). For a code-centric lesson without a widget, this is expected but worth monitoring for student fatigue. The practice section comes late (section 15 of 17). ADHD-friendly design suggests keeping activation energy low -- consider whether the student needs a breather point earlier.
- The lesson has three code blocks that are essentially the same training loop with incremental additions (basic GPU loop, complete GPU loop, mixed precision loop). This is intentional (scaffolding) but could feel repetitive if not justified. The "same heartbeat" framing helps but might need reinforcement.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 3

### Verdict: PASS

All four improvement findings from iteration 1 have been successfully addressed. The manual mixed precision code block is now present and creates effective contrast with autocast. The verbal bridge grounding float16 underflow in training reality is clear and well-placed. The `weights_only=False` explanation reinforces the saving-and-loading lesson's concept. The assembly line analogy is now explicitly mapped in the main content with four stations, faster workers, and logistics framing. The imprecise "about six lessons ago" has been corrected to "several lessons ago." No new critical or improvement-level issues found. Three minor polish items remain.

### Findings

#### [POLISH] — Check 3 answer contains meta-commentary about itself

**Location:** Section 14 (Check 3: Predict-and-Verify), inside the answer reveal
**Issue:** The answer ends with: "This tests understanding of why mixed precision is 'mixed' -- float16 alone is not precise enough for small gradient values." This reads as a note to the lesson builder about what the check is testing, not as part of the student-facing answer. It is inside the `<details>` reveal, so the student sees it after clicking "Show answer."
**Student impact:** Mildly jarring. The student is reading an explanation and encounters a sentence that talks about the check in the third person ("This tests understanding of..."). It breaks the student-facing voice.
**Suggested fix:** Remove the sentence or rewrite it in student-facing voice: "The key idea: mixed precision is 'mixed' because float16 alone is not precise enough for small gradient values."

#### [POLISH] — "Complete, copy-ready" script omits MNISTClassifier definition

**Location:** Section 9 (GPU Training in Practice), subtitle and code block
**Issue:** The subtitle says "The complete pattern, ready to type into Colab" and the code block includes standard imports but uses `MNISTClassifier()` without defining or importing it. The student built this class in the MNIST project (2.2.2) and would bring it into a notebook, but calling the pattern "complete" overpromises slightly. Earlier code snippets using `MNISTClassifier()` are clearly fragments, but this section positions itself as the definitive reference.
**Student impact:** A student copying the code block into a fresh Colab cell would get a `NameError: name 'MNISTClassifier' is not defined`. Momentary confusion before they realize they need their model class.
**Suggested fix:** Add a comment in the code block after the imports: `# (MNISTClassifier class definition goes here -- from your MNIST project)` or soften the subtitle to "The complete pattern, ready to type into Colab alongside your model class."

#### [POLISH] — Colab notebook still does not exist

**Location:** Section 15 (Build It Yourself), Colab link
**Issue:** The href points to `notebooks/2-3-2-gpu-training.ipynb`. This file does not exist in the repository. The link will 404 when clicked.
**Student impact:** Student clicks the link and gets an error page. Frustrating.
**Suggested fix:** Create the notebook before marking the lesson as complete. (Carried forward from iteration 1.)

### Review Notes

**Iteration 1 fixes verified:**
All four improvement findings from iteration 1 have been addressed. Specifically:
1. Manual mixed precision code block added (section 12, ~lines 840-855) -- shows the tedious approach before autocast, creating effective contrast. The "not magic -- automation" connection now mirrors the autograd lesson's manual-then-automated pattern.
2. Verbal bridge for float16 underflow in training (section 11, ~lines 796-804) -- grounds the underflow example in real training scenarios (gradients at 1e-6 to 1e-8 in deep networks). The student no longer needs to take on faith that underflow breaks training.
3. `weights_only=False` explanation added (section 8, ~lines 561-569) -- brief note connecting back to the saving-and-loading lesson. Reinforces when to use True vs False.
4. Assembly line analogy explicitly mapped in main content (section 5, ~lines 352-367) -- four stations (forward, loss, backward, update), faster workers (GPU cores), logistics not a new step (the .to(device) lines). The analogy now does real explanatory work rather than serving as a tagline.

**What works well (confirmed from iteration 1, still holds):**
- The narrative arc remains strong and coherent. "Same heartbeat, new instruments" runs through the entire lesson.
- Problem-before-solution ordering is consistent at every level.
- All five planned modalities are present and effective.
- All five planned misconceptions are addressed with concrete examples.
- Scope boundaries are respected throughout.
- Connections to prior concepts are explicit and numerous (tensors, autograd, training-loop, saving-and-loading, MNIST project).
- Cognitive load is well-managed for a STRETCH lesson.
- The three checks test the right things at the right level.

**Overall assessment:** The lesson is pedagogically sound and ready to ship. The three remaining polish items are minor and can be addressed quickly without re-review. The lesson successfully teaches device-aware training as a natural extension of existing patterns rather than a new paradigm, and introduces mixed precision with clear motivation and appropriate depth for an INTRODUCED concept.
