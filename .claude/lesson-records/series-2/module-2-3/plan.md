# Module 2.3: Practical Patterns — Plan

**Status:** Planning (0 of 3 built)

## Module Goal

The student can save, load, and checkpoint models, train on GPU with device-aware code, and pull together all Series 2 skills in an independent Fashion-MNIST project — completing the bridge from theory to production-ready practice.

## Narrative Arc

Modules 2.1 and 2.2 gave the student the complete PyTorch workflow: build a model, load real data, train it, evaluate it, debug it. But every training run has been ephemeral — close the notebook and the model is gone. The student also hasn't used a GPU, and hasn't attempted a project without step-by-step scaffolding.

Module 2.3 wraps up Series 2 by addressing these three practical gaps:

1. **Saving and Loading** (the durability problem) — Training takes time and compute. If the process crashes at epoch 47 of 50, you lose everything. Checkpointing solves this. The student also needs to understand state_dict as the canonical way to persist and share models. This is the foundation for every real workflow.
2. **GPU Training** (the speed problem) — The student has seen GPUs conceptually (tensor lesson 2.1.1 covered device placement and transfer overhead) but has never trained a model on GPU. This lesson closes the loop: device-aware training code, when GPU actually helps, and a taste of mixed precision.
3. **Fashion-MNIST Project** (the independence test) — A capstone project with less scaffolding than MNIST. The student picks their own architecture, applies regularization, uses checkpointing and GPU training, and tries to beat a baseline. This is the "graduation exam" for Series 2.

The emotional arc: "My models disappear when training ends" -> "I can save, resume, and accelerate" -> "I can do this on my own."

## Lesson Sequence

| # | Slug | Core Concept | Type | Rationale for Position |
|---|------|-------------|------|----------------------|
| 1 | saving-and-loading | Persisting and restoring model state via state_dict and checkpoints | BUILD | Must come before GPU training (checkpointing is used during long GPU runs) and before the project (student needs to save their best model); low-stress API lesson after the debugging CONSOLIDATE |
| 2 | gpu-training | Device-aware training with .to(device) and mixed precision basics | STRETCH | Builds on saving-and-loading (checkpoint includes device state); must come before the project (student should train on GPU); extends tensor lesson's GPU concepts to full training |
| 3 | fashion-mnist-project | Independent end-to-end project combining all Series 2 skills | CONSOLIDATE / PROJECT | Capstone position: uses everything from the series; less scaffolding than MNIST; student proves independence |

## Rough Topic Allocation

| Lesson | Topics | Notes |
|--------|--------|-------|
| saving-and-loading | `model.state_dict()`, `torch.save()`, `torch.load()`, `model.load_state_dict()`, saving/loading optimizers, checkpoint dict pattern, resuming training from checkpoint, `torch.save(model)` vs `torch.save(state_dict)` | Connects to early stopping (1.3.7 "save best model weights") and the debugging lesson's systematic workflow |
| gpu-training | `.to(device)` for model and data, device-aware training loop, CUDA availability checking, timing CPU vs GPU, mixed precision with `torch.amp`, `torch.cuda.amp.GradScaler` | Extends tensors lesson (2.1.1) GPU concepts; connects to "GPU wins at scale" mental model |
| fashion-mnist-project | Architecture design choices, Fashion-MNIST dataset, baseline model, experimentation with hyperparameters, checkpointing best model, GPU training, regularization toolkit | Integrates all of Series 2; connects to MNIST project (2.2.2) but with more independence |

## Cognitive Load Trajectory

```
Lesson 1 (saving-and-loading):     BUILD        — Familiar model objects, new persistence API
Lesson 2 (gpu-training):           STRETCH      — Device management across model + data + training loop
Lesson 3 (fashion-mnist-project):  CONSOLIDATE  — Integration project, no new concepts
```

BUILD -> STRETCH -> CONSOLIDATE. This mirrors Module 2.2's trajectory. The BUILD lesson introduces a clean, self-contained API. The STRETCH lesson requires coordinating device state across model, data, and optimizer. The CONSOLIDATE project pulls everything together without introducing new concepts.

## Module-Level Misconceptions

- **"torch.save(model) is the right way to save"** — Students may save the entire model object instead of the state_dict, creating fragile pickles tied to exact class definitions and file paths. state_dict is portable; the model object is not.
- **"GPU is always faster"** — The student already has the "GPU wins at scale" mental model from 2.1.1, but may still assume GPU training is universally better. Small models on small data can actually be slower on GPU due to transfer overhead.
- **"Moving to GPU is just one line of code"** — Device management requires consistency: model, input data, and targets must all be on the same device. Forgetting to move one tensor causes a device mismatch error.
- **"Saving the model saves everything needed to resume training"** — Resuming training requires optimizer state (momentum buffers, adaptive rates), not just model weights. Without optimizer state, Adam "forgets" its momentum and training regresses.
- **"Mixed precision is complicated and risky"** — Modern PyTorch makes mixed precision nearly automatic with torch.amp. The complexity is in understanding WHEN it helps, not in using the API.
