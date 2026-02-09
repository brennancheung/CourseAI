# Module 2.3: Practical Patterns — Record

**Goal:** The student can save, load, and checkpoint models, train on GPU with device-aware code, and pull together all Series 2 skills in an independent Fashion-MNIST project.
**Status:** In progress (1 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| model.state_dict() as dictionary of named tensors | DEVELOPED | saving-and-loading | Printed immediately to demystify; layer names (fc1.weight, fc1.bias, etc.) mapped to tensor shapes. "Snapshot of all the knobs." |
| torch.save() / torch.load() for state_dicts | DEVELOPED | saving-and-loading | Complete round-trip pattern: save state_dict to .pth, load into fresh model instance, verify with torch.allclose(). |
| load_state_dict() requires matching architecture | DEVELOPED | saving-and-loading | State_dict stores values only, not architecture. Size mismatch error if model class changes. Addressed as misconception (Keras users expect architecture + weights together). |
| optimizer.state_dict() (Adam momentum + adaptive rates) | DEVELOPED | saving-and-loading | Printed to show exp_avg (momentum) and exp_avg_sq (adaptive rate) per parameter. Connected to Optimizers lesson (1.3.5). |
| Checkpoint pattern (model + optimizer + epoch + loss dict) | DEVELOPED | saving-and-loading | Bundle all state into one dictionary; torch.save/load the dict. Integrated into training loop with periodic saves + best-model save. |
| Resume training from checkpoint | DEVELOPED | saving-and-loading | Load checkpoint, restore model and optimizer state_dicts, resume from saved epoch. Loss curve chart showed smooth continuation vs spike without optimizer state. |
| torch.save(model) pickle fragility | INTRODUCED | saving-and-loading | Negative example: save full model, rename class file, ModuleNotFoundError. Contrasted with state_dict via ComparisonRow. Shown as anti-pattern. |
| map_location for cross-device loading | INTRODUCED | saving-and-loading | Brief preview: map_location='cpu' or 'cuda:0' or device variable. Portable pattern shown. Full GPU training deferred to next lesson. |
| weights_only parameter in torch.load() | INTRODUCED | saving-and-loading | weights_only=True for state_dicts (tensors only), weights_only=False for checkpoint dicts with metadata. Security context briefly noted. |
| model.eval() after loading for inference | DEVELOPED | saving-and-loading | Reinforced from mnist-project (2.2.2): dropout disabled, batch norm uses running stats. model.train() for resume-training path. |
| Early stopping implementation with checkpoints | DEVELOPED | saving-and-loading | Connects conceptual early stopping (1.3.7) to concrete PyTorch code: patience counter + save best model + restore best at end. Independent exercise in Colab notebook. |

## Per-Lesson Summaries

### Lesson 1: Saving, Loading, and Checkpoints (`saving-and-loading`)

**Type:** BUILD | **New concepts:** 2 (state_dict, checkpoint pattern) | **Review:** PASS (iteration 2/3)

**Concepts taught:**
- state_dict as the canonical persistence format for both model and optimizer (DEVELOPED)
- Checkpoint pattern bundling model + optimizer + epoch + loss for crash recovery (DEVELOPED)
- torch.save(model) pickle fragility as anti-pattern (INTRODUCED)
- map_location for cross-device loading (INTRODUCED)
- weights_only parameter for torch.load security (INTRODUCED)

**Mental models established:**
- "state_dict is a snapshot of all the knobs" — extends "parameters are knobs" from Series 1
- "Architecture in code, values in file" — state_dict decouples learned values from code structure; this is a feature, not a limitation

**Analogies used:**
- "Photographing the position of every knob" (saving state_dict) / "Setting them back" (loading)
- "Giving an experienced pilot amnesia" (resuming without optimizer state)

**How concepts were taught:**
- state_dict demystified by printing it immediately (concrete before abstract) — student sees familiar layer names mapped to tensors
- Save/load pattern shown as complete code with verification step (torch.allclose proves round-trip fidelity)
- Optimizer state importance demonstrated via Recharts loss curve comparison: smooth resume (with optimizer state) vs loss spike (without). ReferenceLine at resume point makes the divergence visually obvious.
- torch.save(model) failure shown as negative example: rename class file -> ModuleNotFoundError. ComparisonRow contrasts the two approaches.
- Keras misconception explicitly named and reframed: "PyTorch deliberately separates architecture from weights. This is a feature."
- map_location shown briefly with portable device detection pattern; full GPU training deferred.
- model.eval() reinforced with explicit callback to MNIST project (batch norm + dropout behavior).

**What is NOT covered:**
- TorchScript, ONNX export, model compilation
- Distributed checkpointing, multi-GPU saving
- Experiment management tools (MLflow, Weights & Biases)
- Model versioning, production deployment
- Full GPU training (next lesson)

**Exercises (Colab notebook `2-3-1-saving-and-loading.ipynb`):**
1. (Guided) Save and load trained MNIST model, verify predictions match
2. (Supported) Add checkpointing to training loop — every 5 epochs + best model by val loss
3. (Supported) Simulate crash: train 10 epochs, checkpoint, fresh model, resume 10 more, verify continuous loss curve
4. (Independent) Implement full early stopping with checkpoints — patience, save best, restore best at end

**Notes for future lessons:**
- The student now has the save/load and checkpoint patterns. GPU training (next lesson) can build on map_location and device management.
- "Architecture in code, values in file" is a design principle the student should carry forward — relevant when they encounter model hubs (HuggingFace) or transfer learning.
- The loss curve chart (resume with vs without optimizer state) can be referenced when discussing learning rate schedulers or warm restarts.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Available For |
|---------------|---------------|---------------|
| "state_dict is a snapshot of all the knobs" | saving-and-loading | Any future discussion of model persistence, transfer learning, fine-tuning |
| "Architecture in code, values in file" | saving-and-loading | Model hubs, HuggingFace, transfer learning, model sharing |
| "Giving an experienced pilot amnesia" (resume without optimizer state) | saving-and-loading | Learning rate schedulers, warm restarts, any training resumption context |
