# Module 2.3: Practical Patterns — Record

**Goal:** The student can save, load, and checkpoint models, train on GPU with device-aware code, and pull together all Series 2 skills in an independent Fashion-MNIST project.
**Status:** Complete (3 of 3 lessons built)

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
| Device-aware training loop pattern (model + data + targets on same device) | DEVELOPED | gpu-training | Three lines added to existing training loop: model.to(device), inputs.to(device), labels.to(device) inside loop. "Same heartbeat, new instruments." Side-by-side ComparisonRow of CPU vs GPU loop. |
| Device mismatch RuntimeError in training context | DEVELOPED | gpu-training | Elevated from INTRODUCED (tensors, 2.1.1). Shown in multi-component context: model on GPU, DataLoader yields CPU tensors. Forward pass fails. Fix: move each batch inside the loop. |
| "When does GPU help?" timing-based decision | DEVELOPED | gpu-training | Three factors: model size, batch size, transfer overhead. Practical guideline: under 30s on CPU -> GPU probably doesn't help; minutes to hours -> 3-10x speedup. GradientCard triptych. |
| Device-aware checkpoint save/load with map_location | DEVELOPED | gpu-training | Extends checkpoint pattern (saving-and-loading) with device portability. map_location=device remaps tensors to current hardware. weights_only=False for checkpoint dicts with metadata. |
| Mixed precision training concept (float16 forward, float32 gradients) | INTRODUCED | gpu-training | "Mixed" means different precisions for different operations. Float16 underflow demonstrated on isolated tensors; grounded in training reality (gradients at 1e-6 to 1e-8 in deep networks round to zero). |
| torch.amp.autocast context manager | INTRODUCED | gpu-training | Wraps forward pass; automatically chooses float16 where safe, float32 where needed. Connected to "not magic -- automation" from autograd. Manual mixed precision shown first for contrast. |
| torch.amp.GradScaler | INTRODUCED | gpu-training | Scales loss up before backward (prevents float16 gradient underflow), scales gradients back down before optimizer step. 4-line addition to training loop. |
| Gradient underflow in float16 | INTRODUCED | gpu-training | Very small gradients (1e-8) round to zero in float16. Demonstrated with code showing float32 vs float16 values. This is WHY mixed precision is "mixed." |
| Portable device detection pattern in training scripts | DEVELOPED | gpu-training | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` at top of script; all .to(device) calls use this variable. Code runs identically on CPU and GPU. |
| Fashion-MNIST dataset (loading and class structure) | INTRODUCED | fashion-mnist-project | Drop-in replacement for MNIST: datasets.FashionMNIST with same API. 10 clothing classes, different normalization values (mean=0.286, std=0.353). Key insight: some classes are visually confusable (shirt/coat/pullover). |
| Per-class accuracy analysis | INTRODUCED | fashion-mnist-project | Compute accuracy per class separately using torch.argmax and per-class masks. Reveals easy classes (trousers, bags ~95%+) vs hard classes (shirt, coat, pullover ~75-85%). A single accuracy number hides important structure. |
| Independent model design and experimentation workflow | APPLIED | fashion-mnist-project | Student designs architecture, chooses regularization, trains, diagnoses with debugging checklist, and iterates without step-by-step scaffolding. Baseline -> diagnose -> experiment -> analyze -> full pipeline. |
| Complete training pipeline (all Series 2 skills integrated) | APPLIED | fashion-mnist-project | Device detection + data loading + regularized model + training with checkpointing and early stopping + per-class evaluation. Nine prior lessons integrated into one pipeline. |
| Regularization toolkit in a new context | APPLIED | fashion-mnist-project | BatchNorm + Dropout(0.3) + weight_decay applied to Fashion-MNIST. Linear->BatchNorm->ReLU->Dropout ordering. Scissors pattern closes: train accuracy decreases, test accuracy increases. |
| Mixed precision as optional pipeline addition | APPLIED | fashion-mnist-project | autocast + GradScaler added to complete pipeline as optional stretch goal. Reinforces gpu-training (2.3.2) INTRODUCED concept. |
| Baseline-then-improve experimental methodology | INTRODUCED | fashion-mnist-project | Start with simplest model (no regularization), observe results, diagnose (debugging checklist), then improve systematically. The baseline gives a number to beat and confirms the pipeline works. |

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

### Lesson 2: GPU Training (`gpu-training`)

**Type:** STRETCH | **New concepts:** 2-3 (device-aware training loop, mixed precision, "when does GPU help?" refinement) | **Review:** PASS (iteration 2/3)

**Concepts taught:**
- Device-aware training loop: model.to(device), inputs.to(device), labels.to(device) inside loop (DEVELOPED from INTRODUCED in tensors 2.1.1)
- Device mismatch error in multi-component training context (DEVELOPED from INTRODUCED in tensors 2.1.1)
- "When does GPU help?" timing-based decision with three factors (DEVELOPED refinement of existing mental model)
- Device-aware checkpoints with map_location (DEVELOPED, extends checkpoint pattern from saving-and-loading)
- Mixed precision training with torch.amp.autocast and GradScaler (INTRODUCED)
- Gradient underflow in float16 (INTRODUCED)
- Portable device detection pattern for training scripts (DEVELOPED)

**Mental models established:**
- "Same heartbeat, new instruments" (reinforced) -- GPU placement adds instruments to the existing training loop; the forward-loss-backward-step rhythm is unchanged
- "Assembly line with four stations" -- forward, loss, backward, update are stations; GPU upgrades the workers; .to(device) is the truck that transports parts to the faster factory (logistics, not a new manufacturing step)
- "Micrometer -> ruler -> tape measure" precision spectrum -- extends "rough sketch with a micrometer" to float16; float64 is too precise, float32 is right-sized, float16 is good enough for forward pass but not for tiny gradients
- "Not magic -- automation" (reinforced from autograd) -- torch.amp.autocast automates manual dtype management, just as autograd automates manual differentiation

**Analogies used:**
- Assembly line with four stations and faster workers (GPU training loop)
- .to(device) lines as "the truck that transports parts to the faster factory" (device placement as logistics)
- Micrometer / ruler / tape measure for float64 / float32 / float16 precision levels
- Manual mixed precision shown before autocast (mirrors autograd lesson's manual-then-automated pattern)

**How concepts were taught:**
- Device mismatch error shown first (problem before solution): model.to(device) without data movement -> RuntimeError. Familiar error from tensors (2.1.1) in higher-stakes context.
- CPU vs GPU training loop shown side-by-side via ComparisonRow: only 3 lines differ. The "that is all?" moment is the central insight.
- Assembly line analogy explicitly mapped in main content: four stations, faster workers, logistics framing. Does real explanatory work, not just a tagline.
- Timing comparison in hook: small model/tiny data (CPU wins) vs MNIST/full data (GPU wins). Directly disproves "GPU is always faster" misconception.
- Three factors for "when GPU helps" presented via GradientCard triptych: model size, batch size, transfer overhead.
- Device-aware checkpoints connect to saving-and-loading lesson: map_location=device for portable files. weights_only=False explained with callback to prior lesson.
- Float16 underflow demonstrated on isolated tensors, then grounded in training reality (verbal bridge: gradients at 1e-6 to 1e-8 in deep networks disappear at float16 precision).
- Manual mixed precision code shown before autocast: tedious manual casting creates contrast, making autocast feel like relief. Mirrors autograd lesson's pattern.
- torch.amp.autocast + GradScaler shown as 4-line addition to existing GPU training loop. Structure unchanged.
- Three checks test: predict-the-error (device mismatch), transfer (advise colleague about slow GPU), explain-the-mechanism (GradScaler purpose).

**What is NOT covered:**
- Multi-GPU or distributed training
- CUDA programming, kernels, or streams
- Memory management, torch.cuda.empty_cache(), OOM debugging
- pin_memory or num_workers in DataLoader (not even mentioned -- plan considered mentioning them)
- Model parallelism or pipeline parallelism
- TensorCores or hardware-specific optimization
- bfloat16 (mentioned briefly as CPU alternative, not developed)
- Custom CUDA extensions or torch.compile
- Profiling tools (torch.profiler)

**Exercises (Colab notebook `2-3-2-gpu-training.ipynb`):**
1. (Guided) Move MNIST model and training loop to GPU. Time it. Compare to CPU.
2. (Supported) Add device-aware checkpointing. Save during GPU training, load and verify on CPU using map_location.
3. (Supported) Add mixed precision (autocast + GradScaler). Compare training speed with and without.
4. (Independent) Write complete portable training script: detect device, use GPU if available, use mixed precision if on GPU, checkpoint with device portability.

**Notes for future lessons:**
- The student now has a portable training pattern (device detection + GPU + mixed precision + checkpointing) that carries forward to every future project.
- Fashion-MNIST project (next lesson) can assume this pattern is available.
- The "assembly line with four stations" analogy can be extended when new instruments are added (learning rate schedulers, gradient clipping, etc.).
- Mixed precision is at INTRODUCED depth -- the student can use the pattern but should not be expected to debug AMP issues or handle edge cases.
- The "not magic -- automation" pattern has now been used twice (autograd, AMP). Future automation patterns can reference this thread.

**Deviations from plan:**
- Plan suggested showing "robust pattern: always save with model.cpu().state_dict() or use map_location on load." Built lesson only shows map_location approach -- simpler and more common. Reasonable omission noted in review.
- pin_memory and num_workers were planned as MENTIONED items but were omitted entirely. The lesson was already long (1145 lines) and these are optimizations that would distract from the core patterns. Can be mentioned in Fashion-MNIST project if relevant.

### Lesson 3: Fashion-MNIST Project (`fashion-mnist-project`)

**Type:** CONSOLIDATE / PROJECT | **New concepts:** 0 (zero genuinely new concepts) | **Review:** iteration 2/3, deferred notebook as separate task

**Concepts taught (all reinforcement/application of prior concepts):**
- Fashion-MNIST dataset: loading, 10 clothing classes, visual confusability (INTRODUCED)
- Per-class accuracy analysis: compute accuracy per class, identify easy vs hard classes (INTRODUCED)
- Independent model design: student makes architecture and regularization decisions without scaffolding (APPLIED)
- Complete training pipeline integrating all Series 2 skills (APPLIED)
- Baseline-then-improve experimental methodology (INTRODUCED)
- Mixed precision as optional pipeline addition (APPLIED, reinforces gpu-training INTRODUCED)

**Mental models reinforced:**
- "Same heartbeat, new instruments" -- same training loop, different dataset. The pipeline transfers from MNIST to Fashion-MNIST with minimal code changes.
- "The scissors pattern" -- student sees overfitting in their own training (baseline: 94% train / 88% test), applies regularization to close the gap (91% train / 89% test).
- "Debugging is a systematic workflow" -- debugging checklist used proactively in Check 1 to diagnose the baseline before experimenting.

**Analogies used:**
- "Same shape, different challenge" -- Fashion-MNIST looks like MNIST on the API surface (28x28, 10 classes) but classification difficulty is significantly higher.
- "The 5% gap" -- FC model tops out at ~89-90%, CNNs reach ~93-95%. The gap motivates Series 3. Framed as expansion, not correction.

**How concepts were taught:**
- Hook establishes tension with MNIST vs Fashion-MNIST class comparison (GradientCards). The 10-point accuracy gap (97% vs 87%) is the central motivation.
- Baseline model (MNIST architecture) run first, results diagnosed using debugging checklist (Check 1) before any experimentation.
- Three experiments with increasing independence: (1) train longer (one-line change), (2) add regularization (apply known toolkit), (3) architecture decisions (student chooses).
- Negative example: large unregularized model overfits worse, disproves "more neurons = better."
- Per-class accuracy reveals model struggles with visually similar classes (shirt/coat/pullover), grounding the abstract accuracy number in visual intuition.
- Complete pipeline section explicitly names which prior lesson each component comes from (nine lessons, one pipeline).
- Two checks: (1) diagnostic (debugging checklist on baseline), (2) transfer (50-class, 100x100 dataset -- same workflow applies).
- Module Complete and Series Complete celebration blocks mark the graduation.

**What is NOT covered:**
- Convolutional networks (explicitly deferred to Series 3 with the "5% gap" framing)
- Data augmentation beyond ToTensor + Normalize
- Learning rate schedulers or gradient clipping
- Hyperparameter search strategies (grid search, Bayesian optimization)
- Confusion matrix visualization (mentioned as stretch goal only)

**Exercises (Colab notebook `2-3-3-fashion-mnist-project.ipynb` -- NOT YET CREATED):**
1. (Provided) Data loading -- Fashion-MNIST with visualization
2. (Provided) Baseline model -- run and observe ~87-88% accuracy
3. (Lightly scaffolded) Experimentation -- train longer, add regularization, try architectures. Hints in collapsible cells.
4. (Partially guided) Per-class analysis -- computation provided, student interprets results
5. (Independent) Full pipeline -- GPU, checkpointing, early stopping, best architecture
- Stretch goals: confusion matrix, TensorBoard multi-run comparison, best FC model

**Notes for future lessons:**
- Per-class accuracy analysis is at INTRODUCED depth. Series 3 lessons can use it without re-teaching but should not assume the student is fluent with confusion matrices.
- The "5% gap" framing (FC ~89% vs CNN ~93-95%) is explicitly set up to motivate Series 3. The first CNN lesson should callback to this gap.
- The complete training pipeline (device detection + data loading + regularized model + checkpointing + early stopping + evaluation) is the template for every future project. Series 3 changes the model architecture; everything else stays the same.
- Fashion-MNIST is available as a benchmark for comparing FC vs CNN performance in Series 3.
- The baseline-then-improve methodology can be referenced in any future project lesson.

**Deviations from plan:**
- Colab notebook not yet created (deferred as separate work item)
- No embedded sample images on lesson page (uses emoji grid + matplotlib code for notebook instead)
- ComparisonRow with metrics instead of Recharts training curves (student generates their own in notebook)
- Confusion matrix is stretch goal mention only, not shown on lesson page

## Key Mental Models and Analogies

| Model/Analogy | Established In | Available For |
|---------------|---------------|---------------|
| "state_dict is a snapshot of all the knobs" | saving-and-loading | Any future discussion of model persistence, transfer learning, fine-tuning |
| "Architecture in code, values in file" | saving-and-loading | Model hubs, HuggingFace, transfer learning, model sharing |
| "Giving an experienced pilot amnesia" (resume without optimizer state) | saving-and-loading | Learning rate schedulers, warm restarts, any training resumption context |
| "Same heartbeat, new instruments" (reinforced for GPU) | gpu-training | Any future training loop extension (schedulers, gradient clipping, callbacks) |
| "Assembly line with four stations, faster workers" (GPU training loop) | gpu-training | Future loop extensions; the stations metaphor can absorb new instruments |
| "Micrometer -> ruler -> tape measure" (float64 -> float32 -> float16) | gpu-training | Future precision discussions; bfloat16, quantization, model compression |
| "Not magic -- automation" (reinforced for AMP) | gpu-training | Any future automation pattern (torch.compile, schedulers, etc.) |
| "Same heartbeat, new instruments" (reinforced for new dataset) | fashion-mnist-project | The training loop transfers unchanged between datasets; only the model and data change |
| "The scissors pattern" (applied independently) | fashion-mnist-project | Student diagnoses and closes overfitting gap on their own training run for the first time |
| "The 5% gap" (FC ~89% vs CNN ~93-95%) | fashion-mnist-project | Motivates Series 3; framed as expansion, not correction of the FC approach |
| "Baseline-then-improve" experimental methodology | fashion-mnist-project | Start simple, observe, diagnose, improve. The workflow for every future ML project |
