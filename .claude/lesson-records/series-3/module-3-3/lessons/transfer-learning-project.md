# Lesson Plan: Project: Transfer Learning

**Module:** 3.3 — Seeing What CNNs See
**Position:** Lesson 2 of 2 (FINAL lesson in Module 3.3, FINAL lesson in Series 3)
**Slug:** `transfer-learning-project`
**Notebook:** Yes (`3-3-2-transfer-learning-project.ipynb`)
**Cognitive load type:** CONSOLIDATE

---

## Phase 1: Orient — Student State

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Transfer learning (reusing pretrained weights for new tasks) | DEVELOPED | transfer-learning (3.2.3) | Core concept taught via "hire experienced, train specific" analogy. Two strategies: feature extraction and fine-tuning. Student loaded ResNet-18, froze backbone, replaced head. |
| Feature extraction (freeze backbone, replace head) | DEVELOPED | transfer-learning (3.2.3) | Three-step pattern: load pretrained, requires_grad=False on all params, replace model.fc. Student has done this in a CIFAR-10 notebook. |
| Fine-tuning (selectively unfreeze layers with differential LR) | INTRODUCED | transfer-learning (3.2.3) | Unfreeze later stages (layer4) alongside head. Parameter groups in optimizer with different learning rates. Student understands the concept but has not practiced extensively. |
| Differential learning rates (parameter groups in optimizer) | INTRODUCED | transfer-learning (3.2.3) | Pass list of dicts to optimizer. Prevents destroying pretrained features. Student has seen the pattern but not written it independently. |
| Data augmentation transforms | INTRODUCED | transfer-learning (3.2.3) | RandomHorizontalFlip, RandomResizedCrop, ColorJitter. Separate train/val transforms. Connected to overfitting as data-level regularization. |
| Feature transferability spectrum (early universal, later task-specific) | DEVELOPED | transfer-learning (3.2.3) | conv1/layer1 = universal edge/color detectors, layer4 = task-specific, fc = always replaced. Confirmed visually in visualizing-features. |
| Grad-CAM (gradient-weighted class activation mapping) | DEVELOPED | visualizing-features (3.3.1) | Full implementation: forward hook + backward hook, gradient weights via GAP, weighted sum, ReLU. Student implemented in notebook, understands both procedure and interpretation. |
| Filter visualization (conv1 weights as images) | DEVELOPED | visualizing-features (3.3.1) | Accessing model.conv1.weight.data, normalizing, displaying. Connected to "learned filters" from 3.1.1. |
| Activation maps via hooks at multiple depths | DEVELOPED | visualizing-features (3.3.1) | Captured at conv1 (edges), layer2 (textures), layer4 (abstract). Student can register hooks, capture activations, interpret the hierarchy. |
| Shortcut learning (correct prediction from wrong reasoning) | INTRODUCED | visualizing-features (3.3.1) | Husky/wolf example. "Correct prediction does not mean correct reasoning." Framed as the most important practical insight. Student understands the concept but has not diagnosed a real instance themselves. |
| PyTorch hooks (register_forward_hook, register_full_backward_hook) | INTRODUCED | visualizing-features (3.3.1) | "Sensor on a layer" analogy. Four-line pattern. Used for activation capture and Grad-CAM gradient capture. |
| Pretrained models / torchvision.models API | DEVELOPED | transfer-learning (3.2.3) | Load ResNet-18, inspect with print(), access model.fc, model.layer1, etc. ImageNet normalization (mean/std) and 224x224 input. |
| model.train() vs model.eval() | DEVELOPED | resnets (3.2.2), transfer-learning (3.2.3) | Training = per-batch BN stats. Eval = stored running averages. Practical rule: always switch before training/inference. |
| Cross-entropy loss (nn.CrossEntropyLoss) | INTRODUCED | transfer-learning (3.2.3) | Combines log-softmax + NLL. Standard for classification. Training loop unchanged from MSELoss. |
| Transfer learning decision framework (dataset size x domain similarity) | INTRODUCED | transfer-learning (3.2.3) | 2x2 matrix determining feature extraction vs fine-tuning. Practical rule: start simple, add complexity only if needed. |
| Conv-pool-fc architecture pattern | APPLIED | mnist-cnn-project (3.1.3) | Student built a working CNN. Understands the full pipeline. |
| ResNet architecture (basic blocks, skip connections, BN) | DEVELOPED | resnets (3.2.2) | Residual connection formulation, identity vs projection shortcuts, Conv-BN-ReLU pattern. |

### Established Mental Models and Analogies

- "Hire experienced, train specific" (transfer learning = reuse pretrained backbone) (3.2.3)
- Feature extraction vs fine-tuning as a spectrum (3.2.3)
- "Start with the simplest strategy, add complexity only if needed" (3.2.3)
- "Three questions, three tools" (filter viz / activation maps / Grad-CAM) (3.3.1)
- "Visualization is a debugging tool, not just a pretty picture" (3.3.1)
- "Correct prediction does not mean correct reasoning" (3.3.1)
- "The feature hierarchy is real — you saw it" (3.3.1)
- "A filter is a pattern detector" (3.1.1)
- "Architecture encodes assumptions about data" (3.1.3)
- "A residual block starts from identity and learns to deviate" (3.2.2)

### What Was Explicitly NOT Covered

- Fine-tuning from scratch on a novel (non-CIFAR) dataset with domain-specific considerations
- Using visualization tools during or after training to validate model behavior (practiced visualization only on frozen pretrained models, not models the student fine-tuned)
- The full practitioner workflow: train -> visualize -> diagnose -> iterate
- Learning rate scheduling (mentioned as a practical tip in 3.2.3, not developed)
- When to stop training (early stopping, validation plateaus)
- Dealing with class imbalance in small datasets
- Model selection (choosing which pretrained model to use)

### Readiness Assessment

The student is fully prepared. Every concept this project requires has been taught at DEVELOPED or APPLIED depth. The student has:
- Done transfer learning on CIFAR-10 (feature extraction + fine-tuning) in the 3.2.3 notebook
- Implemented Grad-CAM from scratch in the 3.3.1 notebook
- All the PyTorch skills needed (training loops, data loading, transforms, model manipulation)

The project asks the student to combine these into a coherent workflow on a new dataset. No new concepts are required. The key integration challenge is using visualization to *inform decisions* about training, which is a synthesis skill, not a new concept.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to execute the complete transfer learning practitioner workflow — fine-tune a pretrained model on a small custom dataset and use Grad-CAM to verify the model learned meaningful features rather than shortcuts.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Transfer learning (feature extraction + fine-tuning) | APPLIED | DEVELOPED | transfer-learning (3.2.3) | OK | Student needs to execute the workflow independently; DEVELOPED is sufficient because the project provides scaffolding. Depth upgrades to APPLIED through this project. |
| Grad-CAM implementation and interpretation | APPLIED | DEVELOPED | visualizing-features (3.3.1) | OK | Student needs to run Grad-CAM on their own fine-tuned model. DEVELOPED from 3.3.1 notebook is sufficient; this project upgrades to APPLIED by using it in a new context (their own model, not a frozen pretrained model). |
| Data augmentation transforms | INTRODUCED | INTRODUCED | transfer-learning (3.2.3) | OK | Student will apply existing transforms to new data. INTRODUCED is sufficient for this usage — the project is not about inventing new augmentation strategies. |
| Pretrained models / torchvision API | DEVELOPED | DEVELOPED | transfer-learning (3.2.3) | OK | Same API, new dataset. Student already knows how to load, inspect, and modify. |
| Differential learning rates | INTRODUCED | INTRODUCED | transfer-learning (3.2.3) | OK | Student will use parameter groups for fine-tuning. The notebook provides the pattern; student adapts it. INTRODUCED is sufficient with scaffolding. |
| Feature transferability spectrum | DEVELOPED | DEVELOPED | transfer-learning (3.2.3), visualizing-features (3.3.1) | OK | Student uses this understanding to make decisions (which layers to freeze/unfreeze). Confirmed visually in 3.3.1. |
| Shortcut learning detection | INTRODUCED | INTRODUCED | visualizing-features (3.3.1) | OK | Student needs to recognize shortcut patterns in their own Grad-CAM heatmaps. INTRODUCED is sufficient — the concept is clear, this project provides practice. |
| Training loop (DataLoader, optimizer, loss, epochs) | APPLIED | APPLIED | Series 2, transfer-learning (3.2.3) | OK | Student has written multiple training loops. Provided in notebook but student should understand every line. |
| model.train() / model.eval() switching | DEVELOPED | DEVELOPED | resnets (3.2.2) | OK | Critical for correct BN behavior during training and Grad-CAM inference. Student knows the pattern. |
| Cross-entropy loss | INTRODUCED | INTRODUCED | transfer-learning (3.2.3) | OK | Standard classification loss. Student will use it in the training loop. |

All prerequisites are at sufficient depth. No gaps or missing concepts.

### Gap Resolution

No gaps to resolve. All prerequisites are met. This is a CONSOLIDATE lesson that integrates previously taught concepts.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "If validation accuracy is high, the model is good enough — no need to visualize" | High accuracy is the standard metric of success in ML courses. Accuracy is the number the student has been optimizing throughout Series 1-3. It feels like the final answer. | The shortcut learning scenario: model achieves 90%+ accuracy on a biased dataset by learning background correlations (e.g., all "outdoor" photos have grass, all "indoor" photos have walls). Grad-CAM reveals the model focuses on background textures, not the objects. Deploy this model on new data where backgrounds differ and accuracy collapses. | Project brief section — framed as the central question: "Is the model right for the right reasons?" Reinforced in the Grad-CAM validation step of the notebook. |
| "Fine-tuning always beats feature extraction" | More trainable parameters = more flexibility = better model. Fine-tuning feels like the "full" version while feature extraction feels like a shortcut. | On a very small dataset (50-200 images per class), fine-tuning with standard LR destroys pretrained features and performs worse than feature extraction. The student may see this in their own notebook results. With very few examples, the frozen features from ImageNet generalize better than features adapted to a tiny, potentially unrepresentative sample. | The "start simple" guidance in the project brief. The notebook structures the comparison: feature extraction first, fine-tuning second. Student sees results for both and compares. |
| "Grad-CAM on a correctly classified image always highlights the object" | The 3.3.1 lesson showed Grad-CAM highlighting dogs and benches — the model looked at the right thing. Easy to generalize: correct prediction = correct focus. | On a fine-tuned model trained on a small dataset, Grad-CAM may reveal the model focuses on background context, image borders, watermarks, or other artifacts even when predictions are correct. The student's own model may exhibit this. | The Grad-CAM validation step in the notebook. The project brief explicitly asks the student to look for *unexpected* focus patterns, not just confirm expected ones. |
| "The dataset is too small for this to work" | Student has seen ImageNet (1.2M images) and CIFAR-10 (50K images). A dataset of 200-500 images feels hopelessly small. Transfer learning is associated with CIFAR-10 in the student's experience. | Feature extraction with a frozen ImageNet backbone can achieve 85-95% accuracy on 200 images per class for visually distinct categories. The entire point of transfer learning is that the pretrained features do the heavy lifting. The student will see this in their own results. | Hook section — frame the small dataset as the challenge, then show that transfer learning makes it viable. |
| "I need to write all the code from scratch for a real project" | The student has been building toward writing their own code throughout the course. A project feels like it should be fully independent. | Professional ML practitioners use scaffolded pipelines (Lightning, HuggingFace), pretrained models, and standard patterns. The notebook provides structure and utilities. The skill being practiced is decision-making and interpretation, not boilerplate writing. | Project brief — explicitly frame what the student owns (decisions, interpretation, debugging) vs what is provided (data loading, display utilities, training loop skeleton). |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| End-to-end project: feature extraction on a small dataset (200-500 images, 3-5 classes) with Grad-CAM validation | Positive | The primary example — the student executes the full workflow: load pretrained model, prepare data, train via feature extraction, evaluate, validate with Grad-CAM | This is the entire point of the project. Uses a dataset different from CIFAR-10 to demonstrate transfer to a genuinely new domain. Small enough to train in minutes on Colab. Classes should be visually distinct so Grad-CAM heatmaps are interpretable. |
| Fine-tuning comparison: unfreeze layer4 with differential LR, compare to feature extraction | Positive | Shows when fine-tuning helps and when it does not, using the student's own results as evidence | Upgrades fine-tuning from INTRODUCED to hands-on experience. The comparison makes the "start simple" heuristic concrete — the student sees whether the added complexity of fine-tuning improves results on their specific data. |
| Grad-CAM revealing unexpected model focus (e.g., model focuses on background or non-object region despite correct prediction) | Negative | Demonstrates that visualization is a debugging tool and that correct accuracy is not sufficient validation | Directly reinforces the "correct prediction does not mean correct reasoning" mental model from 3.3.1. This may emerge naturally from the student's own model, or the notebook can construct a scenario where it is visible. The student diagnoses a real issue, not a textbook example. |

---

## Phase 3: Design

### Narrative Arc

You have all the pieces. Over the last ten lessons, you learned what convolutions compute, how architectures evolved from LeNet to ResNet, how to reuse pretrained models for new tasks, and how to see what your model actually learned with Grad-CAM. Each of those skills was practiced in isolation. This project puts them together for the first time. You will take a small dataset of images the pretrained model has never seen, fine-tune a ResNet to classify them, and then use Grad-CAM to answer the question that matters most in practice: is the model right for the right reasons? High accuracy is not enough. The shortcut-learning lesson showed you that a model can get 90% accuracy by learning the wrong features. Grad-CAM is how you check. This is the workflow that professional practitioners use every day — train, visualize, diagnose, iterate — and by the end of this project, you will have done it yourself.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | End-to-end project with real images, real training, real Grad-CAM heatmaps on the student's own fine-tuned model | This is a CONSOLIDATE project lesson. The modality IS the hands-on experience. Abstract explanations are unnecessary — the student has the concepts; they need practice applying them. |
| Visual | Grad-CAM heatmaps on the student's fine-tuned model, before/after comparisons of feature extraction vs fine-tuning focus patterns | The capstone visualization skill. The student sees whether their own model looks at the right things. This is the payoff for the entire 3.3 module. |
| Verbal/Reflective | Guided reflection prompts in the web lesson: "What did Grad-CAM reveal? Did the model focus on the objects or the background? What would you change?" | Project lessons need reflection to consolidate learning. Without prompted reflection, the student may execute the notebook mechanically without synthesizing the insights. |
| Comparative | Side-by-side: feature extraction accuracy vs fine-tuning accuracy; Grad-CAM heatmaps before fine-tuning (frozen pretrained) vs after fine-tuning (adapted model) | Comparison makes the effect of fine-tuning visible. The student sees not just accuracy differences but *what changed* in the model's focus. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 0. This is a CONSOLIDATE lesson. Every technique the student uses was taught in a prior lesson.
- **Previous lesson load:** STRETCH (visualizing-features — three new visualization techniques, conceptual framework for interpretability)
- **Appropriateness:** CONSOLIDATE after STRETCH is the ideal pattern. The student just absorbed a dense conceptual lesson and now gets to apply everything in a guided project that reinforces understanding through practice. The cognitive load is low in terms of new concepts but high in terms of integration — combining transfer learning, data handling, training, and visualization into a coherent workflow. This is the right kind of challenge for a capstone.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|------------|
| "Hire experienced, train specific" (3.2.3) | The student literally does this: loads a pretrained ResNet, freezes it, trains a new head for their specific classes. The analogy becomes a lived experience. |
| Feature extraction 3-step pattern (3.2.3) | Student repeats the pattern on new data. Reinforces from DEVELOPED toward APPLIED. |
| "Start with the simplest strategy, add complexity only if needed" (3.2.3) | The project structures feature extraction first, fine-tuning second. The student sees whether the complexity was warranted. |
| "Visualization is a debugging tool" (3.3.1) | The student uses Grad-CAM not for curiosity but to validate their own model's reasoning. The mental model becomes a practiced habit. |
| "Correct prediction does not mean correct reasoning" (3.3.1) | The central question of the Grad-CAM validation step. The student checks their own model, not a textbook example. |
| Feature transferability spectrum (3.2.3, 3.3.1) | Guides the student's freezing decisions. Also visible in Grad-CAM: early features (edges) should be similar pre/post fine-tuning; later features should adapt. |
| "Three questions, three tools" (3.3.1) | The student selects the right tool for their question. Grad-CAM answers "what mattered for this prediction?" which is the most relevant question for model validation. |
| Data augmentation as regularization (3.2.3, 1.3) | Applied to a genuinely small dataset where augmentation matters. |

**Analogies that could be misleading:** The "hire experienced" analogy might suggest transfer learning always works seamlessly. On a very small or domain-shifted dataset, the student may see mediocre results even with transfer learning. The project should frame this as a realistic outcome, not a failure. Transfer learning gives a strong starting point, not a guaranteed result.

### Scope Boundaries

**This lesson IS about:**
- Executing the full transfer learning workflow on a new dataset (not CIFAR-10)
- Using Grad-CAM to validate model reasoning (not just accuracy)
- Comparing feature extraction vs fine-tuning on the student's own data
- Reflecting on what visualization reveals about model behavior
- Series 3 completion moment — acknowledging the full journey from "what is a convolution?" to "I can train, visualize, and validate a CNN"

**This lesson is NOT about:**
- Teaching any new concepts (CONSOLIDATE)
- Advanced fine-tuning techniques (LR scheduling, warmup, cosine annealing)
- Hyperparameter search or architecture selection
- Multi-GPU training or deployment
- Object detection, segmentation, or tasks beyond classification
- Building a custom dataset from scratch (dataset is provided or curated from existing source)
- Advanced interpretability methods beyond Grad-CAM
- Fixing or iterating on a broken model (the project ends with diagnosis, not with a fix-iterate loop)

**Target depths (upgrades from this project):**
- Transfer learning workflow: DEVELOPED -> APPLIED
- Grad-CAM as a debugging tool: DEVELOPED -> APPLIED
- Fine-tuning with differential LR: INTRODUCED -> DEVELOPED
- Shortcut learning detection: INTRODUCED -> DEVELOPED

### Lesson Outline

**1. Context + Constraints**
- This is a capstone project: no new concepts, all integration
- What the student owns: decisions (freeze/unfreeze, LR choices), interpretation (Grad-CAM analysis), reflection
- What is provided: dataset, data loading, display utilities, training loop skeleton, Grad-CAM utilities
- The notebook is where the real work happens; this web lesson is the project brief, guidance, and reflection space
- Estimated time: 30-45 minutes on Colab GPU

**2. Hook — "You have all the pieces"**
- Type: Challenge preview
- Recap the journey: convolutions (3.1) -> architectures (3.2) -> transfer learning (3.2.3) -> visualization (3.3.1)
- Frame the project: "For the first time, you will use all of these together on a task where you do not know in advance whether the model will learn the right features"
- The question: not "how accurate is it?" but "is it right for the right reasons?"
- Why this hook: The student has been building toward this. Each piece was practiced in isolation. The challenge is integration, and the uncertainty of working with a new dataset adds genuine engagement.

**3. Project Brief — The Dataset and the Task**
- Describe the dataset: small image classification dataset with 3-5 visually distinct classes, ~200-500 images total. The dataset should be different from anything the student has used before (not CIFAR-10, not MNIST). A good candidate: a small subset of a domain-specific dataset (flowers, food, textures, medical, or similar).
- The task: classify images into the correct category using a fine-tuned pretrained model
- Why this dataset: small enough to train fast, different enough from ImageNet to make transfer interesting, classes that are visually distinct so Grad-CAM heatmaps are interpretable
- Decision framework recap (one paragraph): student applies the 2x2 matrix from 3.2.3 to their specific dataset — small dataset + similar domain = start with feature extraction

**4. Project Guidance — The Workflow**
- Step-by-step workflow, mirroring the notebook structure:
  1. **Load and explore the data** — understand the classes, sample sizes, image characteristics
  2. **Set up the pretrained model** — load ResNet-18, freeze backbone, replace head for N classes
  3. **Feature extraction** — train with frozen backbone. Track training and validation accuracy/loss.
  4. **Evaluate** — check accuracy. Is it good? (Probably yes, but that is not the final answer.)
  5. **Grad-CAM validation** — run Grad-CAM on correctly classified images from each class. Does the model focus on the object or on background/context?
  6. **Fine-tuning (optional extension)** — unfreeze layer4 with a low LR, retrain, compare accuracy AND Grad-CAM focus
  7. **Reflection** — what did visualization reveal? Was the model right for the right reasons?
- Each step has a brief paragraph of guidance (what to look for, what to expect, common pitfalls)
- Emphasis on the Grad-CAM step as the most important — accuracy alone is not validation

**5. What to Look For — Grad-CAM Interpretation Guide**
- Good signs: heatmap highlights the object or discriminative features of the object (e.g., flower petals for flower classification)
- Warning signs: heatmap highlights background, image borders, watermarks, or non-object regions
- Ambiguous cases: heatmap highlights part of the object + some context. This is normal — context can be a legitimate feature (e.g., "beach" for a beach scene classifier)
- Connection to shortcut learning from 3.3.1: "Remember the husky/wolf example. Your model might have its own version of this."
- Key question for each Grad-CAM heatmap: "If I showed this to someone who does not know ML, would they agree that the model is focusing on the right thing?"

**6. Notebook Link + Getting Started**
- Link to Colab notebook
- Brief setup instructions (runtime type, drive mounting if needed)
- Encouragement: "You have done every piece of this before. The notebook will guide you through combining them."

**7. Reflection Prompts (post-notebook)**
- What accuracy did feature extraction achieve? Was it sufficient?
- Did fine-tuning improve accuracy? By how much?
- What did Grad-CAM reveal about what your model learned? Any surprises?
- Was the model right for the right reasons, or did you find evidence of shortcut learning?
- If you had 10x more data, what would you do differently?
- What was the most useful visualization technique for understanding your model?

**8. Series 3 Completion**
- Celebrate the journey: from "what is a convolution?" through architecture evolution, ResNets, transfer learning, and visualization to "I can train and validate a CNN on a new task"
- Skills acquired across Series 3: understanding convolutional operations, tracing dimension through architectures, knowing why depth helps, using pretrained models, interpreting model behavior with visualization
- The practical superpower: not just "can I get high accuracy?" but "can I understand what my model learned and trust its reasoning?"
- What comes next: Series 4 (LLMs) — a different architecture, a different data modality, but the same practitioner mindset: understand the model, do not just use it
- This is a satisfying conclusion to the CNN arc, not a cliffhanger

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
- [x] At least 3 modalities planned for the core concept, each with rationale
- [x] At least 2 positive examples + 1 negative example, each with stated purpose
- [x] At least 3 misconceptions identified with negative examples
- [x] Cognitive load = 0 new concepts (CONSOLIDATE)
- [x] Every concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated
