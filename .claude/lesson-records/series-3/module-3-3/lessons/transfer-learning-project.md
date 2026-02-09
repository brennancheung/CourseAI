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

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings that would leave the student lost, but one critical issue in the notebook and several improvement-level findings that would significantly strengthen both artifacts. The web lesson is well-structured as a lightweight project brief. The notebook is thorough and well-scaffolded. The main issues are: a notebook bug that would break execution, the notebook providing too many answers for a capstone, missing explicit connection to filter visualization/activation maps (only Grad-CAM is used), and the web lesson's fine-tuning phase being presented as mandatory when the plan called it "optional extension."

### Findings

### [CRITICAL] — Notebook variable `in_features` used before assignment in global scope

**Location:** Notebook cell-12, final print statement
**Issue:** The variable `in_features` is assigned inside the `create_feature_extraction_model` function as a local variable (`in_features = model.fc.in_features`). The print statement at the bottom of the cell references `in_features` in the global scope: `print(f'Only training the classification head ({in_features} -> {NUM_CLASSES})')`. This will raise a `NameError` because `in_features` is local to the function, not available in the cell's global scope.
**Student impact:** The student runs the cell and gets a traceback instead of the verification message. Confusing for a capstone where the student expects the scaffolding to work. They would need to debug the notebook itself rather than focus on the project.
**Suggested fix:** Either (a) capture the return value from the function and use it, e.g., change the function to return both model and in_features, or (b) compute `in_features` outside the function by reading it from `fe_model.fc.in_features` after creation: `in_features = fe_model.fc.in_features` before the print statement.

### [IMPROVEMENT] — Notebook TODOs are pre-filled, reducing student decision-making

**Location:** Notebook cells 12, 19, 22, 23, 30
**Issue:** The planning document frames this as a capstone where "the student owns decisions (freeze/unfreeze, LR choices), interpretation (Grad-CAM analysis), reflection." However, the notebook has every TODO section already filled in with the correct code. Cell 12 has the freeze loop and head replacement already written. Cell 22 has layer4 unfreezing already done. Cell 23 has the differential learning rate optimizer already configured. The student's job reduces to pressing Shift+Enter through the cells and interpreting the outputs. The only cells that require any student action are the interpretation markdown cells.
**Student impact:** The student does not practice making the decisions themselves. The depth upgrade from INTRODUCED to DEVELOPED for fine-tuning with differential LR is not fully earned because the student never had to write the parameter group pattern independently. The capstone becomes a guided demo rather than a project.
**Suggested fix:** Leave the TODO sections as actual TODOs with hints (the hints are already there in comments). Remove the filled-in code so the student must write: (1) the freeze loop in cell 12, (2) the head replacement in cell 12, (3) the layer4 unfreezing in cell 22, (4) the parameter groups optimizer in cell 23. The Grad-CAM visualization code in cells 19 and 30 can remain provided since the goal is interpretation, not reimplementation. This is the capstone---the student should demonstrate they can do the mechanical parts independently.

### [IMPROVEMENT] — Web lesson presents fine-tuning as a required phase, not optional extension

**Location:** Web lesson, PhaseCard 4 (Fine-Tuning)
**Issue:** The planning document's outline section (item 6) describes fine-tuning as "optional extension": "Fine-tuning (optional extension) --- unfreeze layer4 with a low LR, retrain, compare accuracy AND Grad-CAM focus." However, the built web lesson presents all 5 phases as a mandatory linear sequence with no indication that Phase 4 is optional. The notebook also presents it as a required step (no "optional" or "bonus" framing). This is a deviation from the plan.
**Student impact:** Minor for this student (they will likely do it anyway), but it changes the character of the project from "make a decision about whether to fine-tune" to "do both approaches." The "start simple, add complexity only if needed" mental model is weakened because the lesson removes the decision---the student fine-tunes regardless of whether feature extraction was sufficient.
**Suggested fix:** Either (a) add language to PhaseCard 4 marking it as an optional extension the student should attempt if feature extraction accuracy leaves room for improvement, or (b) update the planning document to reflect the deliberate decision to make fine-tuning mandatory (which is defensible for a capstone that wants to upgrade differential LR from INTRODUCED to DEVELOPED). Option (b) is probably better---the comparison is valuable pedagogically, and making it optional risks students skipping the most instructive part.

### [IMPROVEMENT] — Only Grad-CAM is used; filter visualization and activation maps are absent

**Location:** Notebook, entire workflow
**Issue:** The module 3.3 module record establishes "Three questions, three tools" as a key mental model. The planning document's connections table references this. However, neither the web lesson nor the notebook asks the student to use filter visualization or activation maps on their fine-tuned model. Only Grad-CAM is used. This is a missed opportunity to reinforce the "three tools" framework and to let the student see whether fine-tuning changed the conv1 filters (it should not have---they were frozen).
**Student impact:** The student exits the capstone having only applied one of three visualization tools. The "three questions, three tools" mental model is not reinforced through practice. The student does not get the satisfying confirmation that conv1 filters remain identical after fine-tuning (because they were frozen), which would be a concrete payoff for the transferability spectrum concept.
**Suggested fix:** Add one optional section to the notebook (between Phase 3 and Phase 4, or at the end) that has the student: (1) display conv1 filters from the feature extraction model (should match ImageNet pretrained exactly), (2) optionally compare layer4 activations before/after fine-tuning. This does not need to be long---3-4 cells. The web lesson could add one sentence in the "Reflect on Your Results" section: "Which of the three visualization tools was most useful for understanding your model?" (This prompt actually already exists as reflection prompt #6, which is good, but it is orphaned without the notebook actually using all three tools.)

### [IMPROVEMENT] — Reflection prompt #6 references techniques the student did not use in this project

**Location:** Web lesson, Reflection Prompts section, item 6
**Issue:** Prompt 6 asks: "Which visualization technique (filter viz, activation maps, Grad-CAM) was most useful for understanding your model's behavior?" However, the notebook only uses Grad-CAM. The student did not use filter visualization or activation maps on their project model. This prompt asks the student to compare tools they did not actually use in this context.
**Student impact:** Confusion or a vague answer. The student might reference their 3.3.1 experience rather than this project, which defeats the purpose of project-specific reflection. Alternatively, the student may feel they missed something.
**Suggested fix:** Either (a) update the notebook to include filter viz and activation maps as described in the previous finding, making the prompt valid, or (b) rewrite prompt 6 to ask something the student actually did in this project, e.g., "How did Grad-CAM change your understanding of your model's accuracy? Was there a moment where the heatmap surprised you?" or "Would filter visualization or activation maps have added useful information here? Why or why not?"

### [POLISH] — Web lesson Hook section says "ten lessons" but actual count may differ

**Location:** Web lesson, Hook section paragraph 1: "Over the last ten lessons"
**Issue:** The exact count of lessons in Series 3 prior to this one depends on the curriculum. If the count is not exactly ten, this creates a minor credibility issue. The planning document does not specify the exact count.
**Student impact:** Trivial---the student is unlikely to count, and "ten" is close enough to be fine. But if the student does count and gets a different number, it introduces a tiny moment of doubt.
**Suggested fix:** Either verify the exact count and use it, or replace with "Throughout this series" or "Over the last several lessons" to avoid committing to a specific number.

### [POLISH] — Notebook cell-22 has same `in_features` scoping issue as cell-12

**Location:** Notebook cell-22, `create_finetuning_model` function
**Issue:** Same pattern as the CRITICAL finding in cell-12: `in_features` is assigned inside the function but the cell does not reference it outside (the print statement in cell 22 uses `ft_model.layer4.parameters()` and `ft_model.fc.parameters()` directly). So this cell actually works fine---no bug here. However, the variable name reuse could be confusing if the student is debugging cell 12. No action needed on this specific cell.
**Student impact:** None (this cell works correctly).
**Suggested fix:** No change needed for cell 22 specifically; fixing cell 12 is sufficient.

### [POLISH] — Notebook class names are approximate and could confuse a student who Googles them

**Location:** Notebook cell-5, CLASS_NAMES list
**Issue:** The comment says "Label names are approximate (the dataset doesn't include official names in torchvision, but these correspond to visually distinct flower types)." The chosen names (e.g., "Pink Primrose" for class 1, "Globe Thistle" for class 10) may not exactly match the Flowers102 label mapping from the original paper. If a student looks up the official label file, they might find different names and wonder if the wrong classes were selected.
**Student impact:** Minor confusion if the student cross-references. Does not affect the learning objectives at all.
**Suggested fix:** Add a brief note in the markdown cell above or in the code comment: "These names are chosen for display purposes and may not match the original dataset's label file exactly. The important thing is visual distinctness."

### Review Notes

**What works well:**
- The web lesson is genuinely lightweight and appropriate for a CONSOLIDATE capstone. It does not over-explain concepts the student already knows. The "project brief + guidance + reflection" framing is exactly right.
- The narrative arc is compelling. The "You Have All the Pieces" hook works because it is true---the student really does have all the pieces. The journey recap is well-paced and motivating.
- The Grad-CAM interpretation guide (Good Signs / Warning Signs / Ambiguous Cases) is excellent pedagogical scaffolding. It gives the student a framework for interpreting their own results without telling them what to expect.
- The notebook structure mirrors the practitioner workflow cleanly. The five phases are well-sequenced.
- The Series 3 completion celebration is satisfying and well-earned. The framing of "the practical superpower" (understanding, not just accuracy) ties back to the course's core value proposition.
- Misconceptions from the plan are addressed at the right locations: "accuracy is not enough" is the central theme, "fine-tuning always beats feature extraction" is addressed by the comparison structure, and "the dataset is too small" is addressed by the hook.
- The scope boundaries are respected---the lesson does not drift into teaching new concepts.

**Patterns to note:**
- The notebook being pre-filled is a common builder tendency ("make sure it works") but is at odds with the capstone framing. The tension between "scaffolded" and "the student should do it" is the main quality issue.
- The decision to focus exclusively on Grad-CAM (not filter viz / activation maps) is defensible for time reasons but means reflection prompt #6 is orphaned. Either expand the notebook or fix the prompt---do not leave them misaligned.
- The CRITICAL finding (NameError in cell 12) is a genuine runtime bug that must be fixed before the student sees the notebook.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 1
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

All iteration 1 findings were addressed. The web lesson is clean and effective. The notebook now has genuine TODOs for the capstone skills (freeze, head replacement, layer4 unfreezing, differential LR optimizer). However, a new CRITICAL runtime bug was introduced in the optional filter viz section (cell 22), and one improvement-level finding remains about the fine-tuning phase framing.

### Findings

### [CRITICAL] — Notebook cell 22 references `comparison_images` before it is defined

**Location:** Notebook cell 22, line: `sample_img = list(comparison_images.values())[0] if comparison_images else list(correct_samples.values())[0][0][0]`
**Issue:** The variable `comparison_images` is first defined in cell 32 (the Phase 5 side-by-side Grad-CAM comparison cell). Cell 22 is the optional filter viz / activation maps cell that comes after Phase 3 (Grad-CAM validation). If the student runs cells top-to-bottom (the expected and natural order), cell 22 executes before cell 32, and `comparison_images` does not exist in the namespace. Python will raise a `NameError: name 'comparison_images' is not defined`. The `if comparison_images` guard only protects against an empty dict, not a nonexistent variable.
**Student impact:** The student runs the optional cell and gets a traceback. This is particularly frustrating because the cell is labeled as "optional but reinforcing" --- a runtime error in an optional section signals to the student that the notebook is broken and undermines trust in the scaffolding. The fix from iteration 1 (adding this section) introduced the bug.
**Suggested fix:** Remove the `comparison_images` reference entirely. Use only `correct_samples` which is guaranteed to exist at this point (defined in cell 19, which runs before cell 22): `sample_img = list(correct_samples.values())[0][0][0]`. Alternatively, collect a sample image directly from the test dataset within the cell itself.

### [IMPROVEMENT] — Fine-tuning phase is labeled "Optional Extension" in both web and notebook, but notebook execution requires it

**Location:** Web lesson PhaseCard 4 ("Optional Extension"), Notebook cell 23 markdown ("Optional Extension"), Notebook cells 24-27 (fine-tuning implementation and training)
**Issue:** The web lesson correctly labels Phase 4 as "Optional Extension" per the iteration 1 fix. The notebook markdown in cell 23 also says "Optional Extension." However, the notebook's cell 30 (accuracy comparison table) unconditionally references `ft_history`, and cell 32 (side-by-side Grad-CAM) unconditionally uses `ft_model`. If the student skips the fine-tuning cells (24-27) as the "optional" framing invites, cells 30-32 will crash with `NameError`. The notebook is structurally mandatory while being labeled as optional.
**Student impact:** Two failure modes: (1) The student skips the "optional" section and hits errors in Phase 5, or (2) the student does the section because the notebook forces it, but the "optional" framing creates a confusing mixed signal. Neither is ideal.
**Suggested fix:** Either (a) make the notebook truly optional by wrapping cells 30-32 in `try/except` or gating on `'ft_history' in dir()`, with a message like "Fine-tuning was skipped --- showing feature extraction results only," or (b) remove the "Optional Extension" framing from the notebook cell 23 markdown (keeping it only in the web lesson, where it serves as guidance rather than instruction). Option (b) is simpler and preserves the pedagogically valuable comparison. The web lesson can frame it as "optional if you want to stop after feature extraction" while the notebook guides the student through the full comparison.

### [POLISH] — Extra space before `what` in the "Real Skill" aside

**Location:** Web lesson, InsightBlock "The Real Skill" (line ~655): `seeing<em> what</em> it learned`
**Issue:** There is a leading space inside the `<em>` tag: `<em> what</em>` renders as " what" with a space before "what." The intended rendering is "seeing *what* it learned" but the actual rendering will have a double space: "seeing  what it learned."
**Student impact:** Trivial visual issue. The student may not even notice. It is a minor typographic imperfection.
**Suggested fix:** Change `<em> what</em>` to `<em>what</em>` (remove the leading space inside the tag).

### Review Notes

**What was fixed well from iteration 1:**
- The `in_features` NameError in cell 12 is fixed cleanly --- `fc_in = fe_model.fc.in_features` reads the value from the model after creation, sidestepping the scoping issue entirely.
- TODOs are now genuine scaffolding with hints and assertions. Cells 12, 24, and 25 require the student to write freeze loops, head replacement, layer4 unfreezing, and differential LR parameter groups. This is a real capstone now.
- The optional filter viz section (cells 21-22) is a well-framed addition. The "three questions, three tools" callback in the markdown is pedagogically sound. The conv1 filter visualization is a satisfying concrete confirmation of "frozen means frozen."
- Reflection prompt 6 is well-updated with conditional phrasing that works whether or not the student ran the optional section.
- "Throughout this series" replaces the specific count. Class names note added to cell 5 comments.
- Fine-tuning PhaseCard correctly labeled "Optional Extension."

**Patterns to note:**
- The CRITICAL finding is a straightforward variable ordering bug --- the kind of issue that emerges when new cells reference variables from cells that were originally in a different position. Easy to fix.
- The "optional but required" tension in the fine-tuning phase is a design decision more than a bug. The recommendation is to simplify: keep the web lesson's "optional" framing for the student who wants to stop early, but let the notebook guide the full comparison naturally. The comparison is the most instructive part of the capstone.

---

## Review — 2026-02-09 (Iteration 3/3 FINAL)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

All iteration 2 findings have been resolved. The lesson is ready to ship.

**Iteration 2 fixes verified:**
1. **CRITICAL (comparison_images NameError in cell 22):** Fixed. Cell 22 now uses `correct_samples` (defined in cell 19) to source its sample image: `sample_img = list(correct_samples.values())[0][0][0]`. No reference to `comparison_images` remains in cell 22. Variable ordering is correct for top-to-bottom execution.
2. **IMPROVEMENT (fine-tuning optional/required tension):** Fixed. Notebook cell 23 markdown presents Phase 4 as "Fine-Tuning" without "Optional Extension" labeling, guiding the student through the full comparison naturally. Web lesson PhaseCard 4 retains "Optional Extension" subtitle as guidance for students who want to stop after feature extraction. Cells 30-32 work correctly because the notebook flow makes Phase 4 a natural continuation.
3. **POLISH (extra space in `<em>` tag):** Fixed. Line 655 reads `<em>what</em>` with no leading space.

### Findings

None. No critical, improvement, or polish findings on this iteration.

### Review Notes

**Web lesson quality:**
- Lightweight and appropriate for a CONSOLIDATE capstone. Does not over-explain concepts the student already knows.
- The "You Have All the Pieces" hook is compelling and true. The narrative arc works.
- The Grad-CAM Interpretation Guide (Good Signs / Warning Signs / Ambiguous Cases) is excellent scaffolding.
- The Series 3 completion celebration is satisfying. The "practical superpower" framing ties the entire series together.
- All em dashes use `&mdash;` with no spaces. Writing style is clean.
- TypeScript compiles cleanly. No lint errors. Route and module index are correctly wired.

**Notebook quality:**
- Genuine TODOs in cells 12, 24, and 25 require the student to write freeze loops, head replacement, layer4 unfreezing, and differential LR parameter groups. This is a real capstone.
- Variable ordering is correct for top-to-bottom execution. No cross-cell dependency bugs.
- Optional filter viz section (cells 21-22) is well-framed with "three questions, three tools" callback.
- Training utilities, Grad-CAM utility, and display functions are provided as scaffolding.
- Assertions provide immediate feedback if TODO implementations are incorrect.
- The notebook summary and Series 3 completion message mirror the web lesson.

**What the lesson does well:**
- The practitioner workflow (train, evaluate, visualize, diagnose) is the central organizing principle and is executed faithfully across both artifacts.
- The "is the model right for the right reasons?" question drives genuine engagement.
- The comparison structure (FE first, FT second, side-by-side Grad-CAM) makes the value of visualization concrete.
- Reflection prompts close the loop without being prescriptive.
- Scope boundaries are strictly respected --- zero new concepts taught, which is correct for a CONSOLIDATE capstone.

**Depth upgrades earned:**
- Transfer learning workflow: DEVELOPED -> APPLIED (student executes full workflow on new dataset)
- Grad-CAM as debugging tool: DEVELOPED -> APPLIED (student validates their own fine-tuned model)
- Fine-tuning with differential LR: INTRODUCED -> DEVELOPED (student writes parameter groups independently)
- Shortcut learning detection: INTRODUCED -> DEVELOPED (student checks their own model for shortcuts)
