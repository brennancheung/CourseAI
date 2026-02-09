# Module 3.3: Seeing What CNNs See — Plan

## Module Goal

The student can visualize and interpret what a trained CNN has learned — from raw filter weights to activation maps to class-specific Grad-CAM heatmaps — and use these tools to debug, explain, and validate model behavior on a real fine-tuning project.

## Narrative Arc

**Beginning (the problem):** Throughout Series 3, the student has been told that early layers detect edges, middle layers detect textures and parts, and later layers detect high-level object features. They built this mental model from architecture diagrams, parameter counts, and the "zoom out" analogy. But they have never actually *seen* it. The feature hierarchy is still an article of faith, not observed evidence. This module makes it concrete: you will literally see what each layer learned, and use that visibility to make better decisions about your models.

**Middle (building the tools):** The first lesson opens up the black box with three progressively more useful visualization techniques. Raw filter visualization shows what the network's first layer learned (spoiler: edge and color detectors, exactly as promised). Activation maps show what a specific layer responds to when given a particular image. Grad-CAM provides class-specific spatial heatmaps that answer the most practical question: "why did the model predict this class?" Each technique answers a different question and operates at a different level of abstraction.

**End (the payoff):** The second lesson is a capstone project where the student fine-tunes a pretrained model on their own small dataset and uses the visualization tools from Lesson 1 to understand what the model is doing. This combines transfer learning skills from Module 3.2 with interpretability tools from Lesson 1 into a realistic practitioner workflow: train, visualize, iterate.

## Lesson Sequence

| # | Slug | Core Concepts | Type | Rationale for Position |
|---|------|--------------|------|----------------------|
| 1 | visualizing-features | Filter visualization, activation maps, Grad-CAM | STRETCH | Must come first: establishes the visualization toolkit needed for the project; makes the feature hierarchy concrete for the first time |
| 2 | transfer-learning-project | Fine-tuning workflow with visualization-guided debugging | CONSOLIDATE | Applies transfer learning (3.2) + visualization (L1) in a realistic end-to-end project; low new-concept count, high integration |

## Rough Topic Allocation

| Lesson | Topics | Notes |
|--------|--------|-------|
| visualizing-features | Raw filter weights (conv1), activation maps at different layers, Grad-CAM (gradient-weighted class activation mapping), interpreting what each technique shows | Notebook: load pretrained ResNet, visualize filters, run images through and capture activations, compute Grad-CAM heatmaps. Web lesson provides the conceptual framework. |
| transfer-learning-project | Dataset preparation, feature extraction vs fine-tuning decision, training with validation, using Grad-CAM to validate model reasoning, common failure modes (shortcut learning, dataset bias) | Notebook: end-to-end project. Student collects or uses a small dataset, fine-tunes a pretrained model, uses Grad-CAM to check if the model looks at the right regions. Capstone for Series 3. |

## Cognitive Load Trajectory

```
Lesson 1 (visualizing-features):       STRETCH      — Three new visualization techniques, conceptual framework for interpretability
Lesson 2 (transfer-learning-project):  CONSOLIDATE  — No new concepts; integrates transfer learning + visualization in a project
```

STRETCH followed by CONSOLIDATE is a strong pattern. The student gets one dense conceptual lesson, then immediately applies everything in a guided project that reinforces understanding through practice.

## Module-Level Misconceptions

- **"CNNs are black boxes that cannot be interpreted"** — Multiple visualization techniques exist at different levels of abstraction, from raw filters to class-specific heatmaps. CNNs are among the most interpretable deep learning architectures.
- **"Grad-CAM shows what the network 'sees'"** — Grad-CAM shows which spatial regions are important for a *specific class prediction*, not what features the network detected. The distinction between "what activates" and "what matters for this class" is important.
- **"If the model gets the right answer, it must be looking at the right thing"** — Shortcut learning: models can achieve high accuracy by exploiting spurious correlations (watermarks, backgrounds, metadata). Visualization reveals whether the model's reasoning is valid, not just whether the answer is correct.
- **"First-layer filters should look like the objects the network classifies"** — First-layer filters are low-level edge and color detectors. Object-level representations emerge only in later layers through composition. The hierarchy is real but takes many layers to build.
- **"Activation maps and Grad-CAM are the same thing"** — Activation maps show what a layer responds to (feature detection). Grad-CAM weights those activations by their importance for a specific class (decision explanation). Same input data, very different questions.
