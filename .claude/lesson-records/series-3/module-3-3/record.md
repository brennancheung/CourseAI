# Module 3.3: Seeing What CNNs See — Record

**Goal:** The student can visualize and interpret what a trained CNN has learned — from raw filter weights to activation maps to class-specific Grad-CAM heatmaps — and use these tools to debug, explain, and validate model behavior on a real fine-tuning project.
**Status:** In progress (1 of 2 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Filter visualization (displaying conv1 learned weights as images) | DEVELOPED | visualizing-features | 64 filters of shape [3, 7, 7] normalized and displayed as 8x8 grid of tiny RGB images. Inline SVG schematic shows 8 typical filter patterns (vertical/horizontal/diagonal edges, color gradients, stripes, blobs). Code shows accessing model.conv1.weight.data. Connected to "learned filters" from 3.1.1 and "early layers are universal" from 3.2.3. Limitation explicitly stated: only works for conv1 because deeper layers have multi-channel inputs (e.g., [256, 128, 3, 3]) that cannot be displayed as meaningful images. |
| Activation map capture and interpretation at multiple depths | DEVELOPED | visualizing-features | Captured at conv1 (112x112, sharp edges), layer2 (28x28, textures), layer4 (7x7, abstract distributed patterns) via hooks. Inline SVG schematic shows progression from sharp edge detection to abstract blobs. PhaseCards describe each depth. Explicitly confirms "edges -> textures -> parts -> objects" hierarchy from 3.1.2 and 3.2.1. Negative example: layer4 activations are NOT recognizable objects — representation is distributed across 512 channels, no single channel encodes "dog" or "car." |
| Grad-CAM (gradient-weighted class activation mapping) | DEVELOPED | visualizing-features | Taught via 6-step procedure (concrete), then worked example with 3 channels at 2x2 (concrete with numbers), then formula (symbolic). alpha_k = (1/Z) sum(dY^c/dA^k_ij), L = ReLU(sum(alpha_k * A^k)). Inline SVG shows same image producing different heatmaps for "golden retriever" vs "park bench." Code block implements full Grad-CAM with forward hook + backward hook. Connected to backprop (same mechanism, different purpose) and GAP from ResNets (same operation on gradients instead of activations). |
| PyTorch hooks (register_forward_hook, register_full_backward_hook) | INTRODUCED | visualizing-features | Four-line pattern: define hook function, register on layer, run forward pass, remove hook. "Sensor on a layer" analogy — records what passes through without changing the model. register_full_backward_hook noted as modern API replacing deprecated register_backward_hook. Not explored beyond activation/gradient capture. |
| Shortcut learning (correct prediction from wrong reasoning) | INTRODUCED | visualizing-features | Husky/wolf example: model classifies based on background (snow vs forest), not the animal. 90% accuracy is real but reasoning is broken. Framed as the most practically important insight: "correct prediction does not mean correct reasoning." Motivates visualization as a debugging tool, not just a pretty picture. |
| Grad-CAM limitations (resolution, positive-only, not causal) | INTRODUCED | visualizing-features | Resolution limited by last conv layer spatial size (7x7 for ResNet-18 = each cell covers 32x32 input patch). ReLU keeps only positive evidence (regions supporting prediction, not regions arguing against). Correlation not causation — does not prove the model "understands" the object. |
| Class-specific spatial attention (same image, different heatmaps for different classes) | DEVELOPED | visualizing-features | Core property of Grad-CAM. Same image of dog on bench: "golden retriever" highlights dog, "park bench" highlights bench. Different class scores produce different gradients produce different channel weights produce different spatial focus. Taught via inline SVG diagram, explanation, and comprehension check. |

## Per-Lesson Summaries

### visualizing-features
**Status:** Built
**Cognitive load type:** STRETCH
**Widget:** None (inline SVG schematics + Colab notebook)

**What was taught:**
- Three visualization techniques, each answering a different question about what a CNN learned
- Filter visualization: display conv1 weights as tiny RGB images showing edge/color detectors
- Activation maps: capture intermediate layer outputs via hooks, showing the feature hierarchy from concrete (edges) to abstract (distributed patterns)
- Grad-CAM: flow gradients from class score back to last conv layer to produce class-specific spatial heatmaps
- PyTorch hooks as a mechanism for capturing intermediate activations without modifying the model
- Limitations of visualization (resolution, positive-only, not causal)
- Shortcut learning as a critical failure mode where correct predictions hide broken reasoning

**How concepts were taught:**
- **Hook ("You've been taking my word for it"):** Recalls feature hierarchy claims from 3.1 and 3.2 that were accepted on trust. Frames the lesson as empirical investigation: "you have never actually looked." References the same ResNet-18 from transfer-learning lesson. Practical motivation: visualization for debugging, not just curiosity.
- **Filter visualization (4 modalities):**
  - Verbal: "These are the 'questions' the first layer asks at every 7x7 patch"
  - Visual: Inline SVG schematic of 8 typical conv1 filter patterns with labels (vertical, horizontal, diagonal, color, stripes, blob)
  - Symbolic: Code block accessing model.conv1.weight.data, shape [64, 3, 7, 7], normalize and display
  - Concrete: Three categories of what you see (oriented edges, color gradients, high-frequency patterns) and what you do NOT see (no cats, no cars, no objects)
  - Connection to transfer learning: conv1 filters are universal, confirming why freezing conv1 works across domains
  - Limitation: only works for conv1 (deeper layers have multi-channel inputs)
- **Check 1 (predict and verify):** "Feed a cat through conv1 with edge-detection filters — what would the activation maps look like?" Expected: bright at edges, dark in uniform regions. Bridges to activation maps.
- **Hooks section (brief, 3 paragraphs + code):** Problem: model(image) only returns final output, intermediate activations discarded. Solution: register_forward_hook callback. "Placing a sensor on a layer." Four-line pattern: define, register, forward pass, remove. Misconception addressed: "you need to modify the model architecture to see intermediate layers."
- **Activation maps (5 modalities):**
  - Visual: Inline SVG showing input -> conv1 (sharp edges) -> layer2 (blurry textures) -> layer4 (abstract blobs), with spatial resolution labels
  - Verbal: Three PhaseCards describing conv1 (112x112, spatially detailed), layer2 (28x28, textures), layer4 (7x7, abstract)
  - Intuitive: "Filter visualization shows what the layer LOOKS FOR. Activation maps show what the layer FOUND."
  - Concrete: Specific spatial resolutions, channel counts (64 -> 128 -> 512)
  - Comparative: ComparisonRow between filter visualization and activation maps
  - Negative example: layer4 activations are NOT recognizable objects — representation is distributed across channels
- **Check 2 (distinguish techniques):** "An activation map shows all horizontal edges. Is this the same as knowing what class?" Expected: No — feature detection is class-agnostic. Bridges to Grad-CAM.
- **Grad-CAM (6 modalities):**
  - Intuitive: "Ask the network to retrace its reasoning — which parts of the last feature maps were most important?"
  - Concrete steps: 6-step procedure (forward pass, backward pass, GAP gradients, weighted sum, ReLU, upsample)
  - Concrete worked example: GradientCard with 3 channels at 2x2 spatial resolution, actual numbers for gradient weights (0.8, 0.1, 0.5), activation maps, weighted sum computation, ReLU step
  - Symbolic: BlockMath formulas for alpha_k and L_Grad-CAM
  - Visual: Inline SVG showing same image with different Grad-CAM heatmaps for "golden retriever" (highlights dog) vs "park bench" (highlights bench)
  - Code: Full implementation with forward_hook + register_full_backward_hook, forward pass, backward pass, weighted sum, ReLU, normalization
  - Connection to backprop: "same mechanism that trains the network, different purpose"
  - Connection to GAP: "same global average pooling from ResNet classification head, applied to gradients"
- **Check 3 (class-specific heatmaps):** "Same image, Grad-CAM for 'park bench' instead of 'golden retriever'?" Expected: heatmap shifts to bench. Reinforces class-specificity.
- **Activation Maps vs Grad-CAM comparison:** ComparisonRow with 4 parallel items (found vs mattered, class-agnostic vs class-specific, all features vs class-relevant features, different questions).
- **Limitations and shortcut learning:**
  - Three specific limitations listed (resolution, positive-only, not causal)
  - Shortcut learning GradientCard: husky/wolf example — model focuses on snow/forest background, not the animal. 90% accuracy is real, reasoning is broken.
  - Central practical insight: "correct prediction does not mean correct reasoning"
- **Notebook (Colab, scaffolded):** Student implements all three techniques: visualize 64 conv1 filters, register hooks and capture activation maps at three depths, implement Grad-CAM step by step, compare heatmaps for different classes, investigate shortcut learning patterns. Boilerplate provided for image loading, preprocessing, display. Expected: 20-30 minutes on Colab GPU.
- **Summary:** Three questions, three tools framework. SummaryBlock with 4 items covering each technique plus the "debugging tool" insight.
- **Next step:** GradientCard teasing Lesson 2: fine-tune pretrained model on own dataset and use Grad-CAM to verify it learned the right features.

**Mental models established:**
- "Three questions, three tools" — filter viz (what does it look for?), activation maps (what did it find?), Grad-CAM (what mattered for this prediction?) — each technique answers a different question at a different level
- "Visualization is a debugging tool, not just a pretty picture" — shortcut learning makes this concrete
- "Correct prediction does not mean correct reasoning" — the most practically important insight
- "The feature hierarchy is real — you saw it" — transforms the trusted mental model into observed evidence

**Analogies used:**
- "Placing a sensor on a layer" — hooks record what passes through without changing anything
- "Asking the network to retrace its reasoning" — Grad-CAM as explanation via gradient flow
- "Questions the first layer asks" — filters as pattern-matching queries applied at every spatial position
- "Evidence instead of faith" — framing the entire lesson as empirical verification of prior claims

**What was NOT covered (scope boundaries):**
- Activation maximization / deep dream (generating images that maximize neuron activation)
- Saliency maps via input gradients (pixel-level attribution)
- LIME, SHAP, or model-agnostic interpretability methods
- Adversarial examples
- Feature inversion (reconstructing inputs from activations)
- Grad-CAM++ or other variants
- Training or fine-tuning models (deferred to Lesson 2)

**Misconceptions addressed:**
1. "First-layer filters should look like objects" — WarningBlock in filter visualization section. A 7x7 pixel patch cannot contain a dog. Object representations emerge only through many layers of composition. Conv1 filters are always low-level feature detectors.
2. "Deeper activation maps should show recognizable objects" — WarningBlock in activation maps section. Layer4 activations look like abstract blobs. Object-level meaning is encoded in the pattern across all 512 channels, not in any single channel.
3. "Activation maps and Grad-CAM show the same thing" — Addressed via Check 2 (activation maps are class-agnostic), the ComparisonRow after Grad-CAM, and the class-specific heatmap demonstration.
4. "You need to modify the model architecture to visualize intermediate layers" — InsightBlock in hooks section. Hooks let you inspect any layer of any model without touching its code.
5. "If Grad-CAM highlights the right region, the model is correct/trustworthy" — Shortcut learning section. Husky/wolf example: high accuracy, broken reasoning. Grad-CAM is correlation, not causation.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "Three questions, three tools" (filter viz / activation maps / Grad-CAM) | visualizing-features | |
| "Visualization is a debugging tool, not just a pretty picture" | visualizing-features | |
| "Correct prediction does not mean correct reasoning" | visualizing-features | |
| "The feature hierarchy is real — you saw it" (evidence vs faith) | visualizing-features | |
| "Placing a sensor on a layer" (hooks) | visualizing-features | |
| "Asking the network to retrace its reasoning" (Grad-CAM) | visualizing-features | |
