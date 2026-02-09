# Lesson Plan: Visualizing Features

**Module:** 3.3 — Seeing What CNNs See
**Position:** Lesson 1 of 2
**Slug:** `visualizing-features`
**Notebook:** Yes (`3-3-1-visualizing-features.ipynb`)
**Cognitive load type:** STRETCH

---

## Phase 1: Orient — Student State

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Feature hierarchy (edges -> textures -> parts -> objects) | DEVELOPED | building-a-cnn (3.1.2), architecture-evolution (3.2.1) | Taught conceptually via the "zoom out" analogy and receptive field growth. Discussed repeatedly in VGG and ResNet contexts. Student believes it but has never seen it. |
| Learned filters (not hand-designed) | INTRODUCED | what-convolutions-compute (3.1.1) | Student knows that training discovers edge-like filters via backprop. Saw reference to AlexNet/VGGNet first-layer filters but not the actual visualizations. |
| Conv2d, feature maps, sliding filter operation | DEVELOPED | what-convolutions-compute (3.1.1), building-a-cnn (3.1.2) | Student understands what a convolutional layer computes mechanically and can trace dimensions. |
| Pretrained models / torchvision.models API | DEVELOPED | transfer-learning (3.2.3) | Student loaded ResNet-18, inspected it with print(), accessed model.fc, model.layer1, etc. |
| Feature transferability spectrum (early universal, later task-specific) | DEVELOPED | transfer-learning (3.2.3) | Taught that conv1/layer1 detect universal features (edges, textures) and layer4 is task-specific. This is exactly what visualization will confirm. |
| Receptive field (qualitative and quantitative) | DEVELOPED | what-convolutions-compute (3.1.1), architecture-evolution (3.2.1) | Understands RF = n(k-1) + 1 for stacked convs. Knows deeper layers "see" larger regions. |
| Gradients and backpropagation | APPLIED | Series 1 (1.3), Series 2 (autograd) | Student has computed gradients, used .backward(), understands gradient flow. Essential for Grad-CAM. |
| model.eval() vs model.train() | DEVELOPED | resnets (3.2.2), transfer-learning (3.2.3) | Knows eval mode freezes BN running statistics. Relevant for inference-time visualization. |
| "Architecture encodes assumptions about data" | DEVELOPED | mnist-cnn-project (3.1.3), architecture-evolution (3.2.1) | Central mental model of Series 3. Visualization provides evidence for this claim. |
| requires_grad, autograd computational graph | APPLIED | autograd (2.1.2), transfer-learning (3.2.3) | Student has manipulated requires_grad for freezing layers. Grad-CAM requires understanding that gradients can flow to intermediate activations. |

### Established Mental Models and Analogies

- "A filter is a pattern detector" (3.1.1)
- "Feature map is a spatial answer key" (3.1.1)
- "A CNN is a series of zoom-outs" (3.1.2)
- "Spatial shrinks, channels grow, then flatten" (3.1.2)
- "Hire experienced, train specific" (3.2.3)
- Feature transferability spectrum: early = universal, later = task-specific (3.2.3)

### What Was Explicitly NOT Covered

- Actual visualization of learned filters or activation maps (referenced many times, never shown)
- Class activation mapping or any interpretability technique
- Hooks in PyTorch (register_forward_hook)
- Using gradients for anything other than weight updates (gradient w.r.t. activations for interpretation)

### Readiness Assessment

The student is highly prepared. They have strong conceptual understanding of what CNNs compute (filters, feature maps, receptive fields), practical experience with pretrained models, and solid autograd skills. The key gap is PyTorch hooks (needed to capture intermediate activations), which is a small API extension, not a conceptual leap. The student has been promised feature hierarchy evidence for multiple lessons — there is high intrinsic motivation to finally see it.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to visualize and interpret what a trained CNN has learned at three levels: raw filter weights (what patterns the network looks for), activation maps (what the network responds to in a specific image), and Grad-CAM heatmaps (which spatial regions matter for a specific class prediction).

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Convolution as sliding filter | DEVELOPED | DEVELOPED | what-convolutions-compute (3.1.1) | OK | Must understand what a filter does to interpret filter visualizations |
| Feature maps (output of conv layer) | DEVELOPED | DEVELOPED | what-convolutions-compute (3.1.1) | OK | Activation maps ARE feature maps for a specific input — same concept |
| Feature hierarchy (edges -> objects) | DEVELOPED | DEVELOPED | building-a-cnn (3.1.2) | OK | The claim being verified; student needs the mental model to interpret what they see |
| Pretrained models / torchvision API | DEVELOPED | DEVELOPED | transfer-learning (3.2.3) | OK | Loading and inspecting a pretrained model is the starting point |
| Receptive field (qualitative) | INTRODUCED | DEVELOPED | what-convolutions-compute (3.1.1) | OK | Explains why deeper layers show larger-scale features |
| Gradients / backpropagation | DEVELOPED | APPLIED | Series 1 + 2 | OK | Grad-CAM uses gradients of class score w.r.t. activation maps |
| requires_grad / autograd | DEVELOPED | APPLIED | autograd (2.1.2) | OK | Understanding that gradients can be computed w.r.t. any tensor with requires_grad |
| model.eval() mode | INTRODUCED | DEVELOPED | resnets (3.2.2) | OK | Must use eval mode for inference-time visualization |
| PyTorch hooks (register_forward_hook) | INTRODUCED | MISSING | — | MISSING | Needed to capture intermediate layer activations without modifying the model |
| Global average pooling | INTRODUCED | INTRODUCED | resnets (3.2.2) | OK | Grad-CAM is mathematically related to GAP — understanding helps but is not strictly required |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| PyTorch hooks (register_forward_hook) | Small | Brief dedicated section (2-3 paragraphs + code example) before activation maps. The concept is simple: "attach a callback to a layer that captures its output during forward pass." Student already knows nn.Module structure and model.layer1 access. Hooks are a small API extension. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "First-layer filters should look like objects (cats, dogs, cars)" | Intuitive expectation: the network classifies objects, so it should look for objects. Conflating the network's output (class label) with its input-level processing. | Show actual conv1 filters: they are 3x3 or 7x7 edge/color detectors, not object templates. A 7x7 patch cannot contain an object. The object representation emerges only through many layers of composition. | Filter visualization section — the very first reveal |
| "Activation maps and Grad-CAM show the same thing" | Both produce spatial heatmaps overlaid on the input image. Visually similar outputs suggest similar meaning. | Show activation map for a specific filter (e.g., edge detector activating on ALL edges in image) vs Grad-CAM (highlighting only the edges relevant to the predicted class). Activation map for "horizontal edges" lights up everywhere; Grad-CAM for "golden retriever" highlights only the dog region. | After both techniques are introduced — explicit comparison |
| "If Grad-CAM highlights the right region, the model is correct/trustworthy" | Conflating spatial attention with valid reasoning. Grad-CAM showing the dog region feels like proof of understanding. | Example: model predicts "husky" and Grad-CAM highlights the snowy background, not the dog. High accuracy on a biased dataset where all huskies appear in snow. The model learned the shortcut (snow = husky), not the concept (dog features = husky). | Elaborate section — shortcut learning example |
| "You need to modify the model architecture to visualize intermediate layers" | Student has only accessed model outputs. Inspecting internal state feels like it requires architectural changes or rebuilding the model. | Demonstrate hooks: attach a one-line callback, run a forward pass, the hook captures the activation. Model is unchanged. Remove the hook when done. | Hook introduction section |
| "Deeper activation maps should show recognizable objects" | Extension of the "filters should look like objects" misconception to deeper layers. If early layers show edges, surely later layers show faces, cars, etc. | Show actual layer4 activation maps: abstract blob-like patterns, not recognizable objects. The representations are useful for classification but not human-interpretable as images. Object-level meaning is encoded in the *pattern* across many channels, not in any single channel. | Activation maps section — when showing deep layer activations |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| ResNet-18 conv1 filter visualization (64 filters of 7x7x3) | Positive | Show that first-layer filters are edge and color detectors — confirming what the student was told but never shown | Connects directly to "learned filters" (3.1.1) and "feature transferability — early layers are universal" (3.2.3). ResNet-18 is already familiar from transfer-learning lesson. 7x7 filters are large enough to be visually interpretable. |
| Activation maps at conv1, layer2, layer4 for a single image | Positive | Show feature hierarchy in action: early activations respond to edges in the input, middle activations respond to textures/parts, deep activations are abstract spatial patterns | Directly confirms the "edges -> textures -> parts -> objects" hierarchy that has been referenced since 3.1.2. Three layers span the full depth. Using the same image at all three depths makes the progression visible. |
| Grad-CAM on correctly classified image (e.g., dog in a scene) | Positive | Show class-discriminative spatial attention — the model highlights the dog, not the background | The most practically useful technique. Answers "why did the model predict this?" Connects to the "hire experienced, train specific" analogy — the pretrained model knows where to look. |
| Grad-CAM on a shortcut-learning failure (model correct for wrong reasons) | Negative | Show that correct prediction does not mean correct reasoning — Grad-CAM reveals the model focuses on background/context, not the object | Critical for practical use. Challenges the "right answer = right reasoning" assumption. Motivates using visualization as a debugging tool, not just a pretty picture. |
| Activation maps at layer4 — abstract, not human-readable | Negative (boundary) | Show that deep activations are NOT recognizable objects, disproving the expectation that the hierarchy produces increasingly literal representations | Defines the boundary of what visualization can show. The representation is useful for the network but not for human visual inspection at individual-channel level. |

---

## Phase 3: Design

### Narrative Arc

Throughout this entire CNN series, you have been building a mental model on trust. You were told that early layers detect edges, middle layers detect textures, and later layers detect objects. You accepted this because it made sense theoretically — small receptive fields see local patterns, larger receptive fields see global structure. But you have never actually *looked*. This lesson opens the black box. You will load the same ResNet-18 you used for transfer learning, extract its first-layer filters, and see with your own eyes that they are edge and color detectors. Then you will feed an image through the network and watch the activations at different depths, seeing the feature hierarchy unfold from concrete to abstract in a single forward pass. Finally, you will learn Grad-CAM — a technique that answers the most useful question in applied deep learning: "why did the model make this prediction?" By the end, you will have three tools for understanding CNN behavior, each answering a different question, and you will understand both their power and their limitations.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual | Grid display of conv1 filters as small 7x7 images; activation map grids at multiple layers; Grad-CAM heatmaps overlaid on input images | This is fundamentally a visual lesson — the entire point is *seeing* what the network learned. Every core concept is a visualization technique. |
| Concrete example | Specific images run through a real pretrained ResNet-18 with actual filter weights and activation values | Abstract descriptions of "what CNNs learn" are exactly what this lesson replaces. Concrete, specific, real outputs from a real model. |
| Symbolic | Grad-CAM formula: weighted combination of activation maps, weights = global average pool of gradients. alpha_k = (1/Z) * sum(dY/dA_k), then ReLU(sum(alpha_k * A_k)) | The formula grounds the intuition. Student has all the prerequisite math (gradients, GAP, weighted sums). |
| Intuitive | "Filter visualization answers 'what does this layer look for?' Activation maps answer 'what did this layer find in THIS image?' Grad-CAM answers 'what in this image mattered for THIS prediction?'" — three questions, three tools | The three techniques are easily confused. Framing each as answering a different question makes the distinctions memorable. |
| Verbal/Analogy | Grad-CAM as "asking the network to explain its reasoning by retracing its steps" — like asking someone to justify a decision by pointing to the evidence they used | Connects the technical mechanism (gradient flow back to activations) to a familiar experience (explaining a decision). |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3 (filter visualization as a technique, activation map capture via hooks, Grad-CAM)
- **Previous lesson load:** BUILD (transfer-learning, 3.2.3) — practical application, moderate novelty
- **Appropriateness:** This is a STRETCH lesson following a BUILD lesson, which is a good pattern. The three new concepts are related (all are visualization techniques) and build on each other (filters -> activations -> gradient-weighted activations), so the cognitive load is manageable despite the count. The student also has strong motivation — this is the payoff for trust they have been extending since Module 3.1.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|------------|
| "A filter is a pattern detector" (3.1.1) | Filter visualization literally shows the patterns each filter detects. The abstract claim becomes a concrete image. |
| "Feature map is a spatial answer key" (3.1.1) | Activation maps are feature maps for a specific input. Same concept, now made visible. |
| Feature hierarchy "edges -> textures -> parts -> objects" (3.1.2, 3.2.1) | The central claim being verified. Activation maps at different depths show this progression. |
| "A CNN is a series of zoom-outs" (3.1.2) | Activation maps at different depths show increasing spatial scale, confirming the zoom-out analogy. |
| Feature transferability spectrum (3.2.3) | Filter visualization shows that conv1 filters are generic edge/color detectors — directly supporting "early layers are universal." |
| Gradients / backpropagation (Series 1-2) | Grad-CAM uses gradients flowing backward from the class score to the final conv layer. Same mechanism as training, different purpose. |
| requires_grad / autograd (2.1.2) | Hooks and gradient capture rely on the autograd graph. The student's existing autograd knowledge directly enables understanding Grad-CAM's mechanics. |
| Global average pooling (3.2.2) | Grad-CAM weights are computed via global average pooling of gradients — the same operation used in ResNet's classification head. |

**Analogies that could be misleading:** The "zoom out" analogy might lead the student to expect that deeper activation maps look like progressively zoomed-out versions of the input. In reality, deep activations are abstract and don't resemble the input at all. Address this directly when showing layer4 activations.

### Scope Boundaries

**This lesson IS about:**
- Three visualization techniques (filter viz, activation maps, Grad-CAM) at DEVELOPED depth
- PyTorch hooks at INTRODUCED depth (just enough to capture activations)
- Interpreting visualizations: what each technique shows and what it does NOT show
- Limitations of each technique

**This lesson is NOT about:**
- Activation maximization / deep dream (generating images that maximize a neuron's activation)
- Saliency maps via input gradients (pixel-level attribution)
- LIME, SHAP, or other model-agnostic interpretability methods
- Adversarial examples (related but different topic)
- Feature inversion (reconstructing inputs from activations)
- Training or fine-tuning models (that is Lesson 2)
- Grad-CAM++ or other variants (one technique, done well)

**Target depths:**
- Filter visualization: DEVELOPED (can do it and interpret results)
- Activation maps via hooks: DEVELOPED (can capture and interpret at multiple layers)
- Grad-CAM: DEVELOPED (can implement, compute, and interpret — knows what it shows and what it does not)
- PyTorch hooks: INTRODUCED (can use register_forward_hook to capture activations; not expected to use for other purposes)

### Lesson Outline

**1. Context + Constraints**
- This lesson: three visualization techniques, each answering a different question about what a CNN learned
- Not covering: deep dream, saliency maps, model-agnostic methods
- The notebook is where you will run every visualization on real images

**2. Hook — "You've been taking my word for it"**
- Type: Misconception reveal / evidence gap
- Recall the feature hierarchy claim from 3.1 and 3.2: early layers detect edges, later layers detect objects
- "You have been told this repeatedly. Today you will see it — or discover it was wrong."
- Show the student's own quote/belief (from the mental model) and frame this lesson as the experiment that confirms or denies it
- Why this hook: The student has genuine curiosity about whether the mental model is accurate. This frames the lesson as empirical investigation, not passive learning.

**3. Explain 1 — Filter Visualization (technique #1)**
- Load pretrained ResNet-18 (familiar from 3.2.3)
- Access model.conv1.weight — a tensor of shape [64, 3, 7, 7]
- Explain: each of the 64 filters is a 7x7x3 volume. Normalize and display as a 7x7 RGB image.
- Show the 8x8 grid of conv1 filters
- Interpretation: oriented edges at different angles, color gradients, high-frequency patterns
- "These are the 'questions' the first layer asks at every 7x7 patch of the input"
- Connect to "learned filters" from 3.1.1 and "early layers are universal" from 3.2.3
- Negative example: note what you do NOT see — no cats, no cars, no objects. Just edges and colors.
- Limitation: only really works for conv1 (later filters operate on multi-channel feature maps, not RGB — harder to visualize directly)

**4. Check 1 — Predict and Verify**
- "Before I show you activation maps: if you feed a photo of a cat through conv1 with these edge-detection filters, what would the activation maps look like?"
- Expected: bright at edges in the image, dark in uniform regions
- Verify with actual activation maps from conv1

**5. Brief section — PyTorch Hooks**
- Problem: how do you capture intermediate layer outputs without modifying the model?
- Solution: register_forward_hook — attach a callback that stores the layer's output
- Code example: 4 lines — define hook function, register it, run forward pass, remove hook
- "Think of it as placing a sensor on a layer — it records what passes through without changing anything"
- This is a small API introduction, not a deep dive

**6. Explain 2 — Activation Maps (technique #2)**
- Feed a specific image (e.g., a dog in a park) through the network
- Capture activations at conv1, layer2, layer4 using hooks
- Display activation maps at each depth:
  - conv1: responds to edges and color boundaries in the input (sharp, spatially detailed)
  - layer2: responds to textures and local patterns (less spatially precise, more abstract)
  - layer4: abstract spatial patterns, not human-interpretable as individual channels
- "Filter visualization shows what the layer looks FOR. Activation maps show what the layer FOUND in this specific image."
- Confirm feature hierarchy: the "zoom out" progression is visible
- Negative example (boundary): layer4 activations are NOT recognizable objects — the representation is distributed across channels, not localized in any single channel

**7. Check 2 — Distinguish the Techniques**
- "An activation map for a horizontal-edge filter shows all horizontal edges in the image. Is this the same as knowing what class the image is?"
- Expected: No — the activation map shows feature detection (what the layer found), not classification reasoning (what mattered for the class prediction). Edges appear regardless of whether the image is a dog or a cat.
- Bridge to Grad-CAM: "We need a technique that is class-specific."

**8. Explain 3 — Grad-CAM (technique #3)**
- Motivation: activation maps are class-agnostic. We want to know which spatial regions matter for a *specific* prediction.
- Intuition first: "Ask the network to retrace its reasoning. Which parts of the last feature maps were most important for predicting 'golden retriever'?"
- Mechanism (concrete before formula):
  1. Forward pass: get the class score for the predicted class
  2. Backward pass: compute gradients of that class score w.r.t. the last conv layer's activation maps
  3. Global average pool those gradients to get one weight per channel (alpha_k)
  4. Weighted sum of activation maps using those weights
  5. ReLU (keep only positive contributions — regions that help the prediction)
  6. Upsample and overlay on input image
- Formula: alpha_k = (1/Z) * sum_ij (dY^c / dA^k_ij), then L = ReLU(sum_k alpha_k * A^k)
- Connect to prior knowledge: "The backward pass is the same mechanism that trains the network. Here we use it for interpretation instead of weight updates."
- Connect to GAP: "The alpha_k computation is the same global average pooling you saw in ResNet's classification head — but applied to gradients instead of activations."
- Show Grad-CAM for the dog image: heatmap highlights the dog, not the park/background

**9. Check 3 — Transfer Question**
- "For the same image, what would Grad-CAM look like if you asked for the class 'park bench' instead of 'golden retriever'?"
- Expected: heatmap shifts from the dog to the bench region — Grad-CAM is class-specific, same image but different focus
- If available in notebook: actually compute this and verify

**10. Elaborate — Limitations and Shortcut Learning**
- Grad-CAM's resolution is limited by the spatial resolution of the last conv layer (7x7 for ResNet-18) — the heatmap is coarse
- The ReLU means Grad-CAM only shows positive evidence (regions that support the prediction), not negative evidence (regions that argue against it)
- **Shortcut learning example (critical negative example):** A model trained to classify "wolf" vs "husky" achieves 90% accuracy. Grad-CAM reveals it focuses on the background (snow for husky, forest for wolf), not the animal. The accuracy is real but the reasoning is broken. This is a real failure mode.
- "Correct prediction does not mean correct reasoning. Visualization is a debugging tool."

**11. Practice — Notebook**
- Scaffolded Colab notebook:
  - Provided: model loading, image preprocessing, display utilities, Grad-CAM scaffold
  - Student implements: hook registration, activation capture, Grad-CAM computation (fill in the gradient computation and weighted sum)
  - Guided experiments: visualize conv1 filters, activation maps at 3 depths, Grad-CAM for top-2 predicted classes on 2-3 different images
  - Stretch: try Grad-CAM on an image where the model is wrong — does the heatmap reveal why?

**12. Summarize — Three Questions, Three Tools**
- Filter visualization: "What does this layer look for?" (model-level, not input-specific)
- Activation maps: "What did this layer find in this image?" (input-specific, not class-specific)
- Grad-CAM: "What in this image mattered for this prediction?" (input-specific AND class-specific)
- The feature hierarchy is real — you saw it
- Visualization is a debugging tool, not just a pretty picture

**13. Next Step**
- In the next lesson, you will fine-tune a pretrained model on your own dataset and use these visualization tools to verify that the model learned what you intended — not shortcuts or artifacts.

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 2

### Verdict: NEEDS REVISION

No critical issues that would leave the student fundamentally lost, but one critical finding around the visual modality gap and several improvement findings that would meaningfully strengthen the lesson. Another pass needed after fixes.

### Findings

#### [CRITICAL] — Visual modality is entirely deferred to the notebook; lesson text has zero images

**Location:** Entire lesson, particularly Filter Visualization (section 3), Activation Maps (section 6), and Grad-CAM (section 8)
**Issue:** This is a lesson whose entire purpose is *seeing* what a CNN learned. The planning document lists Visual as the first modality and says "This is fundamentally a visual lesson — the entire point is seeing what the network learned." Yet the built lesson contains zero inline images, diagrams, or visual outputs. Every visual is deferred to the notebook with phrases like "When you run this in the notebook, you will see..." or "In the notebook, you will capture activations at three depths." The lesson describes what the student would see in words rather than showing it. The Filter Visualization section says "you will see 64 tiny images" and then describes them as a bullet list. The activation maps section uses three PhaseCards with text descriptions. The Grad-CAM section says "When you overlay the heatmap on a photo of a dog in a park, the result is striking" — but never shows it.
**Student impact:** The student reads the entire lesson without seeing a single visualization. The lesson is about visualization, yet the student's experience is purely verbal/textual until they open the notebook. This makes the lesson feel like an extended preview rather than teaching. The student must hold the conceptual framework in their head without the visual anchors it is built around. For an ADHD-friendly design where "seeing is believing" is the hook, deferring all visuals to a separate environment is a significant modality gap.
**Suggested fix:** Add at least 2-3 static images inline in the lesson: (1) an 8x8 grid of actual ResNet-18 conv1 filters (this is a fixed pretrained model, the image never changes), (2) a side-by-side of conv1/layer2/layer4 activation maps for a sample image, (3) a Grad-CAM heatmap overlay example. These can be pre-generated PNGs committed to the repo and embedded with `<img>` tags or the Next.js Image component. The notebook remains the hands-on practice, but the lesson should show the key visuals inline so the text descriptions land with visual evidence.

#### [IMPROVEMENT] — Explicit comparison between activation maps and Grad-CAM is missing

**Location:** Between section 7 (Check 2) and section 8 (Explain 3: Grad-CAM)
**Issue:** The planning document's misconceptions table includes "Activation maps and Grad-CAM show the same thing" and specifies addressing it "After both techniques are introduced — explicit comparison." The built lesson addresses this distinction indirectly through Check 2 (which bridges to Grad-CAM by noting activation maps are class-agnostic) and through the InsightBlock "The Missing Piece." However, after Grad-CAM is fully introduced, there is no direct side-by-side comparison showing the same image with an activation map vs a Grad-CAM heatmap. The ComparisonRow between Filter Visualization and Activation Maps exists (section 6), but there is no equivalent ComparisonRow or explicit comparison between Activation Maps and Grad-CAM.
**Student impact:** The student understands the conceptual distinction between "class-agnostic" and "class-specific" but has no concrete visual or structural comparison that makes the difference memorable. The misconception is partially addressed but not thoroughly disproved with a concrete side-by-side.
**Suggested fix:** Add a ComparisonRow after the Grad-CAM explanation (or after Check 3) comparing Activation Maps vs Grad-CAM in the same format as the existing Filter Viz vs Activation Maps comparison. Include items like: "Same for all classes / Different for each class," "Shows all detected features / Shows only features relevant to one class," "Answers 'what did the layer find?' / Answers 'what mattered for this prediction?'"

#### [IMPROVEMENT] — Grad-CAM explanation front-loads formula before concrete worked example

**Location:** Section 8, Grad-CAM explanation (lines 566-649)
**Issue:** The plan says "Mechanism (concrete before formula)" and the ordering rules require "concrete before abstract." The built lesson partially follows this: it gives the six-step procedure first (good), then the formula. However, neither the six steps nor the formula include a concrete worked example with actual numbers. The steps describe the procedure abstractly ("compute gradients of that class score...," "global average pool the gradients...") and the formula uses symbols. The lesson then jumps to a full code block. There is no intermediate "here is what happens with a concrete tiny example" — for instance, showing 3 channels (not 512), 2x2 spatial (not 7x7), with actual gradient values, to make the weighted-sum-then-ReLU tangible before scaling to full ResNet.
**Student impact:** The student gets the procedure and the math but may not have the intuition for *what the numbers look like at each step*. The code block is comprehensive but abstract. A student who struggles with the mathematical notation will not have a concrete fallback. The verbal analogy ("ask the network to retrace its reasoning") is good but does not substitute for a small worked example with real numbers.
**Suggested fix:** Add a brief concrete mini-example before or alongside the formula. Example: "Imagine a toy case with 3 channels at 2x2 spatial resolution. Channel 1 has high gradient weight (0.8) and its activation map highlights the top-left. Channel 2 has low weight (0.1). Channel 3 has medium weight (0.5) and highlights bottom-right. The weighted sum: 0.8 * [bright, dim, dim, dim] + 0.1 * [...] + 0.5 * [dim, dim, dim, bright] = [mostly top-left and some bottom-right]. After ReLU, both positive contributions remain." This grounds the formula in specific values before the student sees the full 512-channel code.

#### [IMPROVEMENT] — No explicit "next step" teasing Lesson 2

**Location:** Section 13 (end of lesson, lines 940-955)
**Issue:** The planned outline item 13 states: "In the next lesson, you will fine-tune a pretrained model on your own dataset and use these visualization tools to verify that the model learned what you intended — not shortcuts or artifacts." The built lesson ends with an InsightBlock titled "The Feature Hierarchy Is Real" that summarizes the lesson's confirmatory achievement, which is good. But it has no forward-looking content. The student finishes the lesson with no indication of what comes next or why these tools matter beyond this lesson.
**Student impact:** The student gets closure on this lesson but misses the motivational bridge to Lesson 2. The shortcut learning example earlier creates urgency ("how do I prevent this?"), but the lesson never connects that urgency to the next lesson where the student will actually use visualization as a debugging tool on their own project. The ADHD-friendly principle of "clear next step on every app open" is not served.
**Suggested fix:** Add a brief paragraph or InsightBlock after the summary that previews Lesson 2: "In the next lesson, you will fine-tune a pretrained model on your own small dataset and use Grad-CAM to verify that it learned the right features — not shortcuts."

#### [IMPROVEMENT] — Notebook Part 5 (shortcut learning) uses a placeholder/mismatched image URL

**Location:** Notebook cell-31 (Part 5: Shortcut Learning Example)
**Issue:** The code contains a commented-out ant image URL that was clearly a placeholder, followed by a husky image URL. However, the husky image URL points to a Wikimedia image of a husky at a lake (Lac de Laffrey), not a husky in snow. The entire pedagogical point of this section is that the model focuses on snow (background) rather than the dog. If the background is a lake, the shortcut-learning demonstration does not work as intended. The student will not see the "model focuses on snow" behavior because there is no snow in the image.
**Student impact:** The exercise that is described as "the most practically important part of the notebook" may not demonstrate the intended shortcut-learning behavior. The student reads the lesson text about snow/forest shortcuts and then gets a notebook exercise that does not reproduce that phenomenon, creating a disconnect between theory and practice.
**Suggested fix:** Either (1) find a Wikimedia-licensed image of a husky in a snowy setting where the snow background is prominent, or (2) reframe the notebook section to focus on what Grad-CAM reveals about the model's focus generally (where does it look? is it the animal or the context?) rather than specifically trying to reproduce the snow-shortcut phenomenon with a pretrained ImageNet model. Option 2 may be more honest — a pretrained ImageNet ResNet-18 is not the husky-vs-wolf biased model from the paper, so the shortcut-learning demonstration would require a custom-trained model anyway. The notebook could instead note: "The wolf/husky shortcut example requires a model trained on biased data. With a general ImageNet model, Grad-CAM still reveals interesting focus patterns — the model might attend to context more than you expect."

#### [POLISH] — filter_viz.py code normalizes all filters globally instead of per-filter

**Location:** Section 3, Filter Visualization code block (lines 174-192)
**Issue:** The lesson's inline code uses `filters -= filters.min()` and `filters /= filters.max()` which normalizes all 64 filters together using the global min and max. This means filters with low dynamic range will appear washed out. The notebook code correctly normalizes per-filter (line-by-line in a loop). The lesson code is simplified for clarity, which is reasonable, but the discrepancy could confuse a student who tries the lesson code first and gets different-looking results from the notebook.
**Suggested fix:** Either add a brief comment to the lesson code noting this is a simplified global normalization (and the notebook does per-filter normalization for better visualization), or update the lesson code to match the notebook's per-filter approach. A one-line comment would suffice.

#### [POLISH] — Backward hook uses `register_full_backward_hook` without explaining the "full" distinction

**Location:** Section 8, Grad-CAM code block (line 676)
**Issue:** The hooks section introduces `register_forward_hook` and the Grad-CAM code uses `register_full_backward_hook`. The lesson does not explain why it is "full" backward hook vs `register_backward_hook`. This is a minor API detail, but a curious student might wonder about the distinction.
**Suggested fix:** Add a brief aside or inline comment: "We use `register_full_backward_hook` (the modern API) rather than the deprecated `register_backward_hook`." One sentence is enough.

### Review Notes

**What works well:**
- The narrative arc is excellent. The "you've been taking my word for it" hook creates genuine motivation for a student who has been told about the feature hierarchy for 6+ lessons without seeing it. The framing as empirical investigation rather than passive learning is compelling.
- The three-question framework (what does it look for / what did it find / what mattered for this prediction) is clear, memorable, and well-threaded throughout the lesson. Each technique is introduced with its question, reinforced in the checks, and summarized at the end.
- Misconceptions are well-placed: the "filters should look like objects" warning appears right when the student would form that expectation; the "deep layers should show objects" warning appears when showing layer4 activations; the shortcut learning example lands after the student has gained confidence in Grad-CAM.
- The progression from simple (filter viz) to intermediate (activation maps + hooks) to complex (Grad-CAM) is well-paced. Each technique builds on the previous one.
- The notebook is thorough and well-scaffolded, with appropriate TODO markers and provided utilities.
- Connections to prior concepts are explicit and frequent (Transfer Learning, Building a CNN, autograd, GAP from ResNets).
- Scope boundaries are clearly stated and respected.

**Systemic pattern:** The visual modality gap is the most significant issue. For a lesson specifically about visualization, the web lesson text relies entirely on verbal descriptions of visual outputs, deferring all actual visuals to the notebook. This is like teaching someone about color by describing it in words. Future visualization-heavy lessons should include pre-generated static images inline.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All 7 findings from Iteration 1 have been effectively addressed. The lesson is now pedagogically sound and ready to ship. One minor polish item remains (em dash spacing in SVG/step labels) that does not affect the student experience and requires no re-review.

### Findings

#### [POLISH] — Spaced em dashes in SVG text and worked example step labels

**Location:** Filter visualization SVG (lines 284, 287), activation map SVG (lines 546, 557, 569), Grad-CAM worked example (lines 769, 778, 788)
**Issue:** The writing style rule requires em dashes with no spaces (`word&mdash;word`). Several SVG `<text>` elements and the worked example step labels use ` &mdash; ` with spaces on both sides. Examples: `8 of 64 filters &mdash; actual conv1 kernels`, `112x112 &mdash; sharp edges`, `Step 3 &mdash; Gradient weights:`.
**Student impact:** Minimal. These are labels inside SVG illustrations and step headings in a worked example. The spacing is consistent within its context and does not affect readability. In SVG text, spaced em dashes may actually read better at small font sizes.
**Suggested fix:** Replace ` &mdash; ` with `&mdash;` (no spaces) in the SVG text elements and step labels for consistency with the rest of the lesson's em dash usage, or leave as-is given the SVG context. Not worth a re-review.

### Iteration 1 Fix Verification

All 7 findings from Iteration 1 have been addressed:

1. **[CRITICAL] Visual modality gap — FIXED.** Three inline SVG illustrations added: (a) schematic of 8 typical conv1 filter patterns with labels (lines 215-290), (b) activation maps at three depths showing the edge-to-abstract progression with arrows and resolution labels (lines 511-576), (c) Grad-CAM heatmap overlay showing class-specific focus with two different class targets on the same image (lines 849-912). These are well-designed schematic illustrations that convey the key visual concepts without requiring the student to open the notebook first. The lesson now has genuine visual modality inline.

2. **[IMPROVEMENT] Activation maps vs Grad-CAM comparison — FIXED.** A ComparisonRow (lines 1010-1033) now directly compares Activation Maps vs Grad-CAM with four parallel items: "Shows what the layer FOUND in this image" vs "Shows what mattered for THIS prediction," "Same for all classes" vs "Different for each class," etc. Placed after Check 3, exactly where the misconception would be addressed. Well-structured and memorable.

3. **[IMPROVEMENT] Grad-CAM concrete worked example — FIXED.** A GradientCard (lines 762-796) titled "Worked Example: Grad-CAM with 3 Channels" walks through a toy case with 3 channels at 2x2 spatial resolution. Shows specific gradient weights (0.8, 0.1, 0.5), specific activation maps per channel, the full weighted sum computation with actual numbers, and the ReLU step. This grounds the formula in concrete values before the student encounters the symbolic notation. The example is well-positioned between the six-step procedure and the formula, maintaining the concrete-before-abstract ordering.

4. **[IMPROVEMENT] Next step teaser — FIXED.** A GradientCard (lines 1241-1250) titled "Up Next" now previews Lesson 2: fine-tuning a pretrained model on the student's own dataset and using Grad-CAM to verify it learned the right features. This creates the motivational bridge from the shortcut learning discussion to the next lesson and satisfies the "clear next step" ADHD-friendly principle.

5. **[IMPROVEMENT] Notebook shortcut learning section — FIXED.** The notebook's Part 5 (cell-30, cell-31) has been reframed. The markdown now explicitly acknowledges that the wolf/husky shortcut example requires a model trained on biased data, and that a pretrained ImageNet model cannot reproduce that exact failure. The exercise is now framed as "is the model looking at the object, or at the context?" with images that have prominent backgrounds (husky by a lake, bird with foliage). This is honest and pedagogically sound.

6. **[POLISH] Filter normalization comment — FIXED.** Line 185 now includes the comment `# The notebook normalizes per-filter for better contrast`, explaining the discrepancy between the simplified global normalization in the lesson code and the per-filter approach in the notebook.

7. **[POLISH] Backward hook API comment — FIXED.** Line 931 now includes the comment `# register_full_backward_hook is the modern API (register_backward_hook is deprecated)`, addressing the "full" distinction.

### Review Notes

**What works well (building on Iteration 1 strengths):**
- All three inline SVG illustrations are effective schematic diagrams. They are not screenshots of actual model outputs (which would be static and potentially confusing), but schematic representations that clearly illustrate the core concepts. The filter pattern grid labels each pattern type (vertical, horizontal, diagonal, color, stripes, blob). The activation map progression shows spatial resolution shrinking and representations going abstract. The Grad-CAM diagram shows the same image producing different heatmaps for different class targets. These are the right level of abstraction for inline illustrations.
- The worked example with 3 channels at 2x2 is excellent. It uses monospace formatting to show the actual matrix values, making the weighted sum computation traceable by hand. The progression (gradient weights -> weighted sum -> ReLU) mirrors the six-step procedure and the formula, creating three aligned representations of the same operation.
- The ComparisonRow between activation maps and Grad-CAM completes the "three questions, three tools" framework with explicit structural comparisons between all pairs: Filter Viz vs Activation Maps (line 601), and Activation Maps vs Grad-CAM (line 1011).
- The next step teaser creates a clear motivational arc: shortcut learning creates urgency ("how do I prevent this?") -> summary provides closure -> "Up Next" channels that urgency into the next lesson.
- The notebook reframing of the shortcut learning section is honest and avoids overpromising. The student will not be confused when the pretrained ImageNet model does not reproduce the wolf/husky failure mode.
- The narrative arc, misconception placement, pacing, and scope boundaries continue to be strong from the original build.

**No systemic issues remain.** The visual modality gap identified in Iteration 1 has been resolved with appropriate inline illustrations. The lesson now delivers on its promise of being a visual lesson within the web experience itself, with the notebook serving as hands-on practice rather than the sole source of visual content.
