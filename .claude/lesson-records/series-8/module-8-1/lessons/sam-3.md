# Lesson 2: SAM 3 (sam-3) -- Planning Document

**Module:** 8.1 Vision & Vision-Language Models
**Position:** Lesson 2 of 2+
**Type:** BUILD
**Slug:** sam-3

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Convolution as sliding filter (multiply-and-sum over local region) | DEVELOPED | what-convolutions-compute (3.1.1) | Core CNN operation. Student computed by hand and explored interactively with ConvolutionExplorer widget. Can explain filters, feature maps, edge detection. |
| Hierarchical feature composition (edges -> corners -> shapes -> objects) | DEVELOPED | building-a-cnn (3.1.2) | The conv-pool pattern enables this hierarchy. Each stage expands receptive field, representing increasingly abstract features. |
| Conv-pool-fc architecture pattern | DEVELOPED -> APPLIED | building-a-cnn (3.1.2), mnist-cnn-project (3.1.3) | Full dimension tracking, implemented end-to-end. Mental model: "spatial shrinks, channels grow, then flatten." |
| Vision Transformer / ViT (patchify, process patches as tokens with standard transformer) | DEVELOPED | diffusion-transformers (7.4.2) | Full tensor shape trace of patchify. "Tokenize the image" analogy. Deep understanding from DiT lesson. |
| Scaled dot-product attention and multi-head attention | DEVELOPED | queries-and-keys (4.2.2), multi-head-attention (4.2.4) | Full formula: output = softmax(QK^T / sqrt(d_k)) V. Multi-head: independent heads with dimension splitting, W_O mixing. Student built from scratch. |
| Cross-attention mechanism (Q from one source, K/V from another) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Taught as a one-line change from self-attention: K and V from text, Q from spatial features. Same formula, different sources. |
| Residual connections / skip connections | DEVELOPED | resnets (3.2.2), the-transformer-block (4.2.5) | F(x) + x formulation. "Editing a document, not writing from scratch." Used in ResNets and transformer blocks. |
| U-Net encoder-decoder architecture with skip connections | DEVELOPED | unet-architecture (6.3.1) | Encoder downsampling, bottleneck, decoder upsampling. Skip connections carry spatial detail. "Bottleneck decides WHAT, skips decide WHERE." |
| Transfer learning (reusing pretrained model weights) | DEVELOPED | transfer-learning (3.2.3) | Feature extraction and fine-tuning strategies. "Hire experienced, train specific." |
| Contrastive learning / CLIP (dual encoders, shared embedding space) | DEVELOPED | clip (6.3.3) | Conference analogy, similarity matrix, symmetric cross-entropy. Student deeply understands the shared embedding space. |
| Image segmentation (pixel-level labeling of objects) | NOT TAUGHT | N/A | The student has not been formally taught segmentation as a task. Semantic segmentation, instance segmentation, and panoptic segmentation are all new. The U-Net lesson mentioned its origin in medical image segmentation (MENTIONED depth) but did not teach segmentation as a task. |
| Object detection (bounding boxes around objects) | NOT TAUGHT | N/A | The student has not been formally taught object detection. DETR was not covered. Bounding boxes as a representation are intuitive but have not been discussed. |
| Foundation model concept (one model pretrained on broad data, promptable for many tasks) | INTRODUCED | Multiple contexts | The student has encountered this idea implicitly: GPT as a foundation model for language (Series 4), CLIP as a foundation model for vision-language alignment (6.3.3). The term "foundation model" itself has been used. But the student has not analyzed what makes something a foundation model vs a task-specific model. |
| Promptable interface (conditioning a model on user-provided prompts) | DEVELOPED | Multiple (LLM prompting 4.x, CFG text prompts 6.3.4) | The student understands prompting from LLMs (text prompts -> completions) and diffusion (text prompts -> images via cross-attention). SAM's prompts are different: spatial prompts (points, boxes) and now text prompts (SAM 3). |

**Mental models and analogies already established:**
- "Architecture encodes assumptions about data" (from CNN module)
- "Tokenize the image" (ViT patchify from DiT lesson)
- "Bottleneck decides WHAT, skips decide WHERE" (U-Net dual-path information flow)
- "Three lenses, one embedding" (Q/K/V projections)
- "Hire experienced, train specific" (transfer learning)
- "Two encoders, one shared space" (CLIP/contrastive learning)
- "A filter is a pattern detector" (CNN convolutions)
- "Attention reads, FFN writes" (transformer block)
- "Same formula, different source for K and V" (cross-attention)

**What was explicitly NOT covered in prior lessons (relevant here):**
- Image segmentation as a task (classification labels per pixel, not per image)
- The difference between semantic, instance, and panoptic segmentation
- Dense prediction tasks in general (any task producing a per-pixel output)
- SAM or any segmentation-specific model
- DETR or transformer-based object detection
- Promptable segmentation (spatial prompts: points, boxes, masks)
- The "data engine" approach to dataset creation (human-in-the-loop annotation at scale)
- SAM 2's memory mechanism for video
- SAM 3's concept-based open-vocabulary segmentation

**Readiness assessment:** The student is well-prepared for the architectural concepts. They have deep understanding of ViT (image encoder), cross-attention (prompt-image fusion), and the U-Net / encoder-decoder pattern (mask decoder). The main gap is segmentation itself -- the student has never been formally taught what segmentation IS as a task, how it differs from classification and detection, or what makes it challenging. This gap is manageable because: (1) the student already understands per-pixel outputs from the U-Net lesson (the U-Net produces spatial output maps), (2) the concept is intuitive ("draw an outline around the object"), and (3) a dedicated section can establish segmentation at INTRODUCED depth before using it. The BUILD designation is appropriate: SAM applies familiar architectural components (ViT, cross-attention, encoder-decoder) to a new task domain, rather than introducing fundamentally new mechanisms.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how SAM's promptable segmentation architecture (image encoder + prompt encoder + mask decoder) enables a single model to segment any object in any image given a spatial or text prompt, and how this "foundation model for segmentation" approach evolved from SAM 1 through SAM 3.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| ViT / image encoder (patchify, process patches as tokens) | INTRODUCED | DEVELOPED | diffusion-transformers (7.4.2) | OK | SAM uses a ViT as its image encoder. Student needs to recognize the architecture, not build it. DEVELOPED exceeds requirement. |
| Cross-attention (Q from one modality, K/V from another) | INTRODUCED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | SAM's mask decoder uses cross-attention between prompt tokens and image features. Student has deep understanding. |
| Encoder-decoder architecture with skip connections | INTRODUCED | DEVELOPED | unet-architecture (6.3.1) | OK | SAM's mask decoder is a lightweight encoder-decoder. Student knows the U-Net pattern deeply. |
| Residual connections | INTRODUCED | DEVELOPED | resnets (3.2.2) | OK | Used throughout SAM's architecture. Student has extensive experience. |
| CNN feature maps (spatial grids of filter responses) | INTRODUCED | DEVELOPED | what-convolutions-compute (3.1.1) | OK | Image embeddings are spatial feature maps. Student understands feature maps deeply. |
| Hierarchical features (edges -> shapes -> objects across network depth) | INTRODUCED | DEVELOPED | building-a-cnn (3.1.2), architecture-evolution (3.2.1) | OK | SAM's image encoder produces hierarchical features. Student has this mental model. |
| Transfer learning / pretrained models | INTRODUCED | DEVELOPED | transfer-learning (3.2.3) | OK | SAM as a pretrained model for downstream tasks. Student understands feature extraction and fine-tuning. |
| Image segmentation as a task (per-pixel labeling) | INTRODUCED | MENTIONED | unet-architecture (6.3.1) transfer question | GAP | The U-Net lesson mentioned medical image segmentation as the architecture's origin, but segmentation as a task was never formally taught. Student needs to understand what segmentation produces (a mask) and how it differs from classification (one label per image) and detection (bounding boxes). |
| Object detection (bounding boxes) | MENTIONED | NOT TAUGHT | N/A | GAP | SAM uses bounding boxes as one prompt type. Student needs to recognize what a bounding box is, not understand detection architectures. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Image segmentation as a task (need INTRODUCED for understanding SAM's output) | Small (student understands per-pixel outputs from U-Net, spatial feature maps; just needs the framing as a distinct task with the terminology) | Dedicated section early in the lesson: "The Segmentation Problem." Classification assigns one label to an image. Detection draws bounding boxes. Segmentation labels every pixel -- "which pixels belong to this object?" Show a photograph with (a) classification label, (b) detection box, (c) segmentation mask. The mask is the richest output: exact object boundaries, arbitrary shapes, pixel-precise. Connect to U-Net: "You already know an architecture that produces spatial outputs at full resolution -- that is what segmentation needs." 3-4 paragraphs + a visual comparison. |
| Object detection / bounding boxes (need MENTIONED for understanding box prompts) | Trivial (bounding boxes are intuitively obvious) | Brief inline mention when introducing SAM's prompt types: "A bounding box is a rectangle around the object of interest -- just the rough location and extent, not the precise boundary. The box tells SAM approximately where to look; SAM fills in the precise mask." 1-2 sentences. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "SAM is a classification model that happens to work on images (like ImageNet classifiers)" | The student's primary vision experience is image classification (Series 3). They might map SAM onto that framework: input image, output label. SAM's name ("Segment Anything") does not immediately clarify what segmentation means. | SAM does NOT produce a class label. It produces a binary mask -- a grid of 0s and 1s the same size as the image, where 1 means "this pixel belongs to the object." Classification says "this is a dog." Segmentation says "these exact pixels are the dog." Show a dog photo with classification output ("dog, 97%") vs SAM output (precise silhouette mask). SAM does not even know what the object IS -- it just knows which pixels belong together. | Early, in the segmentation primer section. Establish the distinction before introducing SAM. |
| "SAM needs to be retrained for each new type of object (like fine-tuning a classifier for cats vs dogs)" | The student learned transfer learning with task-specific fine-tuning (3.2.3). They might assume SAM works the same way: pretrain on general data, then fine-tune for specific object types. | SAM works zero-shot on objects it has never seen. Give it a point on a flamingo, and it segments the flamingo -- even if no flamingo was in the training data. This is because SAM does not learn "what is a flamingo." It learns "given a point inside an object, which nearby pixels belong to the same thing." The prompt tells SAM WHERE to look; SAM's learned ability is figuring out WHERE the boundaries are. Any object with coherent boundaries can be segmented. | After explaining the promptable interface. Address explicitly: "SAM does not need to know what the object is." |
| "The image encoder runs once per prompt (every new click re-processes the whole image)" | The student might assume each prompt requires a full forward pass through the entire model. Since the ViT image encoder is the most expensive component, this would make interactive use impractical. | SAM's key design insight: the image encoder runs ONCE per image. The prompt encoder and mask decoder are lightweight (run in ~50ms). You can click 10 different points and get 10 different masks without re-running the expensive image encoder. This amortization is what makes interactive segmentation possible. If the encoder ran per prompt, real-time interaction would be impossible. | When explaining the architecture. Explicitly call out the asymmetric compute: heavy encoder (once) + light decoder (per prompt). |
| "SAM's mask decoder is a U-Net (it must be, since it produces spatial output)" | The student has strong U-Net associations from Series 6. Any architecture producing spatial masks might be mentally mapped to the U-Net pattern. | SAM's mask decoder is NOT a U-Net. It is a lightweight transformer decoder with only two layers. It uses cross-attention between prompt tokens and image embedding tokens, followed by upsampling via transposed convolutions. There is no encoder-decoder-skip-connection structure. The mask decoder is deliberately lightweight to enable real-time interaction. A full U-Net would be too slow for interactive use. | In the architecture section, when introducing the mask decoder. Explicitly contrast with U-Net. |
| "SAM 1, SAM 2, and SAM 3 are completely different architectures (you need to understand each one independently)" | The names suggest three separate models. The student might think understanding SAM 3 requires understanding two prior architectures from scratch. | SAM's evolution is additive, not replacement. SAM 1 established the core: ViT encoder + prompt encoder + mask decoder for images. SAM 2 ADDED a memory mechanism for video (keeping the same core). SAM 3 ADDED concept-level understanding with text/exemplar prompts (keeping both prior contributions). Each version extends the previous one. The core architecture from SAM 1 runs through all three. | In the evolution section. Frame as "SAM 1 + video = SAM 2, SAM 2 + concepts = SAM 3." |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Classification vs detection vs segmentation on the same photograph (a dog in a park) | Positive | Establish what segmentation IS by contrasting with tasks the student already knows. Classification: "dog." Detection: rectangle around the dog. Segmentation: precise pixel mask of the dog's silhouette, including the irregular ear shapes and tail. | Grounds the abstract task taxonomy in a single concrete image. The student can see that segmentation is the richest output -- it captures the exact boundary that classification and detection discard. The dog example is universally familiar. |
| SAM prompted with a point vs a box on the same image (a person holding a cup) | Positive | Show how different prompts produce different segmentation results from the same image. Point on the person's face -> mask of the person. Point on the cup -> mask of the cup. Box around the cup -> mask of the cup (more precise because the box constrains the search region). | Demonstrates the promptable interface concretely. The student sees that the same image encoder output serves multiple prompts, and that different prompt types provide different amounts of guidance. The ambiguity resolution (point on shoulder -- is it the person or the shirt?) introduces SAM's multi-mask output. |
| SAM 3 with text prompt "red car" on a street scene | Positive (stretch) | Show SAM 3's concept-level segmentation: a text prompt finds ALL instances of a concept, not just one clicked object. Multiple red cars get individual masks with unique IDs. | Demonstrates the SAM 1 -> SAM 3 evolution concretely. SAM 1 required clicking each car individually. SAM 3 finds them all from a text description. This is the "GPT-3 moment" comparison: from manual prompting to language-based prompting. |
| SAM on a texture with no clear object boundaries (a blue sky with gradual cloud gradients) | Negative | Show where SAM struggles: when there are no coherent boundaries, the model produces unreliable masks. SAM learns to find object boundaries; if the input has no clear boundaries (gradients, textures, abstract patterns), the prompt does not map to a meaningful segmentation. | Defines the boundary of SAM's capability. SAM is not magic -- it learned boundary detection from 1 billion masks of objects with clear edges. Inputs without clear boundaries produce ambiguous output. This prevents over-generalization of "segment ANYTHING." |

---

## Phase 3: Design

### Narrative Arc

The student has spent the entire course understanding models that look at images and produce labels (classification), generate images from noise (diffusion), or bridge images and text (CLIP/SigLIP). But there is a fundamental task they have never formally confronted: given an image, which exact pixels belong to a specific object? This is image segmentation, and it is harder than it sounds. Classification says "there is a dog." Detection draws a box around the dog. Segmentation traces the precise silhouette -- every pixel of the ear, the tail, the fur boundary against the grass. For decades, segmentation was either manual (a human draws the outline) or required training a specialized model for each type of object (a "cat segmenter," a "car segmenter"). Meta's SAM changed this by asking: what if one model could segment anything, guided only by a simple prompt? A click on the dog. A box around the cup. A text description like "red car." The architecture that makes this possible is elegant and familiar -- a ViT image encoder (the student knows ViTs), a prompt encoder that converts clicks/boxes/text into tokens, and a lightweight mask decoder that uses cross-attention (the student knows cross-attention) to fuse prompt information with image features. The student already has every architectural building block. SAM's contribution is not a new mechanism -- it is a new composition of known mechanisms, trained on an unprecedented dataset of 1 billion masks, creating what amounts to a "foundation model for segmentation." The evolution from SAM 1 (images only) to SAM 2 (images + video via memory) to SAM 3 (images + video + text/concept understanding) mirrors the broader trend the student has seen across the course: start with a focused capability, then extend it.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Verbal/Analogy** | "The universal cookie cutter" -- Traditional segmentation models are like having one cookie cutter per shape (star cutter, circle cutter, tree cutter). You need to buy a new cutter for every new shape. SAM is a programmable cookie cutter that adjusts its shape based on what you point at. You tell it where and it figures out the boundary. SAM 3 adds: you can also describe what you want ("cut out all the stars") and it finds them. | Maps to a familiar physical experience. The "one cutter per shape" limitation mirrors training one model per object class. The "programmable" aspect captures the promptable interface. Extends naturally to SAM 3's text prompting. |
| **Visual (inline diagrams)** | (1) Three-panel comparison: classification output, detection output, segmentation output on the same photograph. (2) SAM architecture diagram: image -> ViT encoder -> image embedding -> [prompt tokens via cross-attention] -> mask decoder -> binary mask. Show the encoder as the heavy component (run once) and the decoder as lightweight (run per prompt). (3) SAM evolution timeline: SAM 1 (images) -> SAM 2 (+ video memory) -> SAM 3 (+ concepts/text). | Three distinct visuals for three concepts. Panel 1 establishes the task. Panel 2 shows the architecture with familiar components labeled. Panel 3 shows the evolution as additive layers. All ground abstract concepts in concrete spatial representations. |
| **Symbolic/Structural** | Architecture breakdown with tensor shapes: Image (1024x1024x3) -> ViT encoder -> Image embedding (64x64x256) -> Prompt encoder (point/box/text -> prompt tokens) -> Mask decoder (cross-attention between prompt tokens and image tokens) -> Mask (256x256 upsampled to original resolution). Shape annotations at each stage, following the dimension-tracking pattern the student knows from CNN and transformer lessons. | The student's strongest technical modality. Tensor shapes make the architecture concrete and verifiable. The dimension tracking pattern is familiar from building-a-cnn, the-transformer-block, and unet-architecture. |
| **Concrete example** | Walk through a specific SAM interaction: (1) User loads a photo of a desk with a laptop, coffee mug, and notebook. (2) ViT processes the image once (takes ~150ms). (3) User clicks a point on the coffee mug. (4) Point is encoded as a positional embedding + type embedding. (5) Mask decoder cross-attends between the point token and image features. (6) Three candidate masks produced (mug only, mug + saucer, mug + saucer + table region). (7) Confidence scores rank them. (8) User picks the mug-only mask. Total decoder time: ~50ms. | Traces the complete interaction loop with specific timing and outputs. The multi-mask output is a critical design choice that the student needs to see concretely (three masks, not one). The timing makes the amortization argument visceral. |
| **Intuitive** | The "of course" chain: (1) "Of course you would separate the heavy encoder from the light decoder -- you want to click around exploring, not wait 150ms per click." (2) "Of course you would produce multiple masks -- a point on a shirt button could mean the button, the shirt, or the person." (3) "Of course the next step after 'click to segment' is 'describe to segment' -- that is the same trajectory language models followed." | Three "of course" moments that make the design decisions feel inevitable rather than arbitrary. Each connects a SAM design choice to an intuition the student already has (interactive UX, ambiguity, language interfaces). |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. Promptable segmentation (the core concept: one model that segments any object given a spatial or text prompt -- this is the paradigm shift)
  2. SAM's three-component architecture (ViT encoder + prompt encoder + lightweight mask decoder -- a specific composition of familiar pieces)
  3. The data engine approach (human-in-the-loop annotation at unprecedented scale to create the foundation dataset -- a different kind of innovation)
- **Previous lesson load:** The SigLIP 2 lesson was BUILD (2 new concepts). If the student does SAM 3 first, they arrive from Series 7 completion (the last lesson was BUILD).
- **This lesson's load:** BUILD -- appropriate. Every architectural component is familiar (ViT, cross-attention, encoder-decoder). The novelty is in the composition and the task domain (segmentation), not in new mechanisms. Segmentation needs a dedicated primer section but is conceptually accessible.
- **Self-contained lesson note:** As a Special Topics lesson, this must work even if the student's CNN/transformer knowledge has faded. Recap sections are heavier than structured series. The segmentation primer is essential since the student has never been formally taught the task.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| ViT / patchify (7.4.2) | SAM 1 uses a MAE-pretrained ViT-H as its image encoder. Same "tokenize the image" pattern. SAM 3 uses a Perception Encoder but the principle is the same: convert image to spatial tokens. |
| Cross-attention (6.3.4) | SAM's mask decoder uses cross-attention between prompt tokens and image tokens. Same formula the student knows deeply (Q from prompts, K/V from image features). Directly extends "same formula, different source for K and V." |
| U-Net encoder-decoder (6.3.1) | SAM's mask decoder is explicitly NOT a U-Net -- it is much lighter. But the student's U-Net knowledge helps understand what SAM's decoder does NOT need (heavy encoder-decoder with skip connections) and why (the image encoder already extracted features). |
| Transfer learning (3.2.3) | SAM is a foundation model that can be used as a pretrained backbone for downstream tasks. Same "hire experienced, train specific" pattern. SAM's ViT encoder features transfer to many vision tasks. |
| CNN hierarchical features (3.1-3.2) | The image encoder produces hierarchical features -- edges at early layers, object parts at later layers. Same feature hierarchy the student learned in Series 3. |
| CLIP shared embedding space (6.3.3) | SAM 3's text prompting uses vision-language embeddings similar to CLIP. The concept of text and image in the same space directly carries over. |
| Foundation model trajectory (Series 4-5) | GPT showed that one model pretrained on broad data can handle many tasks via prompting. SAM is the same idea for segmentation: one model, many objects, prompted interaction. |
| Data scaling (multiple) | SAM's SA-1B dataset (1 billion masks) echoes the data scaling the student has seen: CLIP's 400M pairs, GPT-3's massive text corpus. Scale enables generalization. |

**Analogies from prior lessons that can be extended:**
- "Hire experienced, train specific" -> SAM is the ultimate experienced hire for segmentation: trained on 1B masks, applies to any object
- "Tokenize the image" -> SAM 1 uses ViT which tokenizes the image, SAM 3's Perception Encoder does the same
- "Same formula, different source for K and V" -> SAM's mask decoder uses cross-attention with Q from prompts, K/V from image embedding
- "Architecture encodes assumptions about data" -> SAM's asymmetric design (heavy encoder, light decoder) encodes the assumption that you will prompt the same image many times

**Analogies from prior lessons that could be misleading:**
- "Bottleneck decides WHAT, skips decide WHERE" from U-Net could lead students to expect SAM's decoder follows the same pattern. Need to clarify: SAM's mask decoder is a lightweight transformer, not a U-Net.
- "Two encoders, one shared space" from CLIP could suggest SAM 3's text understanding works identically to CLIP. Need to clarify: SAM 3 uses a unified Perception Encoder that fuses vision and language, rather than two separate encoders.

### Scope Boundaries

**This lesson IS about:**
- What image segmentation is and how it differs from classification and detection
- SAM's core insight: promptable segmentation as a foundation model approach
- SAM 1's three-component architecture (ViT encoder + prompt encoder + mask decoder) and why each component exists
- The prompt types (points, boxes, masks, and SAM 3's text/exemplar prompts)
- The amortized computation design (heavy encoder once, light decoder per prompt)
- The SA-1B dataset and the data engine concept (human-in-the-loop annotation at scale)
- SAM 2's extension to video (memory mechanism, streaming architecture)
- SAM 3's extension to concept-level understanding (text prompts, open-vocabulary, all-instance detection)
- How SAM connects to the broader vision ecosystem

**This lesson is NOT about:**
- Implementing SAM from scratch or training a segmentation model
- The mathematical details of the mask decoder's loss function (focal loss, dice loss)
- DETR or transformer-based object detection architectures in depth
- Semantic vs instance vs panoptic segmentation in formal detail (INTRODUCED only)
- SAM 3D or 3D reconstruction
- EfficientSAM, MobileSAM, or efficiency variants
- The full SA-Co dataset construction methodology
- Benchmark comparisons beyond illustrative examples
- Fine-tuning SAM for specific domains

**Target depths:**
- Promptable segmentation concept (one model segments any object given a prompt): DEVELOPED (can explain the paradigm, why it works, how it differs from task-specific segmentation)
- SAM's three-component architecture (ViT encoder + prompt encoder + mask decoder): DEVELOPED (can trace data flow, explain each component's role, articulate design rationale)
- Image segmentation as a task (per-pixel labeling, contrast with classification/detection): INTRODUCED (can explain what segmentation produces and why it is harder than classification)
- SA-1B dataset and data engine: INTRODUCED (knows the scale, the three-stage annotation process, why the data engine was necessary)
- SAM 2 video extension (memory mechanism, streaming): INTRODUCED (knows the concept and why it matters, not implementation details)
- SAM 3 concept-level segmentation (text prompts, open-vocabulary): INTRODUCED (knows the capability and how it extends SAM 1/2, not architectural details of the Perception Encoder)

### Lesson Outline

**1. Context + Constraints**
What this lesson is about: how SAM (Segment Anything Model) brings the "foundation model" approach to image segmentation -- one model that segments any object in any image, guided by a prompt. What we are NOT doing: implementing SAM, training a segmentation model, or diving into detection architectures. The student will understand SAM's architecture, the promptable interface, and the evolution from SAM 1 to SAM 3.

Self-contained note: This is a Special Topics lesson. The student may have covered ViTs, cross-attention, and U-Nets weeks or months ago. Prerequisites are recapped inline.

**2. Recap (heavier than structured series)**
Brief refresher on the building blocks SAM uses:

1. **ViT image encoder:** Patchify the image, process patches as tokens with a standard transformer. Callback to "tokenize the image" from DiT. The image encoder produces a spatial grid of feature vectors.

2. **Cross-attention:** Q from one source, K/V from another. "Same formula, different source for K and V." SAM uses this to fuse prompt information with image features.

3. **Encoder-decoder pattern:** The student knows U-Net's encoder-decoder with skip connections. SAM's decoder is lighter, but the concept of producing spatial output from encoded features is familiar.

This recap is 4-6 paragraphs, not a re-teaching. The goal is reactivation, connecting known concepts to what comes next.

**3. The Segmentation Problem (gap resolution)**
Type: Dedicated section for the MENTIONED -> INTRODUCED gap.

Present the computer vision task hierarchy on a single photograph:
- **Classification:** "This image contains a dog." One label for the whole image.
- **Detection:** "There is a dog here." A bounding box (rectangle) around the dog.
- **Segmentation:** "These exact pixels are the dog." A binary mask -- every pixel labeled as dog or not-dog.

Visual: three-panel comparison showing the same photograph with classification label, detection box, and segmentation mask.

Key insight: segmentation is the richest output. It captures the exact boundary -- the irregular ear shape, the tail curving behind the leg, the nose against the grass. A bounding box includes background pixels; a mask does not.

Connect to existing knowledge: "You already know an architecture that produces spatial outputs -- the U-Net produces a grid at full resolution. Segmentation is the task that needs this kind of spatial output."

Brief mention of segmentation variants (semantic, instance, panoptic) at MENTIONED depth. SAM does instance-level segmentation: each object gets its own mask.

**4. Hook (the segmentation bottleneck)**
Before SAM, segmentation required:
- A specialized model per object category (train one model for "car segmentation," another for "person segmentation")
- OR expensive per-pixel annotation for every new category
- No general-purpose segmentation model existed

Draw the parallel to language: before GPT-3, NLP had task-specific models (one for sentiment, one for translation, one for summarization). GPT-3 showed that one large pretrained model could handle any text task via prompting. SAM asks: can we do the same for segmentation?

GradientCard: "What if one model could segment anything -- a dog, a car, a cell in a microscope image, an object it has never seen -- guided only by a click or a text description?"

**5. Explain Part 1 -- SAM 1: The Architecture**
The core architecture has three components, each familiar:

**Image Encoder (ViT-H):**
- A MAE-pretrained ViT-H (Huge) processes the image ONCE
- Input: 1024x1024x3 image
- Output: 64x64x256 image embedding (a spatial grid of 256-dim feature vectors)
- This is the heavy computation (~150ms on GPU). Runs once per image.
- Callback: "Same 'tokenize the image' idea from the DiT lesson. Patches become tokens, transformer processes them, output is a grid of features."

**Prompt Encoder:**
- Converts user prompts into tokens the mask decoder can process
- Point prompts: positional encoding + learned type embedding (foreground vs background point)
- Box prompts: encoded as two corner points (top-left + bottom-right) with positional embeddings
- Mask prompts: downsampled and embedded with lightweight convolutions
- Text prompts (SAM 3): text encoded via a language-vision encoder
- Output: a small set of prompt tokens (typically 1-4 tokens)
- Lightweight: runs in milliseconds

**Mask Decoder:**
- A lightweight transformer decoder (only 2 decoder layers)
- Uses cross-attention: Q from prompt tokens, K/V from image embedding tokens
- Produces a low-resolution mask prediction (256x256) that is upsampled to the original image resolution
- Also predicts a confidence score (IoU estimate) for each mask
- Key feature: outputs MULTIPLE candidate masks (typically 3) to handle ambiguity
- Lightweight: ~50ms per prompt

**Architectural diagram:** show the three components with tensor shapes at each stage. Label the ViT as "heavy, runs once" and the prompt encoder + mask decoder as "lightweight, runs per prompt."

**The amortization insight:** This asymmetric design is the key to interactive use. Click on the coffee mug: 50ms. Click on the laptop: 50ms. Click on the notebook: 50ms. The 150ms image encoding is paid once; every subsequent interaction costs only 50ms. This is what makes SAM interactive.

Callback: "Architecture encodes assumptions about data. SAM's architecture encodes the assumption that you will query the same image multiple times with different prompts."

**6. Explain Part 2 -- The Promptable Interface**
What makes SAM a foundation model (not just another segmentation network):

**Point prompts:** Click a single point on the object. SAM infers the full mask. The simplest interaction.

**Box prompts:** Draw a rectangle around the object. More constrained than a point -- reduces ambiguity about which object you mean.

**Mask prompts:** Provide an initial rough mask, SAM refines it. Enables iterative refinement.

**Multi-mask output (critical design choice):** A single point click is inherently ambiguous. Click on a person's shirt button: did you mean the button, the shirt, or the whole person? SAM handles this by predicting 3 masks at different granularities (part, object, scene-level) with confidence scores. The user (or downstream system) picks the right one.

Concrete example: walk through the desk scenario (laptop, coffee mug, notebook). Point on the mug handle -> three masks: (1) the handle only, (2) the mug, (3) the mug + saucer. Scores: 0.95, 0.98, 0.72. The mug mask wins.

**7. Check 1 (predict-and-verify)**
Questions:
- You load a 4000x3000 photograph into SAM. How many times does the ViT image encoder run? (Once -- the embedding is reused for all prompts.)
- You click 5 different objects in the image. Approximately how long does this take? (~150ms for encoding + 5x50ms for decoding = ~400ms total, not 5x200ms.)
- You click a point on the boundary between two touching objects (a cup sitting on a plate). What does SAM output? (Three masks at different granularities: likely the cup alone, the plate alone, or both together, with confidence scores.)

**8. Explain Part 3 -- The SA-1B Data Engine**
SAM's training data is unprecedented: 1.1 billion masks on 11 million images. But this data did not exist -- Meta had to create it.

**The data engine (three stages):**
1. **Assisted-manual:** Human annotators use an early SAM to help draw masks. ~120K images, 4.3M masks. Slow but high-quality.
2. **Semi-automatic:** SAM automatically suggests masks; humans verify and correct. Faster annotation, growing dataset.
3. **Fully automatic:** SAM processes images with a grid of point prompts, generating masks without human input. Scales to 11M images.

**Key insight:** The model improves the data, and better data improves the model. This virtuous cycle -- the "data engine" -- is how SAM scaled to 1 billion masks. The model that creates the annotations gets better as it trains on the annotations it helped create.

Connect to prior experience: "This is a different kind of scaling than what you have seen in language models (just add more text from the internet). SAM's data required active creation through a human-AI partnership."

**Brief comparison:** SA-1B has 400x more masks than any prior segmentation dataset. COCO (a standard benchmark) has ~860K masks on 160K images. SA-1B: 1.1B masks on 11M images.

**9. Explain Part 4 -- SAM 2: Video Extension**
SAM 1 processes single images. SAM 2 (2024) extends to video.

**The problem:** In video, an object appears across many frames. Clicking on every frame is impractical. The model needs to TRACK the object across time.

**The solution -- memory mechanism:**
- SAM 2 processes video frames sequentially (streaming architecture)
- A memory bank stores information about segmented objects from previous frames
- On each new frame, the model attends to its memory of the target object (what it looked like before, where it was)
- If the object becomes occluded (hidden behind something), the memory allows re-identification when it reappears

**Interactive refinement:** The user can click on any frame to correct the segmentation. SAM 2 propagates corrections forward AND backward through the video.

**Practical result:** 3x fewer interactions than previous video segmentation methods. 8x speedup for video annotations.

Connect: "SAM 2's memory mechanism is a form of cross-attention -- the current frame's features attend to stored features from previous frames. Same formula you know, applied across time instead of across spatial positions."

**10. Explain Part 5 -- SAM 3: Concepts and Language**
SAM 3 (November 2025) adds the final piece: language understanding.

**The limitation of SAM 1/2:** You must SHOW the model what to segment -- click a point, draw a box. You cannot SAY what to segment. If you want to segment all red cars in a street scene, you must click each car individually.

**SAM 3's innovation -- concept-level segmentation:**
- Accept text prompts: "red car," "shipping container," "striped umbrella"
- Accept exemplar prompts: provide an image of the target concept
- Find ALL instances of the concept in the image or video, each with a unique mask and ID
- Open-vocabulary: not limited to a fixed set of categories

**Architecture evolution:**
- SAM 3 uses a unified Perception Encoder that fuses visual features with text/exemplar embeddings
- A DETR-style detector finds all instances of the prompted concept
- The SAM 2 memory/tracking module handles video consistency
- A global presence head determines if the concept exists before trying to localize it ("recognition before localization")

**Performance:** 840M parameters (~3.4 GB). 30ms inference per image with 100+ detected objects on H200 GPU. Doubles the segmentation quality (cgF1 score) compared to prior systems on the SA-Co benchmark.

**SA-Co dataset:** 5.2M images, 52.5K videos, 4M+ unique noun phrases, ~1.4B masks. Created with a data engine combining AI annotators, human reviewers, and LLM-generated concept ontologies.

**The trajectory:** SAM 1 (click to segment one object) -> SAM 2 (click to segment and track across video) -> SAM 3 (describe to segment all instances in images and video). Each version is additive: SAM 3 still supports point/box/mask prompts from SAM 1.

**11. Check 2 (transfer question)**
A colleague is building a wildlife monitoring system. Cameras in a national park capture thousands of images daily. They want to automatically segment and count all elephants in each image.

Questions:
- Which version of SAM would they need, and why? (SAM 3 -- they need text prompting "elephant" to find all instances automatically. SAM 1/2 would require clicking each elephant manually.)
- Why is SAM's image encoder amortization less important for this use case? (It is automated batch processing, not interactive use. Each image is processed once with one prompt. The per-prompt efficiency matters less than total throughput.)
- If the cameras also capture video, what SAM 2 capability becomes useful? (Memory mechanism for tracking elephants across frames -- an elephant partially hidden behind a tree can be tracked because the memory stores what it looked like before occlusion.)

**12. Elaborate -- SAM in the Vision Ecosystem**
SAM's impact extends beyond segmentation:

**As a data annotation tool:** SAM dramatically accelerates the creation of segmentation datasets for any domain. What took hours per image now takes seconds. This changes the economics of training specialized models.

**As a foundation model component:** SAM's image encoder (trained on 1B masks) produces rich visual features that transfer to other vision tasks. Like using CLIP or SigLIP as a vision backbone, SAM features encode fine-grained spatial understanding.

**The foundation model pattern:** The student has now seen this pattern three times:
- GPT: one model for all language tasks, prompted with text
- CLIP/SigLIP: one model for vision-language alignment, prompted with images or text
- SAM: one model for all segmentation tasks, prompted with points, boxes, or text

**Connection:** "Every foundation model in this course follows the same recipe: train on a massive, diverse dataset using a task that forces the model to learn general representations. Then use prompting to steer the model to specific tasks without retraining."

**13. Summarize**
Key takeaways:
1. Image segmentation produces per-pixel masks -- richer output than classification (one label) or detection (bounding box)
2. SAM is a promptable segmentation foundation model: ViT image encoder (heavy, once) + prompt encoder (point/box/text) + lightweight mask decoder (fast, per prompt)
3. The asymmetric compute design (heavy encoder once, light decoder per prompt) enables interactive use
4. Multi-mask output handles prompt ambiguity (a click could mean part, object, or scene)
5. The SA-1B data engine (1B masks via human-AI partnership) enabled the foundation model approach
6. SAM evolved additively: SAM 1 (images) -> SAM 2 (+ video memory) -> SAM 3 (+ text/concept understanding)
7. SAM follows the foundation model pattern: massive pretraining + prompting, same trajectory as GPT and CLIP

Echo the mental model: "SAM is the universal cookie cutter. Traditional segmentation needs one cutter per shape. SAM adjusts its shape based on what you point at -- or, with SAM 3, based on what you describe."

**14. Next Step**
"SAM showed us how the 'foundation model for X' pattern applies to segmentation. Every domain in vision is following this same trajectory: build a foundation model, train on unprecedented data, and make it promptable. If there is a topic you want to explore next -- a specific model, technique, or domain -- the Special Topics series is here for that."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (segmentation gap resolved with dedicated primer, detection gap resolved inline)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (segmentation MENTIONED->INTRODUCED, detection NOT TAUGHT->MENTIONED)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2-3: promptable segmentation, SAM architecture, data engine)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-21 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 5
- Polish: 3

### Verdict: NEEDS REVISION

No critical structural failures — the lesson is well-organized, motivates before explaining, and maintains good flow. However, there is one critical finding (a missing modality that was planned), five improvement findings that would meaningfully strengthen the lesson, and three polish items. The critical finding alone does not warrant MAJOR REVISION because the lesson is still usable without it, but combined with the improvement findings, a revision pass is warranted.

### Findings

#### [CRITICAL] — Cookie cutter analogy is buried in an aside, not developed as a primary modality

**Location:** Row.Aside TipBlock "The Cookie Cutter Analogy" (after the "SAM does not need to know what the object is" section, around line 612)
**Issue:** The planning document identifies the "universal cookie cutter" analogy as the primary **Verbal/Analogy modality** — one of the five planned modalities for the core concept. It is supposed to be a developed analogy: "Traditional segmentation models are like having one cookie cutter per shape (star cutter, circle cutter, tree cutter). You need to buy a new cutter for every new shape. SAM is a programmable cookie cutter that adjusts its shape based on what you point at. You tell it where and it figures out the boundary. SAM 3 adds: you can also describe what you want ('cut out all the stars') and it finds them." In the built lesson, this entire analogy is compressed into a 3-sentence aside block. It never appears in the main content flow. The mental model echo at the end references it ("SAM is the universal cookie cutter"), but the student encounters it only as sidebar text that many students skim or skip.
**Student impact:** The student's primary verbal/analogy modality for the core concept is effectively optional reading. The lesson relies heavily on the structural/visual and concrete-example modalities but under-delivers on the verbal/analogy channel. The summary's "mental model echo" references a cookie cutter analogy the student may never have internalized.
**Suggested fix:** Move the cookie cutter analogy into the main content, ideally in the hook section ("The Segmentation Bottleneck") or just after it. Develop it as a full paragraph in Row.Content: one cutter per shape = one model per object class. SAM = a programmable cutter. SAM 3 = you can also describe the shape you want. Then reference it again briefly when introducing SAM 3's text prompts. The aside can keep a condensed version, but the main content must carry the analogy.

#### [IMPROVEMENT] — Concrete worked example (desk scenario) is split across sections and not fully traced

**Location:** Amortization insight section (line 438-466) and Multi-mask output section (line 526-583)
**Issue:** The planning document specifies a single, cohesive concrete example: "Walk through a specific SAM interaction: (1) User loads a photo of a desk with a laptop, coffee mug, and notebook. (2) ViT processes the image once (takes ~150ms). (3) User clicks a point on the coffee mug. (4) Point is encoded as a positional embedding + type embedding. (5) Mask decoder cross-attends between the point token and image features. (6) Three candidate masks produced... (7) Confidence scores rank them. (8) User picks the mug-only mask." In the built lesson, the desk scenario is used for the amortization argument (click mug, click laptop, click notebook — timing), and a separate mug-handle example is used for the multi-mask output. Neither traces the full pipeline end-to-end. Steps 4 and 5 (prompt encoding and cross-attention) are never walked through concretely — the student is told they happen but never sees how prompt tokens interact with image tokens for a specific click.
**Student impact:** The student understands the timing argument and the multi-mask concept separately but never traces a single prompt from click to mask through all three components. The "symbolic/structural" modality (tensor shapes through the pipeline) is present in the GradientCards and Mermaid diagram but never grounded in a specific interaction. The student can recite the components but may not have a unified mental model of data flow.
**Suggested fix:** Add a dedicated "Tracing a Single Prompt" subsection after the architecture cards and before the promptable interface section. Use the desk/mug example: (1) image encoded, show the 64x64x256 grid, (2) user clicks the mug, (3) the click becomes a positional embedding + foreground type embedding = 1 prompt token, (4) mask decoder cross-attends — Q from the prompt token asks "which image regions match this location?", K/V from the image embedding answer, (5) output: 3 masks at different granularities with scores. This unifies the timing, architecture, and multi-mask concepts into one coherent trace.

#### [IMPROVEMENT] — Third positive example (SAM 3 text prompt "red car") is asserted but never developed

**Location:** SAM 3 section, GradientCard "SAM 3's Innovation" (line 881-900)
**Issue:** The planning document lists "SAM 3 with text prompt 'red car' on a street scene" as the third positive example, meant to "show SAM 3's concept-level segmentation: a text prompt finds ALL instances of a concept, not just one clicked object. Multiple red cars get individual masks with unique IDs." In the built lesson, the text prompt capability is listed as a bullet point ("Accept text prompts: 'red car,' 'shipping container,' 'striped umbrella'") but the "red car" example is never developed as a scenario. The student is told SAM 3 can do this but never sees it walked through with concrete detail — how many cars were found, what the masks look like, how IDs work.
**Student impact:** The student understands SAM 3 can accept text prompts in the abstract but lacks a concrete grounding. The distinction between SAM 1 (click each car) and SAM 3 (say "red car," get all of them) is stated but not made visceral through a worked example.
**Suggested fix:** After the SAM 3 innovation card, add a brief concrete scenario: "Imagine a street scene with 4 red cars. With SAM 1, you click each car — 4 clicks, 4 masks. With SAM 3, you type 'red car' — one prompt, 4 masks, each with a unique instance ID. SAM 3 found them all." This can be 2-3 paragraphs or a simple ComparisonRow. It makes the capability concrete rather than listed.

#### [IMPROVEMENT] — Misconception #1 (SAM is a classification model) is addressed implicitly but not called out

**Location:** The Segmentation Problem section (lines 159-247)
**Issue:** The planning document identifies a misconception: "SAM is a classification model that happens to work on images." The planned negative example is: "SAM does NOT produce a class label. It produces a binary mask... Show a dog photo with classification output ('dog, 97%') vs SAM output (precise silhouette mask). SAM does not even know what the object IS — it just knows which pixels belong together." The segmentation primer section does establish the classification/detection/segmentation taxonomy clearly (GradientCards comparing the three), which implicitly addresses this. However, the explicit negative framing ("SAM does NOT produce a class label... SAM does not even know what the object IS") appears much later in the "SAM does not need to know what the object is" section (lines 588-618). The misconception is never explicitly called out as a misconception — it is addressed through the general flow of explanation.
**Student impact:** Students who enter with the classification mental model may not realize their assumption is wrong until much later. The segmentation primer teaches what segmentation IS but does not explicitly say "SAM is not doing classification." The explicit correction comes after the full architecture is presented.
**Suggested fix:** In the segmentation primer section, after the three GradientCards, add one sentence of explicit negative framing: "SAM does not output a class label like 'dog' with a confidence score. It outputs a binary mask — which pixels belong to the object — without knowing what the object is. It knows boundaries, not categories." This plants the correction early, before the student builds the wrong mental model.

#### [IMPROVEMENT] — Misconception #5 (SAM versions are completely different) could be addressed earlier

**Location:** SAM evolution diagram (lines 936-953)
**Issue:** The planning document identifies the misconception "SAM 1, SAM 2, and SAM 3 are completely different architectures (you need to understand each one independently)" and plans to address it in the evolution section with the framing "SAM 1 + video = SAM 2, SAM 2 + concepts = SAM 3." The lesson does eventually address this with the Mermaid diagram and the text "SAM 3 still supports point/box/mask prompts from SAM 1. Each version extends the previous one." However, this comes very late — after the student has already read through SAM 1, SAM 2, and SAM 3 as separate sections. By this point, the student may have already formed the "three separate models" mental model.
**Student impact:** The student reads three distinct sections (SAM 1, SAM 2, SAM 3) and might think they are learning three separate architectures. The "additive" framing comes after the fact rather than setting expectations up front.
**Suggested fix:** Add a brief framing statement before the SAM 2 section (or at the end of the SAM 1 architecture section): "SAM's evolution is additive, not replacement. SAM 1 established the core. SAM 2 adds to it. SAM 3 adds further. The core architecture you just learned carries through all three versions." This sets the right expectation before the student reads SAM 2 and SAM 3.

#### [IMPROVEMENT] — "Of Course" moments are in an aside, not woven into the main explanation flow

**Location:** Row.Aside InsightBlock "Three 'Of Course' Moments" (lines 566-583)
**Issue:** The planning document identifies the "Intuitive" modality as three "of course" moments that make design decisions feel inevitable. In the built lesson, all three are grouped together in a single aside block. This means (a) they are sidebar text that students may skim, and (b) they appear at one location instead of at the three natural points where each insight would land. The first "of course" (separate encoder/decoder) should appear at the amortization section. The second (multi-mask) should appear at the multi-mask section. The third (click to describe trajectory) should appear at the SAM 3 section.
**Student impact:** The intuitive "of course" moments lose their power when grouped together rather than arriving at the moment the student is thinking about each design choice. The student reads about the amortization but the "of course" that makes it click is elsewhere.
**Suggested fix:** Distribute the three "of course" moments to their natural locations in the main content. Each can be a single sentence at the end of its relevant paragraph: "Of course you would separate the heavy encoder from the light decoder..." after the amortization explanation. "Of course you would produce multiple masks..." after introducing multi-mask output. "Of course the next step is 'describe to segment'..." when introducing SAM 3. The grouped aside can be replaced or kept as a synthesis.

#### [POLISH] — MAE acronym is not expanded

**Location:** Image Encoder GradientCard (line 328): "A MAE-pretrained ViT-H (Huge)"
**Issue:** MAE (Masked Autoencoder) is used without expansion. The student has not encountered MAE in previous lessons. While the student does not need to understand MAE in depth (it is not a target concept), an unexpanded acronym creates a small moment of confusion.
**Student impact:** Minor. The student reads "MAE-pretrained" and does not know what MAE stands for. They can infer it is a pretraining method, but the unexpanded acronym is a speedbump.
**Suggested fix:** Expand to "MAE (Masked Autoencoder)-pretrained ViT-H" or add a brief parenthetical: "a ViT-H pretrained with MAE (Masked Autoencoders — a self-supervised method where the model learns to reconstruct masked image patches)."

#### [POLISH] — DETR-style detector mentioned without context

**Location:** SAM 3 architecture evolution bullet list (line 911): "A DETR-style detector finds all instances of the prompted concept"
**Issue:** The planning document explicitly lists DETR as NOT covered: "NOT: DETR or transformer-based object detection architectures in depth." The scope boundaries say the same. Yet the built lesson references "a DETR-style detector" in the SAM 3 architecture description without any explanation of what DETR is. The student has not encountered DETR. The planning doc's prerequisite table confirms object detection is NOT TAUGHT.
**Student impact:** Minor — the student reads "DETR-style" as an opaque reference. They do not know what DETR is, so this bullet point communicates less than it should. It is not confusing enough to be critical (the student can skip it), but it is an unexplained term.
**Suggested fix:** Either remove the DETR reference ("A transformer-based detector finds all instances...") or add a brief inline note ("A DETR-style detector — a transformer-based approach to finding objects in images — finds all instances..."). Given the scope boundary, the simpler fix is to drop the DETR name.

#### [POLISH] — IoU abbreviation not expanded

**Location:** Mask Decoder GradientCard (line 389): "Predicts a confidence score (IoU estimate) for each mask"
**Issue:** IoU (Intersection over Union) is used with only "IoU estimate" as context. The student has not been taught IoU. It is a standard metric for segmentation quality, but the student does not know what it measures.
**Student impact:** Very minor. The student reads "IoU estimate" and infers it is some kind of confidence metric. The parenthetical helps, but the abbreviation is a small speedbump.
**Suggested fix:** Expand to "a confidence score (IoU — Intersection over Union — which measures how well the predicted mask overlaps with the true object)" or simplify to just "a confidence score estimating mask quality."

### Review Notes

**What works well:**
- The segmentation primer is excellent. The three GradientCards (classification/detection/segmentation) effectively establish the task for a student who has never been formally taught it. The connection to U-Net ("you already know an architecture that produces spatial outputs") is smooth.
- The architecture section is clear and well-structured. The three GradientCards for the three components, followed by the Mermaid diagram, give the student multiple angles on the same architecture.
- The hook is compelling. The GPT-3 parallel ("before GPT-3, one model per NLP task; before SAM, one model per object category") is a strong motivator that connects to the student's existing knowledge.
- The ComparisonRow for SAM 1 vs SAM 2 is effective and scannable.
- The transfer question (wildlife monitoring) is well-designed — it tests understanding of SAM 3's capabilities, the amortization tradeoff, and video tracking in a single coherent scenario.
- The "Not a U-Net" WarningBlock and "Not Two Separate Encoders" WarningBlock correctly address two planned misconceptions in well-placed locations.
- Em dash usage is correct throughout (no spaces).
- All interactive elements (details/summary) have cursor-pointer styling.
- The lesson stays within its stated scope boundaries.

**Patterns to watch:**
- Several planned modalities and examples are present but relegated to asides or compressed into bullet points rather than developed in the main content. The lesson reads well at a surface level but may not achieve the intended depth on the intuitive and verbal/analogy channels.
- The lesson is predominantly expository (explain, explain, explain) with two check-your-understanding breaks. The checks are good quality, but the lesson is long enough that the student might benefit from a micro-interaction or concrete trace earlier in the architecture section.

**No notebook expected:** The planning document does not specify exercises or a Practice section, so the absence of a notebook is not a finding.

---

## Review — 2026-02-21 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

All iteration 1 findings were properly addressed. The critical finding (cookie cutter analogy buried in aside) is now resolved — the analogy is developed across two full paragraphs in main content within the hook section, covering the "one cutter per shape" limitation, the "programmable cutter" concept, and SAM 3's "describe what you want" extension. The mental model echo in the summary references an analogy the student has now genuinely internalized. The unified end-to-end data-flow trace (Steps 1-5 tracing a mug click) is a strong addition that ties the architecture together. The SAM 3 "red car" ComparisonRow makes the SAM 1 vs SAM 3 distinction concrete. The "of course" moments are now distributed to their natural locations (amortization, multi-mask, SAM 3 trajectory). The explicit "SAM is not classification" correction is planted early in the segmentation section. The additive framing note before SAM 2 sets the right expectation. All three polish items (MAE expanded, DETR removed, IoU expanded) are fixed. However, two improvement-level issues remain from a fresh reading, and two new polish items surfaced.

### Findings

#### [IMPROVEMENT] — End-to-end trace and multi-mask section both use the mug example with slightly inconsistent details

**Location:** "Tracing a single prompt" section (lines 504-579) and "Multi-mask output" section (lines 638-688)
**Issue:** The end-to-end trace (added in iteration 1 fixes) walks through clicking the coffee mug and ends with three masks: handle (0.85), whole mug (0.98), mug + saucer (0.72). Then the multi-mask output section, several rows later, presents a nearly identical scenario: "you click on the handle of a coffee mug on a desk. SAM returns three masks" with the same scores (handle 0.85, whole mug 0.98, mug + saucer 0.72). The student reads what is essentially the same worked example twice — same object, same masks, same confidence scores. The first time (in the trace) it demonstrates end-to-end data flow. The second time (in multi-mask) it demonstrates ambiguity resolution. These are different pedagogical goals, but using identical details makes the second encounter feel redundant rather than reinforcing.
**Student impact:** The student hits deja vu. The second mug example does not add new information — it restates what was already traced. The multi-mask concept deserves its own fresh example to prove it generalizes beyond the mug scenario. A student might also wonder if this is an editing artifact.
**Suggested fix:** Keep the mug example in the end-to-end trace (where it is well-integrated into the pipeline walkthrough). For the multi-mask section, switch to a different ambiguous scenario — the shirt button example is already mentioned in the preceding paragraph ("Click on a person's shirt button — did you mean the button, the shirt, or the whole person?"). Develop that into the concrete multi-mask example instead: Mask 1: button only (0.82), Mask 2: the shirt (0.96), Mask 3: the whole person (0.68). This proves the multi-mask concept generalizes beyond mugs and avoids redundancy.

#### [IMPROVEMENT] — SAM 2 section introduces "3x fewer interactions" claim without grounding

**Location:** ComparisonRow for SAM 1 vs SAM 2 (line 967): "3x fewer interactions than prior methods"
**Issue:** The SAM 2 comparison table includes the claim "3x fewer interactions than prior methods" as a bullet point. The planning document mentions "3x fewer interactions than previous video segmentation methods" and "8x speedup for video annotations" — but the built lesson only includes the first claim, and neither is grounded in any explanation. The student has no context for what "prior methods" means (they have not been taught any video segmentation method), so "3x fewer" is an unanchored number. It is a factoid that the student cannot evaluate or connect to anything they know.
**Student impact:** Minor — the student reads "3x fewer interactions" and accepts it as a claimed improvement, but it has no pedagogical value because there is no reference point. The student cannot judge whether 3x is impressive or expected. It is the kind of number that sounds good in a paper abstract but does not teach anything here.
**Suggested fix:** Either (a) briefly ground it: "SAM 2 requires 3x fewer user corrections per video compared to prior frame-by-frame annotation tools, because the memory mechanism propagates a single click across many frames" — this connects the number to the memory concept the student just learned; or (b) remove the specific claim and replace with a qualitative statement: "Far fewer user interactions — prompt once, track across all frames."

#### [POLISH] — Segmentation variants paragraph uses smaller italic text that breaks visual rhythm

**Location:** The Segmentation Problem section, final paragraph (lines 237-245): "Brief note on variants..."
**Issue:** The segmentation variants paragraph (semantic, instance, panoptic) is wrapped in `<em>` tags and uses `text-sm`, making it visually smaller and italic compared to surrounding prose. The planning document specifies this content at MENTIONED depth, which is appropriate, but the visual treatment makes it look like a disclaimer or footnote rather than content worth reading. It stands out as visually different from every other paragraph in the lesson.
**Student impact:** Very minor. The student might skip it because it visually signals "this is less important." For MENTIONED-depth content that the student may want to revisit later (when they encounter these terms in papers), it would be better to not visually deprioritize it this much.
**Suggested fix:** Remove the `text-sm` class and `<em>` wrapper. Use standard paragraph styling with a brief lead-in: "There are several variants of segmentation: semantic segmentation labels every pixel with a class..." The content stays MENTIONED-depth (brief, not developed) but does not look visually demoted.

#### [POLISH] — The "Predict and Verify" check question 3 answer is slightly imprecise

**Location:** Check 1, Question 3 (lines 789-805): "You click a point on the boundary between two touching objects (a cup sitting on a plate). What does SAM output?"
**Issue:** The answer states SAM would output "likely the cup alone, the plate alone, or both together." However, SAM's three masks are typically at different *granularities* of the same region (part, object, scene-level), not three different objects. A boundary click would more likely produce: (1) one of the two objects, (2) the other object, or (3) both together — but SAM does not have a mechanism to separately identify cup vs plate from a single ambiguous click. The three masks would more likely be: small region near the boundary, one of the two objects (whichever the features more strongly indicate), and both objects together. The answer as written slightly over-promises SAM's disambiguation ability.
**Student impact:** Very minor. The student might form a slightly too-optimistic mental model of SAM's boundary disambiguation. In practice, a boundary click is one of SAM's hardest cases and the masks may not cleanly separate into "cup alone" and "plate alone."
**Suggested fix:** Adjust the answer: "Three masks at different granularities — likely the region near the click, one of the two objects (whichever the model's features more strongly associate with the click point), or both together. A boundary click is inherently one of SAM's hardest cases — this is where box prompts or multiple point prompts help disambiguate."

### Review Notes

**Resolution of iteration 1 findings:**

All 9 findings from iteration 1 were properly addressed:

1. **CRITICAL (cookie cutter analogy):** Fully resolved. The analogy is now developed in two substantial paragraphs in the hook section's main content (lines 305-322). It covers the one-cutter-per-shape limitation, the programmable cutter concept, and SAM 3's "describe what you want" extension. The mental model echo in the summary now references an analogy the student has genuinely encountered and internalized.

2. **IMPROVEMENT (unified end-to-end trace):** Resolved. A dedicated "Tracing a single prompt: click to mask" subsection (lines 504-579) walks through Steps 1-5 with the mug example, including prompt encoding and cross-attention details that were previously missing. This is a strong addition. (However, see finding about redundancy with the multi-mask section.)

3. **IMPROVEMENT (SAM 3 "red car" example):** Resolved. A concrete ComparisonRow (lines 1023-1062) contrasts SAM 1 (click each car, 4 clicks) with SAM 3 (type "red car," one prompt, 4 masks). The distinction is now visceral rather than abstract.

4. **IMPROVEMENT ("SAM is not classification" misconception):** Resolved. Lines 225-229 now explicitly state: "SAM is not doing classification. It does not output a class label like 'dog, 97% confidence.' It outputs a binary mask... SAM knows boundaries, not categories." This is planted in the segmentation section, before the architecture is introduced.

5. **IMPROVEMENT (additive framing before SAM 2):** Resolved. Lines 886-897 add a framed note: "SAM 2 and SAM 3 are not separate models built from scratch. Each version adds to the previous one." This sets the right expectation before the student reads SAM 2 and SAM 3.

6. **IMPROVEMENT ("of course" moments distributed):** Resolved. The three moments now appear at their natural locations: line 488 after the amortization explanation, line 676 after multi-mask output, and line 991 when introducing SAM 3's text prompting.

7. **POLISH (MAE expanded):** Resolved. Line 357: "A MAE (Masked Autoencoder)-pretrained ViT-H (Huge)."

8. **POLISH (DETR removed):** Resolved. Line 1072: "A transformer-based detector finds all instances of the prompted concept" — DETR name dropped.

9. **POLISH (IoU expanded):** Resolved. Line 418: "IoU — Intersection over Union — estimating how well the mask fits the true object."

**Did fixes introduce new issues?**

One new issue emerged: the end-to-end trace reuses the exact same mug/handle/saucer example with identical confidence scores that appears again in the multi-mask section. This is a direct consequence of adding the trace — it now overlaps with existing content. The fix is straightforward (use the shirt-button scenario for multi-mask).

**What works well (fresh evaluation):**

- The lesson's narrative arc is strong. The segmentation primer -> bottleneck hook -> architecture -> promptable interface -> data engine -> evolution structure flows naturally with clear transitions.
- The cookie cutter analogy in the hook now serves as an effective anchor. It grounds the entire "promptable foundation model" concept before any technical detail arrives.
- The end-to-end trace (Steps 1-5) is the strongest addition — it transforms the architecture from "three components I can name" to "a pipeline I can trace." The numbered steps with timing make it concrete.
- The "red car" ComparisonRow is effective. The side-by-side SAM 1 (4 clicks) vs SAM 3 (one prompt) makes the capability gap visceral.
- The three "of course" moments, now distributed, work well. Each arrives at the moment the student is thinking about the design choice, making it feel inevitable rather than arbitrary.
- The explicit "SAM is not classification" correction (lines 225-229) is well-placed — it catches the misconception before the student would form it.
- All five planned misconceptions are addressed at appropriate locations.
- Modality count for the core concept: verbal/analogy (cookie cutter, developed in main content), visual (Mermaid diagrams, GradientCards), symbolic/structural (tensor shapes, architecture breakdown), concrete example (desk/mug trace, red car comparison), intuitive ("of course" moments). Five modalities present and substantive.
- Em dash formatting is correct throughout (no spaces).
- All interactive elements (details/summary) have cursor-pointer styling.
- The lesson stays within its stated scope boundaries.

**No notebook expected:** The planning document does not specify exercises or a Practice section, so the absence of a notebook is not a finding.

---

## Revision -- 2026-02-21 (Depth Rework)

### Why This Rework Is Needed

User feedback: "The SAM 3 lesson seems more like a layman's lesson rather than going into the inner workings of SAM 3 to the level the course covers other topics at. Usually there's a lot more math and technical teachings."

The user is right. Comparing the current SAM lesson to the course's standard:

- **CLIP lesson** has: loss formulas in LaTeX, PyTorch pseudocode with tensor shapes, a hand-traced 4x4 similarity matrix, step-by-step softmax computation, side-by-side code for training steps.
- **SigLIP 2 lesson** has: sigmoid loss formula derived and traced on specific cells with concrete numbers (sigma(+1 * 0.9 * 10) = ...), side-by-side CLIP vs SigLIP PyTorch code, explicit gradient independence proof.
- **Queries and Keys lesson** has: actual W_Q and W_K matrices with specific numbers, hand-computed Q and K vectors for each token, the full QK^T matrix traced cell-by-cell, softmax applied row-by-row, side-by-side comparison of raw XX^T vs projected QK^T.

The SAM lesson has: prose descriptions of what each component does, a Mermaid architecture diagram, timing numbers, and GradientCards with bullet points. It describes WHAT SAM does but never shows HOW it works mathematically. No formulas, no code, no traced computations. The student finishes knowing the architecture names but not the mechanics.

**Specific gaps in the current lesson:**
1. **Prompt encoding**: Says "positional encoding + type embedding" but never shows the encoding. How does a click at pixel (342, 517) become a vector? What are the dimensions?
2. **Mask decoder**: Says "cross-attention between prompt tokens and image tokens" but never shows the actual operations. How many layers? What happens in each layer? What are the token compositions?
3. **Loss function**: Explicitly scoped OUT ("NOT: mask decoder loss function details"). The user wants this. Focal loss, dice loss, multi-mask minimum-loss assignment are all teachable at this course's level.
4. **SAM 2 memory**: Says "memory bank" and "memory attention" without showing the mechanism. How are memory tokens produced? How does cross-attention over the memory bank work?
5. **No code**: Zero PyTorch pseudocode. Every other technical lesson in this course has code blocks showing the forward pass with tensor shapes.
6. **No traced computation**: No specific numbers traced through any operation. The attention lessons trace q_i dot k_j with actual vectors. The SAM lesson never does this.

**Scope change:** The scope boundary "NOT: mask decoder loss function details (focal loss, dice loss)" is REMOVED. The user explicitly wants this depth. The lesson will cover focal loss, dice loss, IoU prediction loss, and multi-mask minimum-loss training.

### Revised Section Outline

The lesson retains its good structural elements (segmentation primer, cookie cutter analogy, data engine, evolution arc, checks) but adds substantial technical depth to the architecture, prompt encoding, mask decoder, loss function, and SAM 2 memory sections. New sections are marked with **(NEW)** or **(EXPANDED)**.

---

**1. Context + Constraints** (kept, updated scope)
Same framing. Update the scope to INCLUDE loss function details, mask decoder internals, prompt encoding math, and PyTorch pseudocode.

**2. Recap** (kept as-is)
Brief refresher on ViT, cross-attention, encoder-decoder. No changes needed.

**3. The Segmentation Problem** (kept as-is)
Gap resolution for segmentation as a task. Three-panel comparison. Works well.

**4. Hook: The Segmentation Bottleneck** (kept as-is)
Cookie cutter analogy (now in main content per iteration 1 fix). GPT-3 parallel. Works well.

**5. SAM 1 Architecture Overview** (kept, minor restructure)
The three-component diagram and high-level architecture. Keep the GradientCards for image encoder, prompt encoder, mask decoder. Keep the Mermaid diagram. This section gives the "map" before the detailed "territory" sections that follow.

**6. Prompt Encoding: From Clicks to Tokens (NEW)**

This is a new dedicated section showing the mathematical mechanics of how user prompts become tokens.

**Content:**

**(a) Point prompt encoding:**
- A click at pixel coordinates (x, y) is encoded using Fourier/sinusoidal positional encoding
- The positional encoding maps the 2D coordinate to a high-dimensional vector
- Formula in LaTeX:

  PE(p) = [sin(2^0 * pi * p_x), cos(2^0 * pi * p_x), sin(2^0 * pi * p_y), cos(2^0 * pi * p_y), ..., sin(2^(L-1) * pi * p_x), cos(2^(L-1) * pi * p_x), ...]

  where p_x, p_y are normalized to [0, 1] and L is the number of frequency bands (128 bands -> 256-dim per coordinate -> 512-dim total positional encoding, then projected to match prompt token dimension of 256)
- A learned type embedding is added: foreground point vs background point (different learned vectors)
- Final prompt token = positional_encoding(x, y) + type_embedding

**(b) Box prompt encoding:**
- A bounding box is encoded as two corner points: top-left and bottom-right
- Each corner gets its own positional encoding + a learned type embedding (top-left type vs bottom-right type)
- Result: 2 prompt tokens for a box prompt

**(c) Mask prompt encoding:**
- A coarse mask input is downsampled to 256x256 and processed through lightweight convolutions (two 2x2 stride-2 convolutions + GELU + one 1x1 convolution)
- Output: a spatial embedding at the same resolution as the image embedding (64x64x256)
- Added element-wise to the image embedding (not as a token)

**(d) Concrete traced computation:**
- Walk through a specific point prompt: user clicks at pixel (342, 517) on a 1024x1024 image
- Normalize: p_x = 342/1024 = 0.334, p_y = 517/1024 = 0.505
- Show the first few terms of the Fourier encoding: sin(pi * 0.334), cos(pi * 0.334), sin(pi * 0.505), cos(pi * 0.505), sin(2*pi * 0.334), ...
- Compute actual numbers: sin(1.049) = 0.866, cos(1.049) = 0.500, sin(1.587) = 1.000, cos(1.587) = 0.001, ...
- The 512-dim positional vector is projected to 256-dim via a learned linear layer
- Add the foreground type embedding (a learned 256-dim vector)
- Result: one 256-dim prompt token

**Modalities:** LaTeX formula for positional encoding, concrete traced numbers, connection to sinusoidal position encodings from the transformer lessons ("you have seen this before -- same idea as the sinusoidal position encoding from Series 4, but encoding spatial position instead of sequence position").

**(e) PyTorch pseudocode:**
```python
# Point prompt encoding
def encode_point(point_xy, image_size, foreground=True):
    # Normalize to [0, 1]
    normalized = point_xy / image_size              # [2]

    # Fourier positional encoding (128 frequency bands)
    freqs = 2.0 ** torch.arange(128)               # [128]
    # For each coordinate, compute sin and cos at each frequency
    angles_x = normalized[0] * freqs * math.pi     # [128]
    angles_y = normalized[1] * freqs * math.pi     # [128]
    pe = torch.cat([
        angles_x.sin(), angles_x.cos(),            # [256]
        angles_y.sin(), angles_y.cos(),             # [256]
    ])                                              # [512]

    # Project to prompt token dimension
    pe = linear_projection(pe)                      # [256]

    # Add learned type embedding
    type_emb = fg_embedding if foreground else bg_embedding  # [256]
    return pe + type_emb                            # [256]
```

**7. Mask Decoder: The Two-Layer Transformer (EXPANDED)**

The current lesson says "lightweight transformer decoder with only 2 layers" and "uses cross-attention." This section expands to show the exact operations.

**Content:**

**(a) Token composition entering the decoder:**
- Prompt tokens from the prompt encoder (1-4 tokens depending on prompt type)
- Learned output tokens: one per mask candidate (SAM produces 3 masks, so 3 output mask tokens + 1 IoU prediction token = 4 learned tokens)
- Total tokens entering decoder: prompt_tokens + 4 output_tokens
- For a single point prompt: 1 + 4 = 5 tokens total

**(b) Each decoder layer performs three attention operations (in order):**

**Step 1: Self-attention among all tokens**
- Q, K, V all come from the token set (prompt tokens + output tokens)
- Standard self-attention: tokens communicate with each other
- The output tokens learn to specialize (one for small mask, one for medium, one for large)
- This is where the mask tokens "discuss" which granularity each will handle

**Step 2: Token-to-image cross-attention**
- Q = token embeddings (prompt + output tokens)
- K, V = image embedding (64x64 = 4096 spatial tokens, each 256-dim)
- Each token queries the image: "which spatial locations are relevant to me?"
- This is the core operation: prompt information fused with image features
- Formula: Attention(Q_tokens, K_image, V_image) = softmax(Q_tokens * K_image^T / sqrt(d)) * V_image

**Step 3: Image-to-token cross-attention**
- Q = image embedding tokens
- K, V = updated token embeddings (from step 2)
- The image features attend BACK to the prompt/output tokens
- This updates the image embedding with prompt-specific information
- "Which parts of the image are most relevant given what the prompt is asking?"

**(c) After 2 decoder layers:**
- Extract the output mask tokens (3 tokens, each 256-dim)
- Extract the IoU prediction token (1 token, 256-dim)
- Each mask token is projected through a small MLP to produce a per-pixel mask logit vector
- The mask logit vector is dotted with the upsampled image features to produce a spatial mask

**(d) Upsampling to produce the final mask:**
- Decoder output is at 64x64 resolution (matching the image embedding)
- Two transposed convolution layers upsample: 64x64 -> 128x128 -> 256x256
- Each mask token produces one 256x256 mask prediction
- Final upsampling from 256x256 to original image resolution (e.g., 1024x1024) via bilinear interpolation

**(e) IoU prediction head:**
- The IoU token is passed through a small MLP
- Outputs 3 scores (one per mask candidate)
- Each score predicts the IoU (Intersection over Union) between the predicted mask and the true object mask
- Used to rank the mask candidates: highest predicted IoU = best mask

**(f) Concrete traced cross-attention computation:**

Pick specific dimensions and trace the token-to-image cross-attention step. This follows the pattern from the Queries and Keys lesson:

- 5 tokens (1 prompt + 4 output), each 256-dim
- 4096 image positions (64x64 flattened), each 256-dim
- Q = W_Q * tokens: [5, 256] -> [5, 256]
- K = W_K * image: [4096, 256] -> [4096, 256]
- V = W_V * image: [4096, 256] -> [4096, 256]
- Scores = Q * K^T / sqrt(256): [5, 256] x [256, 4096] = [5, 4096]
- Weights = softmax(Scores, dim=-1): [5, 4096] -- each token has an attention distribution over all 4096 image positions
- Output = Weights * V: [5, 4096] x [4096, 256] = [5, 256]

"The prompt token (row 0) produces a 4096-element attention distribution over the 64x64 image grid. High weights correspond to image positions near the clicked point. The output is a 256-dim vector that aggregates image features from the attended positions -- this is how the click location 'reads' the image."

**(g) PyTorch pseudocode for the full mask decoder forward pass:**
```python
class MaskDecoder(nn.Module):
    def forward(self, image_embedding, prompt_tokens):
        # image_embedding: [1, 256, 64, 64]
        # prompt_tokens: [1, N_prompt, 256]  (N_prompt = 1 for point, 2 for box)

        # Concatenate prompt tokens with learned output tokens
        output_tokens = self.mask_tokens.weight    # [4, 256] (3 mask + 1 IoU)
        tokens = torch.cat([prompt_tokens, output_tokens], dim=1)  # [1, N_prompt+4, 256]

        # Flatten image embedding for attention
        image_tokens = image_embedding.flatten(2).permute(0, 2, 1)  # [1, 4096, 256]

        # Two decoder layers
        for layer in self.layers:
            # Step 1: Self-attention among tokens
            tokens = layer.self_attn(q=tokens, k=tokens, v=tokens)

            # Step 2: Token-to-image cross-attention
            tokens = layer.cross_attn_token_to_image(
                q=tokens, k=image_tokens, v=image_tokens
            )

            # Step 3: Image-to-token cross-attention
            image_tokens = layer.cross_attn_image_to_token(
                q=image_tokens, k=tokens, v=tokens
            )

        # Extract output tokens
        mask_tokens = tokens[:, -4:-1, :]    # [1, 3, 256] -- 3 mask tokens
        iou_token = tokens[:, -1:, :]        # [1, 1, 256] -- IoU token

        # Generate masks via dot product with upsampled image features
        upsampled = self.upsample(image_tokens)  # [1, 256, 256, 256] via transposed conv
        # Each mask token produces one mask
        masks = torch.einsum('bmc,bcwh->bmwh',
            self.mask_mlp(mask_tokens),          # [1, 3, 256]
            upsampled                            # [1, 256, 256, 256]
        )                                        # [1, 3, 256, 256]

        # Predict IoU scores
        iou_scores = self.iou_head(iou_token)    # [1, 3]

        return masks, iou_scores
```

**Modalities:** LaTeX for cross-attention formula, PyTorch code block, traced tensor shapes at each step, concrete cross-attention dimension trace (following Queries and Keys pattern), architectural diagram showing the three attention operations per layer.

**8. Loss Function: Training SAM (NEW)**

This is an entirely new section. The previous scope explicitly excluded loss function details. The user wants this depth, and the course teaches loss functions with formulas and code (see CLIP's symmetric cross-entropy, SigLIP's sigmoid loss).

**Content:**

**(a) The problem: class imbalance**
- In a segmentation mask, most pixels are background (label 0), few are foreground (label 1)
- Standard cross-entropy would be dominated by the easy background pixels
- SAM uses two complementary losses that address this from different angles

**(b) Focal loss (addresses hard-pixel focus):**
- Modification of cross-entropy that down-weights well-classified pixels
- Formula:

  L_focal = -alpha * (1 - p_t)^gamma * log(p_t)

  where p_t = p if y=1, (1-p) if y=0
- gamma controls focusing: gamma=0 recovers standard cross-entropy, gamma=2 (SAM's default) heavily down-weights easy pixels
- alpha balances foreground/background contribution
- Concrete trace: for a correctly classified background pixel with p=0.95: (1 - 0.95)^2 * log(0.95) = 0.0025 * 0.051 = 0.000128 -- nearly zero loss. For a misclassified foreground pixel with p=0.1: (1 - 0.1)^2 * log(0.1) = 0.81 * 2.303 = 1.865 -- high loss. The ratio is ~14,000x. Focal loss automatically focuses on the hard cases.
- Connection: "This is the same principle as the temperature discussion in attention scaling -- controlling the sharpness of a distribution to focus on what matters."

**(c) Dice loss (directly optimizes overlap):**
- Directly measures the overlap between predicted and ground-truth masks
- Formula:

  L_dice = 1 - (2 * |P intersection G| + 1) / (|P| + |G| + 1)

  In continuous form: L_dice = 1 - (2 * sum(p_i * g_i) + 1) / (sum(p_i) + sum(g_i) + 1)
- Ranges from 0 (perfect overlap) to 1 (no overlap)
- The +1 smoothing term prevents division by zero when both masks are empty
- Why this complements focal loss: focal loss operates per-pixel (each pixel contributes independently), dice loss operates on the mask as a whole (the sum couples all pixels). Together they provide both local and global training signal.

**(d) Combined loss:**

  L = lambda_focal * L_focal + lambda_dice * L_dice

  SAM uses lambda_focal = 20, lambda_dice = 1 (focal loss weighted 20x higher)

**(e) IoU prediction loss:**
- The IoU prediction head is trained separately with MSE loss
- L_iou = MSE(predicted_iou, actual_iou)
- actual_iou is computed from the predicted mask and ground truth during training
- This teaches the model to estimate its own mask quality

**(f) Multi-mask training (minimum loss assignment):**
- SAM outputs 3 masks per prompt
- During training, compute the loss for ALL 3 masks against the ground truth
- Backpropagate only through the mask with the LOWEST loss
- This is minimum-loss assignment: each mask candidate is free to specialize in a different granularity
- Without this: all 3 masks converge to the same prediction (why produce 3 identical masks?)
- With this: one mask learns "small region," another learns "full object," another learns "object + context"
- Formula: L_mask = min(L_1, L_2, L_3) where L_i = lambda_focal * L_focal(mask_i, gt) + lambda_dice * L_dice(mask_i, gt)

**(g) PyTorch pseudocode:**
```python
# SAM loss computation
def compute_loss(pred_masks, pred_ious, gt_mask):
    # pred_masks: [3, H, W] -- 3 candidate masks
    # pred_ious: [3] -- predicted IoU for each mask
    # gt_mask: [H, W] -- ground truth binary mask

    losses = []
    actual_ious = []
    for i in range(3):
        # Focal loss (per-pixel, focuses on hard examples)
        p_t = pred_masks[i].sigmoid()
        focal = -alpha * (1 - p_t)**gamma * F.binary_cross_entropy_with_logits(
            pred_masks[i], gt_mask, reduction='none'
        )

        # Dice loss (mask-level overlap)
        intersection = (p_t * gt_mask).sum()
        dice = 1 - (2 * intersection + 1) / (p_t.sum() + gt_mask.sum() + 1)

        losses.append(20.0 * focal.mean() + 1.0 * dice)

        # Actual IoU for IoU prediction training
        pred_binary = (p_t > 0.5).float()
        intersection = (pred_binary * gt_mask).sum()
        union = pred_binary.sum() + gt_mask.sum() - intersection
        actual_ious.append(intersection / (union + 1e-6))

    # Minimum loss assignment: backprop through best mask only
    losses = torch.stack(losses)
    best_mask_idx = losses.argmin()
    mask_loss = losses[best_mask_idx]

    # IoU prediction loss
    actual_ious = torch.stack(actual_ious)
    iou_loss = F.mse_loss(pred_ious, actual_ious)

    return mask_loss + iou_loss
```

**Modalities:** LaTeX formulas for focal loss and dice loss, concrete traced pixel values, PyTorch code, connection to prior loss functions (cross-entropy from CLIP, MSE from diffusion).

**9. Check 1: Predict and Verify** (kept, expanded)
Keep the existing check questions. Add one new question about the loss function:
- "Why does SAM use minimum-loss assignment instead of averaging the loss across all 3 masks?" (Because averaging would pressure all masks to be similar. Minimum-loss lets each mask specialize in a different granularity -- the model is rewarded for having at least ONE good mask, not for having three mediocre ones.)

**10. The SA-1B Data Engine** (kept as-is)
Three-stage data engine. Works well. No technical depth changes needed -- this is about the data methodology, not the architecture.

**11. SAM 2: Video Extension (EXPANDED)**

The current section describes SAM 2 conceptually. Add the technical mechanism.

**Content:**

**(a) Memory encoder:**
- After segmenting a frame, the predicted mask and image features are combined
- The mask is downsampled and passed through lightweight convolutions
- Combined with the image encoder output via element-wise addition
- Result: memory tokens for this frame (64x64x64 spatial memory features)

**(b) Memory bank:**
- Stores memory tokens from N most recent frames (default N=6)
- Also stores memory tokens from any frame where the user provided a prompt (prompted frame memories are never evicted)
- Structure: a set of [N_frames, 64x64, 64] spatial memory features

**(c) Memory attention (cross-attention over the memory bank):**
- Current frame's image features attend to the memory bank
- Q = current frame image features: [4096, 256]
- K, V = memory bank tokens: [N_frames * 4096, 64] (projected to match dimensions)
- Standard cross-attention: current frame "reads" the memory of recent frames
- This is how temporal consistency works: the current frame sees where the object was in previous frames
- Connection: "Same cross-attention formula you know. Q from current frame, K/V from memory. The memory bank IS the context window, just for video instead of text."

**(d) Occlusion handling:**
- When the object is fully occluded, the mask prediction for that frame will be empty (or near-empty)
- The memory bank still contains pre-occlusion memories
- When the object reappears, the current frame's features will have high cross-attention scores with the pre-occlusion memory tokens (the object looks similar to how it looked before)
- This is not a special mechanism -- it falls out naturally from cross-attention over a memory bank that retains past frames

**(e) Tensor shape summary for one frame of video processing:**
```
Frame t:
  Image encoder: frame [3, 1024, 1024] -> features [256, 64, 64]
  Memory attention: features [4096, 256] x memory_bank [N*4096, 64] -> updated features [4096, 256]
  Prompt encoder: click/box -> prompt tokens [N_prompt, 256]
  Mask decoder: (updated features, prompt tokens) -> masks [3, 256, 256], IoU scores [3]
  Memory encoder: (mask, features) -> memory tokens [64, 64, 64] -> added to memory bank
```

**Modalities:** Cross-attention formula connecting to prior lessons, tensor shape trace for a full video frame, pseudocode showing the memory bank update loop.

**12. SAM 3: Concepts and Language** (kept, minor additions)
Keep the current content. The SAM 3 architecture uses a unified Perception Encoder and a transformer-based detector, but since these are architecturally more complex and the student has not been taught DETR, this section stays at INTRODUCED depth. The red car ComparisonRow stays. Add a brief note on how the text prompt is encoded (via a text encoder that produces embedding tokens, similar to CLIP/SigLIP text encoding the student already knows).

**13. Check 2: Transfer Question** (kept as-is)
Wildlife monitoring scenario. Works well.

**14. SAM in the Vision Ecosystem** (kept as-is)
Foundation model pattern synthesis. Works well.

**15. Summarize** (expanded)
Add takeaways for:
- Prompt encoding uses Fourier positional encoding + learned type embeddings (same sinusoidal encoding idea from transformers, applied to spatial coordinates)
- The mask decoder runs three types of attention per layer: self-attention among tokens, token-to-image cross-attention, image-to-token cross-attention
- The loss combines focal loss (hard-pixel focus) + dice loss (global overlap) with minimum-loss assignment across the 3 mask candidates
- SAM 2's memory mechanism is cross-attention over stored frame memories

**16. Next Step** (kept as-is)

---

### Updated Scope Boundaries

**This lesson IS about (additions marked with +):**
- What image segmentation is and how it differs from classification and detection
- SAM's core insight: promptable segmentation as a foundation model approach
- SAM 1's three-component architecture (ViT encoder + prompt encoder + mask decoder) and why each component exists
- (+) Prompt encoding math: Fourier positional encoding for spatial coordinates, learned type embeddings, box encoding as two corner points
- (+) Mask decoder internals: the three attention operations per layer (self-attention, token-to-image cross-attention, image-to-token cross-attention), token composition, upsampling via transposed convolutions
- (+) Loss function: focal loss, dice loss, combined loss, IoU prediction head, multi-mask minimum-loss training
- (+) PyTorch pseudocode for prompt encoding, mask decoder forward pass, and loss computation
- (+) Traced computations: point encoding with specific pixel coordinates, cross-attention dimension trace through the mask decoder
- The prompt types (points, boxes, masks, and SAM 3's text/exemplar prompts)
- The amortized computation design (heavy encoder once, light decoder per prompt)
- The SA-1B dataset and the data engine concept
- (+) SAM 2's memory mechanism with technical detail: memory encoder, memory bank, memory attention as cross-attention
- SAM 3's extension to concept-level understanding

**This lesson is NOT about (updated):**
- ~~The mathematical details of the mask decoder's loss function (focal loss, dice loss)~~ **REMOVED -- now IN scope**
- Implementing SAM from scratch or training a segmentation model end-to-end
- DETR or transformer-based object detection architectures in depth
- Semantic vs instance vs panoptic segmentation in formal detail (INTRODUCED only)
- SAM 3D or 3D reconstruction
- EfficientSAM, MobileSAM, or efficiency variants
- The full SA-Co dataset construction methodology
- Benchmark comparisons beyond illustrative examples
- Fine-tuning SAM for specific domains
- The Perception Encoder architecture internals (SAM 3 specific -- complex, and student lacks DETR prerequisites)

### Updated Target Depths

- Promptable segmentation concept: DEVELOPED (unchanged)
- SAM's three-component architecture: DEVELOPED (unchanged, but now with mechanical detail -- the student can trace data flow with tensor shapes, not just name the components)
- (+) Prompt encoding (Fourier positional encoding + type embedding): DEVELOPED (can explain the encoding, write pseudocode, trace specific coordinates)
- (+) Mask decoder operations (three attention types per layer, token composition, upsampling): DEVELOPED (can explain each operation, trace tensor shapes, identify Q/K/V sources)
- (+) SAM loss function (focal + dice + minimum-loss assignment): DEVELOPED (can write the formulas, explain why each component exists, trace the loss for specific pixels)
- Image segmentation as a task: INTRODUCED (unchanged)
- SA-1B dataset and data engine: INTRODUCED (unchanged)
- (+) SAM 2 memory mechanism (memory encoder, memory bank, memory attention): INTRODUCED -> DEVELOPED (can explain the cross-attention mechanism with tensor shapes)
- SAM 3 concept-level segmentation: INTRODUCED (unchanged)

### Cognitive Load Reassessment

The new technical depth adds several sub-concepts (Fourier encoding, focal loss, dice loss, memory attention), but these are all applications of concepts the student already knows:
- Fourier positional encoding = the sinusoidal position encoding from transformers (Series 4), applied to 2D spatial coordinates
- Focal loss = modified cross-entropy (the student knows cross-entropy deeply from CLIP)
- Dice loss = set overlap metric (conceptually simple)
- Memory attention = cross-attention (the student knows cross-attention from Series 4 and Series 6)

The genuinely new concepts remain the same 2-3 from the original plan:
1. Promptable segmentation as a paradigm
2. SAM's three-component architecture composition
3. The data engine approach

The technical details are *applications* of known mechanisms in a new context, not new mechanisms. The cognitive load stays at BUILD.

However, the lesson is now substantially longer. This is acceptable: the attention series has 6 lessons covering attention mechanics at this depth. A single SAM lesson covering architecture + prompt encoding + mask decoder + loss + memory + evolution is dense but coherent because the student has all the building blocks.

### Modality Coverage for New Sections

| Section | LaTeX Formulas | PyTorch Code | Traced Computation | Connection to Prior | Diagram |
|---------|---------------|-------------|-------------------|-------------------|---------|
| Prompt Encoding | Fourier PE formula | `encode_point()` pseudocode | Click at (342, 517) traced through encoding | "Same sinusoidal encoding from transformers" | -- |
| Mask Decoder | Cross-attention formula | Full `MaskDecoder.forward()` | 5 tokens x 4096 image positions traced | "Same Q/K/V formula, Q from tokens, K/V from image" | Three-operation-per-layer diagram |
| Loss Function | Focal loss, dice loss formulas | `compute_loss()` pseudocode | Specific pixel losses traced (p=0.95 bg, p=0.1 fg) | "Focal = modified cross-entropy, IoU = MSE" | -- |
| SAM 2 Memory | -- | Memory bank update loop | Tensor shapes for one video frame | "Memory bank IS the context window for video" | -- |

---

## Review — 2026-02-21 (Post-Depth-Rework, Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 3

### Verdict: NEEDS REVISION

The depth rework is a substantial improvement. The lesson now has the technical weight the user requested: LaTeX formulas, PyTorch pseudocode, traced computations, and tensor dimension traces throughout. The new sections (prompt encoding, mask decoder deep dive, loss function, SAM 2 memory) are well-structured and consistently connect back to prior knowledge. The core narrative arc (segmentation primer, cookie cutter hook, architecture overview, detailed mechanics, data engine, evolution) flows well despite the longer length. However, there is one critical math accuracy issue in the traced computation, three improvement-level findings, and three polish items.

### Findings

#### [CRITICAL] — Incorrect traced values in the Fourier positional encoding computation

**Location:** "Tracing a specific click" section (lines 610-676), specifically the first frequency band values at lines 630-634.
**Issue:** The lesson traces the first frequency band for a click at (342, 517) on a 1024x1024 image with px=0.334, py=0.505. Two values are incorrect:

- The lesson says `cos(pi * 0.334) = cos(1.049) = 0.500`. The actual value is `0.498`. This is minor rounding.
- The lesson says `cos(pi * 0.505) = cos(1.587) = 0.001`. The actual value is `-0.016`. **The sign is wrong** (should be negative, not positive), and the magnitude is off by 16x.

The cos(pi * 0.505) error is the most important: pi * 0.505 = 1.587, which is just past pi/2 = 1.571. Cosine crosses zero at pi/2 and goes negative, so cos(1.587) must be slightly negative. The lesson claims it is 0.001 (positive, near zero), when it is actually -0.016 (negative, near zero).

Additionally, in the second frequency band: `sin(2*pi * 0.334) = sin(2.099) = 0.866` should be `0.864`, and `cos(2*pi * 0.334) = cos(2.099) = -0.500` should be `-0.504`. These are minor rounding discrepancies but accumulate.

**Student impact:** The student is tracing a concrete computation to build intuition for how Fourier encoding works. Getting a wrong sign, even on a small value, undermines the credibility of the trace. A careful student who verifies with a calculator will lose trust. More importantly, the sign of cosine near pi/2 is pedagogically relevant: the aside (line 669) says "sin(pi * 0.505) approximately 1.0 tells us the y-coordinate is near the middle" -- the cos companion being slightly negative (not slightly positive) reinforces that we just crossed the pi/2 boundary, which is actually a useful observation. The wrong sign obscures this.
**Suggested fix:** Correct all traced values to accurate computations (round to 3 decimal places consistently):
- `cos(pi * 0.334) = cos(1.049) = 0.498` (not 0.500)
- `cos(pi * 0.505) = cos(1.587) = -0.016` (not 0.001)
- `sin(2*pi * 0.334) = sin(2.099) = 0.864` (not 0.866)
- `cos(2*pi * 0.334) = cos(2.099) = -0.504` (not -0.500)

#### [IMPROVEMENT] — Focal loss traced formula drops the negative sign, creating inconsistency with the stated formula

**Location:** "Tracing focal loss on two pixels" section (lines 1318-1364), specifically the traced computations at lines 1333-1334 and 1348-1349.
**Issue:** The focal loss formula is correctly stated as: `L_focal = -alpha * (1 - p_t)^gamma * log(p_t)`. This is negative alpha times the focusing term times log(p_t). Since 0 < p_t < 1, log(p_t) is negative, and the leading minus sign makes the overall loss positive.

However, the traced computations write:
- Easy pixel: `(1 - 0.95)^2 * log(0.95) = 0.0025 * 0.051 = 0.000128`
- Hard pixel: `(1 - 0.1)^2 * log(0.1) = 0.81 * 2.303 = 1.865`

Both traces drop the `-alpha` factor AND silently negate the log value. `log(0.95) = -0.0513`, not `0.051`. `log(0.1) = -2.303`, not `2.303`. The traces show positive intermediate values that only work if you implicitly absorb the negation.

The student who tries to verify: "(1 - 0.95)^2 * log(0.95) = 0.0025 * (-0.051) = -0.000128" gets a different (negative) answer. This is confusing because the formula says `-alpha * (...)` but the trace ignores both the negative sign and alpha.

**Student impact:** A student who carefully follows the formula will get a negative result from `(1-p_t)^gamma * log(p_t)` and wonder why the lesson shows a positive number. The key pedagogical point (the 14,600x ratio) is correct and unaffected, but the intermediate math is inconsistent. This is especially problematic in a lesson that was specifically reworked to add mathematical rigor.
**Suggested fix:** Show the trace consistently with the formula. Either: (a) trace using `-log(p_t)` (which is the cross-entropy term) and note that the negative sign from the formula converts log(p_t) to -log(p_t), or (b) include the full formula in the trace: `-alpha * (1-0.95)^2 * log(0.95) = -0.25 * 0.0025 * (-0.051) = 0.000032`. Option (a) is cleaner -- write the trace as `(1-0.95)^2 * [-log(0.95)] = 0.0025 * 0.051 = 0.000128` with a brief note that `-log(p_t)` is the standard cross-entropy term the student knows. Also clarify whether alpha is included or omitted (the formula uses alpha but the traces omit it; SAM uses alpha=0.25, and the PyTorch pseudocode at line 1508 does use alpha=0.25).

#### [IMPROVEMENT] — Memory attention tensor dimension mismatch between prose and code

**Location:** Memory attention tensor shapes (lines 1816-1848) and video pipeline code block (lines 1884-1896).
**Issue:** The memory attention section states: `K, V = memory bank tokens: [N_frames * 4096, 64] (projected to match dims)`. The memory encoder section states memory tokens are `64x64x64 spatial memory features`. So each frame produces [4096, 64] memory tokens. But the current frame's Q is [4096, 256]. For cross-attention Q*K^T to work, Q and K must have matching last dimensions. The parenthetical "(projected to match dims)" hand-waves this projection, but the output is stated as [4096, 256] -- matching the original feature dimension, not the memory dimension.

The video pipeline code block says: `Memory attention: features [4096, 256] x memory_bank [N*4096, 64] -> updated features [4096, 256]`. The cross-attention between a 256-dim Q and a 64-dim K does not work without projection. The code does not show this projection, while the prose mentions it parenthetically.

This is not incorrect per se (the projection does exist), but it is under-explained for a lesson that traces every other operation in explicit detail. The mask decoder cross-attention gets a full formula and dimension trace; the memory attention gets a hand-wave.

**Student impact:** A student who has just carefully traced Q[5,256] * K^T[256,4096] in the mask decoder section will notice that Q[4096,256] * K^T[64, N*4096] does not work. The dimension mismatch is visible. The student may wonder: "Does this use multi-head attention with head_dim=64? Is there a projection layer? Why is memory 64-dim when the rest of the model is 256-dim?" None of these are answered.
**Suggested fix:** Either (a) add one sentence explaining the projection: "The memory features (64-dim) are projected up to 256-dim to match the current frame's features before cross-attention" and update the tensor shape display to show `[N*4096, 64] -> projected -> [N*4096, 256]`; or (b) show the shapes post-projection: `K, V = memory bank tokens (projected): [N_frames * 4096, 256]`. Option (b) is simpler and parallels how the mask decoder section shows post-projection shapes.

#### [IMPROVEMENT] — The lesson has no end-to-end prompt-to-mask trace for the rebuilt version

**Location:** The lesson overall (sections 6-8).
**Issue:** The iteration 2 review of the original lesson noted that a "Tracing a single prompt" section was added, and praised it as "the strongest addition." The depth rework revised outline (section 5 in the revised outline) states the architecture overview should "give the 'map' before the detailed 'territory' sections that follow." However, the rebuilt lesson has the detailed territory (prompt encoding math section 7, mask decoder deep dive section 8) but no unified trace that walks through a single concrete example end-to-end connecting all three components.

The original lesson (pre-depth-rework) had a dedicated "Tracing a single prompt: click to mask" subsection with Steps 1-5. The rebuilt lesson replaces this with much deeper per-component traces (the Fourier encoding trace for prompt encoding, the dimension trace for cross-attention in the mask decoder), but these are in separate sections. There is no moment where the student sees: "Here is what happens from the moment you click (342, 517) to the moment you see a mask: Step 1 image encoder already ran, Step 2 the click becomes a 256-dim token (just traced above), Step 3 the token enters the mask decoder with 4 output tokens, Step 4 three attention operations run (just traced above), Step 5 three masks come out."

The amortization section (lines 482-522) touches on this with the desk/mug timing example, but it is about timing, not data flow.

**Student impact:** The student understands each component deeply in isolation but may struggle to integrate them into a single mental model of the full pipeline. The individual traces are excellent, but the connecting tissue between them is implicit. A student might finish the mask decoder section knowing exactly how cross-attention works in the decoder but not connecting it back to the specific prompt token that came from the Fourier encoding section.
**Suggested fix:** Add a brief (3-4 paragraph) "Putting it all together" subsection after the mask decoder PyTorch pseudocode and before the promptable interface section. This does not need new depth -- it references what was just taught: "You have now traced the entire pipeline. A click at (342, 517) becomes a 256-dim token via Fourier encoding. That token, together with 4 learned output tokens, enters the mask decoder. Two layers of self-attention + token-to-image cross-attention + image-to-token cross-attention produce 3 specialized masks via dot product with upsampled features. Total time: ~50ms. The 150ms image encoding was already done." This is a synthesis paragraph, not new content.

#### [POLISH] — Alpha parameter appears in the formula and PyTorch code but is not explained in the traced computation

**Location:** Focal loss section (lines 1275-1364) and PyTorch pseudocode (lines 1500-1563).
**Issue:** The formula includes alpha: `L_focal = -alpha * (1 - p_t)^gamma * log(p_t)`. The description says "alpha balances foreground/background contribution" (in the planning doc, though this exact phrase is not in the lesson -- the lesson only mentions alpha in the formula). The PyTorch code uses `alpha=0.25`. But the traced computation at lines 1333-1349 does not include alpha at all -- it computes `(1-p_t)^2 * log(p_t)` without alpha. The student sees alpha in the formula, sees it used in the code with value 0.25, but the concrete trace ignores it.
**Student impact:** Very minor. The pedagogical point (the ratio between easy and hard) is unaffected by alpha since it is a constant multiplier on both. But the student might notice the omission if they compare the trace to the formula or code.
**Suggested fix:** Either include alpha in the trace (multiply both results by 0.25, which does not change the ratio) or add a brief note: "We trace the focusing factor (1-p_t)^gamma * [-log(p_t)] here, setting alpha aside since it is a constant that applies equally to all pixels and does not affect the ratio."

#### [POLISH] — The `p_t` definition uses cases/piecewise notation that may confuse students unfamiliar with LaTeX rendering

**Location:** Focal loss formula (line 1295): `p_t = { p if y=1, 1-p if y=0 }`
**Issue:** The piecewise definition uses LaTeX `\begin{cases}` notation. While this renders correctly, some students may be unfamiliar with piecewise function notation. The concept itself is simple (p_t is the probability assigned to the correct class), but the notation adds a small speed bump.
**Student impact:** Very minor. The student can likely parse it from context. The surrounding prose says "p_t is the model's predicted probability for the correct class," which is clear enough.
**Suggested fix:** Add one sentence after the formula: "In plain terms: if the pixel is foreground (y=1), p_t is the model's prediction for foreground. If the pixel is background (y=0), p_t is the model's prediction for background. Either way, higher p_t means the model is more confident about the correct answer."

#### [POLISH] — Missing "of course" moment acknowledgement for memory mechanism

**Location:** SAM 2 memory section (lines 1729-1898).
**Issue:** The original lesson had three "of course" moments distributed at their natural locations: (1) amortization, (2) multi-mask, (3) SAM 3 text prompting. All three are present in the rebuilt lesson. However, the SAM 2 memory mechanism is a natural candidate for a fourth "of course" moment: "Of course you would store what the object looked like in previous frames -- that is how you track something across time." The section describes the memory mechanism as cross-attention over past frames, but the "of course" feeling (making the design choice feel inevitable) is not explicitly invoked. The occlusion handling section (lines 1851-1875) comes close with "This is not a special mechanism. It falls out naturally from cross-attention over a memory bank" -- but this is about the elegance of the mechanism, not the inevitability of the design choice.
**Student impact:** Very minor. The section is clear and well-motivated. This is a missed opportunity rather than a problem.
**Suggested fix:** Add one sentence after the memory mechanism introduction or the occlusion paragraph: "Of course you would give the model a memory of what it saw before -- tracking an object means remembering what it looked like."

### Review Notes

**What works well:**

- The depth rework delivers exactly what the user asked for. The lesson now has the same level of mathematical and code-level rigor as the CLIP, SigLIP, and attention lessons. Formulas are in LaTeX, every PyTorch pseudocode block has tensor shape annotations, and concrete computations are traced with specific numbers.

- The prompt encoding section is particularly strong. The connection to sinusoidal position encoding from the transformer lessons ("same idea, applied to 2D space instead of 1D sequence") is exactly the right pedagogical move. The traced computation (click at pixel 342, 517 -> normalized -> first few frequency bands -> project -> add type embedding -> 256-dim token) makes the abstract formula concrete.

- The mask decoder deep dive with the three PhaseCards (self-attention, token-to-image, image-to-token) followed by the full tensor dimension trace is clear and thorough. Labeling Q/K/V sources explicitly at each step prevents the student from getting lost.

- The loss function section is a strong addition. The class imbalance motivation, the two complementary losses (local per-pixel + global mask-level), and the minimum-loss assignment for multi-mask specialization form a coherent narrative. The traced pixel comparison (14,600x ratio) makes the focal loss's focusing behavior visceral.

- The lesson maintains good flow despite being much longer. Transitions between sections are explicit ("Now that we know how prompts become tokens, let us trace what happens inside the mask decoder"). The structural choice -- architecture overview first (the "map"), then deep dives into each component (the "territory") -- works well.

- The cookie cutter analogy (developed in the hook, echoed in the summary) continues to serve as an effective anchor from the pre-rework version.

- The iteration 2 fixes from the original reviews are preserved: the shirt-button multi-mask example (not reusing the mug), the "red car" SAM 1 vs SAM 3 ComparisonRow, the explicit "SAM is not classification" correction planted early, the additive framing before SAM 2, the distributed "of course" moments. The depth rework did not regress these fixes.

- Em dash formatting is correct throughout (no spaces).

- All interactive elements (details/summary) have cursor-pointer styling.

- The lesson stays within its updated scope boundaries.

- All five planned misconceptions from the original plan are addressed at appropriate locations.

**Modality count for core concepts:**
- Promptable segmentation: verbal/analogy (cookie cutter), visual (Mermaid diagram, GradientCards), symbolic (tensor shapes, architecture breakdown), concrete example (desk/mug amortization, shirt-button multi-mask), intuitive ("of course" moments) = 5 modalities.
- Prompt encoding: symbolic (LaTeX Fourier formula), concrete (traced (342, 517) computation), code (PyTorch pseudocode), connection to prior (sinusoidal from transformers) = 4 modalities.
- Mask decoder: symbolic (cross-attention formula, tensor dimension trace), code (PyTorch MaskDecoder.forward()), visual (PhaseCards for 3 operations), concrete (5 tokens x 4096 positions dimension trace) = 4 modalities.
- Loss function: symbolic (focal loss + dice loss formulas), concrete (traced pixels, 14,600x ratio), code (PyTorch compute_loss()), intuitive ("local + global" insight) = 4 modalities.

**No notebook expected:** The planning document does not specify exercises or a Practice section, so the absence of a notebook is not a finding.

---

## Fix Pass — 2026-02-21 (Post-Depth-Rework, Iteration 1/3)

### Findings Addressed

1. **[CRITICAL] Incorrect Fourier positional encoding traced values — FIXED.**
   - Updated normalized coordinates to 4 decimal places: p_x = 0.3340, p_y = 0.5049.
   - Fixed cos(pi * 0.5049) from 0.001 to -0.016 (sign was wrong).
   - Updated all Band 0 and Band 1 values to use consistent precision with `≈` rounding.
   - Added explanatory text after Band 0 noting the negative cosine (pi/2 crossing) and after Band 1 noting all-negative y-values (past pi). These reinforce that negative values are normal and informative.
   - Updated the aside TipBlock to reference the corrected negative cos value.

2. **[IMPROVEMENT] Focal loss trace drops negative sign and alpha — FIXED.**
   - Added an introductory paragraph before the traced computation that rewrites the formula as `alpha * (1-p_t)^gamma * [-log(p_t)]`, explicitly showing that `-log(p_t)` is the standard cross-entropy term.
   - Stated that the trace shows the focusing factor with alpha set aside, since SAM uses alpha=0.25 as a constant multiplier that does not affect the ratio.
   - Updated both trace cards to use `[-log(p_t)]` notation instead of bare `log(p_t)`, making the positive values consistent with the formula.
   - Added a closing note that multiplying by alpha=0.25 does not change the 14,600x ratio.

3. **[IMPROVEMENT] Memory attention tensor dimension mismatch — FIXED.**
   - Replaced the hand-waving "(projected to match dims)" with an explicit projection step: memory bank [N*4096, 64] -> linear -> [N*4096, 256].
   - Shows the full dimension trace: Q[4096, 256], K/V[N*4096, 256], Scores[4096, N*4096], Output[4096, 256].
   - Updated the video pipeline code block to show the projection step explicitly.

4. **[IMPROVEMENT] No end-to-end prompt-to-mask trace — FIXED.**
   - Added a "Putting it all together: click to mask" subsection after the mask decoder PyTorch pseudocode and before the promptable interface section.
   - Traces Steps 1-5: image encoder -> prompt encoding -> assemble tokens -> mask decoder (2 layers of 3 attention ops) -> generate masks.
   - References concrete values from earlier traces (pixel 342,517 -> 256-dim token -> 5 tokens -> 4096 positions -> 3 masks).
   - Includes timing at each step. Added an aside noting that SAM's contribution is composition, not invention.

5. **[POLISH] p_t piecewise notation clarification — FIXED.**
   - Added plain-English sentence after the piecewise formula: "p_t is the model's confidence in the correct answer — p for foreground pixels, (1-p) for background pixels."

6. **[POLISH] Memory mechanism "of course" moment — FIXED.**
   - Added "Of course you would store what the object looked like before — that is how you find it again when it reappears" to the occlusion handling paragraph.

### Findings Not Addressed

None — all 6 findings (1 critical, 3 improvement, 2 polish) were fixed.

---

## Review — 2026-02-21 (Post-Depth-Rework, Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 3

### Verdict: NEEDS REVISION

The iteration 1 critical finding (wrong sign on cos(pi * 0.505)) is properly fixed — the lesson now correctly shows -0.016 with an explanatory note about the pi/2 crossing. The focal loss formula/trace consistency fix is well-executed: the rewritten formula as `alpha * (1-p_t)^gamma * [-log(p_t)]` and the explicit note about setting alpha aside makes the trace verifiable. The memory attention dimension projection is now explicit with the full dimension chain shown. The end-to-end "Putting it all together" trace (lines 1088-1164) effectively connects all three components into a unified pipeline. The p_t plain-English clarification and memory "of course" moment are both present. However, three of the four "minor rounding discrepancies" flagged alongside the critical finding in iteration 1 were NOT actually corrected despite the fix notes claiming "Updated all Band 0 and Band 1 values to use consistent precision with ≈ rounding." Two improvement-level issues remain, plus three polish items.

### Findings

#### [IMPROVEMENT] — Three Fourier encoding traced values are still inaccurately rounded

**Location:** "Tracing a specific click" section (lines 630-651), Band 0 and Band 1.
**Issue:** The iteration 1 review flagged both the critical sign error AND three "minor rounding discrepancies." The fix notes claim "Updated all Band 0 and Band 1 values to use consistent precision with ≈ rounding." The critical error (cos sign) was fixed, but three values remain inaccurate:

- Line 631: `cos(pi × 0.3340) = cos(1.0493) ≈ 0.500` — actual is 0.498. The lesson rounds 0.498 up to 0.500, which is not valid rounding to 3 decimal places.
- Line 648: `sin(2π × 0.3340) = sin(2.0987) ≈ 0.866` — actual is 0.864. The lesson rounds 0.864 up to 0.866.
- Line 649: `cos(2π × 0.3340) = cos(2.0987) ≈ -0.500` — actual is -0.504. The lesson rounds -0.504 toward zero to -0.500.

None of these are sign errors, and all are close. But this is a lesson that was specifically reworked to add mathematical rigor. The `≈` symbol conveys "approximately equal," but 0.498 ≈ 0.500 reads like the value is 0.500 ± small epsilon, when the actual third decimal place is 8, not 0. These are the kind of values a motivated student will check with a calculator.

Additionally, the aside at line 684 uses `0.505` while the main content at line 622 uses `0.5049` for the y-coordinate. This is an inconsistency between the aside and the traced computation.

**Student impact:** Moderate. Individually each value is close enough to not cause confusion. But there are three of them, all in the same section, and the lesson explicitly labels them with `≈`. A student who verifies any of these with a calculator will get a different third decimal digit, which chips away at the mathematical trustworthiness of the trace — especially right after the iteration 1 fix established that getting these values right matters. The aside/main-content inconsistency (0.505 vs 0.5049) is a minor additional source of confusion.
**Suggested fix:** Update the three values to match 3-decimal-place rounding of the exact computation: `cos(1.0493) ≈ 0.498`, `sin(2.0987) ≈ 0.864`, `cos(2.0987) ≈ -0.504`. Update the aside at line 684 to use `0.5049` consistently with the main content.

#### [IMPROVEMENT] — Focal loss trace intermediate value -log(0.95) shown as 0.051, actual is 0.051**3**

**Location:** Focal loss traced computation (lines 1454-1455).
**Issue:** The easy pixel trace shows: `(1 - 0.95)^2 * [-log(0.95)] = 0.0025 × 0.051 = 0.000128`. The actual value of `-log(0.95)` is 0.0513 (natural log). The lesson truncates to 0.051 (dropping the 3). While the final result `0.000128` is correct when computed from the more precise value (0.0025 × 0.0513 = 0.000128), the displayed intermediate `0.0025 × 0.051 = 0.0001275`, not `0.000128`. So the trace shows three numbers where the product of the first two does not equal the third.

This is a much smaller issue than the iteration 1 focal loss finding (which was about sign consistency). The pedagogical point (14,600× ratio) is completely unaffected. But in a lesson that now goes to significant effort to make traces verifiable, this is a minor arithmetic inconsistency.

**Student impact:** Low. Most students will not multiply `0.0025 × 0.051` by hand. Those who do will get `0.0001275`, round to `0.000128`, and likely not notice. But if they compute `0.0025 × 0.051 = 0.000127` (truncating instead of rounding), they will get a slightly different answer than the lesson shows.
**Suggested fix:** Show `-log(0.95) ≈ 0.051` as `0.0513` for consistency, or show the product as `≈ 0.000128` which is technically what you get with the more precise intermediate.

#### [POLISH] — Aside y-coordinate inconsistency (0.505 vs 0.5049)

**Location:** Aside TipBlock at line 684 uses `sin(pi × 0.505)` and `cos(pi × 0.505)`, while the main content Step 1 at line 622 uses `p_y = 0.5049`.
**Issue:** The aside references `0.505` (3 decimal places) while the main content uses `0.5049` (4 decimal places) for the same normalized y-coordinate. This creates a minor inconsistency — the student may wonder whether the value is 0.505 or 0.5049.
**Student impact:** Very minor. The aside is referencing the same value with less precision. No confusion about the concept, just a visual mismatch.
**Suggested fix:** Update the aside to use `0.5049` for consistency with the main content.

#### [POLISH] — Check 1 Question 2 answer says "~400ms" but the sum is ~150 + 5×50 = 400ms, not "approximately"

**Location:** Check 1, Question 2 answer (lines 1720-1726).
**Issue:** The question asks about clicking 5 objects. The answer says "~150ms for the image encoding + 5 × 50ms for the mask decoder = ~400ms." The arithmetic is exact: 150 + 250 = 400. Using "~" twice and then producing an exact sum creates a minor logical hiccup — if both components are approximate, the sum should be "~400ms" for the right reason (because the individual times are approximate), but the lesson presents it as `150 + 5 × 50 = 400` which is exact arithmetic, then wraps it in `~`. This is fine and arguably correct (the 150ms and 50ms are approximations), but the earlier amortization section at line 499 presents the equivalent calculation for 3 prompts as `~300ms` and also notes "not 3 × 200ms = 600ms." The check answer says "Not 5 × 200ms = 1,000ms" — but earlier the lesson never stated the combined time is 200ms per prompt; it said ~150ms + ~50ms. The "200ms" framing makes sense (150 + 50 = 200) but it is derived, not stated, in the lesson.
**Student impact:** Very minor. The student can follow the math. The "200ms" framing is intuitive (total without amortization). No real confusion.
**Suggested fix:** No fix needed. This is technically correct and communicates well. Flagging only for completeness.

#### [POLISH] — The "Putting it all together" section could reference the loss function

**Location:** End-to-end trace (lines 1088-1164).
**Issue:** The end-to-end trace walks through Steps 1-5 from click to mask output. It does not mention the loss function at all, which makes sense since the trace is about inference, not training. However, the closing paragraph (lines 1145-1152) says "Every operation in this pipeline uses building blocks you have traced before." This is a natural place to briefly note: "And during training, the output masks are compared to ground truth using the focal + dice loss with minimum-loss assignment you will see next." This would connect the inference pipeline to the training section that follows.
**Student impact:** Very minor. The loss function section follows naturally. But a forward reference would help the student anticipate why the loss function matters for this pipeline.
**Suggested fix:** Optional. Add one sentence at the end of the synthesis: "During training, these 3 masks are compared to ground truth via the loss function we examine next."

### Review Notes

**Resolution of iteration 1 findings:**

All 6 findings from iteration 1 were addressed in the fix pass:

1. **[CRITICAL] Incorrect Fourier PE traced values (wrong sign on cos):** The critical sign error is fixed — cos(pi × 0.5049) now correctly shows -0.016 with explanatory text about the pi/2 crossing. However, three other values flagged as "minor rounding discrepancies" in the same finding were NOT corrected to accurate 3-decimal-place values. The fix notes say "Updated all Band 0 and Band 1 values to use consistent precision with ≈ rounding" but the values 0.500, 0.866, and -0.500 are unchanged from the pre-fix version. See Improvement finding #1 above.

2. **[IMPROVEMENT] Focal loss trace sign/alpha consistency:** Well-executed. The rewritten formula with `[-log(p_t)]` makes the trace internally consistent. The note about alpha being set aside is clear. One minor intermediate value issue remains (see Improvement finding #2 above).

3. **[IMPROVEMENT] Memory attention dimension mismatch:** Fully resolved. The memory bank projection from [N*4096, 64] -> linear -> [N*4096, 256] is now explicitly shown (lines 1957-1961). The full dimension chain is visible and correct.

4. **[IMPROVEMENT] No end-to-end prompt-to-mask trace:** Fully resolved. The "Putting it all together: click to mask" subsection (lines 1088-1164) is a strong synthesis. It references concrete values from earlier traces, includes timing, and connects all three components. The aside "Composition, Not Invention" is a well-placed insight.

5. **[POLISH] p_t piecewise notation clarification:** Fully resolved. Lines 1393-1398 add clear plain-English explanation.

6. **[POLISH] Memory mechanism "of course" moment:** Fully resolved. Lines 2013-2018 include the "of course" moment naturally in the occlusion paragraph.

**Did fixes introduce new issues?**

No new structural or conceptual issues were introduced. The only issues found in this iteration are:
- Residual inaccurate rounding from pre-fix values that were claimed to be fixed but were not actually updated
- A very minor focal loss intermediate value precision issue
- An aside/main-content inconsistency in the y-coordinate value

**What works well (fresh evaluation):**

- The overall lesson structure and flow are strong. The segmentation primer -> bottleneck hook -> architecture overview -> detailed mechanics (prompt encoding, mask decoder, loss) -> data engine -> evolution -> synthesis arc is clear and well-paced.

- The technical depth now matches the course standard. LaTeX formulas, PyTorch pseudocode with tensor shape annotations, and traced computations are present throughout. The loss function section (focal + dice + minimum-loss + IoU prediction) is a well-motivated, well-structured addition.

- The end-to-end trace (Steps 1-5, lines 1088-1164) is the strongest structural element. It transforms isolated component understanding into an integrated pipeline mental model.

- The three "of course" moments (amortization at line 507, multi-mask at line 1260, SAM 3 trajectory at line 2097) plus the fourth (memory mechanism at line 2013) are well-distributed at their natural locations.

- The cookie cutter analogy (developed in the hook at lines 321-337, echoed in the summary at line 2460) serves as an effective conceptual anchor.

- The shirt-button multi-mask example (lines 1239-1263) is distinct from the end-to-end mug trace, avoiding the redundancy flagged in the pre-rework iteration 2.

- The "red car" ComparisonRow (lines 2143-2166) makes the SAM 1 vs SAM 3 capability gap concrete and visceral.

- The explicit "SAM is not classification" correction (lines 237-243) is planted early, before the architecture is introduced.

- The additive framing note before SAM 2 (lines 1856-1864) sets correct expectations.

- All five planned misconceptions are addressed at appropriate locations: (1) SAM is classification -> segmentation primer, (2) SAM needs retraining -> promptable interface section, (3) encoder runs per prompt -> amortization, (4) decoder is a U-Net -> WarningBlock, (5) versions are independent -> additive framing note.

- Modality count for core concepts is strong: promptable segmentation has 5+ modalities, prompt encoding has 4, mask decoder has 4, loss function has 4.

- The focal loss trace, even with the minor intermediate precision issue, effectively conveys the 14,600× ratio that makes the focusing behavior visceral.

- Em dash formatting is correct throughout (no spaces).

- All interactive elements (details/summary) have cursor-pointer styling.

- The lesson stays within its updated scope boundaries.

**No notebook expected:** The planning document does not specify exercises or a Practice section, so the absence of a notebook is not a finding.
