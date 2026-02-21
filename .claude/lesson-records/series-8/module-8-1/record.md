# Module 8.1: Vision & Vision-Language Models -- Record

**Goal:** The student can explain how modern vision-language models bridge image and text understanding, tracing the architectural and loss-function innovations that make contrastive pretraining practical and effective, and how these models serve as building blocks for larger systems.
**Status:** Complete (2 of 2 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Sigmoid loss for contrastive learning (replacing softmax cross-entropy with per-pair binary classification via sigmoid + BCE) | DEVELOPED | siglip-2 | Core concept. Taught via "multiple choice vs true/false" analogy, side-by-side code (one line differs), 4x4 matrix SVG diagrams, concrete cell-level loss tracing with real numbers, and the bucket vs independent gauges spatial metaphor. Student can explain the mechanism, trace through an example, and articulate why it removes batch-size dependency. |
| Batch-size dependency in softmax contrastive loss (why CLIP's softmax denominator creates a structural requirement for large batches) | DEVELOPED | siglip-2 | Taught by walking through the softmax denominator at batch=4 vs batch=32,768, showing how each negative's share of probability mass shrinks. GradientCards compare the two scales. Student understands this is a mathematical property of the normalization, not a data quantity issue. |
| SigLIP 2 multi-stage training (Stage 1: contrastive pretraining, Stage 2: self-distillation, Stage 3: multi-resolution fine-tuning) | INTRODUCED | siglip-2 | Three-stage pipeline presented via GradientCards. Student knows the stages and why each helps, but has not implemented or deeply analyzed any stage. |
| Self-distillation (model teaches itself using soft targets from a previous training checkpoint) | INTRODUCED | siglip-2 | Explained inline as a NOT-PREVIOUSLY-TAUGHT concept. Distinguished from standard knowledge distillation (same architecture, no separate larger model). Grounded with concrete example: hard target [0, 0, 1, 0] vs soft target [0.05, 0.10, 0.70, 0.15] for "golden retriever on a beach." Student can explain in own words. |
| Multi-resolution training (processing images at multiple resolutions during later training stages) | INTRODUCED | siglip-2 | Brief explanation. Student knows this helps develop robust features across image sizes and aspect ratios. |
| SigLIP as downstream building block (vision encoder for VLMs like PaliGemma) | INTRODUCED | siglip-2 | Three downstream uses covered: zero-shot classification, image retrieval, VLM vision backbone. Student understands why contrastive pretraining produces vision features naturally aligned with language. Connected to transfer learning "hire experienced, train specific" analogy. |
| Contrastive loss formula (symmetric cross-entropy on similarity matrix) — DEEPENED | DEVELOPED | siglip-2 | Was INTRODUCED from clip (6.3.3). Gap resolution: dedicated recap walks through the softmax denominator, showing how every similarity in the row affects every probability. Student now understands the normalization mechanism deeply enough to explain why it creates batch-size coupling. |
| Promptable segmentation architecture (ViT encoder + prompt encoder + lightweight mask decoder; one model segments any object given a spatial or text prompt) | DEVELOPED | sam-3 | Core concept. Taught via three-component GradientCards, Mermaid architecture diagram, full end-to-end tensor dimension trace (click to mask), PyTorch pseudocode for prompt encoding and mask decoder forward pass. Student can trace the complete pipeline: image [3, 1024, 1024] -> ViT -> embedding [256, 64, 64] -> prompt tokens -> 2-layer decoder -> 3 masks [3, 256, 256]. |
| Fourier positional encoding for spatial prompts (sin/cos at 128 frequency bands encoding 2D coordinates) | DEVELOPED | sam-3 | Traced with real numbers: click (342, 517) normalized to (0.334, 0.505), first two frequency bands computed step-by-step with actual sin/cos values (0.864, 0.498, 1.000, -0.016...). LaTeX formula, PyTorch pseudocode, connection to transformer sinusoidal PE ("same idea, applied to 2D space"). Student sees negative cosine values and understands what they encode spatially. |
| Mask decoder cross-attention mechanism (three attention operations per layer: self-attention among tokens, token-to-image, image-to-token) | DEVELOPED | sam-3 | Full tensor dimension trace: 5 tokens [5, 256] querying 4,096 image positions [4096, 256], producing [5, 4096] attention scores. Three PhaseCards for the three operations. PyTorch pseudocode for full forward pass. Mask generation via dot product between learned mask tokens and upsampled features. Connected to "same formula, different source for K and V" from text conditioning. |
| Focal loss (down-weighting well-classified pixels via (1-p_t)^gamma modulation) | DEVELOPED | sam-3 | LaTeX formula, concrete traced computation on two pixels (easy background p_t=0.95 -> loss 0.000128, hard foreground p_t=0.1 -> loss 1.865, ratio ~14,600x). Connection to cross-entropy the student knows. Student understands gamma controls suppression aggressiveness. |
| Dice loss (mask-level overlap metric as a loss function) | DEVELOPED | sam-3 | LaTeX formula with smoothing term. Explained as global complement to focal loss's per-pixel signal: "focal says 'fix this pixel,' dice says 'the overall shape is wrong.'" Combined loss: 20x focal + 1x dice. |
| Multi-mask minimum-loss training (backpropagate through best mask only, allowing specialization) | DEVELOPED | sam-3 | L = min(L1, L2, L3) formula. Explained why min instead of mean: mean pressures identical masks, min allows specialization. Connected to the three granularity levels (part, object, scene). PyTorch pseudocode for full loss computation. |
| Image segmentation as a task (per-pixel binary mask output, contrast with classification and detection) | INTRODUCED | sam-3 | Gap resolution from MENTIONED (U-Net lesson). Three GradientCards comparing classification/detection/segmentation on a dog photograph. Explicit "SAM is not classification" correction planted early. Semantic/instance/panoptic variants mentioned briefly. Connected to U-Net spatial output. |
| SA-1B data engine (three-stage human-AI annotation partnership scaling to 1B masks) | INTRODUCED | sam-3 | Three PhaseCards (assisted-manual, semi-automatic, fully automatic). Key insight: "model improves data, better data improves model." Scale comparison: SA-1B has 400x more masks than COCO. Distinguished from language model scaling ("not just add more internet text"). |
| SAM 2 video memory mechanism (memory encoder, memory bank, memory attention via cross-attention over stored frames) | INTRODUCED | sam-3 | Three PhaseCards for memory encoder/bank/attention. Tensor shapes traced: memory bank [N*4096, 64] projected to [N*4096, 256], cross-attention with current frame [4096, 256]. Occlusion handling explained as natural consequence of cross-attention. Full frame pipeline in code. Higher technical detail than typical INTRODUCED because of depth rework. |
| SAM 3 concept-level segmentation (text prompts, exemplar prompts, open-vocabulary, all-instance detection) | INTRODUCED | sam-3 | Concrete "red car" ComparisonRow: SAM 1 (4 clicks, 4 masks) vs SAM 3 (one text prompt, 4 masks with unique IDs). Perception Encoder distinguished from CLIP dual encoders. Performance numbers (840M params, 30ms inference). SA-Co dataset briefly described. |
| IoU (Intersection over Union) as mask quality metric | INTRODUCED | sam-3 | Expanded inline: "IoU -- Intersection over Union -- estimating how well the mask fits the true object." IoU prediction trained with MSE loss against actual IoU computed during training. Student understands it as the confidence ranking mechanism for multi-mask output. |

## Per-Lesson Summaries

### Lesson 1: SigLIP 2 (siglip-2)

**Type:** BUILD | **Cognitive load:** 2 new concepts

**Concepts taught:**
1. Sigmoid loss for contrastive learning (DEVELOPED) — the core innovation. Replacing softmax cross-entropy with per-pair sigmoid binary classification removes batch-size dependency.
2. SigLIP 2 training improvements (INTRODUCED) — multi-stage training, self-distillation, multi-resolution. A different kind of advance: training recipe rather than loss function.

**Concepts deepened:**
- Contrastive loss formula: INTRODUCED -> DEVELOPED (gap resolution via detailed softmax denominator walkthrough)

**Mental models established:**
- "Multiple choice vs true/false" — CLIP's softmax loss is a multiple-choice exam (pick the correct match from N options, difficulty depends on N). SigLIP's sigmoid loss is a true/false exam (for each pair, answer: match or not? each question stands alone).
- "Bucket vs independent gauges" — softmax = bucket containing exactly 1 unit of probability divided among all items (adding items dilutes each share). Sigmoid = N independent gauges each reading 0-1 on their own (adding gauges changes nothing).

**Analogies used:**
- "Multiple choice vs true/false" (core analogy, carries entire lesson)
- "Bucket vs independent gauges" (spatial modality for batch-size independence)
- Extended "name tag matching at a conference" from CLIP lesson
- Extended "hire experienced, train specific" from transfer learning for SigLIP as VLM backbone
- Connected softmax competition property (from attention, classification, temperature) to batch-size coupling

**How concepts were taught:**
- Sigmoid loss: 6 modalities — verbal/analogy (multiple choice vs true/false), visual (side-by-side 4x4 SVG matrix diagrams showing connected vs independent cells), symbolic/code (PyTorch side-by-side, one line differs), concrete example (tracing loss for individual cells with real numbers), geometric/spatial (bucket vs gauges), intuitive (two "of course" moments).
- Batch-size dependency: GradientCards comparing batch=4 vs batch=32,768, showing probability mass dilution. Explicit warning that this is a mathematical property, not a data quantity issue.
- Self-distillation: verbal explanation + concrete example (hard vs soft targets with golden retriever scenario). Distinguished from standard knowledge distillation.
- SigLIP 2 stages: three sequential GradientCards (one per stage).
- SigLIP as building block: three GradientCards for downstream uses + explanation of why contrastive pretraining produces language-aligned vision features.

**Misconceptions addressed:**
1. "SigLIP is completely different from CLIP" — addressed early in recap with explicit statement that only the loss function changes.
2. "Sigmoid loss wastes negatives" — addressed with ComparisonRow showing CLIP has 2N loss terms vs SigLIP has N^2 (more training signal per batch).
3. "Batch-size dependency is a data quantity issue" — addressed with WarningBlock explaining it is a mathematical property of softmax normalization.
4. "SigLIP means small-batch training" — addressed with boundary example GradientCard showing SigLIP at batch 32,768 is even better than at 256.
5. "Self-distillation = knowledge distillation (large teacher -> small student)" — addressed with ConceptBlock distinguishing the two.

**What is NOT covered (relevant for future lessons):**
- Implementing SigLIP from scratch
- Training a contrastive model from scratch
- Other contrastive learning variants (MoCo, SimCLR, BYOL)
- The full PaliGemma or Gemini architecture
- Open-vocabulary detection as a downstream use
- Mathematical proof of batch-size independence (intuitive argument only)
- Detailed multilingual training methodology

### Lesson 2: SAM 3 (sam-3)

**Type:** BUILD | **Cognitive load:** 2-3 new concepts (underwent depth rework after initial build)

**Concepts taught:**
1. Promptable segmentation architecture (DEVELOPED) -- the core concept. One model segments any object given a spatial or text prompt. Three-component architecture: ViT image encoder (heavy, once) + prompt encoder (lightweight, per prompt) + mask decoder (lightweight, per prompt).
2. SAM's internal mechanics at full mathematical depth (DEVELOPED) -- Fourier positional encoding formula with traced computation, mask decoder's three attention operations per layer with tensor dimension trace, loss function (focal + dice + minimum-loss assignment) with traced numbers and PyTorch pseudocode.
3. Foundation model for segmentation pattern (INTRODUCED via SAM 2 memory, SAM 3 text prompting, SA-1B data engine) -- how the "massive pretraining + prompting" recipe applies to segmentation and extends from images to video to language.

**Concepts deepened:**
- Image segmentation: MENTIONED -> INTRODUCED (gap resolution via dedicated primer with classification/detection/segmentation comparison)
- Cross-attention: DEVELOPED (already) -> applied in new context (token-to-image, image-to-token, memory attention)

**Mental models established:**
- "The universal cookie cutter" -- Traditional segmentation = one cookie cutter per shape. SAM = a programmable cutter that adjusts based on what you point at. SAM 3 = describe the shape you want. Developed in hook section's main content (two paragraphs), echoed in summary.
- "Composition, not invention" -- SAM's architecture uses no novel individual components (Fourier encoding, cross-attention, transposed convolutions, dot-product decoding). The innovation is how they compose into a promptable segmentation pipeline.
- "Architecture encodes assumptions" -- Extended from CNN module. SAM's asymmetric design (heavy encoder, light decoder) encodes the assumption that you will query the same image multiple times.

**Analogies used:**
- "The universal cookie cutter" (core analogy, developed in main content)
- Extended "tokenize the image" from DiT lesson for SAM's ViT encoder
- Extended "same formula, different source for K and V" from text conditioning for mask decoder cross-attention
- Extended "architecture encodes assumptions about data" from CNN module for encoder/decoder asymmetry
- Extended "hire experienced, train specific" from transfer learning for SAM as foundation model
- GPT-3 parallel: before GPT-3, one model per NLP task; before SAM, one model per object category
- "Memory bank IS the context window, just for video instead of text" for SAM 2

**How concepts were taught:**
- Promptable segmentation: 5+ modalities -- verbal/analogy (cookie cutter, developed in 2 paragraphs), visual (Mermaid architecture diagram, GradientCards for three components, three-panel classification/detection/segmentation comparison), symbolic (LaTeX formulas for Fourier PE, cross-attention, focal loss, dice loss; PyTorch pseudocode for prompt encoding, mask decoder, loss computation), concrete example (traced click at (342, 517) through Fourier PE with actual sin/cos values; shirt-button multi-mask example with scores; "red car" SAM 1 vs SAM 3 comparison), intuitive (three distributed "of course" moments at natural locations).
- Fourier positional encoding: LaTeX formula, step-by-step traced computation (normalize -> first frequency band -> second frequency band -> project -> add type), PyTorch pseudocode, connection to transformer sinusoidal PE.
- Mask decoder: token composition entering decoder (5 tokens = 1 prompt + 3 mask + 1 IoU), three PhaseCards for three attention operations, cross-attention formula with tensor dimension trace [5, 256] x [4096, 256] -> [5, 4096], PyTorch pseudocode for full forward pass, upsampling pipeline, dot-product mask generation.
- Loss function: focal loss formula + traced computation on two pixels (easy bg 0.000128 vs hard fg 1.865, ~14,600x ratio), dice loss formula with intersection/union explanation, combined loss (20x focal + 1x dice), minimum-loss assignment formula with specialization rationale, IoU prediction MSE loss, PyTorch pseudocode for complete loss computation.
- Data engine: three PhaseCards (assisted-manual, semi-automatic, fully automatic), scale comparison (SA-1B = 400x COCO).
- SAM 2 memory: three PhaseCards (memory encoder, memory bank, memory attention), tensor shape trace for cross-attention over memory, occlusion handling, full frame pipeline in code.
- SAM 3: GradientCard for innovation, concrete ComparisonRow (SAM 1 vs SAM 3 on "red car" scenario), architecture evolution bullets, Mermaid evolution timeline.

**Misconceptions addressed:**
1. "SAM is a classification model" -- addressed early in segmentation primer: "SAM is not doing classification. It does not output a class label... It outputs a binary mask... SAM knows boundaries, not categories."
2. "SAM needs retraining for each object type" -- addressed after promptable interface: SAM learns boundaries, not categories. Any object with coherent boundaries can be segmented zero-shot.
3. "The image encoder runs per prompt" -- addressed in architecture section with timing breakdown and amortization insight.
4. "SAM's mask decoder is a U-Net" -- addressed with WarningBlock: "Not a U-Net. It is a lightweight 2-layer transformer decoder."
5. "SAM 1/2/3 are completely different architectures" -- addressed with framing note before SAM 2: "SAM's evolution is additive, not replacement."

**What is NOT covered (relevant for future lessons):**
- Implementing SAM from scratch or training end-to-end
- DETR or transformer-based detection architectures in depth
- Efficiency variants (EfficientSAM, MobileSAM)
- SAM 3D or 3D reconstruction
- The full SA-Co dataset construction methodology
- Fine-tuning SAM for specific domains
- Mathematical proof of focal loss convergence properties
- Detailed Perception Encoder architecture internals

**Build notes:** This lesson underwent a significant depth rework after initial review. The first build was accurate and well-structured but too conceptual -- it described what SAM does without showing how it works mathematically. The rework added: LaTeX formulas with traced computations, PyTorch pseudocode for three systems (prompt encoder, mask decoder, loss), tensor dimension traces, and a full loss function section (focal loss, dice loss, combined loss, min-loss assignment, IoU prediction). The loss function section was originally scoped OUT but was added per user feedback that the lesson did not match the course's standard depth. Two additional review iterations refined: (1) duplicated mug example -> shirt-button for multi-mask, (2) SAM 2 "3x fewer interactions" claim grounded qualitatively.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "Multiple choice vs true/false" (softmax = pick one from N, sigmoid = binary yes/no per pair) | siglip-2 | |
| "Bucket vs independent gauges" (softmax = fixed probability mass divided among items, sigmoid = independent readings) | siglip-2 | |
| "Two encoders, one shared space" (carried from CLIP) | clip (6.3.3) | siglip-2, sam-3 (contrasted: SAM 3 uses unified Perception Encoder, NOT dual encoders) |
| "Name tag matching at a conference" (carried from CLIP) | clip (6.3.3) | siglip-2 |
| "Hire experienced, train specific" (carried from transfer learning) | transfer-learning (3.2.3) | siglip-2, sam-3 |
| Temperature as "sharpness knob" (carried from multiple) | what-is-a-language-model (4.1.1) | siglip-2 |
| "The universal cookie cutter" (traditional segmentation = one cutter per shape, SAM = programmable cutter, SAM 3 = describe the shape) | sam-3 | |
| "Composition, not invention" (SAM's components are all familiar -- the innovation is the composition into a promptable pipeline) | sam-3 | |
| "Architecture encodes assumptions about data" (carried from CNN module; SAM's asymmetric design encodes "query same image multiple times") | what-convolutions-compute (3.1.1) | sam-3 |
| "Same formula, different source for K and V" (carried from text conditioning; extended to prompt-to-image cross-attention in SAM) | text-conditioning-and-guidance (6.3.4) | sam-3 |
| "Tokenize the image" (carried from DiT; applied to SAM's ViT image encoder) | diffusion-transformers (7.4.2) | sam-3 |
