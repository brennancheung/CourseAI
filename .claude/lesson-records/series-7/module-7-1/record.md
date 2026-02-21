# Module 7.1: Controllable Generation -- Record

**Goal:** The student can explain how ControlNet and IP-Adapter add structural and semantic conditioning to a frozen Stable Diffusion model, understanding the architectural patterns (trainable encoder copy, zero convolution, decoupled cross-attention) well enough to reason about when and why to use each technique.
**Status:** Complete (3 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Trainable encoder copy architecture (clone the frozen U-Net encoder, initialize with original weights, train the copy on spatial maps in parallel, add its outputs to the frozen encoder's skip connections at each resolution level) | DEVELOPED | controlnet | Core new concept #1. Taught via the "of course" chain: (1) spatial features live in the encoder, (2) training new spatial features requires an encoder that takes spatial maps, (3) you cannot retrain the frozen encoder, (4) so copy it, initialize from original weights, train the copy. Architecture diagram (Mermaid) with color-coded frozen (gray), trainable (violet), zero conv (amber) components and explicit "+" merge nodes. Pseudocode forward pass showing `e_i + z_i` as the only change from the standard U-Net forward pass. Parameter breakdown: ~300M trainable params (~35% of U-Net), encoder half only, not a full duplicate. |
| Zero convolution mechanism (1x1 conv initialized to all-zero weights and all-zero bias; guarantees zero initial contribution from ControlNet to frozen decoder; control signal fades in gradually as training progresses) | DEVELOPED | controlnet | Core new concept #2. Taught with concrete numerical examples at initialization (output = all zeros regardless of input, so `e_i + 0 = e_i`) and after 100 training steps (weight drifted to 0.03, small signal). "Nothing, Then a Whisper" metaphor: starts silent, gradually gains volume, then a clear voice. Explicit ComparisonRow with LoRA B=0 initialization: same principle (ensure frozen model starts unchanged), different scale (feature-level vs weight-level, ~300M vs ~1M params). WarningBlock: "Zero convolution is not a special operation--it is a standard 1x1 conv with a specific initialization." |
| ControlNet coexistence with text conditioning (ControlNet adds WHERE without replacing WHAT; timestep = WHEN via adaptive norm, text = WHAT via cross-attention, ControlNet = WHERE via additive encoder features; all three mechanisms coexist simultaneously) | INTRODUCED | controlnet | Taught via concrete example: same Canny edge map with two different text prompts producing different content/style but identical structure. Three-column GradientCard grid (WHEN/WHAT/WHERE) synthesizing all conditioning dimensions. Addresses misconception that ControlNet replaces text conditioning. |
| ControlNet as a modular, swappable component (architecture is spatial-map agnostic; same design for edges, depth, pose; one ControlNet per map type but architecture identical; swap checkpoints without retraining; SD v1.5 ControlNets work with any v1.5-compatible model but NOT with SDXL) | INTRODUCED | controlnet | ComparisonRow (Canny edge ControlNet vs depth map ControlNet) showing identical architecture, different preprocessing/training data. Extends "three translators, one pipeline" modularity concept from 6.4.1 to "four translators." Compatibility note on base model version matching. |
| Cross-attention reactivation (Q from spatial features, K/V from text embeddings; each spatial location attends independently to all text tokens; spatially-varying text conditioning) | REINFORCED | controlnet | Brief 2-3 sentence reactivation of cross-attention from 6.3.4 (~15+ lessons ago). Not re-taught; reconnected as context for understanding that ControlNet does not replace cross-attention. Framed as one of three conditioning dimensions (timestep=WHEN, text=WHAT, ControlNet=WHERE). |
| ControlNet does NOT modify frozen weights (zero conv output added to encoder features, not to weights; different from LoRA where bypass merges with weight computation via Wx + BAx; disconnect ControlNet and frozen model is bit-for-bit identical) | DEVELOPED | controlnet | Dedicated WarningBlock titled "ControlNet Does NOT Modify Frozen Weights." Explicitly contrasts feature-level addition (ControlNet) vs weight-level computation (LoRA). Addresses the primary misconception from student's deep LoRA knowledge. Check Question 2 (disconnection test) reinforces this. |
| Canny edge detection preprocessing (binary edge map from intensity gradients; two thresholds control sensitivity; too low = noisy edges, too high = sparse edges; quality directly determines ControlNet output quality) | DEVELOPED | controlnet-in-practice | Taught as a black-box tool: input photo, output edge map. Threshold tuning demonstrated with three settings (too few, good, too many) using vivid descriptions of what ControlNet produces from each. "Garbage in, garbage out" insight. Code: `cv2.Canny(image, low_threshold, high_threshold)`. Notebook Exercise 1 provides hands-on threshold tuning. |
| MiDaS depth estimation preprocessing (monocular depth map from a single photograph; lighter = closer, darker = farther; controls 3D structure, perspective, and spatial layering; no threshold tuning required) | DEVELOPED | controlnet-in-practice | Taught as a black-box tool: input photo, output grayscale depth map. Contrasted with Canny (no user-facing thresholds). When to use: landscape composition, scene depth, spatial arrangement over exact contours. Code: `pipeline("depth-estimation")`. Notebook Exercise 2 compares depth output to Canny and OpenPose. |
| OpenPose skeleton detection preprocessing (body keypoints connected into stick-figure skeleton; identifies joints and connections; optionally includes hand/face keypoints; controls human body pose) | DEVELOPED | controlnet-in-practice | Taught as a black-box tool: input photo with visible person, output skeleton on black background. Pose transfer highlighted as the "most magical" application: extract skeleton from one person, generate a different person in the same pose. Code: `OpenposeDetector.from_pretrained()`. Notebook Exercise 2 provides hands-on use. |
| Conditioning scale / control-creativity tradeoff (manual volume knob for spatial control at inference time; low scale = spatial map is a suggestion, high scale = rigid adherence; sweet spot typically 0.7-1.0; same concept as CFG guidance scale but for spatial control) | DEVELOPED | controlnet-in-practice | Connected to three prior concepts: (1) CFG guidance scale as the same kind of dial for text, (2) "Nothing, Then a Whisper" zero conv fade-in extended to inference-time manual control, (3) img2img strength parameter as the same "how much to deviate" idea. Six-value sweep (0.3 to 2.0) with vivid descriptions of what each produces. Misconception addressed: scale=1.0 does not disable text conditioning. Notebook Exercise 3 provides hands-on sweep. |
| Multi-ControlNet stacking (multiple ControlNets independently contribute additive features to skip connections; compose by summation: `e_i + z_i_canny + z_i_depth`; complementary maps from same source image compose well; conflicting maps or excessive scales produce artifacts) | DEVELOPED | controlnet-in-practice | Taught as extension of modularity: "four translators" extends to adding more translators. API pattern: list of ControlNets, list of images, list of per-ControlNet conditioning scales. ComparisonRow contrasting complementary vs conflicting stacking. Practical guidelines: start with one, moderate scales (0.5-0.8), same source image. Misconception addressed: stacking is not doubling control strength. Notebook Exercise 4 provides hands-on stacking. |
| ControlNet coexistence with text conditioning (WHEN/WHAT/WHERE framing) | DEVELOPED | controlnet-in-practice | Elevated from INTRODUCED (controlnet) to DEVELOPED. Student directly manipulated the text-spatial balance via conditioning scale. Two-prompt experiment at same scale demonstrates text conditioning remains active. The conditioning scale is the "volume knob for WHERE" while text prompt carries WHAT. |
| ControlNet as a modular, swappable component (map-agnostic architecture, checkpoint swapping) | DEVELOPED | controlnet-in-practice | Elevated from INTRODUCED (controlnet) to DEVELOPED. Student saw identical pipeline code for three different map types (only preprocessor and checkpoint change). Multi-ControlNet stacking demonstrates composability. "Notice What Stays the Same" section concretely shows the pipeline does not know or care what map type it receives. |
| Other ControlNet preprocessor types (lineart, scribble, normal maps, segmentation maps) | MENTIONED | controlnet-in-practice | Named for vocabulary breadth. Each has its own checkpoint. Same pattern: extract spatial map, load matching checkpoint, run pipeline. Not developed. |
| Decoupled cross-attention (parallel K/V projections for image embeddings alongside existing text K/V projections; shared Q; two attention outputs combined via weighted addition: `text_out + scale × image_out`) | DEVELOPED | ip-adapter | Core new concept #1. Taught via "of course" chain: (1) CLIP image embeddings live in shared text-image space, (2) cross-attention reads from K/V, (3) add separate K/V projections for image embeddings, (4) two outputs added with scale, (5) Q shared because spatial features ask the same question of both sources. Architecture diagram (Mermaid) with color-coded frozen (gray), trainable (violet), combined output (green). Pseudocode forward pass. Shape walkthrough at 16×16 resolution: Q [256, d_k], K_text/V_text [77, d_k], K_image/V_image [257, d_k]. "Two reference documents, one reader" analogy extending the established reference document framing from 4.2.5 and 6.3.4. |
| IP-Adapter as a lightweight, general-purpose image conditioning adapter (trained once on millions of image-text pairs, works with ANY reference image at inference; ~22M trainable params; purely additive to frozen SD model) | DEVELOPED | ip-adapter | Core new concept #2. Taught by contrast with per-concept methods: textual inversion trains per concept (768 params), LoRA trains per style/subject (~1M params), IP-Adapter trains once and works with any image (~22M params). Dedicated WarningBlock: "Trained Once, Works With Any Image." Three ComparisonRows: IP-Adapter vs LoRA, vs textual inversion, vs ControlNet. |
| IP-Adapter scale parameter (volume knob for image influence; scale=0 means text only, scale=1.0 means strong image influence; text always contributes at full strength; addition not averaging) | DEVELOPED | ip-adapter | Connected to conditioning scale pattern from controlnet-in-practice ("same knob, new context"). Three-value sweep (0.0, 0.5, 1.0) with vivid descriptions using ornate vase example. Dedicated WarningBlock: "Addition, Not Averaging"--text_out has no scale factor, scale=0.5 means "full text, plus half-strength image." |
| CLIP image encoder sequence output (ViT-H/14 produces 257 tokens: 256 patches + 1 CLS token; IP-Adapter uses the full sequence, not just the pooled CLS vector; small trainable projection maps to U-Net cross-attention dimensions) | INTRODUCED | ip-adapter | Gap fill from CLIP lesson (6.3.3) where only the pooled vector was covered. Brief recap paragraph paralleling the text encoder's 77 token sequence. Not deeply developed--used as a black box input to the K/V projections. |
| IP-Adapter coexistence with text conditioning (image provides WHAT-IT-LOOKS-LIKE, text provides WHAT, decoupled not replaced; same reference image with different text prompts produces clearly different outputs) | DEVELOPED | ip-adapter | Taught via golden retriever example: same reference photo with "painting of a dog in a garden" vs "dog running on a beach at sunset." Parallels ControlNet coexistence example (same edge map, two prompts). Misconception addressed: "Image Prompting Is Not Image Replacement." |
| IP-Adapter vs LoRA architectural distinction (LoRA modifies EXISTING W_K/W_V via bypass Wx+BAx, changes how text is processed; IP-Adapter adds NEW W_K_image/W_V_image, adds a new information source; remove LoRA and text output changes, remove IP-Adapter and text output is identical) | DEVELOPED | ip-adapter | Highest-priority comparison. ComparisonRow with five-point contrast. "Remove and check" test as the concrete differentiator. "Same highway, add a detour" (LoRA) vs "build a second highway" (IP-Adapter). Addresses the primary misconception from student's deep LoRA knowledge. |
| Four conditioning channels in frozen SD (WHEN via timestep/adaptive norm, WHAT via text/cross-attention K/V, WHERE via ControlNet/additive encoder features, WHAT-IT-LOOKS-LIKE via IP-Adapter/decoupled cross-attention K/V) | DEVELOPED | ip-adapter | Module-level synthesis. Extends WHEN/WHAT/WHERE to include WHAT-IT-LOOKS-LIKE. All four channels are additive, composable, and target different parts of the U-Net. InsightBlock: "Four Conditioning Channels." ModuleCompleteBlock listing all three lessons. |
| Zero initialization pattern across adapters (zero convolution in ControlNet, B=0 in LoRA, zero K/V projections in IP-Adapter; same principle: new components start contributing nothing so the frozen model is undisturbed at training start) | REINFORCED | ip-adapter | Third reinforcement of the pattern (first: LoRA 4.4.4, second: ControlNet 7.1.1). InsightBlock: "Same Safety Pattern." Connected in training overview and summary. |
| IP-Adapter is NOT img2img (encodes reference with CLIP for semantic representation, not VAE for pixel-level; reference image never enters denoising loop as a latent; IP-Adapter with pure random noise still produces output influenced by reference) | DEVELOPED | ip-adapter | Dedicated WarningBlock: "Not Img2Img." Placed after the architecture explanation so the student can contrast the two pathways (CLIP -> cross-attention vs VAE -> starting latent). |
| IP-Adapter composability with ControlNet (IP-Adapter provides WHAT-IT-LOOKS-LIKE via cross-attention K/V, ControlNet provides WHERE via encoder features; different U-Net targets, composable; e.g., reference photo for visual style + edge map from different image for spatial structure) | INTRODUCED | ip-adapter | Transfer question and Exercise 4 (Independent). Student predicts composability from architectural knowledge. Not deeply practiced beyond one notebook exercise. |
| IP-Adapter variants (IP-Adapter Plus for higher fidelity, IP-Adapter Face ID for face identity preservation, IP-Adapter + ControlNet composition) | MENTIONED | ip-adapter | Named for vocabulary breadth. Brief paragraph. Not developed. |

## Per-Lesson Summaries

### controlnet (Lesson 1)
**Status:** Built, reviewed (PASS on iteration 3)
**Cognitive load:** STRETCH (2 new concepts: trainable encoder copy architecture, zero convolution mechanism)
**Notebook:** `notebooks/7-1-1-controlnet.ipynb` (4 exercises: inspect ControlNet architecture, verify zero-initialization property, trace forward pass, ControlNet vs vanilla SD comparison)

**Concepts taught:**
- Trainable encoder copy architecture (DEVELOPED)--clone the U-Net encoder, initialize with original weights, process spatial maps in parallel, add outputs to frozen skip connections
- Zero convolution mechanism (DEVELOPED)--1x1 conv initialized to all zeros, guarantees zero initial contribution, control signal fades in gradually
- ControlNet coexistence with text conditioning (INTRODUCED)--WHEN/WHAT/WHERE framing, three conditioning dimensions coexisting
- ControlNet as modular, swappable component (INTRODUCED)--map-agnostic architecture, checkpoint swapping, base model compatibility
- Cross-attention reactivation (REINFORCED)--brief callback to Q/K/V from 6.3.4

**Mental models established:**
- "The 'of course' chain"--each architectural decision follows from the previous constraint: spatial features live in the encoder, you cannot retrain the frozen encoder, so copy it, initialize at zero so nothing breaks. The student could have designed this themselves.
- "Nothing, Then a Whisper"--zero convolution starts silent (all zeros), gradually gains volume as training progresses. The frozen model hears nothing at first, then a whisper, then a clear voice.
- "WHEN/WHAT/WHERE"--three conditioning dimensions (timestep via adaptive norm, text via cross-attention, ControlNet via additive encoder features), three mechanisms, all coexisting.

**Analogies used:**
- "Nothing, then a whisper, then a clear voice" (zero convolution fade-in; replaces "training wheels" analogy which was rejected for inaccuracy)
- "Highway and detour" extended to "detour that starts as a dead end" (zero convolution as a bypass that starts disconnected)
- "Bottleneck decides WHAT, skip connections decide WHERE" extended to "ControlNet influences WHERE via the skip connections"
- "Three translators, one pipeline" extended to "four translators" with ControlNet as the spatial translator

**How concepts were taught:**
- Quick recap: 2-3 sentence cross-attention reactivation (Q from spatial features, K/V from text). Framed as two existing conditioning dimensions (WHEN, WHAT) with the transition: "But what about WHERE?"
- Hook: spatial control problem. Text is the wrong tool for spatial precision--you cannot control where the house sits, what angle the hill has, where the horizon falls. Challenge: "You have all the pieces to design this yourself."
- Design challenge: `<details>` reveal with three premises (U-Net encoder extracts spatial features, you know how to freeze models, additive connections can be initialized safely). Student prompted to think before reading the solution.
- Architecture explanation: "of course" chain--six steps, each following from the previous constraint. Mermaid diagram with color-coded components and explicit "+" merge nodes (ADD1/ADD2/ADD3). Pseudocode forward pass showing `e_i + z_i` as the only change from standard U-Net. ComparisonRow (frozen SD model vs ControlNet addition) with parameter counts.
- Check: three predict-and-verify questions (initial contribution = zero, disconnection test = no change, weight initialization from original = faster training).
- Zero convolution: definition (1x1 conv, weights=0, bias=0). Concrete numerical example at initialization and after 100 training steps. "Nothing, Then a Whisper" InsightBlock. WarningBlock: ControlNet does NOT modify frozen weights (features vs weights distinction, contrast with LoRA). ComparisonRow: LoRA B=0 vs zero convolution (same principle, different scale).
- Coexistence: same edge map, two prompts producing different content with identical structure. Three-column GradientCard grid (WHEN/WHAT/WHERE). Map-agnostic architecture with Canny vs depth ComparisonRow. Compatibility note on base model versions.
- Transfer check: style vs structure question (ControlNet is not for style), architecture justification (why encoder architecture not MLP--multi-resolution features needed at skip connections).
- Practice: notebook with 4 exercises in Guided -> Guided -> Supported -> Independent progression. Each exercise isolates or integrates a core concept.

**Misconceptions addressed:**
1. "ControlNet fine-tunes or modifies the original SD model's weights"--dedicated WarningBlock contrasting feature-level addition vs weight-level computation. Disconnect test in Check Q2. Bit-for-bit identical frozen model.
2. "ControlNet replaces text conditioning"--two-prompt example with same edge map. Three-column WHEN/WHAT/WHERE GradientCards.
3. "The trainable encoder copy is a second U-Net (doubles the model)"--ComparisonRow with parameter counts. Encoder half only (~35%), not full duplicate. One decoder runs.
4. "Zero convolution is a complex mechanism"--TipBlock: just a 1x1 conv with zero initialization. The cleverness is the initialization, not the operation.
5. "ControlNet needs to be trained from scratch for each base model"--compatibility note: SD v1.5 ControlNets work with v1.5-compatible models but not SDXL.

**What is NOT covered (deferred):**
- How to preprocess images into spatial maps (Canny, depth, OpenPose)--lesson 2
- Conditioning scale parameter / control-creativity tradeoff--lesson 2
- Stacking multiple ControlNets--lesson 2
- IP-Adapter / image-based conditioning--lesson 3
- T2I-Adapter (mentioned for vocabulary only in references)
- Training a ControlNet from scratch

**Review notes:**
- Iteration 1: MAJOR REVISION--1 critical (notebook missing, resolved by parallel build), 3 improvement (Mermaid merge point ambiguity, missing explicit Misconception #1 WarningBlock, training wheels analogy inaccurate), 2 polish (spaced em dashes in code comments, z4 computed but unused in pseudocode)
- Iteration 2: NEEDS REVISION--0 critical, 2 improvement (notebook training wheels analogy not synced with lesson fix, notebook Exercise 4 OOM risk from dual pipeline loading), 1 polish (notebook spaced em dashes)
- Iteration 3: PASS--all findings resolved. Lesson and notebook aligned on terminology, VRAM management production-quality, zero findings.

### controlnet-in-practice (Lesson 2)
**Status:** Built, reviewed (PASS on iteration 2)
**Cognitive load:** CONSOLIDATE (0 new concepts; practical skill-building only)
**Notebook:** `notebooks/7-1-2-controlnet-in-practice.ipynb` (5 exercises: Canny threshold tuning, three-preprocessor comparison, conditioning scale sweep, multi-ControlNet stacking, independent composition)

**Concepts taught:**
- Canny edge detection preprocessing (DEVELOPED)--threshold tuning, quality-output relationship, "garbage in, garbage out"
- MiDaS depth estimation preprocessing (DEVELOPED)--monocular depth maps for 3D structure and layering control
- OpenPose skeleton detection preprocessing (DEVELOPED)--body keypoint extraction for pose transfer
- Conditioning scale / control-creativity tradeoff (DEVELOPED)--volume knob for spatial control, sweet spot 0.7-1.0, connected to CFG guidance scale, img2img strength, and zero conv fade-in
- Multi-ControlNet stacking (DEVELOPED)--additive feature composition, complementary vs conflicting maps, per-ControlNet scales
- ControlNet coexistence with text conditioning (elevated INTRODUCED -> DEVELOPED)--directly manipulated text-spatial balance via conditioning scale
- ControlNet as modular, swappable component (elevated INTRODUCED -> DEVELOPED)--identical pipeline code for all map types, checkpoint swapping, stacking composability
- Other preprocessor types: lineart, scribble, normal maps, segmentation (MENTIONED)--named for breadth only

**Mental models established:**
- No new mental models. This lesson extends and concretizes three existing models from lesson 1:
  - "WHEN/WHAT/WHERE" extended with a "volume knob for WHERE" (conditioning scale)
  - "Nothing, Then a Whisper" extended from training-time fade-in to inference-time manual control
  - "Four translators, one pipeline" extended to multi-translator stacking (adding a fifth translator)

**Analogies used:**
- "Volume knob" for conditioning scale (extends "Nothing, Then a Whisper" from training to inference)
- "Two knobs, one mixing board" (CFG guidance scale for text influence, conditioning scale for spatial influence)
- "Socket and plug" (pipeline is the socket, ControlNet checkpoint is the plug--swap for different behavior)
- "Garbage in, garbage out" (preprocessing quality determines output quality)

**How concepts were taught:**
- Hook: "The Missing Piece"--student has seen edge maps but never knew where they came from. One-line Canny call reveals the anticlimactic answer, then pivots to: the quality of this step determines everything.
- Preprocessors: each gets a GradientCard (what it extracts, what it controls, when to use) plus a code snippet. Canny gets the deepest treatment with three threshold settings (too few, good, too many) described vividly as predictions for the notebook. MiDaS contrasted with Canny (no thresholds). OpenPose highlighted for pose transfer. Mermaid diagram shows the full workflow: photograph -> preprocessor -> spatial map -> ControlNet -> SD pipeline.
- "Notice What Stays the Same": explicit code block showing identical pipeline call with swappable checkpoint comment. Addresses misconception that different map types need different APIs.
- Check: three scenario-based preprocessor selection questions (fantasy castle, dancer pose, landscape layering).
- Conditioning scale: connected to CFG guidance scale ("same dial, different signal"), to "Nothing, Then a Whisper" ("manual volume knob at inference"), and to img2img strength ("how much to deviate from this reference"). Six-value sweep (0.3-2.0) with vivid descriptions framed as predictions. WarningBlock addressing misconception that scale=1.0 disables text. Code example showing same edges + two prompts producing different content.
- Check: two predict-and-verify questions (high scale on detailed edges, low scale on same map).
- Multi-ControlNet stacking: summation formula `e_i + z_i_canny + z_i_depth`. Code pattern with list API. ComparisonRow: complementary vs conflicting stacking. Practical guidelines (start with one, moderate scales, same source image). WarningBlock addressing misconception that stacking doubles control strength.
- Check: "debugging a friend's stack"--transfer question applying all stacking guidelines.
- Practice: five notebook exercises in Guided -> Guided -> Supported -> Supported -> Independent progression. Exercises are cumulative (Exercise 1's best map feeds Exercise 3, Exercise 2's depth map feeds Exercise 4).

**Misconceptions addressed:**
1. "Conditioning scale 1.0 means the model ignores the text prompt"--WarningBlock with two-prompt experiment at same scale. WHEN/WHAT/WHERE: scale controls WHERE volume, not whether WHAT is on.
2. "Higher conditioning scale is always better"--six-value sweep showing degradation above 1.5 (rigid textures, ghost lines, mechanical quality).
3. "Different spatial map types need different APIs"--"Notice What Stays the Same" section with identical code block. Only preprocessor and checkpoint change.
4. "Stacking two ControlNets doubles control strength"--ComparisonRow: complementary (different types of control) not redundant (more of the same). Conflicting maps at high scales produce artifacts.
5. "Preprocessor quality does not matter much"--Canny threshold comparison with vivid descriptions of degraded output. "Garbage in, garbage out" as the lesson's most important practical insight.

**What is NOT covered (deferred):**
- ControlNet architecture internals (lesson 1; brief callbacks only)
- How preprocessors work internally (Canny algorithm, MiDaS architecture, OpenPose architecture)--used as black-box tools
- Training a ControlNet from scratch
- IP-Adapter / image-based conditioning--lesson 3
- T2I-Adapter or other control mechanisms
- Every possible preprocessor type (lineart, scribble, normal maps, segmentation)--mentioned only
- ControlNet for SDXL (different checkpoints, same principle)
- Production workflows or ComfyUI integration

**Review notes:**
- Iteration 1: NEEDS REVISION--0 critical, 3 improvement (no actual images in visual-centric lesson, misconception #5 not concrete enough, notebook Exercise 3 `pass` fails downstream), 2 polish (spaced em dashes in notebook, img2img connection omitted from lesson)
- Iteration 2: PASS--0 critical, 0 improvement, 2 polish (Canny threshold values differ slightly between lesson and notebook, reference title spaced em dash). All iteration 1 findings addressed: GradientCards rewritten with vivid visual descriptions framed as predictions, notebook guard check added, spaced em dashes fixed, img2img connection integrated.

### ip-adapter (Lesson 3)
**Status:** Built, reviewed (PASS on iteration 2)
**Cognitive load:** BUILD (2 new concepts: decoupled cross-attention, IP-Adapter as general-purpose adapter)
**Notebook:** `notebooks/7-1-3-ip-adapter.ipynb` (4 exercises: load and generate, scale sweep, text-image coexistence, compose with ControlNet)

**Concepts taught:**
- Decoupled cross-attention (DEVELOPED)--parallel K/V projections for CLIP image embeddings alongside existing text K/V; shared Q; weighted addition of two attention outputs
- IP-Adapter as a general-purpose image conditioning adapter (DEVELOPED)--trained once on millions of image-text pairs, works with any reference image at inference, ~22M trainable params
- IP-Adapter scale parameter (DEVELOPED)--volume knob for image influence, connected to conditioning scale pattern from ControlNet
- CLIP image encoder sequence output (INTRODUCED)--gap fill: 257 tokens (256 patches + CLS), not just pooled vector; small trainable projection to U-Net dimensions
- IP-Adapter coexistence with text conditioning (DEVELOPED)--image provides WHAT-IT-LOOKS-LIKE, text provides WHAT, decoupled not replaced
- IP-Adapter vs LoRA architectural distinction (DEVELOPED)--LoRA modifies existing weights, IP-Adapter adds new K/V path; "remove and check" test
- Four conditioning channels (DEVELOPED)--WHEN/WHAT/WHERE extended to WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE
- Zero initialization pattern (REINFORCED)--third instance of the pattern across adapters
- IP-Adapter is NOT img2img (DEVELOPED)--CLIP semantic encoding vs VAE pixel-level encoding
- IP-Adapter composability with ControlNet (INTRODUCED)--different U-Net targets, composable
- IP-Adapter variants: Plus, Face ID (MENTIONED)--vocabulary breadth only

**Mental models established:**
- "Two reference documents, one reader"--extends the "reading from a reference document" analogy from 4.2.5 and 6.3.4. Standard cross-attention reads from one reference document (text). Decoupled cross-attention reads from two simultaneously (text and image), each with its own translation layer (K/V projections), and combines what it reads.
- "WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE"--extends the three-dimensional conditioning framework to four dimensions. Timestep (WHEN) via adaptive norm, text (WHAT) via cross-attention K/V, ControlNet (WHERE) via additive encoder features, IP-Adapter (WHAT-IT-LOOKS-LIKE) via decoupled cross-attention K/V.
- "Same safety pattern"--zero initialization is a recurring theme across adapters: zero convolutions (ControlNet), B=0 (LoRA), zero K/V projections (IP-Adapter). New components start contributing nothing.

**Analogies used:**
- "Two reference documents, one reader" (decoupled cross-attention as reading from two documents simultaneously)
- "Same highway, add a detour" vs "build a second highway" (LoRA vs IP-Adapter: modifying existing weights vs adding new K/V pathway)
- "Volume knob" extended from ControlNet conditioning scale to IP-Adapter scale parameter
- "Four translators" extended from ControlNet lesson to include IP-Adapter as the fourth translator

**How concepts were taught:**
- Recap: two concepts reactivated--cross-attention K/V mechanism (callback to 6.3.4) and CLIP image encoder output (gap fill: sequence not just pooled vector, 257 tokens, small trainable projection to U-Net dimensions).
- Hook: "The Description Problem"--text is lossy for precise visual identity (color palette, lighting mood, material texture). Spatial maps cannot help either (edges capture contours, not color). "What if you could just SHOW the model?" Design challenge: student has the pieces (CLIP image encoder, cross-attention, shared embedding space)--how to feed image features into cross-attention without disrupting the text path? Parallels the ControlNet design challenge from lesson 1.
- Core explanation: "Of course" chain (5 steps, each following from the previous). Mermaid architecture diagram with color-coded components (gray=frozen, violet=new trainable, green=combined output). Pseudocode forward pass showing the minimal change. Shape walkthrough table at 16x16 resolution with concrete tensor dimensions. Key insight: text attention path completely untouched.
- Misconceptions addressed immediately after core explanation: "Not Img2Img" (CLIP semantic vs VAE pixel-level, placed after architecture so student can contrast pathways), "Addition, Not Averaging" (text contributes at full strength, no scale on text_out, different attention weight shapes [256,77] vs [256,257]).
- Check #1: three predict-and-verify questions (scale=0, remove IP-Adapter entirely, why share Q).
- IP-Adapter in Practice: golden retriever example with two prompts (painting in garden vs beach at sunset)--demonstrates text-image coexistence. Scale parameter sweep (0.0, 0.5, 1.0) with ornate vase--demonstrates the volume knob pattern.
- Comparisons: IP-Adapter vs LoRA (ComparisonRow, highest-priority, "remove and check" test), IP-Adapter vs textual inversion (ComparisonRow, intervention point and expressiveness), IP-Adapter vs ControlNet (ComparisonRow, semantic vs spatial). Negative example: room layout--IP-Adapter captures visual style but NOT spatial arrangement.
- Misconception addresses: "Image Prompting Is Not Image Replacement" (WarningBlock), "Trained Once, Works With Any Image" (WarningBlock).
- Style transfer example: sunset photograph (no subject) + "cat in a field"--demonstrates CLIP captures color palette, lighting, mood, not just object identity.
- Training overview: brief, for understanding. Frozen SD model, only new K/V projections + image projection trained. Zero initialization. ~22M params.
- Variants: IP-Adapter Plus, Face ID, composability with ControlNet--mentioned for vocabulary.
- Check #2: two transfer questions (style matching recommendation: IP-Adapter vs LoRA; composability: IP-Adapter + ControlNet from different images).
- Practice: four notebook exercises (Guided: load and generate, Guided: scale sweep, Supported: text-image coexistence with three prompts, Independent: compose with ControlNet).
- Module completion: ModuleCompleteBlock listing achievements across all three lessons. Four conditioning channels summary.

**Misconceptions addressed:**
1. "IP-Adapter replaces the text prompt with an image"--WarningBlock: "Image Prompting Is Not Image Replacement." Golden retriever example with two prompts producing different outputs. Image provides semantic flavor; text controls content and composition.
2. "IP-Adapter is just LoRA trained on images (same mechanism)"--ComparisonRow with five-point contrast. "Remove and check" test: remove LoRA and text output changes, remove IP-Adapter and text output is identical. LoRA modifies existing weights; IP-Adapter adds new K/V path.
3. "IP-Adapter needs to be trained per-concept (like textual inversion)"--WarningBlock: "Trained Once, Works With Any Image." Trained on millions of image-text pairs. At inference, feed any image.
4. "IP-Adapter works like img2img (encodes with VAE, feeds into denoising loop)"--WarningBlock: "Not Img2Img." CLIP semantic encoding vs VAE pixel-level. Reference never enters denoising loop as latent.
5. "Decoupled cross-attention output is just averaging text and image"--WarningBlock: "Addition, Not Averaging." Text contributes at full strength (no scale factor). Different attention weight shapes. Scale=0.5 means "full text, plus half-strength image."

**What is NOT covered (deferred):**
- IP-Adapter training procedure in detail (briefly mentioned for understanding)
- Implementing IP-Adapter from scratch
- IP-Adapter Plus, Face ID, or other variants (mentioned for vocabulary only)
- CLIP image encoder internals (ViT architecture--used as black box)
- IP-Adapter for SDXL or other model variants
- Attention processor / custom attention implementations in diffusers

**Review notes:**
- Iteration 1: MAJOR REVISION--1 critical (notebook missing), 3 improvement (section ordering: comparisons before examples violating concrete-before-abstract, misconception #5 average vs addition not explicitly addressed, "Not Img2Img" WarningBlock placed before core explanation), 2 polish (spaced em dashes in code comments, image projection not mentioned in recap)
- Iteration 2: PASS--0 critical, 0 improvement, 2 polish (spaced em dashes in JSX comments, notebook Exercise 3 partial completion edge case). All iteration 1 findings addressed: notebook built with 4 exercises, section ordering restored to concrete-before-abstract, "Addition, Not Averaging" WarningBlock added, "Not Img2Img" moved after architecture explanation, em dashes fixed, image projection sentence added to recap.
