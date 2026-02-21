# Lesson: ControlNet in Practice -- Planning Document

**Module:** 7.1 (Controllable Generation)
**Position:** Lesson 2 of 3 in module, Lesson 2 of 11 in series
**Slug:** `controlnet-in-practice`

---

## Phase 1: Orient

### Student State

The student just completed the ControlNet architecture lesson and deeply understands how ControlNet works internally. They have never *used* ControlNet with real images or preprocessors. Their mental model is architectural--they know the topology, the zero convolution safety guarantee, and the additive merge at skip connections--but they have not yet experienced the practical workflow of turning a photograph into a spatial map and feeding it to ControlNet.

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Trainable encoder copy architecture (clone encoder, train copy on spatial maps, add outputs to frozen skip connections via zero convolutions) | DEVELOPED | controlnet (7.1.1) | Core concept from lesson 1. Pseudocode forward pass with `e_i + z_i`. Parameter breakdown (~300M trainable, ~35% of U-Net). The student can trace the data flow and explain why the architecture preserves the original model. |
| Zero convolution mechanism (1x1 conv initialized to all zeros, zero initial contribution, gradual fade-in) | DEVELOPED | controlnet (7.1.1) | "Nothing, Then a Whisper" metaphor. Numerical example at init (output = 0) and after 100 steps (weight ~0.03). ComparisonRow with LoRA B=0. Student verified zero-initialization property in notebook Exercise 2. |
| ControlNet coexistence with text conditioning (WHEN/WHAT/WHERE framing) | INTRODUCED | controlnet (7.1.1) | Three conditioning dimensions: timestep=WHEN (adaptive norm), text=WHAT (cross-attention), ControlNet=WHERE (additive encoder features). Same edge map + two different prompts = different content, identical structure. |
| ControlNet as modular, swappable component (map-agnostic architecture, checkpoint swapping) | INTRODUCED | controlnet (7.1.1) | Canny vs depth ComparisonRow showing identical architecture, different preprocessing. "Four translators, one pipeline." Base model compatibility noted. |
| ControlNet does NOT modify frozen weights (feature-level addition, not weight-level) | DEVELOPED | controlnet (7.1.1) | Dedicated WarningBlock contrasting with LoRA's `Wx + BAx`. Disconnect test: frozen model is bit-for-bit identical. |
| Full SD pipeline data flow (prompt -> CLIP -> U-Net denoising loop with CFG -> VAE decode) | DEVELOPED | 6.4.1 (stable-diffusion-architecture) | "Three translators, one pipeline" (extended to four with ControlNet). Tensor shapes at every handoff. |
| Classifier-free guidance (CFG) and guidance scale | DEVELOPED | 6.3.4 (text-conditioning-and-guidance) | "Contrast slider" analogy. Formula: epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond). Tradeoff between fidelity and diversity. Student has adjusted guidance scale in practice. |
| Img2img partial denoising (add noise to an existing image, denoise from there) | DEVELOPED | 6.5.2 (img2img-and-inpainting) | Noise strength parameter controls how much of the original image survives. Student used this in practice. |
| Edge detection as a concept (Canny edges mentioned as spatial map type) | MENTIONED | controlnet (7.1.1) | Canny edges used as the primary example spatial map throughout lesson 1. Student saw edge map images but never ran a Canny detector or understood what it computes. |
| Depth maps as a concept (depth estimation mentioned as spatial map type) | MENTIONED | controlnet (7.1.1) | Depth map appeared in the Canny-vs-depth ComparisonRow. Student knows it is a different type of spatial map but has no experience with depth estimation models. |
| OpenPose as a concept (pose estimation mentioned as spatial map type) | MENTIONED | controlnet (7.1.1) | Named in the lesson 1 scope boundaries as "deferred to lesson 2." Not described. |

**Established mental models the student carries:**
- "Nothing, Then a Whisper" (zero convolution fade-in)
- "WHEN/WHAT/WHERE" (three conditioning dimensions coexisting)
- "Four translators, one pipeline" (SD modularity with ControlNet)
- "The 'of course' chain" (ControlNet architecture follows from constraints)
- "Highway and detour" / "detour that starts as a dead end" (residual connections, zero conv)
- "Contrast slider" (CFG guidance scale tradeoff)

**What was explicitly NOT covered that is relevant:**
- How preprocessors extract spatial maps from images (Canny edge detection algorithm, monocular depth estimation, OpenPose skeleton extraction)--deferred from lesson 1
- Conditioning scale parameter (how much the model follows the ControlNet signal vs generates freely)--deferred from lesson 1
- Stacking multiple ControlNets simultaneously--deferred from lesson 1
- Failure modes and limitations of ControlNet in practice

**Readiness assessment:** The student is fully prepared. The architecture is at DEVELOPED depth, and this lesson requires no new architectural knowledge. The new content is entirely practical: (1) how preprocessors turn photos into spatial maps, (2) what the conditioning scale parameter does, and (3) how to combine multiple ControlNets. All of this builds on the WHEN/WHAT/WHERE mental model and the modular component understanding from lesson 1. No gaps.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to use ControlNet end-to-end with real preprocessors, control the strength of spatial conditioning via the conditioning scale, and stack multiple ControlNets to combine different types of structural control.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Trainable encoder copy architecture | INTRODUCED | DEVELOPED | controlnet (7.1.1) | OK | Student needs to recall what ControlNet does (takes a spatial map, produces features added to skip connections) but does not need to trace the full forward pass. INTRODUCED is sufficient for practical use; they have it at DEVELOPED. |
| Zero convolution mechanism | INTRODUCED | DEVELOPED | controlnet (7.1.1) | OK | Student needs to recall that zero convolutions ensure safe initialization and gradual fade-in. Not re-deriving the mechanism--just understanding that the conditioning scale modulates the same signal that zero convolutions introduce. |
| ControlNet coexistence with text conditioning | DEVELOPED | INTRODUCED | controlnet (7.1.1) | GAP (small) | Conditioning scale is the dial between "follow the spatial map" and "follow the text prompt." To understand the tradeoff, the student needs to actively reason about how text and spatial conditioning interact, not just recognize that they coexist. Needs elevation from INTRODUCED to DEVELOPED through hands-on experience (which this lesson provides). |
| ControlNet as modular, swappable component | DEVELOPED | INTRODUCED | controlnet (7.1.1) | GAP (small) | Stacking multiple ControlNets requires understanding modularity at a deeper level--that each ControlNet independently contributes additive features and they compose by summation. Currently at INTRODUCED (recognized the pattern); needs DEVELOPED (used the pattern). |
| Full SD pipeline data flow | INTRODUCED | DEVELOPED | 6.4.1 | OK | Student needs to know where ControlNet sits in the pipeline (during U-Net denoising). Has this well beyond required depth. |
| Classifier-free guidance (CFG) and guidance scale | INTRODUCED | DEVELOPED | 6.3.4 | OK | Conditioning scale is conceptually parallel to CFG guidance scale--both are "how much to follow this signal" dials. The student has the CFG tradeoff at DEVELOPED, which provides the right mental framework for conditioning scale. |

**Gap resolution:**

| Gap | Size | Resolution |
|-----|------|------------|
| ControlNet coexistence at INTRODUCED, needs DEVELOPED | Small | Lesson 1 INTRODUCED this with the two-prompt example and WHEN/WHAT/WHERE framing. This lesson elevates it to DEVELOPED by having the student directly manipulate conditioning scale and observe the text-vs-spatial tradeoff. No recap section needed--the hands-on exploration IS the gap resolution. A brief callback to WHEN/WHAT/WHERE when introducing conditioning scale is sufficient. |
| ControlNet modularity at INTRODUCED, needs DEVELOPED | Small | Lesson 1 INTRODUCED this with the Canny-vs-depth ComparisonRow. This lesson elevates it to DEVELOPED through multi-ControlNet stacking, where the student loads two ControlNets simultaneously and observes how their contributions compose. No recap section needed--the stacking section IS the gap resolution. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Conditioning scale 1.0 means the model follows the spatial map perfectly and ignores the text prompt" | The student has seen CFG guidance scale where higher values mean "follow the text more strongly." Natural assumption: conditioning_scale=1.0 means "100% follow the control." | Generate with conditioning_scale=1.0 and two different text prompts using the same edge map. The images have identical structure but clearly different content/style. The text prompt still matters significantly at scale=1.0. The scale modulates spatial strength, it does not disable text conditioning. | Section 4 (Explain) when introducing conditioning scale. Predict-and-verify exercise. |
| "Higher conditioning scale is always better--more control is more desirable" | The student wants spatial precision. If ControlNet provides spatial control, maximizing it seems logical. More control = better result. | Generate with conditioning_scale=0.5, 1.0, 1.5, 2.0 on the same edge map. At high values (1.5+), the model over-constrains: fine details become rigid, textures flatten, the image looks mechanical or artifacted. The model is trying too hard to match every pixel of the control map, sacrificing the natural variation and detail that makes the image look good. The sweet spot is typically 0.7-1.0. | Section 4 (Explain), demonstrated via the conditioning scale sweep. |
| "You need different code/APIs for different spatial map types (edges vs depth vs pose)" | Each spatial map type produces a visually different control signal. The student might expect that different map types require different ControlNet APIs, different pipeline configurations, or different inference code. | The diffusers API is identical for all map types. The only thing that changes is (1) which preprocessor extracts the map from the image, and (2) which ControlNet checkpoint you load. The pipeline code, the conditioning_scale parameter, and the inference call are all identical. `StableDiffusionControlNetPipeline` does not know or care what type of spatial map it receives. | Section 3 (Explain) after showing all three preprocessors. An explicit "notice what stayed the same" moment. |
| "Stacking two ControlNets doubles the control strength" | If one ControlNet adds features at skip connections, two ControlNets add twice the features. Linear stacking = linear scaling of control. | Stack Canny edges + depth map on the same image. The result is not "2x more controlled"--it is "controlled in two different ways simultaneously." Edges enforce contours, depth enforces layering/perspective. They provide complementary information, not redundant information. In fact, if you set both conditioning scales too high, they can conflict (edges say "sharp boundary here" but depth says "smooth transition here"), producing artifacts. | Section 5 (Elaborate) during multi-ControlNet stacking. |
| "The preprocessor quality does not matter much--ControlNet will figure it out" | The student has seen that ControlNet is robust (zero conv, gradual fade-in, trained on data). They might assume the spatial map quality is not critical. | Run ControlNet with a poor-quality Canny edge map (wrong threshold, too many noisy edges or too few meaningful edges) vs a well-tuned Canny edge map. The generation quality degrades dramatically with bad preprocessing. ControlNet faithfully follows whatever spatial map you give it--if the map is garbage, the output is garbage. Preprocessing is the most impactful practical skill. | Section 3 (Explain) when demonstrating Canny threshold tuning. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Canny edge extraction with threshold tuning: same photo, three different Canny threshold pairs (too few edges, good edges, too many edges) and their ControlNet outputs | Positive | Shows that preprocessing is not a black box--the threshold directly affects output quality. Demonstrates the "garbage in, garbage out" principle. | Canny is the simplest and most tunable preprocessor. The threshold parameter makes preprocessing quality visible and controllable. Three thresholds show the full range from under-detection to over-detection. |
| Depth map extraction via MiDaS: photo -> depth map -> ControlNet generation with depth conditioning | Positive (generalization) | Shows that a different spatial map type (depth instead of edges) produces a fundamentally different kind of control (layering/perspective vs contours). Same pipeline code, different preprocessor, different ControlNet checkpoint. | Depth maps are visually and conceptually distinct from edges--they encode 3D structure, not 2D contours. This confirms that the modular architecture generalizes and produces qualitatively different control. |
| OpenPose skeleton extraction: photo of a person -> skeleton map -> ControlNet generation preserving pose | Positive (generalization) | Shows a third map type with its own unique control semantics (body pose). Completes the trio of preprocessor types (edges, depth, pose) and demonstrates the breadth of spatial control. | Pose is the most "magical" ControlNet application--extracting a stick figure and generating a completely different person in the same pose. High-impact demo that lands the practical value. |
| Conditioning scale sweep: same edge map, same prompt, conditioning_scale from 0.3 to 2.0 in steps | Positive (parameter exploration) | Makes the control-creativity tradeoff visible. Low scale = creative but ignores structure. High scale = rigid but loses natural detail. Sweet spot in between. | The conditioning scale is the key practical parameter. A sweep across values, presented as a comparison grid, gives the student intuition for what this dial does. Connects to the existing CFG "contrast slider" mental model. |
| Multi-ControlNet stacking: Canny edges + depth map on the same image, both active simultaneously | Positive (composition) | Shows that ControlNets compose by providing complementary structural constraints. The student sees that edges enforce contours while depth enforces layering, and the combination produces more precisely controlled output than either alone. | Stacking is the practical payoff of ControlNet modularity--it demonstrates that the additive architecture allows independent spatial constraints to combine. |
| Over-constrained stacking: two ControlNets with conflicting signals at high conditioning scales, producing artifacts | Negative | Shows the limits of stacking. When two spatial maps provide contradictory information at high strength, the result degrades. Demonstrates that more control is not always better and that the conditioning scale per-ControlNet matters. | Provides the boundary case. The student needs to see failure to understand the tradeoff space. Prevents the "just stack everything at max" misconception. |

### Widget Consideration

**Widget needed:** No custom interactive widget.

**Reasoning:** This is a CONSOLIDATE lesson focused on practical, hands-on use in a Colab notebook. The core learning happens through running code, varying parameters, and observing results. The key interactive experiences (conditioning scale sweep, preprocessor comparison, multi-ControlNet stacking) are best served by notebook cells where the student runs real inference with real models and sees real generated images. A web widget would either require a diffusion model backend (impractical) or simulate results with pre-computed images (less valuable than the real thing). The lesson component in the app should present the conceptual framing (what preprocessors do, what conditioning scale means, how stacking works) with comparison images, and the notebook provides the interactive exploration.

The lesson component will include:
- Static comparison images (before/after preprocessors, conditioning scale sweep grid, stacking results)
- ComparisonRows for different preprocessor types
- A conditioning scale conceptual diagram (not interactive--the notebook is where the student tunes this)

---

## Phase 3: Design

### Narrative Arc

You know how ControlNet works--the trainable encoder copy, zero convolutions, additive features at skip connections. You could draw the architecture from memory. But you have never used it. You have never turned a photograph into a Canny edge map, adjusted the threshold until the edges capture the right level of detail, fed it to ControlNet alongside a creative text prompt, and watched the model follow your spatial intent while filling in everything else from the text. You have never discovered that setting the conditioning scale too high makes the output rigid and lifeless, or that stacking a depth map on top of an edge map produces something neither could achieve alone. This lesson is where the architecture becomes a tool. The emphasis is entirely practical: which preprocessor to use and when, how to tune the control-creativity tradeoff, and how to combine multiple spatial constraints. No new theory. No new architecture diagrams. Just the payoff of what you already understand.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual (comparison grids)** | Side-by-side image grids: (1) source photo -> preprocessed map -> ControlNet output, for each of three preprocessor types. (2) Conditioning scale sweep: same edge map at 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, showing the progression from creative-but-loose to rigid-but-precise. (3) Stacking grid: edges only vs depth only vs edges+depth. | This is a visual lesson. The concepts (preprocessing quality, conditioning strength, stacking) are best understood by seeing the images. The comparison grids make the differences immediately apparent without requiring verbal explanation. |
| **Concrete example (hands-on notebook)** | Notebook cells where the student runs Canny edge detection with different thresholds, generates images at different conditioning scales, and stacks two ControlNets. Real inference with real models, not pre-computed. | The entire lesson thesis is "use it." The notebook IS the primary modality. Pre-computed images in the lesson component provide the conceptual framework; the notebook provides the experiential understanding. |
| **Intuitive / Analogy** | Conditioning scale as a "volume knob" for spatial control: at zero, the spatial map is muted and the model generates freely from text alone; at moderate levels, the spatial map provides structure while text fills in details; at maximum, the spatial map dominates and the model rigidly follows every contour, losing natural variation. Extension of the "Nothing, Then a Whisper" metaphor from lesson 1 (which described training fade-in) to the inference-time control dial. Also: CFG parallel--conditioning scale is to spatial control what guidance scale is to text control. Two knobs, same mixing board. | Builds directly on two existing mental models (whisper/volume, contrast slider). The student should feel that conditioning scale is "obvious"--it is just another volume knob on the same mixing board where they already adjusted CFG. |
| **Verbal (preprocessor taxonomy)** | Clear explanation of what each preprocessor extracts and why it matters: Canny = 2D contours (edges), MiDaS = 3D structure (depth layering), OpenPose = human body pose (skeleton keypoints). Why you would choose each: edges for architectural/compositional control, depth for perspective/layering, pose for character positioning. | The student needs a mental taxonomy for choosing preprocessors. This is practical knowledge that cannot be discovered by running code--you need to know what each preprocessor is good for BEFORE you decide which one to use. |
| **Symbolic (code patterns)** | Diffusers API code showing: (1) preprocessor call, (2) ControlNet pipeline setup, (3) generation with conditioning_scale. Explicit "notice what stays the same" when switching preprocessors--only the preprocessor function and ControlNet checkpoint change. Stacking code: list of ControlNets, list of images, list of conditioning_scales. | The code patterns are what the student will actually use. Seeing the API structure reveals the modularity concretely--the pipeline does not know or care what kind of spatial map it receives. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 0 genuinely new theoretical concepts. The lesson introduces three practical skills:
  1. **Preprocessing spatial maps** (Canny, depth, pose)--practical skill, not theory
  2. **Conditioning scale as the control-creativity tradeoff**--one new parameter with a clear analogy to CFG
  3. **Multi-ControlNet stacking**--extension of the modularity concept from INTRODUCED to DEVELOPED
- **Previous lesson load:** STRETCH (2 new concepts: trainable encoder copy, zero convolution)
- **This lesson's load:** CONSOLIDATE. Zero new theory. The cognitive work is in building practical intuition--what happens when I turn this knob, when do I use this preprocessor, what does stacking actually produce. This is the "hands-on win" after the architectural STRETCH of lesson 1. The module plan explicitly calls this CONSOLIDATE.
- **Load trajectory:** STRETCH (lesson 1) -> CONSOLIDATE (this lesson) -> BUILD (lesson 3). Appropriate--the student gets practical payoff and builds confidence before the conceptual delta of IP-Adapter.

### Connections to Prior Concepts

- **CFG guidance scale -> conditioning scale:** "You already know the guidance scale dial: higher values mean 'follow the text prompt more strongly' at the cost of diversity. Conditioning scale is the same idea for spatial control: higher values mean 'follow the spatial map more strictly' at the cost of natural variation. Two knobs on the same mixing board."
- **"Nothing, Then a Whisper" -> conditioning scale at inference:** "During training, zero convolutions gradually turn up the volume of spatial control from zero. At inference time, conditioning scale is your manual volume knob for the same signal. You choose how loud the spatial control is."
- **Component modularity ("four translators") -> stacking:** "You know ControlNet is a fourth translator that connects via tensor handoffs. Stacking is just adding a fifth translator. Each ControlNet independently contributes additive features to the skip connections. They compose by summation."
- **LoRA checkpoint swapping -> ControlNet checkpoint swapping:** "Same pattern as LoRA: load a different checkpoint, get a different behavior, no retraining. Swap 'controlnet-canny' for 'controlnet-depth' and the pipeline produces depth-conditioned output instead of edge-conditioned output."
- **Img2img strength parameter -> conditioning scale:** "In img2img, the strength parameter controls how much of the original image survives. Conditioning scale similarly controls how much of the spatial map's structure survives in the output. Both are 'how much to deviate from this reference' dials."

**Analogies from prior lessons that extend here:**
- "Contrast slider" (CFG) -> extends to "volume knob" for conditioning scale. Same concept: dial that trades precision for creativity.
- "Four translators, one pipeline" -> extends to multi-translator stacking. Multiple spatial translators can run simultaneously.
- "Nothing, Then a Whisper" -> extends from training behavior to inference-time control.

**Analogies from prior lessons that could be misleading:**
- The "contrast slider" for CFG might lead the student to expect that conditioning_scale=0 produces the same output as vanilla SD (no ControlNet). This is roughly true but not exactly--the ControlNet path is still active at scale=0 in some implementations. Need to clarify that scale=0 means the ControlNet features are multiplied by 0, effectively muting them, but the cleanest "no ControlNet" baseline is to not load ControlNet at all.

### Scope Boundaries

**This lesson IS about:**
- How to preprocess images into spatial maps using Canny edge detection, MiDaS depth estimation, and OpenPose skeleton extraction
- Using ControlNet via the diffusers library with real images
- The conditioning_scale parameter and the control-creativity tradeoff
- Stacking multiple ControlNets for combined spatial control
- Practical guidelines: when to use which preprocessor, how to tune conditioning scale, failure modes of bad preprocessing

**This lesson is NOT about:**
- ControlNet architecture internals (covered in lesson 1; brief callbacks only)
- How the preprocessors themselves work internally (Canny algorithm, MiDaS architecture, OpenPose architecture)--they are used as black-box tools
- Training a ControlNet from scratch
- IP-Adapter or image-based conditioning (lesson 3)
- T2I-Adapter or other control mechanisms
- Every possible preprocessor type (lineart, scribble, normal maps, segmentation maps)--mentioned for breadth, not developed
- ControlNet for SDXL (different checkpoints, same principle--briefly noted)
- Production workflows or ComfyUI integration

**Target depths:**
- Preprocessing spatial maps (Canny, depth, pose): DEVELOPED (used hands-on, understands what each extracts and when to use it)
- Conditioning scale / control-creativity tradeoff: DEVELOPED (tuned it, seen the full range, has intuition for the sweet spot)
- Multi-ControlNet stacking: DEVELOPED (stacked two ControlNets, seen complementary and conflicting results)
- ControlNet coexistence with text conditioning: elevated from INTRODUCED to DEVELOPED (directly manipulated the text-spatial balance)
- ControlNet as modular component: elevated from INTRODUCED to DEVELOPED (swapped checkpoints, stacked multiple)

### Lesson Outline

**1. Context + Constraints**
- "Last lesson you learned the architecture. This lesson you use it."
- Capstone tone: this is a practical session. The student drives.
- Scope: preprocessors, conditioning scale, stacking. No new architecture. No internals of how Canny/MiDaS/OpenPose work (black-box tools).
- By the end: the student can turn any photograph into a spatial control signal, generate controlled images, tune the strength, and combine multiple control types.

**2. Recap (minimal)**
- No dedicated recap section needed. Prerequisites are solid and recent (lesson 1 was the immediately preceding lesson).
- Brief inline callbacks as needed: "Remember WHEN/WHAT/WHERE" when introducing conditioning scale, "Remember the additive features at skip connections" when explaining stacking.

**3. Hook: The Missing Piece (type: before/after)**
- Present a photograph and the edge map from lesson 1's examples. "You saw this edge map in the last lesson. But where did it come from? You have been looking at preprocessed spatial maps without knowing how they were made."
- Quick "behind the curtain" reveal: show the one-line Canny preprocessor call that turns the photo into the edge map. "That is it. One function call. But the quality of this preprocessing step determines the quality of everything that follows."
- Preview the lesson's three practical skills: (1) extract spatial maps with different preprocessors, (2) tune how strongly ControlNet follows them, (3) stack multiple types of spatial control.

**4. Explain: Preprocessors (three map types)**
- **Structure:** Each preprocessor gets a focused subsection: what it extracts, what it controls, a visual (source photo -> map -> generation), and a practical guideline for when to use it.
- **Canny edge detection:**
  - What it extracts: binary edge map from intensity gradients. Two thresholds (low, high) control edge sensitivity.
  - What it controls: 2D contours and silhouettes. The model follows the drawn edges.
  - Threshold tuning: show three threshold settings (too few edges = model ignores composition, good edges = clean structure, too many edges = noisy artifacts). "This is the most impactful practical decision you will make with ControlNet."
  - When to use: architectural scenes, product design, any case where you want precise contour control.
  - Code: `canny = cv2.Canny(image, low_threshold, high_threshold)`
- **MiDaS depth estimation:**
  - What it extracts: monocular depth map (grayscale, lighter = closer, darker = farther). Uses a pre-trained depth estimation model.
  - What it controls: 3D structure, perspective, layering. Objects at different depths maintain their relative positioning.
  - When to use: landscape composition, scene depth control, when you care about spatial arrangement more than exact contours.
  - Code: `depth = depth_estimator(image)` (diffusers provides the preprocessor)
- **OpenPose skeleton detection:**
  - What it extracts: body keypoints connected into a stick-figure skeleton. Optionally includes hand and face keypoints.
  - What it controls: human body pose. The model generates a figure matching the skeleton's pose.
  - When to use: character positioning, pose transfer (extract pose from one photo, generate a different person in the same pose).
  - Code: `pose = openpose(image)` (via controlnet_aux)
- **Explicit "notice what stays the same" moment:** The diffusers ControlNet pipeline call is identical for all three map types. Only the preprocessor and the ControlNet checkpoint change. The architecture you learned in lesson 1 is genuinely map-agnostic.
- **Brief mention of other preprocessor types** (lineart, scribble, normal maps, segmentation) for vocabulary breadth. Not developed.

**5. Check: Preprocessor Selection**
- Scenario-based questions (predict which preprocessor to use):
  - "You want to generate a fantasy castle that follows the skyline of a real city photograph." (Canny or depth--edges for silhouette, depth for layering.)
  - "You want to generate a character in the same pose as a photo of a dancer." (OpenPose.)
  - "You want to generate a landscape that preserves the foreground/background layering of a reference photo but changes the content entirely." (Depth.)

**6. Explain: Conditioning Scale (the control-creativity tradeoff)**
- **Connect to existing mental model:** "CFG guidance scale is a dial: higher means 'follow the text more strongly.' Conditioning scale is the same dial for spatial control: higher means 'follow the spatial map more strictly.' Two knobs on the same mixing board."
- **The tradeoff:** Low conditioning scale = the model generates freely, spatial map is a suggestion. High conditioning scale = the model rigidly follows the spatial map, losing natural variation and fine detail.
- **Conditioning scale sweep:** Visual grid showing the same edge map + same prompt at scale values 0.3, 0.5, 0.7, 1.0, 1.5, 2.0. The student sees the progression from "creative but loose" to "precise but rigid."
- **The sweet spot:** Typically 0.7-1.0 for most use cases. Below 0.5 the spatial control is too weak to matter. Above 1.5 the output becomes over-constrained.
- **Address misconception:** "Conditioning scale 1.0 does not mean the text prompt is ignored. Generate with scale=1.0 and two different text prompts: the structure is identical but the content differs. Both conditioning dimensions are still active."
- **Per-ControlNet conditioning scale:** When stacking, each ControlNet has its own conditioning_scale. You can weight edges more heavily than depth, or vice versa.
- **Extend "Nothing, Then a Whisper":** "During training, zero convolutions gradually turn up the volume. At inference, conditioning_scale is your manual volume knob for the same signal."

**7. Check: Conditioning Scale Prediction**
- Predict-and-verify: "You set conditioning_scale=2.0 on a Canny edge map of a detailed architectural drawing. What do you expect to see?" (Over-constrained: the model tries to follow every edge pixel, producing rigid textures and possible artifacts where edges are ambiguous.)
- Predict-and-verify: "You set conditioning_scale=0.3 on the same edge map. What do you expect?" (The model mostly ignores the edges. The composition might roughly follow the spatial layout but fine details are the model's own.)

**8. Explain: Multi-ControlNet Stacking**
- **The concept:** Each ControlNet independently contributes additive features to the skip connections. They compose by summation: `e_i + z_i_canny + z_i_depth`. The decoder receives the combined spatial information.
- **Code pattern:** A list of ControlNets, a list of preprocessed images, a list of conditioning_scales. The pipeline handles the rest.
  ```python
  pipe = StableDiffusionControlNetPipeline.from_pretrained(
      model_id, controlnet=[controlnet_canny, controlnet_depth]
  )
  result = pipe(prompt, image=[canny_map, depth_map],
                controlnet_conditioning_scale=[0.8, 0.6])
  ```
- **Complementary stacking:** Canny edges + depth map on the same image. Edges enforce contours, depth enforces layering. The combination produces more precisely controlled output than either alone. Visual comparison: edges-only vs depth-only vs both.
- **Conflicting stacking (negative example):** Two ControlNets with contradictory spatial signals (e.g., edges from one image + depth from a different image). At high conditioning scales, they fight each other, producing artifacts. The model cannot simultaneously follow two spatial maps that disagree about where things are.
- **Practical guideline:** Start with one ControlNet, get it working, then add a second. Use moderate conditioning scales (0.5-0.8) when stacking. Keep both maps derived from the same source image for complementary control.

**9. Check: Stacking Transfer Question**
- "Your friend stacks three ControlNets (Canny + depth + pose) all at conditioning_scale=1.5 and gets muddy, artifacted results. What would you suggest?" (Lower the conditioning scales. Three ControlNets at 1.5 each over-constrains the model. Start with 0.5-0.7 for each and increase selectively. Also check if the spatial maps are consistent--all from the same source image.)

**10. Practice: Notebook Exercises (Colab)**
- **Exercise 1 (Guided): Canny Edge Preprocessing and Generation**
  - Load a photograph and extract Canny edges with cv2.Canny
  - Experiment with three threshold pairs: (50, 100), (100, 200), (200, 300)
  - For each, generate an image with ControlNet and observe how edge quality affects output
  - Predict before running: "Which threshold setting will produce the best-controlled generation?"
  - **Tests:** preprocessing quality impact, threshold tuning as a practical skill
  - **Key insight:** garbage in, garbage out. Preprocessing is the most impactful step.

- **Exercise 2 (Guided): Three Preprocessors, One Pipeline**
  - Take the same source photograph and extract: Canny edges, depth map (MiDaS), pose (OpenPose, requires a photo with a person)
  - Generate with each preprocessor + its corresponding ControlNet checkpoint
  - Compare outputs side-by-side: what kind of control does each provide?
  - **Tests:** preprocessor taxonomy, modular checkpoint swapping, "what stays the same" in the API
  - **Key insight:** same pipeline code, different preprocessor and checkpoint, qualitatively different control.

- **Exercise 3 (Supported): Conditioning Scale Sweep**
  - Use the best Canny edge map from Exercise 1
  - Generate at conditioning_scale values: 0.3, 0.5, 0.7, 1.0, 1.5, 2.0
  - Display as a comparison grid
  - Student writes the sweep loop (scaffolded: function signature and display code provided, student fills in the generation call with varying scale)
  - Then: same edge map, scale=1.0, two different text prompts. Verify text conditioning still active.
  - **Tests:** conditioning scale tradeoff, text-spatial coexistence at the practical level
  - **Key insight:** the sweet spot is typically 0.7-1.0. Higher is not better.

- **Exercise 4 (Supported): Multi-ControlNet Stacking**
  - Load Canny and depth ControlNet checkpoints
  - Extract both maps from the same source image
  - Generate with edges only, depth only, and both stacked
  - Compare results in a 3-column grid
  - Student writes the stacking pipeline call (scaffolded: model loading done, student fills in the pipeline construction with a list of ControlNets and the generation call with lists of images and scales)
  - **Tests:** stacking composition, complementary vs conflicting control
  - **Key insight:** complementary maps from the same source image compose well. The combination is more precise than either alone.

- **Exercise 5 (Independent): Your Composition**
  - The student chooses their own source image (provided options: a cityscape, a portrait, a landscape)
  - They select the best preprocessor(s) for their creative intent
  - They tune conditioning scale(s) to achieve a specific visual goal
  - No scaffolding. The student must decide: which preprocessor, what thresholds, what scale, whether to stack.
  - **Tests:** practical decision-making, integrated application of all skills
  - **Key insight:** the workflow is choose image -> choose preprocessor(s) -> tune preprocessing -> tune conditioning scale -> iterate.

- **Scaffolding progression:** Guided -> Guided -> Supported -> Supported -> Independent. Two guided exercises establish the building blocks (preprocessing, preprocessor types). Two supported exercises explore the key parameters (conditioning scale, stacking). One independent exercise integrates everything.
- **Exercises are cumulative:** Exercise 1's best edge map is reused in Exercise 3. Exercise 2's depth map is reused in Exercise 4. Exercise 5 draws on all prior exercises.
- **Solutions should emphasize:** Practical decision-making ("why did this threshold produce better results?"), the tradeoff space ("what was lost when you increased the scale?"), and the "notice what stays the same" insight ("same pipeline call, different inputs").

**11. Summarize**
- Three types of spatial control: edges (Canny) for contours, depth (MiDaS) for 3D structure, pose (OpenPose) for body positioning. Many more exist--these are the three most common.
- Preprocessing quality is the most impactful practical decision. Garbage in, garbage out.
- Conditioning scale is a volume knob: controls the tradeoff between spatial precision and creative freedom. Sweet spot is typically 0.7-1.0.
- Multiple ControlNets stack by summing their additive features. Use complementary maps from the same source, moderate scales, and add complexity gradually.
- Echo the mental model: "The WHEN/WHAT/WHERE framework now has a volume knob for WHERE. And you can layer multiple WHERE signals from different spatial translators."

**12. Next Step**
- Next lesson: IP-Adapter. So far, all conditioning has been either text (WHAT) or spatial maps (WHERE). But what if you want to say "generate something that looks like *this reference image*"--not its edges or depth, but its semantic content and style? IP-Adapter adds a fourth conditioning dimension by injecting image embeddings into cross-attention via a decoupled K/V path. The architecture you know grows one more translator.

---

## Review -- 2026-02-19 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings. The lesson is structurally sound, follows the plan closely, and the student would not get lost or form wrong mental models. However, three improvement-level findings weaken the lesson enough to warrant a revision pass before shipping.

### Findings

#### [IMPROVEMENT] -- Modality rule: no actual images in a visual-centric lesson

**Location:** Sections 4-8 (preprocessors, conditioning scale sweep, stacking)
**Issue:** The planning document specifies "Static comparison images (before/after preprocessors, conditioning scale sweep grid, stacking results)" as a core modality. The built lesson has zero actual images. The Canny threshold comparison (Section 4) uses three GradientCards with bullet-point descriptions of what the output looks like ("Missing structural detail," "Clean edges capture structure," etc.). The conditioning scale sweep (Section 6) uses six small GradientCards with text descriptions. The stacking comparison uses a ComparisonRow with text. All of these describe visuals in words rather than showing them. For a CONSOLIDATE lesson whose entire thesis is "see the practical results," this is a significant gap.
**Student impact:** The student reads text descriptions of visual phenomena they have never seen. "Clean edges capture structure" and "Model follows composition precisely" are abstract claims without visual evidence. The lesson tells the student what to expect rather than showing them, undermining the practical/experiential nature of a CONSOLIDATE lesson. The notebook becomes the sole source of visual experience, but the lesson component should provide the conceptual framework with enough visual grounding that the student enters the notebook with clear expectations.
**Suggested fix:** Add at least one real comparison image for each of the three key topics: (1) a before/after showing good vs bad Canny thresholds on a real photograph, (2) a conditioning scale sweep showing 3-4 scale values on real output, (3) a stacking comparison (edges only vs depth only vs stacked). These could be static images committed to the repo or generated once and saved. Even placeholder `<img>` tags with descriptive alt text and a TODO would be better than pure text descriptions.

#### [IMPROVEMENT] -- Misconception #5 (preprocessor quality) addressed but not as a concrete negative example

**Location:** Section 4 (Canny Edge Detection) -- the three GradientCards for threshold comparison
**Issue:** The planning document identifies "The preprocessor quality does not matter much--ControlNet will figure it out" as misconception #5, with the prescribed negative example being: "Run ControlNet with a poor-quality Canny edge map vs a well-tuned Canny edge map. The generation quality degrades dramatically." The lesson shows the three threshold settings as text descriptions in GradientCards, and there is a "Garbage In, Garbage Out" InsightBlock in the aside. But there is no concrete negative example showing actual degraded output. The text says "Result: rigid, artifacted output" and "Result: loose, imprecise structure" without demonstrating it. A misconception needs a concrete counter-example that the student can see, not just a verbal assertion.
**Student impact:** The student is told that preprocessing quality matters but does not see the evidence until the notebook. This weakens the "garbage in, garbage out" message in the lesson itself. The student might skim past it as a generic warning rather than internalizing it as the lesson's most important practical insight.
**Suggested fix:** This is closely related to the images finding above. At minimum, strengthen the text to be more vivid and specific about what goes wrong (rather than generic labels like "rigid, artifacted output"). Ideally, include real comparison images. The notebook handles this well (Exercise 1 is exactly this experiment), so the lesson could also more explicitly frame these descriptions as "predictions for what you will see in the notebook" to create the predict-before-run setup.

#### [IMPROVEMENT] -- Notebook Exercise 3 has a code cell with `pass` that will fail downstream

**Location:** Notebook cell-20 (Exercise 3 -- conditioning scale sweep)
**Issue:** The TODO cell ends with `pass`, so `sweep_results` remains an empty dict. The very next cell (cell-21) iterates over `sweep_results[scale]` for display, which will raise a `KeyError` because the dict is empty. The student is expected to fill in the loop, but if they run cells in order without completing the TODO (e.g., to see the solution first, or if they get confused), they will hit an immediate error with no helpful error message. Contrast with Exercise 4, which has the same pattern but the cell references `img_stacked` which would fail with a clear `NameError` ("name 'img_stacked' is not defined").
**Student impact:** A student who runs cells sequentially without completing the TODO gets a confusing `KeyError: 0.3` on the display cell rather than a clear message about what they need to do. This is a friction point for a Supported exercise where the student is supposed to have some scaffolding.
**Suggested fix:** Add a guard at the top of the display cell (cell-21) that checks `if not sweep_results:` and prints a helpful message like "sweep_results is empty -- go back to the previous cell and fill in the TODO." Alternatively, add a brief assertion or check in the TODO cell itself after the loop: `assert len(sweep_results) == len(scales), "Fill in the TODO above to generate images for each scale value."` This is a standard notebook UX pattern for Supported exercises.

#### [POLISH] -- Notebook uses spaced em dashes throughout

**Location:** Multiple markdown cells and print statements in the notebook
**Issue:** The notebook consistently uses ` — ` (space-em dash-space) in both markdown cells and print statement strings. Examples: "ideal for all three preprocessors — ideal," "Same prompt, same seed — different preprocessor," "Stacking is not doubling control strength — it is providing two complementary types." The writing style rule specifies em dashes must have no spaces: `word—word` not `word — word`.
**Student impact:** No pedagogical impact. This is a stylistic inconsistency with the project conventions.
**Suggested fix:** Find-and-replace ` — ` with `—` throughout the notebook's markdown cells and print/title strings. The lesson .tsx file correctly uses `&mdash;` or `&ndash;` entities which render without spaces, so the notebook is the outlier.

#### [POLISH] -- Conditioning scale section mentions img2img connection in planning but omits it from lesson

**Location:** Section 6 (Conditioning Scale) and the planning document's "Connections to Prior Concepts" section
**Issue:** The planning document identifies a connection: "In img2img, the strength parameter controls how much of the original image survives. Conditioning scale similarly controls how much of the spatial map's structure survives in the output. Both are 'how much to deviate from this reference' dials." This connection is not in the built lesson. The lesson does connect conditioning scale to CFG guidance scale and to "Nothing, Then a Whisper," but omits the img2img parallel.
**Student impact:** Minor. The two connections that ARE present (CFG parallel, zero conv volume knob) are strong and sufficient. The img2img connection would be a nice additional bridge but is not critical for understanding.
**Suggested fix:** Consider adding a brief aside or inline mention: "You have seen this tradeoff before with img2img strength--how much of the original image survives versus how much the model generates freely. Conditioning scale is the same idea for spatial maps." This is a one-sentence addition that strengthens the connection web. Not required but would be a quality improvement.

### Review Notes

**What works well:**
- The lesson follows the planning document closely. Structure matches, misconceptions are addressed at planned locations, examples cover the planned ground.
- The lesson's pacing is excellent for CONSOLIDATE load. It moves briskly through practical content without bogging down in theory. The inline callbacks to lesson 1 concepts (WHEN/WHAT/WHERE, zero conv, modularity) are brief and well-placed.
- The "Notice What Stays the Same" section is a strong design moment that addresses misconception #3 effectively. The code block showing the identical pipeline call with a swappable checkpoint comment is clean.
- Check questions are well-designed: the preprocessor selection scenarios test practical judgment (not recall), the conditioning scale predictions build intuition, and the "debugging a friend's stack" question is a good transfer task.
- The notebook is comprehensive, well-scaffolded, and handles VRAM management carefully. The exercise progression (Guided -> Guided -> Supported -> Supported -> Independent) matches the plan exactly.
- All five misconceptions from the planning document are addressed in the lesson.

**The core weakness:**
The lesson is a CONSOLIDATE lesson about practical, visual results -- but the lesson component itself has no images. It reads as "here is what you will see in the notebook" rather than "look at these results." The notebook compensates by being excellent, but the lesson component should stand on its own as a conceptual framework with visual grounding. This is not critical (the notebook does the job), but it is a meaningful gap for a visual-centric topic.

**Notebook quality:**
The notebook is strong. Exercises are well-scaffolded, predict-before-run is consistently enforced, solutions include reasoning and common mistakes, VRAM management is careful and production-quality, and the cumulative exercise design (Exercise 1's best map feeds Exercise 3, Exercise 2's depth map feeds Exercise 4) is thoughtful. The main notebook finding (Exercise 3 `pass` issue) is a UX guard rail, not a conceptual problem.

---

## Review -- 2026-02-19 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All five findings from iteration 1 have been addressed. The GradientCards are now vivid and concrete, the Canny threshold failure modes describe specific visual phenomena rather than generic labels, the notebook Exercise 3 guard check is in place, spaced em dashes in the notebook have been fixed, and the img2img strength connection has been integrated into the conditioning scale section. No new issues were introduced by the fixes. Two minor polish items remain, neither of which affects the student's learning experience.

### Findings

#### [POLISH] -- Canny threshold values differ between lesson and notebook

**Location:** Section 4 (Canny threshold comparison GradientCards) vs notebook Exercise 1
**Issue:** The lesson's "Too Many Edges" GradientCard uses thresholds (30, 80), while the notebook's corresponding low-threshold pair is (50, 100). Similarly, the lesson's "Too Few Edges" uses (200, 300) which matches the notebook, and "Good Edges" uses (100, 200) which also matches. Only the "too many" case differs. The lesson frames these as "predictions you should carry into the notebook," but the student will find different threshold values when they open Exercise 1.
**Student impact:** Minimal. The student will see that (50, 100) already produces too many edges in the notebook, which confirms the lesson's principle. The lesson used a more extreme value (30, 80) to make the description vivid. But the mismatch could cause a brief moment of "wait, the lesson said 30/80 and the notebook says 50/100." No conceptual confusion results.
**Suggested fix:** Either change the lesson's "Too Many Edges" thresholds to (50, 100) to match the notebook exactly, or add a brief note in the GradientCard like "Notebook Exercise 1 uses (50, 100) which is already in the 'too many' range." Alternatively, leave as-is since the principle transfers regardless of specific threshold values.

#### [POLISH] -- Reference title uses spaced em dash

**Location:** References section, second reference title
**Issue:** The reference title `controlnet_aux — Preprocessors for ControlNet` uses a spaced em dash (` — `). This is rendered text visible to the student. The writing style rule specifies em dashes must have no spaces: `word—word`.
**Student impact:** None. This is a stylistic inconsistency in a reference link title.
**Suggested fix:** Change to `controlnet_aux—Preprocessors for ControlNet` or restructure as `controlnet_aux: Preprocessors for ControlNet` (using a colon, which may be more natural for a library subtitle).

### Review Notes

**Iteration 1 fixes verified:**
1. **GradientCards rewritten with vivid visual descriptions:** Confirmed. The "Too Few Edges" card now describes "walls shift position, proportions drift, the face may not align with the silhouette." The "Too Many Edges" card describes "skin looks like cracked plaster because the model is faithfully following the texture edges." The conditioning scale sweep cards describe specific visual phenomena at each level ("ghost lines from the edge map bleeding through surfaces"). These function as effective predict-before-run setups rather than vague assertions.
2. **Canny threshold failure modes made concrete:** Confirmed. Each threshold card now has a two-part structure: what the edge map looks like, then what ControlNet produces from it. Specific and vivid.
3. **Notebook Exercise 3 guard check added:** Confirmed. Cell-21 now starts with `if not sweep_results:` and prints a helpful message directing the student back to the TODO cell.
4. **Spaced em dashes replaced in notebook:** Confirmed. The notebook now uses unspaced em dashes throughout markdown cells and print statements.
5. **Img2img strength connection added:** Confirmed. The conditioning scale section (lines 601-607) now includes: "If the tradeoff feels familiar, it should--in Img2Img and Inpainting, the strength parameter controlled how much of the original image survived versus how much the model generated freely. Conditioning scale is the same idea applied to spatial maps." Integrated naturally into the existing paragraph.

**What works well:**
- The lesson is a clean CONSOLIDATE lesson. Zero new theory, all practical skill-building. The cognitive load trajectory (STRETCH -> CONSOLIDATE -> BUILD) is appropriate.
- The predict-before-run pattern is consistently used: the lesson's GradientCards frame descriptions as predictions, and the notebook exercises explicitly ask "predict before running."
- All five misconceptions from the planning document are addressed at the planned locations.
- The "Notice What Stays the Same" section remains one of the strongest design moments, concretely demonstrating the map-agnostic architecture through identical code with swappable checkpoints.
- The conditioning scale section now has three well-integrated connections to prior concepts (CFG guidance scale, "Nothing, Then a Whisper," img2img strength), making it feel like a natural extension of existing mental models rather than a new concept.
- The notebook is excellent: well-scaffolded (Guided -> Guided -> Supported -> Supported -> Independent), VRAM-conscious, cumulative (exercises feed forward), and all solutions include reasoning and common-mistake warnings.

**On the "no actual images" residual:**
The iteration 1 review flagged that a visual-centric lesson has no images. The fix reframed the text descriptions as vivid, concrete predictions for the notebook. This is now a defensible design choice: the lesson provides the conceptual framework and predictions, the notebook provides the visual experience. The framing paragraph ("These are the predictions you should carry into the notebook") makes this explicit. This is not ideal for a standalone lesson component, but it is adequate given that the notebook is the primary modality for a CONSOLIDATE lesson.
