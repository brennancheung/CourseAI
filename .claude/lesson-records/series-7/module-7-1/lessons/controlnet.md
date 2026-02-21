# Lesson: ControlNet -- Planning Document

**Module:** 7.1 (Controllable Generation)
**Position:** Lesson 1 of 3 in module, Lesson 1 of 11 in series
**Slug:** `controlnet`

---

## Phase 1: Orient

### Student State

The student has completed all of Series 6 (Stable Diffusion from scratch) and is entering the capstone series. Their knowledge base is deep and well-connected.

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| U-Net encoder-decoder architecture (encoder downsampling, bottleneck, decoder upsampling, skip connections) | DEVELOPED | 6.3.1 (unet-architecture) | Core mental model: "bottleneck decides WHAT, skip connections decide WHERE." Dimension walkthrough (64x64x64 -> 32x32x128 -> 16x16x256 -> 8x8x512 -> back up). Pseudocode forward pass with `cat(up(b), e2)` pattern. |
| U-Net skip connections as essential for denoising | DEVELOPED | 6.3.1 (unet-architecture) | "Keyhole and side doors" analogy. Without skips, decoder must reconstruct from bottleneck alone = blurry. Skip connections carry fine-grained spatial details. |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings) | DEVELOPED | 6.3.4 (text-conditioning-and-guidance) | "Same formula, different source for K and V." ComparisonRow with self-attention. Shape walkthrough (Q: 256xd_k, K: Txd_k). Per-spatial-location attention demonstrated. ~15+ lessons ago--NEEDS REACTIVATION per Reinforcement Rule. |
| Classifier-free guidance (training with text dropout, two forward passes, amplifying text direction) | DEVELOPED | 6.3.4 (text-conditioning-and-guidance) | Formula: epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond). "Contrast slider" analogy. Guidance scale tradeoff understood. |
| Full SD pipeline data flow (prompt -> CLIP -> U-Net denoising loop with CFG -> VAE decode) | DEVELOPED | 6.4.1 (stable-diffusion-architecture) | "Three translators, one pipeline." Tensor shapes at every handoff. Component modularity with parameter counts. "You know every line" of the inference pseudocode. |
| Component modularity (three independently trained models, swappable via tensor handoffs) | INTRODUCED | 6.4.1 (stable-diffusion-architecture) | Three GradientCards with parameter counts. Swappability demonstrated. WarningBlock: "Not End-to-End." |
| Frozen-model pattern (train new parameters while keeping original model frozen) | DEVELOPED | 6.5.1 (lora-finetuning), 6.5.3 (textual-inversion) | LoRA: freeze U-Net, train bypass matrices at cross-attention projections. Textual inversion: freeze entire model, train one embedding row. "Same detour, different highway." |
| Residual connections / bypass pattern | DEVELOPED | 3.2 (ResNet), 4.4.4 (LoRA) | ResNet: f(x) + x. LoRA: W_0 x + BA x. "Highway and detour" analogy deeply established. |
| Latent diffusion as architectural pattern | DEVELOPED | 6.3.5 (from-pixels-to-latents) | VAE encode -> diffuse in latent space -> VAE decode. 48x compression. "Translator between two languages." |
| Adaptive group normalization for timestep conditioning | DEVELOPED | 6.3.2 (conditioning-the-unet) | Timestep-dependent gamma and beta. "Same clock, different question." Global conditioning pattern. |

**Established mental models the student carries:**
- "Bottleneck decides WHAT, skip connections decide WHERE" (U-Net information flow)
- "Highway and detour" (residual/bypass connections, extended to LoRA)
- "Three translators, one pipeline" (SD modularity)
- "Same formula, different source for K and V" (cross-attention as a delta from self-attention)
- "Contrast slider" (CFG as text amplification)
- "The conductor's score" (timestep conditioning via adaptive group norm)
- "Finetuning is a refinement, not a revolution" (LoRA, textual inversion)

**What was explicitly NOT covered that is relevant:**
- ControlNet or any form of spatial conditioning beyond text (deferred from 6.5)
- Zero convolution (entirely new concept)
- Trainable encoder copy architecture (new)
- T2I-Adapter or other control mechanisms (out of scope for this lesson)
- How to combine frozen weights with new trainable weights via additive connections (LoRA does this at the weight level; ControlNet does it at the feature level)

**Readiness assessment:** The student is well-prepared. Every building block for ControlNet exists at DEVELOPED depth: U-Net encoder-decoder, skip connections, cross-attention, frozen-model pattern, residual/bypass connections, latent diffusion. The only gap is cross-attention reactivation (~15+ lessons ago). No missing prerequisites. The assembly of these pieces into ControlNet's architecture is the new insight.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how ControlNet adds spatial control to a frozen Stable Diffusion model by cloning the encoder, training only the clone on spatial maps, and connecting it to the frozen decoder via zero convolutions that guarantee the original model starts unchanged.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| U-Net encoder-decoder architecture | DEVELOPED | DEVELOPED | 6.3.1 | OK | Student needs to understand what "clone the encoder" means--encoder downsampling path, feature maps at each resolution, how they connect to decoder via skip connections. Has this at DEVELOPED with dimension walkthrough and pseudocode. |
| U-Net skip connections | DEVELOPED | DEVELOPED | 6.3.1 | OK | ControlNet's outputs are additive to the existing skip connections. Student must understand the encoder-to-decoder skip connection mechanism. Has this at DEVELOPED with "keyhole and side doors" analogy. |
| Cross-attention mechanism | INTRODUCED | DEVELOPED | 6.3.4 | OK (needs reactivation) | Student needs to know that text conditioning still operates via cross-attention even with ControlNet active. Does not need to re-derive cross-attention, just recall "Q from spatial, K/V from text." DEVELOPED but ~15+ lessons ago--brief reactivation per Reinforcement Rule. |
| Frozen-model training pattern | DEVELOPED | DEVELOPED | 6.5.1, 6.5.3 | OK | ControlNet freezes the original SD model and trains only the copy. Student has this pattern deeply from LoRA ("freeze everything except the bypass") and textual inversion ("freeze everything except one embedding row"). |
| Residual / additive bypass connections | DEVELOPED | DEVELOPED | 3.2, 4.4.4 | OK | Zero convolution outputs are *added* to the frozen skip connections, exactly like a residual bypass. Student has "highway and detour" at APPLIED depth from LoRA. Direct transfer. |
| Full SD pipeline data flow | DEVELOPED | DEVELOPED | 6.4.1 | OK | Student needs to understand where ControlNet fits in the pipeline (after CLIP, alongside the U-Net denoising loop). Pipeline is thoroughly mapped. |
| Component modularity | INTRODUCED | INTRODUCED | 6.4.1 | OK | ControlNet is another modular component that connects via tensor handoffs. Student has this at INTRODUCED, which is sufficient--they need to recognize the pattern, not apply it. |
| Latent diffusion (diffusion operates on 64x64x4 latents, not pixels) | INTRODUCED | DEVELOPED | 6.3.5 | OK | ControlNet processes spatial maps in latent space. Student needs to know that the U-Net operates on latent tensors. Has this well beyond the required depth. |

**Gap resolution:** No gaps. All prerequisites are met at or above required depth. Cross-attention requires brief reactivation (2-3 sentences + a callback, not a full section) because it was DEVELOPED ~15+ lessons ago and has not been directly used since. The series plan's reinforcement rule specifically calls this out: "Q from spatial features, K/V from text. ControlNet adds a new source of spatial features."

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "ControlNet fine-tunes or modifies the original SD model's weights" | The student has seen LoRA, which modifies the forward pass of the original model (even if weights are frozen, LoRA's output merges with the original). Natural assumption: ControlNet is like LoRA but for spatial control. | If ControlNet modified the original model, removing the ControlNet checkpoint would change generation quality. But it does not--the frozen model is bit-for-bit identical with or without ControlNet connected. The zero convolutions start at zero, so the ControlNet's initial output is exactly zero. Disconnect ControlNet and the output is the original model's output, unchanged. | Section 5 (Explain) after introducing zero convolution. WarningBlock. |
| "ControlNet replaces text conditioning--you use either text OR spatial maps" | The student has seen conditioning as monolithic: timestep conditioning or text conditioning. Adding a new conditioning signal might seem like it replaces the existing one. | Generate the same edge map with two different text prompts: "a cat in a living room" vs "a dog in a garden." Same edges, radically different images. Text still controls semantics; ControlNet controls structure. Both active simultaneously. | Section 7 (Elaborate). Concrete two-prompt example. |
| "The trainable encoder copy is a second U-Net (doubles the model size)" | The student hears "copy of the encoder" and thinks of a full duplicate model. The encoder is about half the U-Net's parameters, so this seems expensive. | The trainable copy is the encoder half only (not the decoder). And the frozen original encoder still runs--ControlNet's copy runs *in parallel*, its outputs are added to the frozen encoder's skip connections before they reach the decoder. Total added parameters are ~35-40% of the original U-Net, not 100%. Also, at inference, only one decoder runs. | Section 5 (Explain) when introducing the architecture. ConceptBlock with parameter breakdown. |
| "Zero convolution is a complex mechanism that does something clever with the spatial maps" | The name "zero convolution" sounds technical and novel. The student may expect a sophisticated operation. | A zero convolution is a 1x1 convolution initialized to all-zero weights and all-zero bias. That is the entire definition. The "cleverness" is not in the operation--it is in the *initialization*. Starting at zero means the ControlNet's output is exactly zero before any training, so connecting it to the frozen model changes nothing initially. Training gradually moves the weights away from zero, gradually introducing control signal. | Section 5 (Explain). Address immediately when introducing zero convolution. TipBlock. |
| "ControlNet needs to be trained from scratch for each new base model" | Reasonable assumption given that the trainable copy is initialized from the base model's encoder weights. | ControlNet checkpoints trained on SD v1.5 work with any SD v1.5-compatible model (same architecture, same latent space). The encoder copy learns to produce control signals in the same feature space as the original encoder, so it transfers across fine-tuned variants of the same base. However, a ControlNet trained for SD v1.5 does NOT work with SDXL (different architecture). | Section 7 (Elaborate). Brief note, not a full treatment. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Edge map (Canny) as spatial control: same edge map, two different text prompts producing different images with identical structure | Positive | Shows ControlNet's core value: spatial control is independent of semantic content. Text still matters for style/content. Demonstrates that ControlNet and text conditioning coexist. | Most intuitive spatial map--edges are visually obvious and easy to reason about. Two prompts demonstrate text/spatial orthogonality. |
| ControlNet architecture diagram: frozen U-Net + trainable encoder copy + zero convolution connections to decoder | Positive | Shows the full architecture clearly--what is frozen, what is trainable, where the outputs merge. Makes the "copy + connect" pattern concrete. | The architecture IS the concept. A diagram is essential for seeing the parallel processing paths and merge points. |
| Zero convolution at initialization: ControlNet freshly initialized, output is exactly zero, SD model produces its normal output | Positive (concrete numerical) | Demonstrates the "safe connection" property. At step 0 of training, ControlNet contributes nothing. The frozen model is provably unchanged. | Makes the zero-initialization property concrete and verifiable. This is the key insight that makes the architecture safe. |
| LoRA vs ControlNet comparison: LoRA changes *what* cross-attention projections compute, ControlNet adds *where* information via spatial features | Negative (boundary) | Clarifies the distinction between modifying existing conditioning (LoRA) and adding new conditioning (ControlNet). Prevents conflating the two frozen-model techniques. | Student has deep LoRA knowledge. Explicit comparison prevents the "ControlNet is LoRA for spatial stuff" misconception. |
| Depth map as second spatial map type: same architecture, different preprocessing, different control signal | Positive (generalization) | Shows the ControlNet pattern generalizes beyond edges. Different spatial maps provide different types of structural control, but the architecture is identical. Prevents over-fitting to "ControlNet = edge detection." | Confirms that the architecture is map-agnostic. The preprocessor changes; the ControlNet architecture does not. |

### Widget Consideration

**Widget needed:** No interactive widget for this lesson.

**Reasoning:** The core concept is architectural (how pieces connect), not mathematical (how values transform). The best representation is a clear architecture diagram showing frozen components, trainable components, zero convolution connections, and data flow. This is served well by a static Mermaid diagram or custom SVG, not an interactive widget. The student does not need to manipulate parameters to understand the architecture--they need to see the topology.

The *next* lesson (controlnet-in-practice) is where interactive exploration belongs: varying conditioning scale, comparing different spatial maps, seeing the control-creativity tradeoff. That is a parameter-space exploration that benefits from interactivity. This lesson is about understanding the architecture that makes it possible.

---

## Phase 3: Design

### Narrative Arc

You have built Stable Diffusion from the ground up. You can generate images from text, control style with LoRA, transform existing images with img2img, and even teach the model new concepts with textual inversion. But try to describe *spatial structure* in words. "Put the building on the left, with its edges following this specific contour, and a person standing at exactly this position with their arm raised at this angle." Text is the wrong tool for spatial precision. The prompt "a person standing with their right arm raised" gives you a person with a raised arm, but you cannot control *where* in the frame, *what pose* exactly, or *which edges* the composition follows. What you need is a way to hand the model a spatial map--an edge image, a depth map, a skeleton--and say "follow this structure, but use the text prompt for everything else." ControlNet solves this by adding a trainable encoder alongside the frozen SD model, connected in a way that *mathematically guarantees* the original model starts completely unchanged. Every piece of this architecture--the U-Net encoder, skip connections, frozen-model training, residual bypasses--is something you already know. The new idea is how they are assembled.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual (architecture diagram)** | Mermaid or SVG diagram showing the frozen U-Net (encoder + decoder with skip connections) alongside the trainable encoder copy, with zero convolution connections merging into the decoder. Color-coded: gray for frozen, violet for trainable, amber for zero convolutions. | The architecture *is* the concept. Topology is inherently visual. The student needs to see that two encoder paths exist in parallel, that the trainable copy's outputs are added at each resolution level, and that the frozen model's own encoder still operates. No amount of text alone communicates this as effectively as a diagram. |
| **Symbolic (pseudocode)** | Pseudocode showing the forward pass: frozen encoder produces features e1-e4, trainable encoder produces c1-c4 from spatial map input, zero convolutions produce z1-z4 (initially all zeros), decoder receives (e_i + z_i) at each skip connection. | Makes the additive connection mechanism precise and unambiguous. Student can trace the data flow line by line. Connects to the `cat(up(b), e2)` pseudocode from 6.3.1--the only change is `cat(up(b), e2 + zero_conv(c2))`. |
| **Intuitive / "Of course"** | The "of course" chain: (1) You want spatial control. (2) The encoder is where spatial features are extracted. (3) You cannot retrain the encoder without breaking the model. (4) So copy the encoder, train the copy on spatial maps, and add its outputs to the original. (5) Initialize the connection at zero so the original model starts unchanged. Each step follows from the previous one. | Transforms the architecture from "clever trick" to "obvious in retrospect." The student should feel that they could have designed this themselves. This is the capstone-series tone: guided frontier exploration, not lecture. |
| **Concrete example (numerical)** | Zero convolution at initialization: a 1x1 conv with weight=0.0 and bias=0.0, applied to a feature map of any values, produces all zeros. e_i + 0 = e_i. The frozen model's output is unchanged. After 100 training steps, weight might be 0.03, producing a small signal. The control signal fades in gradually. | Makes the zero-initialization property tangible with actual numbers. Addresses the "zero convolution is complex" misconception by showing it is literally a conv with zeros. The gradual fade-in is the key safety property. |
| **Verbal/Analogy** | "The training wheels don't touch the ground at first." Zero convolutions start at zero output, like training wheels raised off the pavement. They gradually lower as training progresses, taking on more of the steering. The bike (original model) rides exactly as before until the training wheels engage. | Captures the gradual-engagement property that makes ControlNet safe. Extends the student's existing "highway and detour" mental model--this is a detour that starts as a dead end and gradually connects. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts
  1. **Trainable encoder copy architecture** -- cloning the U-Net encoder, processing spatial maps through it in parallel, adding outputs to the frozen encoder's features
  2. **Zero convolution** -- 1x1 conv initialized to all zeros as the connection mechanism; guarantees zero initial contribution
- **Everything else is connection and application:** U-Net encoder (DEVELOPED), skip connections (DEVELOPED), frozen-model pattern (DEVELOPED), residual/additive connections (DEVELOPED), cross-attention reactivation (brief callback)
- **Previous lesson load:** The student's most recent lesson was textual-inversion (6.5.3), which was STRETCH. Before that, img2img-and-inpainting (6.5.2, BUILD) and lora-finetuning (6.5.1, BUILD). The student has had a mix of load levels.
- **This lesson's load:** STRETCH. Two genuinely new concepts, and the architectural assembly is novel even though all pieces are familiar. But this is bounded STRETCH--the new concepts are simple individually (copy an encoder, initialize a conv to zero), and the cognitive challenge is in seeing how they fit together. The student's deep U-Net knowledge makes this achievable.
- **Series context:** First lesson in a new series. The series plan says "ControlNet before theory" because it gives the student a concrete "win" before the theoretical work in Module 7.2. High motivation expected. The capstone-series tone ("let's read the frontier together") should make this feel exciting, not overwhelming.

### Connections to Prior Concepts

- **U-Net encoder -> ControlNet trainable copy:** "You know the encoder path: 64x64 -> 32x32 -> 16x16 -> 8x8. ControlNet makes an exact copy of this path. Same layers, same initial weights, same downsampling. But it takes a spatial map as input instead of a noisy latent."
- **Skip connections -> ControlNet merge points:** "You know that encoder features are concatenated with decoder features at matching resolutions. ControlNet's outputs are *added* to these encoder features before the concatenation happens. Same skip connections, enriched with spatial control."
- **LoRA "highway and detour" -> Zero convolution "detour that starts disconnected":** "LoRA adds a trainable bypass (BA) to frozen weights W. The bypass starts near-zero (B=0 initialization). ControlNet does the same thing at the *feature* level: the zero convolution starts at exactly zero, so the frozen features pass through unchanged. Same principle, different scale."
- **Frozen-model pattern -> ControlNet freezing:** "LoRA freezes the U-Net weights and trains small bypass matrices. Textual inversion freezes the entire model and trains one embedding row. ControlNet freezes the entire original model and trains a copy of the encoder. Same pattern, third application."
- **Cross-attention -> Still active with ControlNet:** "Remember: Q from spatial features, K/V from text. ControlNet does not replace cross-attention. It enriches the spatial features that produce Q. The text conditioning pathway is unchanged."
- **Component modularity -> ControlNet as another swappable component:** "SD is three independently trained models connected by tensor handoffs. ControlNet adds a fourth: you can swap ControlNet checkpoints (edge, depth, pose) without retraining anything else."

**Analogies from prior lessons that extend here:**
- "Highway and detour" (LoRA) -> extends to "detour that starts as a dead end" (zero convolution)
- "Bottleneck decides WHAT, skip connections decide WHERE" -> extends to "ControlNet influences WHERE via the skip connections"
- "Three translators, one pipeline" -> extends to "four translators" with ControlNet as the spatial translator

**Analogies from prior lessons that could be misleading:**
- "Same detour, different highway" (LoRA) could suggest ControlNet is LoRA applied to the encoder. It is not--ControlNet trains a full copy of the encoder, not small bypass matrices. ControlNet's trainable parameter count is ~300M, not ~1M. Explicit distinction needed.

### Scope Boundaries

**This lesson IS about:**
- The ControlNet architecture: trainable encoder copy + zero convolution + additive connection to frozen decoder
- Why this architecture preserves the original model (zero-initialization safety)
- How spatial maps (edges, depth, pose) serve as the conditioning input
- Where ControlNet fits in the SD pipeline
- How ControlNet coexists with text conditioning

**This lesson is NOT about:**
- How to preprocess images into spatial maps (Canny, depth estimation, OpenPose)--that is lesson 2
- Conditioning scale parameter or control-creativity tradeoff--lesson 2
- Stacking multiple ControlNets--lesson 2
- IP-Adapter or image-based conditioning--lesson 3
- T2I-Adapter (mentioned for vocabulary only)
- Training a ControlNet from scratch (this series does not include training from scratch)
- Implementation details beyond pseudocode-level forward pass

**Target depths:**
- Trainable encoder copy architecture: DEVELOPED
- Zero convolution mechanism: DEVELOPED
- ControlNet's coexistence with text conditioning: INTRODUCED
- ControlNet as a modular, swappable component: INTRODUCED

### Lesson Outline

**1. Context + Constraints**
- This is the first lesson of Series 7: Post-SD Advances. The student has built SD from scratch and customized it.
- Capstone tone: "You understand the machine. Now let's see what the field built on top of it."
- This lesson covers the ControlNet architecture. We will not use ControlNet hands-on (that is next lesson). We will not cover IP-Adapter (that is lesson 3).
- By the end: the student can draw the ControlNet architecture, explain why zero convolution makes it safe, and trace the forward pass.

**2. Recap (brief cross-attention reactivation)**
- 2-3 sentences reactivating cross-attention: "Q from spatial features, K/V from text embeddings. Each spatial location attends independently to all text tokens, producing spatially-varying conditioning."
- Callback to the 6.3.4 mental model: "Timestep tells the network WHEN (adaptive norm, global). Text tells it WHAT (cross-attention, spatially varying)."
- Transition: "But what about WHERE? What if you want to specify spatial structure directly?"

**3. Hook: The Spatial Control Problem (type: before/after + challenge preview)**
- Present the problem concretely: show a text prompt ("a house on a hill, watercolor painting") and note that you cannot control the composition--where the house is, what angle the hill has, where the horizon falls.
- Present what you *want*: a spatial map (edge image) alongside a text prompt, producing an image that follows the edges precisely while using the text for style and content.
- Challenge the student: "You have all the pieces to design this. The U-Net encoder extracts spatial features. You know how to freeze models and train new parameters alongside them. You know that additive connections (residual/bypass) can be initialized safely. How would you add spatial control without breaking anything?"
- Brief think-before-reveal moment. The student should attempt a mental design before seeing ControlNet's actual architecture.

**4. Explain: ControlNet Architecture**
- **The constraint:** You cannot modify the frozen SD model's weights. Any new conditioning must be *additive* and must start with zero contribution (so the model is unchanged at initialization).
- **The insight (intuitive "of course" chain):**
  1. Spatial features live in the encoder. (Student knows this.)
  2. Training new spatial features requires an encoder that takes spatial maps as input.
  3. You cannot retrain the existing encoder (it is frozen, it works).
  4. So: copy the encoder. Initialize it with the original weights (good starting point). Train the copy on spatial maps.
  5. Add the copy's outputs to the original encoder's outputs at each resolution.
  6. Initialize the connections at zero so the model starts unchanged.
- **Architecture diagram:** Mermaid or SVG showing:
  - Left: frozen U-Net (encoder path -> bottleneck -> decoder path, with skip connections)
  - Right: trainable encoder copy (same structure as encoder, takes spatial map as input)
  - Zero convolution connections from trainable copy to decoder at each resolution level
  - Color coding: gray = frozen, violet = trainable, amber = zero convolution
- **Pseudocode forward pass:**
  ```
  # Frozen encoder (unchanged)
  e1 = frozen_encoder_block_1(z_t)    # 64x64
  e2 = frozen_encoder_block_2(e1)     # 32x32
  e3 = frozen_encoder_block_3(e2)     # 16x16
  e4 = frozen_encoder_block_4(e3)     # 8x8

  # Trainable copy (spatial map input)
  c1 = copy_encoder_block_1(spatial_map)  # 64x64
  c2 = copy_encoder_block_2(c1)           # 32x32
  c3 = copy_encoder_block_3(c2)           # 16x16
  c4 = copy_encoder_block_4(c3)           # 8x8

  # Zero convolution connections (initialized to zero)
  z1 = zero_conv_1(c1)  # starts at 0.0
  z2 = zero_conv_2(c2)  # starts at 0.0
  z3 = zero_conv_3(c3)  # starts at 0.0
  z4 = zero_conv_4(c4)  # starts at 0.0

  # Decoder receives enriched skip connections
  d4 = decoder_block_4(bottleneck)
  d3 = decoder_block_3(cat(d4, e3 + z3))  # original + control
  d2 = decoder_block_2(cat(d3, e2 + z2))
  d1 = decoder_block_1(cat(d2, e1 + z1))
  ```
- **Connect to existing mental model:** "The only change from the U-Net you know is `e_i + z_i` instead of `e_i`. One addition per resolution level. Everything else is identical."

**5. Check: Predict-and-Verify**
- Question 1: "Before any training, what is the ControlNet's contribution to the decoder?" (Answer: exactly zero. Zero convolutions produce all zeros. The model's output is identical to vanilla SD.)
- Question 2: "If you disconnect the ControlNet entirely (remove the z_i additions), does the output change?" (Answer: no. The frozen model was never modified. This is the safety guarantee.)
- Question 3: "Why initialize the trainable copy from the original encoder's weights instead of random weights?" (Answer: the copy starts with knowledge of how to extract spatial features from images. Training only needs to adapt it from "extract features from noisy latents" to "extract features from spatial maps." Much easier than learning from scratch.)

**6. Explain: Zero Convolution**
- **What it is:** A 1x1 convolution with weights initialized to 0.0 and bias initialized to 0.0. That is the complete definition.
- **Concrete numerical example:**
  - Feature map: any values (e.g., [[0.5, -0.3], [1.2, 0.8]])
  - Zero conv weight: 0.0, bias: 0.0
  - Output: [[0.0, 0.0], [0.0, 0.0]]
  - Added to frozen encoder feature: unchanged. `e_i + 0 = e_i`.
- **After 100 training steps:** weight might be 0.03, bias 0.001. Output: [[0.016, -0.008], [0.037, 0.025]]. A tiny signal. The control fades in gradually.
- **"Training wheels" analogy:** The training wheels do not touch the ground at first. As training progresses, they gradually lower and begin to steer.
- **Connect to LoRA B=0 initialization:** "LoRA initializes matrix B to zero so the bypass output starts at zero: BA_x = 0. Zero convolution initializes the entire conv to zero for the same reason: zero_conv(c_i) = 0. Same principle--ensure the frozen model starts unchanged--applied at the feature level instead of the weight level."
- **Address misconception:** WarningBlock: "Zero convolution is not a special operation. It is a standard 1x1 conv layer with a specific initialization. The name describes the initialization, not the operation."

**7. Elaborate: Coexistence with Text Conditioning**
- **Address the "replaces text" misconception:**
  - Concrete example: same Canny edge map, two prompts: "a cat in a living room, oil painting" and "a robot in a factory, cyberpunk style." Same spatial structure, completely different content and style.
  - The edge map controls WHERE things go. The text prompt controls WHAT those things are and what they look like.
- **Where each conditioning type operates:**
  - Text conditioning: cross-attention (Q from spatial features, K/V from CLIP text embeddings). Unchanged by ControlNet.
  - Timestep conditioning: adaptive group norm. Unchanged by ControlNet.
  - Spatial conditioning (ControlNet): additive to encoder features at skip connections. A new, separate conditioning pathway.
- **Reactivation payoff:** "Timestep says WHEN. Text says WHAT. ControlNet says WHERE. Three conditioning signals, three mechanisms, all coexisting."

**8. Check: Transfer Questions**
- Question 1: "A friend says they want to use ControlNet to change the *style* of an image. Is ControlNet the right tool?" (Answer: No. ControlNet provides spatial/structural control. For style, use LoRA or prompt engineering. ControlNet controls edges, depth, pose--structure, not style.)
- Question 2: "Why does the trainable copy process the spatial map through the *same architecture* as the encoder (conv blocks, downsampling, etc.) rather than a simple MLP?" (Answer: The spatial map needs to produce features at every resolution level to match the decoder's expectations at each skip connection. An MLP would produce a single feature vector, not multi-resolution feature maps. The encoder architecture is designed for exactly this multi-resolution spatial feature extraction.)

**9. Practice: Notebook Exercises (Colab)**
- **Exercise 1 (Guided): Inspect the ControlNet Architecture**
  - Load a pre-trained ControlNet (Canny) and the frozen SD model via diffusers
  - Print parameter counts: frozen U-Net, ControlNet, total
  - Identify which parameters are trainable vs frozen
  - Verify that ControlNet's encoder blocks match the U-Net's encoder blocks in structure
  - **Tests:** architecture recognition, modular component concept

- **Exercise 2 (Guided): Verify the Zero-Initialization Property**
  - Create a fresh (untrained) zero convolution layer: `nn.Conv2d(c, c, 1)` with weights and bias set to 0
  - Pass a random feature map through it, verify output is all zeros
  - Add the output to a "frozen" feature tensor, verify the tensor is unchanged
  - **Tests:** zero convolution mechanism, safety guarantee property

- **Exercise 3 (Supported): Trace the Forward Pass**
  - Load a pre-trained ControlNet and manually trace one forward pass
  - Feed a Canny edge map (provided as a pre-computed tensor, no preprocessing needed)
  - Inspect feature map shapes at each resolution level from both the frozen encoder and the ControlNet copy
  - Verify that the additive connections produce features with matching shapes
  - **Tests:** architectural understanding, multi-resolution feature extraction, additive connection

- **Exercise 4 (Independent): ControlNet vs Vanilla SD Comparison**
  - Generate the same prompt with and without ControlNet conditioning
  - Use the same seed, same prompt, same sampler settings
  - Observe: without ControlNet, the model generates freely; with ControlNet + edge map, the composition follows the edges
  - Vary the text prompt while keeping the edge map fixed--verify that structure is preserved while content changes
  - **Tests:** coexistence of text and spatial conditioning, ControlNet's role in the pipeline

- **Scaffolding progression:** Guided -> Guided -> Supported -> Independent. First two exercises isolate the two core concepts (architecture, zero conv). Third exercise integrates them. Fourth requires the student to design their own experiment.
- **Exercises are independent** (not cumulative)--each can be done standalone.
- **Solutions should emphasize:** The "of course" reasoning ("the output is zero because the weights are zero, of course"). The safety guarantee ("disconnect ControlNet and nothing changes, of course"). The coexistence insight ("edges control structure, text controls content, of course").

**10. Summarize**
- ControlNet adds spatial conditioning to a frozen SD model via three pieces: (1) a trainable copy of the encoder that processes spatial maps, (2) zero convolutions that connect the copy to the frozen decoder with zero initial contribution, (3) additive connections at each resolution level that enrich the skip connection features.
- Echo the mental model: "The trainable copy extracts spatial features from your control image. Zero convolutions ensure it starts silent. Training gradually turns up the volume. The frozen model never changes."
- Key distinction: ControlNet adds a new conditioning *dimension* (WHERE), not a new conditioning *mechanism*. The mechanism is the same as everything the student already knows: additive connections, frozen weights, encoder-decoder architecture.

**11. Next Step**
- Next lesson: ControlNet in Practice. We will use real preprocessors (Canny edge detection, depth estimation, OpenPose), explore the conditioning scale parameter (how much to follow the control), and stack multiple ControlNets for combined control. The architecture you learned today is the engine; next lesson you drive it.

---

## Review — 2026-02-16 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: MAJOR REVISION

One critical finding (missing notebook) and three improvement findings exist. The lesson's pedagogical design is strong overall -- the "of course" chain, capstone tone, modality coverage, and scope discipline are all well-executed. The critical finding is procedural (the notebook build is in-progress per task #6, not a lesson design failure), and the improvements target specific weak spots that would make the lesson genuinely better for the student.

### Findings

#### [CRITICAL] — Notebook missing

**Location:** Practice section (Section 10)
**Issue:** The planning document specifies 4 exercises (Guided: inspect architecture, Guided: verify zero initialization, Supported: trace forward pass, Independent: ControlNet vs vanilla comparison). The lesson's practice section describes these exercises and links to a Colab URL (`notebooks/7-1-1-controlnet.ipynb`). However, no notebook file exists in the repository at this path or any variant of it.
**Student impact:** The student reads about 4 hands-on exercises, clicks the Colab link, and gets a 404 or empty notebook. The practice section -- which is essential for moving the two core concepts from INTRODUCED to DEVELOPED -- is entirely missing. Without it, the lesson teaches architecture conceptually but the student never touches real tensors or verifies the zero-initialization property with actual code.
**Suggested fix:** Complete the notebook build (task #6 is in-progress). The review can re-evaluate the notebook in iteration 2 once it exists. The lesson component itself does not need changes for this finding.

#### [IMPROVEMENT] — Mermaid diagram topology ambiguous at the additive merge point

**Location:** Architecture diagram (Section 5, the Mermaid diagram)
**Issue:** The diagram shows two separate pathways to each decoder block: (1) the frozen encoder's skip connections labeled `"skip + z₃"` with dotted arrows from E3 to D3, and (2) the zero convolution outputs with solid arrows `C3 --> ZC3 --> D3`. These appear as two independent input paths to each decoder block. The additive merge (`e_i + z_i`) that makes ControlNet safe is not visually represented as a single merge point -- it looks like the decoder receives two separate inputs.
**Student impact:** The student might interpret the diagram as: "the frozen encoder sends its skip connection AND the zero conv sends a separate signal, and the decoder somehow combines them internally." This is subtly wrong -- the addition happens BEFORE the decoder receives the concatenated features. The pseudocode (`cat(d4, e3 + z3)`) clarifies this, but the diagram is the first visual the student encounters, and first impressions are sticky. The diagram and pseudocode tell slightly different stories about where the merge happens.
**Suggested fix:** Modify the Mermaid diagram to show explicit merge/addition nodes at each resolution level. For example, add `ADD3["+"]` nodes where the frozen encoder skip and the zero conv output meet, with the combined output then flowing to the decoder block. This makes the additive nature visually explicit and aligns the diagram with the pseudocode. Alternatively, add a caption below the diagram clarifying that the zero conv outputs are added to the skip connections, not sent separately to the decoder.

#### [IMPROVEMENT] — Misconception #1 (ControlNet modifies original model weights) not explicitly addressed

**Location:** Entire lesson
**Issue:** The planning document identifies "ControlNet fine-tunes or modifies the original SD model's weights" as Misconception #1, to be addressed in Section 5 with a WarningBlock after introducing zero convolution. The negative example planned is: "If ControlNet modified the original model, removing the ControlNet checkpoint would change generation quality. But it does not." The built lesson addresses this implicitly through Check Q2 (disconnection test) and through the zero-initialization explanation, but there is no dedicated WarningBlock that explicitly names and disproves this misconception. The student has to infer from Q2 that the original weights are unchanged.
**Student impact:** A student with deep LoRA knowledge might carry the assumption that ControlNet, like LoRA, modifies the forward pass in a way that the model "knows about" the new component. The implicit treatment through Q2 may not fully dislodge this assumption because Q2 focuses on "does the output change?" (behavioral equivalence) rather than "are the weights changed?" (parameter-level). The distinction matters because LoRA's forward pass is `Wx + BAx` (the original weight W participates alongside the bypass), while ControlNet's frozen model weights are literally never touched -- the addition happens at the feature level, not the weight level.
**Suggested fix:** Add a WarningBlock in the zero convolution section (Section 7, after the numerical example) explicitly stating: "ControlNet does NOT modify the original model's weights. Not a single parameter in the frozen U-Net changes. The zero convolution output is added to the encoder's features, not to its weights. Disconnect the ControlNet and the frozen model is bit-for-bit identical -- it never knew ControlNet existed." This makes the distinction explicit rather than requiring the student to infer it.

#### [IMPROVEMENT] — "Training wheels" analogy is slightly misleading

**Location:** Zero Convolution section, InsightBlock "Training Wheels Analogy"
**Issue:** The analogy says "The training wheels do not touch the ground at first. As training progresses, they gradually lower and begin to steer." Training wheels in reality are always touching the ground -- they are safety devices that prevent falling, not steering devices. They do not "gradually lower." The analogy mixes the metaphor: (a) training wheels that do not touch the ground is not how real training wheels work, (b) training wheels steer the bike is not what training wheels do -- they prevent tipping. The zero convolution's behavior (starts at zero, gradually contributes a steering signal) maps better to something like a co-pilot who is initially silent and gradually starts making suggestions.
**Student impact:** Mild confusion. The student likely understands the intended meaning (starts at zero contribution, gradually increases), but the analogy's vehicle does not hold up under scrutiny. For a capstone-level student who thinks carefully about analogies, this may feel off. It will not cause a wrong mental model, but it is weaker than it could be.
**Suggested fix:** Either revise the analogy to something more accurate (e.g., "a backup singer who starts with their microphone muted -- the sound engineer gradually raises the volume as they find their pitch") or simplify to the more generic language already used in the prose: "The control signal fades in gradually -- nothing at first, then a whisper, then a clear voice." The "whisper to clear voice" metaphor in the main prose is actually stronger than the training wheels analogy.

#### [POLISH] — Spaced em dashes in Python code comments

**Location:** Lines 349 and 564 (inside `CodeBlock` code strings)
**Issue:** Two instances of spaced em dashes in Python code comments: `# Frozen encoder (unchanged — same forward pass you know)` and `# A tiny signal — the control is fading in gradually`. While these are inside code blocks (not prose), the writing style rule says "Em dashes must have no spaces: `word—word` not `word — word`." Code comments are student-visible text.
**Student impact:** Negligible. These are code comments, and the spacing is a minor style inconsistency.
**Suggested fix:** Replace ` — ` with `—` in both code comment strings, or rephrase the comments to avoid em dashes entirely (e.g., `# Frozen encoder (unchanged, same forward pass you know)` and `# A tiny signal: the control is fading in gradually`).

#### [POLISH] — E4 skip connection missing from Mermaid diagram

**Location:** Architecture diagram (Section 5, Mermaid diagram)
**Issue:** The diagram shows skip connections from E1, E2, and E3 to D1, D2, and D3 respectively, but E4 does not have a skip connection to D4. The flow goes `E4 --> BN --> D4`. Meanwhile, the zero convolution output from C4 goes `C4 --> ZC4 --> D4`. In a standard U-Net, the innermost encoder block's output goes through the bottleneck rather than via a skip connection -- so this is architecturally correct. However, the pseudocode says `d3 = decoder_block_3(cat(d4, e3 + z3))` which starts at d4, not showing how z4 connects. The z4 connection is defined (`z4 = zero_conv_4(c4)`) but never used in the pseudocode's decoder lines. The diagram shows `ZC4 --> D4` but the pseudocode does not show this connection.
**Student impact:** Minor inconsistency. A careful student might notice that z4 is computed but never appears in the decoder pseudocode. They might wonder where z4 goes. The diagram suggests it goes to D4 but the pseudocode does not show it.
**Suggested fix:** Either add a comment in the pseudocode showing where z4 is used (e.g., `d4 = decoder_block_4(bottleneck + z4)` or a comment `# z4 is added at the bottleneck-to-decoder transition`), or note that the actual ControlNet paper has connections at the bottleneck as well and show this in the pseudocode. Alternatively, remove z4 from the pseudocode if it is not used, and simplify the diagram to match.

### Review Notes

**What works well:**
- The "of course" chain is the lesson's strongest pedagogical move. It transforms ControlNet from "a clever paper trick" into "the obvious solution given constraints you already understand." This is exactly the right tone for a capstone series.
- The design challenge (Section 4) engages the student as a co-designer before revealing the architecture. Excellent for capstone-level engagement.
- Modality coverage is strong (5 distinct modalities for the core concept). The pseudocode + diagram + numerical example + analogy + intuitive chain cover different learning styles effectively.
- Scope discipline is excellent. The lesson stays tightly within its boundaries and the "NOT" items in the scope block are specific and useful.
- The capstone tone is well-maintained throughout -- "guided frontier exploration, not hand-holding."
- The three-column WHEN/WHAT/WHERE GradientCards are a satisfying synthesis that ties the entire conditioning story together.

**Systemic pattern:**
- The Mermaid diagram ambiguity (merge point not visually clear) is the same type of issue seen in earlier lessons where the diagram and code/prose tell slightly different structural stories. Architecture diagrams benefit from explicit merge/addition nodes when additive connections are the key concept.

**Iteration guidance:**
- Fix the three improvement findings (diagram merge points, explicit Misconception #1 WarningBlock, training wheels analogy).
- The notebook critical will be resolved by completing the notebook build (task #6).
- Re-invoke review after fixes for iteration 2.

---

## Fixes Applied — 2026-02-16 (Post-Iteration 1)

### Finding #1 [CRITICAL] — Notebook missing
**Status:** RESOLVED (built in parallel, no lesson component changes needed).

### Finding #2 [IMPROVEMENT] — Mermaid diagram topology ambiguous at the additive merge point
**Status:** FIXED. Added explicit `ADD1`, `ADD2`, `ADD3` merge nodes (circle shape with `"+"` label) inside the frozen model subgraph. Frozen encoder skip connections now flow through the `+` nodes before reaching decoder blocks. Zero convolution outputs also flow into these same `+` nodes. This makes the additive merge visually explicit and aligns the diagram with the pseudocode (`cat(d4, e3 + z3)`). Added a `merge` classDef (emerald) for the `+` nodes. ZC4 still flows directly to D4 (bottleneck connection, no skip at that level).

### Finding #3 [IMPROVEMENT] — Misconception #1 not explicitly addressed
**Status:** FIXED. Added a dedicated `WarningBlock` titled "ControlNet Does NOT Modify Frozen Weights" after the zero conv numerical example and InsightBlock. Explicitly states: features not weights, contrasts with LoRA's `Wx + BAx`, bit-for-bit identical frozen model, disconnect test.

### Finding #4 [IMPROVEMENT] — "Training wheels" analogy misleading
**Status:** FIXED. Replaced the "Training Wheels Analogy" InsightBlock with "Nothing, Then a Whisper"—using the volume/whisper/clear-voice metaphor already present in the prose. The metaphor maps accurately to zero conv behavior (starts silent, gradually gains volume).

### Finding #5 [POLISH] — Spaced em dashes in Python code comments
**Status:** FIXED. Rephrased both code comments to avoid em dashes entirely: `unchanged — same` -> `unchanged, same` and `A tiny signal — the control` -> `A tiny signal: the control`.

### Finding #6 [POLISH] — z4 computed but never used in decoder pseudocode
**Status:** FIXED. Changed `d4 = decoder_block_4(bottleneck)` to `d4 = decoder_block_4(bottleneck + z4)` with a `# z4 added at bottleneck` comment. This aligns with the diagram (which shows ZC4 --> D4) and uses z4 consistently.

---

## Review — 2026-02-16 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 1

### Verdict: NEEDS REVISION

All iteration 1 findings were correctly applied in the lesson component. The Mermaid diagram now has explicit `+` merge nodes, the WarningBlock for Misconception #1 is present and explicit, the whisper metaphor replaced training wheels, code comment em dashes are fixed, and z4 is used consistently in the pseudocode. The lesson `.tsx` is strong. The remaining issues are in the notebook, which was built in parallel and did not receive the same iteration 1 fixes.

### Iteration 1 Fix Verification

| Finding | Status | Verified |
|---------|--------|----------|
| #1 [CRITICAL] Notebook missing | RESOLVED | Notebook exists at `notebooks/7-1-1-controlnet.ipynb` with 4 exercises matching the plan. |
| #2 [IMPROVEMENT] Mermaid merge points | FIXED | Lines 285-287: explicit `ADD1`, `ADD2`, `ADD3` circle nodes with `"+"` labels. Zero conv outputs and frozen encoder skips both flow into these merge nodes. The merge classDef is emerald-colored. Diagram now aligns with the pseudocode. |
| #3 [IMPROVEMENT] Misconception #1 WarningBlock | FIXED | Lines 598-606: dedicated `WarningBlock` titled "ControlNet Does NOT Modify Frozen Weights." Explicitly contrasts features vs weights, references LoRA's `Wx + BAx`, states bit-for-bit identical, disconnect test. Well-placed after the numerical example. |
| #4 [IMPROVEMENT] Training wheels analogy | FIXED | Lines 586-591: InsightBlock now titled "Nothing, Then a Whisper" using the volume/whisper/clear-voice metaphor. Maps accurately to zero conv behavior. Prose in line 579 also uses the same metaphor ("nothing at first, then a whisper, then a clear voice"). Consistent throughout the lesson. |
| #5 [POLISH] Spaced em dashes in code comments | FIXED | Line 354: `# Frozen encoder (unchanged, same forward pass you know)` — comma, not em dash. Line 569: `# A tiny signal: the control is fading in gradually` — colon, not em dash. Both rephrased cleanly. |
| #6 [POLISH] z4 unused in pseudocode | FIXED | Line 373: `d4 = decoder_block_4(bottleneck + z4)` with `# z4 added at bottleneck` comment. Consistent with the diagram (ZC4 --> D4). |

All six iteration 1 findings are correctly resolved in the lesson component.

### Findings

#### [IMPROVEMENT] — Notebook uses replaced "training wheels" analogy

**Location:** Notebook cell-13, final print statements (Exercise 2: Verify the Zero-Initialization Property)
**Issue:** The lesson replaced the "training wheels" analogy with the "Nothing, Then a Whisper" metaphor in iteration 1 (Finding #4). However, the notebook still prints: `"This is the 'training wheels' analogy: they do not touch the ground at first, then gradually lower and begin to steer."` The notebook was built in parallel with the lesson and did not receive this fix. The lesson and notebook now use different metaphors for the same concept.
**Student impact:** The student reads "Nothing, Then a Whisper" in the lesson, then opens the notebook and encounters "training wheels" — a different analogy that was specifically rejected for being inaccurate (training wheels do not start raised off the ground, and they prevent tipping rather than steering). The terminology mismatch undermines the coherence between the two learning artifacts. The student may wonder which metaphor is "right" or feel the lesson and notebook were not written together.
**Suggested fix:** Replace the training wheels print statements in cell-13 with the whisper metaphor: `"The zero conv started silent (all zeros) and gradually learned to produce"`, `"a small signal. Nothing at first, then a whisper, then a clear voice."`, `"Training gradually turns up the volume."` This aligns the notebook with the lesson's revised language.

#### [IMPROVEMENT] — Notebook Exercise 4 solution loads two full pipelines simultaneously, risking OOM on T4

**Location:** Notebook cell-24 (Exercise 4 solution)
**Issue:** The solution creates both `pipe_cn` (StableDiffusionControlNetPipeline) and `pipe_vanilla` (StableDiffusionPipeline), each calling `.from_pretrained()` and `.to(device)`. At this point in the notebook, the student already has `controlnet`, `unet`, and `text_encoder` on GPU from Exercise 3. Loading two additional full pipelines means the GPU holds: the standalone ControlNet model (~600MB fp16), the standalone U-Net (~1.7GB fp16), the standalone text encoder (~250MB fp16), the ControlNet pipeline's U-Net + text encoder + VAE (~3.5GB fp16), and the vanilla pipeline's U-Net + text encoder + VAE (~3.5GB fp16). Total: ~9.5GB+ on a T4 with 16GB VRAM. This will likely OOM or be extremely tight.
**Student impact:** The student follows the solution, hits an OOM error on a free-tier T4, and does not understand why. The solution's note says "you may need to delete the previous models" but this is buried at the end and comes after the OOM has already occurred. An Independent exercise that crashes on the expected hardware is a frustrating experience.
**Suggested fix:** Add a VRAM cleanup cell before Exercise 4's code cell that deletes the standalone models from earlier exercises (`del controlnet, unet, unet_device, controlnet_device, text_encoder; torch.cuda.empty_cache()`). Also restructure the solution to load sequentially: build vanilla pipeline, generate vanilla image, delete vanilla pipeline, build ControlNet pipeline, generate both ControlNet images. This is the same pattern used in the 6-5-1 notebook after iteration 2 fixed the same issue.

#### [POLISH] — Notebook has three spaced em dashes in student-visible text

**Location:** Notebook cell-7 code comment (line 181), cell-20 code comment (line 606), cell-20 print statement (line 625)
**Issue:** Three instances of ` — ` (spaced em dash) in student-visible text: (1) `# and "zero_convs" — the zero convolution connections.` (2) `# This should produce a different noise prediction — the spatial conditioning is absent.` (3) `print(f"in both cases — the difference comes entirely from the additive z_i terms.")`. The lesson's code comments were fixed in iteration 1 (Finding #5), but the notebook was not updated.
**Student impact:** Negligible. Style inconsistency between lesson and notebook code comments.
**Suggested fix:** Rephrase to avoid em dashes, matching the lesson's fix pattern: (1) use a comma or colon, (2) use a colon or period, (3) use a colon.

### Review Notes

**What works well:**
- All six iteration 1 findings were correctly applied in the lesson component. The fixes are clean and well-integrated — they do not feel patched on.
- The WarningBlock for Misconception #1 (lines 598-606) is particularly well-written. It explicitly names the feature-vs-weight distinction and contrasts with LoRA's `Wx + BAx` formula, which is exactly the comparison a student with deep LoRA knowledge needs.
- The "Nothing, Then a Whisper" InsightBlock (lines 586-591) is a better metaphor than the replaced training wheels. The volume/whisper/voice progression maps one-to-one onto the zero conv's numerical behavior (0.0 -> 0.03 -> learned weights).
- The Mermaid diagram merge nodes (ADD1/ADD2/ADD3) make the additive connection visually unambiguous. The emerald color distinguishes them from the frozen (gray), trainable (violet), and zero conv (amber) components.
- The lesson component is now at a quality level suitable for shipping. The remaining issues are notebook-only.

**Systemic pattern:**
- The notebook was built in parallel with the lesson and did not receive the iteration 1 lesson fixes. This is expected — the notebook build (task #6) ran concurrently. Both improvement findings are notebook-lesson alignment issues that would not exist if the notebook had been built after the lesson was finalized. For future lessons, consider building the notebook after the lesson passes review, or include a "sync notebook with lesson" step when lesson revisions affect shared terminology or analogies.

**Iteration guidance:**
- Fix the two improvement findings (training wheels analogy in notebook, VRAM management in Exercise 4 solution).
- Fix the polish finding (spaced em dashes in notebook).
- Re-invoke review for iteration 3.

---

## Fixes Applied — 2026-02-16 (Post-Iteration 2)

### Finding #1 [IMPROVEMENT] — Notebook uses replaced "training wheels" analogy
**Status:** FIXED. Cell-13 final print statements now use the whisper metaphor: "The zero conv started silent (all zeros) and gradually learned to produce a small signal. Nothing at first, then a whisper, then a clear voice. Training gradually turns up the volume." No "training wheels" language remains in the notebook.

### Finding #2 [IMPROVEMENT] — Notebook Exercise 4 solution loads two full pipelines simultaneously
**Status:** FIXED. Added cell-23 as a dedicated VRAM cleanup cell before Exercise 4 that deletes standalone models from Exercises 1-3 (`del controlnet, unet, controlnet_device, unet_device, text_encoder; gc.collect(); torch.cuda.empty_cache()`). Restructured cell-25 solution to load sequentially: build vanilla pipeline, generate, delete vanilla pipeline, build ControlNet pipeline, generate both ControlNet images. Matches the pattern from the 6-5-1 notebook fix.

### Finding #3 [POLISH] — Notebook has three spaced em dashes in student-visible text
**Status:** FIXED. All three instances rephrased: (1) cell-7 comma, (2) cell-20 "because" rephrasing, (3) cell-20 colon.

---

## Review — 2026-02-16 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

### Iteration 2 Fix Verification

| Finding | Status | Verified |
|---------|--------|----------|
| #1 [IMPROVEMENT] Notebook training wheels analogy | FIXED | Cell-13 print statements: "The zero conv started silent (all zeros) and gradually learned to produce / a small signal. Nothing at first, then a whisper, then a clear voice. / Training gradually turns up the volume." No training wheels language present. Matches the lesson's "Nothing, Then a Whisper" InsightBlock. |
| #2 [IMPROVEMENT] Notebook Exercise 4 OOM risk | FIXED | Cell-23: dedicated VRAM cleanup cell (`del controlnet, unet, controlnet_device, unet_device, text_encoder; gc.collect(); torch.cuda.empty_cache()`). Cell-25 solution: sequential pipeline loading (vanilla pipeline -> generate -> delete -> ControlNet pipeline -> generate -> delete). Memory report in cleanup cell confirms freed VRAM. |
| #3 [POLISH] Notebook spaced em dashes | FIXED | Cell-7: `and "zero_convs," the zero convolution connections.` (comma). Cell-20 comment: `because the spatial conditioning is absent.` (rephrased). Cell-20 print: `in both cases: the difference comes entirely` (colon). No spaced em dashes remain in student-visible notebook text. |

All three iteration 2 findings are correctly resolved.

### Findings

None. Both the lesson component and the notebook pass review.

### Review Notes

**What works well across the complete lesson+notebook package:**
- The lesson's pedagogical design was strong from the start. The "of course" chain, the capstone tone, and the modality coverage (5 modalities for 2 core concepts) make this a compelling lesson for the target student.
- The notebook exercises form a clean progression from inspection (Ex1) to verification (Ex2) to integration (Ex3) to independent experimentation (Ex4). Each exercise directly tests a concept from the lesson.
- Terminology alignment between lesson and notebook is now consistent after the iteration 2 fix. The whisper metaphor, the WHEN/WHAT/WHERE framing, the `e_i + z_i` notation, and the LoRA B=0 connection all appear in both artifacts with identical language.
- The VRAM management in the notebook is now production-quality. The cleanup cell before Exercise 4 and the sequential pipeline loading in the solution ensure the notebook runs on a free-tier T4 (16 GB VRAM) without OOM.
- The notebook's predict-before-run pattern is genuinely implemented in both Guided exercises (cells 3 and 9 pose specific predictions with expected answers).
- The Supported exercise (Ex3) provides appropriate scaffolding: the API signature is hinted but not given as complete code, and the TODO markers are 1-2 lines each.
- The Independent exercise (Ex4) provides no code scaffolding, only a task description and verification criteria.

**Iteration retrospective:**
- Iteration 1 caught the most impactful issues: the missing notebook (critical), the ambiguous Mermaid merge points (improvement), and the missing explicit misconception treatment (improvement). These were structural issues.
- Iteration 2 caught notebook-lesson alignment issues that arose from parallel development. The training wheels analogy mismatch and the VRAM management problem were both real issues that would have degraded the student's experience.
- Iteration 3 confirms all fixes are clean. The lesson is ready for Phase 5 (Record).

**Recommendation:** Proceed to Phase 5 (Record) in the planning skill. Update the module record with this lesson's concept index and per-lesson summary.
