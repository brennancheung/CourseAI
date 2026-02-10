# Lesson Plan: U-Net Architecture

**Slug:** `unet-architecture`
**Module:** 6.3 (Architecture & Conditioning)
**Position:** Lesson 1 of 5 in module; Lesson 10 overall in Series 6
**Cognitive load:** BUILD (re-entering theoretical mode after CONSOLIDATE capstone)

---

## Phase 1: Orient (Student State)

### Relevant concepts the student has, with depths and sources

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Encoder-decoder architecture (hourglass: compress through bottleneck, reconstruct) | DEVELOPED | autoencoders (6.1) | Built in Colab notebook. Knows Conv2d encoder -> bottleneck -> ConvTranspose2d decoder. Understands the bottleneck forces learning what matters. |
| Skip/residual connections (identity shortcut around one or more layers) | DEVELOPED | ResNet lesson (3.2) | Knows the identity shortcut, why it helps gradient flow, and that it lets the network learn residuals. Applied in exercises. The term "skip connection" is familiar. |
| CNN feature hierarchy (edges -> textures -> parts -> objects at increasing depth) | DEVELOPED | Series 3 (Convolutions) | Knows that early layers detect simple features, deeper layers detect complex features. Connected to multi-scale denoising in the-diffusion-idea. |
| Multi-scale denoising progression (coarse-to-fine) | DEVELOPED | the-diffusion-idea + sampling-and-generation (6.2) | High noise = structural decisions, medium = refine structure, low = fine details. Experienced via DenoisingTrajectoryWidget. Connected to CNN feature hierarchy. |
| Bottleneck / latent representation | DEVELOPED | autoencoders (6.1) | Learned compression of what matters. Knows the bottleneck is the learning mechanism, not an obstacle. Explored at multiple sizes (4 to 256 dims). |
| U-Net skip connections (encoder features concatenated with decoder) | INTRODUCED | build-a-diffusion-model (6.2 capstone) | First encounter in code. "The autoencoder's bottleneck forces compression, but for denoising we want the decoder to have access to high-resolution features -- skip connections pass them through directly." Student read annotated code with 2 skip connections. Did NOT build from scratch. |
| ConvTranspose2d (learned upsampling) | INTRODUCED | autoencoders (6.1) | Knows "small spatial -> large spatial" and that it reverses Conv2d's spatial shrinking. Used in autoencoder decoder. Does not know the math. |
| Forward/reverse diffusion process | DEVELOPED/APPLIED | Module 6.2 | Implemented both. Knows the training algorithm, sampling algorithm, noise schedule, closed-form formula. The neural network was a black box. |
| Timestep embedding (network receives t as input) | INTRODUCED | learning-to-denoise + build-a-diffusion-model (6.2) | Knows the concept (one network for all timesteps, conditioned on t) and saw a minimal implementation (linear projection of normalized t). Full timestep embedding mechanism deferred. |
| "Same building blocks, different question" mental model | DEVELOPED | Module 6.1 + 6.2 | Recurring theme: conv layers, MSE loss, backprop are shared building blocks. The question (classification vs reconstruction vs noise prediction) is what changes. |

### Mental models and analogies already established

- **"Force it through a bottleneck; it learns what matters"** -- from autoencoders
- **"Sculpting from marble: rough shape -> details -> polish"** -- for iterative refinement / coarse-to-fine denoising
- **"Same building blocks, different question"** -- recurring throughout the course
- **CNN feature hierarchy as a recipe for building** -- low-level features compose into high-level structure
- **The autoencoder's latent space has gaps** -- the U-Net does not need to learn a latent space; it operates in image space

### What was explicitly NOT covered

- How the U-Net is structured internally beyond the minimal 2-skip-connection version from the capstone
- Why skip connections are essential for denoising (vs just nice-to-have)
- How the multi-resolution structure maps to the coarse-to-fine denoising progression
- Attention layers within the U-Net (self-attention, cross-attention)
- Group normalization
- Residual blocks within the U-Net (distinct from ResNet residual connections)
- Any conditioning mechanism beyond "the network receives t"

### Readiness assessment

The student is well-prepared. They have encoder-decoder at DEVELOPED, skip connections at DEVELOPED (from ResNet), multi-scale denoising at DEVELOPED, and a first encounter with U-Net skip connections at INTRODUCED (from the capstone). The lesson needs to DEVELOP the U-Net architecture -- deepening the "why" of each design choice, not teaching encoder-decoder or skip connections from scratch. The primary gap is connecting these familiar pieces into the specific U-Net configuration and explaining WHY each choice matters for denoising specifically.

---

## Phase 2: Analyze

### Target concept

This lesson teaches the student to explain why the U-Net's encoder-decoder architecture with skip connections is the natural choice for multi-scale denoising -- understanding how the downsampling path captures context at multiple resolutions, how the upsampling path reconstructs spatial detail, and why skip connections are essential (not optional) for preserving the fine-grained information that the bottleneck would otherwise destroy.

### Prerequisites table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Encoder-decoder architecture | DEVELOPED | DEVELOPED | autoencoders (6.1) | OK | Need to understand the hourglass shape and why it exists. Student built one. |
| Skip/residual connections | DEVELOPED | DEVELOPED | ResNet (3.2) | OK | Need to understand identity shortcuts and why they help. Student applied them. |
| CNN feature hierarchy | DEVELOPED | DEVELOPED | Series 3 | OK | Need to connect low-level/high-level features to U-Net's multi-resolution processing. Student knows this well. |
| Multi-scale denoising progression | DEVELOPED | DEVELOPED | the-diffusion-idea + sampling-and-generation (6.2) | OK | Core connection: coarse-to-fine maps to U-Net's multi-resolution structure. Already established. |
| Bottleneck as learned compression | INTRODUCED | DEVELOPED | autoencoders (6.1) | OK | Need to understand what the bottleneck preserves and what it loses. Student explored this with the bottleneck size slider. |
| ConvTranspose2d | INTRODUCED | INTRODUCED | autoencoders (6.1) | OK | Need to know upsampling exists. Student knows "small spatial -> large spatial." Sufficient for this lesson. |
| U-Net skip connections in code | INTRODUCED | INTRODUCED | build-a-diffusion-model (6.2) | OK | Student saw 2 skip connections in annotated code. This lesson develops the concept. Existing INTRODUCED depth is the expected starting point. |
| Diffusion forward/reverse process | DEVELOPED | DEVELOPED/APPLIED | Module 6.2 | OK | Need to understand what the denoising network does (predict noise at various levels). Student implemented both processes. |

### Gap resolution

No gaps. All prerequisites are at or above required depth. The lesson builds on solid foundations.

### Misconceptions table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The U-Net is just an autoencoder used for denoising" | The student built autoencoders in 6.1 and the U-Net looks like an autoencoder (encoder-decoder shape). The hourglass shape is visually identical. | Without skip connections, the decoder must reconstruct fine details from the compressed bottleneck alone. At 32x32 -> 8x8 -> 32x32, fine edge positions and textures are lost through the bottleneck. The autoencoder's blurry reconstructions (which the student saw) demonstrate this -- and denoising requires pixel-precise output. An autoencoder-without-skips produces blurry denoised images; the U-Net with skips preserves sharpness. | Section 6 (Elaborate) -- after skip connections are explained, show what happens without them. |
| "Skip connections are just a nice optimization trick, not architecturally essential" | The student learned skip connections in ResNet as a training aid (helping gradients flow, enabling deeper networks). They may think U-Net skips serve the same purpose -- helpful but not critical. | In ResNet, removing skip connections makes training harder but the architecture can still function. In the U-Net, removing skip connections fundamentally changes what the decoder can produce -- it loses access to high-resolution features entirely and must hallucinate fine details from the bottleneck. This is not a training difficulty; it is an information loss. | Section 5 (core explanation) -- motivate skip connections via the information loss problem before showing them as the solution. |
| "The bottleneck is unnecessary -- why not keep full resolution throughout?" | If skip connections bypass the bottleneck, why have a bottleneck at all? The student might think the whole downsampling path is unnecessary overhead. | Without downsampling, the network only sees local pixel neighborhoods (limited receptive field). At high noise levels, the network needs to make global structural decisions (is this a face or a landscape?). The bottleneck/low-resolution layers give the network a global view. You need BOTH: global context from the bottleneck AND local details from the skip connections. | Section 6 (Elaborate) -- after both components are explained, address why you need both. |
| "Each resolution level in the U-Net handles one specific noise level" | The student knows coarse-to-fine denoising progression and might map it too literally: "the lowest resolution handles high noise, the highest resolution handles low noise." | The U-Net processes ALL resolution levels for EVERY timestep. At t=900, the bottleneck layers do the heavy lifting (global structure), but the high-resolution layers still run. At t=50, the high-resolution layers do the fine work, but the bottleneck still provides context. It is not a routing mechanism; it is parallel multi-scale processing. | Section 5 (core explanation) -- address when explaining the multi-resolution connection. |

### Examples planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Autoencoder-without-skips denoising attempt: encode noisy image through bottleneck, decode. Compare output sharpness to U-Net-with-skips output. | Negative | Shows that the autoencoder shape alone is insufficient -- the bottleneck destroys fine details that denoising needs to preserve. | The student already knows autoencoders produce blurry reconstructions (from 6.1). Extending this to denoising makes the skip connection necessity visceral rather than theoretical. |
| U-Net processing a heavily noised image (t=900): trace what each resolution level contributes. Low-res captures "this is roughly a shoe shape," mid-res captures "the sole is here, the opening is there," high-res captures edge positions and textures. | Positive | Shows how multi-resolution processing maps to the coarse-to-fine denoising the student already understands. Makes the abstract architecture concrete. | Connects directly to the DenoisingTrajectoryWidget experience from sampling-and-generation. The student watched denoising progress from structure to detail -- now they see WHERE in the architecture each level of detail is handled. |
| U-Net processing a lightly noised image (t=50): trace the same path. Now the bottleneck barely changes anything (structure is already correct), but the high-resolution skip connections carry the fine detail corrections. | Positive | Shows the same architecture adapting its behavior to different noise levels -- not through routing, but through what information matters at each scale. | Contrasts with the t=900 example to demonstrate that the architecture handles both extremes naturally. Prevents misconception #4 (one resolution per noise level). |
| Simple diagram: 64x64 -> 32x32 -> 16x16 -> 32x32 -> 64x64 with and without skip connections. Show channel counts and what information flows where. | Positive (structural) | Provides a concrete spatial/visual representation of the architecture with specific tensor dimensions. | The student learned encoder-decoder via dimension walkthroughs in 6.1. Extending that pattern to the U-Net maintains consistency and makes the architecture tangible. |

---

## Phase 3: Design

### Narrative arc

The student just built a working pixel-space diffusion model. The neural network was a black box that took in a noisy image and a timestep and predicted noise. The student saw it work -- the denoised images emerged from static. But the network's architecture was provided, not understood. Why is it shaped like an hourglass? Why does it have those skip connections? Why not just use a stack of conv layers?

The answer connects two things the student already knows deeply but has not yet linked. From autoencoders (6.1): the encoder-decoder shape captures information at multiple scales of abstraction, with a bottleneck that forces the network to learn what matters. From the diffusion trajectory (6.2): denoising naturally works coarse-to-fine -- early steps decide global structure, late steps refine fine details. The U-Net is the architecture that maps one to the other. Its multi-resolution processing is not an arbitrary design choice; it is the spatial analog of the temporal coarse-to-fine progression the student already observed. And skip connections are the key difference between "autoencoder for denoising" (which would produce blurry results) and "U-Net for denoising" (which preserves sharpness). The student needs to feel why skip connections are essential, not just nice-to-have -- they carry the fine-grained information that the bottleneck cannot preserve.

### Modalities planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual/Spatial | Architecture diagram: the U-Net as a symmetric encoder-decoder with skip connection arrows, annotated with tensor dimensions at each level (e.g., 64x64x64 -> 32x32x128 -> 16x16x256 -> 32x32x128 -> 64x64x64). Mermaid or SVG diagram. | The U-Net is an inherently spatial/structural concept. A dimension walkthrough made the autoencoder click in 6.1 -- the same pattern will work here. The skip connection arrows make the information flow visible. |
| Verbal/Analogy | "The autoencoder compresses everything through a keyhole. The U-Net opens side doors." The bottleneck (keyhole) captures global structure; the skip connections (side doors) let fine details bypass the bottleneck and reach the decoder directly. | Extends the bottleneck analogy from 6.1 ("describe a shoe in 32 words"). The keyhole captures the gist; the side doors carry the details that were too fine-grained for the gist. |
| Concrete example | Trace a 64x64 noisy image through the U-Net at two different noise levels (t=900 and t=50). At t=900, the bottleneck features dominate (global structure decisions). At t=50, the skip connection features dominate (fine detail corrections). Show what each resolution level contributes with specific descriptions. | Makes the abstract architecture concrete by connecting it to the denoising progression the student experienced in the DenoisingTrajectoryWidget. |
| Symbolic/Code | PyTorch-style pseudocode showing the forward pass: encoder blocks, bottleneck, decoder blocks with skip connection concatenation. Channel dimension doubling after concatenation. | The student has seen real U-Net code in the capstone. Pseudocode at a slightly higher level of abstraction lets them see the pattern without getting lost in implementation details. |
| Intuitive | "Why this shape makes sense" synthesis: the bottleneck gives global context (needed for high-noise structural decisions), the skip connections preserve local detail (needed for low-noise refinement), and the multi-resolution structure processes both simultaneously for every timestep. It is not that different resolutions handle different noise levels -- it is that the network always has access to both global context and local detail, and the noise level determines which matters more. | Addresses misconception #4 and crystallizes the core insight into an "of course" moment. |

### Cognitive load assessment

- **New concepts in this lesson:** 2 genuinely new
  1. U-Net skip connections as essential for denoising (DEVELOPED -- elevated from INTRODUCED in the capstone; the "why" is new)
  2. Multi-resolution processing mapped to coarse-to-fine denoising (new connection between existing concepts)
- **Supporting concepts reviewed/connected:** encoder-decoder architecture, CNN feature hierarchy, residual blocks, bottleneck compression, ConvTranspose2d
- **Previous lesson load:** CONSOLIDATE (build-a-diffusion-model capstone -- zero new theory)
- **Assessment:** BUILD is appropriate. The student is re-entering theoretical mode after pure implementation, but the theoretical content builds on well-established foundations (encoder-decoder, skip connections, multi-scale features). Two new concepts is within the 2-3 limit. The lesson connects familiar pieces in a new way rather than introducing entirely unfamiliar material.

### Connections to prior concepts

- **Encoder-decoder from autoencoders (6.1):** Direct extension. "Remember the autoencoder's hourglass? The U-Net starts there but adds crucial bypass routes." The encoder path IS a CNN encoder. The decoder path uses ConvTranspose2d just like the autoencoder decoder.
- **Skip connections from ResNet (3.2):** Same principle, different purpose. In ResNet, skip connections help gradient flow in deep networks. In the U-Net, skip connections carry high-resolution spatial information from encoder to decoder. Same mechanism, different "why."
- **CNN feature hierarchy (Series 3):** The U-Net's downsampling path IS a CNN feature hierarchy. Early layers: edges and textures. Deeper layers: shapes and structures. Bottleneck: global composition. This is the same hierarchy the student explored in "Seeing What CNNs See" (3.3).
- **Multi-scale denoising (6.2):** The DenoisingTrajectoryWidget showed denoising progressing from global structure to fine details. The U-Net's multi-resolution processing is the architectural mechanism that enables this. The connection should feel like "of course -- the architecture mirrors the process."
- **Bottleneck compression (6.1):** The autoencoder's bottleneck forces lossy compression. The student saw this as blurry reconstructions. The U-Net's skip connections solve this specific problem for denoising -- they carry the details that the bottleneck loses.

**Potentially misleading prior analogies:**
- The "describe a shoe in 32 words" analogy from autoencoders could mislead: the U-Net does NOT force all information through the bottleneck. The skip connections are like being allowed to also send photos alongside your 32-word description. Need to explicitly correct this.
- The ResNet skip connection analogy could mislead: ResNet skips are primarily about gradient flow; U-Net skips are primarily about information flow. Same mechanism, different primary purpose. Need to distinguish these explicitly.

### Scope boundaries

**This lesson IS about:**
- The U-Net's encoder-decoder architecture with skip connections at DEVELOPED depth
- WHY this architecture suits multi-scale denoising (the connection to coarse-to-fine)
- What information flows through the skip connections vs through the bottleneck
- Tensor dimensions at each resolution level (concrete, not abstract)
- Residual blocks within the U-Net (briefly, at INTRODUCED depth -- they are the building blocks of each level)

**This lesson is NOT about:**
- Timestep conditioning (how the network receives t) -- that is Lesson 11
- Self-attention or cross-attention layers within the U-Net -- introduced briefly as "these exist" but not developed until Lessons 12-13
- Group normalization details (MENTIONED as a component, not explained)
- Text conditioning or any form of guided generation
- Implementation (no Colab notebook for this lesson -- the capstone already had implementation)
- The original medical U-Net from Ronneberger et al. (MENTIONED for historical context, not developed)
- 3D convolutions or video U-Nets

**Target depth:** U-Net architecture at DEVELOPED. The student should be able to draw the architecture, explain why each component exists, and trace information flow through it. Not yet at APPLIED (no implementation exercise in this lesson; implementation comes in later lessons and Module 6.4).

### Lesson outline

1. **Context + Constraints**
   - "In Module 6.2, the neural network was a black box. You fed it a noisy image and a timestep, and it predicted noise. Today we open that box."
   - Scope: the spatial architecture only. Timestep conditioning is next lesson. Text conditioning is two lessons out.
   - We are DEVELOPING the U-Net -- deepening from the minimal version in the capstone to the real architecture.

2. **Recap** (brief, since prerequisites are solid)
   - Callback to the capstone: "You used a minimal U-Net with 2 skip connections and a simple timestep embedding. It generated recognizable MNIST digits. Now let us understand WHY that architecture was shaped like that."
   - Callback to the autoencoder: "In Module 6.1, you built an encoder-decoder that compressed images through a bottleneck. The reconstructions were blurry. Hold that thought."
   - No extended recap needed -- both concepts are at DEVELOPED depth and were used recently.

3. **Hook** (type: puzzle/before-after)
   - "Why not just use a stack of conv layers?" Start with the simplest possible denoising architecture: a series of conv layers, same resolution throughout. It works for very light noise but fails catastrophically for heavy noise. Why?
   - The answer: at heavy noise (t=900), the network needs to make global structural decisions (face vs landscape). A stack of same-resolution conv layers has a limited receptive field -- it can only see local neighborhoods. It cannot reason about global structure.
   - "You need two things simultaneously: a way to see the big picture (global context) AND a way to preserve fine details (local precision). The U-Net gives you both."

4. **Explain -- The Encoder Path (Downsampling)**
   - Start with the encoder as a familiar CNN: Conv -> ReLU -> Conv -> ReLU -> Downsample. Each level doubles the channels and halves the spatial resolution.
   - Dimension walkthrough: 64x64x3 -> 64x64x64 -> 32x32x128 -> 16x16x256 -> 8x8x512. (Specific dimensions for concreteness; actual SD U-Net uses different sizes but the pattern is identical.)
   - Connect to CNN feature hierarchy: "This IS the feature hierarchy from Series 3. Early layers capture edges and textures. Deeper layers capture shapes and structure. The bottleneck captures global composition."
   - Key insight: as spatial resolution shrinks, each pixel in the feature map represents a larger region of the original image. The 8x8 feature map has a global receptive field -- each "pixel" sees the entire input.

5. **Explain -- The Bottleneck**
   - The lowest resolution: maximum channels, minimum spatial size. Global context.
   - "At t=900, this is where the important decisions happen. The image is mostly noise, but the bottleneck features capture whatever faint global structure remains."
   - Connect to autoencoder bottleneck: "Same principle -- information compression. But unlike the autoencoder, the U-Net does NOT rely solely on the bottleneck to reconstruct the output."

6. **Explain -- The Decoder Path (Upsampling) and Skip Connections**
   - The decoder mirrors the encoder: Upsample -> Conv -> ReLU -> Conv -> ReLU. Each level halves the channels and doubles the spatial resolution.
   - WITHOUT skip connections (negative example): "This is just an autoencoder. Remember the blurry reconstructions from Module 6.1? The decoder must reconstruct all spatial detail from the bottleneck alone. For denoising, this means the fine details -- edge positions, textures, the precise boundary between a shoe and its background -- are lost."
   - WITH skip connections (the solution): "Each decoder level receives features from the corresponding encoder level via concatenation. The encoder's 32x32x128 features are concatenated with the decoder's 32x32x128 features to produce 32x32x256 features. The decoder has access to BOTH the global context (from the upsampled lower levels) and the local detail (from the skip connections)."
   - The "keyhole and side doors" analogy: "The bottleneck is a keyhole -- global structure passes through. The skip connections are side doors -- fine details bypass the keyhole entirely."
   - Address misconception #1 explicitly: "This is why the U-Net is NOT just an autoencoder. The skip connections fundamentally change what information the decoder can access."

7. **Explain -- Why Both: The Two-Path Information Flow**
   - "Why not skip the bottleneck entirely and just pass everything through skip connections?"
   - Address misconception #3: without the downsampling path, the network has no global context. At t=900, it would try to refine pixel-level details in what is essentially pure static. It needs to first decide "this is probably a shoe" (bottleneck) before it can refine "the toe is here" (skip connections).
   - Synthesis: the bottleneck provides WHAT (global structure), the skip connections provide WHERE (precise spatial details). You need both.

8. **Explain -- Multi-Resolution Maps to Coarse-to-Fine Denoising**
   - This is the "of course" moment. The U-Net's architecture mirrors the denoising task:
     - At high noise: bottleneck layers dominate. They are making structural decisions from almost nothing. The skip connections carry mostly noise.
     - At low noise: skip connection features dominate. The structure is already correct (from earlier denoising steps). The high-resolution features carry the precise detail corrections.
   - "The architecture does not route different noise levels to different resolution layers. Every timestep processes through all resolution levels. But the IMPORTANCE of each level shifts with the noise level."
   - Address misconception #4 explicitly.
   - Connect to DenoisingTrajectoryWidget: "When you watched denoising in the previous lesson, you saw structure emerge first and details appear later. The U-Net's multi-resolution processing is WHY."
   - Trace the t=900 example and t=50 example (from Phase 2 examples).

9. **Explain -- Residual Blocks (briefly)**
   - At INTRODUCED depth: each "level" of the U-Net is not a single conv layer but a residual block (or multiple). Conv -> Norm -> Activation -> Conv -> Norm -> Activation + skip.
   - Connect to ResNet: "These are the same residual blocks from Series 3.2, with group normalization instead of batch normalization."
   - Group normalization: MENTIONED only. "Group norm works better than batch norm for small batch sizes common in diffusion training. You do not need the details."
   - This section is brief -- the important concept is that each level has internal skip connections (residual blocks) in addition to the long-range skip connections between encoder and decoder.

10. **Check -- Predict and verify**
    - "Your colleague proposes removing all skip connections to reduce memory usage. What happens to the denoised output at t=50 vs t=900?"
    - At t=900: barely affected -- the bottleneck is doing most of the work anyway.
    - At t=50: catastrophically worse -- fine details are lost. The output becomes blurry, like autoencoder reconstructions.
    - This tests whether the student understands the differential importance of skip connections at different noise levels.

11. **Check -- Transfer question**
    - "In the original 2015 paper, U-Net was designed for medical image segmentation (labeling each pixel of a brain scan). Why would the same architecture be useful for a completely different task (denoising)?"
    - Both tasks require pixel-precise output that combines global understanding with local detail. Segmentation needs to know "this region is a tumor" (global) AND "the boundary of the tumor is exactly here" (local). Denoising needs "this is a shoe" (global) AND "the edge of the shoe is exactly here" (local). Same architectural need, different task.

12. **Explore -- Architecture Visualization**
    - Full U-Net architecture diagram with dimensions at each level. Skip connection arrows clearly visible. Channel counts annotated.
    - The student traces information flow through the diagram. "Follow a pixel from input to bottleneck to output. What path does it take? What information does it pick up along the way?"
    - Brief mention: "In real diffusion U-Nets, attention layers are interleaved at certain resolution levels. You will see those in Lessons 12-13." MENTIONED, not developed.

13. **Summarize**
    - Key takeaways:
      1. The U-Net is an encoder-decoder with skip connections -- not just an autoencoder. The skip connections carry fine-grained spatial details that the bottleneck would destroy.
      2. The multi-resolution structure maps to the coarse-to-fine denoising progression. Low-resolution layers provide global context; high-resolution layers preserve local detail. Both are needed at every timestep.
      3. Skip connections are essential, not optional. Without them, the decoder cannot produce pixel-precise output -- it would produce the same blurry results as an autoencoder.
    - Mental model: "The bottleneck decides WHAT. The skip connections decide WHERE. The U-Net gives you both at every resolution."

14. **Next step**
    - "You now understand the spatial architecture. But the network needs to know one more thing: what noise level is it denoising? The same U-Net handles t=50 (barely noisy) and t=900 (pure static), but it needs to behave very differently for each. Next, you will see how timestep embeddings give the network that awareness."

### Exercise design (no standalone Colab for this lesson)

This is a conceptual/theoretical lesson with no separate notebook. The student's implementation experience was the capstone in Module 6.2. The next implementation opportunity comes when they build a fuller U-Net in later lessons. The checks within the lesson (predict-and-verify, transfer question) serve as the active learning components.

Widget consideration: A static architecture diagram (Mermaid or SVG) showing the full U-Net with dimensions, skip connection arrows, and annotations at each level. No interactive widget is strictly necessary -- the concept is structural/spatial rather than dynamic. If a widget is built, it could show information flow through the U-Net at different noise levels (highlighting which paths carry the most important information), but this is optional.

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 planned: visual/spatial, verbal/analogy, concrete example, symbolic/code, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (4 identified)
- [x] Cognitive load at most 3 new concepts (2 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-10 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 2

### Verdict: MAJOR REVISION

One critical finding (missing autoencoder recap that the plan explicitly calls for) plus four improvement findings. The lesson is well-structured and the narrative arc is strong, but there are gaps between what was planned and what was built that would leave the student weaker than intended.

### Findings

#### [CRITICAL] — Missing autoencoder recap callback in context section

**Location:** Section 2 (Context + Constraints) and Section 3 (Hook)
**Issue:** The planning document's outline item 2 (Recap) explicitly specifies: "Callback to the autoencoder: 'In Module 6.1, you built an encoder-decoder that compressed images through a bottleneck. The reconstructions were blurry. Hold that thought.'" This callback is essential because the entire skip connection argument later depends on the student remembering that autoencoders produce blurry reconstructions. The built lesson does not include this recap. The "What You Already Know" TipBlock in Section 2 only mentions the capstone U-Net, not the autoencoder blurriness.
**Student impact:** When the lesson later says "Remember the blurry reconstructions from Autoencoders?" (line 298-299), the student has not been primed. The autoencoder lesson was at least 4 lessons ago (6.1 -> five 6.2 lessons -> here). Without a recap, the "blurry reconstructions" reference may not land with the emotional weight needed to make skip connections feel essential. The Reinforcement Rule says concepts introduced more than 3 lessons ago need reinforcement; the blurriness of autoencoder reconstructions is the central negative example for this lesson.
**Suggested fix:** Add an explicit autoencoder recap in the "What You Already Know" TipBlock or as a separate brief paragraph in Section 2. Something like: "In Autoencoders (Module 6.1), you built an encoder-decoder that compressed images through a bottleneck. The reconstructions were blurry -- fine details were lost through the bottleneck. Hold that thought." This primes the student for the skip connection argument.

#### [IMPROVEMENT] — Architecture diagram placed after the explanation instead of integrated with it

**Location:** Section 7 (The Full Architecture) vs the planning document's outline item 12 (Explore)
**Issue:** The planning document places the full architecture diagram in section 12 (Explore), after all the conceptual explanations. The built lesson places it in Section 7, which is between the decoder/skip connection explanation (Section 6) and the "Why You Need Both Paths" section (Section 8). This is actually a reasonable deviation -- the student benefits from seeing the full picture after encoder + bottleneck + decoder are explained. However, the planning document also specifies the diagram should come with an activity: "The student traces information flow through the diagram. 'Follow a pixel from input to bottleneck to output. What path does it take? What information does it pick up along the way?'" This tracing activity is absent. The diagram is presented passively with a "Reading the Diagram" aside but no active engagement prompt.
**Student impact:** The student sees the diagram but is not asked to actively engage with it. Active tracing (even mentally) builds stronger spatial understanding than passive observation. The diagram placement is fine; the missing active prompt weakens it.
**Suggested fix:** Add a brief prompt below the diagram asking the student to trace a path. For example: "Trace the path a pixel takes: through all three encoder levels, through the bottleneck, back up through all three decoder levels. At each decoder level, it picks up features from the matching encoder level. Count: how many times do the encoder features influence the output?" This could be a TryThisBlock or a simple italicized prompt.

#### [IMPROVEMENT] — Misconception #2 (skip connections are just a training trick) not explicitly addressed

**Location:** Throughout the lesson, but especially Section 6 and Section 8
**Issue:** The planning document identifies Misconception #2: "Skip connections are just a nice optimization trick, not architecturally essential." The planned negative example contrasts ResNet skips (training aid) with U-Net skips (information pathway). The built lesson does distinguish U-Net skips from autoencoders (no-skips = blurry, with-skips = sharp), but it never addresses the ResNet analogy directly. The student learned skip connections in the ResNet context (Series 3.2) as a training aid for gradient flow. The lesson never says "In ResNet, skip connections helped gradients flow. Here, skip connections serve a fundamentally different purpose -- they carry spatial information." The "Two Kinds of Skip Connections" ConceptBlock in Section 11 (line 661-667) mentions short-range vs long-range skip connections but frames both as "same mechanism, different scales" rather than "same mechanism, different primary purpose."
**Student impact:** The student might still think of U-Net skip connections as primarily a training aid (helping gradients flow through the deep encoder-decoder) rather than as an information pathway (carrying high-resolution features that the bottleneck destroys). This misconception would not cause the student to be lost, but it would leave them with a weaker mental model of why skip connections are essential.
**Suggested fix:** In Section 6 or the "Two Kinds of Skip Connections" ConceptBlock, explicitly contrast the two purposes: "In ResNet, skip connections primarily help gradients flow during training. In the U-Net, skip connections primarily carry high-resolution spatial information from encoder to decoder. Same mechanism, fundamentally different purpose."

#### [IMPROVEMENT] — Ordering of Sections 7 and 8 could cause premature question

**Location:** Section 7 (Architecture Diagram) and Section 8 (Why You Need Both Paths)
**Issue:** The full architecture diagram (Section 7) prominently shows skip connections bypassing the bottleneck. A curious student seeing this would immediately wonder: "If the skip connections bypass the bottleneck, why bother with the bottleneck at all?" This is exactly Misconception #3 from the planning document. But the lesson does not address this question until Section 8, after the diagram. In the interim, the student sits with the misconception. The planning document places the "Why Both" explanation (outline item 7) before the architecture diagram (outline item 12), which would prevent this issue.
**Student impact:** The student sees the diagram and forms a question (or misconception) that is not immediately addressed. One section of unresolved confusion is not catastrophic, but it creates unnecessary cognitive tension. The student may start to question the bottleneck's value, which could undermine their understanding of the encoder path explanation they just read.
**Suggested fix:** Either (a) move the "Why You Need Both Paths" section before the full architecture diagram, so the student understands the dual-path necessity before seeing it visualized, or (b) add a brief one-sentence forward reference after the diagram: "You might wonder: if skip connections bypass the bottleneck, why have a bottleneck at all? That is exactly the right question -- the next section answers it."

#### [IMPROVEMENT] — "Series 3" reference is vague

**Location:** Section 4 (The Encoder Path), line 167
**Issue:** The lesson says "it is the same feature hierarchy from Series 3." The student has no way to quickly recall what "Series 3" contained -- it was many lessons ago and the reference is to a series, not a specific lesson. The planning document specifies the connection should be to "Seeing What CNNs See (3.3)" and the CNN feature hierarchy concept from "Series 3 (Convolutions)." A more specific reference would help the student recall the right mental model.
**Student impact:** The student reads "Series 3" and may not immediately recall the feature hierarchy. The aside partially compensates ("The encoder path IS a CNN feature hierarchy. Early layers capture edges and textures..."), but the main text reference is still vague.
**Suggested fix:** Change "from Series 3" to "from the convolution lessons" or "from Seeing What CNNs See" to give the student a more specific anchor.

#### [POLISH] — Spaced em dashes in dimension level labels

**Location:** Lines 191, 201, 211, 221 (Level 0/1/2/Bottleneck labels)
**Issue:** The dimension labels use spaced em dashes: "Level 0 &mdash; 64x64x64". The writing style rule specifies em dashes should have no spaces. These are labels rather than prose, so the impact is minimal, but consistency with the style rule matters.
**Student impact:** None -- purely a style consistency issue.
**Suggested fix:** Change to "Level 0&mdash;64x64x64" or use a different separator (colon, pipe) for labels if the unspaced em dash looks too cramped.

#### [POLISH] — Attention mention paragraph feels abrupt

**Location:** Section 14 (Brief mention of attention), lines 750-761
**Issue:** The paragraph about attention layers appears between the transfer question check and the summary. There is no transition -- the lesson jumps from a check question about medical segmentation directly into "One more piece you will encounter soon: attention layers." This feels like a tagged-on afterthought.
**Student impact:** Minor. The student gets the information, but the flow sags slightly at this point. The lesson's energy peaks at the checks (active engagement) and then drops to a passive mention before the summary.
**Suggested fix:** Either move the attention mention to after the residual blocks section (Section 11), which would create a natural "here are two more components inside the U-Net" grouping, or add a brief transition: "Before we summarize, one preview of what is coming next..."

### Review Notes

**What works well:**
- The hook (Section 3) is strong. Starting with "why not just conv layers?" creates genuine motivation before introducing the U-Net. The student feels the need for multi-scale processing before being shown the solution.
- The autoencoder comparison (ComparisonRow in Section 6) is one of the best elements. It concretely shows what skip connections add by contrasting with something the student already experienced (blurry autoencoder reconstructions).
- The "Predict and Verify" check (Section 12) is excellent pedagogy. It tests the differential importance of skip connections at different noise levels, which is the lesson's core insight.
- The pseudocode (Section 10) is at the right level of abstraction -- high enough to see the pattern, low enough to be concrete. The `cat(up(b), e2)` pattern makes skip connections visible in code.
- The cognitive load is well-managed. Two genuinely new concepts (skip connections as essential, multi-resolution mapping to coarse-to-fine), with everything else connecting to familiar material.
- Scope boundaries are respected. The lesson does not drift into timestep conditioning, attention mechanisms, or implementation.

**Modality check (3 required):**
1. Visual/Spatial: Mermaid architecture diagram with dimensions -- present
2. Verbal/Analogy: "Keyhole and side doors" -- present (InsightBlock, line 338)
3. Concrete example: t=900 vs t=50 walkthrough showing what each resolution contributes -- present (Section 9)
4. Symbolic/Code: Pseudocode forward pass -- present (Section 10)
5. Intuitive: "The architecture mirrors the denoising task" synthesis -- present (Section 9)

All 5 planned modalities are present. Requirement met.

**Example check (2 positive + 1 negative required):**
- Positive 1: t=900 heavy noise walkthrough (Section 9) -- present
- Positive 2: t=50 light noise walkthrough (Section 9) -- present
- Positive 3: Dimension walkthrough through encoder levels (Section 4) -- present
- Negative 1: Autoencoder without skips (Section 6, ComparisonRow) -- present

Requirement met.

**Misconception check (3 required):**
- Misconception #1 (U-Net is just an autoencoder): Addressed in Section 6 ComparisonRow and WarningBlock -- present
- Misconception #2 (Skip connections are just a training trick): NOT explicitly addressed -- see IMPROVEMENT finding above
- Misconception #3 (Bottleneck is unnecessary): Addressed in Section 8 WarningBlock -- present
- Misconception #4 (Each resolution handles one noise level): Addressed in Section 9 WarningBlock ("Not a Routing Mechanism") -- present

3 of 4 planned misconceptions explicitly addressed. Misconception #2 is implicitly addressed (the lesson shows skip connections carry information, not just help training) but not explicitly contrasted with the ResNet purpose.

**Row component check:** All content uses Row compound components. No manual flex layouts found. Requirement met.

**Notebook check:** No notebook exists. Planning document explicitly states "no standalone Colab for this lesson." No finding needed.

**Interaction design check:** The only interactive elements are `<details>` reveal toggles in the check questions. The `<summary>` elements use `cursor-pointer` class. Requirement met.

---

## Review — 2026-02-10 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All seven findings from iteration 1 have been properly addressed. The autoencoder recap is now present and well-positioned. The architecture diagram has an active tracing prompt. Misconception #2 (ResNet vs U-Net skips) is explicitly contrasted. The forward reference after the diagram prevents premature confusion about the bottleneck. "Series 3" was replaced with "the convolution lessons." Spaced em dashes were fixed. The attention mention now has a smooth transition. Two minor polish items remain but neither affects the student's learning experience.

### Iteration 1 Fix Verification

| Iteration 1 Finding | Status | How It Was Fixed |
|---------------------|--------|------------------|
| CRITICAL: Missing autoencoder recap | FIXED | "What You Already Know" TipBlock now includes: "In Autoencoders, you built an encoder-decoder that compressed images through a bottleneck. The reconstructions were blurry -- fine details were lost through the bottleneck. Hold that thought." Primes the student for the skip connection argument. |
| IMPROVEMENT: Architecture diagram lacks active tracing prompt | FIXED | TryThisBlock added in the aside next to the diagram: "Follow a single pixel through the U-Net... Count: how many times do the encoder features influence the final output?" Active engagement, not passive observation. |
| IMPROVEMENT: Misconception #2 (ResNet vs U-Net skips) not addressed | FIXED | "Two Kinds of Skip Connections" ConceptBlock in Section 11 now explicitly contrasts: "In ResNet, skip connections primarily help gradients flow during training. In the U-Net, the long-range skip connections primarily carry high-resolution spatial information from encoder to decoder. Same mechanism, fundamentally different purpose." |
| IMPROVEMENT: Forward reference needed after diagram | FIXED | One-sentence forward reference added after the diagram: "You might wonder: if skip connections bypass the bottleneck, why have a bottleneck at all? That is exactly the right question -- the next section answers it." |
| IMPROVEMENT: "Series 3" reference too vague | FIXED | Changed to "the convolution lessons" -- more specific anchor for the student. |
| POLISH: Spaced em dashes in level labels | FIXED | Labels now use unspaced em dashes: "Level 0&mdash;64x64x64". |
| POLISH: Attention mention feels abrupt | FIXED | Transition added: "Before we summarize, one preview of what is coming next." Smooth flow from checks to attention preview to summary. |

### Findings

#### [POLISH] — Mermaid diagram does not visually resemble the letter U

**Location:** Section 7 (The Full Architecture), Mermaid diagram
**Issue:** The lesson text says "it looks like the letter U, which is where the name comes from." However, the Mermaid diagram uses `graph TD` (top-down layout), rendering as a vertical pipeline from top to bottom with horizontal dashed arrows for skip connections. It does not visually resemble the letter U -- it looks like a linear top-to-bottom flow with side branches. The traditional U-Net diagram has the encoder going down-left, the bottleneck at the bottom, and the decoder going up-right, forming the U shape.
**Student impact:** Minimal. The student reads "looks like the letter U," looks at the diagram, and does not see a U shape. This is a momentary "huh" that resolves quickly since the information flow is still clear. The diagram communicates encoder-bottleneck-decoder structure and skip connections effectively regardless of orientation.
**Suggested fix:** Either (a) remove the "looks like the letter U" claim and just say "this is the U-Net architecture," since the diagram communicates the structure without needing the visual metaphor, or (b) if Mermaid supports it, adjust the layout to better suggest the U shape (e.g., using subgraph positioning to place encoder left, bottleneck bottom, decoder right). Option (a) is simpler and sufficient.

#### [POLISH] — Pseudocode omits timestep parameter without comment

**Location:** Section 10 (The Forward Pass in Code), pseudocode
**Issue:** The pseudocode signature is `def forward(self, x)`, taking only the noisy image. The student's most recent encounter with a U-Net (the capstone) used `forward(self, x, t)` which takes both the image and the timestep. The lesson's scope boundary correctly excludes timestep conditioning ("NOT: timestep conditioning -- how the network receives t is the next lesson"), but the pseudocode does not comment on why `t` is missing from the signature. The student might briefly wonder "where did the timestep go?"
**Student impact:** Minor. The scope boundary is clear at the top of the lesson, and the "Next Step" section explicitly bridges to timestep conditioning. A student reading carefully would understand the omission. But a student scanning the code quickly might have a moment of confusion.
**Suggested fix:** Add a brief inline comment in the pseudocode, e.g., `def forward(self, x):  # timestep t omitted -- next lesson` or mention the omission in the paragraph preceding the code block. This is a one-line fix that prevents any momentary confusion.

### Review Notes

**What works well (confirming iteration 1 strengths still hold):**
- The hook remains one of the strongest elements. "Why Not Just Conv Layers?" with the receptive field argument creates genuine motivation.
- The autoencoder comparison is now even stronger with the explicit recap priming the student. The callback to blurry reconstructions lands with full weight.
- The "Predict and Verify" check is excellent -- tests the lesson's core insight (differential importance of skip connections at different noise levels).
- The pseudocode is at the right abstraction level. The `cat(up(b), e2)` pattern makes skip connections visible.
- Cognitive load is well-managed at 2 new concepts within the 2-3 limit.
- Scope boundaries are respected throughout.
- The "Two Kinds of Skip Connections" ConceptBlock now effectively addresses all 4 planned misconceptions. The lesson hits 4/4 instead of the 3/4 from iteration 1.
- The forward reference after the diagram is a particularly elegant fix -- it acknowledges the student's natural question and channels it into the next section rather than leaving it unresolved.

**Modality check (3 required):** All 5 planned modalities present. No change from iteration 1. Requirement met.

**Example check (2 positive + 1 negative required):** 3 positive + 1 negative. No change from iteration 1. Requirement met.

**Misconception check (3 required):** All 4 planned misconceptions now explicitly addressed (up from 3/4 in iteration 1). Requirement exceeded.

**Row component check:** All content uses Row compound components. No manual flex layouts. Requirement met.

**Notebook check:** No notebook exists. Planning document specifies none needed. No finding.

**Interaction design check:** `<details>` reveals with `cursor-pointer` on summaries. Requirement met.

**Writing style check:** No spaced em dashes in rendered content. JSX comments use spaced em dashes but those are not rendered to the student. Requirement met.
