# conditioning-the-unet -- Lesson Planning Document

**Module:** 6.3 (Architecture & Conditioning)
**Position:** Lesson 2 of 5
**Type:** BUILD (re-entering theoretical mode with heavily leveraged prior knowledge)
**Previous lesson:** unet-architecture (BUILD)
**Next lesson:** clip (STRETCH)

---

## Phase 1: Student State

### Relevant concepts the student has

| Concept | Depth | Source Lesson | Notes |
|---------|-------|--------------|-------|
| Sinusoidal positional encoding (sin/cos at exponentially increasing wavelengths, formula, four requirements, clock analogy) | DEVELOPED | embeddings-and-position (4.1.3) | Student implemented from formula in notebook. Understands uniqueness, smoothness, arbitrary-length, deterministic. "Clock with many hands" mental model. This is the direct conceptual bridge to timestep embeddings. |
| U-Net encoder-decoder architecture (encoder downsampling, bottleneck, decoder upsampling, skip connections) | DEVELOPED | unet-architecture (6.3.1) | Full dimension walkthrough, Mermaid diagram, pseudocode. Pseudocode explicitly omitted the timestep parameter with a comment noting it would be covered "next lesson." Student has the complete spatial architecture but no conditioning mechanism. |
| Multi-resolution processing mapped to coarse-to-fine denoising (bottleneck dominates at high noise, skips dominate at low noise) | DEVELOPED | unet-architecture (6.3.1) | The student understands that the same architecture processes all noise levels, with different paths mattering more at different timesteps. This directly motivates WHY the network needs to know the noise level -- the behavior should shift. |
| Dual-path information flow (bottleneck = WHAT, skip connections = WHERE) | DEVELOPED | unet-architecture (6.3.1) | Core mental model from the previous lesson. Relevant because conditioning must modulate BOTH paths, not just one. |
| Residual blocks within the U-Net (Conv -> Norm -> Activation -> Conv -> Norm -> Activation + skip) | INTRODUCED | unet-architecture (6.3.1) | The student knows residual blocks exist inside the U-Net and their general structure. Normalization layers are present but not deeply explored. These blocks are where conditioning gets injected. |
| Timestep embedding as concept (one network handles all timesteps by receiving t as input) | INTRODUCED | learning-to-denoise (6.2.3) | Student knows the network is conditioned on t and that architecture details were deferred. |
| Timestep embedding as simple linear projection (normalize t to [0,1], 2-layer MLP, add to bottleneck features) | INTRODUCED | build-a-diffusion-model (6.2.5) | The capstone's minimal implementation. Explicitly noted as "no sinusoidal encoding, no adaptive group norm." This is the concrete "before" that this lesson upgrades. |
| Batch normalization (normalize activations, learned gamma/beta, Conv-BN-ReLU) | DEVELOPED (in CNN context) | training-dynamics (1.3.6), resnets (3.2) | Student knows the formula, learned scale/shift parameters (gamma/beta), and has used nn.BatchNorm2d. This is the foundation for understanding adaptive normalization. |
| Layer normalization (same formula, different axis: per-token across features) | INTRODUCED | the-transformer-block (4.2.5) | Student knows how layer norm differs from batch norm. Relevant as a normalization variant. |
| Group normalization | MENTIONED | unet-architecture (6.3.1) | Named as "works better than batch norm for small batch sizes." Student has recognition only. This lesson needs to elevate it to at least INTRODUCED before building adaptive group norm on top of it. |
| nn.Linear as learned transformation / "learned lens" | DEVELOPED | queries-and-keys (4.2.2) | Student has used nn.Linear extensively. The MLP that projects the timestep embedding is just nn.Linear layers. |

### Mental models and analogies already established

- **"Clock with many hands"** for sinusoidal encoding (different frequencies capture different granularities) -- directly transferable to timestep embeddings
- **"Bottleneck decides WHAT, skip connections decide WHERE"** -- the conditioning must influence the what/where balance
- **"Same building blocks, different question"** (recurring theme) -- sinusoidal PE is the same building block, applied to a different question (noise level instead of token position)
- **"Keyhole and side doors"** for U-Net information flow -- conditioning enters through the keyhole AND the side doors

### What was explicitly NOT covered in prior lessons (relevant here)

- How the U-Net receives the timestep t internally (deferred from learning-to-denoise AND unet-architecture)
- Why sinusoidal embeddings are better than the simple linear projection used in the capstone
- Adaptive normalization / feature modulation (how timestep information modulates feature maps)
- Group normalization details (only mentioned by name)
- The concept of "global conditioning" (information that affects every spatial location uniformly)

### Readiness assessment

The student is well-prepared. The two core new concepts (sinusoidal timestep embedding, adaptive normalization) both connect directly to established knowledge:
- Sinusoidal timestep embedding is a direct transfer from positional encoding (DEVELOPED)
- Adaptive normalization builds on batch/layer normalization (DEVELOPED/INTRODUCED) and the learned gamma/beta parameters the student already knows

The only gap is group normalization (MENTIONED), which needs a brief elevation before adaptive group norm can be built on it. This is a small gap -- the student has both batch norm (DEVELOPED) and layer norm (INTRODUCED) as bridges.

---

## Phase 2: Analysis

### Target concept

This lesson teaches the student how the U-Net knows what noise level it is denoising -- via sinusoidal timestep embeddings that encode the noise level into a rich vector, and adaptive group normalization that uses this vector to modulate the network's behavior at every layer.

### Prerequisites table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Sinusoidal positional encoding (formula + intuition) | DEVELOPED | DEVELOPED | embeddings-and-position (4.1.3) | OK | Same mechanism repurposed. Student needs to recall the formula, the multi-frequency intuition, and the "uniqueness" property. All at DEVELOPED depth with notebook practice. |
| U-Net architecture (encoder-decoder + skip connections) | DEVELOPED | DEVELOPED | unet-architecture (6.3.1) | OK | This lesson adds conditioning to the architecture. Student needs to know where residual blocks live and how information flows. Taught last lesson. |
| Multi-resolution processing mapped to denoising | DEVELOPED | DEVELOPED | unet-architecture (6.3.1) | OK | Motivates why the network needs to know the noise level: the balance between bottleneck and skip-connection features should shift with t. |
| Residual blocks within the U-Net | INTRODUCED | INTRODUCED | unet-architecture (6.3.1) | OK | Student needs to know the block structure (conv, norm, activation, skip) to understand where conditioning is injected. INTRODUCED is sufficient -- we show where in the block, not ask the student to build one. |
| Timestep embedding concept (network receives t) | INTRODUCED | INTRODUCED | learning-to-denoise (6.2.3) | OK | Student knows the concept exists and was deferred. This lesson delivers on that promise. |
| Simple timestep embedding implementation (linear projection) | INTRODUCED | INTRODUCED | build-a-diffusion-model (6.2.5) | OK | The "before" picture. Student has seen and read annotated code of the minimal version. This lesson explains why it is insufficient and what replaces it. |
| Batch normalization (formula, learned gamma/beta) | DEVELOPED (concept) | DEVELOPED | training-dynamics (1.3), resnets (3.2) | OK | The foundation for understanding adaptive normalization. Student knows normalize -> scale by gamma -> shift by beta. The "adaptive" part replaces learned gamma/beta with timestep-dependent gamma/beta. |
| Layer normalization | INTRODUCED | INTRODUCED | the-transformer-block (4.2.5) | OK | Provides normalization variant awareness. Not directly required but useful context. |
| Group normalization | INTRODUCED | MENTIONED | unet-architecture (6.3.1) | GAP (small) | Student recognizes the name but does not know how it works. Adaptive GROUP normalization requires understanding what group norm is. Brief recap needed. |
| nn.Linear as projection | DEVELOPED | DEVELOPED | Multiple (Series 2, 4.2) | OK | The MLP that processes the timestep embedding is nn.Linear layers. No gap. |

### Gap resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Group normalization (MENTIONED -> INTRODUCED needed) | Small (has batch norm and layer norm as bridges; just needs to understand the "groups" variant) | Brief recap section (2-3 paragraphs + visual). Present group norm as "the middle ground between batch norm and layer norm": divide channels into groups, normalize within each group. One inline diagram showing the three normalization axes (batch norm = across batch per channel, layer norm = across all channels per token, group norm = across channel groups per token). Connect to the U-Net context: small batch sizes in diffusion training make batch norm statistics unreliable, group norm avoids this. This elevates to INTRODUCED. |

### Misconceptions table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Timestep conditioning is just concatenating t to the input image" | The capstone added the timestep at the bottleneck only. The student might generalize that conditioning = adding at one point. Also, many simple conditioning approaches in other ML contexts work via concatenation. | Concrete argument: a single number t concatenated to the input has negligible influence on a 64x64x64 feature map. The network can trivially learn to ignore one extra channel. Also, by the time information passes through several conv layers, the influence of that single input channel is diluted. Compare: adding conditioning to every layer vs adding at input only. | Section 4 (Hook) -- motivate why the simple approach fails, then Section 7 (Elaborate) for the "at every layer" argument |
| "The timestep embedding is just a lookup table (like token embeddings)" | Token embeddings use nn.Embedding, which is a lookup table. The student might expect timestep embeddings to work the same way. | Timestep t is continuous (or at least takes 1000 values, too many for a small lookup table to generalize). More importantly, the sinusoidal encoding provides smooth interpolation between timesteps -- t=500 and t=501 should produce nearly identical embeddings. A random lookup table has no such smoothness guarantee. The four requirements from positional encoding (unique, smooth, any-length, deterministic) apply here for the same reasons. | Section 5 (Explain -- sinusoidal embedding), directly after connecting to positional encoding |
| "Adaptive normalization changes the network's weights based on the timestep" | "Adaptive" sounds like the weights change. The student might conflate "the network behaves differently at different timesteps" with "the weights are different." | The weights are fixed. What changes are the normalization parameters (gamma, beta). The network has one set of conv weights used at all timesteps. Conditioning does not change weights; it changes the statistics of the intermediate features via scale and shift after normalization. Concrete: same conv weights, different gamma/beta -> different output. | Section 6 (Explain -- adaptive normalization) with explicit WarningBlock |
| "You need separate conditioning for each resolution level of the U-Net" | The U-Net has multiple resolution levels with different channel counts. The student might think you need to design separate conditioning for each level. | The timestep embedding is projected to a single vector (e.g., 512-dim). At each residual block, a separate learned linear layer projects this vector to the correct number of channels for that level. Same embedding, different projections per level -- the same "learned lens" pattern from Q/K/V. | Section 6 (Explain -- per-block projection) |
| "Sinusoidal embeddings are outdated / the simple linear projection from the capstone works fine with more training" | The capstone's simple approach "worked" for MNIST. The student might not see why complexity is needed. | The linear projection maps t to a point on a line in embedding space. All timestep information is encoded in magnitude along one direction. Sinusoidal embedding spreads information across many dimensions with different frequencies, giving the MLP a much richer representation to work with. Analogy: describing a position with one number (distance along a line) vs a GPS coordinate (latitude, longitude, altitude). The simple approach works for MNIST because MNIST is easy; it fails at scale. | Section 5 (Explain -- why sinusoidal) as a direct comparison |

### Examples planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Positional encoding callback: same formula, s/position/timestep | Positive (bridge) | Show that the sinusoidal embedding formula is identical to positional encoding with position replaced by timestep. "You already know this formula." Concrete: PE(t=500, dim=0) = sin(500/10000^0) vs PE(pos=500, dim=0) = sin(500/10000^0). Same computation. | Leverages the strongest prior knowledge (DEVELOPED with notebook practice). Makes the new concept feel immediately familiar rather than novel. Reduces cognitive load by showing this is transfer, not invention. |
| Simple projection vs sinusoidal: what each encodes | Positive (comparison) | Side-by-side: the capstone's linear projection maps t to a 128-dim vector along one learned direction. Sinusoidal maps t to a 512-dim vector where each pair of dimensions oscillates at a different frequency. Visualize: the linear projection's output for t=100 and t=900 are two points on a line; the sinusoidal embedding's outputs are two distinct patterns across 512 dimensions. | Directly addresses misconception #5 (simple projection is fine). Provides concrete, visual evidence of why the richer representation matters. Connects to the clock analogy: one hand vs many hands. |
| Batch norm gamma/beta -> adaptive gamma/beta: the minimal conceptual step | Positive (bridge) | Standard batch norm: normalize, then apply learned gamma and beta (fixed after training). Adaptive: normalize, then apply gamma(t) and beta(t) that are FUNCTIONS of the timestep. The formula changes from `gamma * x_norm + beta` to `gamma(t) * x_norm + beta(t)`. One line of change. | This is the core conceptual move. Showing it as a minimal delta from something the student already knows (batch norm gamma/beta at DEVELOPED) minimizes perceived novelty. The student should feel "oh, you just make gamma and beta depend on t." |
| Negative: conditioning at input only (concatenate t as extra channel) | Negative | Show that conditioning at the input gets diluted through conv layers. By the time information reaches the bottleneck (4 downsampling steps), the influence of one extra input channel on 512 feature channels is negligible. The network effectively ignores it. | Directly addresses misconception #1. Motivates why conditioning needs to happen at EVERY layer, not just the input. Establishes the principle of "global conditioning = inject at every processing stage." |
| Negative: random (non-smooth) timestep embedding | Negative | If timestep embeddings were random (like an untrained lookup table), t=500 and t=501 would map to completely different vectors. The network would treat adjacent timesteps as unrelated, unable to leverage the fact that denoising at t=500 is nearly identical to denoising at t=501. Smooth embeddings let the network generalize across nearby timesteps. | Addresses misconception #2 and reinforces the "smoothness" requirement from positional encoding. The student already understands this requirement from 4.1.3; this example reconnects to it in the new context. |

---

## Phase 3: Design

### Narrative arc

The previous lesson opened the U-Net's architecture and showed its elegant multi-resolution design for denoising -- but left a conspicuous gap. The pseudocode had a comment: "timestep parameter omitted, covered next lesson." The student knows the network is supposed to handle all 1000 noise levels with one set of weights, and they know from the coarse-to-fine mental model that the network's behavior SHOULD shift dramatically between t=900 (hallucinating structure from static) and t=50 (polishing fine details). But HOW does the network know which mode to operate in? The capstone's answer was a simple linear projection added at the bottleneck -- and the student saw it work for MNIST. This lesson reveals why that approach is a toy version of the real mechanism, and builds the actual conditioning pipeline used in diffusion models: sinusoidal timestep embeddings (a direct callback to positional encoding from transformers) processed through an MLP, then injected into every residual block via adaptive group normalization. The emotional arc: "Wait, I already know this formula" (sinusoidal) -> "Oh, you just make gamma and beta depend on t" (adaptive norm) -> "Now the pseudocode is complete."

### Modalities planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Verbal/Analogy | "Same clock, different question" -- sinusoidal PE used a clock with many hands to encode position; here the same clock encodes noise level. Also: "The timestep is the conductor's score -- same orchestra (weights), different passage (behavior) depending on what measure the conductor points to." | The clock analogy is already established at DEVELOPED depth. Reusing it makes the transfer explicit: position and timestep both need unique, smooth, deterministic encodings. The conductor analogy captures adaptive normalization: the conductor does not change the instruments, but signals which dynamics to use. |
| Visual (diagram) | (1) Normalization axis comparison: batch norm vs layer norm vs group norm as 3D tensor slices. (2) Residual block diagram with conditioning injection point clearly marked. (3) Side-by-side: simple projection (points on a line) vs sinusoidal (rich multi-frequency pattern). | Group norm needs a visual to bridge from batch/layer norm. The residual block diagram concretizes WHERE conditioning enters. The projection comparison makes the "why sinusoidal" argument visual rather than purely verbal. |
| Symbolic (formula/code) | (1) Sinusoidal formula with timestep: emb(t, 2i) = sin(t / 10000^(2i/d)), explicitly shown alongside the positional encoding formula for comparison. (2) Adaptive group norm formula: AdaGN(x, t) = gamma(t) * GroupNorm(x) + beta(t), where [gamma(t), beta(t)] = Linear(emb_t). (3) Pseudocode completing the U-Net forward pass with conditioning. | The formulas ground the concepts precisely. The side-by-side sinusoidal comparison makes the transfer undeniable. The pseudocode completes the promise from the previous lesson -- the student finally sees the full forward pass. |
| Concrete example | Specific numbers: compute sinusoidal embedding for t=500 at dimensions 0, 1, 2, 3 (showing different frequencies). Then: gamma(500) = 1.3, beta(500) = -0.2 applied to a normalized feature map. Then: gamma(50) = 0.8, beta(50) = 0.5 for the same feature map -- different timestep, different modulation, different output. | Numbers make the abstract concrete. Showing two different timesteps with different gamma/beta values demonstrates that the same architecture produces different behaviors depending on t. This is the core insight. |
| Intuitive | The "of course" moment: if the network needs to behave differently at high noise vs low noise, but has fixed weights, SOMETHING must signal the noise level -- and it must do so at every processing stage, not just the input. The solution: a rich encoding of t injected at every layer via scale and shift of normalized features. | The motivation chain (the network needs to know t -> injection at input is insufficient -> inject at every layer) creates a sense of inevitability rather than arbitrary design. |

### Cognitive load assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. **Sinusoidal timestep embedding** -- but this is a direct transfer from sinusoidal positional encoding (DEVELOPED), so the novelty is low. The new part is "same formula, different application."
  2. **Adaptive group normalization** -- genuinely new mechanism, but built on batch norm gamma/beta (DEVELOPED). The conceptual move is small: "make gamma and beta depend on t."

  Group normalization needs elevation from MENTIONED to INTRODUCED, but this is a brief gap-fill (how it differs from batch/layer norm), not a full new concept.

- **Previous lesson load:** BUILD (unet-architecture: re-entering theoretical mode after CONSOLIDATE capstone)
- **This lesson's load:** BUILD -- appropriate. Two new concepts, both heavily leveraged from prior knowledge. The genuine novelty is lower than a STRETCH lesson. Follows the trajectory: BUILD -> BUILD -> STRETCH (CLIP next).

### Connections to prior concepts

| Prior Concept | How It Connects | Risk of Misleading |
|---------------|----------------|--------------------|
| Sinusoidal positional encoding (4.1.3) | Same formula, same intuition. Position in a sequence -> noise level in diffusion. "Clock with many hands" analogy extends directly. | Low risk. The transfer is clean. The only difference is the input (timestep integer vs position integer) and the purpose (encoding noise level vs encoding position). |
| Batch normalization gamma/beta (1.3, 3.2) | Adaptive norm replaces learned (fixed) gamma/beta with timestep-dependent gamma/beta. The normalization step is identical; only the source of scale/shift changes. | Medium risk: student might think "adaptive" means the normalization statistics change, not just gamma/beta. Address explicitly: the normalization (mean/variance computation) is standard; what is adaptive is the scale and shift AFTER normalization. |
| Residual blocks within U-Net (6.3.1) | Conditioning is injected inside each residual block, specifically at the normalization step. The block structure (conv -> norm -> activation -> conv -> norm -> activation + skip) gains a conditioning injection point. | Low risk. Student has the block structure at INTRODUCED depth, which is sufficient to understand the injection point. |
| "Same building blocks, different question" (recurring) | Sinusoidal encoding is a building block; the question changes from "what position?" to "what noise level?" | No risk. This analogy has been reliable throughout the course. |
| nn.Linear as "learned lens" (4.2) | The MLP that projects the timestep embedding, and the per-block linear layers that produce gamma/beta, are nn.Linear -- the same projection tool. | No risk. Familiar operation. |
| Multi-resolution conditioning (6.3.1) | The lesson connects conditioning to the coarse-to-fine insight: at high t, the network should emphasize bottleneck processing (structural decisions); at low t, emphasize skip connections (fine details). Conditioning via adaptive norm at every level is what enables this differential behavior. | Low risk. The connection enriches existing understanding without contradicting it. |

### Analogies to extend or retire

- **Extend "clock with many hands"** to timestep embeddings. Same clock mechanism, encoding noise level instead of position.
- **Extend "learned lens" / nn.Linear** to the projection layers.
- **Introduce "conductor's score"** for adaptive normalization: the conductor (timestep embedding) does not change the instruments (conv weights) or the sheet music (architecture), but signals the dynamics (loud/soft = scale) and key (sharp/flat = shift) for each passage. Different measure numbers (timesteps) produce different performances from the same orchestra.
- **Batch norm gamma/beta as "tone controls"**: standard norm has fixed tone controls set during training; adaptive norm has tone controls that respond to the timestep signal. Like a graphic equalizer that adjusts based on what genre is playing.

### Scope boundaries

**This lesson IS about:**
- How the timestep integer t becomes a rich embedding vector (sinusoidal encoding + MLP)
- How that embedding vector modulates the U-Net's behavior at every layer (adaptive group normalization)
- Why conditioning needs to happen at every residual block, not just at the input
- Completing the pseudocode from the previous lesson with the timestep parameter
- The concept of "global conditioning" -- information that affects every spatial location

**This lesson is NOT about:**
- Text conditioning or cross-attention (Lesson 4: text-conditioning-and-guidance)
- CLIP or text embeddings (Lesson 3: clip)
- Self-attention within the U-Net (mentioned in unet-architecture, deferred to later)
- Classifier-free guidance (Lesson 4)
- Implementation / notebook (this is a conceptual BUILD lesson; the full implementation is the Module 6.4 capstone)
- The details of how group normalization is computed (elevated to INTRODUCED, not DEVELOPED)
- Alternative conditioning mechanisms (FiLM, conditional instance norm, etc.) -- may mention FiLM as the general name for this pattern
- Training details or loss function changes due to conditioning (the training algorithm is unchanged)

**Depth targets:**
- Sinusoidal timestep embedding: DEVELOPED (understand the formula, the connection to PE, why it is better than linear projection, how the MLP processes it)
- Adaptive group normalization: DEVELOPED (understand the mechanism -- timestep-dependent gamma/beta applied after normalization -- and why it is injected at every block)
- Group normalization: INTRODUCED (know what it does, how it differs from batch/layer norm, why it is used in diffusion)
- Global conditioning pattern: INTRODUCED (the principle that conditioning information should be injected at every processing stage)
- FiLM conditioning: MENTIONED (name-drop as the general framework; adaptive norm is an instance of FiLM)

### Lesson outline

**1. Context + Constraints**
- This lesson completes the U-Net architecture by adding timestep conditioning -- how the network knows what noise level it is denoising.
- Scope: sinusoidal timestep embeddings and adaptive normalization. NOT text conditioning (that comes later).
- By the end: the pseudocode from the previous lesson will be complete.

**2. Recap (brief -- group normalization gap fill)**
- The student knows batch norm (normalize across batch per channel) and layer norm (normalize across all channels per token).
- Group norm is the middle ground: divide channels into groups (e.g., 32 groups), normalize within each group. Works for any batch size because statistics are computed per-example, per-group.
- Inline visual: normalization axis diagram showing batch norm / layer norm / group norm as slices of a [batch, channel, height, width] tensor.
- Why diffusion uses group norm: batch sizes are small (expensive 64x64 or 256x256 images), so batch norm statistics are unreliable. Group norm is independent of batch size.
- Elevates group normalization from MENTIONED to INTRODUCED.

**3. Hook: "The pseudocode is incomplete"**
- Type: Callback/puzzle.
- Recall the pseudocode from unet-architecture. There was a comment: "timestep parameter omitted." The student was promised this lesson would fill it in.
- Pose the question: The network has one set of weights. It must denoise at t=900 (global structure from static) AND at t=50 (fine detail polish). These are radically different tasks. How does the same network do both?
- Briefly recall the capstone's solution: normalize t to [0,1], pass through a 2-layer MLP, add to bottleneck. That "worked" for MNIST. But it is like describing your GPS position with a single number instead of latitude + longitude + altitude. And it only injects conditioning at ONE point (the bottleneck), not at every processing stage.
- Promise: "You already know the formula for the solution. You saw it in Module 4.1."

**4. Explain -- Sinusoidal Timestep Embedding (DEVELOPED)**
- **Connection to positional encoding:** Side-by-side formulas.
  - PE(pos, 2i) = sin(pos / 10000^(2i/d_emb))
  - TE(t, 2i) = sin(t / 10000^(2i/d_emb))
  - "Same formula. Different input. Different question."
- **"Same clock, different question":** In transformers, the clock encodes position in a sequence. In diffusion, the same clock encodes the noise level. The four requirements still hold: unique (each timestep gets a distinct pattern), smooth (nearby timesteps get similar patterns), works for any range, deterministic.
- **Concrete example:** Compute TE(t=500) at dimensions 0, 1, 2, 3 and TE(t=50) at the same dimensions. Show that dim 0 oscillates rapidly (captures fine timestep differences) while dim 3 changes slowly (captures broad noise level). This is the clock analogy in action: the "second hand" (dim 0) distinguishes t=500 from t=501; the "hour hand" (dim 3) distinguishes "early in denoising" from "late in denoising."
- **Why not the simple linear projection?** ComparisonRow:
  - Simple projection: t -> normalize -> MLP -> 128-dim vector. The MLP can only learn to spread t along directions in the embedding space. Adjacent timesteps have no guaranteed similarity. All timestep information is encoded in magnitude/direction along learned axes.
  - Sinusoidal + MLP: t -> sinusoidal encoding -> MLP -> 512-dim vector. The sinusoidal encoding provides a rich, multi-frequency starting point. The MLP refines it. Adjacent timesteps are inherently similar because sine/cosine vary smoothly.
- **The MLP refinement step:** The sinusoidal encoding is the input to a 2-layer MLP (Linear -> GELU -> Linear) that produces the final timestep embedding vector. The MLP learns to combine and transform the raw frequencies into features useful for denoising. This is a standard pattern: provide a structured input, let the network refine it.
- **Predict-and-verify check:** "If you doubled the number of dimensions in the sinusoidal encoding, would the embedding be better?" (More dimensions = more frequency bands = finer discrimination between timesteps, but with diminishing returns. In practice, 128 or 256 sinusoidal dimensions processed by an MLP to 512 is common.)

**5. Check (predict-and-verify)**
- "The sinusoidal encoding for t=500 and t=501 -- would you expect them to be nearly identical, completely different, or somewhere in between?" (Nearly identical -- the sine/cosine functions are smooth. This is a direct callback to the smoothness requirement from positional encoding.)
- "The sinusoidal encoding for t=500 and t=50 -- same question." (Quite different -- the "hour hand" dimensions have changed significantly. This captures the large behavioral difference between high-noise and low-noise denoising.)

**6. Explain -- Adaptive Group Normalization (DEVELOPED)**
- **Problem statement:** The timestep embedding is a vector. The U-Net processes 2D feature maps. How does the vector influence the feature maps?
- **Build from what the student knows (batch norm gamma/beta):**
  - Standard batch/group norm: normalize features, then apply learned scale (gamma) and shift (beta). These are FIXED parameters after training -- the same gamma and beta for every input, every timestep.
  - The conceptual move: what if gamma and beta DEPENDED on the timestep?
  - AdaGN formula: AdaGN(x, t) = gamma(t) * GroupNorm(x) + beta(t)
  - Where [gamma(t), beta(t)] = Linear(timestep_embedding)
  - "One line of change. The normalization is standard. The scale and shift are timestep-dependent."
- **Concrete example with numbers:**
  - At t=500: gamma(500) = [1.3, 0.7, 1.1, ...], beta(500) = [-0.2, 0.4, 0.0, ...]
  - At t=50: gamma(50) = [0.8, 1.2, 0.9, ...], beta(50) = [0.5, -0.1, 0.3, ...]
  - Same feature map after normalization. Different gamma/beta -> different output features.
  - "Same orchestra, different dynamics. The conductor (timestep) tells the musicians (features) how loud to play (gamma) and what key to play in (beta)."
- **WarningBlock -- "Adaptive does NOT mean the weights change":**
  - The conv weights are fixed. The normalization computation (mean, variance) is standard. What changes are ONLY the scale and shift parameters after normalization. The network has one set of weights; conditioning changes how intermediate features are scaled and shifted, which changes the network's effective behavior without changing any weights.
- **Where in the residual block:**
  - Updated residual block: Conv -> AdaGN(t) -> Activation -> Conv -> AdaGN(t) -> Activation + skip
  - Diagram showing a single residual block with the timestep embedding arrow entering at both normalization points.
  - The timestep embedding feeds a per-block linear layer that produces [gamma, beta] with the right number of channels for that block.
- **Per-block projection (connect to "learned lens"):**
  - The timestep embedding is ONE vector (e.g., 512-dim).
  - Each residual block has its own small linear layer: Linear(512, 2 * channels) -> split into gamma and beta.
  - Different blocks at different resolution levels have different numbers of channels (64, 128, 256, 512). Each block's linear layer projects the SAME timestep embedding to the right size.
  - "Same embedding, different lens per block. You know this pattern from Q/K/V -- one embedding, multiple learned projections."

**7. Elaborate -- Why at Every Layer? (negative example + global conditioning pattern)**
- **Negative example: conditioning at input only.**
  - If you concatenate t as an extra channel to the input image, the conditioning signal passes through 4 downsampling blocks before reaching the bottleneck. Each conv layer blends the conditioning channel with spatial features. By the bottleneck, the conditioning signal is thoroughly mixed and diluted.
  - The capstone's approach (add at bottleneck only) is better but still limited: the decoder must recover conditioning information from the bottleneck features, and the skip connections carry NO conditioning information at all.
- **Adaptive norm at every block means:**
  - At EVERY resolution level, on BOTH the encoder and decoder paths, the network's behavior is modulated by the timestep.
  - The high-resolution encoder blocks get timestep awareness (important: they process the least-noised features, so they need to know whether to emphasize fine details or ignore them).
  - The bottleneck gets timestep awareness (important: it makes global structural decisions).
  - The decoder blocks get timestep awareness (important: they decide how aggressively to reconstruct details).
  - The skip connections carry spatial information that is then MODULATED by the decoder's timestep-aware processing.
- **Global conditioning pattern (INTRODUCED):**
  - Name the pattern: "global conditioning" -- information that should influence every spatial location and every processing stage.
  - The timestep is the canonical example: it applies equally to every pixel. Whether you are denoising the top-left corner or the bottom-right corner, the noise level is the same.
  - Later in the module: text conditioning via cross-attention is a DIFFERENT pattern -- "spatially-varying conditioning" where different spatial locations attend to different parts of the text. Preview this contrast briefly (1-2 sentences) without developing it.
- **FiLM conditioning (MENTIONED):**
  - Name-drop: "This pattern -- predicting scale and shift parameters from a conditioning signal -- is called Feature-wise Linear Modulation (FiLM). Adaptive group norm is a specific instance of the FiLM pattern."
  - 1-2 sentences. No development.

**8. Check (transfer question)**
- "The previous lesson's mental model was 'bottleneck decides WHAT, skip connections decide WHERE.' With adaptive normalization, how does the timestep influence the what/where balance?"
  - Expected: At high t (heavy noise), the adaptive norm parameters at bottleneck blocks could amplify structural features (the WHAT). At low t (fine detail refinement), the adaptive norm at high-resolution blocks could amplify fine spatial features. The timestep does not change which path information flows through, but it changes what each path emphasizes via scale/shift.
- "Your colleague says: 'Conditioning at every layer is wasteful. Most of the information is the same timestep repeated. You only need it at the bottleneck.' What is wrong with this reasoning?"
  - Expected: The timestep is the same, but the PROJECTION is different at each block. Each block's linear layer learns to extract different aspects of the timestep for its resolution level. The bottleneck block might emphasize "how much structure to hallucinate," while a high-resolution encoder block might emphasize "how much to trust the fine details."

**9. Explain -- Complete Forward Pass (synthesis)**
- Present the complete U-Net forward pass pseudocode, now with timestep conditioning. This fulfills the promise from the previous lesson.
- Pseudocode showing:
  1. Compute sinusoidal encoding of t
  2. Pass through MLP to get timestep embedding
  3. Encoder path: each block processes features with AdaGN(t)
  4. Bottleneck: processes with AdaGN(t)
  5. Decoder path: each block processes concatenated features (encoder skip + decoder) with AdaGN(t)
  6. Output: predicted noise
- Highlight the additions from the previous lesson's pseudocode in a different color/style. The spatial architecture is unchanged; only the conditioning is new.
- InsightBlock: "The U-Net from the previous lesson was the skeleton. This lesson added the nervous system -- the timestep signal that tells each bone how to move."

**10. Practice -- Notebook exercises (Colab)**
- **Exercise design rationale:** This lesson introduces two conceptual mechanisms (sinusoidal embedding, adaptive norm) that are straightforward to implement in isolation. The exercises should verify understanding of both, then combine them.
- **Exercise sequence (cumulative):**
  1. **(Guided) Sinusoidal timestep embedding:** Implement the sinusoidal encoding function from the formula. Compute embeddings for t=0, 50, 500, 950, 999. Visualize as a heatmap (reusing the mental model from the PositionalEncodingHeatmap in 4.1.3). Predict-before-run: "Will t=500 and t=501 look similar or different in the heatmap?" Verify smoothness. Compare to a random embedding (torch.randn for each t) to see the smoothness contrast.
  2. **(Guided) Timestep MLP:** Build the 2-layer MLP (Linear -> GELU -> Linear) that processes the sinusoidal encoding. Pass the same timesteps through and observe the refined embeddings. Visualize with cosine similarity matrix: t=500 vs t=501 should be very similar, t=500 vs t=50 should be quite different.
  3. **(Supported) Adaptive group normalization:** Implement AdaGN as a small module: takes feature maps and timestep embedding, applies GroupNorm, then scales and shifts with timestep-dependent gamma/beta. Test with a dummy feature map at two different timesteps. Verify that the output changes when the timestep changes, even though the feature map is the same.
  4. **(Independent) Compare simple vs sinusoidal conditioning:** Take the capstone's minimal U-Net (provided). Replace the simple linear timestep embedding with sinusoidal + MLP. Train both versions on MNIST for 10 epochs. Compare loss curves. The sinusoidal version should converge faster or to a lower loss. Reflection: why?
- **Solutions should emphasize:** The smoothness property of sinusoidal embeddings (adjacent timesteps produce similar embeddings), the minimal conceptual delta from standard normalization to adaptive normalization, and the per-block projection pattern.

**11. Summarize**
- Two mechanisms complete the unconditional DDPM architecture:
  1. **Sinusoidal timestep embedding:** Same formula as positional encoding from transformers. Encodes the noise level as a rich, multi-frequency vector. Processed by an MLP into the timestep embedding.
  2. **Adaptive group normalization:** The timestep embedding modulates the network's behavior at every residual block by predicting timestep-dependent scale (gamma) and shift (beta) parameters. Same architecture, same weights -- different behavior at different noise levels.
- Mental model echo: "The U-Net is the orchestra. The timestep embedding is the conductor's score. The adaptive norm is how the conductor communicates dynamics to each section."
- The pseudocode is now complete.

**12. Next step**
- The U-Net now knows the noise level, completing the unconditional DDPM architecture. But Stable Diffusion generates images from TEXT descriptions. For that, the network needs a second conditioning signal: text.
- The next lesson introduces CLIP -- a model that creates a shared embedding space for text and images. CLIP provides the text embeddings that will eventually be injected into the U-Net alongside the timestep.
- "The timestep tells the network WHEN it is in the denoising process. The text will tell it WHAT to generate."

---

## Widget Assessment

**Widget needed:** No dedicated interactive widget.

**Rationale:** The core concepts (sinusoidal embedding, adaptive normalization) are best understood through formulas, comparisons, and concrete numerical examples rather than interactive exploration. The sinusoidal encoding is already familiar from the PositionalEncodingHeatmap widget in 4.1.3 -- rehashing it would not add value. Adaptive normalization is a formula-level concept (gamma(t) * norm(x) + beta(t)) that is best conveyed through worked examples and a clear residual block diagram.

The lesson uses:
- Inline diagrams (normalization axis comparison, residual block with conditioning injection)
- ComparisonRow (simple projection vs sinusoidal)
- Concrete numerical examples (specific gamma/beta values at different timesteps)
- Pseudocode completion (the payoff from the previous lesson)
- Colab notebook exercises (the interactive component)

If a widget were added, the most valuable would be a small demo showing the same feature map modulated by different timestep gamma/beta values -- but this is better served by the notebook Exercise 3.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (group norm gap resolved with brief recap)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (group norm: brief recap section)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept (verbal/analogy, visual, symbolic, concrete example, intuitive -- 5 modalities)
- [x] At least 2 positive examples + 1 negative example (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 2 new concepts (within limit of 3)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-10 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 2

### Verdict: NEEDS REVISION

No critical conceptual failures that would leave the student lost. One critical finding (missing notebook) blocks the lesson from being complete. Four improvement findings that would meaningfully strengthen the lesson. Two polish items.

### Findings

### [CRITICAL] -- Notebook missing; planning doc specifies 4 exercises

**Location:** Practice section (Section 11 in the lesson, Phase 3 outline point 10 in the planning doc)
**Issue:** The planning document specifies 4 notebook exercises (Guided: sinusoidal embedding, Guided: timestep MLP, Supported: adaptive group norm, Independent: compare simple vs sinusoidal). The lesson component describes all 4 exercises in its "Practice: Hands-On Exercises" section. However, no notebook file exists at `notebooks/6-3-2-conditioning-the-unet.ipynb` or any similar path. The review skill is explicit: "Notebook missing -- planning doc specifies exercises but no notebook was created" is a CRITICAL finding.
**Student impact:** The student reads about 4 exercises they should do, clicks through to... nothing. The exercises are where the DEVELOPED depth targets for sinusoidal timestep embedding and adaptive group normalization would actually be achieved. Without the notebook, these concepts remain at INTRODUCED depth at best.
**Suggested fix:** Create the notebook with all 4 exercises following the scaffolding progression specified in the planning doc (Guided -> Guided -> Supported -> Independent). Ensure the first code cell handles pip installs and imports. Set random seeds. Include `<details>` solution blocks for exercises 3 and 4.

### [IMPROVEMENT] -- Concrete sinusoidal example uses fabricated denominators

**Location:** "Sinusoidal Timestep Embedding" section, the concrete example block for t=500 and t=50
**Issue:** The lesson claims to compute embeddings "using d=256" and lists values like "dim 0: sin(500/1)", "dim 2: sin(500/10)", "dim 4: sin(500/100)", "dim 6: sin(500/1000)". The actual formula is `sin(t / 10000^(2i/d))`. For d=256: dim 0 (i=0) gives 10000^0 = 1 (correct). Dim 2 (i=1) gives 10000^(2/256) = 10000^0.0078 = ~1.07 (not 10). Dim 4 (i=2) gives 10000^(4/256) = 10000^0.0156 = ~1.15 (not 100). The denominators 1, 10, 100, 1000 look like they come from d=8 (where 10000^(2/8)=10, 10000^(4/8)=100, 10000^(6/8)=1000), not d=256. The numerical values (-0.47, -0.51, 0.98, 0.48) are computed using these wrong denominators.
**Student impact:** A careful student who plugs the formula in with d=256 will get different numbers and be confused. A less careful student will absorb incorrect intuition about how quickly the frequencies change across dimensions. Either way, the concrete example--which is supposed to ground the concept--contains incorrect math.
**Suggested fix:** Either (a) change d to 8 and label the dimensions correctly (dim 0 through dim 6, i=0 through i=3), explicitly noting this is a small d for illustration, or (b) keep d=256 but use the actual denominators (1, ~1.07, ~1.15, ~1.23) with correct output values, and pick dimensions that are more spread out (e.g., dim 0, dim 64, dim 128, dim 192) to show the frequency progression clearly. Option (a) is simpler and pedagogically cleaner.

### [IMPROVEMENT] -- Group norm recap lacks the planned visual (normalization axis diagram)

**Location:** Section 3 "Quick Recap: Group Normalization"
**Issue:** The planning doc under "Gap resolution" specifies: "One inline diagram showing the three normalization axes (batch norm = across batch per channel, layer norm = across all channels per token, group norm = across channel groups per token)." The built lesson has a 3-column GradientCard comparison (Batch Norm / Layer Norm / Group Norm) with bullet points, but no visual diagram showing the normalization axes on a tensor. These are different modalities. The bullet-point cards are text; the planned diagram is visual/spatial and would show the student geometrically which elements are grouped for normalization.
**Student impact:** The student gets a verbal description of three normalization variants but never sees the spatial pattern of "which elements in the tensor are grouped together." This is a concept where visual representation is particularly effective--the standard "colored cube slices" diagram in the Wu & He 2018 paper is iconic precisely because it makes the grouping immediately obvious in a way text cannot.
**Suggested fix:** Add a normalization axis diagram above or alongside the GradientCards. This could be a simple inline illustration (even ASCII-style or a static image) showing a [B, C, H, W] tensor with colored regions indicating which elements are normalized together for each variant. The GradientCards can remain as a text summary alongside the visual.

### [IMPROVEMENT] -- No residual block diagram with conditioning injection point

**Location:** Section 7 (Adaptive Group Normalization), around the residual block pseudocode
**Issue:** The planning doc under "Modalities planned" specifies: "Residual block diagram with conditioning injection point clearly marked." The built lesson has a code block showing the residual block with AdaGN but no visual diagram of the block with arrows showing where the timestep embedding enters. There is a full U-Net architecture Mermaid diagram later (Section 10) but it shows the macro architecture, not the internal structure of a single residual block.
**Student impact:** The pseudocode tells the student the order of operations, but a diagram would show the flow visually--particularly the two injection points for t_emb within a single block. This is the "parts before whole" principle: the student should see the conditioning flow inside one block before seeing it in the full architecture diagram.
**Suggested fix:** Add a small Mermaid diagram or inline illustration showing a single residual block: Input -> Conv1 -> AdaGN(t_emb) -> Activation -> Conv2 -> AdaGN(t_emb) -> Activation -> (+skip) -> Output, with the t_emb arrow entering at both AdaGN nodes. Place it near the existing residual block pseudocode.

### [IMPROVEMENT] -- Misconception #4 (separate conditioning per resolution) is only implicitly addressed

**Location:** Per-block projection section (around lines 646-689)
**Issue:** The planning doc identifies misconception #4: "You need separate conditioning for each resolution level of the U-Net." The plan says to address it in Section 6 (per-block projection). The built lesson does explain the per-block projection pattern and connects it to Q/K/V projections. However, it never explicitly states the misconception as something the student might wrongly believe, nor does it have a WarningBlock or explicit callout. The text addresses it implicitly by showing how one embedding serves all blocks, but the misconception is never surfaced and debunked. Compare to misconception #3 ("adaptive means weights change") which gets an explicit WarningBlock.
**Student impact:** A student who holds this misconception might read through the per-block projection section and not realize their mental model was wrong, because the lesson never says "you might think you need separate conditioning for each level, but you do not." The student might even interpret the per-block linear layers as confirming their misconception ("see, there ARE separate layers for each level").
**Suggested fix:** Add a brief explicit callout (a sentence or two, or a TipBlock) that names the misconception and corrects it: "You might expect each resolution level to need its own conditioning signal. It does not. All blocks share the same timestep embedding. Each block only has its own small projection layer to extract the aspect of the timestep relevant at its resolution. One source, many views."

### [POLISH] -- The lesson describes exercises but has no Colab link

**Location:** Section 11 "Practice: Hands-On Exercises"
**Issue:** The exercise section describes 4 exercises with scaffolding levels but does not include a link to open the notebook in Colab. There is no `href` or button to click. (This is partially a consequence of the CRITICAL finding that the notebook does not exist, but even once the notebook is created, the lesson needs a link.)
**Student impact:** Minor friction--the student sees exercise descriptions but has to navigate elsewhere to find the notebook. Low activation energy principle requires a direct link.
**Suggested fix:** Once the notebook is created, add a Colab link button at the top of the exercises section using the standard pattern: `https://colab.research.google.com/github/{user}/{repo}/blob/main/notebooks/6-3-2-conditioning-the-unet.ipynb`.

### [POLISH] -- Predict-and-verify check from planning doc (doubling sinusoidal dimensions) is absent

**Location:** End of Section 5 (Sinusoidal Timestep Embedding), before the first Check section
**Issue:** The planning doc's outline point 4 includes a predict-and-verify check: "If you doubled the number of dimensions in the sinusoidal encoding, would the embedding be better?" This was planned to test whether the student understands the diminishing-returns nature of adding more frequency bands. It is not present in the built lesson. The existing check section (Section 6) has two good questions about smoothness, but the dimensionality question tests a different aspect of understanding.
**Student impact:** Minor. The two existing questions are solid. The missing question would have added another angle of understanding but its absence does not leave a gap.
**Suggested fix:** Consider adding this as a third question in the first "Predict and Verify" GradientCard, or as a brief aside. Low priority.

### Review Notes

**What works well:**
- The lesson follows the planned narrative arc faithfully. The "pseudocode is incomplete" hook creates genuine motivation, the sinusoidal embedding section leverages prior knowledge effectively, and the adaptive group norm section builds from what the student knows (batch norm gamma/beta) with a genuinely minimal conceptual delta.
- The connection to positional encoding is handled well: side-by-side formulas, clock analogy callback, four requirements callback. The student should feel "I already know this" rather than "this is new."
- The conductor/orchestra analogy for adaptive normalization is effective and threaded consistently.
- The two "Check Your Understanding" sections ask genuine transfer questions that require integration, not just recall.
- Scope boundaries are well maintained. The lesson does not drift into text conditioning, cross-attention, or implementation details.
- The global conditioning vs spatially-varying conditioning contrast at the end (1-2 sentences) is a good preview without overloading.
- The complete forward pass pseudocode fulfills the promise from the previous lesson satisfyingly.

**Pattern observations:**
- The lesson is text-heavy compared to what was planned. The planning doc specified 3 visual modalities (normalization axis diagram, residual block diagram, projection comparison), of which only the Mermaid U-Net architecture diagram and the ComparisonRow were built. The lesson compensates with more prose and code blocks, but the visual modality is underrepresented for the adaptive group norm concept specifically.
- The misconceptions are addressed but not always surfaced explicitly. Misconception #3 (weights change) gets the WarningBlock treatment; the others are addressed through explanation rather than explicit "you might think X, but actually Y" framing. Consider whether additional WarningBlocks or explicit callouts would help.

---

## Review -- 2026-02-10 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All four Improvement findings from iteration 1 have been correctly addressed. The fixes do not introduce any new issues. The lesson meets all pedagogical principles, follows the plan faithfully, and would work well for the target student. The notebook is complete, well-scaffolded, and aligned with the lesson content.

### Findings

### [POLISH] -- Spaced em dashes in sinusoidal concrete example

**Location:** Sinusoidal Timestep Embedding section, concrete example list items (lines 349, 352)
**Issue:** Two instances of `&mdash;` with spaces on both sides in rendered inline text: `sin(500/1) = **-0.47** &mdash; oscillates rapidly` and `sin(500/1000) = **0.48** &mdash; changes slowly`. The writing style rule specifies em dashes with no spaces: `word&mdash;word` not `word &mdash; word`.
**Student impact:** Negligible. The student will not notice a style difference in em dash spacing. Purely a consistency issue.
**Suggested fix:** Remove the spaces around `&mdash;` on both lines, e.g., `<strong>&minus;0.47</strong>&mdash;oscillates rapidly`.

### Review Notes

**Iteration 1 fixes verified:**
1. **Sinusoidal denominators (Improvement #1):** Fixed correctly. The lesson now uses d=8 for illustration and the denominators (1, 10, 100, 1000) match the formula `10000^(2i/d)` with d=8. All six numerical values (sin values for t=500 and t=50 at four dimensions) were independently verified against the actual formula and are correct to two decimal places.
2. **Group norm ASCII visual (Improvement #2):** Added. A clear ASCII diagram shows batch norm (columns span batch), layer norm (rows span all channels), and group norm (half-rows span channel groups). The diagram is accompanied by the three GradientCards that were already present. The visual+text combination now provides two modalities for the normalization comparison.
3. **Residual block Mermaid diagram (Improvement #3):** Added. Shows a single residual block with Input -> Conv1 -> AdaGN -> Activation -> Conv2 -> AdaGN -> Activation -> (+) -> Output, with t_emb arrows entering both AdaGN nodes. The amber highlighting on AdaGN nodes and the purple highlighting on the residual skip (+) make the conditioning injection points visually clear. Placed immediately after the residual block pseudocode, following the "parts before whole" principle.
4. **Misconception #4 WarningBlock (Improvement #4):** Added as "One Embedding, Not Many" in the per-block projection aside. Explicitly surfaces the misconception ("You might expect each resolution level to need its own conditioning signal. It does not.") and corrects it. Placed right next to the per-block projection code, where the student would most likely form this misconception.
5. **Colab link (Polish #6):** Added. A styled link button appears at the top of the exercises section, matching the standard Colab link pattern.
6. **Extra predict-verify question (Polish #7):** Intentionally not addressed per iteration 1 notes. The two existing questions are solid and the missing dimensionality question does not leave a pedagogical gap.

**Notebook evaluation:**
- All 4 exercises match the planning doc specification (Guided -> Guided -> Supported -> Independent).
- Guided exercises have genuine predict-before-run prompts ("Will t=500 and t=501 look similar or different?").
- Supported exercise (AdaGN) uses TODO markers with hints, has a `<details>` solution block with reasoning before code.
- Independent exercise (comparison) provides the baseline model and infrastructure, asks the student to modify one component, has a `<details>` solution block.
- Setup cell imports standard libraries (no pip installs needed for Colab), sets random seeds, detects GPU.
- Terminology aligns with the lesson ("clock with many hands," "conductor's score," same formula references).
- No new concepts introduced beyond what the lesson teaches.

**What works well (in addition to iteration 1 observations):**
- The ASCII normalization diagram is particularly effective. It shows the grouping pattern at a glance in a way that the text description alone could not.
- The Mermaid residual block diagram complements the pseudocode well. The amber highlighting for AdaGN and purple for the residual skip creates an immediate visual hierarchy.
- The WarningBlock for misconception #4 is well-placed. It directly follows the per-block projection code, surfacing and correcting the misconception at exactly the point where it would form.
- The notebook's Exercise 3 (AdaGN) is well-designed: the student implements the mechanism, tests it with two different timesteps on the same feature map, and sees that the output changes. This concretely demonstrates the core insight.
- The notebook's Exercise 1 comparison of sinusoidal vs random embeddings (cosine similarity) is a strong practical demonstration of the smoothness property.

**Overall:** The lesson is ready to ship. The iteration 1 fixes resolved all substantive issues without introducing new problems. The remaining Polish finding is trivial and can be fixed at any time.
