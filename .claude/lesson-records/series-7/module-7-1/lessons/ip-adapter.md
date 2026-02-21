# Lesson: IP-Adapter -- Planning Document

**Module:** 7.1 (Controllable Generation)
**Position:** Lesson 3 of 3
**Slug:** `ip-adapter`
**Cognitive load type:** BUILD
**Date planned:** 2026-02-19

---

## Phase 1: Orient (Student State)

### Relevant Concepts with Depths and Sources

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings; same QKV formula as self-attention) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Deep understanding of the one-line change from self-attention. Concrete attention weight table for per-spatial-location attention. Shape walkthrough: Q (256xd_k), K (Txd_k), attention weights (256xT). Three-lens framing (seeking/advertising/contributing). Practiced in notebook. |
| Per-spatial-location cross-attention (each spatial location generates its own query, attends independently to all text tokens, creating spatially-varying conditioning) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Core insight: different spatial locations attend to different text tokens. Cat's face attends to "cat," sky region attends to "sunset." |
| CLIP dual-encoder architecture (separate image encoder + text encoder, no shared weights, loss-only coupling) | INTRODUCED | clip (6.3.3) | Student knows CLIP has two separate encoders producing vectors in a shared embedding space. Has not built CLIP from scratch. |
| Shared embedding space (text and image vectors in the same geometric space, enabling cross-modal cosine similarity) | DEVELOPED | clip (6.3.3) | Deep understanding from SVG diagrams, ComparisonRow with VAE latent space, and notebook exercises. Knows alignment comes from contrastive loss, not architecture. |
| ControlNet architecture (trainable encoder copy, zero convolution, additive feature-level merging) | DEVELOPED | controlnet (7.1.1) | Core understanding of the frozen-model + trainable-copy pattern. Knows zero conv ensures safe initialization. Can explain the full "of course" chain. |
| ControlNet coexistence with text conditioning (WHEN/WHAT/WHERE framing) | DEVELOPED | controlnet-in-practice (7.1.2) | Practiced the three-dimensional conditioning framework. Directly manipulated text-spatial balance via conditioning scale. |
| LoRA mechanism (low-rank bypass Wx + BAx, B=0 initialization, merge at inference) | DEVELOPED | lora-finetuning (4.4.4, reinforced 6.5.1) | Deep knowledge of the bypass pattern, including its application to diffusion U-Net cross-attention projections. "Highway and detour" mental model. |
| LoRA placement in diffusion U-Net (cross-attention projections as primary target) | DEVELOPED | lora-finetuning (6.5.1) | Student understands that cross-attention is where text meets image features, making it the natural target for style/subject adaptation. |
| Textual inversion (pseudo-token embedding optimization, expressiveness boundaries) | DEVELOPED | textual-inversion (6.5.3) | Student understands embedding-level customization: adding one word to CLIP's vocabulary, 768 trainable params, limited to object identity not complex styles. |
| Block ordering at attention resolutions (residual block -> self-attention -> cross-attention at 16x16 and 32x32) | INTRODUCED | text-conditioning-and-guidance (6.3.4) | Student knows where cross-attention sits in the U-Net block and why only at middle resolutions (O(n^2) cost). |

### Established Mental Models and Analogies

- **"Same formula, different source for K and V"** -- cross-attention is self-attention with K/V projected from text embeddings instead of spatial features
- **"Three lenses, one embedding"** -- W_Q (seeking), W_K (advertising), W_V (contributing) as learned projections
- **"WHEN/WHAT/WHERE"** -- timestep via adaptive norm, text via cross-attention, ControlNet via additive encoder features
- **"Three translators, one pipeline" / "Four translators"** -- SD components as modular translators, extended with ControlNet
- **"Same detour, different highway"** -- LoRA as a reusable bypass pattern across architectures
- **"Inventing a new word in CLIP's language"** -- textual inversion as embedding-level concept injection
- **"Two encoders, one shared space"** -- CLIP's alignment comes from contrastive loss, not shared architecture
- **"Three knobs on three different parts of the pipeline"** -- customization spectrum: weights (LoRA), inference (img2img), embeddings (textual inversion)

### What Was Explicitly NOT Covered

- CLIP image encoder internals (ViT architecture MENTIONED only in 6.3.3)
- How to extract CLIP image embeddings in code (only text embeddings used in SD pipeline)
- Decoupled cross-attention as a concept (the module plan's core concept for this lesson)
- IP-Adapter specifically (mentioned in "What is NOT covered" of lessons 1 and 2)
- Image prompting or image-based conditioning beyond img2img
- Any adapter that modifies cross-attention at the K/V level (LoRA modifies weight matrices; textual inversion modifies embeddings)

### Readiness Assessment

The student is well-prepared for this lesson. Cross-attention is at DEVELOPED depth with deep Q/K/V intuition from six lessons in Module 4.2 and concrete diffusion application in 6.3.4. The CLIP shared embedding space is at DEVELOPED depth with the understanding that both image and text encoders produce vectors in the same geometric space. The student has seen three different ways to customize frozen SD models (LoRA, textual inversion, ControlNet) and can reason about where each intervenes in the pipeline. The key gap is that the student has never seen CLIP's image encoder produce embeddings that are fed into the U-Net's cross-attention -- this is the conceptual delta of this lesson, and it is small given the deep foundation.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student how IP-Adapter adds image-based semantic conditioning to a frozen Stable Diffusion model by injecting a parallel set of K/V projections for CLIP image embeddings alongside the existing text K/V projections in the U-Net's cross-attention layers.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Cross-attention mechanism (Q from spatial, K/V from conditioning) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Student must understand the Q/K/V formula deeply enough to reason about adding a second K/V source. DEVELOPED is sufficient. |
| Per-spatial-location cross-attention | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | IP-Adapter's image conditioning is spatially varying (each location attends to image features). Student already understands this for text. |
| CLIP shared embedding space | DEVELOPED | DEVELOPED | clip (6.3.3) | OK | Student must understand that CLIP image embeddings and text embeddings live in the same geometric space, which is what makes image conditioning meaningful. |
| CLIP dual-encoder architecture | INTRODUCED | INTRODUCED | clip (6.3.3) | OK | Student needs to know CLIP has a separate image encoder. INTRODUCED is sufficient -- we use it as a black box (encode image -> get embedding), same as text encoder in SD pipeline. |
| ControlNet architecture | INTRODUCED | DEVELOPED | controlnet (7.1.1) | OK | Comparison point: ControlNet adds WHERE (structural), IP-Adapter adds WHAT-IT-LOOKS-LIKE (semantic). Student's deep ControlNet knowledge enables rich comparison. |
| LoRA mechanism | INTRODUCED | DEVELOPED | lora-finetuning (4.4.4, 6.5.1) | OK | Comparison point for how adapters attach to cross-attention. LoRA modifies W_Q/W_K/W_V weight computation. IP-Adapter adds a parallel K/V path. Student's deep LoRA knowledge enables precise architectural comparison. |
| Textual inversion | INTRODUCED | DEVELOPED | textual-inversion (6.5.3) | OK | Comparison point for concept injection. Textual inversion operates at the embedding input to CLIP. IP-Adapter operates at the cross-attention output inside the U-Net. Different intervention points, different expressiveness. |
| Block ordering at attention resolutions | INTRODUCED | INTRODUCED | text-conditioning-and-guidance (6.3.4) | OK | Student needs to know where cross-attention sits in the U-Net to understand where IP-Adapter intervenes. INTRODUCED is sufficient. |
| CLIP image encoder output format | INTRODUCED | MENTIONED | clip (6.3.3) | GAP | Student knows CLIP has an image encoder producing a vector, but has only seen the single [1, 768] summary vector used for similarity. IP-Adapter uses the full sequence of image token embeddings (similar to how text conditioning uses [77, 768], not just the pooled vector). Brief recap needed. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| CLIP image encoder output format (sequence vs pooled vector) | Small (student knows CLIP image encoder exists and produces embeddings in the shared space; just needs to learn it produces a sequence, not just one vector) | Brief recap paragraph (2-3 sentences) within the core explanation section. Callback to the text-side parallel: "Remember that CLIP's text encoder produces a SEQUENCE of 77 token embeddings, not just a single summary vector. The same is true for the image encoder -- it produces a sequence of image patch embeddings (typically 257 tokens for ViT-H/14: 256 patches + 1 CLS token). IP-Adapter uses this sequence, not just the pooled CLS vector." This mirrors the recap at the start of 6.3.4 that clarified CLIP produces a sequence of text embeddings. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "IP-Adapter replaces the text prompt with an image prompt" | Student has seen image-based conditioning (img2img) that uses an image as the starting point, and may assume IP-Adapter similarly replaces text. The word "image prompting" suggests substitution. | Same reference image with two different text prompts producing clearly different outputs (e.g., reference image of a golden retriever + prompt "a painting of a dog in a garden" vs "a dog running on a beach"). Text still controls content and composition; the image provides semantic flavor. | Section 7 (Elaborate) -- dedicated WarningBlock after the core mechanism is established. Also foreshadowed in the hook by framing the problem as "adding" not "replacing." |
| "IP-Adapter is just LoRA trained on images (same mechanism, different training data)" | Student has deep LoRA knowledge and knows LoRA targets cross-attention. IP-Adapter also targets cross-attention. Natural assumption: same technique, different data. | Architecture comparison: LoRA modifies the EXISTING W_Q/W_K/W_V weight matrices (Wx + BAx changes the weight computation for every input). IP-Adapter adds SEPARATE K/V projection matrices that process a DIFFERENT input (image embeddings instead of text embeddings). LoRA changes how text is processed; IP-Adapter adds a second source of information. Concrete: remove LoRA and text conditioning changes. Remove IP-Adapter and text conditioning is unchanged -- the original K/V path is untouched. | Section 5 (Explain) -- ComparisonRow immediately after showing the architecture. Critical to address early because the student's LoRA depth makes this the highest-risk misconception. |
| "IP-Adapter needs to be trained per-concept (like textual inversion trains per-concept)" | Student has trained textual inversion embeddings for specific concepts. IP-Adapter also uses image conditioning. Natural assumption: one IP-Adapter per concept. | IP-Adapter is trained once on millions of image-text pairs (like CLIP). At inference, you feed ANY image. The K/V projections learned to extract general visual features, not one specific concept. Textual inversion: 1 concept = 1 trained embedding. IP-Adapter: 1 trained adapter = ANY image at inference. | Section 7 (Elaborate) -- after the student understands the training setup. |
| "IP-Adapter works by encoding the image with the VAE and feeding it into the denoising process (like img2img)" | Student has seen img2img, which encodes an image with the VAE and uses it as the denoising starting point. IP-Adapter also takes an image as input. Natural confusion between these two pathways. | IP-Adapter encodes the reference image with CLIP (semantic representation), not the VAE (pixel-level representation). The reference image never enters the denoising loop as a latent tensor. It enters via cross-attention as a semantic signal, not via the starting noise. Concrete: IP-Adapter + pure random noise starting point still produces output influenced by the reference image. Img2img + no text prompt produces output that structurally resembles the input image. Different pathways, different effects. | Section 4 (Hook/Explain) -- address early when introducing the CLIP image encoding step. |
| "The decoupled cross-attention output is just the average of text and image attention outputs" | Student understands weighted averages from attention. Simple mental model: add the two outputs together. | The outputs are added (not averaged), and each has its own learned W_K and W_V projections that were trained independently. The text path and image path produce different-shaped attention weight matrices (spatial_tokens x 77 for text vs spatial_tokens x 257 for image). They are not interchangeable or symmetric. The addition is weighted by a scale parameter, not fixed 50/50. | Section 5 (Explain) -- during the formula presentation. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Same reference image (golden retriever photo), two different text prompts ("a painting of a dog in a garden" and "a dog running on a beach at sunset") | Positive | Demonstrates that text and image conditioning coexist -- the reference image provides the dog's visual appearance while the text controls scene, style, and composition | Directly parallels the ControlNet coexistence example from lesson 1 (same edge map, two prompts). Student will recognize the pattern: image reference provides semantic identity (WHAT-IT-LOOKS-LIKE), text provides everything else. |
| Reference image of an ornate ceramic vase, prompt "a vase on a wooden table, photorealistic" at three IP-Adapter scales (0.0, 0.5, 1.0) | Positive | Demonstrates the scale parameter as a "volume knob" for image influence, directly paralleling the ControlNet conditioning scale from lesson 2 | Extends the "volume knob" analogy from ControlNet's conditioning scale to IP-Adapter's scale parameter. Student sees the same control pattern: 0 = image has no effect, 0.5 = blended, 1.0 = strong image influence. Familiar pattern, new application. |
| Attempt to use IP-Adapter for precise spatial control (reference image of a specific room layout, prompt "a living room") -- IP-Adapter captures the color palette and style but NOT the exact furniture positions | Negative | Demonstrates IP-Adapter's semantic (not structural) nature. IP-Adapter captures "what things look like" not "where things are." | Boundary-drawing example that distinguishes IP-Adapter from ControlNet. ControlNet with a depth map would preserve the room layout. IP-Adapter captures visual style, color palette, and mood but cannot control spatial arrangement. This is the critical distinction for the module. |
| Reference image of a sunset (no subject), prompt "a cat sitting in a field" | Positive (stretch) | Demonstrates that IP-Adapter can extract style/mood/color from a reference image without transferring subject matter | Shows the most powerful use case: style transfer without explicit style description in text. The sunset's warm colors and atmospheric quality influence the generation without the cat needing to be in a sunset scene. Demonstrates that CLIP image features capture more than just "what objects are present." |

---

## Phase 3: Design

### Narrative Arc

You have learned to add spatial control to Stable Diffusion with ControlNet -- edges, depth maps, and poses that tell the model WHERE to put things. But there is an entire category of visual intent that spatial maps cannot capture and text struggles to describe. Consider a reference photograph with a specific color palette, lighting mood, or material texture. You could try to describe it in words: "warm amber lighting with soft bokeh and muted earth tones." But that description is lossy -- it captures some of the feeling but misses the precise quality you want. And for a specific object (your cat, a particular ceramic vase, a company's product), text fails entirely. You cannot describe the exact visual identity of a specific object in 77 tokens.

The student already has all the pieces to solve this. CLIP's image encoder can extract semantic features from any photograph. CLIP's shared embedding space means image features and text features live in the same geometric space. Cross-attention already provides a mechanism for conditioning U-Net spatial features on external information. The question is: how do you feed image features into cross-attention WITHOUT replacing the text features that are already there? The answer -- decoupled cross-attention -- is an elegant, small architectural change that adds a parallel K/V pathway for image embeddings alongside the existing text K/V pathway.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual/Diagram | Mermaid architecture diagram showing the decoupled cross-attention block: spatial features -> W_Q (shared), text embeddings -> W_K_text/W_V_text (existing, frozen), image embeddings -> W_K_image/W_V_image (new, trainable). Two parallel attention computations merging via addition. Color-coded: gray for frozen, violet for new trainable components. | The core concept is an architectural change to cross-attention. A diagram showing the parallel K/V paths side by side is essential for understanding what "decoupled" means. The student has seen cross-attention data flow diagrams in 6.3.4 -- this extends that visual with a second branch. |
| Symbolic/Code | Pseudocode showing the decoupled cross-attention forward pass: `output = cross_attn(Q, K_text, V_text) + scale * cross_attn(Q, K_image, V_image)`. Side-by-side with the standard cross-attention forward pass showing the minimal change. | The student responds well to pseudocode (established pattern across the course). The one-line formula makes the architectural change concrete and shows that it is genuinely small. Parallels the "one-line change" framing from 6.3.4 (self-attention -> cross-attention). |
| Concrete example | Reference image of a golden retriever with two different text prompts, described with vivid output predictions (as in controlnet-in-practice). Scale parameter sweep (0.0, 0.5, 1.0) with a ceramic vase reference. | The student has been trained throughout the course to predict-before-verify. Concrete examples with vivid descriptions of expected outputs ground the architectural explanation in observable behavior. Parallels the conditioning scale sweep from lesson 2. |
| Intuitive/"of course" | "Of course" chain: (1) CLIP image embeddings capture visual semantics, (2) cross-attention reads from K/V embeddings, (3) adding a second K/V path for image embeddings is the natural way to feed image semantics into the U-Net, (4) keeping the text K/V path separate preserves text control. Each step follows from the previous. | The student has responded well to "of course" chains (ControlNet lesson 1 used this to great effect). This frames decoupled cross-attention as the obvious solution given the student's existing knowledge. |
| Verbal/Analogy | "Two reference documents" -- extend the "reading from a reference document" analogy from 4.2.5 and 6.3.4. Standard cross-attention: the U-Net reads from one reference document (text embeddings). Decoupled cross-attention: the U-Net reads from two reference documents simultaneously (text and image embeddings), each with its own translation layer (K/V projections), and combines what it reads. | Extends an established analogy that the student already has at depth. The "reference document" framing was used in Module 4.2 and 6.3.4. Adding a second document is a natural, low-cognitive-load extension. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2
  1. Decoupled cross-attention (adding parallel K/V projections for image embeddings alongside text K/V)
  2. IP-Adapter as a lightweight, general-purpose image conditioning adapter (trained once, works with any image)
- **Previous lesson load:** CONSOLIDATE (controlnet-in-practice -- 0 new concepts, practical skill-building)
- **Load trajectory:** STRETCH -> CONSOLIDATE -> BUILD. This follows the module plan. BUILD is appropriate: the conceptual delta is small (one additional K/V path in cross-attention) but the architectural insight is meaningful. The student's deep cross-attention knowledge means the "parallel K/V" idea is a clean extension, not a conceptual leap.

### Connections to Prior Concepts

| Prior Concept | How This Lesson Connects | Risk of Misleading? |
|---------------|-------------------------|---------------------|
| Cross-attention K/V from text (6.3.4) | IP-Adapter adds a SECOND set of K/V projections for image embeddings. Same Q, two K/V sources. "Same formula, different source for K and V" extends naturally to "same formula, TWO sources for K and V." | Low risk. The extension is clean and additive. |
| LoRA bypass on cross-attention (4.4.4, 6.5.1) | Both LoRA and IP-Adapter target cross-attention. Critical distinction: LoRA modifies the existing weight matrices (changes how text is processed). IP-Adapter adds new weight matrices (adds a new information source). | MODERATE RISK. Student's deep LoRA knowledge may lead them to assume IP-Adapter works the same way. Must address explicitly with ComparisonRow. |
| Textual inversion (6.5.3) | Both inject visual concepts into the generation process. Textual inversion operates at the INPUT to CLIP (embedding level, 768 params). IP-Adapter operates INSIDE the U-Net at cross-attention (projection level, ~22M params). Different intervention points, different expressiveness. | Low risk. The student already understands the "three knobs" customization spectrum. IP-Adapter is a fourth knob at a different location. |
| ControlNet additive architecture (7.1.1) | Both add capabilities to a frozen model. ControlNet adds WHERE (spatial/structural). IP-Adapter adds WHAT-IT-LOOKS-LIKE (semantic/visual identity). Both preserve text conditioning. Both are trained separately and plugged in. | Low risk. The comparison is clean and helps position IP-Adapter in the module's controllable generation taxonomy. |
| CLIP shared embedding space (6.3.3) | IP-Adapter relies on CLIP image embeddings living in the same geometric space as text embeddings. This is what makes it possible for the U-Net's cross-attention to process image features through a parallel K/V path -- the features are semantically meaningful in the same way text features are. | Low risk. Deepens the student's appreciation of why the shared space matters. |
| "Nothing, Then a Whisper" / zero conv initialization (7.1.1) | IP-Adapter's new K/V projections are initialized to produce zero output (similar principle to zero convolution). At the start of training, the adapter contributes nothing. As training progresses, image conditioning gradually fades in. Same safety pattern, different mechanism. | Low risk. Reinforces the "safe initialization for frozen models" pattern. |
| Conditioning scale as volume knob (7.1.2) | IP-Adapter has its own scale parameter controlling image influence strength, directly paralleling ControlNet's conditioning scale. Same user-facing concept. | No risk. Clean parallel. |

### Scope Boundaries

**This lesson IS about:**
- The decoupled cross-attention architecture (parallel K/V projections for image embeddings)
- Why CLIP image embeddings are the right conditioning signal (not VAE latents)
- How IP-Adapter preserves text conditioning (decoupling, not replacement)
- IP-Adapter's scale parameter for controlling image influence
- How IP-Adapter compares to LoRA, textual inversion, and ControlNet in the customization/control taxonomy
- IP-Adapter as a general-purpose adapter (trained once, works with any image)

**This lesson is NOT about:**
- IP-Adapter training procedure in detail (briefly mentioned for understanding, not the focus)
- Implementing IP-Adapter from scratch
- IP-Adapter Plus or Face ID variants (MENTION for vocabulary breadth)
- Combining IP-Adapter with ControlNet simultaneously (MENTION as a natural extension, not developed)
- CLIP image encoder internals (ViT architecture -- used as black box)
- Style transfer methods beyond IP-Adapter (NST, style LoRA comparison is sufficient context)
- IP-Adapter for SDXL or other model variants
- Attention processor / custom attention implementations in diffusers

**Target depth for core concept (decoupled cross-attention):** DEVELOPED -- the student should be able to explain the architecture, trace the data flow, reason about why the parallel path preserves text conditioning, and predict behavior at different scale values.

### Lesson Outline

#### 1. Context + Constraints
What this lesson covers: how IP-Adapter adds image-based semantic conditioning to a frozen SD model via decoupled cross-attention. What we are NOT doing: implementing from scratch, training, or covering every variant. This is the final lesson in Module 7.1, completing the controllable generation story: ControlNet (WHERE) + IP-Adapter (WHAT-IT-LOOKS-LIKE) + text (WHAT).

#### 2. Recap (brief)
Reactivate two concepts needed for the lesson:
- **Cross-attention K/V mechanism:** 2-3 sentences. "Q from spatial features, K/V from text embeddings. Each spatial location attends independently to text tokens. This is the WHAT channel." Callback to 6.3.4 and the WHEN/WHAT/WHERE framing from 7.1.1.
- **CLIP image encoder output:** Brief gap fill. Student knows CLIP has two encoders in a shared space, but has only seen the pooled [1, 768] summary vector. Clarify that the image encoder (ViT) also produces a SEQUENCE of patch token embeddings (e.g., [1, 257, 1024] for ViT-H/14), similar to how the text encoder produces [1, 77, 768]. IP-Adapter projects these image tokens into the U-Net's expected dimensions via a small trainable projection.

#### 3. Hook: The Description Problem
Type: **before/after challenge**
Motivating problem: Show a reference photograph with distinctive visual qualities (specific lighting, color palette, material textures). Challenge: "Describe this image precisely enough that Stable Diffusion would reproduce its visual character." The student tries and realizes that text is lossy for precise visual identity. Then: "What if you could just SHOW the model the image?" The student already has the pieces (CLIP image encoder, cross-attention, shared embedding space). Challenge: "You know CLIP can encode images into the same space as text. You know cross-attention reads from K/V embeddings. How would you feed image features into cross-attention without disrupting the text path?"

This parallels the ControlNet hook from lesson 1 where the student was challenged to design the solution before being told. The student should be able to sketch the answer: add a second K/V path for image embeddings.

#### 4. Explain: Decoupled Cross-Attention
Core concept presentation with planned modalities:

**"Of course" chain:**
1. CLIP's image encoder produces embeddings in the shared text-image space (student knows this from 6.3.3)
2. Cross-attention's K/V projections translate conditioning embeddings into the U-Net's internal language (student knows this from 6.3.4)
3. Adding a SEPARATE set of K/V projections for image embeddings feeds image semantics into the U-Net without touching the text path
4. The two attention outputs are added: `output = text_attn + scale * image_attn`
5. The Q projection is shared -- spatial features ask the same questions of both text and image references

**Architecture diagram (Mermaid):**
Show the standard cross-attention block (frozen, gray) alongside the new IP-Adapter branch (trainable, violet). Shared Q, two parallel K/V paths, addition at the output. Color-coded consistently with prior lessons.

**Pseudocode:**
```python
# Standard cross-attention (frozen)
Q = W_Q(spatial_features)       # shared query
K_text = W_K_text(text_emb)     # existing frozen K
V_text = W_V_text(text_emb)     # existing frozen V
text_out = attention(Q, K_text, V_text)

# IP-Adapter branch (trainable)
K_image = W_K_image(image_emb)  # NEW trainable K
V_image = W_V_image(image_emb)  # NEW trainable V
image_out = attention(Q, K_image, V_image)

# Decoupled output
output = text_out + scale * image_out
```

**Shape walkthrough:**
- Q: [n_spatial, d_k] (e.g., [256, 320] at 16x16 resolution) -- from spatial features, SHARED
- K_text, V_text: [77, d_k] -- from CLIP text encoder, FROZEN projections
- K_image, V_image: [257, d_k] -- from CLIP image encoder + trainable projection, NEW
- text_out: [n_spatial, d_v] -- text attention output
- image_out: [n_spatial, d_v] -- image attention output
- Attention weights for text: [n_spatial, 77] -- same as standard SD
- Attention weights for image: [n_spatial, 257] -- new, rectangular, not square

**Key insight:** The text attention path is completely untouched. No frozen weights are modified. The image path is purely additive, with its own learned projections. This is why text conditioning is preserved.

#### 5. Check #1 (predict-and-verify)
Three questions:
1. "If you set the IP-Adapter scale to 0, what happens?" (Image branch contributes nothing; output is identical to standard SD with text only. Same principle as ControlNet at scale=0.)
2. "If you remove the IP-Adapter entirely (delete the new W_K_image and W_V_image), does the frozen model change?" (No -- IP-Adapter is purely additive. Bit-for-bit identical without it. Same principle as ControlNet disconnect test from lesson 1.)
3. "Why does IP-Adapter share the Q projection instead of having its own W_Q_image?" (Spatial features determine what each location is "seeking." The same spatial location should ask the same question of both text and image -- "what should I look like here?" The answer comes from different sources, but the question is the same.)

#### 6. Explore: IP-Adapter in Practice
Practical demonstration via notebook:
- Load IP-Adapter with diffusers
- Reference image of a golden retriever + two different text prompts -- text controls composition, reference controls visual identity
- Scale parameter sweep (0.0, 0.5, 1.0) with vivid descriptions of expected output at each level
- Connect scale parameter to the "volume knob" pattern from ControlNet lesson 2

#### 7. Elaborate: Comparisons, Boundaries, and Misconceptions

**IP-Adapter vs LoRA (ComparisonRow -- highest-priority comparison):**
- LoRA: modifies EXISTING W_K, W_V weight matrices. Changes how text is processed for ALL prompts. Trained on a specific concept/style. Baked into the model.
- IP-Adapter: adds NEW W_K_image, W_V_image matrices. Adds a new information source. Trained once, works with ANY reference image at inference. The text path is untouched.
- Key test: remove LoRA and text-to-image output changes (because the text K/V projections have been modified). Remove IP-Adapter and text-to-image output is identical (because text K/V projections were never touched).

**IP-Adapter vs Textual Inversion (ComparisonRow):**
- Textual inversion: 768 trainable params. Operates at the INPUT to CLIP (embedding lookup table). One embedding per concept. Limited to object identity.
- IP-Adapter: ~22M trainable params across all cross-attention layers. Operates INSIDE the U-Net at cross-attention. General-purpose -- works with any image at inference.
- Shared principle: both preserve the frozen model. Both are purely additive.

**IP-Adapter vs ControlNet (ComparisonRow):**
- ControlNet: structural/spatial control (WHERE). Adds features via the encoder copy + zero conv. Input: spatial maps (edges, depth, pose).
- IP-Adapter: semantic/visual identity control (WHAT-IT-LOOKS-LIKE). Adds K/V via decoupled cross-attention. Input: CLIP image embeddings.
- Together: ControlNet provides structure, IP-Adapter provides visual identity, text provides additional specificity. All three are additive and composable.

**Negative example (spatial control boundary):**
Reference image of a specific room layout. IP-Adapter captures color palette, lighting, and furniture style but NOT the exact positions. ControlNet with a depth map would preserve the layout. IP-Adapter is semantic, not spatial.

**Misconception #1 address (WarningBlock: "Image Prompting Is Not Image Replacement"):**
Same reference image + two different text prompts producing different outputs. The image provides semantic flavor; text still controls content and composition.

**Misconception #3 address (general-purpose, not per-concept):**
IP-Adapter is trained once on millions of image-text pairs. At inference, feed ANY image. Contrast with textual inversion (one training per concept) and LoRA (one training per style/subject).

**Training overview (brief, for understanding):**
- Training data: large image-text pair dataset (same kind of data CLIP was trained on)
- Loss: same DDPM noise prediction loss as standard diffusion training
- What is frozen: entire SD model (U-Net, CLIP, VAE)
- What is trained: the new W_K_image and W_V_image projections at every cross-attention layer + a small image projection network
- Initialization: new projections initialized to zero output (callback to zero convolution pattern from lesson 1 and LoRA B=0 from 4.4.4)
- Result: ~22M trainable parameters (much smaller than ControlNet's ~300M, but larger than LoRA's ~1M or textual inversion's 768)

**Variants (MENTION for vocabulary breadth):**
- IP-Adapter Plus: uses more CLIP image features for higher fidelity
- IP-Adapter Face ID: specialized for face identity preservation
- IP-Adapter + ControlNet: composable -- IP-Adapter provides "what it looks like" while ControlNet provides "where things go"

#### 8. Check #2 (transfer questions)
1. "A colleague wants to generate images that match a specific painting's color palette and brushstroke style. They are deciding between LoRA and IP-Adapter. What would you recommend and why?" (IP-Adapter: no training needed, just provide the painting as reference. LoRA would require collecting training images and running training. If they need this specific style permanently for thousands of generations, LoRA might be worth the training investment.)
2. "Could you use IP-Adapter with a photograph as the reference image AND a ControlNet with an edge map from a completely different image?" (Yes -- IP-Adapter provides semantic/visual identity from the reference photo via cross-attention, ControlNet provides spatial structure from the edge map via encoder features. They target different parts of the U-Net and are composable.)

#### 9. Practice: Notebook Exercises (Colab)
Four exercises, cumulative progression:

**Exercise 1 (Guided):** Load IP-Adapter and generate with a reference image + text prompt. Compare output with and without IP-Adapter (scale=0 vs scale=0.6). Observe that text still controls composition.

**Exercise 2 (Guided):** Scale parameter sweep. Same reference image, same prompt, five scale values (0.0, 0.3, 0.5, 0.7, 1.0). Observe the transition from text-dominant to image-dominant. Connect to conditioning scale pattern from ControlNet lesson 2.

**Exercise 3 (Supported):** Same reference image + three different text prompts. Observe that text controls content/scene while reference image controls visual character. Addresses misconception #1 (image replaces text).

**Exercise 4 (Independent):** Combine IP-Adapter with ControlNet. Use a reference image for visual style (IP-Adapter) and an edge map from a DIFFERENT image for spatial structure (ControlNet). The student designs their own experiment and interprets the results.

Exercises are cumulative: Exercise 1's reference image carries through. Each exercise tests a specific concept at a specific depth:
- Ex 1: IP-Adapter basic functionality (IP-Adapter scale at INTRODUCED)
- Ex 2: Scale parameter behavior (volume knob pattern at DEVELOPED)
- Ex 3: Text-image coexistence (decoupled cross-attention at DEVELOPED)
- Ex 4: Composability with ControlNet (module-level synthesis at APPLIED)

#### 10. Summarize
Key takeaways echoing the mental models:
- IP-Adapter adds a parallel K/V path for CLIP image embeddings alongside the existing text K/V path -- "two reference documents, one reader"
- The text path is completely untouched -- decoupled, not replaced
- IP-Adapter provides WHAT-IT-LOOKS-LIKE control; ControlNet provides WHERE control; text provides WHAT. Three complementary conditioning channels.
- The controllable generation spectrum is now complete: text (WHAT) + ControlNet (WHERE) + IP-Adapter (WHAT-IT-LOOKS-LIKE) + timestep (WHEN)
- Same frozen-model + additive-adapter pattern: zero initialization, purely additive, disconnect and nothing changes

#### 11. Next Step
Module 7.1 is complete. The student has learned three controllable generation techniques that compose with a frozen SD model: ControlNet for spatial structure, IP-Adapter for visual identity, and text for semantic content. Preview the next module in Series 7 (SDXL, consistency models, or flow matching, depending on the series plan).

ModuleCompleteBlock listing the three lessons and what was achieved.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (CLIP image encoder sequence output is a small gap with resolution plan)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept (visual/diagram, symbolic/code, concrete example, intuitive/"of course", verbal/analogy = 5 modalities)
- [x] At least 2 positive examples + 1 negative example (3 positive + 1 negative), each with stated purpose
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: MAJOR REVISION

One critical finding (missing notebook) and three improvement findings require a revision pass. The lesson component itself is strong--the critical finding is an artifact gap, not a pedagogical flaw in the built .tsx. However, the notebook is integral to the lesson's exercise progression and must exist before the lesson can ship.

### Findings

#### [CRITICAL] -- Notebook missing

**Location:** `notebooks/7-1-3-ip-adapter.ipynb` (expected path)
**Issue:** The planning document specifies 4 exercises (Guided -> Guided -> Supported -> Independent) and the lesson component references a Colab URL for `notebooks/7-1-3-ip-adapter.ipynb`, but no notebook file exists at that path. The lesson's Practice section (lines 1086-1157) describes all four exercises and links to Colab, but the student would click the link and get a 404.
**Student impact:** Student reaches the Practice section, clicks the Colab link, and gets nothing. The lesson's exercise progression--which is designed to move decoupled cross-attention from INTRODUCED to DEVELOPED and test composability at APPLIED--is entirely missing. Without the notebook, the lesson teaches but does not practice.
**Suggested fix:** Build the notebook following the planning document's Exercise specifications (Section 9). Ensure: (1) first code cell installs dependencies and imports, (2) random seeds for reproducibility, (3) Exercise 1 is genuinely predict-before-run, (4) Exercises 3 and 4 have `<details>` solution blocks, (5) same reference image carries through all exercises, (6) terminology matches the lesson component exactly.

#### [IMPROVEMENT] -- Section ordering deviates from plan: comparisons before practical examples

**Location:** IP-Adapter vs LoRA section (lines 556-616) appears before IP-Adapter in Practice section (lines 618-744)
**Issue:** The planning document specifies Section 6 (Explore: IP-Adapter in Practice--golden retriever examples, scale sweep) BEFORE Section 7 (Elaborate: Comparisons--LoRA, textual inversion, ControlNet). The built lesson reverses this: the LoRA comparison comes after Check #1 but before the concrete practice examples. This violates the "concrete before abstract" ordering principle. The student sees an architectural comparison (IP-Adapter vs LoRA mechanism) before seeing IP-Adapter used with a real reference image and text prompt.
**Student impact:** The student understands the mechanism from the diagram and pseudocode, then is immediately asked to compare it to LoRA at an architectural level. They have not yet "experienced" IP-Adapter through concrete examples (golden retriever, vase). The comparison would land more powerfully after the student has seen IP-Adapter produce results.
**Suggested fix:** Move the "IP-Adapter vs LoRA" section (lines 556-616) to after the "IP-Adapter in Practice" and "The Scale Parameter" sections (lines 618-744). This restores the plan's ordering: explain -> check -> see it in action -> compare to other tools.

#### [IMPROVEMENT] -- Misconception #5 (average vs weighted addition) lacks explicit address

**Location:** Shape walkthrough section (lines 389-485) and the formula `text_out + scale * image_out`
**Issue:** The planning document identifies misconception #5: "The decoupled cross-attention output is just the average of text and image attention outputs." The plan specifies addressing this "during the formula presentation" with a concrete negative example showing different attention weight shapes (spatial_tokens x 77 vs spatial_tokens x 257). The built lesson shows the formula and the shape table, and the aside (line 477) mentions different token counts, but there is no explicit WarningBlock or negative example that says "this is NOT averaging." The student must infer from the formula that it is weighted addition, not averaging. Given that the other four misconceptions each get their own explicit WarningBlock, the asymmetry is noticeable.
**Student impact:** A student who mentally models the combination as "50/50 averaging of text and image influence" would not be corrected. The formula `text_out + scale * image_out` does show it is not averaging (it is addition with a scale parameter), but the student might still think scale=0.5 means "equal weight to both" without understanding that the text contribution has no scale factor (implicit scale=1.0). The asymmetry--text always contributes fully, image is scaled--is not explicitly called out.
**Suggested fix:** Add a brief WarningBlock after the formula or shape walkthrough, titled something like "Addition, Not Averaging." Two key points: (1) the text path contributes at full strength always (no scale on text_out), (2) the scale parameter controls only the image contribution, so scale=0.5 does not mean "half text, half image"--it means "full text, plus half-strength image." This is a small addition that prevents a subtle but consequential misunderstanding.

#### [IMPROVEMENT] -- "Not Img2Img" misconception placed before core explanation

**Location:** WarningBlock "Not Img2Img" (lines 235-250), appears between the Design Challenge and the Decoupled Cross-Attention explanation
**Issue:** The plan specifies addressing this misconception in "Section 4 (Hook/Explain)--address early when introducing the CLIP image encoding step." The built lesson places it AFTER the Design Challenge reveal but BEFORE the formal explanation of decoupled cross-attention. At this point, the student has seen the one-sentence reveal ("add a second set of K/V projections for image embeddings") but has not yet seen the architecture diagram, pseudocode, or shape walkthrough. The WarningBlock says "IP-Adapter encodes the reference image with CLIP (semantic representation, not pixel-level) and injects it via cross-attention as a semantic signal"--but the student has not yet been shown how this injection works. They are being told what IP-Adapter does NOT do (use VAE) before being shown what it DOES do (decoupled cross-attention).
**Student impact:** The student gets a "not this" before "here's what it actually is." The WarningBlock references cross-attention injection that has not yet been explained. The student must hold the "not img2img" correction in memory through the entire explanation section before it fully makes sense. This is a sequencing issue that increases cognitive load during the most important section.
**Suggested fix:** Move the "Not Img2Img" WarningBlock to immediately AFTER the pseudocode (line 386) or after the shape walkthrough (line 485). At that point, the student has seen exactly how IP-Adapter works (CLIP -> K/V projections -> attention -> addition). The contrast with img2img (VAE -> starting latent) will be sharper because the student can mentally compare the two pathways.

#### [POLISH] -- Spaced em dashes in CodeBlock comments

**Location:** Lines 361 and 367 in the CodeBlock
**Issue:** The Python comments contain spaced em dashes: `# Standard cross-attention (frozen — unchanged from vanilla SD)` and `# IP-Adapter branch (trainable — the only new thing)`. The writing style rule requires no spaces around em dashes.
**Student impact:** Negligible--these are code comments. But they are visible to the student and inconsistent with the lesson's prose style.
**Suggested fix:** Change to unspaced em dashes: `(frozen—unchanged from vanilla SD)` and `(trainable—the only new thing)`.

#### [POLISH] -- Image projection network mentioned in recap but not elaborated

**Location:** Recap section, CLIP image encoder paragraph (line 122-127)
**Issue:** The recap mentions "257 tokens for ViT-H/14: 256 patches + 1 CLS token" but the planning document also specifies mentioning "a small trainable projection network that adapts CLIP image features to the U-Net's expected dimensions." The training overview section (line 980) mentions this projection network, but the recap section where the student first encounters the CLIP image sequence does not mention that the raw CLIP image embeddings (e.g., 1024-dim for ViT-H/14) need to be projected to match the U-Net's cross-attention dimension (e.g., 320 or 768). The shape walkthrough table lists K_image/V_image as [257, d_k] with source "CLIP image encoder + trainable projection"--but the recap does not prepare the student for this projection step.
**Student impact:** Minor. The student might wonder how 1024-dim CLIP image embeddings become d_k-dimensional K/V vectors, but the shape table's "trainable projection" note and the training section's mention of "a small image projection network" cover it. The gap is just a matter of when the student learns about the projection step--later rather than sooner.
**Suggested fix:** Add one sentence to the recap's CLIP image encoder paragraph: "IP-Adapter includes a small trainable projection that maps these image patch embeddings to the U-Net's expected cross-attention dimensions." This primes the student for the shape walkthrough.

### Review Notes

**What works well:**
- The "of course" chain is genuinely strong. Each step follows from the previous, and the student's deep cross-attention knowledge makes this feel like a natural extension rather than a new concept.
- The Design Challenge paralleling the ControlNet lesson is excellent pedagogy--it reinforces the pattern of "you have the pieces to solve this yourself."
- The LoRA comparison with the "remove and check" test is the best comparison in the lesson. It makes the architectural difference concrete and testable.
- Five modalities for the core concept (diagram, pseudocode, examples, "of course" chain, analogy) is thorough.
- The module completion framing (WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE) is a satisfying synthesis.
- Scope boundaries are well-maintained--the lesson does not drift.

**Pattern to watch:**
- The ordering deviation (comparisons before examples) is a recurring temptation: it feels natural to compare immediately after explaining. But the plan's ordering (see it work, then compare) better serves the "concrete before abstract" principle. Future lessons should follow the plan's ordering unless there is an explicit reason to deviate.

**The critical finding (missing notebook) is a build artifact gap, not a pedagogical design problem.** The lesson component itself is well-built. Once the notebook is created and the three improvement findings are addressed, this lesson should pass on the next review iteration.

---

## Review -- 2026-02-20 (Iteration 2/3)

### Iteration 1 Findings -- Verification

All 6 findings from Iteration 1 have been addressed:

1. **[CRITICAL] Notebook missing** -- RESOLVED. `notebooks/7-1-3-ip-adapter.ipynb` now exists with 4 exercises (Guided -> Guided -> Supported -> Independent). First cell installs dependencies and imports. Random seeds set. Reference image carries through all exercises. Exercises 3 and 4 have `<details>` solution blocks. Terminology matches the lesson component.

2. **[IMPROVEMENT] Section ordering (comparisons before examples)** -- RESOLVED. The lesson now follows the plan's ordering: Section 7 (IP-Adapter in Practice: golden retriever, scale sweep) comes BEFORE Section 8 (IP-Adapter vs LoRA comparison) and Section 9 (Elaborate: more comparisons). Concrete examples precede abstract comparisons.

3. **[IMPROVEMENT] Misconception #5 (average vs weighted addition)** -- RESOLVED. A dedicated WarningBlock titled "Addition, Not Averaging" now appears at lines 489-507, immediately after the shape walkthrough. It explicitly states: (1) the text path contributes at full strength always (no scale on text_out), (2) scale=0.5 means "full text, plus half-strength image," not "half and half," (3) the two paths produce different-shaped attention weight matrices and are not symmetric.

4. **[IMPROVEMENT] "Not Img2Img" WarningBlock placed before core explanation** -- RESOLVED. The "Not Img2Img" WarningBlock now appears at lines 472-487, after the architecture diagram, pseudocode, and shape walkthrough. The student has seen how IP-Adapter works (CLIP -> K/V projections -> attention -> addition) before being told what it is NOT (VAE -> starting latent).

5. **[POLISH] Spaced em dashes in CodeBlock comments** -- RESOLVED. The Python code block comments now use unspaced em dashes: `(frozen—unchanged from vanilla SD)` and `(trainable—the only new thing)`.

6. **[POLISH] Image projection mentioned in recap** -- RESOLVED. The recap's CLIP image encoder paragraph (lines 126-128) now includes: "A small trainable projection maps these image patch embeddings to the U-Net's expected cross-attention dimensions before they are fed into the K/V projections." This primes the student for the shape walkthrough.

### Fresh Review

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

The lesson is well-built and ready to ship. All iteration 1 findings have been addressed. The fresh review found only two minor polish items, neither of which affects student learning or warrants another revision cycle.

### Findings

#### [POLISH] -- Spaced em dashes in JSX comments

**Location:** Lines 43, 99, 143, 237, 509, 578, 768, 1032, 1049, 1108 (JSX comments like `{/* Section 3: Recap — Cross-Attention */}`)
**Issue:** JSX comments use spaced em dashes (e.g., `Recap — Cross-Attention`). These are not student-facing (they are code comments stripped at build time), but they are inconsistent with the writing style rule applied everywhere else in the lesson.
**Student impact:** Zero. These comments never render.
**Suggested fix:** Replace with unspaced em dashes in JSX comments for codebase consistency. No urgency.

#### [POLISH] -- Notebook Exercise 3 uses `pass` placeholder without guard

**Location:** Notebook cell-15, the `for p in prompts:` loop
**Issue:** The Supported exercise asks the student to fill in the `pass` placeholder with actual generation code. The next display cell (cell-16) checks `if not coexistence_results:` and prints a helpful message. This is good scaffolding. However, if the student partially fills in the loop (e.g., generates for some prompts but makes an error for others), the `coexistence_results` dict will be non-empty but missing keys, causing a KeyError in the display cell's `zip(prompts, short_labels)` iteration rather than a helpful error message.
**Student impact:** Minor. A student who gets 1 of 3 prompts working but errors on the rest would see a confusing KeyError instead of a useful message. This is an edge case -- most students will either complete the loop or leave it as `pass`.
**Suggested fix:** In cell-16, add a guard: `if len(coexistence_results) != len(prompts):` with a message like "Not all prompts generated -- check for errors in the loop above." Low priority.

### Review Notes

**What works well (carried forward and confirmed):**
- The "of course" chain remains the lesson's strongest moment. Each step follows naturally from the student's existing knowledge, and the shared Q insight (step 5) produces a genuine "of course" feeling.
- The Design Challenge paralleling ControlNet lesson 1 is excellent--same pedagogical pattern (you have the pieces, design the solution), reinforcing the student's agency.
- The LoRA comparison with the "remove and check" test is now even stronger in its new position (after concrete examples). The student has seen IP-Adapter work before being asked to compare it architecturally.
- Section ordering now follows concrete-before-abstract throughout: explain -> check -> see it in practice -> compare to other tools -> elaborate.
- The misconception coverage is thorough: all 5 planned misconceptions have dedicated WarningBlocks with concrete reasoning.
- The notebook is well-constructed: predict-before-run pattern in Guided exercises, proper cleanup between exercises for VRAM management, same reference image throughout, `<details>` solutions with reasoning (not just code dumps), and terminology matches the lesson exactly.
- The module completion framing (WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE with four additive channels) provides satisfying synthesis.
- Scope boundaries are maintained--the lesson does not drift into implementation details, training procedures, or variant architectures.

**Modality count for core concept (decoupled cross-attention):** 5 modalities present (visual/Mermaid diagram, symbolic/pseudocode, concrete examples/golden retriever + vase, intuitive/"of course" chain, verbal/analogy "two reference documents"). Exceeds the 3-modality minimum.

**Example count:** 3 positive examples (golden retriever two-prompt, vase scale sweep, sunset style transfer) + 1 negative example (room layout spatial control boundary) + 1 stretch example (sunset for style without words). Exceeds the 2+1 minimum.

**Notebook alignment:** All 4 exercises match the planning document's specifications. Exercise progression (Guided -> Guided -> Supported -> Independent) is correct. Exercise 1 starts with the simplest possible test (scale=0 vs scale=0.6). Exercise 4 is the only Independent exercise and comes at the end. The notebook introduces no new concepts--it practices what the lesson teaches.
