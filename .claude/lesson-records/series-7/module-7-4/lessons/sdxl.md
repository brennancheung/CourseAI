# Lesson: SDXL

**Slug:** `sdxl`
**Series:** 7 (Post-SD Advances), **Module:** 7.4 (Next-Generation Architectures), **Position:** Lesson 9 of 11 in series, Lesson 1 of 3 in module
**Cognitive Load:** BUILD (2-3 new concepts that extend familiar patterns: dual text encoders, micro-conditioning, refiner model)

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Full SD v1.5 pipeline data flow (text -> CLIP -> U-Net denoising loop with CFG -> VAE decode, with tensor shapes at every handoff) | DEVELOPED | stable-diffusion-architecture (6.4.1) | The student traced the complete pipeline with "a cat sitting on a beach at sunset." Tensor shapes at every stage: [77] tokens -> [77, 768] embeddings -> z_T [4, 64, 64] -> 50 steps -> z_0 -> VAE decode -> [3, 512, 512]. Mermaid pipeline diagram and annotated pseudocode. |
| CLIP dual-encoder architecture (separate image encoder + text encoder, shared embedding space, contrastive learning) | DEVELOPED | clip (6.3.3) | The student understands CLIP's training setup (contrastive learning on 400M pairs), dual encoders with no shared weights, shared embedding space for cross-modal retrieval. Specifically: text encoder produces [77, 768] contextual embeddings. The student knows CLIP limitations (typographic attacks, spatial reasoning, counting). |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings, per-spatial-location conditioning) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | The student can trace Q/K/V through cross-attention, knows the shape (e.g., [256, 77] attention weights at 16x16), understands per-spatial-location text conditioning. Reactivated in Module 7.1 (ControlNet, IP-Adapter). |
| Classifier-free guidance (training with text dropout, two forward passes, epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | The student knows the CFG formula, worked through a concrete numerical example, understands the guidance scale tradeoff (w=1 no amplification, w=7.5 typical, w=20 oversaturated). Negative prompts as CFG application from 6.4.1. |
| Component modularity (CLIP, U-Net, VAE independently trained, connected by tensor handoffs) | INTRODUCED | stable-diffusion-architecture (6.4.1) | "Three translators, one pipeline." The student knows components can be swapped independently. Parameter counts: CLIP ~123M, U-Net ~860M, VAE ~84M. |
| Img2img as partial denoising (VAE encode input, noise to t_start, denoise from t_start to t=0; strength parameter maps to starting timestep) | DEVELOPED | img2img-and-inpainting (6.5.2) | The student implemented img2img from scratch in the notebook. Understands strength parameter, nonlinear alpha-bar mapping, "starting mid-hike" analogy. The refiner model uses this exact mechanism. |
| U-Net encoder-decoder architecture (encoder downsampling, bottleneck, decoder upsampling, skip connections) | DEVELOPED | unet-architecture (6.3.1) | "Bottleneck decides WHAT, skip connections decide WHERE." The student traced dimensions through the architecture (64x64x64 -> 32x32x128 -> 16x16x256 -> 8x8x512 -> back up). Knows SD v1.5 channel progression: 320 -> 640 -> 1280 -> 1280. |
| Adaptive group normalization for timestep conditioning (timestep-dependent gamma/beta, global conditioning at every processing stage) | DEVELOPED | conditioning-the-unet (6.3.2) | The student knows the formula AdaGN(x, t) = gamma(t) * Normalize(x) + beta(t), the per-block linear projection pattern, and the distinction between global conditioning (timestep) and spatially-varying conditioning (text). |
| Latent diffusion (encode with VAE, diffuse in latent space [4, 64, 64], decode back to pixels; 48x compression from 512x512x3) | DEVELOPED | from-pixels-to-latents (6.3.5) | "Same orchestra, smaller hall." The student understands frozen-VAE pattern, the 8x spatial downsampling, why diffusion in latent space is computationally tractable. |
| LoRA fine-tuning for diffusion (cross-attention projections as primary target, Wx + BAx bypass, 4-50 MB files) | DEVELOPED | lora-finetuning (6.5.1) | "Same detour, different highway." The student trained a style LoRA in the notebook. Understands placement, rank, composition, and why cross-attention is the target. |
| Four conditioning channels in SD (WHEN via timestep/adaptive norm, WHAT via text/cross-attention, WHERE via ControlNet, WHAT-IT-LOOKS-LIKE via IP-Adapter) | DEVELOPED | ip-adapter (7.1.3) | Module 7.1 synthesis. All four channels are additive, composable, and target different parts of the U-Net. |
| Flow matching (straight-line interpolation, velocity prediction, fewer ODE steps needed) | DEVELOPED | flow-matching (7.2.2) | The student trained a 2D flow matching model. Understands curved vs straight trajectories, velocity parameterization, and why flow matching enables fewer steps. Knows SD3/Flux use flow matching (INTRODUCED in 7.2.2). |
| Vision Transformer / ViT (transformer on image patches) | MENTIONED | clip (6.3.3) | Named as one of CLIP's image encoder options. 2-sentence ConceptBlock. The student knows ViT exists and processes patches like tokens but has not seen it developed. |
| Speed landscape taxonomy (three levels: better solvers, straighter paths, trajectory bypass) | DEVELOPED | the-speed-landscape (7.3.3) | The student has a comprehensive decision framework for acceleration approaches. SDXL Turbo was discussed at DEVELOPED depth in 7.3.2. |
| SDXL Turbo / adversarial diffusion distillation | DEVELOPED | latent-consistency-and-turbo (7.3.2) | The student understands ADD (hybrid loss: diffusion + adversarial), knows SDXL Turbo is a specific SDXL-based model, discussed in the speed landscape. But the base SDXL architecture was NOT covered--only the distilled variant. |

### Mental Models and Analogies Already Established

- **"Three translators, one pipeline"** -- CLIP, U-Net, VAE as independently trained, modular components connected by tensor handoffs.
- **"Bottleneck decides WHAT, skip connections decide WHERE"** -- U-Net dual-path information flow.
- **"WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE"** -- four conditioning dimensions, each targeting a different U-Net mechanism.
- **"Volume knob"** -- conditioning scale, guidance scale, IP-Adapter scale as adjustable influence dials.
- **"Same detour, different highway"** -- LoRA bypass applied to different contexts (LLM, diffusion, speed adapter).
- **"Starting mid-hike"** -- img2img as partial denoising from an intermediate noise level.
- **"Menu, not upgrade path"** -- acceleration approaches as options with tradeoffs, not a linear progression.
- **"Curved vs straight"** -- flow matching's core geometric insight about trajectory shape.
- **"Of Course" chain** -- design insights that feel inevitable given the right framework. Used in ControlNet, IP-Adapter, flow matching.

### What Was Explicitly NOT Covered

- SDXL's base architecture (dual text encoders, refiner model, micro-conditioning, higher resolution, larger U-Net) -- SDXL was mentioned as a base model for SDXL Turbo and LCM, and ControlNet/LoRA compatibility notes referenced SDXL, but the architectural innovations were never taught.
- Multi-encoder text conditioning (the student only knows single-CLIP-encoder conditioning from SD v1.5).
- Micro-conditioning or resolution-conditional generation (no prior lesson addresses conditioning on image size or crop parameters).
- The refiner pipeline as a specialized second denoising pass (img2img mechanism is known but the two-model pipeline is new).
- Training on multiple resolutions and the artifacts this produces (the student assumes models are trained at one fixed resolution).

### Readiness Assessment

The student is well-prepared. They have:
1. Deep understanding of the full SD v1.5 pipeline with tensor shapes at every stage (DEVELOPED in 6.4.1)
2. CLIP text encoding at DEVELOPED depth, including architecture, contrastive learning, and limitations (6.3.3)
3. Cross-attention as the text conditioning mechanism at DEVELOPED depth (6.3.4)
4. Img2img as partial denoising at DEVELOPED depth, including implementation from scratch (6.5.2)
5. U-Net architecture at DEVELOPED depth with channel progression traced (6.3.1)
6. Conditioning mechanisms (timestep, text, spatial) at DEVELOPED depth (6.3.2, 6.3.4, 7.1)
7. The capstone series context: the student has been reading about frontier models for 8 lessons in Series 7

The student is missing SDXL-specific knowledge (dual encoders, refiner, micro-conditioning), but every prerequisite concept is at DEVELOPED depth. There are no gaps to fill. The lesson introduces new applications of familiar patterns, which is exactly the BUILD profile.

The main challenge is making the lesson engaging despite being "just" refinements to the U-Net architecture. The narrative frame ("the U-Net's last stand") provides the engagement: every improvement in this lesson is an attempt to push the U-Net architecture further, and the implicit question is whether there is a ceiling. The next lesson (DiT) answers that question.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain the key architectural and conditioning innovations in SDXL (dual text encoders, micro-conditioning, and the refiner model) and why these represent the limits of what U-Net-based scaling can achieve before the paradigm shift to transformers.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| CLIP text encoder producing [77, 768] contextual embeddings | DEVELOPED | DEVELOPED | clip (6.3.3) | OK | Need to understand what SD v1.5's text encoder does so the student can see what SDXL changes. The student knows CLIP's architecture, training, and limitations. Dual encoders will be taught as an extension. |
| Cross-attention (Q from spatial, K/V from text, per-spatial-location conditioning) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Need to understand how text embeddings enter the U-Net. SDXL's dual encoders concatenate their outputs before cross-attention. The mechanism is unchanged; only the K/V source tensor grows. |
| Full SD v1.5 pipeline (CLIP -> U-Net -> VAE with tensor shapes) | INTRODUCED | DEVELOPED | stable-diffusion-architecture (6.4.1) | OK | Need the student to know the baseline pipeline so SDXL's changes are clear. "Here is what SD v1.5 does. Here is what SDXL changes." |
| Img2img as partial denoising (VAE encode, noise to t_start, denoise from there) | DEVELOPED | DEVELOPED | img2img-and-inpainting (6.5.2) | OK | The refiner model is img2img applied with a specialized second model. The mechanism is identical: take an intermediate result, noise it partially, denoise with a different model. The student implemented img2img from scratch. |
| U-Net architecture (encoder-decoder with skip connections, channel progression) | INTRODUCED | DEVELOPED | unet-architecture (6.3.1) | OK | Need to understand the baseline U-Net so the student can see SDXL's changes (wider channels, more attention layers). The student has the architecture at DEVELOPED depth. |
| Classifier-free guidance (CFG formula, guidance scale tradeoff) | INTRODUCED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | SDXL uses CFG with dual text encoders. The formula is the same; the text embeddings are richer. |
| Component modularity ("three translators, one pipeline") | INTRODUCED | INTRODUCED | stable-diffusion-architecture (6.4.1) | OK | Need to understand that components can be swapped. SDXL adds a second text encoder and a refiner model. Same modularity principle. |
| Conditioning mechanisms (adaptive norm for timestep, cross-attention for text) | INTRODUCED | DEVELOPED | conditioning-the-unet (6.3.2) | OK | Micro-conditioning adds new conditioning inputs alongside the timestep. The student understands how conditioning inputs are injected. |

### Gap Resolution

No gaps. All prerequisites are at or above required depth. The student has every concept needed at DEVELOPED depth, giving headroom for a BUILD lesson.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "SDXL is a fundamentally different architecture from SD v1.5" | The name "XL" and the dramatically better outputs suggest a new design. The student has seen SDXL mentioned alongside DiT and SD3 as "next-generation." | SDXL's denoising backbone is still a convolutional U-Net with encoder-decoder, skip connections, cross-attention at middle resolutions, and adaptive group normalization. The same diagram from 6.3.1 still applies with wider channels. If you removed the dual encoders and micro-conditioning, you would have a larger SD v1.5. | Early in the lesson, after presenting the "what changed" overview. Explicit statement: "The U-Net is the same architecture. Everything on this list is about what goes IN or what goes AROUND the U-Net, not the U-Net itself." |
| "Two text encoders means two separate cross-attention mechanisms" | The student knows one CLIP encoder feeds one cross-attention K/V path. Two encoders might mean two separate cross-attention paths, like IP-Adapter's decoupled cross-attention. | SDXL concatenates the two encoders' outputs into one embedding tensor before cross-attention. The U-Net still has one cross-attention path; the K/V source tensor is simply wider ([77, 768] from CLIP ViT-L concatenated with [77, 1280] from OpenCLIP ViT-bigG = [77, 2048]). This is fundamentally different from IP-Adapter's decoupled approach (separate K/V projections, separate attention computations, weighted addition). | When introducing dual text encoders. ComparisonRow: SDXL concatenation vs IP-Adapter decoupled attention. "Same cross-attention, wider input" vs "two cross-attentions, weighted sum." |
| "The refiner model is a separate, specialized architecture" | "Refiner" sounds like a post-processing step or a different kind of model. The student might imagine a GAN-based sharpener or a super-resolution network. | The refiner is another U-Net with the same SDXL architecture, fine-tuned on high-quality images at high resolution. It takes the base model's output, applies img2img (noise to an intermediate level, denoise from there), and produces sharper high-frequency details. Same mechanism the student implemented in 6.5.2. The two-model pipeline is conceptually: base generates structure at 1024x1024, refiner polishes details. | When introducing the refiner. Explicit connection to img2img: "This is the img2img mechanism you already know, applied with a second model that specializes in fine detail." |
| "Micro-conditioning is a minor implementation detail" | "Feeding image size as a conditioning input" sounds like a simple bookkeeping parameter, not a significant innovation. | Without micro-conditioning, SDXL would need to throw away all training images below 1024x1024 (wasting most of the training data) OR accept that images trained at lower resolutions produce artifacts at 1024x1024. Micro-conditioning lets the model learn "when the original image was 512x512, the output looked like this; when it was 1024x1024, the output looked like this"--and then at inference you ask for 1024x1024 quality regardless of what the training image resolution was. It eliminates an entire class of artifacts (cropped heads, off-center compositions) that plagued multi-resolution training. | After explaining the multi-resolution training problem. Present the problem first (artifacts), then the solution (micro-conditioning), so the student feels the significance. |
| "Higher resolution just means scaling up the latent space" | The student might think going from 512x512 to 1024x1024 simply means changing the VAE's latent from [4, 64, 64] to [4, 128, 128] and everything else stays the same. | Higher resolution dramatically increases compute: attention is O(n^2) in spatial tokens. At 128x128 latent, self-attention has 16,384 tokens at 16x16 resolution (vs 4,096 at 64x64 latent). More importantly, the model needs to have SEEN 1024x1024 images during training to generate them well--SD v1.5 was trained primarily on 512x512 and produces poor results at higher resolutions. The resolution improvement in SDXL required both architectural scaling (more compute) and training strategy changes (multi-resolution training with micro-conditioning). | When discussing resolution. Address computational and training-data requirements, not just the spatial dimension change. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| SD v1.5 vs SDXL side-by-side on the same prompt (e.g., "a cat sitting on a beach at sunset"--the running example from 6.4.1) | Positive | Show the output quality improvement. The student sees that SDXL produces dramatically better results on the same prompt they traced through the pipeline in 6.4.1. This motivates the question: what changed? | Uses the familiar callback prompt from 6.4.1. The student already traced this exact prompt through SD v1.5. Seeing SDXL's output on the same prompt makes the improvement tangible and personal. |
| SDXL dual encoder embedding concatenation with concrete tensor shapes | Positive | Make the dual encoder mechanism concrete. CLIP ViT-L produces [77, 768]. OpenCLIP ViT-bigG produces [77, 1280]. Concatenated: [77, 2048]. One cross-attention path, wider K/V. The student can trace the tensor shapes as they did in 6.4.1. | Continues the "trace the pipeline with tensor shapes" pattern from 6.4.1. The student is comfortable with this format. Concrete dimensions make the change tangible rather than abstract. |
| The refiner as img2img with a specialized model (base output at timestep ~200, refiner denoises from ~200 to 0) | Positive (stretch) | Connect the refiner to img2img, which the student implemented from scratch. The refiner takes the base model's output, noises it to an intermediate level (e.g., timestep 200 of 1000), and denoises from there with a model specialized for fine detail. This is the "starting mid-hike" mechanism from 6.5.2 with a different hiker (model). | Anchors the new concept to something the student built themselves. The img2img notebook exercise had the student implement the exact pipeline that the refiner uses. The only difference is which model does the denoising. |
| Micro-conditioning negative example: a training image that was 256x256 upscaled to 1024x1024, producing blurry/artifacted output because the model does not know the original was low-resolution | Negative | Show WHY micro-conditioning matters by showing what happens without it. If the model trains on a mix of 256x256, 512x512, and 1024x1024 images all resized to 1024x1024, it learns to produce an average that includes blurriness from the upscaled small images. Micro-conditioning tells the model "this training example was originally 256x256"--and at inference you say "I want original_size=1024x1024 quality." | Shows the problem before the solution. Without this negative example, micro-conditioning sounds like a minor detail. With it, the student understands that multi-resolution training is a real problem requiring an engineering solution. |

---

## Phase 3: Design

### Narrative Arc

The student has spent eight lessons in Series 7 studying what can be done with Stable Diffusion's frozen U-Net: add spatial control (ControlNet), add image prompting (IP-Adapter), understand the theoretical foundations better (score functions, flow matching), and make generation faster (consistency models, adversarial distillation). In every case, the U-Net itself stayed the same--a convolutional encoder-decoder with cross-attention, trained to predict noise. But during the same period that these advances emerged, Stability AI shipped SDXL, which asks a different question: what happens if you make the U-Net itself better?

SDXL is not a new idea. It is the U-Net approach pushed to its practical ceiling. Larger backbone, higher training resolution, dual text encoders for richer conditioning, a refiner model for fine detail, and micro-conditioning to handle multi-resolution training data. Every one of these is a scaling or conditioning improvement--not a new architecture. The student should feel this as both impressive (the quality jump from SD v1.5 is dramatic) and limiting (every improvement is about working around the U-Net's constraints rather than transcending them). This sets up the implicit question that the next lesson answers: what if you replaced the U-Net entirely?

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example** | Tensor shape trace through the SDXL pipeline, mirroring the SD v1.5 trace from 6.4.1. CLIP ViT-L: [77, 768]. OpenCLIP ViT-bigG: [77, 1280]. Concatenated: [77, 2048]. Pooled embeddings concatenated for time embedding: [768] + [1280] = [2048]. Latent: [4, 128, 128] at 1024x1024. The student can compare these numbers to the SD v1.5 trace they already know. | The student learned SD v1.5 through tensor shape tracing. Using the same format for SDXL makes the differences immediately visible: where the shapes diverge from SD v1.5, something changed. Where they are the same, the mechanism is preserved. This is the most efficient way to communicate "what changed and what did not." |
| **Visual** | Side-by-side pipeline diagram: SD v1.5 (one CLIP encoder, 512x512, no refiner) vs SDXL (two CLIP encoders, 1024x1024, optional refiner). Color-coded to highlight what is new (dual encoders, refiner) vs what is the same (VAE, U-Net backbone, cross-attention, CFG). | A visual comparison makes the scope of changes instantly clear. The student can see that the U-Net box is the same shape in both diagrams--only what feeds into it and what happens after it changed. This supports the "same architecture, better conditioning" thesis. |
| **Verbal/Analogy** | "The U-Net's last stand"--every SDXL innovation is an attempt to push the U-Net architecture further without replacing it. Dual encoders give it better instructions. The refiner gives it a second chance. Micro-conditioning removes training data noise. Higher resolution gives it more room to work. But the U-Net itself is the same species. The implicit question: is there a ceiling? | Sets up the narrative arc for the entire module. The student should finish this lesson feeling that SDXL is impressive but fundamentally incremental, primed for the paradigm shift in lesson 10 (DiT). The "last stand" framing also helps retention--it gives the lesson a clear place in the series narrative. |
| **Intuitive** | The "volume and clarity" intuition for dual encoders: CLIP ViT-L is a good translator that misses nuance. OpenCLIP ViT-bigG is a great translator that captures subtlety. Using both is not about two different channels--it is about a richer, more detailed translation of the same text. The concatenated embedding gives the cross-attention mechanism more to work with at every spatial location. | Makes dual encoders feel natural rather than arbitrary. The student already understands cross-attention as "reading from a reference document." Having a richer reference document (more detailed embeddings) directly improves the reading. No new mechanism needed--just better source material. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 new concepts that are extensions of familiar patterns:
  1. Dual text encoders with concatenated embeddings -- NEW mechanism (two encoders instead of one, concatenated output) but the student deeply understands CLIP encoding, cross-attention, and the existing single-encoder pipeline. The extension is clear: wider K/V source tensor.
  2. Micro-conditioning (original_size, crop_top_left, target_size as conditioning inputs) -- NEW concept, no direct analogue in prior lessons. However, the student deeply understands conditioning mechanisms (timestep via adaptive norm from 6.3.2, text via cross-attention from 6.3.4, spatial via ControlNet from 7.1.1). Micro-conditioning is "more conditioning inputs" using the same injection pattern (added to the timestep embedding).
  3. Refiner model as specialized img2img -- NEW application of a known mechanism. The student implemented img2img from scratch in 6.5.2. The refiner uses the identical pipeline with a second, specialized model. The conceptual leap is minimal.
- **Previous lesson load:** the-speed-landscape was CONSOLIDATE (0 new concepts, pure synthesis)
- **Is this appropriate?** BUILD is appropriate. The student had a CONSOLIDATE lesson as the previous lesson, so a BUILD with 2-3 familiar-pattern extensions is a natural step up. None of the new concepts require a conceptual leap--they are all "more of the same but better." The main cognitive work is integration: understanding how these refinements work together to produce SDXL's quality improvement.

### Connections to Prior Concepts

| New Concept | Connects To | How |
|-------------|------------|-----|
| Dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG, concatenated to [77, 2048]) | CLIP text encoder [77, 768] (6.3.3) + cross-attention K/V (6.3.4) | "SD v1.5 uses one CLIP encoder producing [77, 768]. SDXL uses two, producing [77, 2048]. The cross-attention mechanism is unchanged--the K/V source tensor is just wider, giving the U-Net a richer description to attend to at every spatial location." |
| Micro-conditioning (original_size, crop_top_left, target_size fed alongside timestep) | Timestep conditioning via adaptive group norm (6.3.2) + conditioning channels framework (7.1.3) | "You know the timestep is injected at every processing stage via adaptive group norm. Micro-conditioning adds more numbers alongside the timestep: the original image size, the crop position, the target size. Same injection mechanism, more information." Extends the WHEN/WHAT/WHERE framework to include resolution-awareness. |
| Refiner model (SDEdit-style partial denoising with a second specialized model) | Img2img as partial denoising (6.5.2) | "You implemented img2img: encode, noise to intermediate level, denoise from there. The refiner is exactly this, but with a model fine-tuned on high-quality high-resolution images. Same mechanism, specialized hiker." |
| SDXL's larger U-Net (wider channels, more attention at higher resolutions) | SD v1.5 U-Net channel progression 320/640/1280/1280 (6.3.1, 6.4.1) | "Same architecture diagram from 6.3.1. Wider channels: 320/640/1280 removed in favor of higher base. More attention blocks at higher resolutions. Bigger, not different." |
| SDXL's 1024x1024 base resolution | Latent space dimensions [4, 64, 64] at 512x512 (6.3.5) | "At 1024x1024, the VAE produces [4, 128, 128] latents. Same 8x downsampling ratio, larger spatial grid. Attention at 32x32 = 1024 tokens (still tractable). Attention at 64x64 = 4096 tokens (getting expensive)." |

### Analogies to Extend

- **"Three translators, one pipeline"** from 6.4.1 -- SDXL adds a fourth translator (OpenCLIP ViT-bigG) that works alongside the original CLIP translator. But they speak into the same microphone (concatenated embeddings), not separate channels.
- **"Starting mid-hike"** from 6.5.2 -- the refiner is exactly this: the base model hikes from the summit to partway down, then a different hiker (the refiner) completes the descent. Same trail, different hiker for the final stretch.
- **"WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE"** from 7.1.3 -- micro-conditioning adds another dimension: "AT-WHAT-QUALITY" (resolution-awareness). Same conditioning philosophy, one more input.

### Analogies That Could Be Misleading

- **"Two reference documents, one reader"** from IP-Adapter (7.1.3) could mislead because IP-Adapter uses decoupled cross-attention (two separate K/V paths, weighted addition). SDXL does NOT do this--it concatenates the two encoders' outputs into one embedding vector and uses a single cross-attention path. Address by explicit ComparisonRow: "SDXL concatenates before cross-attention. IP-Adapter computes two separate attentions and adds. Same goal (richer conditioning), different mechanism."
- **"Volume knob"** from ControlNet/IP-Adapter could mislead if the student thinks the two encoders have individual scale parameters. They do not. The embeddings are concatenated and treated as one input. There is no knob to adjust the balance between CLIP ViT-L and OpenCLIP ViT-bigG at inference time.

### Scope Boundaries

**This lesson IS about:**
- What changed between SD v1.5 and SDXL: dual text encoders, larger U-Net, higher base resolution, refiner model, micro-conditioning
- WHY each change was made: the problem it solves and the tradeoff it introduces
- Tensor shape trace through the SDXL pipeline (mirroring the 6.4.1 trace)
- The refiner as img2img with a specialized model
- Micro-conditioning as a solution to multi-resolution training artifacts
- Positioning SDXL as the U-Net's ceiling, setting up DiT

**This lesson is NOT about:**
- Training SDXL from scratch (the student will not train anything; Series 6 had the training experience)
- SDXL Turbo or LCM-LoRA for SDXL (already covered in Module 7.3)
- Every architectural detail of the SDXL U-Net (internal channel counts, exact attention layer placement)
- SDXL ControlNet or IP-Adapter variants (extensions of Module 7.1 concepts, not new)
- Production deployment or optimization
- VAE changes (SDXL uses a slightly different VAE but the change is not pedagogically significant)
- The DiT architecture (next lesson)

**Depth targets:**
- Dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG, concatenated embeddings): DEVELOPED (tensor shapes, why two encoders, how they combine, what changes in cross-attention)
- Micro-conditioning (original_size, crop_top_left, target_size): INTRODUCED (the problem it solves, how it is injected, why it matters, but not the implementation details of how these numbers are encoded)
- Refiner model: INTRODUCED (what it does, connection to img2img mechanism, when to use it, but not the refiner's specific training procedure or architectural differences)
- SDXL as U-Net ceiling / narrative positioning: INTRODUCED (qualitative understanding, not a formal argument)

---

### Lesson Outline

#### 1. Context + Constraints

- "This is the first lesson in Module 7.4: Next-Generation Architectures. Every model you have worked with in Series 7--ControlNet, LCM, SDXL Turbo--uses the same U-Net backbone from Series 6. This module asks: what about the architecture itself?"
- "We start with SDXL, the U-Net pushed to its limit. The next two lessons replace the U-Net entirely."
- ConstraintBlock: This lesson covers WHAT changed in SDXL and WHY. It does not cover SDXL training, SDXL Turbo (covered in 7.3.2), ControlNet/IP-Adapter for SDXL (same concepts from Module 7.1), or every internal architectural detail. We are reading the SDXL paper together, extracting the key innovations.

#### 2. Recap

Brief reactivation of three concepts (no gaps to fill, so this is pure reactivation):
- **SD v1.5 pipeline** (from 6.4.1): 2-3 sentences. "You traced the full pipeline: text -> CLIP [77, 768] -> U-Net denoising loop with CFG -> VAE decode [3, 512, 512]. The U-Net has channels 320/640/1280/1280, cross-attention at 16x16 and 32x32, and generates in [4, 64, 64] latent space."
- **Cross-attention** (from 6.3.4): 1-2 sentences. "Q from spatial features, K/V from text embeddings. Each spatial location independently attends to all 77 text tokens. The quality of the text embedding directly determines how well the U-Net follows the prompt."
- **Img2img** (from 6.5.2): 1-2 sentences. "Take an image, encode with VAE, noise to an intermediate level, denoise from there. Strength parameter controls the starting timestep. You implemented this from scratch."
- Transition: "SDXL starts from this exact foundation. Everything you know about SD v1.5 still applies. The question is: what did Stability AI change, and why?"

#### 3. Hook

Type: **Before/after + "what changed?" challenge**

"Here is the same prompt you traced through SD v1.5 in lesson 6.4.1: 'a cat sitting on a beach at sunset.'"

Two vivid description GradientCards side-by-side:
- **SD v1.5 at 512x512** (amber): Recognizable scene, decent composition, but soft details. Cat's fur lacks individual strands. Sand has a slight plastic texture. Sunset colors are pleasant but lack the complexity of real light scattering. Text following is approximate--the cat might be standing rather than sitting, the beach might lack water.
- **SDXL at 1024x1024** (emerald): Dramatically sharper. Individual fur strands visible. Sand has realistic granularity. Sunset has complex color gradients and atmospheric haze. Text following is noticeably better--the cat is sitting, the beach has waves, the sunset is prominent.

Challenge: "This improvement comes from five changes. Before I tell you what they are, predict: what would YOU change about SD v1.5 to get better results? Think about the components you know--what are the weaknesses?"

Design challenge in `<details>` reveal: The student should be able to identify at least 2-3 of the following from their existing knowledge:
1. Text understanding is limited (CLIP ViT-L has only 123M params, 77 tokens, known limitations)
2. Resolution is stuck at 512x512 (model trained at this resolution, generates poorly at higher)
3. Details are soft (single denoising pass must handle both composition and fine detail)

"SDXL addresses all three--plus two more you might not have predicted."

#### 4. Explain: The Five Changes (Overview)

Five GradientCards as a brief overview, each with a one-sentence description:
1. **Dual Text Encoders** -- Two CLIP models instead of one, for richer text understanding
2. **Larger U-Net** -- Wider channels, more attention at higher resolutions
3. **Higher Base Resolution** -- 1024x1024 instead of 512x512
4. **Refiner Model** -- A second U-Net for fine detail polish
5. **Micro-Conditioning** -- Resolution and crop metadata as conditioning inputs

InsightBlock: "Notice what is NOT on this list: a new architecture. The U-Net is the same species. Every change is about what goes IN to the U-Net, what comes OUT, or what goes AROUND it. The U-Net backbone itself is still a convolutional encoder-decoder with skip connections and cross-attention."

Address Misconception #1: "SDXL is still a U-Net model. The same diagram from lesson 6.3.1 still applies. The improvements are about conditioning and scale, not a new denoising architecture."

#### 5. Explain: Dual Text Encoders

**Part A: Why two encoders?**

"SD v1.5 uses CLIP ViT-L/14, producing [77, 768] embeddings. This encoder has 123M parameters and was trained on 400M image-text pairs. It is good at understanding common concepts and compositions but struggles with nuance: specific art styles, precise spatial descriptions, fine-grained attributes."

"SDXL adds a second encoder: OpenCLIP ViT-bigG/14. This is a much larger CLIP model (354M parameters in the text encoder alone) trained on 2B image-text pairs (LAION-2B). It produces [77, 1280] embeddings."

"The two encoders' outputs are concatenated along the embedding dimension: [77, 768] + [77, 1280] = [77, 2048]. This combined tensor becomes the K/V source for cross-attention."

**Part B: How concatenation works**

Tensor shape trace (continuing the format from 6.4.1):
```
Prompt: "a cat sitting on a beach at sunset"

CLIP ViT-L/14:    [77 tokens, 768 dims]   -- SD v1.5's original encoder
OpenCLIP ViT-bigG: [77 tokens, 1280 dims]  -- SDXL's addition
Concatenated:      [77 tokens, 2048 dims]  -- One embedding per token, richer

Cross-attention K/V source: [77, 2048]  (was [77, 768] in SD v1.5)
```

"The cross-attention mechanism is unchanged. Q still comes from spatial features. K and V come from this wider embedding. Each spatial location still independently attends to all 77 tokens. The only difference: the embedding at each token position carries 2048 dimensions of meaning instead of 768."

Address Misconception #2: ComparisonRow -- SDXL concatenation vs IP-Adapter decoupled attention:

| | SDXL Dual Encoders | IP-Adapter Decoupled Attention |
|---|---|---|
| Inputs combined | Before cross-attention (concatenated embeddings) | After cross-attention (weighted addition of two outputs) |
| Cross-attention paths | One (wider K/V source) | Two (separate text and image K/V projections) |
| Runtime balance | Fixed (no scale parameter) | Adjustable (IP-Adapter scale) |
| Information type | Both are text (same prompt, different encoders) | Text vs image (different modalities) |

InsightBlock: "Same goal--richer conditioning. Different approach. SDXL combines two text encoders before the bottleneck. IP-Adapter combines text and image after the bottleneck."

**Part C: Pooled embeddings for global conditioning**

"Beyond the per-token sequence embeddings, SDXL also uses the pooled (CLS) embedding from OpenCLIP ViT-bigG. This is a single [1280]-dim vector summarizing the entire prompt. It is concatenated with the timestep embedding and fed through the adaptive norm pathway--the same global conditioning mechanism you know from 6.3.2."

"So SDXL uses text in two ways: per-token embeddings [77, 2048] through cross-attention (spatially varying) and a pooled vector [1280] through adaptive norm (global). Two injection points, two purposes."

#### 6. Check #1

Three predict-and-verify questions:
1. "In SDXL, the cross-attention K/V source is [77, 2048]. In SD v1.5, it was [77, 768]. Does this change the cross-attention output shape?" (Answer: No. The attention output shape is determined by Q and V dimensions. Q comes from spatial features, same as before. The attention weights are still [n_spatial, 77]. The output dimension per spatial location changes from 768 to 2048, but this is handled by the projection layers within the attention block.)
2. "If you loaded an SD v1.5 LoRA into SDXL, would it work?" (Answer: No. The LoRA targets cross-attention projection matrices. In SD v1.5, W_K and W_V project from 768 dimensions. In SDXL, they project from 2048 dimensions. The matrix shapes are incompatible. This is why "SD v1.5 LoRAs" and "SDXL LoRAs" are separate categories--same mechanism, different dimensions.)
3. "A colleague says: 'SDXL uses two encoders, so it takes twice as long to encode the prompt.' Is this accurate?" (Answer: Roughly yes for the text encoding step--two forward passes through two models. But text encoding is a negligible fraction of total inference time (one pass each, vs 50+ U-Net passes for denoising). The actual compute cost increase comes from the larger U-Net, not the dual encoders.)

#### 7. Explain: Micro-Conditioning

**Part A: The multi-resolution training problem**

"Training SDXL required massive datasets--hundreds of millions of images. These images come in every resolution: 256x256 thumbnails, 512x512 web images, 1024x1024 photographs, 2000x3000 DSLR shots. How do you train on all of them?"

"Option 1: Only use images at or above 1024x1024. This throws away most of the dataset. Bad--you need the data."

"Option 2: Resize everything to 1024x1024. A 256x256 thumbnail upscaled to 1024x1024 is blurry. The model learns 'sometimes images are blurry' and occasionally generates blurry output. Bad--the model learns to produce artifacts."

"Option 3: Crop 1024x1024 regions from larger images. A 2000x3000 photo cropped to 1024x1024 might cut off heads, miss the main subject, or show off-center compositions. The model learns 'sometimes the subject is partially visible.' Bad--the model generates cropped compositions."

Present the problem concretely: "SD v1.5 suffered from these exact issues at 512x512. Images with cut-off heads, off-center subjects, and inconsistent quality. The larger and more diverse the dataset, the worse these artifacts become."

**Part B: The solution**

"Micro-conditioning tells the model what it is looking at. Three additional numbers fed alongside the timestep:"

Three-item GradientCard:
- **original_size** -- The resolution of the training image before any resizing. "This image was originally 256x256" vs "This image was originally 2048x1536."
- **crop_top_left** -- Where the 1024x1024 crop was taken from. (0, 0) means top-left corner, (256, 128) means cropped from the middle. The model learns that (0, 0) crops often have cut-off bottoms and (512, 512) crops are well-centered.
- **target_size** -- The resolution the model should generate. Always set to (1024, 1024) at inference. During training, it matches the training resolution.

"At inference, you set original_size = (1024, 1024), crop_top_left = (0, 0), and target_size = (1024, 1024). This tells the model: 'generate as if the original image was high-resolution and well-centered.' The model has learned to separate image quality from image content."

**Part C: How micro-conditioning is injected**

"These numbers are encoded and added to the timestep embedding. Same injection mechanism as the timestep: through the adaptive norm pathway at every processing stage. The U-Net already has the plumbing for global conditioning inputs--micro-conditioning just sends more information through the same pipes."

Connection to existing conditioning framework: "Timestep tells the U-Net WHEN in the denoising process. Micro-conditioning tells the U-Net AT-WHAT-QUALITY the training data was. Both are global (same at every spatial location). Both use adaptive norm injection."

Address Misconception #4: "This is not a minor detail. Without micro-conditioning, SDXL would either waste most of its training data (only using 1024x1024 images) or produce resolution-dependent artifacts (blurry outputs from upscaled training data, cropped compositions from random crops). Micro-conditioning is what makes it possible to train on a massive, diverse dataset AND generate high-quality output."

#### 8. Check #2

Two predict-and-verify questions:
1. "At inference, you set original_size = (1024, 1024) and crop_top_left = (0, 0). What would happen if you set original_size = (256, 256) and crop_top_left = (0, 0) instead?" (Answer: The model would generate as if the original image was a 256x256 thumbnail. The output would likely be softer, less detailed--the model learned that 256x256 originals tend to be lower quality. This is sometimes used intentionally to produce a specific aesthetic, but generally you want original_size = target_size for best quality.)
2. "Micro-conditioning is injected through the same adaptive norm pathway as the timestep. Could you add other metadata the same way--for example, the camera model or the aesthetic quality rating of the training image?" (Answer: Yes, in principle. Any global metadata about the training image can be encoded and added to the timestep embedding. Some models do exactly this--for example, conditioning on aesthetic scores to steer toward higher-quality outputs at inference. The key insight is that the adaptive norm pathway is a general-purpose global conditioning channel, not specialized for timesteps.)

#### 9. Explain: The Refiner Model

**Part A: The problem**

"SDXL's base model generates at 1024x1024 with dramatically better composition and text following than SD v1.5. But a single U-Net must handle everything: global composition, mid-level structure, and fine-grained details (skin pores, fabric weave, individual leaves). The denoising process allocates these responsibilities across timesteps (high noise = composition, low noise = details), but asking one model to be expert at both is a lot."

**Part B: The solution**

"The SDXL refiner is a second U-Net, fine-tuned specifically on high-quality, high-resolution images. It specializes in the low-noise timesteps where fine detail matters."

"The two-model pipeline:"
1. Base model denoises from t=T (pure noise) to t=t_switch (e.g., t=200)--handles composition, structure, color
2. Refiner model denoises from t=t_switch to t=0--handles fine detail, texture, sharpness

Address Misconception #3: "This is the img2img mechanism you already know. The base model's output at t_switch is a partially denoised latent. The refiner takes this latent, treats it as an img2img input at strength corresponding to t_switch, and completes the denoising. Same mechanism you implemented in notebook 6.5.2, applied with a different model."

Connection: "In img2img, you started from a real image noised to an intermediate level. Here, you start from a model-generated intermediate result. The pipeline is identical: latent at intermediate noise level -> denoise from there to t=0. The only difference: the model doing the denoising was specifically trained for this finishing role."

TipBlock: "The refiner is optional. SDXL's base model generates good results without it. The refiner adds polish--sharper details, more realistic textures--at the cost of a second denoising pass. Whether it is worth the compute depends on the use case."

#### 10. Check #3

Two predict-and-verify questions:
1. "The refiner model receives a partially denoised latent at timestep t_switch. Does it also receive the text prompt?" (Answer: Yes. The refiner uses the same dual text encoder embeddings for cross-attention guidance during its denoising pass. The text prompt continues to guide the generation at the fine-detail level. Without text conditioning, the refiner would polish details without semantic guidance--it might sharpen fur on a dog when the prompt said "cat.")
2. "A colleague suggests setting t_switch = T (the refiner starts from pure noise, doing all the denoising). What would happen?" (Answer: The refiner was fine-tuned on the low-noise timestep range. It was not trained to handle high-noise denoising (global composition, structure). Starting from pure noise would likely produce poor results because the refiner's specialization is in fine detail, not composition. This is like asking a detail painter to also sketch the entire scene from scratch--different skills.)

#### 11. Explain: Resolution and Architecture Scale

Brief section connecting the resolution and U-Net scaling:

"SDXL generates at 1024x1024 base resolution. With the same VAE (8x downsampling), the latent is [4, 128, 128]--four times the spatial area of SD v1.5's [4, 64, 64]. The U-Net processes this larger latent with a correspondingly larger architecture: deeper attention at higher resolutions, wider channels."

"This is a straightforward scaling. The architecture is the same--just bigger to handle bigger inputs. And this points to the U-Net's fundamental constraint: scaling a convolutional architecture is ad hoc. You can widen channels, add more attention blocks, increase resolution--but there is no systematic recipe. Each scaling decision requires manual engineering. Transformers, as you will see in the next lesson, scale more predictably."

InsightBlock: "The U-Net's Last Stand"--every SDXL innovation is about what goes in (dual encoders), what goes around (refiner), or what goes alongside (micro-conditioning). The U-Net backbone itself is the same species, just larger. The question the next lesson answers: what if you replaced it entirely?

#### 12. Elaborate: SDXL in the Ecosystem

**Part A: What SDXL changed for practitioners**

Brief practical positioning:
- SDXL became the standard base model for the community (replacing SD v1.5)
- ControlNet, IP-Adapter, LoRA all work with SDXL (same mechanisms, different dimensions)
- SDXL Turbo (from 7.3.2) applies adversarial distillation to the SDXL base model
- LCM-LoRA for SDXL (from 7.3.2) enables 4-step generation from SDXL checkpoints
- The speed landscape from 7.3.3 applies directly--all acceleration approaches work with SDXL

**Part B: What SDXL could NOT solve**

- Scaling recipe: there is no clear "double compute, halve the loss" relationship. Each U-Net scaling decision is manual.
- Cross-attention as the only text-image interaction: text tokens and image features interact only at cross-attention layers (16x16, 32x32). The rest of the U-Net processes spatial features without text input. Is this the best way?
- Convolutional inductive biases: convolutions assume local spatial structure. This is useful (translational equivariance) but also limiting (global relationships require many layers and large receptive fields, or attention).

"These limitations are not bugs in SDXL. They are properties of the U-Net architecture itself. To address them, you need a different architecture. That is the subject of the next lesson."

#### 13. Practice (Notebook)

Design: 4 exercises, Guided -> Guided -> Supported -> Independent progression.

**Exercise 1 (Guided): SDXL Pipeline Inspection**
- Load SDXL base model via diffusers
- Inspect the dual text encoders: print model class names, parameter counts, output shapes
- Verify: CLIP ViT-L produces [77, 768], OpenCLIP ViT-bigG produces [77, 1280]
- Compare to SD v1.5 text encoder (if memory allows, load both; otherwise compare to known shapes from 6.4.1)
- Predict-before-run: "What shape will the combined text embedding be?" (Answer: [77, 2048])
- What it tests: the dual encoder pipeline is a real, inspectable configuration, not just a diagram

**Exercise 2 (Guided): SDXL Base Generation and Comparison**
- Generate "a cat sitting on a beach at sunset" with SDXL base at 1024x1024
- Compare to SD v1.5 at 512x512 (use a pre-generated reference image if VRAM is limited)
- Vary guidance_scale (5, 7.5, 10) and observe quality differences from SD v1.5 at same scales
- Predict-before-run: "Will the optimal guidance_scale for SDXL be higher or lower than SD v1.5?" (Answer: Typically lower--the dual encoders provide better text conditioning, so less amplification via CFG is needed. SDXL defaults to guidance_scale ~5-7 vs SD v1.5's ~7.5.)
- What it tests: the practical quality improvement and the connection between richer text encoding and lower required guidance

**Exercise 3 (Supported): Micro-Conditioning Exploration**
- Generate the same prompt with different micro-conditioning values:
  - original_size = (1024, 1024), crop_top_left = (0, 0) -- default, best quality
  - original_size = (256, 256), crop_top_left = (0, 0) -- simulate low-res original
  - original_size = (1024, 1024), crop_top_left = (512, 512) -- simulate center crop
- Compare outputs: the low original_size should produce softer details; the crop offset should shift composition
- What it tests: micro-conditioning is not just metadata--it actively affects generation quality and composition

**Exercise 4 (Independent): Base + Refiner Pipeline**
- Set up the two-stage pipeline: base model denoises for most steps, refiner finishes the last ~20%
- Generate with base-only vs base+refiner, same prompt and seed
- Vary t_switch (the handoff point): try giving the refiner 10%, 20%, 40% of the denoising steps
- Compare sharpness and fine detail across the variants
- Bonus: time the two-stage pipeline vs base-only to quantify the compute tradeoff
- What it tests: the refiner as img2img (same mechanism, specialized model), the detail improvement, and the compute-quality tradeoff

#### 14. Summarize

Key takeaways (echo mental models):

1. **Same architecture, better everything around it.** SDXL is still a U-Net. The improvements are about conditioning (dual text encoders, micro-conditioning), scale (larger U-Net, higher resolution), and pipeline (refiner model). The denoising backbone is the same species as SD v1.5.

2. **Dual encoders = wider text embedding.** CLIP ViT-L [77, 768] + OpenCLIP ViT-bigG [77, 1280] = [77, 2048]. One cross-attention path, richer K/V source. Not decoupled attention--concatenated before the bottleneck.

3. **Micro-conditioning solves multi-resolution training.** Tell the model what it is looking at (original resolution, crop position, target resolution), and it learns to separate content from quality. At inference, ask for high-quality output regardless of training data diversity.

4. **The refiner is img2img with a specialist.** Same mechanism you implemented in 6.5.2. The base model handles composition; the refiner polishes details. Optional but effective.

5. **The U-Net's last stand.** Every SDXL improvement pushes the U-Net architecture further without replacing it. The implicit question: is there a ceiling? The next lesson answers that question.

#### 15. Next Step

"SDXL showed what happens when you push the U-Net to its limit: dramatically better results through better conditioning and scale. But the U-Net itself--a convolutional encoder-decoder with hand-designed skip connections--has no clear scaling recipe. What if you replaced it with an architecture that does?"

"The next lesson introduces the Diffusion Transformer (DiT). It replaces the U-Net entirely with a vision transformer: patchify the latent into tokens, process with standard transformer blocks, use adaptive layer norm for conditioning. The architecture you learned in Series 4 meets the diffusion framework you learned in Series 6. Everything converges."

---

## Review -- 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 1

### Verdict: NEEDS REVISION

No critical findings. The lesson is well-structured, follows the plan closely, and should genuinely work for the student. Two improvement findings concern a missing planned modality and a partially addressed misconception that weaken the lesson compared to what was designed.

### Findings

### [IMPROVEMENT] -- Missing visual modality: side-by-side pipeline diagram

**Location:** Planned in Phase 3 "Modalities Planned" (Visual), not present in the built lesson
**Issue:** The planning document specifies a side-by-side pipeline diagram: "SD v1.5 (one CLIP encoder, 512x512, no refiner) vs SDXL (two CLIP encoders, 1024x1024, optional refiner). Color-coded to highlight what is new (dual encoders, refiner) vs what is the same (VAE, U-Net backbone, cross-attention, CFG)." This visual modality was not built. The lesson currently has 3 modalities (concrete example via tensor shape trace, verbal/analogy via "U-Net's last stand," and symbolic via CodeBlock). It meets the minimum of 3, but a planned modality was dropped without documentation. A pipeline diagram would make the "same architecture, better conditioning" thesis immediately visible at a glance--the student could see that the U-Net box is the same shape in both diagrams. This is particularly valuable because the lesson's central argument is architectural continuity, and a visual comparison is the most efficient way to communicate that.
**Student impact:** The student must reconstruct the architectural comparison mentally from prose descriptions rather than seeing it visually. For a BUILD lesson where the core insight is "same backbone, different inputs," a visual would reduce cognitive effort and strengthen retention.
**Suggested fix:** Add a pipeline comparison visual. Options: (1) An ASCII art or Mermaid-style diagram showing SD v1.5 pipeline flow vs SDXL pipeline flow, placed in Section 5 alongside the five-change overview. (2) A structured comparison using two GradientCards with parallel structure showing the pipeline stages. The diagram should highlight what changed (dual encoders in blue, refiner in emerald, micro-conditioning in orange) vs what stayed the same (VAE, U-Net backbone, cross-attention in gray/muted).

### [IMPROVEMENT] -- Misconception #5 partially addressed (higher resolution = just scaling up)

**Location:** Section 12 (Resolution and Architecture Scale)
**Issue:** The planning document identifies Misconception #5: "Higher resolution just means scaling up the latent space." The planned negative example includes specific attention token counts ("At 128x128 latent, self-attention has 16,384 tokens at 16x16 resolution vs 4,096 at 64x64 latent") and the training data requirement ("the model needs to have SEEN 1024x1024 images during training to generate them well"). The built lesson mentions "attention is O(n^2)" and "scaling a convolutional architecture is ad hoc" but omits the concrete token count comparison and the training data point. This makes the section feel hand-wavy rather than grounded.
**Student impact:** The student understands that SDXL is bigger but does not get a concrete sense of HOW MUCH bigger the compute is or WHY simply changing the latent size is not enough. The "ad hoc scaling" claim is asserted rather than demonstrated.
**Suggested fix:** Add 2-3 sentences with the concrete numbers from the plan: "At 1024x1024, the VAE produces [4, 128, 128] latents. Self-attention at 32x32 resolution now operates on 1,024 tokens (vs 256 at SD v1.5's 16x16). At 64x64 resolution, that is 4,096 tokens. The compute scales quadratically with spatial tokens." Also add the training data point: "And scaling the architecture alone is not enough--the model must be trained on 1024x1024 images to generate them well. SD v1.5 was trained primarily on 512x512 and produces poor results when asked to generate at higher resolutions."

### [POLISH] -- Spaced em dashes in micro-conditioning GradientCard

**Location:** Section 8, the "Micro-Conditioning Inputs" GradientCard (lines 596, 602, 609)
**Issue:** Three list items use spaced em dashes: `<strong>original_size</strong> — The resolution...`, `<strong>crop_top_left</strong> — Where the 1024×1024 crop...`, `<strong>target_size</strong> — The resolution...`. The writing style rule requires no-space em dashes: `word—word` not `word — word`.
**Student impact:** Minor inconsistency with the rest of the lesson's formatting (the header description uses the correct `—` without spaces).
**Suggested fix:** Change ` — ` to `—` in these three locations. Alternatively, since these are list items where the em dash separates a term from its definition, a colon (`:`) might be more appropriate and would sidestep the formatting question entirely.

### Review Notes

**What works well:**
- The lesson's narrative arc is strong. The "U-Net's last stand" framing provides coherence and motivation throughout, and sets up the DiT lesson naturally.
- The problem-before-solution ordering is consistent. Every section motivates the need before introducing the mechanism. The micro-conditioning section is particularly well-structured (three bad options, then the solution).
- The IP-Adapter comparison (ComparisonRow in Section 6) is an excellent anticipatory misconception address. The student would absolutely confuse SDXL's concatenation with IP-Adapter's decoupled attention without this explicit comparison.
- The check-your-understanding questions are consistently strong. They test reasoning and application, not memorization. The LoRA incompatibility question (Check #1, Q2) is especially good--it connects a practical concern to the tensor shape change.
- The notebook is thorough and well-scaffolded. Exercise progression mirrors the lesson arc. Solutions include reasoning, common mistakes, and alternative approaches.
- Connections to prior knowledge are explicit and specific throughout. The refiner-to-img2img connection is particularly well-developed across two paragraphs.

**Pattern observation:**
- The dropped visual modality is a recurring pattern from Module 7.3 reviews (both the-speed-landscape and consistency-models had planned visual modalities that were downgraded during building). For a lesson whose central argument is architectural continuity, a visual comparison would add significant value. Prioritize this for the revision.

---

## Review -- 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

All three iteration 1 findings have been adequately resolved. No new critical or improvement findings. The lesson is ready to ship.

### Iteration 1 Fix Verification

**1. Side-by-side pipeline diagram (was IMPROVEMENT):** Resolved. A ComparisonRow was added in Section 5b comparing the SD v1.5 and SDXL pipelines in parallel. The left column (amber) shows the familiar SD v1.5 stages with tensor shapes. The right column (emerald) shows SDXL's stages, making it immediately visible what changed (dual encoders, wider K/V, refiner, micro-conditioning) and what stayed the same (CFG mechanism, VAE decode, core denoising loop). An aside explains how to read the comparison, and an italic summary reinforces the "same core loop" thesis. This fulfills the planned visual modality and provides the at-a-glance architectural comparison that was missing.

**2. Concrete resolution scaling numbers (was IMPROVEMENT):** Resolved. Section 12 now includes specific token counts (256 tokens at 16x16 in SD v1.5 vs 1,024 tokens at 32x32 in SDXL, 4,096 at 64x64), the quadratic compute relationship (4x tokens = ~16x compute per attention layer), and the training data requirement ("the model must be trained on 1024x1024 images to generate them well"). The section is now grounded in concrete numbers rather than hand-wavy assertions.

**3. Spaced em dashes in micro-conditioning GradientCard (was POLISH):** Resolved. The three list items in the "Micro-Conditioning Inputs" GradientCard now use unspaced em dashes (`original_size</strong>—The`, `crop_top_left</strong>—Where`, `target_size</strong>—The`). No spaced em dashes remain in any rendered content in the lesson or notebook.

### Review Notes

**What works well (all observations from iteration 1 still hold, plus):**
- The ComparisonRow pipeline comparison is well-integrated. It appears after the five-change overview, providing a visual anchor before the detailed explanations. The parallel structure makes same-vs-different immediately scannable.
- The resolution section's concrete numbers (256 vs 1,024 vs 4,096 tokens, quadratic scaling) ground the "ad hoc scaling" argument in observable compute costs. The student now understands both the magnitude of the scaling challenge and why architectural scaling alone is insufficient.
- The lesson achieves a strong pedagogical balance: 2-3 new concepts (all BUILD-level extensions of familiar patterns), seven check-your-understanding questions testing reasoning rather than memorization, explicit connections to prior knowledge at every stage, and a compelling narrative arc that positions SDXL as both impressive and fundamentally limited.
- The notebook is thorough, self-contained, and well-aligned with the lesson. Exercise scaffolding (Guided -> Guided -> Supported -> Independent) mirrors the lesson's conceptual progression. Solutions include reasoning, code, common mistakes, and observation guidance.
- The lesson's strongest pedagogical moves: (a) the IP-Adapter vs SDXL ComparisonRow that pre-empts the most likely misconception, (b) the three-bad-option problem framing for micro-conditioning that makes the solution feel necessary rather than arbitrary, (c) the refiner-to-img2img connection that leverages the student's hands-on implementation experience.

**No remaining concerns.** The lesson is pedagogically sound, follows the plan, addresses all five planned misconceptions, uses four modalities (concrete example, visual comparison, verbal/analogy, intuitive), and maintains a coherent narrative arc that sets up the DiT lesson naturally.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (no gaps)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps to resolve)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution: "the U-Net pushed to its ceiling")
- [x] At least 3 modalities planned for the core concept, each with rationale (4 modalities: concrete example, visual, verbal/analogy, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 2-3 new concepts (dual text encoders, micro-conditioning, refiner)--all extensions of familiar patterns
- [x] Every new concept connected to at least one existing concept (dual encoders to CLIP/cross-attention, micro-conditioning to adaptive norm/conditioning framework, refiner to img2img)
- [x] Scope boundaries explicitly stated
