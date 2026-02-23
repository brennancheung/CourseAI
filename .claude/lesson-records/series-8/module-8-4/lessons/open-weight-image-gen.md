# Lesson Planning: Open Weight Image Generation Models

**Slug:** `open-weight-image-gen`
**Module:** 8.4 (Image Generation Landscape)
**Type:** Survey / Landscape
**Duration:** 45-60 minutes (longer than typical -- breadth, not depth)
**Cognitive Load:** CONSOLIDATE

---

## Phase 1: Orient (Student State)

The student has completed all of Series 6 (Stable Diffusion) and Series 7 (Post-SD Advances), plus Module 8.3 (Architecture Analysis). They have deep understanding of image generation architectures and can reason about model design from first principles.

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| U-Net architecture for denoising (encoder-decoder with skip connections, multi-resolution feature maps) | DEVELOPED | 6.3.1 (u-net-architecture) | Built and traced tensor shapes through the full encoder-bottleneck-decoder path |
| Cross-attention conditioning (text embeddings as K/V, image features as Q, at specific U-Net resolutions) | DEVELOPED | 6.3.4 (text-conditioning) | Traced attention weight matrices, visualized per-token attention maps |
| VAE encoding/decoding (8x spatial compression, 4-channel latent space, separately trained) | DEVELOPED | 6.1.4 (variational-autoencoders) | Built VAE intuition from autoencoders, understands the reparameterization trick |
| CLIP text encoder (contrastive learning, [77,768] embeddings, image-text alignment) | DEVELOPED | 6.3.3 (clip-for-diffusion) | Understands CLIP's role as translator, its strengths (visual concepts) and weaknesses (compositional reasoning) |
| Classifier-free guidance (unconditional + conditional prediction, guidance scale w) | DEVELOPED | 6.3.5 (classifier-free-guidance) | "Contrast slider" analogy, understands the formula and tradeoffs |
| Full SD v1.5 pipeline (CLIP -> U-Net denoising loop with CFG -> VAE decode, tensor shapes at every handoff) | DEVELOPED | 6.4.1 (stable-diffusion-architecture) | Traced the complete pipeline, understands component modularity |
| DDPM/DDIM/DPM-Solver sampling (ODE perspective, sampler as inference-time choice) | DEVELOPED | 6.4.2 (samplers-and-efficiency) | "Predict and leap" for DDIM, ODE perspective, higher-order solvers |
| SDXL architecture (dual text encoders CLIP+OpenCLIP, larger U-Net, micro-conditioning, refiner model, 1024x1024) | DEVELOPED | 7.4.1 (sdxl) | "The U-Net's last stand"--every improvement is IN, AROUND, or ALONGSIDE the U-Net |
| DiT / Diffusion Transformers (patchify, adaLN-Zero, standard transformer blocks, scaling recipe) | DEVELOPED | 7.4.2 (diffusion-transformers) | "Tokenize the image," "Two knobs not twenty," understood the convergence of transformers + diffusion |
| MMDiT joint attention (concatenate text + image tokens, shared self-attention, separate Q/K/V projections per modality) | DEVELOPED | 7.4.3 (sd3-and-flux) | "One room one conversation," four attention types, "shared listening separate thinking" |
| T5-XXL as text encoder (4.7B params, deep linguistic understanding complementing CLIP) | INTRODUCED | 7.4.3 (sd3-and-flux) | Understands why SD3 uses three text encoders and T5's role |
| Triple text encoder setup in SD3 (CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL) | INTRODUCED | 7.4.3 (sd3-and-flux) | Traced the progression: 1 encoder -> 2 -> 3 |
| Rectified flow / flow matching (straight-line interpolation, velocity prediction, fewer inference steps) | DEVELOPED | 7.2.2 (flow-matching) + 7.4.3 (sd3-and-flux) | Understands the training objective and why it enables fewer steps |
| SD3 vs Flux architectural positioning (both MMDiT family, Flux uses single-stream blocks in later layers) | MENTIONED | 7.4.3 (sd3-and-flux) | Brief ComparisonRow, vocabulary-level only |
| S3-DiT single-stream architecture (shared projections + refiner layers, ~50% parameter savings) | INTRODUCED | 7.4.4 (z-image) | "Translate once then speak together," parameter counting |
| Qwen3-4B as text encoder (single LLM replacing triple encoder setup) | INTRODUCED | 7.4.4 (z-image) | Text encoder evolution trajectory: 1 CLIP -> 2 CLIPs -> 3 encoders -> 1 LLM |
| 3D Unified RoPE (temporal + spatial positional encoding) | INTRODUCED | 7.4.4 (z-image) | Orthogonal axes for text and image tokens |
| DMD / Decoupled-DMD / DMDR (distillation + RL post-training) | INTRODUCED | 7.4.4 (z-image) | "Spear and shield," "teacher becomes guardrail" |
| ControlNet (zero-convolution adapter, copy of encoder half) | DEVELOPED | 7.1.1 (controlnet) | Understands the mechanism and zero-initialization safety pattern |
| LoRA fine-tuning (low-rank adaptation of attention weights) | DEVELOPED | 6.5.1 (lora-finetuning) | Has fine-tuned a model, understands the math |
| Consistency models / consistency distillation (direct mapping from any noise level to clean output) | INTRODUCED | 7.3.2 (consistency-models) | Understands the one-step generation concept |
| IP-Adapter (decoupled cross-attention for image conditioning) | DEVELOPED | 7.1.3 (ip-adapter) | "What it looks like" conditioning channel |
| Autoregressive image generation (discrete visual tokenization, GPT-style next-token prediction on image tokens) | INTRODUCED | 8.3.1 (nano-banana-pro) | Understands the paradigm as distinct from diffusion |

**Mental models already established:**
- "Three translators, one pipeline" (SD modularity)
- "The U-Net's last stand" (SDXL as U-Net ceiling)
- "Tokenize the image" (patchify / DiT)
- "Two knobs not twenty" (transformer scaling recipe)
- "One room, one conversation" (MMDiT joint attention)
- "Shared listening, separate thinking" (modality-specific projections)
- "Translate once, then speak together" (S3-DiT single-stream)
- "Convergence, not revolution" (frontier architectures combine known components)
- "Simplicity beats complexity" (Z-Image competitive with fewer params)

**What was explicitly NOT covered:**
- SD v2.x architecture details (different CLIP encoder, v-prediction)
- Detailed Flux architecture beyond vocabulary-level positioning vs SD3
- PixArt, Playground, Kandinsky, DeepFloyd IF, Stable Cascade, Hunyuan-DiT, Kolors, AuraFlow as specific models
- DALL-E or Midjourney internals (closed models)
- The broader ecosystem of who built what and why (lab histories, researcher lineage)
- Model selection criteria for practical use
- Open weight vs open source licensing distinctions

**Readiness assessment:** The student is exceptionally well-prepared for this lesson. Every core technology (U-Net, DiT, MMDiT, S3-DiT, VAE, CLIP, T5, cross-attention, joint attention, flow matching, adaLN-Zero, CFG, LoRA, consistency distillation, DMD) is at INTRODUCED or DEVELOPED depth. This lesson does not teach any of these--it uses them as vocabulary for a landscape survey. The student should feel like they are reading a map in a language they already speak fluently.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to place any open weight image generation model into the architectural and historical landscape, identifying its denoising backbone, text encoder(s), training objective, key innovation, and lineage within the broader evolution.

Note: this is a survey/consolidation lesson. The "target concept" is not a single technical idea but an organized mental map of the field. This is acceptable for a CONSOLIDATE lesson where the value is synthesis, not introduction.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| U-Net denoising architecture | INTRODUCED | DEVELOPED | 6.3.1 | OK | Student needs to recognize U-Net as one of two backbone types. Has full understanding. |
| DiT / patchify / transformer blocks for denoising | INTRODUCED | DEVELOPED | 7.4.2 | OK | Student needs to recognize DiT as the alternative backbone. Has full understanding. |
| MMDiT joint attention | INTRODUCED | DEVELOPED | 7.4.3 | OK | Student needs to distinguish MMDiT from standard DiT. Has full understanding. |
| S3-DiT single-stream architecture | MENTIONED | INTRODUCED | 7.4.4 | OK | Only need recognition level for the comparison table. |
| CLIP text encoder | INTRODUCED | DEVELOPED | 6.3.3 | OK | Need to recognize CLIP as text encoder type. Has deep understanding. |
| T5-XXL text encoder | INTRODUCED | INTRODUCED | 7.4.3 | OK | Need to recognize T5 as text encoder type. Understands its role. |
| DDPM training objective | INTRODUCED | DEVELOPED | 6.2.2 | OK | Need to distinguish DDPM from flow matching as training approaches. |
| Flow matching / rectified flow | INTRODUCED | DEVELOPED | 7.2.2 | OK | Need to recognize as alternative training objective. |
| VAE for latent space compression | INTRODUCED | DEVELOPED | 6.1.4 | OK | Need to recognize VAE as standard component. |
| CFG / classifier-free guidance | INTRODUCED | DEVELOPED | 6.3.5 | OK | Need to recognize as standard inference technique. |
| SDXL full architecture | MENTIONED | DEVELOPED | 7.4.1 | OK | Need to place SDXL in the landscape. Has detailed knowledge. |
| SD3 full architecture | MENTIONED | INTRODUCED | 7.4.3 | OK | Need to place SD3 in the landscape. Has solid understanding. |
| Flux architecture positioning | MENTIONED | MENTIONED | 7.4.3 | OK | Only need vocabulary-level recognition. |
| Z-Image / S3-DiT full architecture | MENTIONED | INTRODUCED | 7.4.4 | OK | Need to place Z-Image in the landscape. Has solid understanding. |
| LoRA fine-tuning | MENTIONED | DEVELOPED | 6.5.1 | OK | Relevant for ecosystem discussion. |
| ControlNet | MENTIONED | DEVELOPED | 7.1.1 | OK | Relevant for ecosystem discussion. |
| Consistency distillation | MENTIONED | INTRODUCED | 7.3.2 | OK | Relevant for acceleration techniques comparison. |

**Gap resolution:** No gaps. Every prerequisite is at or above required depth. This is expected for a CONSOLIDATE lesson at the end of the student's image generation journey.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Newer models are always better across every dimension" | Natural assumption from seeing the field evolve. Each new model sounds like it supersedes the previous one. | SD v1.5 has vastly more LoRAs, ControlNets, and community tooling than SD3 or Flux. For fine-tuning workflows, the ecosystem can matter more than raw quality. SDXL remains widely used because its tooling is mature. | Model comparison section -- include "ecosystem maturity" and "fine-tuning ecosystem" columns alongside architecture. |
| "All these models are fundamentally different architectures" | Each model has its own name, branding, and marketing. PixArt-alpha, SD3, and Hunyuan-DiT sound completely different. | PixArt-alpha, PixArt-sigma, SD3, Flux, and Hunyuan-DiT are ALL DiT-based architectures with variations. Once you identify the backbone (U-Net vs DiT vs MMDiT), most of the mystery dissolves. The number of truly distinct architectural paradigms is small (3-4). | Architecture taxonomy section -- group models by backbone type first, then distinguish within groups. |
| "Open weight means open source" | The terms are often used interchangeably in casual discussion. Model download pages say "open" without specifying what. | Flux.1 Dev releases weights under a non-commercial license--you can download and run it, but cannot use it commercially without a separate license. SD v1.5 (CreativeML OpenRAIL-M) allows commercial use with restrictions. True open-source would include training code, data, and full reproducibility. | Scope boundaries section -- brief clarification, not a deep dive into licensing. |
| "The shift from U-Net to DiT was a single event" | SD3's paper made it seem like a sudden pivot. | PixArt-alpha (Nov 2023) used DiT before SD3 (Feb 2024). The Peebles & Xie DiT paper (Dec 2022, ICCV 2023) preceded all of them. Hunyuan-DiT (May 2024) developed independently at Tencent. The shift was a gradual convergence across multiple labs, not a single switch. | Historical timeline section -- make the parallel development visible. |
| "Flux is just SD3 with different branding" | Both use MMDiT + flow matching. Same researchers. Sounds like a rename. | Flux introduces architectural innovations SD3 lacks: single-stream blocks in later layers (reducing parameter count), rotary position embeddings instead of learned, guidance distillation for Schnell variant. They share lineage but diverged architecturally. | SD3 vs Flux comparison -- specific architectural differences listed. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| The Stability AI -> Black Forest Labs lineage (Robin Rombach et al. create SD v1, leave Stability, found BFL, create Flux) | Positive | Shows how researcher lineage explains architectural evolution. SD and Flux share DNA because the same people made them. | Makes the abstract "lineage" concept concrete and human. The student can trace why Flux continues the MMDiT direction. |
| The parallel DiT convergence (PixArt-alpha Nov 2023, Hunyuan-DiT May 2024, SD3 Feb 2024--three labs independently adopt DiT) | Positive | Shows that architectural shifts are driven by the underlying idea's merit, not a single lab's decision. Multiple teams converge on the same solution because it works. | Breaks the misconception that the U-Net -> DiT shift was a single event. Three independent data points make the trend undeniable. |
| SD v1.5 ecosystem vs SD3 ecosystem (v1.5 has thousands of LoRAs, dozens of ControlNet models, massive community; SD3 has fewer despite being newer and architecturally superior) | Negative | Newer is not always better for practical use. Architecture quality is one of many factors in model selection--ecosystem, tooling, community support, and fine-tuning availability matter enormously. | Directly addresses the "newer is always better" misconception with a concrete, observable example the student can verify on Civitai or Hugging Face. |

### Additional Research Notes for Build Phase

**Models to cover in detail (open weight, with architecture breakdown):**

1. **Stable Diffusion v1.x (Aug 2022)** -- Stability AI / CompVis / Runway
   - U-Net backbone, CLIP ViT-L/14 text encoder, KL-VAE (f=8, 4 channels), DDPM/DDIM training, 512x512
   - Key innovation: First widely available open weight text-to-image model. Latent diffusion made high-quality generation accessible on consumer GPUs.
   - ~860M U-Net params, ~123M CLIP params, ~84M VAE params (~1.07B total)

2. **Stable Diffusion v2.x (Nov 2022)** -- Stability AI
   - U-Net backbone (larger), OpenCLIP ViT-H/14 text encoder (replacing CLIP ViT-L), same VAE, v-prediction (alternative parameterization to epsilon-prediction), 512x512 and 768x768 variants
   - Key innovation: v-prediction parameterization, improved training at higher resolutions. But the switch to OpenCLIP ViT-H broke compatibility with v1.x ecosystem and the community largely rejected it.
   - Notable as a cautionary tale: better architecture does not guarantee adoption if it breaks ecosystem compatibility.

3. **SDXL (Jul 2023)** -- Stability AI
   - Larger U-Net backbone, dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG, concatenated to [77,2048]), micro-conditioning, optional refiner model, 1024x1024
   - Key innovation: "The U-Net's last stand"--pushed U-Net to its practical limits with better conditioning and higher resolution. ~3.5B U-Net params.
   - Student already knows this in depth from 7.4.1.

4. **DeepFloyd IF (Apr 2023)** -- Stability AI / DeepFloyd
   - Pixel-space diffusion (not latent!), T5-XXL text encoder (first major use of T5 for diffusion), cascaded generation (64x64 -> 256x256 -> 1024x1024 with three separate models)
   - Key innovation: Demonstrated T5-XXL's superiority for text understanding in diffusion models. Excellent text rendering for its era. Influenced SD3's adoption of T5.
   - Notable: pixel-space diffusion at 64x64 base was extremely expensive. The cascaded approach did not scale well. But the T5 insight was hugely influential.

5. **Kandinsky 2.x (Apr-Jul 2023)** -- Sber AI (Russia)
   - U-Net backbone, CLIP image encoder + image prior model (unCLIP-style) + CLIP text encoder
   - Key innovation: unCLIP approach (text -> image embedding prior -> diffusion conditioned on image embedding). Different conditioning strategy from SD.
   - Notable for demonstrating the unCLIP/DALL-E 2 approach in open weight form.

6. **PixArt-alpha (Nov 2023)** -- Huawei
   - DiT backbone (first major open weight DiT for text-to-image), T5-XXL text encoder, cross-attention conditioning (not joint attention), 512x512 to 1024x1024
   - Key innovation: Showed DiT works for text-to-image with cross-attention (not just class-conditional ImageNet). Training efficiency: competitive quality at ~10% of SD's training cost through decomposed training strategy (first learn image distribution, then learn text-image alignment).
   - ~600M params. Much more efficient than SD.

7. **PixArt-sigma (Apr 2024)** -- Huawei
   - Same DiT backbone as PixArt-alpha but with improved VAE, better data curation, and weak-to-strong training transfer
   - Key innovation: 4K resolution support, improved VAE. Demonstrated that DiT could scale to very high resolutions.

8. **Stable Cascade / Wurstchen (Dec 2023 research, Feb 2024 release)** -- Stability AI
   - Three-stage cascade in latent space: Stage C (high-level composition) operates in highly compressed 24x24 latent space, Stage B (detail) upsamples latent, Stage A (pixel decode). Uses a different VAE compression paradigm.
   - Key innovation: Extreme latent compression (42:1 spatial ratio vs SD's 8:1). Trained much faster (24K A100 hours vs 200K+ for SD). But the cascaded approach added inference complexity.
   - Notable: architecturally interesting but did not gain wide adoption.

9. **Playground v2.5 (Feb 2024)** -- Playground AI
   - SDXL architecture (same U-Net backbone) with different training: aesthetic-focused dataset, EDM framework (Karras et al.), modified noise schedule
   - Key innovation: Demonstrated that training recipe matters as much as architecture. Same SDXL architecture but reportedly better aesthetic quality through curated data and training improvements.
   - Notable: proved you can get significant quality gains without changing the architecture.

10. **Stable Diffusion 3 / SD3.5 (Feb 2024 paper, Jun 2024 release / Oct 2024 for SD3.5)** -- Stability AI
    - MMDiT backbone, triple text encoders (CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL), flow matching training, 1024x1024
    - Key innovation: MMDiT joint attention (text and image tokens attend to each other bidirectionally), flow matching training objective for fewer inference steps
    - SD3.5 Large: 8B params. SD3.5 Medium: 2.5B params. SD3.5 Large Turbo: distilled for 4-step generation.
    - Student already knows this in depth from 7.4.3.

11. **Hunyuan-DiT (May 2024)** -- Tencent
    - DiT backbone with cross-attention and adaLN conditioning, bilingual CLIP + bilingual T5 text encoders (Chinese + English), 1024x1024
    - Key innovation: First major bilingual (Chinese-English) open weight model. Uses a dialogue-based text encoder fine-tuned with multi-turn conversation data for better prompt understanding.
    - ~1.5B params.

12. **Kolors (Jul 2024)** -- Kuaishou/KWAI
    - SDXL-like U-Net backbone, ChatGLM-6B text encoder (Chinese LLM), 1024x1024
    - Key innovation: Used a large Chinese LLM (ChatGLM) as text encoder, demonstrating that general-purpose LLMs can replace CLIP for conditioning. Strong bilingual support. Interesting as an SDXL-architecture model with LLM text encoding.
    - This may be what the user refers to as "QWEN"--it's from the Chinese AI ecosystem and uses an LLM text encoder.

13. **AuraFlow (Jun 2024)** -- Fal.ai
    - MMDiT-style architecture, open source training code + data pipeline
    - Key innovation: Fully open training pipeline (code, data curation, training scripts). Community-driven alternative to corporate models.
    - ~6.8B params.

14. **Flux.1 (Aug 2024)** -- Black Forest Labs (founded by Robin Rombach et al., original SD creators)
    - MMDiT backbone with single-stream blocks in later layers, dual text encoders (CLIP ViT-L + T5-XXL, dropping second CLIP), rotary position embeddings, guidance distillation (Schnell), flow matching training, 1024x1024+
    - Key innovation: Hybrid double-stream + single-stream blocks (19 double-stream then 38 single-stream in Dev). Schnell variant with 1-4 step generation via guidance distillation. Rotary embeddings for resolution flexibility.
    - Flux.1 Dev: ~12B params (non-commercial license). Flux.1 Schnell: ~12B params (Apache 2.0). Flux.1 Pro: API-only.
    - Student knows this at vocabulary level from 7.4.3.

15. **Z-Image Base / Z-Image Turbo (Jan 2025)** -- Freepik
    - S3-DiT single-stream backbone, Qwen3-4B text encoder (single LLM), 3D Unified RoPE, DMDR post-training, 1024x1024+
    - Key innovation: Single-stream architecture with refiner layers. Single LLM text encoder replacing triple encoder setup. Post-training with DMD + RL. Competitive quality at 6.15B params vs Flux's ~12B.
    - Student already knows this in depth from 7.4.4.

**Models to mention briefly (closed, but historically important for context):**

- **DALL-E (Jan 2021)** -- OpenAI. Autoregressive (discrete VAE + GPT). First major text-to-image model. Proved the concept.
- **DALL-E 2 (Apr 2022)** -- OpenAI. Diffusion-based (unCLIP: CLIP text -> image prior -> diffusion decoder). Kandinsky follows this approach.
- **DALL-E 3 (Sep 2023)** -- OpenAI. Improved prompt following via synthetic captions. Integrated into ChatGPT.
- **Imagen (May 2022)** -- Google. Pixel-space cascaded diffusion with T5-XXL. Never released. Demonstrated T5's value for text encoding (influenced DeepFloyd IF and SD3).
- **Midjourney (various)** -- Midjourney Inc. Architecture undisclosed. Dominant in aesthetic quality. Discord-based interface. Important reference point but provides no architectural learning.

**Video generation model to mention:**
- **Wan 2.1 (Feb 2025)** -- Alibaba/Tongyi Wanxiang. Primarily a video generation model, not image. Uses a 3D VAE + DiT backbone for video. Mentioned for completeness but clearly distinguished as video, not image.

**Flux 2 distinction:**
- **Flux 2 (announced 2025)** -- Black Forest Labs. API-only, NOT open weight. Includes variants (Max, Pro, Flex, Klein) but none have released weights. Should be mentioned briefly as the closed successor to Flux.1, noting the shift from open to closed distribution.

---

## Phase 3: Design

### Narrative Arc

You have spent the last two series learning how image generation works from the inside out--U-Nets, DiTs, MMDiTs, VAEs, CLIP, T5, flow matching, cross-attention, joint attention. You can trace a denoising step, explain why CFG works, and describe the difference between SD v1.5 and SD3 at the architectural level. But if someone asks "what are the major open weight image generation models and how do they relate to each other?"--you might struggle to give a coherent answer. The individual trees are clear. The forest is not.

This lesson steps back to see the forest. Not to learn new trees--every technology you will encounter here is something you have already studied. The value is in the map: seeing which innovations propagated across models, which architectural choices were dead ends, and how the field moved from a single U-Net-based model in 2022 to a diverse ecosystem of transformer-based architectures in 2025. By the end, you should be able to pick up any new model announcement and immediately place it in this landscape--"ah, this is a DiT-based model with flow matching and an LLM text encoder, so it is in the same family as Flux and SD3, but with the text encoder approach of Z-Image."

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual (timeline) | Horizontal timeline of model releases (2022-2025) showing release dates, labs, and key innovations | The landscape is fundamentally temporal--the student needs to see what came first and how innovations propagated. A timeline makes temporal relationships visible at a glance. |
| Visual (architecture taxonomy diagram) | Tree diagram grouping models by backbone type: U-Net family (SD v1, v2, SDXL, Playground v2.5, Kolors), DiT family (PixArt-alpha, PixArt-sigma, Hunyuan-DiT), MMDiT family (SD3, Flux, AuraFlow), S3-DiT (Z-Image), Other (DeepFloyd IF pixel-space, Kandinsky unCLIP, Stable Cascade) | Grouping by backbone type immediately reveals that the number of truly distinct architectures is small. Reduces the overwhelming model count to a manageable taxonomy. |
| Verbal/Analogy | "Family tree" analogy--models have parents, siblings, and cousins. SD v1 and Flux are parent-child (same creators). SD3 and Flux are siblings (same era, same lab lineage, divergent design). PixArt and SD3 are cousins (independent convergence on DiT). | Makes the relationships between models feel natural and memorable. The student already uses family-tree reasoning intuitively. |
| Concrete (comparison table) | Large comparison table with columns: Model, Release Date, Lab, Backbone, Text Encoder(s), Training Objective, Base Resolution, Approx Params, Key Innovation, Open Weight License | The centerpiece deliverable. A single reference artifact the student can consult when encountering any model. Concrete data in every cell--no vague descriptions. |
| Symbolic (architecture evolution diagrams) | Three evolution lines traced with key technical details: (1) Denoising backbone: U-Net -> U-Net (larger) -> DiT -> MMDiT -> S3-DiT, (2) Text encoding: CLIP -> CLIP+OpenCLIP -> CLIP+OpenCLIP+T5 -> CLIP+T5 -> single LLM, (3) Training objective: DDPM -> v-prediction -> flow matching -> rectified flow + distillation | Seeing the three evolution lines separately reveals that architecture, text encoding, and training objective evolved somewhat independently. A model's position is defined by its choices on each axis. |
| Intuitive | "Convergence, not revolution" callback -- every model on this map is built from components the student already knows. The landscape looks overwhelming until you realize there are only ~4 backbone types, ~4 text encoding strategies, and ~3 training objectives. The combinatorial space is small. | The emotional payoff: the student should feel that their deep knowledge makes the landscape navigable, not that breadth is overwhelming. |

### Cognitive Load Assessment

- **How many new concepts in this lesson?** Zero genuinely new concepts. Every technology referenced was taught in Series 6-7 or Module 8.3. The organizational framework (landscape map, model taxonomy) is new but is a synthesis structure, not a technical concept.
- **What was the load of the previous lesson?** If the student is coming from Module 8.3 (nano-banana-pro), that was STRETCH. If from Module 7.4 (z-image), that was BUILD. Either way, CONSOLIDATE is appropriate as a follow-up.
- **Is this lesson's load appropriate?** Yes. The breadth is high (many models) but the depth per model is deliberately shallow--the student already has deep knowledge of the key architectures (SD v1, SDXL, SD3, Flux, Z-Image) and only needs landscape-level awareness of the others. The cognitive work is organizational, not conceptual.

### Connections to Prior Concepts

| Prior Concept | How Connected |
|---------------|---------------|
| "The U-Net's last stand" (SDXL, 7.4.1) | Extended to cover the full U-Net lineage (SD v1 -> v2 -> SDXL -> Playground v2.5 -> Kolors). SDXL was the ceiling; everything after moves to transformers. |
| "Tokenize the image" (DiT, 7.4.2) | Used as the anchor for the DiT family: PixArt-alpha, PixArt-sigma, Hunyuan-DiT all use this approach. The patchify operation is the common thread. |
| "One room, one conversation" (MMDiT, 7.4.3) | Used to identify the MMDiT family: SD3, Flux, AuraFlow. Joint attention is the distinguishing feature. |
| "Translate once, then speak together" (S3-DiT, 7.4.4) | Used for Z-Image's position in the taxonomy--a further simplification of the MMDiT approach. |
| "Convergence, not revolution" (multiple lessons) | The meta-narrative for the whole landscape. The field did not invent new physics--it combined transformers, flow matching, and better text encoders in different configurations. |
| "Two knobs not twenty" (DiT scaling, 7.4.2) | Explains why the field moved to transformers: U-Net scaling is ad hoc, transformer scaling is systematic. This drove the U-Net -> DiT shift across multiple labs simultaneously. |
| Text encoder evolution (6.3.3, 7.4.1, 7.4.3, 7.4.4) | The progression CLIP -> CLIP+OpenCLIP -> CLIP+OpenCLIP+T5 -> CLIP+T5 -> single LLM is a key narrative thread the student has built incrementally. This lesson makes it explicit as a historical trend across models. |

**Potentially misleading prior analogies:** None identified. The student's existing mental models all generalize correctly to the landscape level.

### Scope Boundaries

**This lesson IS about:**
- The landscape of open weight image generation models (2022-2025)
- Architecture identification: backbone type, text encoder(s), training objective for each model
- Historical timeline and lineage (which lab, which researchers, what came from where)
- The key innovation each model contributed to the field
- A reference comparison table
- How to place new model announcements into this map

**This lesson is NOT about:**
- Teaching any new architecture, training objective, or technique (all assumed known)
- Detailed architectural deep dives into models not already covered (PixArt, Hunyuan-DiT, Kolors, etc. get brief descriptions, not full lessons)
- Training procedures, datasets, or compute requirements
- Model quality benchmarks or rankings (too subjective and rapidly outdated)
- Licensing details beyond the open-weight vs open-source distinction
- Video generation models (Wan 2.1 mentioned briefly for completeness)
- Closed models beyond brief historical context (DALL-E, Midjourney, Imagen)
- Practical deployment, inference optimization, or model selection for specific tasks
- Fine-tuning any of these models

**Depth target:** The five models the student already knows in depth (SD v1, SDXL, SD3, Flux, Z-Image) serve as anchors. All other models are at MENTIONED depth--the student should recognize the name, know the backbone type and key innovation, and place it in the taxonomy. No model goes beyond what the student already has.

### Widget Needed

**No custom widget needed.** The lesson's visual needs (timeline, taxonomy tree, comparison table, evolution diagrams) are well-served by existing block components: Mermaid diagrams for the timeline and taxonomy, a standard table for the comparison, and GradientCards or progression diagrams for the evolution lines. No interactive computation or draggable visualization is required for a survey/consolidation lesson.

### Lesson Outline

#### 1. Context + Constraints
- "This is a survey lesson. You already know the technologies. We are organizing them."
- Scope: open weight models, 2022-2025. Closed models mentioned for historical context only.
- Not a ranking. Not a recommendation engine. A map.
- Duration: longer than typical (45-60 min). Take breaks if needed.

#### 2. Recap (light)
- No formal recap needed--prerequisites are solid. But the opening should reactivate key vocabulary by referencing the student's existing knowledge: "You know U-Nets (Series 6), DiTs (7.4.2), MMDiTs (7.4.3), S3-DiTs (7.4.4), flow matching (7.2), and the full text encoder evolution from CLIP to T5 to Qwen. This lesson uses all of that as vocabulary."

#### 3. Hook: The Overwhelming Landscape
- Type: challenge preview
- Present a wall of model names: SD v1.5, SD v2.1, SDXL, PixArt-alpha, PixArt-sigma, DeepFloyd IF, Kandinsky 2.1, Stable Cascade, Playground v2.5, SD3, SD3.5, Hunyuan-DiT, Kolors, AuraFlow, Flux.1 Dev, Flux.1 Schnell, Z-Image Base, Z-Image Turbo...
- "If you tried to learn each model from scratch, this would take weeks. But you do not need to. Once you know the building blocks--and you do--the number of truly distinct architectures is surprisingly small."
- Reveal: "There are really only 4-5 backbone types and 3-4 text encoding strategies. The rest is combinations."

#### 4. The Timeline (Historical Context)
- Horizontal timeline from 2022-2025 showing major releases
- Brief context on closed model milestones that influenced the open weight ecosystem: DALL-E (proved concept), DALL-E 2 (unCLIP approach), Imagen (T5 for diffusion), Midjourney (aesthetic benchmark)
- The open weight story begins with SD v1 (Aug 2022) and CompVis/Stability AI
- Key inflection points marked: SD v1 launch, SDXL consolidation, DiT paper, SD3/Flux pivot to transformers
- Researcher lineage: Robin Rombach and team create SD at CompVis/Stability, later found BFL and create Flux

#### 5. Architecture Taxonomy
- Tree diagram grouping all models by backbone type:
  - **U-Net family:** SD v1.x, SD v2.x, SDXL, Playground v2.5, Kolors, Kandinsky
  - **Pixel-space / Cascaded:** DeepFloyd IF, Stable Cascade (different VAE paradigm)
  - **DiT family (cross-attention):** PixArt-alpha, PixArt-sigma, Hunyuan-DiT
  - **MMDiT family (joint attention):** SD3/SD3.5, Flux.1, AuraFlow
  - **S3-DiT (single-stream):** Z-Image
- For each group: brief description of what defines the group (the shared architectural choice), then per-model one-liner of what distinguishes it within the group
- Callback to established mental models: "The U-Net's last stand," "Tokenize the image," "One room one conversation," "Translate once then speak together"

#### 6. Three Evolution Lines
- **Backbone evolution:** U-Net (SD v1) -> larger U-Net (SDXL) -> DiT (PixArt) -> MMDiT (SD3/Flux) -> S3-DiT (Z-Image). Each step motivated by limitations of the previous (scaling recipe, text-image interaction, parameter efficiency).
- **Text encoder evolution:** CLIP ViT-L (SD v1) -> CLIP + OpenCLIP (SDXL) -> T5-XXL added (DeepFloyd IF pioneered, SD3 adopted) -> CLIP + T5 (Flux) -> single LLM (Z-Image/Kolors). Each step motivated by text understanding limitations.
- **Training objective evolution:** DDPM epsilon-prediction (SD v1) -> v-prediction (SD v2) -> flow matching (SD3/Flux/Z-Image). Plus acceleration: consistency distillation, guidance distillation (Flux Schnell), DMD/DMDR (Z-Image Turbo). Student already knows all of these--this just places them on the timeline.

#### 7. Check #1: Taxonomy Placement
- Assessment type: predict-and-verify
- Present 3-4 model descriptions WITHOUT names. Student identifies backbone type and likely text encoder strategy.
  - "A model that concatenates text and image tokens into one sequence for self-attention, uses CLIP + T5 for text encoding, and trains with flow matching." (Answer: Flux.1 or SD3 family)
  - "A model using a U-Net backbone with a single Chinese LLM as text encoder." (Answer: Kolors)
  - "A model that runs diffusion in pixel space at 64x64, upsamples with two cascaded models, and uses T5-XXL for text." (Answer: DeepFloyd IF)

#### 8. The Comparison Table (Centerpiece)
- Large table with one row per model:
  - Model | Release | Lab | Backbone | Text Encoder(s) | Training Objective | Base Resolution | Approx Params | Key Innovation
- Models in chronological order
- Each row is one sentence: no deep dives, just identification
- The table should feel like a reference card the student returns to

#### 9. Notable Innovations by Model
- Brief section highlighting the non-obvious innovations:
  - SD v2's v-prediction (a different way to parameterize the denoising target--predict the "velocity" rather than the noise)
  - DeepFloyd IF's proof that T5 works for diffusion (influenced SD3's text encoder choice)
  - PixArt-alpha's training efficiency (decomposed training: learn images first, then learn text-image alignment)
  - Playground v2.5's demonstration that training recipe beats architecture (same SDXL arch, better results)
  - Stable Cascade's extreme latent compression (42:1 vs 8:1)
  - Kolors' use of ChatGLM as text encoder (LLM-as-encoder before Z-Image)
  - Flux Schnell's guidance distillation for 1-4 step generation
- For each: one paragraph max. Connect to concepts the student knows.

#### 10. Check #2: Innovation Attribution
- Assessment type: explain-it-back
- "Which model first demonstrated that T5 could work as a text encoder for diffusion?" (DeepFloyd IF)
- "What was Playground v2.5's key insight, given that it used the same architecture as SDXL?" (Training recipe > architecture)
- "Why did SD v2.x fail to replace SD v1.5 in the community, despite being technically improved?" (Broke ecosystem compatibility--different CLIP encoder, incompatible with v1.x LoRAs/ControlNets)

#### 11. The Closed Models (Brief Context)
- One paragraph each for DALL-E lineage, Midjourney, Imagen
- Why mention them: they set the benchmarks and pioneered techniques the open models adopted
- DALL-E 1: proved autoregressive text-to-image works
- DALL-E 2: unCLIP approach (Kandinsky follows this)
- DALL-E 3: synthetic captioning for prompt following
- Imagen: T5-XXL for text encoding in diffusion (DeepFloyd IF and SD3 follow)
- Midjourney: aesthetic quality benchmark, architecture unknown
- Note: Flux 2 (BFL, 2025) is API-only/closed--the Flux lineage moved from open to closed

#### 12. Video Generation (Brief Mention)
- Wan 2.1 (Alibaba): primarily a video generation model using 3D VAE + DiT. The user asked about it, so acknowledge it, but clearly distinguish: video generation is a different domain that shares architectural DNA (DiT backbone, flow matching) but adds temporal modeling.
- One paragraph, no deep dive. This is not the focus of the lesson.

#### 13. How to Read New Model Announcements
- Practical synthesis: when a new model drops, ask these questions:
  1. What is the denoising backbone? (U-Net / DiT / MMDiT / S3-DiT / something new?)
  2. What text encoder(s)? (CLIP / T5 / LLM / multi-encoder?)
  3. What training objective? (DDPM / v-prediction / flow matching?)
  4. What is the claimed innovation? (Architecture / training recipe / data / post-training / scale?)
  5. What is the lineage? (Which lab, which prior models, which researchers?)
- This is the "teach a person to fish" section: the taxonomy is the tool for navigating future releases.

#### 14. Check #3: Apply the Framework
- Assessment type: transfer question
- Present a hypothetical model announcement: "Lab X releases ModelY, a 10B parameter model using a transformer backbone with cross-attention text conditioning, trained with rectified flow on 2B image-text pairs, using Llama 3 as the text encoder."
- Student places it: DiT family (cross-attention, not joint), LLM text encoder (Z-Image/Kolors lineage), flow matching (SD3/Flux training lineage). Closest existing comparison: PixArt-sigma with an LLM text encoder and flow matching training.

#### 15. Summarize
- The landscape has 4-5 backbone types, 3-4 text encoding strategies, 3 training objectives
- The evolution is driven by: better scaling recipes (U-Net -> DiT), better text understanding (CLIP -> T5/LLM), and faster inference (DDPM -> flow matching -> distillation)
- "Convergence, not revolution": every model on this map is built from components you already understand
- The taxonomy is the tool--new models slot into this framework

#### 16. Next Step
- This landscape map is a living document--new models will emerge
- The student can now explore specific models in depth (e.g., trying Flux.1 Schnell vs SD3.5 vs Z-Image Turbo in practice)
- Point toward practical exploration: the student has the knowledge to understand what they are running

### Practice (Notebook)

**No notebook for this lesson.** This is a survey/consolidation lesson. The "practice" is the taxonomy framework itself--the student's exercise is using it to navigate future model announcements. If a notebook were added, it would be a hands-on comparison exercise (generate with SD v1.5, SDXL, and Flux.1 Schnell, compare results and map differences to architectural choices), but this is optional and can be a separate follow-up.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises (no widget)
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (6 modalities planned)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 0 new concepts (CONSOLIDATE)
- [x] Every concept connected to existing knowledge (all callbacks to established mental models)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-23 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 4

### Verdict: NEEDS REVISION

No critical issues -- the lesson is functional and the student would not be lost or form wrong mental models. However, four improvement-level findings meaningfully weaken the lesson's effectiveness. The lesson is strong in structure, scope discipline, and taxonomy visualization, but has gaps in misconception handling, narrative flatness in the middle sections, and a missed opportunity to make the comparison table more scannable.

### Findings

#### [IMPROVEMENT] — Misconception #3 (open weight vs open source) not addressed in the lesson

**Location:** Entire lesson -- absent
**Issue:** The planning document identifies five misconceptions. Four of them are addressed in the built lesson: (1) "newer is always better" -- addressed via the SD v2 cautionary tale and the Check #2 Q3 about ecosystem compatibility; (2) "all these models are fundamentally different" -- addressed by the taxonomy grouping and the hook's "only 4-5 backbone types"; (4) "the U-Net to DiT shift was a single event" -- addressed by the parallel DiT convergence section with a WarningBlock; (5) "Flux is just SD3 with different branding" -- addressed by the MMDiT family card listing specific differences (single-stream blocks, rotary embeddings, guidance distillation). However, misconception #3 ("open weight means open source") is entirely absent from the lesson. The scope boundary says "NOT: licensing details beyond the open-weight vs open-source distinction," implying this distinction SHOULD be made. The planning document specifies addressing it in the "Scope boundaries section -- brief clarification, not a deep dive." The ConstraintBlock mentions "open weight" in the title and scope but never clarifies what "open weight" means vs "open source."
**Student impact:** The student will continue conflating "open weight" and "open source." This is a common confusion in the field and the planning doc explicitly identified it as worth a brief clarification. A one-sentence aside is all that's needed.
**Suggested fix:** Add a TipBlock or ConceptBlock aside (perhaps next to the ConstraintBlock or in the Section 11 closed models area) with a 2-3 sentence clarification: open weight = weights downloadable but may restrict commercial use or not include training code; open source = full reproducibility (code, data, weights). Mention Flux.1 Dev (non-commercial license) vs Flux.1 Schnell (Apache 2.0) as a concrete example. Keep it brief per scope boundaries.

#### [IMPROVEMENT] — Middle sections (Notable Innovations, Section 9) feel like a catalog rather than a narrative

**Location:** Section 9 (Notable Innovations by Model), lines ~887-988
**Issue:** The Notable Innovations section is a sequence of seven GradientCards. Each card is well-written individually. But the section as a whole reads like a catalog: card, card, card, card, card, card, card. There is no narrative thread connecting them. The intro paragraph says "some models contributed innovations that transcended their own adoption," but then the cards are in roughly chronological order without any connecting prose explaining WHY these particular innovations mattered to each other or how they form a progression. The student reads each card in isolation.
**Student impact:** The student processes seven disconnected innovation stories. They might not notice the meta-patterns: that text encoding innovations propagated (DeepFloyd IF -> SD3 -> Flux -> Z-Image), that training recipe insights propagated (Playground v2.5 -> everyone), or that the ecosystem compatibility lesson of SD v2 was never repeated because subsequent models created new ecosystems rather than extending old ones. The cards are individually good but their collective impact is diluted by lack of connective tissue.
**Suggested fix:** Add 1-2 sentences of connecting prose between groups of related cards. For example, after the DeepFloyd IF card and before PixArt-alpha: "DeepFloyd IF's T5 insight spread across the field. But another idea propagated just as widely: that you do not need massive compute if you structure training intelligently." After the Kolors card and before Flux Schnell: "Kolors foreshadowed a trajectory that Z-Image would complete. Meanwhile, the acceleration problem was being solved from a different angle." These transitions transform a catalog into a narrative.

#### [IMPROVEMENT] — The comparison table lacks visual backbone-family grouping

**Location:** Section 8 (The Comparison Table), lines ~672-882
**Issue:** The comparison table is the lesson's centerpiece, with 16 model rows in chronological order. The planning document says "one row per model, chronological order." This is implemented correctly. However, the "Backbone" column contains text labels ("U-Net", "DiT (cross-attn)", "MMDiT", etc.) but there is no visual grouping or color coding. The student just spent time learning the five-family taxonomy with a color-coded Mermaid diagram (amber for U-Net, cyan for DiT, violet for MMDiT, emerald for S3-DiT, rose for pixel-space). That color language is completely absent from the table. The student must mentally re-classify each row.
**Student impact:** The table is harder to scan than it needs to be. The student who just internalized "amber = U-Net, violet = MMDiT" gets no visual reinforcement when scanning 16 rows. The primary organizational insight (few distinct backbone families) is invisible in the table's visual layout. The table and the taxonomy diagram feel disconnected.
**Suggested fix:** Add subtle left-border color coding to each table row matching the Mermaid diagram colors. For example, U-Net rows get a left border in amber, DiT rows in cyan, MMDiT in violet, S3-DiT in emerald. This can be done with a simple `className="border-l-2 border-amber-500"` on the `<tr>`. Alternatively, add a small colored dot or badge in the Backbone column. The goal is not to replace the text but to create visual continuity between the taxonomy diagram and the reference table.

#### [IMPROVEMENT] — Check #1 has four questions but no preamble framing the exercise

**Location:** Section 7 (Check: Taxonomy Placement), lines ~574-667
**Issue:** The section jumps directly from the SectionHeader to the first GradientCard question. There is no framing text explaining what the student should do. The subtitle says "Can you identify the model from its description?" but the student sees four GradientCards with "Question 1," "Question 2," etc. and descriptions. There is no instruction like "For each description, try to identify the backbone family and specific model before revealing the answer." The student might read these passively (read description, immediately click Reveal) rather than actively (try to classify, then check).
**Student impact:** Without explicit instructions to predict before revealing, the check becomes a reading exercise rather than a retrieval exercise. The reveal/predict pattern only works if the student actually commits to a prediction before clicking. The planning document says "Assessment type: predict-and-verify" but the built lesson does not make the predict step explicit.
**Suggested fix:** Add a brief paragraph before the first question: "For each description below, try to identify the backbone family and the specific model before clicking Reveal. The descriptions use only information from the taxonomy above -- backbone type, text encoder strategy, and distinctive features." This primes the student to treat it as an active exercise.

#### [POLISH] — Z-Image paper arxiv link may be incorrect

**Location:** References section, line ~1365
**Issue:** The Z-Image paper is referenced with arxiv ID `2511.22699`. The arxiv ID prefix `2511` indicates a November 2025 paper, but the lesson places Z-Image's release at January 2025. This may be the correct ID if the paper was published later on arxiv, or it may be an incorrect ID. Cannot verify without web access, but the discrepancy is worth noting.
**Student impact:** If the student clicks the link and gets a 404 or a different paper, that erodes trust in the reference. Minor issue since the student is unlikely to follow every link.
**Suggested fix:** Verify the arxiv ID is correct. If the Z-Image paper is known by a different ID, update it.

#### [POLISH] — SD3 and SD3.5 appear as both a combined row and a separate row in the comparison table

**Location:** Section 8 (The Comparison Table), lines ~789-865
**Issue:** Row 9 is "SD3 / SD3.5" with release "Jun 2024". Row 15 is "SD3.5" with release "Oct 2024". This double-listing could be intentional (SD3 as the initial release, SD3.5 as the updated release with new variants), but it creates mild confusion: does the student count this as one model or two? The "SD3 / SD3.5" row already mentions SD3.5 Large and Large Turbo. The separate SD3.5 row adds "Medium/Large/Large Turbo variants; improved training" which is partially redundant.
**Student impact:** Minor confusion. The student might wonder why SD3.5 appears twice. The reference table should have clean, non-overlapping entries.
**Suggested fix:** Either (a) merge into a single "SD3 / SD3.5" row covering both the June 2024 and October 2024 releases (with a note about the variant sizes), or (b) split cleanly: "SD3" for June 2024 and "SD3.5" for October 2024, with the first row NOT mentioning SD3.5. Option (a) is simpler.

#### [POLISH] — The lesson uses "family tree" vocabulary from the plan without the explicit analogy

**Location:** Throughout the lesson, but especially sections 4-5
**Issue:** The planning document proposes a "family tree" analogy: "models have parents, siblings, and cousins. SD v1 and Flux are parent-child (same creators). SD3 and Flux are siblings." This analogy appears in the Modalities Planned table. In the built lesson, the genealogical language ("lineage," "DNA," "same people made them") appears organically but the formal family-tree analogy is never explicitly stated. The student picks up the metaphor implicitly rather than explicitly.
**Student impact:** Very minor. The implicit approach works. The explicit analogy might land more memorably, but it is not essential.
**Suggested fix:** No change strictly necessary. If revising for other reasons, consider adding one sentence making the analogy explicit: "Think of models as having parents, siblings, and cousins -- models share DNA when the same researchers created them."

#### [POLISH] — "Qwen3-4B" naming may cause confusion with Qwen versioning

**Location:** Multiple locations (S3-DiT card, comparison table, text encoder evolution diagram)
**Issue:** The lesson uses "Qwen3-4B" consistently, matching the Z-Image lesson. This is fine for internal consistency. However, if the student later encounters external discussions using "Qwen2.5" or other version numbers, the "3" in "Qwen3-4B" could cause confusion. This is not the lesson's fault -- it is using the Z-Image paper's terminology -- but a brief note could help.
**Student impact:** Negligible. The student already encountered this in the Z-Image lesson and should be calibrated.
**Suggested fix:** No change needed. Internal consistency is maintained. If confusion arises later, it can be addressed then.

### Review Notes

**What works well:**

1. **Taxonomy structure is excellent.** The five-family grouping with color-coded Mermaid diagrams provides genuine organizational value. The student goes from "18 confusing names" to "5 families" in one visual. This is the lesson's core payoff and it lands.

2. **Scope discipline is strong.** The lesson stays firmly in CONSOLIDATE territory. No new concepts are introduced. Every technology referenced has been taught. The ConstraintBlock is thorough and the lesson respects every boundary it sets.

3. **The three evolution lines (Section 6) are well-designed.** Separating backbone, text encoding, and training objective into independent evolution lines with their own Mermaid diagrams is pedagogically sound. It reveals that these axes evolved somewhat independently and that a model is a point in a 3D space. The InsightBlock "Three Axes, One Model" captures this cleanly.

4. **The "five questions" framework (Section 13) provides lasting practical value.** This is the "teach a person to fish" section. The student leaves with a reusable tool, not just a static reference. The hypothetical model exercise (Check #3) tests this directly.

5. **Connection to prior mental models is thorough.** "The U-Net's last stand," "Tokenize the image," "One room, one conversation," "Translate once, then speak together" -- these are all referenced at the right moments, creating continuity with previous lessons.

6. **The SD v2 cautionary tale is well-deployed.** Using it as a negative example for "better architecture does not guarantee adoption" is the right lesson to draw, and it surfaces naturally in both the timeline section and the innovations section.

**Systemic observation:** The lesson is well-structured for breadth but could benefit from slightly more narrative connective tissue in the long middle sections (Sections 8-9). Survey lessons have an inherent risk of feeling like catalogs. The first half (hook, timeline, taxonomy, evolution lines) has strong narrative flow. The second half (table, innovations) needs the same treatment.

---

## Review — 2026-02-23 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 3

### Verdict: PASS

All four iteration 1 IMPROVEMENT findings have been cleanly resolved. The fixes are well-executed and introduce no new issues. The lesson is now a strong consolidation survey that organizes the student's existing knowledge into a coherent landscape map. Three minor polish-level observations remain but none affect the lesson's pedagogical effectiveness.

### Iteration 1 Fix Status

1. **Misconception #3 (open weight vs open source) -- RESOLVED.** A WarningBlock titled "Open Weight != Open Source" was added as an aside in Section 11 (The Closed Models). It explains the distinction concisely with the Flux.1 Dev (non-commercial) vs Flux.1 Schnell (Apache 2.0) concrete example. Well-placed and appropriately scoped -- brief clarification, not a deep dive, exactly per the planning doc's scope boundary.

2. **Middle sections feel like a catalog -- RESOLVED.** Two connecting prose paragraphs were added between GradientCards in Section 9 (Notable Innovations). The first (after DeepFloyd IF, before PixArt-alpha) bridges the T5 insight to training efficiency. The second (after Kolors, before Flux Schnell) bridges the LLM-as-encoder trajectory to the acceleration problem. Both transitions are concise and genuinely connect the surrounding cards into a narrative rather than just providing filler. The section now reads as a story with progression rather than a catalog of disconnected cards.

3. **Comparison table lacks visual backbone-family grouping -- RESOLVED.** Every table row now has a `border-l-2` class with the backbone family color (amber for U-Net, rose for pixel-space/cascaded, cyan for DiT, violet for MMDiT, emerald for S3-DiT). An explanatory paragraph before the table maps each color to its family. This creates strong visual continuity between the Mermaid taxonomy diagram and the reference table.

4. **Check #1 has no preamble framing the exercise -- RESOLVED.** A framing paragraph was added before the first question: "For each description below, commit to a prediction before clicking Reveal. Identify the backbone family... and the text encoder strategy first, then name the specific model." This primes the student for active retrieval rather than passive reading. The instruction is specific enough to guide behavior (identify backbone first, then text encoder, then model name).

### Findings

#### [POLISH] — Check #2 lacks the same predict-before-reveal framing that was added to Check #1

**Location:** Section 10 (Check: Innovation Attribution), lines ~1028-1097
**Issue:** The iteration 1 fix added a framing paragraph to Check #1 instructing the student to "commit to a prediction before clicking Reveal." Check #2 (Innovation Attribution) and Check #3 (Apply the Framework) do not have equivalent framing. Check #2 has a SectionHeader subtitle ("Can you trace which model pioneered which idea?") which implies active recall but does not explicitly instruct the student to predict before revealing. Check #3's GradientCard includes "Place this model in the taxonomy using the five questions" which serves as adequate instruction. The inconsistency is only between Check #1 and Check #2.
**Student impact:** Minor. By Check #2, the student has already internalized the predict-before-reveal pattern from Check #1. The risk of passive reading is low by this point in the lesson.
**Suggested fix:** Add a one-sentence preamble before Check #2's first question: "For each question, commit to an answer before clicking Reveal." Brief, consistent with Check #1's pattern.

#### [POLISH] — The "family tree" analogy explanation could be one sentence shorter

**Location:** Section 5 (Architecture Taxonomy), lines 279-289
**Issue:** The paragraph introducing the taxonomy says: "Think of it as a family tree: models have parents, siblings, and cousins -- they share DNA when the same researchers or the same architectural ideas produced them. Once you identify the backbone, most of the mystery dissolves." This is followed by: "The families are defined by how the denoising network processes image features and how it incorporates text conditioning." The second sentence is a slightly flat restatement of what the student is about to see in the taxonomy diagram. It adds precision but reduces the momentum of the family-tree analogy.
**Student impact:** Negligible. The student reads one extra sentence that is informative but slightly anticlimactic after the analogy.
**Suggested fix:** No change required. The sentence is factually useful as a framing device for the taxonomy. If revising for other reasons, consider merging it with the previous sentence: "The families are defined by how the denoising network processes images and incorporates text -- once you identify the backbone, most of the mystery dissolves."

#### [POLISH] — Stable Cascade release date inconsistency between timeline and table

**Location:** Mermaid timeline (line 197) vs comparison table (line 787)
**Issue:** The Mermaid timeline shows "Dec: Stable Cascade research (Stability AI)" in the 2023 section. The comparison table lists Stable Cascade's release as "Feb 2024." Both are arguably correct: the Wurstchen research paper predates the official Stable Cascade release. However, the two sections show different dates for the same model without explaining the discrepancy.
**Student impact:** Minor. A careful student might notice the date difference (Dec 2023 vs Feb 2024) and be momentarily confused. Most students will not cross-reference the timeline and table closely enough to notice.
**Suggested fix:** Change the timeline entry to "Dec : Stable Cascade research (Stability AI)" -> "Feb : Stable Cascade (Stability AI)" and move it to the 2024 section to match the table. Alternatively, keep both dates and add "(research)" to the timeline and "(release)" to the table to make the distinction explicit. The timeline already says "research" which helps.

### Review Notes

**What works well (in addition to iteration 1 observations, which remain valid):**

1. **Iteration 1 fixes are clean.** All four improvements were implemented precisely as suggested without over-engineering. The open weight vs open source aside is concise and well-placed. The transition sentences between innovation cards are brief and genuinely connective. The table border colors create strong visual continuity. The Check #1 preamble primes active recall without being heavy-handed.

2. **The color-coded table is a genuine upgrade.** The left-border visual coding transforms the comparison table from a 16-row text reference into a visually scannable artifact. The student can now glance at the table and immediately see clusters of amber (U-Net era), then cyan and violet (transformer pivot), then emerald (S3-DiT frontier). This visual pattern reinforces the taxonomy.

3. **The open weight vs open source aside is well-scoped.** It could have been over-done (deep licensing discussion) but it stays at exactly the right level: one concrete example (Flux Dev vs Schnell), the core distinction (weights vs full reproducibility), and the actionable takeaway ("check the license, not just the download button").

4. **No new issues introduced by the fixes.** The four changes integrate smoothly into the existing lesson without creating inconsistencies, breaking narrative flow, or adding cognitive load. This is good revision craftsmanship.

**Overall assessment:** The lesson is now ready to ship. It is a strong consolidation survey that effectively organizes the student's existing knowledge. The taxonomy structure is clear, the visual language is consistent (color-coded families across diagrams and table), the checks provide active recall opportunities, and the five-question framework gives lasting practical value. The three polish findings are genuinely minor and do not affect pedagogical effectiveness.
