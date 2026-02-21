# Module 7.4: Next-Generation Architectures -- Record

**Goal:** The student can trace the architectural evolution from Stable Diffusion v1.5 through SDXL to the Diffusion Transformer (DiT) and SD3/Flux, understanding why the field moved from refined U-Nets to transformers and how flow matching, joint text-image attention, and stronger text encoders converge in the current frontier architecture.
**Status:** Complete (4 of 4 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Dual text encoders (CLIP ViT-L [77,768] + OpenCLIP ViT-bigG [77,1280] concatenated to [77,2048]) | DEVELOPED | sdxl | Tensor shapes traced explicitly. Concatenation along embedding dimension before cross-attention—single K/V path, wider source. Explicitly distinguished from IP-Adapter's decoupled cross-attention via ComparisonRow. |
| Pooled text embedding for global conditioning ([1280] from OpenCLIP ViT-bigG CLS token, injected via adaptive norm alongside timestep) | INTRODUCED | sdxl | Explained as global conditioning channel complementing per-token cross-attention. Two injection points, two purposes—per-token for "what goes where," pooled for "overall vibe." |
| Micro-conditioning (original_size, crop_top_left, target_size as conditioning inputs via adaptive norm) | INTRODUCED | sdxl | Problem-before-solution: three bad options for multi-resolution training (discard data, upscale artifacts, crop artifacts). Solution: tell the model the training image's context. Injected via same adaptive norm pathway as timestep. At inference, set to high-quality values. Extends the WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE framework to AT-WHAT-QUALITY. |
| Refiner model as specialized img2img (second U-Net denoising from t_switch to t=0) | INTRODUCED | sdxl | Explicitly connected to img2img mechanism from 6.5.2—"same trail, different hiker for the final stretch." Base handles composition (high-noise timesteps), refiner handles fine detail (low-noise timesteps). Optional polish, not required. |
| SDXL as U-Net ceiling / narrative positioning (every improvement is conditioning or scale, not new architecture) | INTRODUCED | sdxl | "The U-Net's last stand"—framing for the module arc. Five changes (dual encoders, larger U-Net, higher resolution, refiner, micro-conditioning) all go IN, AROUND, or ALONGSIDE the U-Net. Limitations identified: no scaling recipe, limited text-image interaction, convolutional inductive biases. Sets up DiT. |
| SDXL resolution scaling (1024x1024 base, [4,128,128] latent, quadratic attention cost) | INTRODUCED | sdxl | Concrete token counts: 256 at 16x16 (SD v1.5) vs 1,024 at 32x32 (SDXL), 4,096 at 64x64. O(n^2) attention means ~16x compute per layer. Training on 1024x1024 images required, not just architectural scaling. |
| SDXL adapter compatibility (ControlNet, IP-Adapter, LoRA work with SDXL but require SDXL-specific versions due to dimension changes) | MENTIONED | sdxl | Brief ecosystem positioning—same mechanisms, different weight dimensions. SD v1.5 adapters incompatible due to matrix shape mismatches (e.g., W_K/W_V project from 2048 not 768). |
| Patchify operation (split latent [C,H,W] into non-overlapping p×p patches, flatten, linear project to d_model) | DEVELOPED | diffusion-transformers | Full tensor shape trace: [4,32,32] → 256 patches of dim 16 → [256, 1152] via nn.Linear. Structural correspondence to text tokenization made explicit via ComparisonRow: words→BPE→embedding vs spatial grid→patches→linear projection. Both produce [n, d_model] sequences. "Tokenize the image" analogy established. |
| Patch size as resolution-compute tradeoff (L = (H/p) × (W/p), smaller p = more tokens = O(L²) attention cost) | DEVELOPED | diffusion-transformers | Three concrete examples: p=8 gives 64 tokens, p=4 gives 256, p=2 gives 1024 on a [4,64,64] latent. Connected to SDXL token counts (256 at 16×16, 1024 at 32×32). "Patch size is DiT's resolution knob." |
| Positional embeddings for patches (learned, extending GPT-2 positional encoding to 2D patch positions) | INTRODUCED | diffusion-transformers | Explained as DiT's only spatial prior—without them, the transformer treats the sequence as an unordered set. Everything else about spatial relationships is learned from data. Connected to GPT-2's learned positional embeddings from Series 4. |
| DiT block as standard transformer block (MHA + FFN + residual + norm, no convolutions, no encoder-decoder hierarchy) | DEVELOPED | diffusion-transformers | Every component explicitly checked off against Series 4 knowledge: self-attention from 4.2.4, FFN from 4.2.5, residual connections from 4.2.5, pre-norm from 4.2.5. "You already know every component except the conditioning mechanism." Side-by-side ComparisonRow: U-Net (multi-resolution, skip connections, convolutions, attention at middle resolutions, ad hoc scaling) vs DiT (single resolution, no skip connections, no convolutions, attention everywhere, systematic scaling). |
| adaLN-Zero conditioning (adaptive layer norm with scale γ, shift β, and gate α per sub-layer, 6 parameters per block from conditioning vector c) | DEVELOPED | diffusion-transformers | Full formula traced: c → MLP → (γ₁,β₁,α₁,γ₂,β₂,α₂). MHA sub-layer: h = γ₁·LN(x)+β₁, h = MHA(h), x' = x+α₁·h. FFN sub-layer: same pattern with γ₂,β₂,α₂. Connected to adaptive group norm from 6.3.2 (same idea, different normalization type) with explicit ComparisonRow showing the differences. |
| Zero-initialized gate in adaLN-Zero (α=0 at initialization makes each block an identity function) | DEVELOPED | diffusion-transformers | Math shown explicitly: x' = x + 0·MHA(…) = x. Connected to ControlNet zero convolution (7.1.1) and LoRA B-matrix zero initialization—same safety pattern: start contributing nothing, learn what to add. "The 'Zero' in adaLN-Zero is not just a name—it is the critical design choice." |
| DiT scaling recipe (increase d_model and N—same two knobs as GPT-2→GPT-3) | INTRODUCED | diffusion-transformers | DiT model family traced: DiT-S (384/12/33M) through DiT-XL (1152/28/675M). Connected to GPT-2→GPT-3 scaling from 4.2.6. Directly addresses SDXL's "no scaling recipe" limitation. "Two knobs, not twenty." Scaling analogy: LEGO bricks (transformer) vs house renovation with custom parts (U-Net). |
| DiT-XL/2 as state-of-the-art class-conditional ImageNet generation (FID 2.27 on ImageNet 256×256) | INTRODUCED | diffusion-transformers | Positioned as evidence for scaling argument. Emphasis on the trend (bigger = better, smooth curve) over the specific number. |
| Convolutions vs attention tradeoff at scale (CNNs data-efficient at small scale, transformers more capable at large scale) | INTRODUCED | diffusion-transformers | Honest presentation of the tradeoff: convolutions embed useful spatial priors (local connectivity, translational equivariance). At modern training scale (hundreds of millions of images), learned spatial relationships outperform hard-coded assumptions. Mirrors NLP: RNNs had useful sequential biases, but transformers won at scale. |
| DiT replaces only the denoising network (VAE, text encoder, noise schedule, sampling algorithm unchanged) | INTRODUCED | diffusion-transformers | Explicit pipeline trace: 11 steps with clear annotation of which are DiT-specific (steps 4-8) and which are unchanged from Series 6 (steps 9-11). "Same pipeline, different denoising network." WarningBlock: do not think of DiT as a complete replacement of the SD pipeline. |
| DiT as convergence of transformers (Series 4) and latent diffusion (Series 6) | INTRODUCED | diffusion-transformers | Two GradientCards showing the knowledge threads meeting. "Of Course" chain: fixed-size tensor → patches → tokens → transformer. The lesson is designed to feel inevitable given the student's existing knowledge. |
| MMDiT joint attention (concatenate text tokens [77, d_model] and image patch tokens [256, d_model] into one [333, d_model] sequence; standard self-attention on the combined sequence; provides four attention types simultaneously: image-to-text, text-to-image, image-to-image, text-to-text) | DEVELOPED | sd3-and-flux | Core new concept #1. Taught via "one room, one conversation" analogy contrasting with cross-attention's "one-way mirror" (image reads text, text is frozen). Concrete tensor shape trace: 77 text + 256 image = 333 combined tokens, [333, 333] attention matrix. ComparisonRow: cross-attention (two operations per block, one-directional) vs joint attention (one operation, bidirectional). Four attention types enumerated explicitly. Design challenge in `<details>` reveal: three options for adding text to DiT, option 3 (concatenation) is what SD3 chose. Addresses misconception that MMDiT is more complex than cross-attention ("simpler, not more complex"). |
| Modality-specific Q/K/V projections in MMDiT (separate W_Q, W_K, W_V matrices for text and image tokens; projections concatenated before shared attention; separate FFN layers per modality after attention; "shared listening, separate thinking") | DEVELOPED | sd3-and-flux | Key architectural nuance distinguishing MMDiT from naive concatenation. CodeBlock trace: text tokens -> text W_Q/W_K/W_V, image tokens -> image W_Q/W_K/W_V, concatenate Q/K/V, shared attention, split output, modality-specific FFNs. "French speaker and Japanese speaker" negative example for why shared projections would fail. WarningBlock: "Not Naive Concatenation." Contrast with IP-Adapter's approach (separate K/V sources, weighted addition) noted. |
| T5-XXL as text encoder for diffusion (4.7B parameter encoder-decoder transformer providing deep linguistic understanding alongside CLIP's visual alignment; produces [77, 4096] embeddings vs CLIP's [77, 768]; trained on text alone, not image-text pairs) | INTRODUCED | sd3-and-flux | Gap fill from MENTIONED (4.2.6) to INTRODUCED. Problem-before-solution: CLIP trained on image-text pairs, weak at compositional reasoning. T5 trained on text alone, captures linguistic structure. CodeBlock comparing three encoders: CLIP ViT-L (123M, [77,768]), OpenCLIP ViT-bigG (354M, [77,1280]), T5-XXL (4.7B, [77,4096]). ComparisonRow: CLIP encoders vs T5-XXL (training data, embedding alignment, strengths). Addresses misconception that T5 replaces CLIP ("T5 does not replace CLIP. All three encoders contribute simultaneously."). |
| Triple text encoder setup in SD3 (CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL; CLIP pooled embeddings for global adaLN-Zero conditioning, per-token embeddings from all three for joint attention text token sequence) | INTRODUCED | sd3-and-flux | Extension of SDXL dual encoder pattern. Progression traced: SD v1.5 (one encoder, [77,768]) -> SDXL (two encoders, [77,2048]) -> SD3 (three encoders including T5 [77,4096]). Dual-path text conditioning: per-token path via joint attention (spatially varying) + global path via adaLN-Zero (pooled CLIP). Connected to SDXL's same dual-path pattern (per-token via cross-attention, pooled via adaptive norm). |
| Rectified flow applied in SD3 (same flow matching training objective from 7.2.2 at scale; straight-line interpolation, velocity prediction, MSE loss; 20-30 inference steps vs 50+ for DDPM) | INTRODUCED | sd3-and-flux | Explicitly framed as application, not new content: "You already know everything about flow matching that SD3 uses." Same interpolation, velocity target, MSE loss from 7.2.2. Rectified flow (trajectory straightening via model-aligned pairs) applied for even straighter aggregate trajectories. Practical payoff: 20-30 steps. |
| Logit-normal timestep sampling (biasing training timestep distribution toward intermediate noise levels around t=0.5 where denoising is hardest; a training optimization, not a fundamental change to the objective) | INTRODUCED | sd3-and-flux | Minor new detail. Intermediate timesteps are where the model makes the most important compositional decisions. TipBlock emphasizes this is an optimization, not a paradigm change. |
| SD3/Flux as convergence architecture (every component traces to a prior lesson: transformers from Series 4, latent diffusion from Series 6, flow matching from 7.2, DiT from 7.4.2, text encoding from 6.3.3/7.4.1/this lesson) | INTRODUCED | sd3-and-flux | The emotional and intellectual core of the lesson and the series conclusion. Five-thread convergence map (GradientCard grid). Annotated 13-step SD3 pipeline with every step traced to its source lesson. InsightBlock: "You Already Knew All of This." Three independent advances summarized (architecture DiT->MMDiT, training DDPM->flow matching, text encoding CLIP->CLIP+T5). |
| SD3 vs Flux architectural positioning (both MMDiT family; Flux uses single-stream blocks in later layers, drops second CLIP encoder, available as Dev/Schnell/Pro variants) | MENTIONED | sd3-and-flux | ComparisonRow with concrete distinguishing details. For vocabulary positioning: when the student encounters "Flux.1 Schnell" or "SD3.5 Large," they know these are MMDiT variants with flow matching training. |
| S3-DiT single-stream architecture (shared Q/K/V projections and shared SwiGLU FFN across all token types after lightweight refiner layers; eliminates ~50% parameter overhead of MMDiT's dual-stream per-layer duplication) | INTRODUCED | z-image | "Translate once, then speak together." Parameter counting makes the efficiency argument concrete: for d_model=3840 and 30 layers, single-stream saves ~2.5B parameters vs dual-stream. Refiner layers (2 per modality) pre-process embeddings into a shared representation space, enabling shared projections. Misconception addressed: "single-stream does not mean no modality awareness--the awareness is concentrated upfront." ComparisonRow: MMDiT dual-stream vs S3-DiT single-stream. |
| Refiner layers for modality pre-processing (2 lightweight transformer layers per modality that map raw embeddings into a shared representation space before the main 30-layer backbone) | INTRODUCED | z-image | Solves the same problem as MMDiT's separate projections (text and image embeddings in different spaces) but with different allocation: 2 dedicated layers upfront vs duplication across 30 layers. Connected to the "French/Japanese speaker" negative example from sd3-and-flux. |
| Qwen3-4B as text encoder (single LLM replacing triple encoder setup CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL; richer compositional reasoning, bilingual support, instruction-following baked into embeddings) | INTRODUCED | z-image | Text encoder evolution trajectory traced: SD v1.5 (1 CLIP) -> SDXL (2 CLIPs) -> SD3 (2 CLIPs + T5) -> Z-Image (1 LLM). "Convergence to simplicity"--from multiple specialized encoders back to one powerful encoder. Misconception addressed: "LLM provides embeddings, not chat responses--richer embeddings from diverse language training, not inference-time reasoning." |
| Prompt Enhancer (PE) integrated during supervised fine-tuning (VLM that expands short prompts with descriptive detail; compensates for 6B model's limited world knowledge; zero inference cost because reasoning baked in during SFT) | INTRODUCED | z-image | Concrete example: "West Lake at sunset" expanded with visual details. Key insight: no inference-time cost because the PE teaches during training, model generates independently at inference. |
| 3D Unified RoPE (temporal axis d_t=32 for text sequential position, height axis d_h=48 and width axis d_w=48 for image spatial position; orthogonal axes mean cross-modal attention has no positional bias) | INTRODUCED | z-image | Extension of 1D RoPE from Series 4. Concrete dimension allocations shown. Text tokens: temporal=position, spatial=0. Image tokens: temporal=0, spatial=row,col. Cross-modal attention depends on content, not position. Generalizes to arbitrary resolutions without retraining (continuous rotation frequencies vs learned embeddings). |
| adaLN with shared low-rank down-projection (shared low-rank compression of conditioning vector + layer-specific up-projections; LoRA pattern applied to timestep conditioning) | INTRODUCED | z-image | Parameter-efficient extension of adaLN-Zero from diffusion-transformers. Connected to LoRA's low-rank factorization pattern. |
| DMD / Distribution Matching Distillation (third distillation paradigm: CFG-augmented regression + distribution matching loss; distinct from consistency distillation and adversarial distillation) | INTRODUCED | z-image | Gap fill: student had NOT TAUGHT. Framed as third entry in distillation taxonomy from Module 7.3. Two components: CFG augmentation ("spear") pushes quality, distribution matching ("shield") prevents mode collapse. |
| Decoupled-DMD (spear/shield decomposition with separate noise schedules; spear at high noise for large-scale composition, shield at low noise for fine-detail distribution matching; eliminates artifacts from forced coupling) | INTRODUCED | z-image | Z-Image's own metaphor. Pseudocode showing coupled vs decoupled training steps. Conceptual advance: understand WHY the method works, then optimize each component independently. Misconception addressed: "not a minor tweak--a conceptual reframing that enables DMDR." |
| DMDR / DMD + Reinforcement Learning (combines Decoupled-DMD with DPO stage for text rendering/counting and GRPO stage for aesthetics/photorealism; breaks the teacher ceiling by using DMD as regularizer and RL as quality driver) | INTRODUCED | z-image | "The teacher becomes the guardrail, not the ceiling." DPO and GRPO directly transferred from Series 5 to image generation. DMD prevents reward hacking (same problem from Series 5). Two-stage RL: DPO for objective correctness (binary preferences), GRPO for subjective quality (scalar rewards). |
| Z-Image performance positioning (6.15B params vs Flux 32B; #1 open-source on Artificial Analysis; 87.4% Good+Same rate vs Flux Dev; Z-Image Turbo: 8 steps, sub-second on H800, <16GB VRAM) | MENTIONED | z-image | "Simplicity beats complexity"--competitive performance from architecture simplification + training innovation + post-training, not architectural novelty. |
| Three-phase training curriculum (Phase 1: low-res pre-training 256x256, Phase 2: omni pre-training arbitrary resolution, Phase 3: PE-aware SFT high-quality data) | MENTIONED | z-image | Brief overview. Progressive resolution training mirrors SDXL's approach but formalized. $630K total training cost (314K H800 GPU hours). |

## Per-Lesson Summaries

### sdxl (Lesson 1)
**Status:** Complete
**Cognitive load:** BUILD
**Notebook:** `notebooks/7-4-1-sdxl.ipynb`
**Review:** PASS on iteration 2/3 (iteration 1 had 2 IMPROVEMENT + 1 POLISH, all fixed)

**Concepts taught:** Dual text encoders (DEVELOPED), micro-conditioning (INTRODUCED), refiner model (INTRODUCED), SDXL as U-Net ceiling (INTRODUCED), resolution scaling (INTRODUCED), adapter compatibility (MENTIONED).

**Mental models established:**
- "The U-Net's last stand"—SDXL as the final refinement of the U-Net paradigm before DiT replaces it. Every improvement is about what goes IN (dual encoders), AROUND (refiner), or ALONGSIDE (micro-conditioning) the U-Net.
- AT-WHAT-QUALITY as a fifth conditioning dimension extending the WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE framework from 7.1.3.
- "Same trail, different hiker"—refiner as img2img with a specialized model (extending the "starting mid-hike" analogy from 6.5.2).

**Analogies used:**
- "Two translators, one microphone"—dual encoders concatenate into one embedding, not separate channels.
- "Same trail, different hiker for the final stretch"—refiner as img2img with a specialist for the low-noise denoising range.
- Extended the four conditioning channels to five: WHEN (timestep), WHAT (text), WHERE (ControlNet), WHAT-IT-LOOKS-LIKE (IP-Adapter), AT-WHAT-QUALITY (micro-conditioning).

**How concepts were taught:**
- Dual encoders: tensor shape trace mirroring 6.4.1 format (CLIP ViT-L [77,768] + OpenCLIP ViT-bigG [77,1280] = [77,2048]). ComparisonRow explicitly contrasting SDXL concatenation vs IP-Adapter decoupled attention. Pooled embedding explained as global conditioning via adaptive norm.
- Micro-conditioning: problem-before-solution with three bad options (discard, upscale, crop) presented as GradientCards. Solution: three metadata inputs via adaptive norm. ComparisonRow comparing timestep conditioning vs micro-conditioning injection.
- Refiner: connected to img2img (6.5.2) with explicit "this is the img2img mechanism you already know." Two-model pipeline described as base (t=T to t_switch) + refiner (t_switch to t=0).
- Resolution scaling: concrete token counts and quadratic cost. Training data requirement alongside architectural scaling.
- Pipeline comparison: ComparisonRow showing SD v1.5 vs SDXL pipeline stages side-by-side.

**What is NOT covered:**
- Training SDXL from scratch
- SDXL Turbo or LCM-LoRA for SDXL (covered in Module 7.3)
- Every internal U-Net detail (exact channel counts, attention placement)
- ControlNet or IP-Adapter for SDXL (same concepts from Module 7.1)
- VAE changes between SD v1.5 and SDXL
- DiT architecture (next lesson)

### diffusion-transformers (Lesson 2)
**Status:** Complete
**Cognitive load:** STRETCH
**Notebook:** `notebooks/7-4-2-diffusion-transformers.ipynb`
**Review:** PASS on iteration 2/3 (iteration 1 had 3 IMPROVEMENT + 2 POLISH, all fixed; iteration 2 found only 2 POLISH)

**Concepts taught:** Patchify operation (DEVELOPED), patch size as resolution-compute tradeoff (DEVELOPED), positional embeddings for patches (INTRODUCED), DiT block as standard transformer block (DEVELOPED), adaLN-Zero conditioning (DEVELOPED), zero-initialized gate (DEVELOPED), DiT scaling recipe (INTRODUCED), DiT-XL/2 results (INTRODUCED), convolutions vs attention tradeoff at scale (INTRODUCED), DiT replaces only the denoising network (INTRODUCED), DiT as convergence of two knowledge threads (INTRODUCED).

**Mental models established:**
- "Tokenize the image"—patchify is to images what tokenization is to text. Words→tokens→embeddings parallels spatial grid→patches→linear projection. Both produce [n, d_model] sequences that the transformer processes identically.
- "Two knobs, not twenty"—scaling DiT means increasing d_model and N (same recipe as GPT-2→GPT-3). Contrast with U-Net where doubling parameters requires many ad hoc engineering decisions.
- "Same pipeline, different denoising network"—DiT replaces only the U-Net. VAE, text encoder, noise schedule, and sampling are unchanged.
- "Of Course" chain for DiT—(1) denoising network processes fixed-size tensor, (2) fixed-size tensor can be split into patches, (3) patches are tokens, (4) you know the best architecture for token sequences, (5) of course the field replaced the U-Net with a transformer.

**Analogies used:**
- "Tokenize the image"—structural correspondence between text tokenization pipeline and image patchify pipeline, developed via side-by-side ComparisonRow.
- "Two knobs, not twenty" / LEGO vs house renovation—scaling a transformer is like stacking LEGO (increase brick size or add layers), scaling a U-Net is like renovating a house with custom parts (each room is different).
- Zero-initialization safety pattern—connected adaLN-Zero's α=0 to ControlNet zero convolution and LoRA B-matrix initialization. "Start contributing nothing, learn what to add."

**How concepts were taught:**
- Patchify: concrete tensor shape trace mirroring Series 6 format ([4,32,32] → 256 patches → [256,1152]). ComparisonRow mapping text pipeline (words→BPE→embedding) to image pipeline (spatial grid→patches→projection). Three concrete sequence length examples at different patch sizes.
- DiT block: checklist of components from Series 4 (MHA, FFN, residual, norm) all identified as known. ComparisonRow: U-Net architecture (7 properties) vs DiT architecture (7 properties). InsightBlock: "notice what DiT removes vs what it keeps."
- adaLN-Zero: problem statement (how to condition a transformer block), connection to AdaGN (same idea, different norm type), formula trace with all 6 parameters per block, zero-initialization math showing identity property. ComparisonRow: AdaGN (2 params, no gate, standard init) vs adaLN-Zero (3 params per sub-layer, gate, zero init).
- Scaling: SDXL "no scaling recipe" limitation restated, transformer "two knobs" recipe introduced, DiT model family table (DiT-S through DiT-XL), connection to GPT-2→GPT-3. Honest tradeoff: CNNs data-efficient at small scale, transformers win at modern training scale.
- Pipeline: 11-step pipeline trace with clear annotation of which steps are DiT-specific (4-8) and which are unchanged from Series 6 (9-11). Three GradientCards previewing what DiT enables (MMDiT, architecture-agnostic training, clear scaling path).

**What is NOT covered:**
- Text conditioning in DiT (original DiT is class-conditional on ImageNet, not text-conditioned)
- ViT pretraining on classification tasks (DiT trained from scratch on diffusion)
- Full DiT training from scratch (too compute-intensive)
- Every DiT variant (U-ViT, MDT, etc.—only the original DiT from Peebles & Xie 2023)
- SD3/Flux architecture, MMDiT, joint text-image attention (next lesson)
- Detailed training procedures, learning rate schedules, data requirements
- Other conditioning variants from the DiT paper (in-context, cross-attention)—only adaLN-Zero

### sd3-and-flux (Lesson 3)
**Status:** Complete
**Cognitive load:** BUILD
**Notebook:** `notebooks/7-4-3-sd3-and-flux.ipynb`
**Review:** PASS on iteration 2/3 (iteration 1 had 3 IMPROVEMENT + 3 POLISH, all improvements fixed)

**Concepts taught:**
- MMDiT joint attention (DEVELOPED)--concatenate text + image tokens, standard self-attention on combined sequence, four attention types (image-to-text, text-to-image, image-to-image, text-to-text), replaces cross-attention with one simpler operation
- Modality-specific Q/K/V projections (DEVELOPED)--separate projections per modality, shared attention computation, separate FFN layers, "shared listening, separate thinking"
- T5-XXL as text encoder (INTRODUCED)--gap fill from MENTIONED, 4.7B params, deep linguistic understanding complementing CLIP's visual alignment
- Triple text encoder setup (INTRODUCED)--CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL, dual-path text conditioning (per-token via joint attention, global via adaLN-Zero)
- Rectified flow in SD3 (INTRODUCED)--same flow matching objective from 7.2.2 applied at scale, 20-30 inference steps
- Logit-normal timestep sampling (INTRODUCED)--bias training toward intermediate noise levels, a training optimization
- SD3/Flux as convergence architecture (INTRODUCED)--every component traces to a prior lesson, five knowledge threads converge
- SD3 vs Flux positioning (MENTIONED)--vocabulary-level distinction for real-world encounters

**Mental models established:**
- "One room, one conversation"--cross-attention is two rooms with a one-way mirror (image watches text). Joint attention puts everyone in one room: text hears image, image hears text, symmetric interaction. Simpler (one operation instead of two) and richer (bidirectional).
- "Shared listening, separate thinking"--text and image tokens attend together through shared attention (same "room") but maintain separate Q/K/V projections and separate FFNs. They "speak their own language" but "hear each other."
- "Convergence, not revolution"--SD3/Flux is not a new paradigm but the combination of concepts built across the entire course. Every component traces to a lesson already completed. The frontier IS the student's understanding, combined.

**Analogies used:**
- "One room, one conversation" vs "one-way mirror" (joint attention vs cross-attention for text-image interaction)
- "French speaker and Japanese speaker using the same dictionary" (why shared projections would fail--modalities need their own projection matrices)
- "Shared listening, separate thinking" (shared attention, modality-specific processing)
- Contrast with IP-Adapter's "two reference documents, one reader" (IP-Adapter keeps sources separate, MMDiT merges them)

**How concepts were taught:**
- Recap: three concepts reactivated (DiT architecture from 7.4.2, flow matching from 7.2.2, cross-attention for text conditioning from 6.3.4). Transition: "DiT solved the architecture problem. Flow matching solved the training objective. But DiT is class-conditional--no text. And cross-attention is one-directional."
- Hook: five-thread convergence map (GradientCard grid: transformers, latent diffusion, flow matching, DiT, text encoding). Design challenge in `<details>` reveal: three options for adding text to DiT (cross-attention, adaLN-Zero, concatenation). "Option 3 is what SD3 and Flux do."
- Text encoder problem: CLIP limitations revisited (contrastive training, weak at composition). T5-XXL as solution (4.7B params, trained on text alone). CodeBlock comparing three encoders. Why keep CLIP alongside T5 (complementary: visual alignment vs linguistic understanding). ComparisonRow. Misconception addressed: "T5 does not replace CLIP."
- Check #1: two predict-and-verify questions (total model size, what dropping CLIP would lose).
- MMDiT joint attention: cross-attention limitation (one-directional, two operations per block). The simple idea: concatenate and self-attend. Concrete shape trace (77+256=333 tokens). Four attention types enumerated. ComparisonRow: cross-attention vs joint attention. "One room, one conversation" analogy. InsightBlock: "Simpler, not more complex."
- Check #2: three predict-and-verify questions (compute cost comparison, why split after attention, "treats text and image identically" misconception).
- MMDiT block detail: modality-specific projections (separate W_Q/W_K/W_V per modality). CodeBlock trace through full block. Negative example (naive concatenation with shared projections). "Shared listening, separate thinking." IP-Adapter contrast. Timestep conditioning via adaLN-Zero unchanged from DiT. Dual-path text conditioning (per-token + global) mirroring SDXL.
- Check #3: two predict-and-verify questions (adaLN-Zero parameter count, IP-Adapter with MMDiT).
- Flow matching in practice: SD3 uses identical training objective from 7.2.2. "You already know everything." Rectified flow application. Logit-normal timestep sampling as minor optimization. TipBlock: "a training optimization, not a fundamental change."
- Check #4: two predict-and-verify questions (what changes at scale, DDPM vs SD3 training comparison).
- The convergence: full 13-step SD3 pipeline annotated with source lessons. InsightBlock: "You Already Knew All of This." Three independent advances (architecture, training, text encoding). SD3 vs Flux brief positioning with ComparisonRow.
- Series conclusion: 11-lesson evolution traced (7.1 conditioning, 7.2 score-based perspective, 7.3 fast generation, 7.4 architecture). ModuleCompleteBlock.
- Practice: four notebook exercises (Guided: pipeline inspection, Guided: joint attention structure, Supported: generation with step count variation, Independent: convergence pipeline trace).

**Misconceptions addressed:**
1. "MMDiT's joint attention is more complex than cross-attention"--ComparisonRow showing one attention operation replaces two. "Simpler, not more complex." The "M" in MMDiT sounds exotic but the mechanism is standard self-attention on a multimodal sequence.
2. "Flow matching in SD3 is new content"--explicit statement: "You already know everything about flow matching that SD3 uses." Same interpolation, velocity target, MSE loss from 7.2.2. Only logit-normal timestep sampling is new (minor detail).
3. "T5-XXL replaces CLIP in SD3/Flux"--ComparisonRow showing complementary information. CLIP provides image-aligned embeddings and pooled CLS for adaLN-Zero. T5 provides deep linguistic understanding. All three encoders contribute simultaneously.
4. "Joint attention means the model treats text and image identically"--separate Q/K/V projections and separate FFNs. "Shared listening, separate thinking." Not naive concatenation. Check #2 Q3 directly tests this.
5. "SD3/Flux is a completely new system unrelated to prior knowledge"--convergence theme throughout. Five-thread GradientCard grid. Annotated 13-step pipeline. "Every step traces to a lesson you completed."

**What is NOT covered:**
- Full SD3/Flux training procedures (dataset, hyperparameters, compute requirements)
- Every Flux variant (dev/schnell/pro/fill)--mentioned for vocabulary only
- Implementing MMDiT from scratch (too much architecture code for one lesson)
- Video extensions, multimodal extensions, or ControlNet/IP-Adapter for SD3/Flux
- T5's internal architecture or training (used as a black box; the student knows transformers)
- Licensing, business considerations, open-source vs closed-source dynamics
- Detailed SD3 vs Flux architectural differences (both treated as the same MMDiT family)

**Review notes:**
- Iteration 1: NEEDS REVISION--0 critical, 3 improvement (inconsistent image token counts 256 vs 1024 between sections without bridging, SD3 vs Flux comparison thin and asymmetric, notebook Exercise 2 uses synthetic attention weights without disclosure), 3 polish (plan called for three-column comparison but two-column built, "learned lens" connection not used, Check #3 Q1 adaLN-Zero answer vague)
- Iteration 2: PASS--all three improvement findings fixed (bridging sentence added for token count change, Flux comparison made concrete with architectural details, Exercise 2 reframed as "Joint Attention Structure" with honest synthetic data disclosure). 2 polish remaining (aside repeats body text in "One Room" section, adaLN-Zero answer still hedges on concrete count). Lesson accepted as the series capstone.

### z-image (Lesson 4)
**Status:** Complete
**Cognitive load:** BUILD
**Notebook:** `notebooks/7-4-4-z-image.ipynb`
**Review:** PASS on iteration 2/3 (iteration 1 had 1 CRITICAL + 3 IMPROVEMENT + 2 POLISH, all fixed; iteration 2 found only 2 POLISH)

**Concepts taught:**
- S3-DiT single-stream architecture (INTRODUCED)--shared Q/K/V projections and shared SwiGLU FFN across all token types after lightweight refiner layers, eliminating ~50% parameter overhead of MMDiT's dual-stream design
- Refiner layers for modality pre-processing (INTRODUCED)--2 lightweight transformer layers per modality mapping raw embeddings into a shared representation space
- Qwen3-4B as text encoder (INTRODUCED)--single LLM replacing triple encoder setup, text encoder evolution trajectory traced
- Prompt Enhancer (INTRODUCED)--VLM that expands prompts during SFT, zero inference cost
- 3D Unified RoPE (INTRODUCED)--temporal axis for text, spatial axes for image, orthogonal cross-modal attention
- adaLN with shared low-rank down-projection (INTRODUCED)--LoRA pattern applied to timestep conditioning
- DMD / Distribution Matching Distillation (INTRODUCED)--gap fill, third distillation paradigm with spear/shield decomposition
- Decoupled-DMD (INTRODUCED)--separate noise schedules for spear (high noise) and shield (low noise)
- DMDR (INTRODUCED)--DMD + DPO + GRPO breaking the teacher ceiling
- Z-Image performance positioning (MENTIONED)--6.15B vs Flux 32B, competitive with 1/5 parameters
- Three-phase training curriculum (MENTIONED)--progressive resolution training

**Mental models established:**
- "Translate once, then speak together"--S3-DiT concentrates modality-specific processing in refiner layers, then uses fully shared projections and FFN. Contrasts with MMDiT's "shared listening, separate thinking" (modality-specific at every layer).
- "Spear and shield"--Decoupled-DMD separates distillation into quality-driving CFG augmentation (spear) and distribution-preserving regularization (shield). Separate noise schedules for each.
- "The teacher becomes the guardrail, not the ceiling"--DMDR uses DMD as regularizer and RL as quality driver. The teacher's role shifts from quality ceiling to guardrail preventing reward hacking.
- "Simplicity beats complexity"--Z-Image's competitive performance comes from being simpler than MMDiT, not more complex. Architecture simplification + training innovation + post-training beats architecture complexity.
- "Convergence to simplicity" (text encoders)--from 1 CLIP to 2 CLIPs to 3 encoders back to 1 LLM. The trajectory is "one powerful enough encoder replaces all."

**Analogies used:**
- "Translate once, then speak together" vs "shared listening, separate thinking" (S3-DiT vs MMDiT modality handling)
- "Spear and shield" (Z-Image's own metaphor for DMD's two components)
- "The teacher becomes the guardrail, not the ceiling" (DMDR's reframing of teacher-student relationship)
- "Convergence to simplicity" (text encoder trajectory from multiple specialized to one powerful)
- LoRA pattern applied to timestep conditioning (shared low-rank down-projection + per-layer up-projections)
- Extended the "French/Japanese speaker" analogy from sd3-and-flux (refiner layers as upfront translation)

**How concepts were taught:**
- S3-DiT: parameter counting (d_model=3840, 30 layers, ~50% savings from shared projections/FFN). ComparisonRow: MMDiT dual-stream (5 properties) vs S3-DiT single-stream (5 properties). Negative example: single-stream without refiner layers fails for the same reason naive concatenation fails in MMDiT. The refiner layers solve the same problem differently--"translate once" vs "translate at every exchange."
- Qwen3-4B: text encoder evolution CodeBlock tracing the progression (SD v1.5 -> SDXL -> SD3 -> Z-Image). Practical benefits listed (compositional reasoning, bilingual, instruction-following). Misconception addressed: LLM provides embeddings, not chat responses.
- Prompt Enhancer: concrete before/after example ("West Lake at sunset" -> expanded description). Key insight: integrated during SFT, zero inference cost.
- 3D Unified RoPE: concrete dimension allocations (d_t=32, d_h=48, d_w=48). Worked examples for text token at position 5 and image patch at row 3, col 7. Orthogonal axes mean cross-modal attention has no positional bias--content determines relationships.
- DMD: gap fill from NOT TAUGHT. Framed as third entry in distillation taxonomy. Two components: CFG augmentation (spear) + distribution matching (shield).
- Decoupled-DMD: pseudocode CodeBlock showing coupled vs decoupled training (separate noise schedules). Conceptual advance: understand WHY, then optimize independently.
- DMDR: two-stage RL (DPO for text rendering/counting, GRPO for aesthetics). DMD prevents reward hacking (connected to Series 5). GradientCards for DPO and GRPO stages.
- Source code references: five Z-Image repo files mapped to course concepts. Positioned as capstone reading exercise.
- Check-your-understanding: 8 predict-and-verify questions across 4 checks, testing comprehension of refiner layers, parameter savings, 3D RoPE cross-modal behavior, text encoder simplification, spear-without-shield failure, decoupling benefits, DPO vs GRPO selection, DMD as reward hacking prevention.

**Misconceptions addressed:**
1. "Single-stream means the model treats text and image identically with no modality awareness"--No. Refiner layers provide modality-specific pre-processing. The awareness is concentrated upfront (2 layers) instead of duplicated across 30 layers. WarningBlock.
2. "Distillation cannot produce models better than the teacher"--DMDR combines distillation with RL. DMD is the regularizer, RL is the quality driver. The teacher shifts from ceiling to guardrail. WarningBlock.
3. "Using an LLM as text encoder means chatbot-like understanding"--Qwen3-4B provides embeddings, not chat responses. Richer embeddings from diverse training, not inference-time reasoning. WarningBlock.
4. "Z-Image's efficiency comes from a fundamentally different architecture"--Architecture is simpler (standard transformer with SwiGLU, RMSNorm, RoPE). Efficiency from single-stream parameter savings + better training + post-training. InsightBlock.
5. "Decoupled-DMD is a minor tweak"--Conceptual advance: understand which component drives quality vs stability, then optimize independently. Enables DMDR. WarningBlock.

**What is NOT covered:**
- Implementing S3-DiT from scratch (architectural understanding, not coding)
- Full training procedure details (dataset construction, hyperparameters, World Knowledge Topological Graph)
- The Prompt Enhancer VLM's architecture or training in detail (scoped to brief overview with concrete example)
- Every ablation from the paper (only key design choices)
- Z-Image's data curation pipeline
- Comparing to every other architecture (SD3/Flux as primary comparison)
- Deploying Z-Image in production
- Visual semantic tokens from Z-Image's editing variants (explicitly scoped out as editing-specific)

**Review notes:**
- Iteration 1: MAJOR REVISION--1 critical (notebook missing), 3 improvement (visual semantic tokens introduced without adequate explanation, Decoupled-DMD loss pseudocode missing, Prompt Enhancer explanation shallow/disconnected), 2 polish (em dashes with spaces in CodeBlock, "Flux.2 Dev" reference possibly incorrect). All six findings fixed.
- Iteration 2: PASS--0 critical, 0 improvement, 2 polish (aside repeats body text in "Simplicity, Not Novelty" InsightBlock, notebook Exercise 3 TODO markers could be more helpful). Both genuinely minor. Lesson accepted as the series and module capstone.
