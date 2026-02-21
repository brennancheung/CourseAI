# Lesson: SD3 & Flux

**Slug:** `sd3-and-flux`
**Series:** 7 (Post-SD Advances), **Module:** 7.4 (Next-Generation Architectures), **Position:** Lesson 11 of 11 in series, Lesson 3 of 3 in module
**Cognitive Load:** BUILD (2-3 new concepts that extend the DiT architecture the student just learned: MMDiT joint attention as a simplification of cross-attention, T5-XXL as a stronger text encoder, rectified flow applied in practice)

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| DiT block as standard transformer block (MHA + FFN + residual + norm, no convolutions, no encoder-decoder hierarchy) | DEVELOPED | diffusion-transformers (7.4.2) | The student checked off every component against Series 4 knowledge. ComparisonRow: U-Net (7 properties) vs DiT (7 properties). "You already know every component except the conditioning mechanism." |
| Patchify operation (split latent [C,H,W] into non-overlapping p x p patches, flatten, linear project to d_model; produces [L, d_model] sequence) | DEVELOPED | diffusion-transformers (7.4.2) | Full tensor shape trace: [4,32,32] -> 256 patches of dim 16 -> [256, 1152]. Structural correspondence to text tokenization via ComparisonRow. "Tokenize the image" analogy established. |
| adaLN-Zero conditioning (adaptive layer norm with scale gamma, shift beta, and gate alpha per sub-layer, 6 parameters per block, zero-initialized gate making each block start as identity) | DEVELOPED | diffusion-transformers (7.4.2) | Full formula traced. Connected to adaptive group norm from 6.3.2 and zero convolution from 7.1.1. "The 'Zero' in adaLN-Zero is the critical design choice." |
| DiT scaling recipe (increase d_model and N; same two knobs as GPT-2 -> GPT-3; predictable loss improvement with scale) | INTRODUCED | diffusion-transformers (7.4.2) | DiT model family traced from DiT-S (33M) to DiT-XL (675M). "Two knobs, not twenty." LEGO vs house renovation analogy. |
| DiT replaces only the denoising network (VAE, text encoder, noise schedule, sampling unchanged) | INTRODUCED | diffusion-transformers (7.4.2) | Explicit 11-step pipeline trace annotating which steps are DiT-specific (4-8) and which are unchanged (9-11). "Same pipeline, different denoising network." |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings, per-spatial-location conditioning) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Reactivated multiple times across Series 7 (ControlNet, IP-Adapter). The student can trace Q/K/V through cross-attention, knows shapes, understands per-spatial-location text conditioning. This is the mechanism MMDiT replaces. |
| Self-attention (Q, K, V all from same input sequence; every token attends to every other token; O(n^2) in sequence length) | DEVELOPED | multi-head-attention (4.2.4) | Built from scratch. Traced full worked example with 4 tokens. Understands head specialization, dimension splitting, output projection. This is what MMDiT uses for both text and image tokens. |
| IP-Adapter decoupled cross-attention (separate K/V projections for text and image; shared Q; weighted addition of two attention outputs) | DEVELOPED | ip-adapter (7.1.3) | "Two reference documents, one reader." The student deeply understands this as one approach to multi-source attention. Important contrast: IP-Adapter keeps text and image K/V separate; MMDiT merges them. |
| SDXL dual text encoders (CLIP ViT-L [77,768] + OpenCLIP ViT-bigG [77,1280] concatenated to [77,2048]; single cross-attention path with wider K/V) | DEVELOPED | sdxl (7.4.1) | Tensor shapes traced. Distinguished from IP-Adapter's decoupled approach via ComparisonRow. Pooled text embedding for global conditioning via adaptive norm. |
| Conditional flow matching (straight-line interpolation x_t = (1-t)*x_0 + t*epsilon; velocity prediction v = epsilon - x_0; straight trajectories by construction) | DEVELOPED | flow-matching (7.2.2) | The student trained a 2D flow matching model. Understands curved vs straight trajectories, velocity parameterization, conversion formulas. "Design the trajectory, then derive the training objective." |
| Rectified flow (iterative trajectory straightening; generate aligned pairs from trained model, retrain; each round produces straighter aggregate trajectories) | INTRODUCED | flow-matching (7.2.2) | Taught at intuition level via PhaseCards. "Retracing a hand-drawn line with a ruler." 1-2 rounds sufficient in practice. |
| Connection between flow matching and SD3/Flux (SD3 and Flux use flow matching as training objective; independent of architecture change) | INTRODUCED | flow-matching (7.2.2) | "Two Independent Changes" GradientCard separating training objective from architecture. Practical benefit: 20-30 steps vs 50+. |
| T5 as encoder-decoder transformer | MENTIONED | decoder-only-transformers (4.2.6) | Named in three-variant comparison (encoder-only BERT, encoder-decoder T5, decoder-only GPT). One GradientCard. The student knows T5 exists and is an encoder-decoder transformer but has no understanding of its capabilities, size, or why it would be used as a text encoder for diffusion. |
| Four conditioning channels (WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE/AT-WHAT-QUALITY) | DEVELOPED | ip-adapter (7.1.3), sdxl (7.4.1) | The student has a complete framework for understanding how different conditioning signals enter the denoising network. MMDiT replaces the cross-attention mechanism for WHAT. |
| Flow matching as same family as diffusion (not a new paradigm; different training objective and trajectory shape, same generative model type) | INTRODUCED | flow-matching (7.2.2) | "Same family, different member." Three-column comparison: diffusion SDE (stochastic, curved), probability flow ODE (deterministic, curved), flow matching ODE (deterministic, straight). |
| Conversion between noise, score, and velocity parameterizations | DEVELOPED | flow-matching (7.2.2) | Explicit formulas. Three are interconvertible. Same architecture works for all three. |

### Mental Models and Analogies Already Established

- **"Tokenize the image"** -- patchify is to images what tokenization is to text. The transformer processes both identically. Established in 7.4.2.
- **"Two knobs, not twenty"** -- scaling DiT means increasing d_model and N. Same recipe as GPT-2 -> GPT-3. Established in 7.4.2.
- **"Same pipeline, different denoising network"** -- DiT replaces only the U-Net. Everything else (VAE, text encoder, sampling) is unchanged.
- **"Of Course" chain** -- design insights that feel inevitable given the right framework. Used in ControlNet, IP-Adapter, flow matching, DiT.
- **"Curved vs straight"** -- flow matching's core geometric insight. Straight trajectories need fewer ODE solver steps.
- **"GPS recalculating vs straight highway"** -- curved paths need constant recalculation, straight paths do not.
- **"Same landscape, different lens"** -- diffusion SDE, probability flow ODE, flow matching ODE as three lenses on the same generative process.
- **"The U-Net's last stand"** -- SDXL as the final U-Net refinement, setting up DiT.
- **"Two reference documents, one reader"** -- IP-Adapter's decoupled cross-attention. Two K/V sources, shared Q, weighted addition.
- **"Attention reads, FFN writes"** -- the transformer block's dual-function structure.
- **Zero initialization pattern** -- ControlNet zero conv, LoRA B=0, adaLN-Zero alpha=0. Start contributing nothing, learn what to add.
- **"Same safety pattern"** -- recurring zero initialization theme across adapters and architectures.

### What Was Explicitly NOT Covered

- **MMDiT / joint text-image attention:** Explicitly deferred from the DiT lesson. The DiT lesson covered class-conditional generation only. Joint attention where text tokens and image tokens are concatenated into one sequence and attend to each other was previewed in three GradientCards at the end of the DiT lesson but not taught.
- **T5-XXL as a text encoder for diffusion:** T5 was only MENTIONED in the three-variant comparison in 4.2.6. The student knows T5 exists and is an encoder-decoder transformer but has no understanding of why T5 would be better than CLIP for text conditioning, what T5-XXL's capabilities are, or how its embeddings differ from CLIP's.
- **Rectified flow in practice / logit-normal timestep sampling:** Rectified flow was INTRODUCED at intuition level in 7.2.2. Logit-normal timestep sampling (biasing training toward intermediate noise levels) has not been taught at all.
- **SD3 or Flux architecture specifics:** No lesson has covered the actual SD3 or Flux pipeline. The DiT lesson's preview GradientCards mentioned MMDiT, flow matching, and scaling but did not explain any of them in the SD3/Flux context.
- **How text conditioning works in a DiT-based model:** The original DiT uses class labels, not text. How to add text conditioning to a transformer-based denoising network was explicitly identified as the open question for this lesson.
- **Separate vs shared parameters for text and image streams in MMDiT:** The distinction between SD3's fully separate QKV projections per modality vs simpler approaches has not been discussed.

### Readiness Assessment

The student is exceptionally well-prepared. They have:

1. **Deep DiT knowledge (DEVELOPED in 7.4.2):** patchify, adaLN-Zero, transformer blocks as denoising backbone, scaling properties. The previous lesson was a STRETCH that successfully landed. The student understands every component of the DiT architecture.

2. **Deep self-attention AND cross-attention knowledge (DEVELOPED across Series 4 and 6):** The student built self-attention from scratch, understands Q/K/V projections at formula level, and has seen cross-attention used for text conditioning in the U-Net. MMDiT's joint attention is a simplification: concatenate text and image tokens, run standard self-attention. The student has all the pieces.

3. **Flow matching at DEVELOPED depth (7.2.2):** The student trained a flow matching model. They understand straight-line interpolation, velocity prediction, and why flow matching enables fewer inference steps. Rectified flow at INTRODUCED depth provides additional context for SD3's training objective.

4. **CLIP text encoder at DEVELOPED depth and dual encoders at DEVELOPED depth (6.3.3, 7.4.1):** The student understands CLIP's strengths and limitations (77 tokens, contrastive training, poor at counting/spatial reasoning). They traced SDXL's dual encoder concatenation with tensor shapes. T5-XXL is a natural "what if you used a better text encoder?" extension.

5. **The DiT lesson's explicit preview of SD3/Flux:** Three GradientCards at the end of the DiT lesson previewed exactly what this lesson teaches: (1) joint text-image attention (MMDiT), (2) architecture-agnostic training objectives (flow matching with DiT), (3) clear scaling path. The student is primed for these topics.

6. **Series conclusion context:** This is lesson 11 of 11 in the final series. The student has built from "what is learning?" through neural networks, CNNs, transformers, LLMs, diffusion, and now frontier architectures. The convergence framing should feel earned.

The main challenge is that T5-XXL is a gap: the student knows T5 exists (MENTIONED in 4.2.6) but has no understanding of its capabilities or why it matters for text conditioning. This is a medium gap -- the student deeply understands CLIP's limitations (from 6.3.3) and transformer architectures (from Series 4), so explaining "T5-XXL is a much larger language model that produces richer text understanding" is straightforward. A dedicated section within this lesson resolves the gap.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how SD3 and Flux combine the DiT architecture with joint text-image attention (MMDiT), stronger text encoding (T5-XXL alongside CLIP), and flow matching to create the current frontier diffusion architecture, understanding why these three changes converge and how each addresses a specific limitation of prior approaches.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| DiT block (MHA + FFN + residual + adaLN-Zero, standard transformer on patch tokens) | DEVELOPED | DEVELOPED | diffusion-transformers (7.4.2) | OK | MMDiT extends the DiT block. The student must understand the base architecture to see what MMDiT changes (joint attention replaces class-conditional adaLN-Zero for text information). |
| Patchify operation (latent -> patch tokens -> [L, d_model] sequence) | DEVELOPED | DEVELOPED | diffusion-transformers (7.4.2) | OK | SD3/Flux still patchify the latent. The student needs this to understand that the image tokens in MMDiT's joint sequence come from the same patchify operation. |
| adaLN-Zero conditioning (gamma, beta, alpha from conditioning vector) | DEVELOPED | DEVELOPED | diffusion-transformers (7.4.2) | OK | MMDiT still uses adaLN-Zero for timestep conditioning. The change is that text moves from adaLN-Zero (class label in DiT) to joint attention (text tokens in MMDiT). The student must understand adaLN-Zero to see what role it retains vs what joint attention takes over. |
| Self-attention (Q, K, V from same sequence; every token attends to every other) | DEVELOPED | DEVELOPED | multi-head-attention (4.2.4) | OK | MMDiT's core mechanism is standard self-attention on a concatenated text+image sequence. The student must be able to trace Q/K/V through the joint sequence. |
| Cross-attention (Q from spatial features, K/V from text; per-location text conditioning) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | MMDiT replaces cross-attention with joint self-attention. The student needs to understand what is being replaced and why joint attention is simpler (one attention operation instead of two). |
| CLIP text encoder (77 tokens, [77, 768], contrastive training, limitations) | DEVELOPED | DEVELOPED | clip (6.3.3) | OK | T5-XXL is added ALONGSIDE CLIP, not replacing it. The student must understand CLIP's limitations (which motivate T5) and its strengths (which are retained). |
| SDXL dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG, concatenated to [77, 2048]) | DEVELOPED | sdxl (7.4.1) | DEVELOPED | OK | SD3 extends the dual-encoder approach to a triple-encoder approach. The student's understanding of concatenation and why multiple encoders help is the foundation. |
| Conditional flow matching (straight-line interpolation, velocity prediction, fewer steps) | DEVELOPED | DEVELOPED | flow-matching (7.2.2) | OK | SD3/Flux use flow matching as the training objective. The student must understand what flow matching does (straight trajectories, velocity prediction) to see how it fits into the SD3/Flux training pipeline. |
| Rectified flow (iterative trajectory straightening for even fewer steps) | INTRODUCED | INTRODUCED | flow-matching (7.2.2) | OK | SD3 uses rectified flow. INTRODUCED depth is sufficient -- the student understands the concept (generate aligned pairs, retrain for straighter aggregate trajectories) and this lesson applies it rather than deepening it. |
| T5 as encoder-decoder transformer | MENTIONED | MENTIONED | decoder-only-transformers (4.2.6) | GAP | T5-XXL is a key component of SD3/Flux. The student knows T5 exists but not what it does, how large it is, or why its text embeddings are better than CLIP's for diffusion. Needs elevation. |
| DiT replaces only the denoising network (VAE, text encoder, sampling unchanged) | INTRODUCED | INTRODUCED | diffusion-transformers (7.4.2) | OK | SD3/Flux preserve this modularity. The student needs to know that the pipeline structure is preserved even though the denoising network changes further. |
| IP-Adapter decoupled cross-attention (separate K/V paths, weighted addition) | INTRODUCED | DEVELOPED | ip-adapter (7.1.3) | OK | Important contrast: IP-Adapter keeps text and image attention separate; MMDiT merges them into one attention. Different design philosophies for multi-source attention. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| T5-XXL at MENTIONED, need INTRODUCED | Small (the student deeply understands CLIP's architecture and limitations from 6.3.3, knows transformer architectures from Series 4, and has seen T5 named as an encoder-decoder transformer in 4.2.6. The gap is specifically: why T5's text embeddings are richer than CLIP's, T5-XXL's scale, and how its embeddings enter the model.) | Brief dedicated section (2-3 paragraphs + comparison). Frame as problem-before-solution: "CLIP was trained on image-text pairs via contrastive learning. It understands text through the lens of images. T5 was trained purely on text -- it understands language itself. A 4.7B-parameter language model produces much richer text understanding than a 123M CLIP text encoder." No need to teach T5's architecture -- the student already knows transformers. Focus on what T5 provides (richer embeddings, longer context, better compositional understanding) and how it complements CLIP (T5 for understanding, CLIP for image-text alignment). |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "MMDiT's joint attention is fundamentally different from and more complex than cross-attention" | The student has deeply learned cross-attention as the mechanism for text conditioning. A new mechanism called "joint attention" with a special name (MMDiT) sounds like a more complex system. The name "Multimodal Diffusion Transformer" suggests something exotic. | Joint attention is simpler than cross-attention. Cross-attention requires two separate operations: self-attention on image tokens AND cross-attention where image Q attends to text K/V. Joint attention replaces both with one operation: concatenate text and image tokens, run standard self-attention on the combined sequence. Every token attends to every other token. The student already knows self-attention deeply. MMDiT is not adding a mechanism -- it is removing the asymmetry between text and image. One attention operation instead of two. | When introducing MMDiT. Explicit ComparisonRow: cross-attention approach (two attention operations per block: self-attention on image + cross-attention to text) vs joint attention (one attention operation: self-attention on concatenated text+image). "Simpler, not more complex." |
| "Flow matching in SD3/Flux is new content that this lesson needs to teach" | The student might think that because SD3 uses "rectified flow," it is a different concept from the flow matching they learned in 7.2.2. The name "rectified flow scheduling" sounds like a new technique. | SD3's training objective is the conditional flow matching the student already trained a model with in 7.2.2. Rectified flow is the iterative straightening procedure the student learned about in 7.2.2. Nothing about the training objective is new -- this lesson APPLIES concepts the student already has. The only new detail is logit-normal timestep sampling (a scheduling optimization, not a new paradigm). | Early in the flow matching section. Explicit statement: "You already know everything about flow matching. SD3 uses the same straight-line interpolation and velocity prediction you trained a model with in 7.2.2. Rectified flow is the same trajectory straightening you saw in 7.2.2. The only new detail is how training timesteps are sampled." |
| "T5-XXL replaces CLIP in SD3/Flux" | SD3 uses T5-XXL, which is much bigger and better at text. The student might assume the bigger model simply replaces the smaller one, like upgrading to a better version. | SD3 uses THREE text encoders simultaneously: CLIP ViT-L (from SD v1.5), OpenCLIP ViT-bigG (from SDXL), AND T5-XXL. They are not redundant -- each provides different information. CLIP encoders provide image-aligned text understanding (trained contrastively on image-text pairs). T5 provides deep linguistic understanding (trained purely on text). The pooled CLIP embeddings also provide global conditioning via adaLN-Zero. Removing any one degrades quality. | When introducing T5-XXL and the triple encoder setup. Three-column comparison: what each encoder provides and why all three are needed. |
| "Joint attention means the model treats text tokens and image tokens identically" | In standard self-attention, all tokens are processed identically. The student might assume concatenating text and image tokens means they share everything -- same embeddings, same projections, same normalization. | In SD3's MMDiT, text tokens and image tokens have SEPARATE linear projections for Q, K, and V. Each modality has its own W_Q, W_K, W_V matrices. The projections are modality-specific; only the attention computation itself is shared. After attention, the tokens are split back apart and processed by separate FFN layers. This is different from naive concatenation where everything is shared. The separate projections allow each modality to "speak its own language" while the shared attention lets them "hear each other." | After explaining the basic joint attention idea, when going deeper into the MMDiT block structure. This is the key architectural detail that separates MMDiT from naive concatenation. |
| "SD3/Flux is a completely new system unrelated to what I've learned" | The cumulative weight of changes (new attention mechanism, new text encoder, new training objective) might make the student feel this is an entirely new paradigm. | SD3/Flux is the convergence of concepts the student already knows: (1) transformer blocks from Series 4, (2) latent diffusion pipeline from Series 6, (3) flow matching from 7.2.2, (4) patchify from 7.4.2, (5) adaLN-Zero from 7.4.2. The VAE is the same. The sampling loop is the same. The text encoders are the same CLIP models plus T5. The architecture is DiT plus joint attention. Every piece has been taught -- this lesson shows how they combine. | As the lesson's overarching framing. The convergence theme should run through the entire lesson. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Joint attention traced on a concrete sequence: 77 text tokens + 256 image patch tokens = 333 total tokens, each with d_model dimensions. Q/K/V computed per-modality (separate projections), attention matrix is [333, 333] (every token attends to every other), then tokens are split back into text [77, d_model] and image [256, d_model] for modality-specific FFN processing | Positive | Make joint attention concrete with specific numbers. The student can compare the attention matrix shape to cross-attention (where the attention was [256, 77], image attending to text). In joint attention, both directions exist: image attends to text AND text attends to image, plus text attends to text and image attends to image. The symmetry is the key insight. | Continues the tensor shape tracing format from 6.4.1, SDXL, and DiT. The student is comfortable with this format. The concrete sequence length (333 = 77 + 256) makes the combined sequence tangible. The [333, 333] attention matrix size lets the student compute the cost and compare to the DiT-only [256, 256] self-attention plus [256, 77] cross-attention. |
| SD3's triple text encoder setup with tensor shapes: CLIP ViT-L [77, 768], OpenCLIP ViT-bigG [77, 1280], T5-XXL [77, 4096]. How they combine: CLIP outputs pooled for global conditioning via adaLN-Zero, CLIP + OpenCLIP per-token embeddings projected for one purpose, T5 per-token embeddings projected separately, both concatenated or combined for the text token sequence entering joint attention | Positive | Make the triple encoder concrete. The student traced dual encoders in SDXL ([77, 2048]). SD3 extends this with a third, much larger encoder. Showing the tensor shapes makes the scale visible: T5-XXL's 4096-dim embeddings carry ~5x more information per token than CLIP's 768. The student sees why T5 matters through the numbers, not just the argument. | Builds directly on the SDXL tensor shape trace pattern. The student can see the progression: SD v1.5 (one encoder, 768 dims) -> SDXL (two encoders, 2048 dims concatenated) -> SD3 (three encoders, with T5's 4096 dims dwarfing CLIP's). The scale of the T5 embedding makes the "richer text understanding" argument tangible. |
| Negative example: naive concatenation where text and image tokens share all projections (same W_Q, W_K, W_V, same FFN) vs MMDiT's modality-specific projections. Why naive concatenation fails: text tokens and image tokens have fundamentally different representations (text is discrete language, images are continuous spatial features). Forcing them through the same projections is like asking a French speaker and a Japanese speaker to use the same dictionary -- shared attention lets them hear each other, but each needs to formulate thoughts in their own language. | Negative | Show that MMDiT is not "just concatenate and self-attend." The separate projections per modality are an important design choice. Without them, the model would struggle because text embeddings and image patch embeddings live in different representation spaces. The negative example prevents the student from oversimplifying MMDiT as trivial concatenation. | Addresses Misconception #4 and is the most important architectural nuance. The student's deep understanding of Q/K/V projections (from 4.2.4) means they can appreciate why modality-specific projections matter: the projection matrices learn to map each modality's representation into a shared attention space where dot-product similarity is meaningful. |
| Positive (stretch): the full SD3 pipeline traced end-to-end, annotating which components come from which lesson/series. VAE encode (Series 6) -> patchify (7.4.2) -> triple text encoding (this lesson + 6.3.3 + 7.4.1) -> MMDiT blocks with adaLN-Zero timestep conditioning and joint attention (this lesson + 7.4.2) -> unpatchify (7.4.2) -> flow matching sampling (7.2.2) -> VAE decode (Series 6). Every step annotated with its source. | Positive (stretch) | Show the convergence explicitly. The student should see that every component of the frontier architecture traces back to a lesson they completed. Nothing is unexplained. This is the emotional payoff of the entire series: "you understand the current frontier." | This is the culminating example of the entire course's generative model track (Series 6 + 7). The pipeline trace format is the student's most familiar representational format. Annotating each step with its lesson source makes the convergence tangible and personal. |

---

## Phase 3: Design

### Narrative Arc

The student finished the DiT lesson understanding that transformers can replace U-Nets as diffusion denoising networks, with better scaling properties and simpler architecture. But the DiT they learned was class-conditional on ImageNet -- it generates golden retrievers and volcanos by conditioning on class labels, not text prompts. The three GradientCards at the end of the DiT lesson previewed what comes next: joint text-image attention, architecture-agnostic training objectives, and a clear scaling path. This lesson delivers on that preview.

SD3 and Flux represent the convergence of everything the student has learned across Series 4, 6, and 7. The DiT architecture (from the previous lesson) provides the transformer backbone. Flow matching (from 7.2.2) provides the training objective that produces straight trajectories and enables fewer inference steps. T5-XXL (new in this lesson) provides text understanding that goes far beyond CLIP's image-aligned embeddings. And MMDiT -- the Multimodal Diffusion Transformer -- ties it all together with a surprisingly simple idea: instead of using cross-attention to inject text into the image processing, concatenate text tokens and image tokens into one sequence and let them attend to each other through standard self-attention. Text reads image, image reads text, text reads text, image reads image -- all in one attention operation.

The lesson should feel like arriving at a destination the student has been traveling toward across 50+ lessons. Not a surprise, but a convergence. Every piece was established along the way: transformers, latent diffusion, cross-attention, flow matching, patchify, adaLN-Zero. SD3/Flux is not new -- it is the combination of everything the student already knows. The final lesson of the final series should leave the student feeling that they can read frontier diffusion papers and understand the design choices, because they have built every foundation those papers stand on.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example** | Tensor shape trace through the MMDiT joint attention: 77 text tokens + 256 image tokens = 333-token combined sequence. Modality-specific Q/K/V projections: text W_Q produces [77, d_head], image W_Q produces [256, d_head], concatenated Q is [333, d_head]. Attention matrix [333, 333]. Output split back to [77, d_model] and [256, d_model]. Then modality-specific FFNs. The student can compare this to cross-attention shapes (image Q [256, d_head] attending to text K/V [77, d_head], attention matrix [256, 77]) and see what joint attention adds: text-to-text, text-to-image, and image-to-image attention alongside the existing image-to-text. | The student learned every prior architecture through tensor shape tracing. This format is their most reliable way to build understanding. The concrete numbers make the abstract idea of "joint attention" tangible: a [333, 333] matrix is a real thing with real compute cost and real information flow. Comparing to cross-attention's [256, 77] shows exactly what changed and what new information flows exist. |
| **Visual** | ComparisonRow contrasting three text conditioning approaches: (1) U-Net cross-attention (SD v1.5/SDXL): image features as Q, text as K/V, one-directional text->image influence. (2) DiT class-conditional (original DiT): class label via adaLN-Zero, no text tokens in attention. (3) MMDiT joint attention (SD3/Flux): text and image tokens concatenated, bidirectional attention, both modalities influence each other. Three columns showing the information flow direction in each. | The student has seen cross-attention and class-conditional approaches. A three-column comparison makes the evolution visible: from one-directional text influence (cross-attention) to no text in attention (class-conditional DiT) to bidirectional text-image interaction (MMDiT). The student can trace the progression and see that MMDiT is the synthesis: it provides the text conditioning of cross-attention AND the full self-attention of DiT, in one operation. |
| **Verbal/Analogy** | "One room, one conversation" -- In cross-attention (U-Net), text and image are in separate rooms connected by a one-way mirror: the image can see the text, but the text cannot see or respond to the image. In joint attention (MMDiT), everyone is in the same room having one conversation. Text tokens hear what image tokens say and respond. Image tokens hear what text tokens say and respond. The interaction is symmetric and bidirectional. This is simpler (one room, not two) and richer (two-way interaction, not one-way). | Makes joint attention feel natural and simpler rather than exotic. The one-way mirror captures the key limitation of cross-attention that MMDiT resolves: text conditioning in the U-Net is one-directional (image reads text, but text embeddings are frozen and never updated by what the image features contain). In MMDiT, the text representations ARE updated by the image context, allowing the model to disambiguate prompts based on what it has generated so far. |
| **Intuitive** | The convergence intuition: SD3/Flux is not a new idea but the combination of ideas the student already has. Walk through the "convergence map" -- each component traces to a specific lesson: transformer blocks (4.2.5), patchify (7.4.2), adaLN-Zero (7.4.2), flow matching (7.2.2), CLIP (6.3.3), self-attention (4.2.4). The student should feel "I already knew all of this" -- the same feeling they had when building their first diffusion model in 6.2.5, where every piece traced to a prior lesson. | The convergence feeling IS the pedagogical goal of this lesson. By the end, the student should not feel they learned something new so much as they recognized something inevitable. This mirrors the "Of Course" chain pattern used throughout Series 7: given what you know, of course this is where the field went. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 new concepts that are extensions or simplifications of familiar patterns:
  1. **MMDiT joint attention (concatenate text and image tokens, run standard self-attention with modality-specific projections)** -- This is the primary new concept. However, it is arguably a simplification: it replaces two attention operations (self-attention + cross-attention) with one (self-attention on the combined sequence). The student deeply understands both self-attention and cross-attention. The genuinely new element is the modality-specific projections (separate W_Q, W_K, W_V per modality) and the split-after-attention pattern. The conceptual delta is manageable because the attention mechanism itself is unchanged -- only the input composition changes.
  2. **T5-XXL as text encoder (larger language model providing richer text understanding alongside CLIP)** -- New component but not a new concept. The student understands text encoders deeply (CLIP at DEVELOPED). T5-XXL is "a bigger, better text encoder trained differently." The student needs to understand WHY it is better (trained on text alone, not image-text pairs; much larger; better compositional understanding) and HOW it joins the pipeline (its embeddings form part of the text token sequence in joint attention). The conceptual delta is small.
  3. **Logit-normal timestep sampling (biasing training toward intermediate noise levels)** -- A small optimization detail. The student understands noise schedules and timestep sampling. Logit-normal sampling says "intermediate timesteps are where the model learns the most, so sample more training steps there." This is a minor concept that can be INTRODUCED in 2-3 sentences.

- **Previous lesson load:** diffusion-transformers was STRETCH (2-3 genuinely new concepts)
- **Is this appropriate?** BUILD following STRETCH is the standard recovery pattern. The student had the hardest lesson of the module (DiT) and now applies and extends that knowledge. None of the new concepts require a conceptual leap -- MMDiT is a simplification of something the student knows (cross-attention), T5 is a larger version of something the student knows (text encoders), and logit-normal sampling is a minor detail. The cognitive work is integration: seeing how all the pieces fit together. This naturally consolidates the STRETCH from the previous lesson while adding a manageable amount of new content.

### Connections to Prior Concepts

| New Concept | Connects To | How |
|-------------|------------|-----|
| MMDiT joint attention (concatenate text + image tokens, self-attention with modality-specific projections) | Self-attention from 4.2.4 + cross-attention from 6.3.4 + IP-Adapter decoupled attention from 7.1.3 | "Cross-attention is one-directional: image reads text. Joint attention is bidirectional: image reads text AND text reads image. It is standard self-attention (from 4.2.4) applied to a combined sequence. Compare to IP-Adapter (7.1.3): IP-Adapter keeps text and image K/V separate with weighted addition. MMDiT merges them into one sequence so everything attends to everything. Simpler mechanism, richer interaction." |
| Modality-specific projections in MMDiT (separate W_Q, W_K, W_V per modality) | Q/K/V projections from 4.2.4 + "learned lens" from 4.2.2/6.3.2 | "Each modality has its own projection matrices that map its representations into a shared attention space. This is the 'learned lens' pattern: each modality looks at its input through its own learned projection. The projections are modality-specific; the attention computation is shared." |
| T5-XXL as text encoder (language model embeddings alongside CLIP embeddings) | CLIP text encoder from 6.3.3 + SDXL dual encoders from 7.4.1 | "SDXL added a second CLIP encoder for richer text. SD3 adds a third, fundamentally different encoder: T5-XXL. CLIP understands text through the lens of images (contrastive training). T5 understands text through the lens of language (masked language modeling). Different training means different strengths. Both contribute to the text token sequence." |
| Rectified flow in SD3 (flow matching training objective applied in practice) | Flow matching from 7.2.2 + rectified flow from 7.2.2 | "You trained a flow matching model in 7.2.2. SD3 uses the same training objective: straight-line interpolation, velocity prediction, MSE loss. Rectified flow (trajectory straightening via model-aligned pairs) is applied to make the aggregate trajectories even straighter. Nothing about the training objective is new -- this lesson applies it." |
| Triple text encoder setup (CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL) | SDXL dual encoders from 7.4.1 | "SD v1.5: one CLIP encoder [77, 768]. SDXL: two CLIP encoders [77, 2048]. SD3: two CLIP encoders + T5-XXL [77, 4096]. The progression is consistent: more text encoders providing richer text understanding. Each step adds a new source of text information through the same pipeline pattern." |
| Logit-normal timestep sampling | Noise schedule from 6.2.2 + timestep sampling in training from 6.2.3 | "In DDPM training, timesteps are sampled uniformly. Logit-normal sampling biases toward intermediate timesteps where the denoising task is hardest (not too noisy, not too clean). Same concept as importance sampling: spend more training compute where the model learns the most." |

### Analogies to Extend

- **"Tokenize the image"** from 7.4.2 -- extends naturally. In DiT, you tokenize the image and process with a transformer. In MMDiT, you tokenize the image AND keep the text as tokens, then process BOTH together. "Tokenize both modalities, process together."
- **"Same pipeline, different denoising network"** from 7.4.2 -- still applies. SD3's pipeline is: text encode -> patchify -> MMDiT denoising -> unpatchify -> VAE decode. The VAE and sampling loop are unchanged. The denoising network changed again (from DiT to MMDiT), but the pipeline structure persists.
- **"Of Course" chain** -- can be applied to MMDiT: (1) DiT processes image tokens with self-attention, (2) text conditioning via cross-attention requires a separate attention operation, (3) but both text and image are token sequences, (4) you can concatenate two token sequences into one, (5) standard self-attention on the combined sequence provides both self-attention AND cross-modal attention in one operation, (6) of course the field merged them.
- **"Two knobs, not twenty"** from 7.4.2 -- still applies. MMDiT scales the same way as DiT: increase d_model and N.
- **"Same family, different member"** from 7.2.2 -- flow matching in SD3 is the same family member the student already knows. Applying it to a real model is not learning something new.

### Analogies That Could Be Misleading

- **"Two reference documents, one reader"** from IP-Adapter (7.1.3) -- could mislead because it implies keeping text and image as separate documents the reader consults independently. In MMDiT, the documents are combined into one: text and image tokens are in the same sequence and attend to each other symmetrically. There is no "reader consults two separate sources" -- there is one combined text read by everyone simultaneously. Address by explicit contrast: "IP-Adapter reads from two separate documents and combines the readings. MMDiT puts everything into one document."
- **"Tokenize the image"** from 7.4.2 -- could mislead if the student thinks all tokens are processed identically. In MMDiT, text tokens and image tokens have separate projections and separate FFNs. They are tokenized and attend together, but they maintain distinct processing paths. Address by clarifying: "same attention room, different languages -- each modality has its own way of thinking (FFN) and expressing (Q/K/V projections)."

### Scope Boundaries

**This lesson IS about:**
- MMDiT joint attention: concatenating text and image tokens, running self-attention on the combined sequence, modality-specific Q/K/V projections, split-after-attention, modality-specific FFNs
- T5-XXL as a text encoder: why a language model provides richer text understanding than CLIP, how T5 complements CLIP, the triple encoder setup with tensor shapes
- Flow matching as SD3's training objective: connecting to the student's existing flow matching knowledge from 7.2.2, rectified flow application, logit-normal timestep sampling as a minor optimization
- The full SD3/Flux pipeline traced end-to-end, annotated with lesson sources
- Convergence: how each component traces to a prior lesson, the "you already knew all of this" feeling
- Positioning SD3/Flux as the current frontier and the series conclusion

**This lesson is NOT about:**
- Full SD3/Flux training procedure (training compute requirements, dataset, exact hyperparameters)
- Every Flux variant (dev, schnell, pro, fill, etc.) -- mentioned for vocabulary only
- Video extensions or multimodal extensions of the architecture
- Distilled versions or acceleration beyond what was covered in Module 7.3
- Licensing, business considerations, or open-source vs closed-source dynamics
- Implementing MMDiT from scratch (too much architecture code for a single lesson)
- ControlNet or IP-Adapter for SD3/Flux (same concepts from Module 7.1, different weight dimensions)
- Detailed comparison of SD3 vs Flux architecture differences (both are MMDiT variants; this lesson treats them as the same architectural family)
- T5's internal architecture or training (used as a black box; the student knows transformers)

**Depth targets:**
- MMDiT joint attention: DEVELOPED (student can explain why text and image tokens are concatenated, trace the attention matrix shape, explain modality-specific projections, compare to cross-attention)
- T5-XXL as text encoder: INTRODUCED (student understands why T5 is better than CLIP for text understanding, knows it is a large language model encoder, knows it complements rather than replaces CLIP, but does not know T5's internal details)
- Rectified flow in SD3: INTRODUCED (student understands SD3 uses the flow matching objective they already know, with rectified flow for straighter trajectories; logit-normal sampling is a minor detail)
- SD3/Flux as convergence architecture: INTRODUCED (qualitative understanding of how all pieces fit together; the student can read the architectural description in an SD3/Flux paper and identify each component)

---

### Lesson Outline

#### 1. Context + Constraints

- "This is the final lesson of Series 7 and the final lesson of the course's generative model track. In the previous lesson, you saw how DiT replaces the U-Net with a transformer: patchify the latent, process with standard transformer blocks, use adaLN-Zero for timestep conditioning. But DiT was class-conditional on ImageNet -- it generates golden retrievers and volcanos, not 'a cat sitting on a beach at sunset.' How do you add text conditioning to a transformer-based denoising network?"
- "This lesson answers that question. SD3 and Flux take DiT's transformer backbone, add stronger text encoding, and replace cross-attention with something simpler: joint self-attention on text and image tokens together. Combined with flow matching as the training objective, the result is the current frontier diffusion architecture."
- ConstraintBlock: This lesson covers the MMDiT architecture (joint text-image attention), T5-XXL as a text encoder, and flow matching as SD3's training objective. It does NOT cover full SD3/Flux training procedures, every Flux variant (dev/schnell/pro), video extensions, implementing MMDiT from scratch, or ControlNet/IP-Adapter for SD3/Flux (same concepts from Module 7.1, different dimensions). We are reading the frontier together, tracing each design choice back to what you already know.

#### 2. Recap

Brief reactivation of three concepts (one has a gap fill deferred to a dedicated section):

- **DiT architecture** (from 7.4.2): 2-3 sentences. "You learned DiT last lesson: patchify the latent into tokens, process with standard transformer blocks (self-attention + FFN + residual + norm), condition via adaLN-Zero. Class-conditional on ImageNet. The architecture scales with two knobs: d_model and N."
- **Flow matching** (from 7.2.2): 2-3 sentences. "Straight-line interpolation x_t = (1-t)*x_0 + t*epsilon. Velocity prediction v = epsilon - x_0. Straight trajectories by construction, fewer ODE solver steps needed. You trained a flow matching model from scratch."
- **Cross-attention for text conditioning** (from 6.3.4): 2-3 sentences. "In the U-Net, text enters through cross-attention: Q from spatial features, K/V from text embeddings. Each spatial location attends to all text tokens. One-directional: the image reads the text, but the text embeddings are fixed -- they never respond to what the image contains."
- Transition: "DiT solved the architecture problem. Flow matching solved the training objective problem. But DiT is class-conditional -- no text. And the U-Net's cross-attention is one-directional. SD3 and Flux solve both."

#### 3. Hook

Type: **Convergence map + "trace the sources" challenge**

"Every component of the current frontier architecture traces back to a lesson you have already completed."

A convergence map showing five knowledge threads converging:
- **Thread 1: Transformer blocks** (Series 4) -- self-attention, FFN, residual connections, scaling
- **Thread 2: Latent diffusion** (Series 6) -- VAE, latent space, conditioning, denoising loop
- **Thread 3: Flow matching** (Module 7.2) -- straight trajectories, velocity prediction, fewer steps
- **Thread 4: DiT** (previous lesson) -- patchify, adaLN-Zero, transformer as denoising network
- **Thread 5: Text encoding** (CLIP from 6.3.3, dual encoders from 7.4.1) -- how text conditions the model

"SD3 and Flux combine all five threads. Before I show you how, predict: the original DiT used class labels for conditioning. If you wanted to add text conditioning to a DiT, how would you do it? You have text embeddings from CLIP. You have a transformer processing patch tokens. How do you make the transformer use the text?"

Design challenge in `<details>` reveal: The student should consider options:
1. Cross-attention (the U-Net approach) -- add cross-attention layers where patch tokens attend to text tokens
2. adaLN-Zero (the DiT approach) -- project text into the conditioning vector alongside the timestep
3. Concatenation (the simple approach) -- add text tokens to the patch token sequence and let self-attention handle everything

"Option 3 is what SD3 and Flux do. And it turns out to be both simpler and better."

#### 4. Explain: The Text Encoder Problem

**Part A: CLIP's limitations revisited**

Brief callback to 6.3.3: "CLIP's text encoder was trained on image-text pairs via contrastive learning. It understands text through the lens of 'what image does this describe?' This makes it good at visual concepts (colors, objects, scenes) but weak at compositional reasoning (spatial relationships, counting, negation), long descriptions, and abstract concepts that do not correspond to visual patterns."

"SDXL partially addressed this by adding a second, larger CLIP encoder (OpenCLIP ViT-bigG). But both encoders share the same training paradigm: learn text representations through image-text alignment. What if you used a text encoder trained purely on language understanding?"

**Part B: T5-XXL -- a language model as text encoder**

"T5-XXL is a 4.7 billion parameter encoder-decoder transformer trained on massive text corpora. You saw T5 named in the three-variant comparison in Series 4 as an encoder-decoder transformer (alongside encoder-only BERT and decoder-only GPT)."

"Where CLIP's text encoder has 123M parameters and was trained to match images, T5-XXL has 4.7B parameters and was trained to understand language itself. The result: T5 produces text embeddings that capture compositional structure, complex relationships, and nuanced meaning that CLIP misses."

Key comparison (brief):
- CLIP ViT-L: 123M params, [77, 768] embeddings, trained on image-text pairs
- OpenCLIP ViT-bigG: 354M params (text encoder), [77, 1280] embeddings, trained on image-text pairs
- T5-XXL: 4.7B params, [77, 4096] embeddings, trained on text alone

"T5's embeddings carry ~5x more information per token than CLIP's (4096 vs 768 dimensions). More importantly, they capture different information: linguistic structure rather than visual alignment."

**Part C: Why keep CLIP alongside T5?**

"If T5 is so much better, why not just use T5? Because CLIP and T5 provide complementary information:"

Two-column comparison:
- **CLIP encoders:** Trained on image-text pairs. Embeddings are aligned with visual features. The pooled CLS embedding provides a global summary that works well for adaLN-Zero conditioning. Good at: visual concepts, style, aesthetics.
- **T5-XXL:** Trained on text alone. Embeddings capture linguistic meaning without visual bias. No pooled embedding (encoder output only). Good at: compositional descriptions, spatial relationships, counting, abstract concepts.

"SD3 uses all three: CLIP ViT-L, OpenCLIP ViT-bigG, and T5-XXL. The CLIP pooled embeddings provide global conditioning via adaLN-Zero (same as SDXL). The per-token embeddings from all three are combined to form the text token sequence for joint attention."

Address Misconception #3: "T5 does not replace CLIP. All three encoders contribute simultaneously. Removing any one degrades quality because each provides information the others lack."

#### 5. Check #1

Two predict-and-verify questions:

1. "SD3 uses T5-XXL with 4.7B parameters as one of its text encoders. The denoising network (MMDiT) has ~2B parameters. What does this imply about the total model size?" (Answer: The text encoder alone is larger than the denoising network. Total model size is ~8B+ across all components. This is a significant practical consideration -- you need substantial VRAM just for the text encoders. It also reflects the field's recognition that text understanding is a bottleneck: investing more parameters in understanding the prompt is worth the cost.)

2. "A colleague says: 'T5-XXL is a language model, so it understands text better than CLIP. We should just use T5 and drop CLIP entirely.' What would be lost?" (Answer: CLIP's pooled embedding provides global conditioning via adaLN-Zero -- T5 does not produce this kind of summary vector. More importantly, CLIP's embeddings are aligned with visual features because of contrastive training. T5's embeddings are purely linguistic. The combination of both captures "what does this text mean linguistically" AND "what visual content does this text map to." Dropping either loses information.)

#### 6. Explain: MMDiT -- Joint Text-Image Attention

**Part A: The cross-attention limitation**

"In the U-Net (and in a hypothetical text-conditioned DiT with cross-attention), text conditioning is one-directional:"

- Image tokens compute Q, text tokens provide K and V
- The image reads from the text: "what does the prompt say about this spatial location?"
- But the text embeddings are frozen in the attention computation -- they do not change in response to what the image contains
- Each block has TWO attention operations: self-attention on image tokens (image reads image) AND cross-attention from image to text (image reads text)

"Joint attention asks: what if text and image were in the same sequence?"

**Part B: The simple idea**

"Concatenate the text tokens and image patch tokens into one sequence. Run standard self-attention on the combined sequence."

Concrete example with tensor shapes:

```
Text tokens:  [77, d_model]     (from text encoders, projected to d_model)
Image tokens: [256, d_model]    (from patchify, same as DiT)

Concatenated: [333, d_model]    (77 + 256 = 333 tokens)

Self-attention on [333, d_model]:
  Q: [333, d_head]  (every token produces a query)
  K: [333, d_head]  (every token produces a key)
  V: [333, d_head]  (every token produces a value)
  Attention weights: [333, 333]  (every token attends to every other token)
  Output: [333, d_model]

Split back: text [77, d_model], image [256, d_model]
```

"One attention operation replaces two. And it provides four types of attention simultaneously:"

Four-item list:
- **Image-to-text** (image tokens read text tokens) -- equivalent to cross-attention in U-Net
- **Text-to-image** (text tokens read image tokens) -- NEW: text representations update based on image content
- **Image-to-image** (image tokens read image tokens) -- equivalent to self-attention in DiT
- **Text-to-text** (text tokens read text tokens) -- NEW: text representations refine each other within the block

"Cross-attention provided only the first. Joint attention provides all four."

Address Misconception #1: ComparisonRow -- Cross-attention approach vs Joint attention (MMDiT):

| | Cross-Attention (U-Net/hypothetical DiT) | Joint Attention (MMDiT) |
|---|---|---|
| Attention operations per block | Two: self-attention on image + cross-attention to text | One: self-attention on concatenated text+image |
| Information flow | One-directional: image reads text | Bidirectional: image reads text AND text reads image |
| Text representation | Fixed across blocks (same embeddings at every layer) | Updated by each block (text refines based on image context) |
| Mechanism complexity | Two separate attention computations with different Q sources | One standard self-attention |
| Attention matrix | Self: [256, 256] + Cross: [256, 77] | Joint: [333, 333] |

InsightBlock: "Joint attention is simpler, not more complex. One attention operation instead of two. The 'M' in MMDiT stands for 'Multimodal' -- but the mechanism is just standard self-attention on a multimodal sequence."

**Part C: The "one room" analogy**

"Think of cross-attention as two rooms connected by a one-way mirror. The image room can see the text room, but the text room cannot see or respond to what is happening with the image. Joint attention puts everyone in the same room having one conversation. Text tokens hear what image tokens are saying and can respond. Image tokens hear what text tokens are saying and can respond. The interaction is symmetric."

"This matters because text conditioning in the U-Net is static -- the same text embeddings are used at every denoising step, regardless of what the image looks like at that point. In MMDiT, the text representations evolve through the network's layers, refining their meaning based on the image context. 'A crane near a river' can be disambiguated based on what the image actually contains."

#### 7. Check #2

Three predict-and-verify questions:

1. "In cross-attention, the attention matrix is [256, 77] (256 image tokens attending to 77 text tokens). In joint attention, it is [333, 333]. How does the attention compute cost compare?" (Answer: Cross-attention attention cost is proportional to 256 * 77 = 19,712. But the block also has self-attention at 256 * 256 = 65,536. Total: ~85,248. Joint attention cost: 333 * 333 = 110,889. So joint attention is ~30% more expensive per block in attention alone. However, there is only ONE attention operation instead of two, which reduces other overhead. The practical cost is comparable.)

2. "After the joint attention computation, MMDiT splits the output back into text tokens [77, d_model] and image tokens [256, d_model]. Why split them?" (Answer: Text tokens and image tokens need different post-attention processing -- each modality has its own FFN. Text and image representations live in different spaces, so the same FFN would not serve both well. The split also allows different adaLN-Zero modulation for each modality's FFN. Splitting preserves modality-specific processing while the attention itself is shared.)

3. "A colleague claims: 'Joint attention means SD3 treats text and image identically.' Is this accurate?" (Answer: No. The attention computation is shared, but text and image tokens have separate Q/K/V projections and separate FFN layers. Each modality maintains its own representational identity. They attend to each other through a shared attention mechanism, but they "think" differently. This is the key nuance of MMDiT that distinguishes it from naive concatenation.)

#### 8. Explain: The MMDiT Block in Detail

**Part A: Modality-specific projections**

"The simple version -- 'concatenate and self-attend' -- raises a question: text embeddings (from T5 or CLIP) and image patch embeddings (from patchify) live in very different representational spaces. Should they share the same W_Q, W_K, W_V projection matrices?"

"SD3's answer: no. Each modality gets its own projection matrices."

```
MMDiT Block:
  Text tokens [77, d_model] -> text W_Q, text W_K, text W_V -> text Q/K/V
  Image tokens [256, d_model] -> image W_Q, image W_K, image W_V -> image Q/K/V

  Concatenate: Q = [text_Q; image_Q] -> [333, d_head]
               K = [text_K; image_K] -> [333, d_head]
               V = [text_V; image_V] -> [333, d_head]

  Standard self-attention on concatenated Q/K/V -> [333, d_model]

  Split: text output [77, d_model], image output [256, d_model]

  Text output -> text FFN -> text residual
  Image output -> image FFN -> image residual
```

Address Misconception #4: "This is not naive concatenation where everything shares the same projections. Each modality formulates its Q, K, V through its own learned matrices -- they 'speak their own language.' The shared attention lets them 'hear each other.' The separate FFNs let them 'think independently.' Shared listening, separate thinking."

Negative example: "If text and image shared all projections (same W_Q, W_K, W_V, same FFN), the model would struggle: text embeddings from T5 (trained on language) and patch embeddings from patchify (linear projection of pixel values) have fundamentally different structures. Forcing them through the same projection is like asking a French speaker and a Japanese speaker to use the same dictionary. They need their own way of formulating thoughts, but a shared space where they can understand each other."

**Part B: What about timestep conditioning?**

"The timestep still enters through adaLN-Zero, just like in DiT. The conditioning vector c (timestep embedding + pooled CLIP embedding) produces gamma, beta, alpha for each sub-layer's LayerNorm. This is unchanged from DiT."

"So text enters through TWO paths in MMDiT:"
1. **Per-token path:** T5 and CLIP per-token embeddings join the attention sequence as text tokens (spatially varying, context-dependent)
2. **Global path:** Pooled CLIP embeddings join the timestep in adaLN-Zero (global, same at every spatial location)

"This dual-path text conditioning mirrors SDXL, where per-token CLIP embeddings entered cross-attention and the pooled OpenCLIP embedding entered adaptive norm. Same design principle, different mechanism for the per-token path."

#### 9. Check #3

Two predict-and-verify questions:

1. "In a standard DiT block, the adaLN-Zero MLP produces 6 parameters (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2). In an MMDiT block, which has separate FFNs for text and image, how many adaLN-Zero parameters do you expect?" (Answer: Each FFN sub-layer needs its own gamma, beta, alpha. With separate text and image FFNs, that is 3 params for the text FFN + 3 params for the image FFN + 3 params for the shared attention sub-layer's pre-norm. The exact count depends on implementation details, but the principle is: more sub-layers means more adaLN-Zero parameters per block. The conditioning MLP grows accordingly.)

2. "Could you use IP-Adapter with an MMDiT model? How would the architecture differ from IP-Adapter with a U-Net?" (Answer: In principle, yes. IP-Adapter adds image conditioning via decoupled cross-attention K/V. In an MMDiT, you could add CLIP image tokens to the combined sequence (alongside text and patch tokens), or add a separate decoupled attention path. The concept is the same -- add a new conditioning source -- but the implementation would differ. In practice, MMDiT's joint attention already provides a natural way to incorporate additional token sequences, so adding image reference tokens to the combined sequence is the more elegant approach.)

#### 10. Explain: Flow Matching in Practice

**Part A: Activating existing knowledge**

"SD3's training objective is the conditional flow matching you already know from lesson 7.2.2:"

Brief recap in 3-4 sentences:
- Straight-line interpolation: x_t = (1-t)*x_0 + t*epsilon
- Velocity prediction: v_theta(x_t, t) predicts v = epsilon - x_0
- Training loss: MSE(v_theta(x_t, t), epsilon - x_0)
- Straight trajectories by construction, fewer ODE solver steps needed

Address Misconception #2: "You already know everything about flow matching that SD3 uses. The training objective is identical to what you implemented in your flow matching notebook. This is not new content -- it is the same concept applied to a real model."

**Part B: Rectified flow application**

"SD3 uses rectified flow (from 7.2.2): after initial training, generate aligned (noise, data) pairs using the model, then retrain on these pairs. This makes the aggregate velocity field straighter, reducing the number of inference steps needed even further."

"In practice, SD3/Flux achieves good results in 20-30 steps, compared to 50+ for DDPM-based models. This is the practical payoff of the 'curved vs straight' insight from 7.2.2."

**Part C: Logit-normal timestep sampling (a minor optimization)**

"One new detail: instead of sampling timesteps uniformly during training (every t equally likely), SD3 uses logit-normal sampling. This biases training toward intermediate timesteps (around t=0.5) where the denoising task is hardest -- not pure noise (easy: predict roughly toward data center) and not nearly clean (easy: predict small refinements)."

"The intuition: intermediate noise levels are where the model must make the most important decisions about composition and structure. Spending more training compute at these timesteps improves overall quality."

TipBlock: "Logit-normal sampling is a training optimization, not a fundamental change. The training objective (flow matching velocity prediction) is the same. Only the distribution over which timesteps get trained more changes."

#### 11. Check #4

Two predict-and-verify questions:

1. "SD3 uses flow matching with velocity prediction. The student you trained in 7.2.2 used the same objective on 2D data. What changes when you apply this to a real image generation model?" (Answer: Nothing fundamental changes. The interpolation is the same (linear between noise and data). The training target is the same (velocity). The loss is the same (MSE). What changes is scale: the input is a high-dimensional latent [4, 64, 64] instead of 2D points, the model is a billion-parameter MMDiT instead of a small MLP, and the training data is millions of images. The concept is identical; the engineering is different.)

2. "Compare SD3's training setup to DDPM training from Series 6: what is the same and what is different?" (Answer: Same: sample data, add noise, predict something, MSE loss, backprop. Different: (1) interpolation formula (linear vs sqrt(alpha_bar) weighting), (2) prediction target (velocity vs noise), (3) timestep sampling (logit-normal vs uniform), (4) trajectory shape (straight vs curved). The training LOOP is identical in structure. The OBJECTIVE and SCHEDULE differ.)

#### 12. Elaborate: The Convergence

**Part A: The full SD3 pipeline, annotated**

"Here is the complete SD3 pipeline, with each component annotated by where you learned it:"

```
Full SD3 Pipeline:
1. Prompt -> CLIP ViT-L text encoder [77, 768]           (6.3.3: clip)
2. Prompt -> OpenCLIP ViT-bigG text encoder [77, 1280]    (7.4.1: sdxl)
3. Prompt -> T5-XXL text encoder [77, 4096]               (this lesson)
4. CLIP pooled embeddings + timestep -> c for adaLN-Zero   (7.4.2: diffusion-transformers)
5. Per-token embeddings projected -> text tokens [77, d]    (this lesson)
6. Noisy latent z_t [4, 64, 64] -> patchify -> [1024, d]  (7.4.2: diffusion-transformers)
7. Concatenate: [77 + 1024, d] = [1101, d]                 (this lesson: MMDiT)
8. N MMDiT blocks with joint attention + adaLN-Zero         (this lesson + 7.4.2)
9. Split output -> image tokens [1024, d]                   (this lesson)
10. Unpatchify -> [4, 64, 64]                               (7.4.2: diffusion-transformers)
11. Flow matching sampling step                              (7.2.2: flow-matching)
12. Repeat steps 6-11 for ~28 steps                         (7.2.2: fewer steps needed)
13. VAE decode z_0 -> [3, 1024, 1024]                       (6.3.5: from-pixels-to-latents)
```

"Thirteen steps. Every one traces to a lesson you completed. Nothing in this pipeline is unexplained."

InsightBlock: "This is the 'you already knew all of this' moment. SD3 is not a new paradigm. It is the convergence of concepts you built over the course of this entire series: transformers (Series 4) + latent diffusion (Series 6) + flow matching (7.2) + DiT (7.4.2) + better text encoding (this lesson). The frontier is not beyond your understanding. It is the synthesis of your understanding."

**Part B: What makes SD3/Flux the current frontier**

Three GradientCards summarizing the three independent advances that converge:

- **Architecture (DiT -> MMDiT):** Replace the U-Net with a transformer. Replace cross-attention with joint attention. Result: simpler architecture, better scaling, bidirectional text-image interaction.
- **Training objective (DDPM -> Flow matching):** Replace noise prediction with velocity prediction on straight trajectories. Result: fewer inference steps (20-30 vs 50+), simpler training (no noise schedule to tune).
- **Text encoding (CLIP -> CLIP + T5):** Add a language model alongside the vision-language model. Result: better compositional understanding, handling of complex prompts, richer text conditioning.

"Each advance is independent. You could have a U-Net with flow matching (and some models do). You could have DiT with DDPM noise prediction (the original DiT paper does). You could have T5 with cross-attention (Imagen does). SD3 combines all three."

**Part C: SD3 vs Flux (brief positioning)**

"SD3 and Flux are both MMDiT architectures from the same research lineage (Stability AI / Black Forest Labs). The key difference:"

- **SD3:** Uses the full triple encoder setup (CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL). Multiple size variants (SD3 Medium, SD3.5 Large).
- **Flux:** Further refined architecture, available in several variants: Flux.1 Dev (for development/research), Flux.1 Schnell (distilled for speed, ~4 steps), Flux.1 Pro (commercial API).

"The architectural principles are the same. The differences are in scale, training data, and distillation. For understanding how the architecture works, SD3 and Flux are the same family."

#### 13. Practice (Notebook)

Design: 4 exercises, Guided -> Guided -> Supported -> Independent progression.

**Exercise 1 (Guided): SD3 Pipeline Inspection**
- Load SD3 Medium via diffusers (`StableDiffusion3Pipeline`)
- Inspect the triple text encoder setup: print model class names, parameter counts
- Verify embedding shapes: CLIP ViT-L [77, 768], OpenCLIP ViT-bigG [77, 1280], T5-XXL [77, 4096]
- Inspect the transformer (MMDiT): count parameters, identify modality-specific Q/K/V projections
- Predict-before-run: "How many text encoder parameters vs denoising network parameters?"
- What it tests: the triple encoder setup and MMDiT architecture are real, inspectable configurations

**Exercise 2 (Guided): Joint Attention Visualization**
- Encode a prompt through all three text encoders
- Patchify a random latent (or use a partially denoised latent from the pipeline)
- Trace the joint attention: verify concatenated sequence length (text tokens + image tokens)
- Extract attention weights from one MMDiT block: visualize the [text+image, text+image] attention matrix
- Identify the four quadrants: text-to-text, text-to-image, image-to-text, image-to-image
- Predict-before-run: "Which quadrant will have the highest attention weights?" (Likely image-to-image, since spatial coherence is the primary task)
- What it tests: joint attention is a concrete, inspectable computation with real attention patterns

**Exercise 3 (Supported): SD3 Generation and Flow Matching Steps**
- Generate "a cat sitting on a beach at sunset" (the running example from 6.4.1) with SD3
- Vary the number of inference steps: 10, 20, 30, 50
- Compare quality across step counts: 20-30 steps should be sufficient (the flow matching payoff)
- Compare to SD v1.5 at the same step counts (use pre-generated reference images or load SD v1.5 if VRAM allows)
- Observe: SD3 at 20 steps vs SD v1.5 at 50 steps -- comparable or better quality in fewer steps
- What it tests: the practical benefit of flow matching (fewer steps for good results) and the overall quality improvement

**Exercise 4 (Independent): The Convergence Pipeline Trace**
- Generate an image with SD3, capturing intermediate outputs
- Trace the full pipeline: text encoding (measure shapes), patchify (verify token count), denoising steps (count), VAE decode (verify output shape)
- For each step, annotate which lesson covered the relevant concept
- Compare the SD3 pipeline to the SD v1.5 pipeline the student traced in 6.4.1: what changed and what is preserved?
- Bonus: compare to SDXL pipeline from 7.4.1 -- the three-generation progression
- What it tests: the convergence theme -- the student sees that every component traces to their prior knowledge

#### 14. Summarize

Key takeaways (echo mental models):

1. **One room, one conversation.** MMDiT replaces cross-attention with joint self-attention: concatenate text and image tokens, let everything attend to everything. Bidirectional interaction (text reads image, image reads text), one attention operation instead of two. Simpler and richer.

2. **Modality-specific processing, shared attention.** Text and image tokens have their own Q/K/V projections and their own FFN layers. They attend together but think differently. The projections map each modality into a shared attention space; the FFNs keep their representations distinct.

3. **Three encoders for three kinds of understanding.** CLIP ViT-L (visual alignment), OpenCLIP ViT-bigG (richer visual alignment), T5-XXL (deep linguistic understanding). Pooled CLIP for global conditioning via adaLN-Zero. Per-token embeddings for joint attention. Each contributes information the others cannot.

4. **Flow matching delivers.** The straight-line trajectories from 7.2.2 produce the practical result: SD3/Flux generates good images in 20-30 steps. Rectified flow makes the aggregate trajectories even straighter. Same concept you trained, applied at scale.

5. **Convergence, not revolution.** SD3/Flux combines: transformer blocks (Series 4) + latent diffusion (Series 6) + flow matching (7.2.2) + DiT (7.4.2) + better text encoding (this lesson). Every component traces to a lesson you completed. The frontier is not beyond your understanding -- it IS your understanding, combined.

#### 15. Series Conclusion

"You started Series 7 with Stable Diffusion v1.5 as your reference architecture: a U-Net trained with DDPM noise prediction, conditioned via a single CLIP encoder through cross-attention, generating 512x512 images in 50+ steps."

"Over eleven lessons, you traced the evolution:"
- Module 7.1: Added spatial and image conditioning without changing the model (ControlNet, IP-Adapter)
- Module 7.2: Reframed diffusion through the score function and discovered flow matching -- straight paths instead of curved
- Module 7.3: Collapsed the multi-step process with consistency models and adversarial distillation
- Module 7.4: Replaced the architecture entirely -- U-Net to DiT to MMDiT

"The current frontier (SD3, Flux) is the synthesis: a transformer processing a joint text-image token sequence, trained with flow matching, conditioned by three text encoders including a large language model. Every design choice in this architecture has a reason you now understand."

"You can read a diffusion model paper published today and trace its design choices back to concepts you have built from scratch. That was the goal of this course."

ModuleCompleteBlock + SeriesCompleteBlock.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (one gap: T5-XXL at MENTIONED, resolved with dedicated section)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (T5-XXL elevation from MENTIONED to INTRODUCED via dedicated section in Part 4B)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (convergence of five knowledge threads, answering DiT's text conditioning question)
- [x] At least 3 modalities planned for the core concept, each with rationale (4 modalities: concrete example, visual, verbal/analogy, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (MMDiT joint attention, T5-XXL as text encoder, logit-normal timestep sampling)
- [x] Every new concept connected to at least one existing concept (MMDiT to self-attention + cross-attention, T5 to CLIP + SDXL dual encoders, flow matching application to 7.2.2)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings. Three improvement findings that would meaningfully strengthen the lesson. Another pass needed after fixes.

### Findings

#### [IMPROVEMENT] — Inconsistent image token counts between explanation and pipeline trace

**Location:** "The Simple Idea" section (CodeBlock with 77+256=333 tokens) vs "The Convergence" section (pipeline trace with 77+1024=1101 tokens)
**Issue:** The joint attention explanation uses 256 image tokens (from [4,32,32] latent at p=2, matching the DiT lesson's ImageNet 256x256 example), but the full pipeline trace switches to 1024 image tokens (from [4,64,64] latent for 1024x1024 generation) without acknowledging the change. The numbers 333 and 1101 appear in different sections with no bridging explanation.
**Student impact:** The student traces through the 333-token example carefully, builds their mental model around those numbers, then encounters 1101 in the pipeline trace and wonders: did the patch count change? Did the patch size change? Did something else change? The cognitive overhead of reconciling two different concrete examples without explicit acknowledgment is a distraction from the convergence payoff.
**Suggested fix:** Add a brief parenthetical or sentence when the pipeline trace is introduced, e.g.: "Note: the earlier joint attention example used 256 image tokens from a 256x256 image (matching the DiT lesson). At SD3's native 1024x1024 resolution, patchify produces 1024 tokens, for a joint sequence of 1101." One sentence resolves the confusion and reinforces the patch-size-as-resolution-knob concept from the DiT lesson.

#### [IMPROVEMENT] — SD3 vs Flux comparison is thin and asymmetric

**Location:** "SD3 vs Flux" section (ComparisonRow at line ~1197)
**Issue:** The SD3 side has concrete details (triple encoder setup, size variants), but the Flux side says only "Further refined architecture" with no specifics about what was refined. The ComparisonRow is visually asymmetric (3 concrete items vs 3 vague items). The student cannot distinguish SD3 from Flux after reading this section. The lesson's scope boundaries say this comparison should be brief and for vocabulary positioning, which is fair, but "further refined architecture" teaches the student nothing -- they cannot even tell what kind of refinement.
**Student impact:** The student reads "further refined architecture" and learns nothing actionable. When they encounter "Flux.1 Dev" or "Flux.1 Schnell" in the wild, they know these are MMDiT variants but have no idea what distinguishes Flux from SD3. The vocabulary goal is partially missed.
**Suggested fix:** Replace "Further refined architecture" with one concrete distinguishing detail, e.g., "Single-stream blocks in later layers (text+image share projections, unlike MMDiT's dual-stream)" or "Uses only CLIP + T5 (drops second CLIP encoder)." Even one specific architectural difference gives the student a real distinction. Keep it brief -- one concrete item replaces one vague item.

#### [IMPROVEMENT] — Notebook Exercise 2 uses synthetic attention weights instead of real ones

**Location:** Notebook Exercise 2, Steps 5-7 (cells 20-22)
**Issue:** The exercise is titled "Joint Attention Visualization" and claims to visualize attention patterns in an MMDiT block, but it actually constructs synthetic random Q and K vectors and computes attention weights from those. The hook in Step 2 captures inputs/outputs but the code never extracts real Q/K from the captured data. The student sees a heatmap with labeled quadrants, but the attention pattern is not from the model -- it is from random vectors with manually-injected structure ("text_base" with added noise). The "Guided" label implies the student is observing real model behavior.
**Student impact:** The student follows a guided exercise that claims to show how MMDiT attends, but is actually seeing random patterns with artificial structure. If the student notices (and they might, since the code explicitly says "synthetic"), their trust in the exercise is undermined. If they do not notice, they form impressions about attention patterns that are not grounded in reality. Either way, the exercise does not deliver what it promises.
**Suggested fix:** Either (a) extract real attention weights from the captured hook data by computing Q @ K^T from the actual projected tensors in the attention module (this requires understanding the diffusers Attention module internals, which is doable), or (b) honestly reframe the exercise: rename it to "Joint Attention Structure" and explicitly state "We visualize the STRUCTURE of the four quadrants using synthetic data to understand what each quadrant represents. The notebook does not extract trained attention weights because the diffusers library does not expose them directly. What matters is the structure: which tokens can attend to which." Option (b) is simpler and still pedagogically valuable -- the quadrant structure is the key insight, not the specific weight values.

#### [POLISH] — Plan called for three-column visual comparison but lesson uses two columns

**Location:** "Cross-attention vs Joint attention comparison" ComparisonRow (line ~594)
**Issue:** The planning document's Modalities section specifies "ComparisonRow contrasting three text conditioning approaches: (1) U-Net cross-attention, (2) DiT class-conditional, (3) MMDiT joint attention." The built lesson uses a two-column ComparisonRow (cross-attention vs joint attention), omitting the DiT class-conditional column.
**Student impact:** Minimal. The two-column version is cleaner and the DiT class-conditional approach is adequately covered in the recap section. The three-column version would show the full progression but might dilute the core comparison.
**Suggested fix:** No action needed unless the builder wants to add the third column. Document the deviation as intentional: the two-column version focuses on the mechanism being replaced (cross-attention) vs the mechanism replacing it (joint attention), which is the clearest contrast.

#### [POLISH] — "learned lens" connection from plan not used in lesson

**Location:** "The MMDiT Block in Detail" section (line ~742)
**Issue:** The planning document's Connections to Prior Concepts table links modality-specific projections to the "learned lens" concept from 4.2.2/6.3.2. The built lesson uses the "speak their own language" / "hear each other" analogy instead, which is effective, but does not make the explicit "learned lens" callback.
**Student impact:** Negligible. The "speak their own language" analogy works well. The "learned lens" callback would be an additional reinforcement of a prior concept but is not needed for understanding.
**Suggested fix:** Optionally add "This is the 'learned lens' pattern: each modality looks at the input through its own learned projection" alongside the existing language analogy. Not required.

#### [POLISH] — Check #3 Q1 answer about adaLN-Zero parameter count is vague

**Location:** Check #3, Question 1 reveal (line ~887)
**Issue:** The answer says "Each FFN sub-layer needs its own gamma, beta, alpha... The exact count depends on implementation details" but the planning document specifies the principle clearly: more sub-layers = more adaLN-Zero parameters. The answer could be more concrete -- e.g., "at minimum 9 parameters: 3 for attention + 3 for text FFN + 3 for image FFN, compared to DiT's 6."
**Student impact:** Minor. The student gets the principle but not a concrete number to verify their prediction against. The predict-and-verify pattern works best when the reveal has a specific answer.
**Suggested fix:** Replace "The exact count depends on implementation details" with a concrete minimum: "At minimum 9 parameters per block: 3 for the shared attention sub-layer + 3 for the text FFN + 3 for the image FFN, compared to DiT's 6. The actual count may be higher if text and image have separate attention pre-norms."

### Review Notes

**What works well:**
- The convergence theme is executed beautifully. The five-thread GradientCard grid, the annotated pipeline trace, and the series conclusion all land the "you already knew all of this" feeling that the planning document identified as the emotional core.
- The motivation rule is thoroughly followed. Every concept is motivated before being introduced: CLIP's limitations before T5, cross-attention's one-directionality before joint attention, class-conditional DiT before text-conditioned MMDiT.
- The "one room, one conversation" analogy is strong and used consistently. It provides a sticky mental model for joint attention.
- All five planned misconceptions are addressed at the right locations in the lesson.
- The four check-your-understanding sections are well-placed and genuinely test prediction, not recall.
- The lesson respects scope boundaries. It does not drift into training procedures, Flux variant details, or implementation code.
- The notebook is well-structured with proper scaffolding progression (Guided -> Guided -> Supported -> Independent) and solution blocks for Supported/Independent exercises.

**Patterns to note:**
- The lesson's strongest sections are the MMDiT explanation (Sections 7-9) and the convergence pipeline trace (Section 13). These are the emotional and intellectual peaks.
- The flow matching section (Section 11) is appropriately brief -- it correctly identifies that the student already knows this material and reinforces rather than re-teaches.
- The series conclusion (Section 16) is well-crafted and provides genuine closure across 11 lessons.

**Notebook concern (not a separate finding, included in IMPROVEMENT #3):**
- Exercise 2's synthetic attention approach is the most significant notebook issue. Exercises 1, 3, and 4 are solid. Exercise 3 has proper `NotImplementedError` guards for TODOs and a detailed solution block. Exercise 4 is appropriately open-ended for an Independent exercise.

---

## Review — 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All three improvement findings from iteration 1 have been correctly applied. No critical or improvement findings remain. Two minor polish items identified.

### Iteration 1 Fix Verification

**IMPROVEMENT #1 (Inconsistent image token counts): FIXED.** A bridging sentence was added at line 1096-1099 of the lesson: "Note: the earlier joint attention example used 256 image tokens from a 256x256 latent (matching the DiT lesson's ImageNet example). At SD3's native 1024x1024 resolution, the latent is [4, 64, 64] and patchify produces 1024 tokens—the same operation at higher resolution." This clearly connects the 333-token example to the 1101-token pipeline trace and reinforces the patch-size-as-resolution-knob concept from the DiT lesson.

**IMPROVEMENT #2 (SD3 vs Flux comparison thin and asymmetric): FIXED.** The Flux side of the ComparisonRow now has three concrete, distinguishing details: (1) "Single-stream blocks in later layers (text+image share projections after initial dual-stream processing)," (2) "Drops the second CLIP encoder—uses only CLIP ViT-L + T5-XXL," (3) variant names with descriptions. The student can now distinguish Flux from SD3 with real architectural knowledge.

**IMPROVEMENT #3 (Notebook Exercise 2 synthetic attention): FIXED.** Exercise 2 was reframed as "Joint Attention Structure" (not "Visualization") with explicit disclosure: "Why synthetic data? The diffusers library does not expose per-head attention weights... Rather than fighting the implementation, we construct synthetic Q/K vectors matching the model's dimensions to demonstrate what each quadrant of the joint attention matrix *represents*." Cells 20-21 also state clearly that weights are synthetic. The pedagogical value (quadrant structure as architectural property) is preserved honestly.

### Findings

#### [POLISH] — "One Room, One Conversation" aside repeats the body text

**Location:** "One Room, One Conversation" section (InsightBlock aside at line ~651)
**Issue:** The aside InsightBlock at line 651-656 reads: "Cross-attention: two rooms, one-way mirror, the image watches the text. Joint attention: one room, everyone hears everyone. Simpler and richer." This is a compressed restatement of the two paragraphs that immediately precede it in Row.Content (lines 630-656). The aside provides no additional information or different perspective beyond what the body text already says.
**Student impact:** Negligible. The student reads the analogy in the body, then sees it compressed in the aside. At worst, it feels slightly repetitive. At best, the aside serves as a quick reference if the student scrolls back.
**Suggested fix:** Optionally replace the aside content with a different angle, e.g., a concrete example of disambiguation ("When you prompt 'a crane near a river,' cross-attention cannot disambiguate—the text embeddings are the same regardless of image content. In MMDiT, the text representation of 'crane' can update based on what the image actually shows."). The disambiguation example is already in the body text (line 644), so alternatively just remove the aside or use a forward pointer ("This bidirectional interaction is what makes SD3 better at complex prompts—you will see this in Exercise 3"). Not required.

#### [POLISH] — Check #3 Q1 adaLN-Zero answer still hedges on the concrete count

**Location:** Check #3, Question 1 reveal (line ~887-893)
**Issue:** Iteration 1 flagged this as Polish, suggesting the answer replace "The exact count depends on implementation details" with a concrete minimum like "at minimum 9 parameters per block." The current answer still says "The exact count depends on implementation details, but the principle is: more sub-layers means more adaLN-Zero parameters per block." The principle is taught, but the student cannot verify their prediction against a specific number.
**Student impact:** Minor. The predict-and-verify pattern works best when the reveal has a concrete answer. "At minimum 9" would give the student something to check against. The current answer teaches the principle without the payoff of a specific number.
**Suggested fix:** Replace "The exact count depends on implementation details, but the principle is: more sub-layers means more adaLN-Zero parameters per block. The conditioning MLP grows accordingly." with "At minimum 9 parameters per block: 3 for the shared attention sub-layer + 3 for the text FFN + 3 for the image FFN, compared to DiT's 6. The conditioning MLP grows accordingly." Not required.

### Review Notes

**What works well:**
- All three iteration 1 improvement fixes were applied correctly and effectively. The bridging sentence for token counts is clean. The Flux comparison is now concrete and asymmetric in a useful way. The notebook exercise reframing is honest and preserves pedagogical value.
- The convergence theme remains the lesson's greatest strength. The five-thread GradientCard grid, the annotated pipeline trace, and the series conclusion all deliver the "you already knew all of this" feeling.
- The motivation rule is consistently followed. Every concept is motivated before introduction.
- All five planned misconceptions are addressed at the right locations with concrete negative examples or explicit statements.
- The four check-your-understanding sections test prediction, not recall, and are placed at appropriate intervals.
- The notebook is well-structured with honest framing, proper scaffolding progression, `NotImplementedError` guards for TODO exercises, and detailed solution blocks.
- The lesson respects its scope boundaries throughout—no drift into training procedures, Flux variant details, or implementation code.
- As the final lesson of Series 7 (and the course's generative model track), the emotional arc lands well. The student finishes understanding that the frontier is the synthesis of concepts they built from scratch across 50+ lessons.

**Patterns to note:**
- The two remaining Polish items are both from iteration 1's Polish findings that were not addressed. They are genuinely minor and do not affect the lesson's effectiveness.
- The lesson's structure closely follows the planning document with one documented deviation (two-column vs three-column comparison), which was correctly identified as intentional in iteration 1.
