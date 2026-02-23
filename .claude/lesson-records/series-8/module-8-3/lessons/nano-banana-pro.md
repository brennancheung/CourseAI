# Lesson: Nano Banana Pro (Gemini 3 Pro Image)

**Module:** 8.3 Architecture Analysis
**Slug:** `nano-banana-pro`
**Status:** Planning complete

---

## Phase 1: Orient (Student State)

The student has completed the entire course through Series 7 and two modules of Series 8. The relevant concept inventory for this lesson:

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Autoregressive generation (text) | APPLIED | building-nanogpt (4.3.1) | Implemented generate() method: forward pass, take last position logits, apply temperature, sample, append. Conceptual loop from 4.1.1. Deeply understood. |
| Decoder-only transformer architecture | APPLIED | building-nanogpt (4.3.1) | Built full GPT in PyTorch: attention heads, MHA, FFN, block stacking, weight tying. Can implement from scratch. |
| Self-attention (Q/K/V, multi-head, scaled dot-product) | APPLIED | Module 4.2 (6 lessons) | Deep treatment across 6 lessons. Hand-traced worked examples, interactive widgets, implemented in code. |
| Cross-attention (Q from one modality, K/V from another) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | "Same formula, different source for K and V." Spatial features produce Q, text tokens produce K/V. Extended in SAM lesson. |
| MMDiT joint attention (text + image tokens concatenated, self-attention on combined sequence) | DEVELOPED | sd3-and-flux (7.4.3) | "One room, one conversation." Four attention types (image-to-text, text-to-image, etc.). Modality-specific Q/K/V projections. |
| VAE encoding/decoding (continuous latent space, encoder-decoder architecture) | DEVELOPED | Module 6.1 (4 lessons) | Built VAE, explored latent spaces, understands bottleneck, KL divergence, reconstruction-vs-regularization tradeoff. |
| Latent diffusion (operate in VAE latent space, not pixel space) | DEVELOPED | Module 6.4 | Understands why latent space (compression, compute efficiency), how VAE encoder/decoder wraps the diffusion process. |
| Diffusion denoising process (forward noise addition, reverse denoising, noise prediction) | DEVELOPED | Module 6.2 (5 lessons) | Full DDPM understanding: noise schedule, forward process, reverse process, training objective, sampling. |
| Classifier-free guidance | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Training with random text dropout, two forward passes at inference, amplification formula. Concrete numerical example. |
| CLIP/SigLIP contrastive learning | DEVELOPED | clip (6.3.3), siglip-2 (8.1.1) | Dual-encoder architecture, shared embedding space, sigmoid vs softmax loss. |
| DiT / patchify operation | DEVELOPED | diffusion-transformers (7.4.2) | "Tokenize the image." Patches as tokens, adaLN-Zero conditioning, scaling recipe. |
| S3-DiT single-stream architecture | INTRODUCED | z-image (7.4.4) | Shared projections + refiner layers. "Translate once, then speak together." |
| Flow matching / rectified flow | INTRODUCED | Module 7.2, sd3-and-flux (7.4.3) | Straight-line interpolation, velocity prediction, 20-30 steps. |
| Quantization (weight quantization for memory reduction) | DEVELOPED | lora-and-quantization (4.4.4) | Absmax quantization with worked example. int8/int4. |
| KV caching for autoregressive inference | DEVELOPED | scaling-and-efficiency (4.3.3) | Cache K/V from previous steps, O(n) vs O(n^2). Concrete compute savings. |
| ConvTranspose2d (learned upsampling) | INTRODUCED | autoencoders (6.1.2) | Small spatial -> large spatial. Used in VAE decoder and U-Net. |
| ViT (Vision Transformer, image patches as tokens) | MENTIONED | clip (6.3.3), diffusion-transformers (7.4.2) | Named as CLIP image encoder option; patchify operation taught in DiT context. Student knows ViT processes patches like tokens. |
| DMD / DMDR (distillation + RL for image generation) | INTRODUCED | z-image (7.4.4) | Third distillation paradigm. Spear/shield decomposition. DPO + GRPO breaking teacher ceiling. |
| Safety stack for image generation | DEVELOPED | image-generation-safety (8.2.1) | Multi-layered defense, CLIP-based classifiers, concept erasure. |

**Mental models and analogies already established:**
- "Tokenize the image" (patchify = tokenization for images)
- "One room, one conversation" (joint attention for multimodal sequences)
- "Shared listening, separate thinking" (modality-specific projections in joint attention)
- "Translate once, then speak together" (S3-DiT refiner layers)
- "Two knobs, not twenty" (transformer scaling recipe)
- "Convergence, not revolution" (frontier systems combine known components)
- Autoregressive loop from text generation (predict next token, append, repeat)

**What was explicitly NOT covered in prior lessons:**
- Discrete visual tokenization (VQ-VAE, ViT-VQGAN, codebook approach) -- the student knows continuous VAE latent spaces but NOT discrete token vocabularies for images
- Autoregressive image generation (generating image tokens one at a time like text tokens) -- entirely untaught
- Google Parti or any autoregressive text-to-image system
- DALL-E 1's autoregressive approach (DALL-E 2/3 were not taught either, only mentioned in passing)
- Chameleon-style unified token streams
- Hybrid autoregressive + diffusion approaches (HART, BLIP3o-NEXT)

**Readiness assessment:** The student is well-prepared. They have deep understanding of both transformers/autoregressive generation (Series 4) and diffusion (Series 6-7), plus vision-language models (8.1). The key gap is discrete visual tokenization (VQ-VAE/ViT-VQGAN), which is a MEDIUM gap -- the student understands continuous VAE latent spaces, so extending to discrete codebook tokenization requires a dedicated section but not a prerequisite lesson. The student has never seen autoregressive image generation, but has all the pieces to understand it: autoregressive text generation + image tokenization = autoregressive image generation.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to analyze how Nano Banana Pro (Gemini 3 Pro Image) likely works by reasoning from observable behavior, disclosed architecture fragments, and published precedents -- and in doing so, understand why autoregressive image generation is a viable alternative to diffusion that excels at text rendering and compositional reasoning.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Autoregressive text generation (predict-next-token loop) | DEVELOPED | APPLIED | building-nanogpt (4.3.1) | OK | Need to understand the generation loop well enough to extend it to image tokens. Student has APPLIED depth -- exceeds requirement. |
| Decoder-only transformer architecture | DEVELOPED | APPLIED | building-nanogpt (4.3.1) | OK | Need to understand the backbone architecture. Student has APPLIED depth. |
| Self-attention (Q/K/V, multi-head) | DEVELOPED | APPLIED | Module 4.2 | OK | Needed to understand why autoregressive attention enables compositional control. |
| VAE encoding/decoding (continuous latent space) | DEVELOPED | DEVELOPED | Module 6.1 | OK | Bridge concept: VQ-VAE extends VAE with discrete tokens. Student needs VAE foundation to understand the extension. |
| Diffusion denoising pipeline (noise prediction, iterative refinement) | DEVELOPED | DEVELOPED | Module 6.2 | OK | Needed as contrast paradigm. Student understands the full diffusion pipeline. |
| Classifier-free guidance | INTRODUCED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Needed to understand why diffusion struggles with text rendering (CFG amplifies but does not create compositional structure). |
| DiT / patchify (image patches as tokens) | INTRODUCED | DEVELOPED | diffusion-transformers (7.4.2) | OK | Bridge to visual tokenization: student already knows images can be turned into token sequences. |
| Cross-attention (text conditions image generation) | INTRODUCED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Needed for understanding how text controls generation in diffusion vs autoregressive. |
| Discrete visual tokenization (VQ-VAE, codebook) | INTRODUCED | NOT TAUGHT | -- | MISSING | Student knows continuous VAE but NOT discrete tokenization. MEDIUM gap: continuous VAE is the foundation, extension to discrete codebook is conceptually similar with key differences. |
| Autoregressive image generation | INTRODUCED | NOT TAUGHT | -- | MISSING | The core new concept. Student has autoregressive text generation + image tokenization pieces but has never seen them combined. MEDIUM gap: both pieces exist, the combination is the new idea. |

### Gap Resolution

| Gap | Size | Action |
|-----|------|--------|
| Discrete visual tokenization (VQ-VAE, codebook) | MEDIUM | Dedicated section within this lesson. Build from the student's VAE knowledge: "VAE encodes to a continuous vector. VQ-VAE encodes to a discrete integer from a codebook -- the same way a text tokenizer maps words to integer IDs." Show the codebook lookup step: encoder output -> find nearest codebook vector -> replace with that vector's integer index. This gives the student enough to understand why autoregressive generation over image tokens works (discrete tokens = same setup as text generation). Target depth: INTRODUCED. |
| Autoregressive image generation | MEDIUM | Core concept of the lesson. Build from autoregressive text generation: "You know the predict-next-token loop for text. If images are discrete tokens, the same loop generates images." The key insight is that the model does not need separate text and image generation -- it just continues the token sequence. Target depth: INTRODUCED. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Autoregressive image generation is just diffusion with a different name" | Both generate images from a model. The student has spent 20+ lessons on diffusion and may pattern-match everything to it. | Diffusion iteratively refines ALL pixels simultaneously from noise; autoregressive generates tokens one at a time in sequence order. Diffusion has no concept of "earlier" or "later" pixels -- it denoises the whole image at once. Autoregressive has explicit ordering. Generate 100 tokens, then stop -- you get a partial image (top half). Stop diffusion at 50% -- you get a blurry full image. Fundamentally different. | Section 3 (Explain: Two Paradigms), immediately after introducing autoregressive image generation. |
| "Text rendering must require a specialized text-detection module" | Diffusion models have been notoriously bad at text, so the student may assume text rendering needs a special pipeline. | Nano Banana Pro renders text accurately because its autoregressive generation processes image content token-by-token, left-to-right / top-to-bottom, with each token's generation conditioned on ALL previous tokens including the text tokens specifying what to write. The model "sees" the text instruction and the already-rendered characters in its context window, just like a language model "sees" already-generated words. No separate text module needed -- the architecture inherently handles it. | Section 4 (Elaborate: Why Text Rendering Works), after explaining the generation process. |
| "The mandatory 'thinking' step is just marketing / unnecessary latency" | Google's "reasoning" language sounds like hype. The student may dismiss it as a gimmick added for branding. | Compare the composition planning needed for "a red car in front of a blue house with the text 'SOLD' on a sign" vs generating token-by-token without planning. Without a thinking step, the model must commit to early tokens (sky, background) before knowing the composition. With thinking, it can decompose the prompt into spatial layout, object placement, and text positioning BEFORE generating the first image token. This is analogous to how chain-of-thought helps LLMs solve multi-step problems. | Section 5 (Elaborate: The Thinking Step), after the core architecture explanation. |
| "We can't know anything useful about how Nano Banana Pro works because there's no paper" | The student may assume that without a published paper, analysis is pure guesswork. | The lesson itself disproves this: by combining observable behavior (94% text accuracy, mandatory thinking, <10s generation) + disclosed fragments (Gemini 3 backbone, GemPix 2, autoregressive tokens) + published precedents (Parti, ViT-VQGAN, Chameleon), we can construct specific, falsifiable hypotheses about the architecture. Many of these hypotheses are strongly constrained by the evidence. | Throughout the lesson; explicitly addressed in the Hook and revisited in the Summary. |
| "Autoregressive must be slower than diffusion because it generates tokens sequentially" | Sequential generation sounds inherently slow. Diffusion does 20-50 steps but processes all pixels in parallel per step. | Nano Banana Pro generates high-resolution images in under 10 seconds. The key is token efficiency: 1,120 tokens for 2K output means only 1,120 forward passes through the decoder, and each forward pass can use KV caching (the student knows this from 4.3.3). Furthermore, the visual tokenizer's codebook compresses spatial information far more than patch-level tokens -- each token represents a meaningful visual unit, not a raw pixel patch. | Section 6 (Elaborate: Speed and Efficiency), after discussing token counts. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Text rendering comparison: "Write 'HELLO WORLD' in a sunset scene" -- diffusion (garbled text, correct scene) vs autoregressive (correct text AND scene) | Positive | Shows the key behavioral difference that motivates the architectural analysis. The student can immediately see WHY autoregressive matters for text. | Text rendering is the most dramatic, observable difference between paradigms. It makes the architectural question visceral: "What is structurally different about this model that it can do what diffusion cannot?" |
| Partial generation comparison: stop autoregressive at 50% tokens vs stop diffusion at 50% denoising steps | Positive (contrastive) | Reveals the fundamental structural difference between paradigms. Autoregressive at 50%: coherent top half, blank bottom. Diffusion at 50%: blurry full image. | This example makes the paradigm difference concrete and visual. The student can predict what each partial result looks like, which forces engagement with the generation mechanics. |
| Codebook tokenization: a 256x256 image tokenized to 32x32 = 1,024 integer tokens, compared to the text "a cat sitting on a mat" tokenized to ~8 integer tokens | Positive | Bridges the gap between text tokenization (which the student knows deeply) and visual tokenization (which is new). Shows structural isomorphism: both produce sequences of integers from a vocabulary. | The parallel between text and image tokenization is the key conceptual bridge. Once the student sees this, autoregressive image generation becomes "obvious." |
| DALL-E 1 vs DALL-E 2/3 trajectory: autoregressive (DALL-E 1) abandoned for diffusion (DALL-E 2/3) because of quality... but now autoregressive returns at larger scale | Negative (with nuance) | Prevents the misconception that autoregressive was always viable. Shows the historical tension: autoregressive had structural advantages but lost on quality at 2021-era scale. At 2025+ scale (Gemini 3 Pro backbone), autoregressive quality matches or exceeds diffusion. | This example teaches the student to reason about WHY paradigms shift: it is not that one is inherently better, but that scale changes the tradeoff. Connects to "convolutions vs attention at scale" from DiT lesson. |

---

## Phase 3: Design

### Narrative Arc

You have spent the last 80+ lessons building two parallel tracks of understanding: how transformers generate text token-by-token (Series 4), and how diffusion models generate images by iteratively denoising noise (Series 6-7). These tracks never crossed -- text generation was autoregressive, image generation was diffusion. Then Google shipped Nano Banana Pro, and the internet noticed something strange: it renders text in images almost perfectly. Not "pretty good for an AI" -- 94%+ accuracy, legible fonts, correct spelling, proper kerning. Every diffusion model you studied struggles with text rendering because diffusion refines all pixels simultaneously with no explicit awareness of character sequence. So what changed? The answer is that Google crossed the tracks: they built an image generator that works like a language model, generating images token-by-token using the same autoregressive architecture that powers Gemini's text generation. But they did not publish a comprehensive paper. In this lesson, you become the architecture detective: starting from what you can observe (the behavior), what Google has disclosed (fragments), and what the research community has published (precedents), you will reconstruct a plausible picture of how this system works. Along the way, you will understand a fundamentally different approach to image generation -- one that was tried and abandoned years ago, but has now returned at a scale where it works.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual/Diagrammatic | Architecture comparison diagram: diffusion pipeline (noise -> iterative denoising -> image) vs autoregressive pipeline (text tokens -> image tokens -> decoder -> image). Mermaid diagram showing the inferred Nano Banana Pro pipeline end-to-end. | The two paradigms need to be seen side-by-side to make the structural differences concrete. A diagram of the inferred pipeline makes the speculation tangible and falsifiable. |
| Concrete example | Worked token-by-token generation trace: show 6-8 image tokens being generated sequentially, each one conditioned on all previous tokens (both text and image). Show what the "context window" looks like at each step. | Autoregressive generation is best understood by watching it happen step by step. The student knows this pattern from text generation -- seeing it applied to image tokens makes the transfer concrete. |
| Verbal/Analogy | "Writing a letter vs painting a mural" -- autoregressive is like writing a letter (each word follows the last, you can reference what you already wrote, including quoted text); diffusion is like painting a mural (you rough in the whole scene, then refine everywhere simultaneously). Text rendering in a letter is natural; text rendering on a mural requires careful planning because you are painting all letters at once. | This analogy captures the fundamental difference in a way that immediately explains the text rendering advantage. |
| Symbolic/Code | Pseudocode for the autoregressive image generation loop, structurally identical to the GPT generate() method from 4.3.1 but producing image tokens. Side-by-side with the diffusion sampling loop. | Pseudocode makes the connection to the student's implementation experience explicit. Seeing the structural identity between text generation and image generation in code is a powerful "of course" moment. |
| Intuitive | "Of course" chain: (1) if images can be tokenized into discrete integers, (2) and you have a transformer that generates tokens autoregressively, (3) then you can generate images the same way you generate text, (4) and of course text rendering works because each character is just another token in the sequence. | The "of course" progression leverages the student's existing knowledge to make autoregressive image generation feel inevitable rather than surprising. |

### Cognitive Load Assessment

- **New concepts:** 2 genuinely new concepts:
  1. Discrete visual tokenization (VQ-VAE/ViT-VQGAN codebook approach)
  2. Autoregressive image generation (generating image tokens sequentially like text)
- **Additional new-ish material:** The architectural analysis/speculation framework is new but lightweight -- it is a meta-skill, not a technical concept with formulas.
- **Previous lesson load:** The last lesson the student completed in the course was either SAM 3 or image-generation-safety (both BUILD). This lesson is STRETCH, which is appropriate after BUILD.
- **Assessment:** 2 new concepts + analytical reasoning framework. Within the 2-3 new concept limit. The analytical framework draws heavily on synthesis of existing knowledge rather than introducing new technical material, so it functions more like a "connection" exercise than a third new concept.

### Connections to Prior Concepts

| Prior Concept | How It Connects | Where in Lesson |
|---------------|----------------|-----------------|
| Autoregressive text generation (4.3.1) | The SAME generation loop, applied to image tokens instead of text tokens. "You already wrote this code." | Core explanation (Section 3). Pseudocode side-by-side. |
| VAE encoding/decoding (6.1) | VQ-VAE extends VAE with discrete codebook. "Same encoder-decoder structure, but the bottleneck is discrete integers instead of continuous vectors." | Gap resolution section (Section 2). |
| DiT patchify / "tokenize the image" (7.4.2) | DiT turns images into token sequences for transformer processing. VQ-VAE does the same thing but with discrete tokens instead of continuous patches. | Gap resolution section and core explanation. |
| KV caching (4.3.3) | Explains how autoregressive generation can be fast despite sequential token generation. Each new token only needs one forward pass. | Speed/efficiency section (Section 6). |
| Cross-attention vs joint attention (6.3.4, 7.4.3) | In diffusion, text conditions image via cross-attention or joint attention. In autoregressive generation, text and image tokens share the SAME sequence -- text conditioning is implicit through self-attention over the shared context. | Architecture comparison (Section 3). |
| Chain-of-thought reasoning (5.2) | The mandatory "thinking" step is analogous to CoT: decompose the problem before generating the answer. | Thinking step section (Section 5). |
| "Convolutions vs attention at scale" (7.4.2) | Same tradeoff pattern: autoregressive lost to diffusion at 2021 scale, wins at 2025 scale. Scale changes which approach dominates. | Historical context (DALL-E trajectory example). |
| S3-DiT / Z-Image single-stream (7.4.4) | Both S3-DiT and Nano Banana Pro process text+image in a unified transformer. But S3-DiT still uses diffusion for generation; NBP uses autoregressive. | Architecture comparison section. |

**Potentially misleading prior analogies:**
- "Tokenize the image" from DiT: DiT patchify produces CONTINUOUS embeddings that feed into a diffusion process. VQ-VAE tokenization produces DISCRETE integers that feed into an autoregressive process. The student might conflate these. Must explicitly address this distinction.
- "One room, one conversation" from MMDiT: In MMDiT, text and image tokens attend jointly but the generation process is still diffusion (simultaneous refinement). In Nano Banana Pro, text and image tokens are in the same sequence but generation is sequential. Same attention mechanism, different generation paradigm.

### Scope Boundaries

**This lesson IS about:**
- Understanding autoregressive image generation as a paradigm alternative to diffusion
- Discrete visual tokenization (VQ-VAE/codebook) at INTRODUCED depth
- Constructing a plausible architectural hypothesis for Nano Banana Pro from evidence
- Why autoregressive excels at text rendering and compositional reasoning
- The hybrid autoregressive+diffusion possibility (HART, BLIP3o-NEXT) at MENTIONED depth

**This lesson is NOT about:**
- Implementing VQ-VAE or any image tokenizer from scratch
- Training an autoregressive image generator
- The mathematical details of VQ-VAE training (commitment loss, codebook collapse, etc.)
- A comprehensive survey of all autoregressive image generation methods
- Google's training infrastructure or compute costs
- The ethics of AI-generated images
- Comparing Nano Banana Pro to every other image generation model (keep focused on architectural understanding)

**Target depths:**
- Discrete visual tokenization: INTRODUCED (explain with intuition, not implement)
- Autoregressive image generation: INTRODUCED (understand the paradigm, trace through examples, not implement)
- Architectural analysis as a meta-skill: INTRODUCED (practice on one example, not formalized into a methodology)

### Lesson Outline

#### 1. Context + Constraints
- This is a Special Topics lesson analyzing a production system without a full technical paper
- We are constructing informed hypotheses, not stating confirmed facts
- We will NOT implement anything -- this is an analysis/reasoning lesson
- We will NOT cover every autoregressive image model -- we focus on understanding the paradigm through one case study
- ConstraintBlock with scope boundaries

#### 2. Hook: The Text Rendering Mystery
- **Type:** Before/after + real-world impact
- Open with the observation: "Diffusion models have been bad at text rendering for years. DALL-E 3, Midjourney, Stable Diffusion -- all struggle with spelling. Then Nano Banana Pro shipped with 94%+ text rendering accuracy. What changed?"
- ComparisonRow: diffusion text rendering (garbled) vs Nano Banana Pro text rendering (accurate)
- This immediately creates a puzzle the student wants to solve. The rest of the lesson is the detective work.
- Also note: mandatory thinking step, native multimodal, under 10 seconds for 4K -- these are all behavioral clues

#### 3. The Evidence: What We Know (Recap + Gap Resolution)

**3a. What Google Has Disclosed**
- Built on Gemini 3 Pro backbone (decoder-only transformer)
- Uses "GemPix 2" rendering engine
- "Reasoning-guided synthesis" -- model plans before rendering
- Autoregressive image token generation
- Token efficiency: 1,120 tokens for 2K output
- Multi-stage internal pipeline with self-correction

**3b. Gap Resolution: Discrete Visual Tokenization (VQ-VAE)**
- Bridge from the student's VAE knowledge: "You built a VAE that encodes images to continuous vectors. VQ-VAE takes one more step: it maps each encoder output to the NEAREST vector in a learned codebook, producing a discrete integer."
- Codebook analogy: "Like a text tokenizer has a vocabulary of ~50,000 words, a visual tokenizer has a codebook of ~8,000-100,000 visual patterns."
- Concrete example: 256x256 image -> encoder -> 32x32 grid of continuous vectors -> each mapped to nearest codebook entry -> 32x32 = 1,024 integer tokens
- Side-by-side: text tokenization ("a cat on a mat" -> [64, 2815, 319, 257, 2420]) vs image tokenization (cat image -> [4821, 1293, 7744, ...])
- Key insight: once you have discrete tokens, the image is just another "language" with a different vocabulary
- ViT-VQGAN: Google's specific visual tokenizer (ViT encoder, VQGAN codebook). Parti used this. Nano Banana Pro likely uses an evolved version.
- TipBlock: "You do NOT need to understand how VQ-VAE is trained. The key idea is: images in, discrete integers out. Same as text tokenization."

**3c. Published Precedents**
- Google Parti (2022): proved autoregressive can match diffusion quality at scale. Used ViT-VQGAN + encoder-decoder transformer (20B parameters).
- Chameleon (Meta): unified autoregressive model for text + images in the same token stream. Early-fusion multimodal.
- DALL-E 1 -> DALL-E 2/3 trajectory: started autoregressive, switched to diffusion for quality. Now autoregressive returns at scale.
- HART: hybrid approach (autoregressive for global structure, diffusion refinement for local detail).
- BLIP3o-NEXT: autoregressive tokens condition a diffusion model.

#### 4. Core Explanation: Two Paradigms for Image Generation

**The paradigm split:**
- Diffusion: start with noise, iteratively refine ALL pixels simultaneously over 20-50 steps. Each step refines the whole image.
- Autoregressive: start with text prompt tokens, generate image tokens ONE AT A TIME, each conditioned on all previous tokens. Each step adds one token.

**"Writing a letter vs painting a mural" analogy:**
- Autoregressive = writing a letter. Each word follows the last. You can reference what you already wrote. Quoting text is natural -- you write one character at a time.
- Diffusion = painting a mural. You rough in the whole scene, then refine everywhere simultaneously. Text on the mural requires careful planning because you paint all letters at once.

**Pseudocode comparison:**
- Show diffusion sampling loop (the student's familiar `for t in reversed(range(T))` loop)
- Show autoregressive image generation loop (structurally identical to GPT generate() from 4.3.1, but producing image tokens)
- "You already wrote this code. The only difference is the vocabulary."

**The partial generation comparison:**
- Stop autoregressive at 50% tokens: coherent top-left, blank bottom-right (raster scan order)
- Stop diffusion at 50% steps: blurry full image
- This reveals the structural difference: autoregressive builds the image piece by piece, diffusion refines the whole image simultaneously

**Misconception #1 address:** "This is NOT diffusion with a different name." Explicit ComparisonRow: diffusion (noise -> all-pixel refinement -> image) vs autoregressive (text tokens -> sequential image tokens -> decode to image).

#### 5. Check #1: Predict and Verify
- "If Nano Banana Pro generates image tokens autoregressively, what happens when you ask it to generate TWO images in a conversation?" (Answer: the second image's tokens are generated with the first image's tokens still in the context window -- this is how it maintains multi-image consistency for up to 5 people across 14 input images.)
- "Why does autoregressive generation naturally handle text rendering while diffusion struggles?" (Answer: each character's pixels are generated token-by-token with awareness of the text prompt AND the already-rendered characters. It is sequential and conditional, like typing.)

#### 6. The Inferred Architecture: Putting the Pieces Together

**The pipeline hypothesis:**

Step 1: User prompt enters the Gemini 3 Pro decoder-only transformer
Step 2: "Thinking" step -- the model plans the composition (see Section 7)
Step 3: The transformer generates image tokens autoregressively (likely from a ViT-VQGAN-evolved codebook)
Step 4: GemPix 2 (likely the visual tokenizer + decoder) converts discrete tokens back to a high-resolution image

**Mermaid diagram:** Full inferred pipeline

**What is GemPix 2?**
- Google's name for the visual tokenizer + decoder component
- Likely an evolved ViT-VQGAN: a visual tokenizer that maps between images and discrete tokens
- "GemPix" suggests Gemini + pixels -- the bridge between the LLM's token space and pixel space
- Token efficiency (1,120 tokens for 2K) suggests a highly optimized tokenizer with large effective patch size or hierarchical tokenization

**Open question: purely autoregressive or hybrid?**
- Evidence for purely autoregressive: Google says "autoregressive image token generation," mandatory thinking step, Gemini backbone is a decoder-only transformer
- Evidence for possible hybrid: under 10s for 4K (fast for pure autoregressive at this resolution), physics-accurate lighting (diffusion excels at this), HART and BLIP3o-NEXT show hybrid is viable
- Best hypothesis: primarily autoregressive with possible diffusion refinement in GemPix 2 decoder
- Honest: we do not know. Both are plausible. Mark this as an open question.

#### 7. Elaborate: Why Text Rendering Works

**The structural argument:**
- In diffusion: text is injected via cross-attention (text embeddings condition the denoising of ALL pixels simultaneously). The model must learn that specific spatial regions should contain specific characters, but it refines all regions at once with no explicit character ordering.
- In autoregressive: the text instruction is in the context window. As the model generates image tokens for the region containing text, it has BOTH the instruction ("write HELLO") AND the already-generated tokens (the H, the E, the L...) in its attention context. It generates one visual token at a time, conditioned on everything before it.
- "This is the same reason GPT can spell words: it generates one token at a time, each conditioned on the previous ones."

**Misconception #2 address:** "No specialized text-detection module. The architecture inherently handles text because autoregressive generation IS sequential, conditional generation -- the same property that makes language models good at text."

#### 8. Elaborate: The Thinking Step

**What the mandatory "thinking" step likely does:**
- Prompt decomposition: "a red car in front of a blue house with 'SOLD' on a sign" -> identify objects (car, house, sign), properties (red, blue), text ("SOLD"), spatial relationships (in front of)
- Composition planning: decide spatial layout before generating image tokens
- This is analogous to chain-of-thought reasoning from Series 5: decompose the problem before committing to the answer

**Why it matters for autoregressive generation specifically:**
- Without planning, the model must commit to early tokens (sky, background) before knowing where objects will go
- With planning, it can allocate spatial regions first, then fill them in order
- This is less critical for diffusion because diffusion refines the whole image simultaneously -- it does not commit to any region first

**Misconception #3 address:** "Not marketing. The thinking step is architecturally motivated -- autoregressive generation benefits from composition planning in a way that diffusion does not."

#### 9. Check #2: Predict and Verify
- "Google says Nano Banana Pro uses 1,120 tokens for 2K output. Parti used 1,024 tokens for 256x256 (via ViT-VQGAN with 8x8 patches). What does the token count tell us about Nano Banana Pro's tokenizer?" (Answer: generating 2K resolution with only ~1,120 tokens means each token represents a much larger spatial region than Parti's -- likely a hierarchical or more efficient tokenizer, or the 1,120 is for a lower intermediate resolution that gets upsampled.)
- "If Nano Banana Pro edits images within the same model, how might that work architecturally?" (Answer: encode the input image to tokens, place those tokens in the context alongside the edit instruction, then generate new image tokens conditioned on both. Same autoregressive mechanism, different input context.)

#### 10. Elaborate: Speed, Efficiency, and the Hybrid Question

**How autoregressive can be fast:**
- KV caching (student knows from 4.3.3): each new token only needs one forward pass through the transformer
- Token efficiency: 1,120 tokens is far fewer forward passes than the millions of pixels in a 2K image
- Contrast with diffusion: 20-50 full forward passes through the U-Net/DiT, each processing ALL tokens

**The hybrid landscape (MENTIONED depth):**
- HART: autoregressive generates coarse tokens, diffusion refines to fine detail. 4.5-7.7x faster than pure autoregressive.
- BLIP3o-NEXT: autoregressive tokens condition a small diffusion model for final synthesis.
- These suggest a spectrum between pure autoregressive and pure diffusion, with hybrids capturing advantages of both.

**Misconception #5 address:** "Sequential does not mean slow. KV caching + efficient tokenization + fewer total steps can make autoregressive competitive with or faster than diffusion."

#### 11. Summary
- Two paradigms for image generation: diffusion (refine all pixels simultaneously from noise) vs autoregressive (generate tokens sequentially like text)
- Autoregressive requires discrete visual tokenization (VQ-VAE/ViT-VQGAN): images -> integer tokens -> same setup as text
- Text rendering works because autoregressive generation is inherently sequential and conditional -- each character token is generated with awareness of previous characters
- Nano Banana Pro likely: Gemini 3 Pro backbone generates image tokens autoregressively -> GemPix 2 decodes to pixels. Mandatory thinking step provides composition planning. Possibly hybrid with diffusion refinement.
- Architectural analysis is possible even without a paper: observable behavior + disclosed fragments + published precedents = informed, falsifiable hypotheses
- Echo the "convergence, not revolution" theme: autoregressive image generation combines transformers (Series 4) + visual tokenization (extending Series 6 VAE) + the scaling that makes it work

#### 12. References
- Google Parti paper (Yu et al., 2022)
- ViT-VQGAN paper (Yu et al., 2021)
- DALL-E paper (Ramesh et al., 2021)
- Chameleon paper (Team et al., Meta, 2024)
- HART paper (Tang et al., 2024)
- BLIP3o-NEXT (Xie et al., 2025)
- Google's Nano Banana Pro blog post / announcement (whatever is publicly available)

#### 13. Next Step
- Link back to the broader Special Topics series
- No specific next lesson required (standalone)

### Widget Decision

**No custom interactive widget.** This lesson is an analysis/reasoning lesson, not a concept-building lesson with interactive exploration. The primary modalities are:
- Mermaid diagrams (architecture comparisons, inferred pipeline)
- ComparisonRows (diffusion vs autoregressive, at multiple levels)
- GradientCards (evidence categories, precedent models)
- PhaseCards (inferred pipeline steps)
- CodeBlocks (pseudocode for both generation loops)

The lesson relies on the student's existing interactive experience with both autoregressive generation (Series 4 widgets and notebooks) and diffusion (Series 6-7 widgets and notebooks) to ground the comparison. Adding a widget here would not add pedagogical value -- the insight comes from connecting two systems the student has already explored interactively.

### Notebook Decision

**No Colab notebook.** This is an analysis lesson, not an implementation lesson. The student is not building or training anything. A notebook exercise would feel forced -- the "practice" in this lesson is the architectural reasoning itself, expressed through the predict-and-verify checks.

---

## Design Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (two gaps identified with resolution plans)
- [x] No multi-concept jumps in exercises
- [x] All gaps have explicit resolution plans (VQ-VAE and autoregressive image gen both have dedicated sections)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (text rendering mystery -> architecture detective work)
- [x] At least 3 modalities planned for the core concept (visual/diagrammatic, concrete example, verbal/analogy, symbolic/code, intuitive "of course" chain)
- [x] At least 2 positive examples + 1 negative example (text rendering comparison, partial generation comparison, codebook tokenization example, DALL-E trajectory as nuanced negative)
- [x] At least 3 misconceptions identified with negative examples (5 identified)
- [x] Cognitive load: 2 new concepts (within limit)
- [x] Every new concept connected to at least one existing concept (VQ-VAE -> VAE, autoregressive image gen -> autoregressive text gen)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-22 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings. The lesson is well-structured, epistemically honest, and builds effectively on the student's prior knowledge. The core paradigm comparison (diffusion vs autoregressive) lands well, the VQ-VAE gap resolution is handled smoothly, and the "of course" chain successfully makes autoregressive image generation feel inevitable rather than surprising. However, four improvement-level findings exist that would meaningfully strengthen the lesson.

### Findings

#### [IMPROVEMENT] — Misconception #4 ("can't know anything without a paper") never explicitly addressed

**Location:** Throughout the lesson (planned as "throughout the lesson; explicitly addressed in the Hook and revisited in the Summary")
**Issue:** The planning document identifies misconception #4: "We can't know anything useful about how Nano Banana Pro works because there's no paper." The plan says this should be "explicitly addressed in the Hook and revisited in the Summary." The Hook does set up the detective work framing, and the WarningBlock on epistemic honesty establishes the hypothesis/fact distinction. However, the misconception is never directly stated and refuted. The lesson models the right behavior (constructing hypotheses from evidence) but never names the misconception the student might hold. The Summary takeaway #5 comes closest ("Architectural analysis is possible without a paper...") but this is a conclusion, not a misconception correction.
**Student impact:** The student who enters thinking "without a paper this is just guessing" may not have that belief challenged directly. They will absorb the reasoning framework implicitly, which is good, but the pedagogical principle says misconceptions should be addressed head-on with negative examples.
**Suggested fix:** Add a brief explicit statement in the Hook section (after the behavioral comparison, around line 170) that directly names and refutes this misconception: something like "You might think that without a published paper, we are stuck guessing. But observable behavior, disclosed fragments, and published precedents are not guesses -- they are constraints that eliminate most possible architectures." This converts an implicit lesson-level demonstration into an explicit misconception correction.

#### [IMPROVEMENT] — Partial generation comparison describes "top half" but autoregressive raster scan is not explained

**Location:** Section "The partial generation test" (lines 537-578)
**Issue:** The partial generation comparison says stopping autoregressive at 50% tokens gives "half the image (e.g. top portion)" and the aside emphasizes this is "NOT diffusion with a different name." But the lesson never explains WHY stopping at 50% gives the top portion. The student has no reason to know that image tokens are generated in raster scan order (top-left to bottom-right). The planning doc mentions "raster scan order" but the built lesson drops this detail. The student would need to infer that image tokens follow some spatial ordering, but the lesson presents it as obvious.
**Student impact:** A student who understands autoregressive text generation (left-to-right) might wonder: "Why the top portion? What determines the order of image tokens?" This is a small but real gap -- the lesson asserts the result without explaining the mechanism that produces it. The student might form incorrect assumptions about how spatial order maps to token order.
**Suggested fix:** Add a parenthetical or brief sentence explaining raster scan order: "Image tokens are generated in raster scan order -- left-to-right, top-to-bottom, like reading a page. So stopping at 50% gives you the top half of the image, fully rendered, with the bottom half blank." One sentence closes the gap.

#### [IMPROVEMENT] — The "two paradigms" core analogy section lacks a genuine negative example

**Location:** Section "Two Paradigms for Image Generation" (lines 415-596)
**Issue:** The planning document lists the DALL-E 1 -> DALL-E 2/3 trajectory as a "negative (with nuance)" example that should prevent the misconception that autoregressive was always viable. The built lesson includes this in the "Published Precedents" section (lines 391-400) as a brief GradientCard, but it reads more like a historical fact than a negative example that defines a boundary. The lesson says "The paradigm that lost on quality at small scale wins at large scale" but does not develop this into a genuine negative example that shows the student when autoregressive DOES NOT work (i.e., at smaller scale, lower quality than diffusion). This means the core paradigm section has strong positive examples and a good contrastive comparison, but the boundary -- "when does autoregressive image generation NOT work well?" -- is underspecified.
**Student impact:** The student might leave thinking autoregressive is strictly better than diffusion, rather than understanding the scale-dependent tradeoff. The lesson mentions the tradeoff but does not develop it with enough specificity to serve as a boundary-defining negative example.
**Suggested fix:** In the DALL-E trajectory card or in the "Two Paradigms" section, add 1-2 sentences that make the negative case concrete: "At 2021-era scale (~12B parameters), autoregressive image quality was visibly worse than diffusion -- blurry, less coherent, weaker fine details. DALL-E 1's images looked noticeably worse than DALL-E 2's. The quality gap only closed at the scale of Gemini 3 Pro (~hundreds of billions of parameters)." This makes the boundary concrete.

#### [IMPROVEMENT] — Speed section's computation comparison is slightly misleading

**Location:** Section "Speed, Efficiency, and the Hybrid Question" (lines 889-933)
**Issue:** The "Token Efficiency" GradientCard says "1,120 tokens is far fewer forward passes than the millions of pixels in a 2K image. Compare with diffusion: 20-50 full forward passes through a DiT, each processing ALL tokens." This comparison is apples-to-oranges and could form a misconception. Diffusion does 20-50 full forward passes through a DiT, but each processes ~1,000-4,000 tokens (latent patches), not "millions of pixels." The lesson correctly taught latent diffusion in Series 6 -- the student knows diffusion operates in latent space, not pixel space. Comparing autoregressive token count to pixel count (instead of latent token count) makes the autoregressive approach seem more efficient than the fair comparison would suggest.
**Student impact:** The student might form an incorrect mental model of the speed comparison. They know from latent diffusion that diffusion operates on compressed latents, so this comparison might even confuse them: "Wait, I thought diffusion worked in latent space, not pixel space?"
**Suggested fix:** Fix the comparison to use latent tokens: "Compare with diffusion: 20-50 full forward passes through a DiT, each processing ~1,000-4,000 latent tokens. With autoregressive: 1,120 forward passes (one per token), but each only computes one new token's logits thanks to KV caching. The total computation is in the same ballpark -- the tradeoff is between parallel processing of all tokens per step (diffusion) vs sequential processing of one token per step with caching (autoregressive)."

#### [POLISH] — Check #1 Question 2 answer repeats the main text almost verbatim

**Location:** Check #1, Question 2 reveal (lines 632-639)
**Issue:** The reveal answer for "Why does autoregressive generation naturally handle text rendering while diffusion struggles?" is nearly identical to the explanation in the "Why Text Rendering Works" section that follows it. The check appears BEFORE the full explanation section, so it serves as a prediction exercise (good), but the reveal text pre-empts the detailed explanation.
**Student impact:** Minor -- the student who reads the reveal gets the same content twice. If they skip the reveal and read the later section, no issue. But for students who open reveals immediately, the later section feels redundant.
**Suggested fix:** Make the Check #1 reveal briefer and more prediction-oriented: "Think about it: in the autoregressive loop, what does the model 'see' when generating the visual tokens for the letter 'E'? The next section will develop this in detail." This preserves the predict-and-verify pattern without duplicating the full explanation.

#### [POLISH] — Missing connection to MMDiT joint attention ("one room, one conversation")

**Location:** Throughout the paradigm comparison sections
**Issue:** The planning document notes a potentially misleading prior analogy: "One room, one conversation" from MMDiT. It says: "In MMDiT, text and image tokens attend jointly but the generation process is still diffusion. In Nano Banana Pro, text and image tokens are in the same sequence but generation is sequential." The built lesson never makes this distinction. The student has the "one room, one conversation" mental model from SD3/Flux and might think Nano Banana Pro's approach is "the same as MMDiT but autoregressive." This is partially right but the distinction matters.
**Student impact:** The student might not cleanly separate "how tokens attend to each other" from "how the image is generated." They know joint attention from MMDiT and might conflate it with autoregressive generation. This is a subtle point and won't cause major confusion, hence POLISH rather than IMPROVEMENT.
**Suggested fix:** Add a brief ConceptBlock or aside in the paradigm comparison section: "In SD3/Flux, text and image tokens share attention ('one room, one conversation') but generation is still diffusion -- all tokens refined simultaneously. In Nano Banana Pro, text and image tokens also share the same sequence, but generation is autoregressive -- one token at a time. Same attention mechanism, different generation paradigm."

#### [POLISH] — The lesson header description says "excels at text rendering" but the lesson also covers compositional reasoning

**Location:** Line 52-53, LessonHeader description
**Issue:** The header description says "why autoregressive generation is a viable alternative to diffusion that excels at text rendering." The lesson also covers compositional reasoning (the thinking step, multi-image consistency), but the header only mentions text rendering. This is a minor accuracy issue.
**Student impact:** Negligible -- the student reads the full lesson regardless. But the header could be slightly more complete.
**Suggested fix:** Consider: "...that excels at text rendering and compositional reasoning" to match the lesson's actual scope.

### Review Notes

**What works well:**
- The epistemic honesty framing is excellent. The three-tier evidence categorization (disclosed/observable/inferred/open) is consistently applied throughout and gives the student a genuine reasoning framework, not just facts.
- The VQ-VAE gap resolution (Section 5) is one of the cleanest gap bridges in the course. It starts from the student's VAE knowledge, takes one precise step (continuous -> discrete via codebook), uses the text tokenization parallel, and the Mermaid diagram crystallizes it. The TipBlock ("You do NOT need to understand how VQ-VAE is trained") is well-placed scope management.
- The "of course" chain (lines 584-596) is pedagogically effective. It leverages the student's existing knowledge to make the new paradigm feel inevitable. This is the lesson's strongest moment.
- The code comparison (diffusion sampling loop vs autoregressive image generation) works powerfully because the student literally wrote the autoregressive loop in Building NanoGPT. The recognition moment is genuine.
- Connections to prior lessons are abundant and specific (Building NanoGPT, VAE, DiT, Chain of Thought, Scaling and Efficiency). The lesson reads as a synthesis of the student's entire course journey, which is appropriate for a Special Topics lesson.
- The hybrid landscape section demonstrates intellectual honesty by presenting evidence for both interpretations and concluding "we do not know." This models good reasoning.
- The lesson stays within its stated scope boundaries. No scope creep.

**Comparison to SigLIP 2 (reference lesson):**
- SigLIP 2 has more modalities for its core concept (6 modalities including SVG diagrams, side-by-side code, traced loss computation with real numbers, spatial metaphor). Nano Banana Pro has 5 planned modalities and delivers on all of them (visual/Mermaid, concrete VQ-VAE example, letter-vs-mural analogy, pseudocode, "of course" chain). Appropriate for an analysis lesson vs a concept-building lesson.
- SigLIP 2's check questions are more rigorous (4 targeted questions in Check #1, each testing a specific independence property). Nano Banana Pro's checks are good prediction exercises but somewhat less rigorous -- appropriate given this is a speculative analysis lesson, not a concept that can be tested with precise numerical tracing.
- SigLIP 2 addresses all 5 planned misconceptions explicitly with named negative examples. Nano Banana Pro addresses 3 of 5 explicitly (misconceptions #1, #2, #5), handles #3 (thinking step is marketing) well, but underdevelops #4 (can't know without a paper). See IMPROVEMENT finding above.

**Pattern observed:** The lesson is strongest in its first half (hook, VQ-VAE gap resolution, paradigm comparison, code side-by-side) and slightly weaker in its second half (architecture hypothesis, speed comparison). The second half has more assertions and fewer worked-through reasoning steps. This is partly inherent to the speculative nature of the analysis, but the speed comparison in particular could be tightened.

---

## Review — 2026-02-22 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 3

### Verdict: PASS

All four iteration 1 IMPROVEMENT findings have been addressed effectively, and all three iteration 1 POLISH findings have been addressed. The lesson is now in strong shape. No critical or improvement-level issues remain. Three minor polish findings exist, none of which affect the student's learning or comprehension.

**Iteration 1 fixes verified:**
- Misconception #4 ("can't know without a paper") is now explicitly addressed at lines 172-178 with a direct statement: "You might think that without a published paper, we are stuck guessing. We are not." This converts the implicit demonstration into an explicit misconception correction.
- Raster scan order is now explained in the partial generation ComparisonRow (line 573): "Raster scan order: left-to-right, top-to-bottom (like reading)." The gap is closed.
- The DALL-E trajectory GradientCard (lines 399-417) now includes concrete quality language ("visibly better quality," "sharper details, more coherent compositions," "DALL-E 1's images looked noticeably worse than DALL-E 2's") and the explicit boundary statement ("diffusion still excels at fine-grained texture details and high-resolution spatial coherence at smaller model scales").
- The speed section Token Efficiency card (lines 939-949) now correctly compares against latent tokens ("20-50 full forward passes through a DiT, each processing ~1,000-4,000 latent tokens") rather than pixels. The "similar ballpark" framing is honest and accurate.
- Check #1 Question 2 reveal (lines 659-665) is now a shorter, prediction-oriented pointer ("Think about it: when generating the visual tokens for the letter 'E,' what does the model 'see' in its context window?") that leads into the next section rather than duplicating it.
- The MMDiT distinction is now addressed in a ConceptBlock aside (lines 613-622): "In SD3/Flux, text and image tokens share attention ('one room, one conversation')—but generation is still diffusion... Same attention mechanism, different generation paradigm."
- The header description (line 52) now includes "and compositional reasoning."

### Findings

#### [POLISH] — "Irrevocable" tokens vs disclosed "self-correction" creates minor tension

**Location:** Thinking Step section (line 851) vs Disclosed by Google card (line 220)
**Issue:** The disclosed facts card lists "Multi-stage internal pipeline with self-correction" as something Google stated publicly. Later, the thinking step section says early tokens are "irrevocable—you cannot go back and change the sky after you have rendered the foreground." These are not contradictory (self-correction likely operates at a pipeline level—e.g., quality check and regeneration—rather than at the token level), but a student reading carefully might notice the apparent tension: "Wait, you said self-correction, but also irrevocable?"
**Student impact:** Minimal. Most students will read "irrevocable" as applying to within a single generation pass, and "self-correction" as a higher-level pipeline concept. But a careful student might pause.
**Suggested fix:** Add a brief parenthetical at line 851: "early tokens are irrevocable within a generation pass—you cannot go back and change the sky after you have rendered the foreground. (The disclosed 'self-correction' likely operates at a pipeline level: generate, evaluate, possibly regenerate—not mid-sequence editing.)"

#### [POLISH] — Check #2 Question 1 introduces Parti tokenization details the student has not seen

**Location:** Check #2, Question 1 (lines 873-876)
**Issue:** The question tells the student "Parti used 1,024 tokens for 256×256 (via ViT-VQGAN with 8×8 patches)." These specific numbers (1,024 tokens, 256×256 resolution, 8×8 patches) never appear earlier in the lesson. The Parti GradientCard (lines 383-389) mentions "20B parameter encoder-decoder transformer" but not the tokenization details. This means Check #2 Q1 is supplying new factual premises within the question itself, making it a "reason from these new numbers" exercise rather than a "recall + reason from what you learned" exercise.
**Student impact:** Minor. The question still works as an inference exercise. But it would feel more like a genuine "check your understanding" if the student had already encountered the Parti tokenization numbers—perhaps in the VQ-VAE section's concrete example or in the Parti GradientCard.
**Suggested fix:** Add "1,024 tokens for 256×256" to the Parti GradientCard (line 383-389) so the student has the baseline before encountering it in the check question. Alternatively, add a sentence in the VQ-VAE concrete example section noting: "Google's Parti (2022) used this exact setup: ViT-VQGAN tokenizing 256×256 images into 1,024 tokens via an 8×8 patch grid."

#### [POLISH] — HART reference title "without Vector Quantization" may briefly confuse

**Location:** References section, HART paper (lines 1100-1103)
**Issue:** The HART paper is listed with the title "Autoregressive Image Generation without Vector Quantization." The student just spent a section learning that autoregressive image generation requires discrete visual tokenization (VQ-VAE). Seeing a paper titled "without Vector Quantization" in the references could create a moment of confusion: "Wait, I thought you needed a codebook?" The lesson's main text mentions HART only as "autoregressive for global structure, diffusion refinement for local detail" (lines 994-995) without noting that HART's innovation is operating on continuous tokens rather than discrete ones.
**Student impact:** Very minor. The student scanning references might briefly wonder, but HART is only at MENTIONED depth and the reference note explains the hybrid approach. The confusion would be fleeting.
**Suggested fix:** Adjust the reference note to address this: "The HART paper. Hybrid autoregressive + diffusion refinement—notably, HART generates continuous tokens rather than discrete codebook tokens, using diffusion to fill in fine detail. An alternative to the VQ-VAE approach."

### Review Notes

**What works well (carrying forward from iteration 1 + improvements):**
- All iteration 1 findings were addressed cleanly. The fixes are natural and well-integrated—they do not feel like patches bolted onto an existing lesson.
- The misconception #4 fix (lines 172-178) is particularly well-written. It directly names the misconception, refutes it in one strong sentence, and uses the word "constraints"—which frames the rest of the lesson as constraint-satisfaction rather than speculation. This is a meaningful pedagogical improvement.
- The speed comparison fix (Token Efficiency card) is now intellectually honest. The "similar ballpark" framing and the "parallel vs sequential" tradeoff description give the student an accurate mental model rather than a misleading efficiency claim.
- The MMDiT distinction aside is well-placed next to the "of course" chain. The student encounters it at exactly the moment they might conflate joint attention with autoregressive generation.
- The lesson's epistemic honesty is consistent and well-calibrated throughout. The four-tier evidence categorization (disclosed/observable/inferred/open), the repeated "hypothesis, not fact" warnings, and the honest "we do not know" conclusion in the hybrid section all model good reasoning practices.
- The VQ-VAE gap resolution remains the cleanest section of the lesson. The bridge from continuous VAE to discrete codebook is handled in exactly the right number of steps.
- The "of course" chain remains the lesson's strongest pedagogical moment—it successfully converts the new paradigm from "surprising" to "inevitable" using only concepts the student already has.

**Overall assessment:** The lesson is ready to ship. The three remaining findings are genuine but minor—they represent the kind of polish that could be addressed but would not meaningfully change the student's learning experience if left as-is. No re-review needed after these fixes.
