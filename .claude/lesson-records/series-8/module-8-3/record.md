# Module 8.3: Architecture Analysis -- Record

**Goal:** The student can analyze production AI systems by reasoning from observable behavior, disclosed architecture fragments, and published precedents -- constructing informed, falsifiable hypotheses even without a comprehensive technical paper.
**Status:** Active (1 of ? lessons built -- open-ended module)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Discrete visual tokenization (VQ-VAE/ViT-VQGAN codebook: encoder output mapped to nearest codebook vector, producing a discrete integer index) | INTRODUCED | nano-banana-pro | Taught as extension of the student's VAE knowledge: "VAE encodes to a continuous vector. VQ-VAE takes one more step: maps to the nearest codebook entry, producing a discrete integer." Codebook analogy parallels text tokenizer vocabulary. Concrete example: 256x256 image -> 32x32 = 1,024 integer tokens. Side-by-side ComparisonRow: text tokenization vs image tokenization. Mermaid diagram of the full pipeline. TipBlock explicitly scopes out VQ-VAE training details (commitment loss, codebook collapse). Connected to DiT patchify ("same intuition -- images as token sequences -- but discrete vs continuous representation"). |
| Autoregressive image generation (generating discrete image tokens one at a time using the same predict-next-token loop as GPT) | INTRODUCED | nano-banana-pro | Core concept. Taught by showing the generation loop is structurally identical to GPT generate() from Building NanoGPT -- "the only difference is the vocabulary." Side-by-side CodeBlocks: diffusion sampling loop vs autoregressive image generation loop. The "of course" chain makes it feel inevitable: (1) images can be tokenized to integers, (2) you have a transformer that generates tokens, (3) of course you can generate images the same way. Partial generation comparison makes the paradigm difference concrete: stop at 50% -> autoregressive gives sharp top half (raster scan order), diffusion gives blurry full image. |
| Autoregressive vs diffusion as image generation paradigms (diffusion refines all pixels simultaneously from noise; autoregressive generates tokens sequentially) | INTRODUCED | nano-banana-pro | Taught via "writing a letter vs painting a mural" analogy. ComparisonRow contrasting the two paradigms across 5 dimensions. Partial generation test as concrete differentiator. Explicit misconception correction: "This is NOT diffusion with a different name." Code side-by-side reinforces structural difference. |
| Reasoning-guided synthesis / thinking step for composition planning (mandatory planning before autoregressive generation because early tokens are irrevocable) | INTRODUCED | nano-banana-pro | Connected to chain-of-thought from Series 5. Explained as architecturally motivated: autoregressive commits to early tokens permanently, so upfront planning is more valuable than for diffusion (which refines everything simultaneously). Concrete example: decomposing "a red car in front of a blue house with 'SOLD' on a sign" into objects, properties, spatial layout, generation order. |
| Architectural analysis from evidence (constructing falsifiable hypotheses from disclosed facts, observable behavior, and published precedents) | INTRODUCED | nano-banana-pro | Meta-skill taught implicitly through the lesson structure and explicitly named in hook and summary. Four-tier evidence categorization: disclosed, observable, inferred, open. Misconception #4 addressed directly: "You might think that without a published paper, we are stuck guessing. We are not." The full lesson models the reasoning process. |
| Google Parti as autoregressive image generation precedent (ViT-VQGAN + 20B encoder-decoder transformer, 2022) | MENTIONED | nano-banana-pro | Presented in "Published Precedents" section as the direct ancestor of Nano Banana Pro's approach. Used ViT-VQGAN for visual tokenization producing 1,024 tokens for 256x256 images. |
| Chameleon unified autoregressive model for text + images (Meta, 2024) | MENTIONED | nano-banana-pro | Early-fusion multimodal -- text and image tokens interleaved in one sequence. Proved concept of single model generating both modalities. |
| DALL-E 1 autoregressive approach and the paradigm shift to diffusion | MENTIONED | nano-banana-pro | DALL-E 1 (2021) was autoregressive but lost to diffusion on quality at 2021 scale. Autoregressive returns at 2025 scale. Used as nuanced negative example: autoregressive is not always viable -- scale changes which approach dominates. Concrete quality language: "visibly better quality... sharper details, more coherent compositions." |
| Hybrid autoregressive + diffusion approaches (HART, BLIP3o-NEXT) | MENTIONED | nano-banana-pro | HART: autoregressive for global structure, diffusion refinement for local detail, using continuous tokens rather than discrete VQ. BLIP3o-NEXT: autoregressive tokens conditioning a diffusion model. Presented as evidence for a spectrum between pure autoregressive and pure diffusion. |
| GemPix 2 as visual tokenizer + decoder component | MENTIONED | nano-banana-pro | Google's name for the rendering engine. Likely an evolved ViT-VQGAN. Token efficiency (~1,120 tokens for 2K output) suggests highly optimized tokenizer. |

## Per-Lesson Summaries

### Lesson 1: Nano Banana Pro (nano-banana-pro)

**Type:** STRETCH | **Cognitive load:** 2 new concepts + analytical reasoning framework

**Concepts taught:**
1. Discrete visual tokenization (INTRODUCED) -- VQ-VAE/ViT-VQGAN codebook approach, bridging from the student's continuous VAE knowledge to discrete integer tokens.
2. Autoregressive image generation (INTRODUCED) -- generating image tokens sequentially using the same GPT generate() loop, applied to a visual codebook vocabulary.

**Concepts reinforced/connected:**
- Autoregressive text generation (from Building NanoGPT) -- the same loop, different vocabulary
- VAE encoding/decoding (from Module 6.1) -- extended to discrete codebook quantization
- KV caching (from Scaling and Efficiency) -- explains how sequential generation can be fast
- Chain-of-thought (from Series 5) -- thinking step as composition planning analogous to CoT
- DiT patchify (from Diffusion Transformers) -- "tokenize the image" in continuous vs discrete form
- MMDiT joint attention (from SD3/Flux) -- distinguished: same attention mechanism, different generation paradigm
- Cross-attention conditioning (from text conditioning) -- contrasted with autoregressive's implicit conditioning through shared sequence
- "Convolutions vs attention at scale" (from DiT) -- same pattern: autoregressive lost to diffusion at small scale, wins at large scale

**Mental models established:**
- "Writing a letter vs painting a mural" -- autoregressive is sequential (each word follows the last, quoting text is natural); diffusion is simultaneous (rough in the whole scene, refine everywhere at once, text on a mural requires painting all letters simultaneously).
- "Convergence, not revolution" (echo) -- autoregressive image generation combines transformers (Series 4) + visual tokenization (extending Series 6 VAE) + scale. Not alien technology -- familiar components in a different configuration.
- Evidence-based architectural analysis: disclosed / observable / inferred / open as four tiers of certainty.

**Analogies used:**
- "Writing a letter vs painting a mural" (core analogy for autoregressive vs diffusion paradigm split)
- Text tokenizer vocabulary parallel for visual codebook (both map inputs to discrete integer IDs from a vocabulary)
- "Developing a photograph vs printing a page" (partial generation test: diffusion = everything emerges simultaneously, autoregressive = each line is final as laid down)
- Chain-of-thought parallel for the mandatory thinking step
- Extended "tokenize the image" from DiT (continuous patches vs discrete codebook entries)
- Extended "one room, one conversation" from MMDiT (same attention, different generation paradigm)

**"Of Course" moments:**
1. "Of course you can generate images the same way you generate text" -- once images are discrete tokens, the GPT loop just works
2. "Of course text rendering works" -- each character is another token in the sequence, generated with full awareness of what came before
3. "Of course the thinking step matters" -- autoregressive commits to early tokens permanently, so upfront planning is architecturally motivated

**How concepts were taught:**
- Discrete visual tokenization: 4 modalities -- verbal/analogy (text tokenizer vocabulary parallel), visual (Mermaid pipeline diagram: image -> encoder -> continuous vectors -> codebook lookup -> integer tokens), concrete example (256x256 -> 32x32 = 1,024 tokens with specific integer IDs), symbolic (ComparisonRow: text tokenization vs image tokenization with example token sequences). Connected to VAE (continuous -> discrete extension) and DiT patchify (same intuition, different representation).
- Autoregressive image generation: 5 modalities -- verbal/analogy (writing a letter vs painting a mural), visual (ComparisonRow across 5 dimensions), symbolic/code (side-by-side CodeBlocks: diffusion sampling loop vs autoregressive image generation loop), concrete example (partial generation comparison: stop at 50%), intuitive ("of course" chain making the paradigm feel inevitable).
- Architectural hypothesis: 4-step PhaseCard pipeline (prompt -> thinking -> token generation -> GemPix 2 decoding), Mermaid end-to-end diagram, evidence-weighted reasoning (GradientCards for disclosed facts, observable behavior, inferred architecture, open questions).

**Misconceptions addressed:**
1. "Autoregressive image generation is just diffusion with a different name" -- addressed with ComparisonRow, partial generation comparison, and explicit WarningBlock: fundamentally different generation mechanics.
2. "Text rendering must require a specialized text-detection module" -- addressed with structural argument: autoregressive generation is inherently sequential and conditional, each character generated with awareness of previous characters. InsightBlock: "No Special Module."
3. "The mandatory thinking step is just marketing" -- addressed by connecting to chain-of-thought reasoning and explaining the architectural motivation: early tokens are irrevocable in autoregressive generation.
4. "We can't know anything useful without a paper" -- addressed explicitly in hook: "You might think that without a published paper, we are stuck guessing. We are not. Observable behavior, disclosed fragments, and published precedents are constraints that eliminate most possible architectures."
5. "Autoregressive must be slower than diffusion because sequential" -- addressed with KV caching explanation, token efficiency comparison against latent tokens (not pixels), and honest "similar ballpark" framing.

**What is NOT covered (relevant for future lessons):**
- Implementing VQ-VAE or any image tokenizer from scratch
- Training an autoregressive image generator
- VQ-VAE training details (commitment loss, codebook collapse, straight-through estimator)
- Comprehensive survey of autoregressive image generation methods
- Continuous-token autoregressive approaches (HART uses continuous tokens, only MENTIONED)
- Masked image modeling (MAE-style, non-autoregressive alternatives)
- Google's training infrastructure or compute costs
- The full Gemini architecture beyond what is relevant to image generation

**Review history:** PASS after 2 iterations. Iteration 1: 0 critical, 4 improvement, 3 polish. All improvements addressed: (1) misconception #4 explicitly stated and refuted, (2) raster scan order explained in partial generation comparison, (3) DALL-E trajectory negative example made concrete with quality language, (4) speed comparison corrected to use latent tokens instead of pixels. All polish addressed: Check #1 Q2 reveal shortened to prediction-oriented pointer, MMDiT distinction added as ConceptBlock aside, header description expanded to include compositional reasoning. Iteration 2: 0 critical, 0 improvement, 3 polish (minor tensions: irrevocable vs self-correction, Check #2 premises, HART reference title).

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "Writing a letter vs painting a mural" (autoregressive = sequential with each token conditioned on previous; diffusion = simultaneous refinement everywhere) | nano-banana-pro | |
| "Convergence, not revolution" (echo from prior lessons: frontier systems combine known components in new configurations) | multiple prior | nano-banana-pro (echoed for autoregressive image gen = transformers + visual tokenization + scale) |
| Disclosed / observable / inferred / open evidence tiers (four-tier epistemic framework for analyzing systems without full papers) | nano-banana-pro | |
| "Developing a photograph vs printing a page" (partial generation comparison: diffusion = everything emerges simultaneously, autoregressive = line by line) | nano-banana-pro | |
| Text tokenizer vocabulary parallel for visual codebook (both produce discrete integer sequences from a vocabulary) | nano-banana-pro | |
