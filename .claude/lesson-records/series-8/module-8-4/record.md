# Module 8.4: Image Generation Landscape -- Record

**Goal:** The student can navigate the open weight image generation ecosystem, identifying each major model's architecture, key innovations, and position in the historical evolution from SD v1 through the current frontier -- enabling informed model selection and architectural reasoning when encountering new releases.
**Status:** Complete (1 of 1 lesson built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Open weight image generation landscape taxonomy (5 backbone families: U-Net, pixel-space/cascaded, DiT cross-attention, MMDiT joint attention, S3-DiT single-stream) | INTRODUCED | open-weight-image-gen | The organizing framework for the entire lesson. Presented as a color-coded Mermaid tree diagram (amber=U-Net, rose=pixel-space, cyan=DiT, violet=MMDiT, emerald=S3-DiT). The student can now classify any model into one of five families by identifying its denoising backbone. Not DEVELOPED because no hands-on practice with classification beyond the three check exercises. |
| Three evolution lines framework (backbone, text encoding, training objective as independent axes defining a model's position) | INTRODUCED | open-weight-image-gen | Each axis gets its own Mermaid evolution diagram. InsightBlock: "Three Axes, One Model -- any model is a point in a 3D space." The student understands that backbone evolution (U-Net -> DiT -> MMDiT -> S3-DiT), text encoding evolution (CLIP -> CLIP+T5 -> single LLM), and training objective evolution (DDPM -> v-prediction -> flow matching -> distillation) progressed somewhat independently. |
| Five-question framework for placing new model announcements (backbone, text encoder, training objective, claimed innovation, lineage) | INTRODUCED | open-weight-image-gen | The "teach a person to fish" section. Presented as an ordered checklist with explanations of what each question reveals. Tested via Check #3 with a hypothetical model announcement. Intended as a lasting practical tool. |
| "Convergence, not revolution" as landscape-level mental model | INTRODUCED | open-weight-image-gen | Echo and expansion of prior usage. Applied at the full landscape level: the entire field evolved by combining known components (transformers, flow matching, better text encoders) in different configurations. The combinatorial space is small (4-5 backbones x 3-4 text encoding strategies x 3 training objectives). |
| SD v1.x (Stability AI / CompVis, Aug 2022: U-Net backbone, CLIP ViT-L, DDPM, 512x512, ~1.1B params) | MENTIONED | open-weight-image-gen | Identified in taxonomy, comparison table, and timeline. The student already has DEVELOPED understanding of SD v1.5 from Series 6. This lesson adds historical context (first open weight model) and ecosystem context (still has the largest LoRA/ControlNet ecosystem). |
| SD v2.x (Stability AI, Nov 2022: U-Net, OpenCLIP ViT-H, v-prediction, ~1.3B params) | MENTIONED | open-weight-image-gen | Presented as a cautionary tale: technically improved but broke ecosystem compatibility by switching text encoder. Used as negative example for "newer is always better" misconception. v-prediction introduced as a name (predicting velocity rather than noise) but not mathematically explained. |
| DeepFloyd IF (Stability AI / DeepFloyd, Apr 2023: pixel-space cascaded, T5-XXL, DDPM, ~4.3B params) | MENTIONED | open-weight-image-gen | Notable innovation card: first major open weight use of T5 for diffusion conditioning. Identified as the model that proved T5's superiority for text understanding, influencing SD3's text encoder choice. Pixel-space cascaded approach (64x64 base) identified as too expensive to scale. |
| Kandinsky 2.x (Sber AI, Apr 2023: U-Net unCLIP approach, CLIP image prior, ~3.3B params) | MENTIONED | open-weight-image-gen | Identified in taxonomy as U-Net family with unCLIP conditioning strategy (following DALL-E 2 approach). One-liner in comparison table. |
| PixArt-alpha (Huawei, Nov 2023: DiT cross-attention, T5-XXL, ~600M params) | MENTIONED | open-weight-image-gen | Notable innovation card: first major open weight DiT for text-to-image, competitive at ~10% of SD's training cost through decomposed training (learn images first, then text-image alignment). Part of the parallel DiT convergence narrative. |
| PixArt-sigma (Huawei, Apr 2024: DiT cross-attention, T5-XXL, up to 4K, ~600M params) | MENTIONED | open-weight-image-gen | Identified as improved PixArt-alpha with better VAE and 4K resolution support. One row in comparison table. |
| Stable Cascade (Stability AI, Feb 2024: cascaded latent, extreme 42:1 compression, ~3.6B params) | MENTIONED | open-weight-image-gen | Notable innovation card: extreme latent compression (42:1 vs SD's 8:1). Architecturally interesting but did not gain wide adoption due to cascaded inference complexity. |
| Playground v2.5 (Playground AI, Feb 2024: SDXL architecture, EDM training, ~3.5B params) | MENTIONED | open-weight-image-gen | Notable innovation card: proved training recipe matters as much as architecture. Same SDXL architecture with different training achieved reportedly better aesthetic quality. |
| Hunyuan-DiT (Tencent, May 2024: DiT cross-attention, bilingual CLIP + bilingual T5, ~1.5B params) | MENTIONED | open-weight-image-gen | Identified as first major bilingual (Chinese-English) open weight model. Part of the parallel DiT convergence narrative (three labs on three continents converging on DiT). |
| AuraFlow (Fal.ai, Jun 2024: MMDiT-style, CLIP + T5, ~6.8B params) | MENTIONED | open-weight-image-gen | Identified as community-driven MMDiT alternative with fully open training pipeline (code, data curation, training scripts). |
| Kolors (Kuaishou/KWAI, Jul 2024: SDXL-like U-Net, ChatGLM-6B LLM text encoder, ~3.5B params) | MENTIONED | open-weight-image-gen | Notable innovation card: LLM as text encoder before Z-Image. ChatGLM-6B (Chinese LLM) replacing CLIP. Foreshadowed the single-LLM text encoding trajectory that Z-Image completed. |
| SD3 / SD3.5 (Stability AI, Jun-Oct 2024: MMDiT, triple text encoders, flow matching, 2-8B params) | MENTIONED | open-weight-image-gen | Student already has INTRODUCED understanding from 7.4.3. This lesson adds: SD3.5 variants (Medium 2.5B, Large 8B, Large Turbo distilled), positioning in the MMDiT family alongside Flux and AuraFlow, comparison table entry. |
| Flux.1 Dev/Schnell (Black Forest Labs, Aug 2024: hybrid MMDiT, CLIP + T5-XXL, ~12B params) | MENTIONED | open-weight-image-gen | Student already has MENTIONED understanding from 7.4.3. This lesson adds: specific architectural details (19 double-stream + 38 single-stream blocks, rotary embeddings), Schnell's guidance distillation for 1-4 step generation, Apache 2.0 licensing. |
| Z-Image Base/Turbo (Freepik / Tongyi Lab, Jan 2025: S3-DiT, Qwen3-4B, flow matching, ~6.15B params) | MENTIONED | open-weight-image-gen | Student already has INTRODUCED understanding from 7.4.4. This lesson adds: positioning as the S3-DiT representative in the taxonomy, comparative context against Flux (~12B), Turbo variant with DMDR post-training. |
| DALL-E lineage (OpenAI: DALL-E 1 autoregressive 2021, DALL-E 2 unCLIP 2022, DALL-E 3 synthetic captions 2023) | MENTIONED | open-weight-image-gen | Brief historical context. DALL-E 1 proved concept, DALL-E 2's unCLIP approach influenced Kandinsky, DALL-E 3 integrated into ChatGPT. Closed models mentioned only to explain lineage of open weight choices. |
| Imagen (Google, 2022: pixel-space cascaded, T5-XXL, never released) | MENTIONED | open-weight-image-gen | Brief historical context. Pioneered T5 for diffusion text encoding. Influenced DeepFloyd IF and SD3. Never released weights. |
| Midjourney (architecture undisclosed, aesthetic quality benchmark) | MENTIONED | open-weight-image-gen | Brief mention. Aesthetic benchmark everyone chases. No architectural learning possible. |
| Flux 2 (BFL, 2025: API-only, not open weight) | MENTIONED | open-weight-image-gen | One-sentence mention. Flux lineage moved from open (Flux.1) to closed (Flux 2). Variants (Max, Pro, Flex, Klein) have not released weights. |
| Wan 2.1 (Alibaba/Tongyi, Feb 2025: video generation, 3D VAE + DiT) | MENTIONED | open-weight-image-gen | One-paragraph mention for completeness. Clearly distinguished as video generation, not image. Shares architectural DNA (DiT, flow matching) but adds temporal modeling. |
| Open weight vs open source distinction | MENTIONED | open-weight-image-gen | WarningBlock aside in closed models section. Open weight = weights downloadable but may restrict commercial use. Open source = full reproducibility (code, data, weights). Concrete example: Flux.1 Dev (non-commercial) vs Flux.1 Schnell (Apache 2.0). |
| v-prediction parameterization (predicting velocity rather than noise) | MENTIONED | open-weight-image-gen | Named in SD v2 context and training objective evolution line. Described as "predicting velocity rather than noise" and as a stepping stone toward flow matching. Not mathematically explained. Student already has vocabulary from flow matching lesson context. |
| EDM training framework (Karras et al.) | MENTIONED | open-weight-image-gen | Named in Playground v2.5 context. Mentioned as alternative training framework to standard DDPM. Not explained. |
| Guidance distillation (distilling CFG-guided model into one that generates without CFG at inference) | MENTIONED | open-weight-image-gen | Named in Flux Schnell context. Briefly described as "distilling the CFG-guided model into a model that generates high-quality images without needing CFG at inference." Student already has consistency distillation context from 7.3.2. |

## Per-Lesson Summaries

### Lesson 1: The Open Weight Image Generation Landscape (open-weight-image-gen)

**Type:** CONSOLIDATE | **Cognitive load:** Zero genuinely new concepts

**Concepts taught:**
1. Open weight image generation landscape taxonomy (INTRODUCED) -- five backbone families organizing all major models from 2022-2025 into a navigable classification system.
2. Three evolution lines framework (INTRODUCED) -- backbone, text encoding, and training objective as three independent axes defining any model's position in the landscape.
3. Five-question framework for new model announcements (INTRODUCED) -- backbone, text encoder, training objective, claimed innovation, and lineage as the reusable tool for placing future releases.

**Concepts reinforced/connected:**
- U-Net architecture (from Series 6) -- used as one of five backbone families in the taxonomy; the student's deep understanding makes the U-Net family instantly recognizable
- DiT / patchify (from Diffusion Transformers 7.4.2) -- used as the DiT cross-attention family anchor; "tokenize the image" mental model referenced
- MMDiT joint attention (from SD3 & Flux 7.4.3) -- used as the MMDiT family anchor; "one room, one conversation" mental model referenced
- S3-DiT single-stream (from Z-Image 7.4.4) -- used as the S3-DiT family anchor; "translate once, then speak together" referenced
- SDXL as "the U-Net's last stand" (from 7.4.1) -- used as the U-Net ceiling, motivating the transformer pivot
- Text encoder evolution (from 6.3.3, 7.4.1, 7.4.3, 7.4.4) -- made explicit as historical trend across the full model landscape (CLIP -> CLIP+OpenCLIP -> CLIP+OpenCLIP+T5 -> CLIP+T5 -> single LLM)
- Flow matching (from 7.2.2, 7.4.3) -- positioned in training objective evolution as the current standard
- "Two knobs not twenty" transformer scaling recipe (from 7.4.2) -- used to explain why multiple labs independently converged on DiT

**Mental models established:**
- The landscape taxonomy itself (5 backbone families as the primary organizing principle)
- "Three axes, one model" (backbone x text encoder x training objective = a model's position)
- Five-question framework as a reusable tool for future model announcements
- "Convergence, not revolution" expanded to full landscape level (the combinatorial space is small)

**Analogies used:**
- "Family tree" for model relationships (parents, siblings, cousins -- models share DNA when the same researchers created them)
- "Seeing the forest" vs individual trees (the student has studied trees in Series 6-7; this lesson shows the forest)
- Color-coded backbone families as visual language (amber=U-Net, rose=pixel-space, cyan=DiT, violet=MMDiT, emerald=S3-DiT) carried consistently from taxonomy diagram to comparison table

**How concepts were taught:**
- Landscape taxonomy: 3 modalities -- visual (color-coded Mermaid tree diagram grouping all models by backbone), verbal (family tree analogy with DNA/lineage language), concrete (per-family GradientCards with one-liner per model). The taxonomy diagram is the lesson's primary visual artifact.
- Three evolution lines: 3 modalities each -- visual (three separate Mermaid flow diagrams showing each axis's progression), verbal (connecting narrative prose linking each step to its motivation), symbolic (specific model names, dates, and param counts anchoring each step)
- Five-question framework: 2 modalities -- verbal (numbered checklist with explanations), concrete (hypothetical model announcement in Check #3 where the student applies all five questions)
- Comparison table: The centerpiece reference artifact. 16 rows, 9 columns, chronological order, left-border color coding matching backbone family. Designed as a reference card to return to.
- Historical timeline: Mermaid timeline diagram showing 2022-2025 releases with labs and key inflection points marked.

**Misconceptions addressed:**
1. "Newer models are always better" -- SD v2 cautionary tale (broke ecosystem compatibility), SD v1.5 ecosystem discussion (more LoRAs/ControlNets than SD3), WarningBlock "Newer Is Not Always Better," Check #2 Q3 asking why SD v2 failed to replace v1.5.
2. "All these models are fundamentally different architectures" -- hook reveals "only 4-5 backbone types and 3-4 text encoding strategies," taxonomy grouping shows models clustering into few families, InsightBlock "Few Distinct Architectures."
3. "Open weight means open source" -- WarningBlock aside in closed models section with Flux.1 Dev vs Schnell concrete example.
4. "The U-Net to DiT shift was a single event" -- parallel DiT convergence section showing PixArt-alpha (Nov 2023), SD3 (Feb 2024), and Hunyuan-DiT (May 2024) independently adopting DiT. WarningBlock "Not a Single Pivot."
5. "Flux is just SD3 with different branding" -- MMDiT family GradientCard lists specific differences: hybrid double/single-stream blocks, rotary embeddings, guidance distillation in Schnell, dropped second CLIP encoder.

**What is NOT covered (relevant for future lessons):**
- Detailed architectural deep dives into PixArt, Hunyuan-DiT, Kolors, DeepFloyd IF, Kandinsky, Stable Cascade, AuraFlow (all MENTIONED only)
- Model quality benchmarks or rankings (deliberately excluded as too subjective and rapidly outdated)
- Licensing details beyond the brief open-weight vs open-source distinction
- Video generation models beyond the one-paragraph Wan 2.1 mention
- Practical deployment, inference optimization, or model selection criteria
- Training datasets, data curation strategies, or compute requirements
- Fine-tuning any of these models (the student has LoRA skills from 6.5 but this lesson does not apply them)

**Review history:** PASS after 2 iterations. Iteration 1: 0 critical, 4 improvement, 4 polish. All improvements addressed: (1) open weight vs open source WarningBlock added in closed models section, (2) transition prose added between Notable Innovation GradientCards for narrative flow, (3) left-border color coding added to comparison table matching taxonomy colors, (4) predict-before-reveal preamble added to Check #1. Iteration 2: 0 critical, 0 improvement, 3 polish (Check #2 missing predict preamble, family tree analogy phrasing, Stable Cascade date inconsistency between timeline and table).

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| Five backbone families taxonomy (U-Net, pixel-space/cascaded, DiT, MMDiT, S3-DiT) | open-weight-image-gen | |
| "Three axes, one model" (backbone x text encoder x training objective) | open-weight-image-gen | |
| Five-question framework for new model announcements | open-weight-image-gen | |
| "Convergence, not revolution" (landscape-level: small combinatorial space of known components) | echoed from prior lessons | open-weight-image-gen (expanded to full landscape) |
| Color-coded backbone families (amber/rose/cyan/violet/emerald) | open-weight-image-gen | |
| "Family tree" for model relationships (parents, siblings, cousins sharing DNA) | open-weight-image-gen | |
| "Seeing the forest" (stepping back from individual trees studied in Series 6-7) | open-weight-image-gen | |
