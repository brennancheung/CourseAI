# Lesson: Textual Inversion

**Module:** 6.5 Customization
**Position:** Lesson 3 of 3 (final lesson in Series 6)
**Slug:** `textual-inversion`

---

## Phase 1: Orient -- Student State

The student arrives at the final lesson of the Stable Diffusion series having completed two BUILD-level customization lessons. They understand LoRA fine-tuning (modifying U-Net cross-attention weights) and img2img/inpainting (modifying the inference process). The "customization spectrum" was introduced in Lesson 2 via three GradientCards that foreshadowed this lesson: "LoRA changes weights, img2img/inpainting changes inference, textual inversion changes embeddings." The student knows something surprising is coming--the Lesson 2 next-step preview told them "you could optimize a single embedding vector in CLIP's space to represent something the model has never seen."

This is a STRETCH lesson. The optimization target (a single embedding vector rather than model weights or inference configuration) is conceptually novel, even though every component it uses is familiar. The surprise is not mechanical complexity but conceptual reframing: instead of changing what the model does, you change what it hears.

**Relevant concepts the student has:**

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Token embeddings as learned lookup table (nn.Embedding maps integer ID to dense vector; one-hot @ matrix = row selection) | DEVELOPED | embeddings-and-position (4.1.3) | Student built embeddings, verified embedding.weight[i] == embedding(tensor([i])), explored the EmbeddingSpaceExplorer widget showing ~120 tokens in semantic clusters. Knows embeddings are learned parameters (requires_grad=True), not preprocessing. |
| Embedding space clustering (similar tokens have nearby vectors after training) | DEVELOPED | embeddings-and-position (4.1.3) | Interactive EmbeddingSpaceExplorer widget with ~120 tokens in semantic clusters. Before/after training comparison: random at init, meaningful clusters after. "Geometry encodes meaning." |
| BPE tokenization (subword tokens, merge table, vocabulary) | APPLIED | tokenization (4.1.2) | Student implemented BPE from scratch. Knows tokens are subword pieces, not words. Understands vocabulary as a fixed set of token IDs. |
| CLIP shared embedding space (text and images in the same geometric space, continuous and meaningful) | DEVELOPED | clip (6.3.3) | Before/after SVG diagram. ComparisonRow vs VAE latent space. Negative example: independently trained encoders produce unaligned spaces. Core mental model: "Two encoders, one shared space -- the loss function creates the alignment, not the architecture." |
| CLIP text encoder produces a sequence of 77 contextual token embeddings ([77, 768]) | INTRODUCED | stable-diffusion-architecture (6.4.1) | Student knows the pipeline: tokenizer splits text -> adds SOT/EOT -> pads to 77 -> CLIP transformer outputs [77, 768] contextual embeddings. WarningBlock: "The U-Net will never see the text." Knows these are contextual embeddings (each token's representation includes context from all others via self-attention). |
| CLIP tokenizer padding to 77 tokens (fixed context length with SOT/EOT special tokens) | INTRODUCED | stable-diffusion-architecture (6.4.1) | Shape invariance: short and long prompts both produce [1, 77, 768]. The fixed shape standardizes the CLIP-to-U-Net interface. |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings; per-spatial-location attention) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | One-line change from self-attention. Concrete attention weight table for "a cat sitting in a sunset." Three-lens callback. Student knows cross-attention is WHERE text meaning enters the U-Net. |
| LoRA placement in diffusion U-Net (cross-attention projections W_Q, W_K, W_V, W_out) | DEVELOPED | lora-finetuning (6.5.1) | Student knows LoRA modifies U-Net weights. "Same detour, different highway" and "of course cross-attention" mental models. This is the primary contrast: LoRA changes what the model DOES with embeddings; textual inversion changes what the model RECEIVES. |
| Diffusion LoRA training loop (DDPM training with frozen base, LoRA params only unfrozen) | DEVELOPED | lora-finetuning (6.5.1) | Student has seen the "freeze everything except X, backprop through X only" pattern applied to LoRA params. This SAME pattern applies to textual inversion, but with X = a single embedding vector. |
| Style vs subject LoRA (rare-token identifier like "sks") | INTRODUCED | lora-finetuning (6.5.1) | Student has seen the idea of associating a rare token with a specific concept. The "sks" identifier in subject LoRA is conceptually related to the pseudo-token in textual inversion, but LoRA trained the weights while keeping the token embedding fixed. Textual inversion does the opposite. |
| DDPM training objective (sample random timestep, noise the image, predict the noise, MSE loss) | DEVELOPED | learning-to-denoise (6.2.3) | Full training loop understood. Used in LoRA training (6.5.1) and will be used again here. |
| Forward process closed-form (x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon) | DEVELOPED | the-forward-process (6.2.2) | Used in training (6.2.3), capstone (6.2.5), LoRA training (6.5.1), img2img (6.5.2), and inpainting (6.5.2). This formula appears for the fifth time in this lesson. |
| Customization spectrum (LoRA = weights, img2img/inpainting = inference, textual inversion = embeddings) | INTRODUCED | img2img-and-inpainting (6.5.2) | Three GradientCards foreshadowing this lesson. "Three knobs on three different parts of the pipeline." Student has the first two knobs at DEVELOPED depth; the third is the target of this lesson. |
| Full SD pipeline data flow (text -> CLIP -> U-Net denoising loop with CFG -> VAE decode) | DEVELOPED | stable-diffusion-architecture (6.4.1) | Complete pipeline traced with tensor shapes at every handoff. Student can locate exactly where embedding modification takes effect. |
| Classifier-free guidance (CFG) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | Two forward passes per step (conditional + unconditional), amplifying text direction. Relevant because textual inversion training often uses CFG-aware loss or operates within the CFG framework. |
| Embeddings are learned parameters (requires_grad=True, updated by backprop) | DEVELOPED | embeddings-and-position (4.1.3) | Student explicitly learned that embeddings are the first layer of the model, not preprocessing. WarningBlock: "Embeddings are the first layer of the model, not a preprocessing step." This is the conceptual foundation for textual inversion: if embeddings are parameters, they can be optimized individually. |

**Mental models and analogies already established:**
- "Two encoders, one shared space -- the loss function creates the alignment" (CLIP)
- "Geometry encodes meaning" (embedding spaces)
- "Same detour, different highway" (LoRA bypass in diffusion)
- "Of course cross-attention" (why LoRA targets cross-attention projections)
- "Three knobs on three different parts of the pipeline" (customization spectrum)
- "The U-Net never sees text" (CLIP is the only translator)
- "Finetuning is a refinement, not a revolution" (why few images suffice)
- "Same formula, different source for K and V" (cross-attention)
- "The API is a dashboard. You built the machine behind it." (parameter comprehension)
- "Embeddings are the first layer of the model, not a preprocessing step" (learned parameters)

**What was explicitly NOT covered that is relevant:**
- Textual inversion (explicitly deferred from both Lessons 1 and 2 of this module)
- Optimizing a single embedding vector while keeping all model weights frozen
- Adding new tokens to a tokenizer's vocabulary (extending the embedding table)
- The distinction between the initial token embedding lookup and the CLIP text encoder's contextual processing
- How gradient flows through the CLIP text encoder back to a single embedding
- Comparing textual inversion's expressiveness to LoRA's (one vector vs thousands of parameters)

**Readiness assessment:** Strong. The student has every prerequisite at sufficient depth. They understand token embeddings as learned parameters (DEVELOPED from 4.1.3), CLIP's shared embedding space as continuous and meaningful (DEVELOPED from 6.3.3), the DDPM training loop (DEVELOPED from 6.2.3, used again in 6.5.1), and cross-attention as the bridge from text to image (DEVELOPED from 6.3.4). The key conceptual leap--that you can optimize a single embedding vector while freezing everything else--is surprising but mechanically simple given what the student already knows. The "freeze everything except X" pattern from LoRA training transfers directly; the only change is what X is.

---

## Phase 2: Analyze

### Target Concept

**This lesson teaches the student to understand how textual inversion creates a new pseudo-token (e.g., `<my-cat>`) and optimizes its embedding vector in CLIP's embedding space to represent a novel concept, while keeping the entire model (U-Net, VAE, CLIP text encoder) frozen.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Token embeddings as learned lookup table (nn.Embedding) | DEVELOPED | DEVELOPED | embeddings-and-position (4.1.3) | OK | Student must understand that each token ID maps to a row in an embedding matrix, and that this row is a trainable parameter. Textual inversion adds a new row and optimizes it. DEVELOPED achieved via building and exploring embeddings. |
| Embedding space clustering / "geometry encodes meaning" | DEVELOPED | DEVELOPED | embeddings-and-position (4.1.3) | OK | Student must understand that embedding positions are meaningful: similar tokens cluster nearby. Textual inversion places a new vector in this space so that it encodes a novel concept. The student's interactive experience with EmbeddingSpaceExplorer provides strong grounding. |
| CLIP shared embedding space (continuous, meaningful) | DEVELOPED | DEVELOPED | clip (6.3.3) | OK | Student must understand that CLIP's text embedding space is continuous and geometrically meaningful. A well-placed vector in this space can represent a concept the model has never seen as text. Student has before/after diagrams and ComparisonRow vs VAE latent space. |
| CLIP text encoder output ([77, 768] contextual embeddings) | INTRODUCED | INTRODUCED | stable-diffusion-architecture (6.4.1) | OK | Student needs to understand the two-stage pipeline: (1) token IDs -> initial embeddings via lookup, (2) initial embeddings -> contextual embeddings via CLIP transformer. Textual inversion operates at stage (1). INTRODUCED is sufficient because this lesson will explain the distinction between stages explicitly. |
| BPE tokenization and vocabulary | DEVELOPED | APPLIED | tokenization (4.1.2) | OK | Student needs to understand that a tokenizer has a fixed vocabulary and that token IDs index into it. Adding a pseudo-token means extending this vocabulary. APPLIED exceeds DEVELOPED. |
| Cross-attention (Q from spatial, K/V from text embeddings) | DEVELOPED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Student must understand that the text embedding sequence feeds into cross-attention as K and V. The optimized pseudo-token's embedding enters the U-Net through this exact pathway. |
| DDPM training loop (sample timestep, noise image, predict noise, MSE loss) | DEVELOPED | DEVELOPED | learning-to-denoise (6.2.3) | OK | The textual inversion training loop IS the DDPM training loop with everything frozen except one embedding vector. Student has seen this loop three times (6.2.3, 6.2.5 capstone, 6.5.1 LoRA training). |
| "Freeze everything except X" pattern (from LoRA training) | DEVELOPED | DEVELOPED | lora-finetuning (6.5.1) | OK | The LoRA lesson froze the base U-Net and optimized only LoRA parameters. Textual inversion uses the same pattern but X = a single embedding vector instead of LoRA matrices. Direct transfer. |
| LoRA as weight modification (the contrast) | DEVELOPED | DEVELOPED | lora-finetuning (6.5.1) | OK | The contrast between LoRA (modifying weights) and textual inversion (modifying embeddings) is the core conceptual payoff. Student needs LoRA firmly understood to feel the surprise. |
| Forward process closed-form | DEVELOPED | DEVELOPED | the-forward-process (6.2.2) | OK | Used in the training loop. Fifth appearance. No gap. |
| Customization spectrum (foreshadowed) | INTRODUCED | INTRODUCED | img2img-and-inpainting (6.5.2) | OK | Student has the three-knob framing. This lesson fills in the third knob at DEVELOPED depth. |

**All prerequisites are OK. No gaps to resolve.**

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Textual inversion modifies the model's weights (U-Net, CLIP encoder)" | The student has just completed two lessons where customization involved weight modification (LoRA) or at least model inference changes (img2img). "Training" implies "weight updates." The student may assume gradient descent = weight changes somewhere. | During textual inversion training, run the same prompt WITHOUT the pseudo-token through the model before and after training. The output is identical. No weights changed. The model behaves exactly as before for any prompt that does not contain the new token. Only prompts containing `<my-cat>` produce different results, because only they route through the new embedding. | Section 5 (Explain)--immediately after presenting the mechanism, explicitly state "zero weights change" and contrast with LoRA. Reinforce in the notebook with a verification exercise. |
| "The pseudo-token's embedding is processed in isolation by CLIP (like a bag of separate embeddings)" | The student knows token embeddings are lookup table entries (one vector per token). They may think the pseudo-token's vector goes directly to cross-attention without interacting with other tokens. This ignores the CLIP text encoder's transformer, which applies self-attention across all tokens. | The prompt "a photo of `<my-cat>` sitting on grass" produces a different contextual embedding for `<my-cat>` than "a sketch of `<my-cat>` in watercolor." The same initial embedding is transformed differently by the CLIP transformer depending on the surrounding words. The pseudo-token interacts with every other token through self-attention. Isolation would mean context does not matter--but context demonstrably changes the output. | Section 5 (Explain)--when tracing the data flow through the two-stage pipeline (lookup -> CLIP transformer -> cross-attention). Critical distinction: textual inversion optimizes the initial embedding (stage 1), but the CLIP transformer contextualizes it (stage 2). |
| "Textual inversion is as expressive as LoRA -- it can capture any style or concept equally well" | The student may generalize from the impressive demo. "If one vector can represent my cat, one vector can represent anything." This overestimates what a single 768-dimensional vector can encode. | Complex styles requiring spatial understanding (e.g., "always place the subject in the lower-left third with backlighting") cannot be captured in a single embedding vector because cross-attention maps text to spatial features at the token level, not at the composition level. LoRA modifies thousands of projection parameters and can encode spatial style patterns. Textual inversion encodes a concept descriptor, not a spatial recipe. Concrete comparison: textual inversion for "my cat" works well (concept = a specific visual object); textual inversion for "Studio Ghibli animation style" works poorly because style involves spatial composition, color grading, and rendering choices that span multiple attention layers. | Section 8 (Elaborate)--dedicated comparison section contrasting textual inversion with LoRA on expressiveness, training speed, file size, and use cases. |
| "You need to train a completely new CLIP model or fine-tune the CLIP text encoder" | The student has seen LoRA fine-tune the U-Net. They may assume textual inversion fine-tunes the CLIP encoder. "Training on text embeddings" could easily be interpreted as "training the text encoder." | The CLIP text encoder weights are completely frozen. Gradients flow THROUGH the CLIP encoder during backpropagation (to reach the embedding vector), but the encoder's weights have zero gradients because they are not in the optimizer's parameter group. Only the single embedding vector is in the optimizer. This is verifiable: check the parameter count being optimized (768 floats for a single token, vs CLIP's ~123M parameters). | Section 5 (Explain)--the gradient flow diagram should make this unambiguous. The CLIP encoder is a frozen pathway through which gradients pass to reach the embedding. |
| "The pseudo-token must somehow be a 'real' word -- you're finding the closest existing word for the concept" | Students may think textual inversion searches through existing vocabulary entries for the best match, or that the optimization converges to an existing token's embedding. The idea of a token that does not correspond to any word feels unusual. | After training, compute the cosine similarity between the learned embedding and all 49,408 entries in CLIP's vocabulary. The learned vector is NOT close to any single existing token. It typically sits in a region of embedding space that is "between" several related tokens--near "cat" and "fur" and the training subject's distinctive features, but not identical to any of them. It is a genuinely new point in the space, not a closest-match selection. | Section 7 (Elaborate)--after the core mechanism is understood. Frame as "the optimization finds a point in the continuous space, not a point in the discrete vocabulary." Connect to the "geometry encodes meaning" mental model: the continuous space has infinitely more points than the discrete vocabulary. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Train `<my-cat>` on 5-10 photos of a specific cat, then generate "a photo of `<my-cat>` wearing a top hat" and "a painting of `<my-cat>` in a forest" | Positive | Core demonstration that a single learned embedding can generalize to novel contexts (compositions, styles) the model has never seen with this subject. Shows the concept transfers across prompts. | The cat example is simple, universally understood, and directly parallels the subject LoRA from Lesson 1 (where the student also learned a specific subject). The novel prompts ("wearing a top hat," "in a forest") prove the embedding encodes the concept, not just a specific image. The student can reason about why this works: CLIP's cross-attention lets the `<my-cat>` embedding interact with "top hat" in the attention mechanism, producing a plausible composition. |
| Textual inversion for a specific artistic texture/pattern (e.g., a particular brush stroke style from 5 reference paintings) using `<my-style>` | Positive | Shows textual inversion applies to styles, not just objects. Generates "a mountain landscape in `<my-style>`" and "a portrait in `<my-style>`." Demonstrates the breadth of what "concept" means in embedding space. | Extends the student's understanding beyond "textual inversion for objects" to "textual inversion for any visual concept that can be described as a token-level attribute." Parallels the style vs subject distinction from Lesson 1's LoRA treatment. Also sets up the negative example: this works for simple, consistent styles but fails for complex multi-aspect styles. |
| Textual inversion for a complex compositional style (e.g., "always place subject in lower-left with dramatic backlighting and film grain") -- shows limitations | Negative | Demonstrates the expressiveness ceiling. A single 768-dim vector cannot encode a compositional recipe involving spatial layout, lighting, and texture simultaneously. The generated images capture some visual flavor but not the spatial composition or lighting pattern consistently. | This is the boundary of textual inversion. Contrasts with LoRA, which can encode compositional style patterns because it modifies spatial attention projections at every layer. Makes the "one vector" limitation tangible. The student should feel the difference: the embedding carries semantic meaning ("this kind of visual thing") but not procedural instructions ("put it here, light it this way"). |
| Comparing outputs from the same prompt with and without `<my-cat>` before/after training to verify zero weight change | Positive (verification) | Proves the core claim: the model is unchanged. Any prompt that does not use the pseudo-token produces identical results before and after textual inversion training. Only prompts containing the pseudo-token are affected. | The most direct proof that textual inversion changes embeddings, not weights. This is the negative example for Misconception #1. The student runs the same prompt+seed before and after training and gets identical output, verifying that the model weights are untouched. Then they run a prompt WITH `<my-cat>` and see the new concept appear. |

---

## Phase 3: Design

### Narrative Arc

You have arrived at the final lesson of the Stable Diffusion series with two customization tools: LoRA, which modifies the model's weights to change how it processes text; and img2img/inpainting, which modifies the inference process to control what input the model works with. Both are powerful. But both have a limitation. LoRA requires hundreds of training steps and modifies thousands of parameters. Img2img requires an existing image to start from. What if you want something simpler--what if you could teach the model a single new word?

Consider a concrete problem: you have five photos of your cat. You want to generate "my cat wearing a spacesuit on Mars." No existing prompt can describe your specific cat precisely enough for Stable Diffusion to produce it. LoRA could work, but it feels like bringing a sledgehammer to a thumbtack. You do not need to change how the model processes text or generates images. You need to change what one word means.

This is the insight behind textual inversion. The model understands language through CLIP's embedding space, where every word maps to a 768-dimensional vector. You learned in Module 4.1 that embeddings are learned parameters--trainable rows in a matrix. You saw in the CLIP lesson that this embedding space is continuous and meaningful: nearby vectors represent similar concepts. What if you created a new token--`<my-cat>`--and optimized its embedding vector so that, when the model processes the prompt "a photo of `<my-cat>`," it generates images that look like your cat? The model's weights stay completely frozen. The U-Net, VAE, and CLIP text encoder do not change at all. You are not changing the machine. You are changing what the machine hears.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual/Diagram | Two-stage pipeline diagram showing: (1) the tokenizer adding `<my-cat>` to vocabulary and assigning it an ID, (2) the embedding lookup table with the new row highlighted (the ONLY trainable parameter, marked in violet while everything else is gray/frozen), (3) the CLIP transformer processing all embeddings (frozen), (4) the contextual embeddings feeding into cross-attention (frozen). Gradient flow arrows showing backpropagation passing through the frozen CLIP encoder to reach the single trainable embedding. | The student needs to see exactly where the optimization target sits in the pipeline and why gradients must flow through frozen components to reach it. The diagram makes the "everything frozen except one vector" claim visually verifiable. Color-coding (violet for trainable, gray for frozen) directly parallels the LoRA lesson's diagram, reinforcing the contrast: in LoRA, the violet was at the cross-attention projections; here, the violet is at the embedding table. |
| Concrete example | Worked example: 5 photos of a cat. Pseudo-token `<my-cat>` assigned ID 49409 (one past the end of CLIP's 49408-entry vocabulary). Initial embedding: random 768-dim vector (or initialized from nearest concept, e.g., "cat"). Training: encode prompt "a photo of `<my-cat>`" through tokenizer and CLIP, run standard DDPM training step (VAE encode training image, sample t, noise, U-Net predicts noise, MSE loss), backprop through all frozen components to update only the 768-float embedding vector. After 3000-5000 steps, the embedding has converged to a point in CLIP's space that represents this specific cat. | Concrete numerical details (token ID 49409, 768 floats, 3000-5000 steps) ground the abstract concept. The student can trace exactly what is trainable (one row of 768 floats) and compare to LoRA (thousands of parameters across multiple projection matrices). The training step parallels the LoRA training step almost exactly--the only difference is what receives the gradient update. |
| Symbolic/Code | Pseudocode showing: (1) extend tokenizer vocabulary (`tokenizer.add_tokens(["<my-cat>"])`), (2) resize embedding (`text_encoder.resize_token_embeddings(len(tokenizer))`), (3) freeze everything (`for p in unet.parameters(): p.requires_grad_(False)`, same for VAE and text_encoder), (4) unfreeze ONLY the new embedding (`token_embeds[new_token_id].requires_grad_(True)`), (5) optimizer = Adam([token_embeds[new_token_id]], lr=5e-4). The training loop is the same DDPM loop from 6.2.3 and 6.5.1. | Code makes the "freeze everything except one vector" pattern mechanically clear and verifiable. The student can see that the optimizer receives exactly ONE parameter tensor of shape [768]. The `requires_grad_(False)` / `requires_grad_(True)` pattern directly parallels the LoRA lesson, making the contrast explicit: same pattern, different target. |
| Verbal/Analogy | "Inventing a new word" analogy: imagine you speak a language (CLIP's embedding space) where every word has a precise geometric position. Your cat has no word in this language. Textual inversion is like coining a new word--finding the exact position in the language's semantic space where, if a word existed there, speakers (the U-Net) would understand it to mean your specific cat. You are not changing the grammar (U-Net weights) or the accent (inference process). You are adding one word to the dictionary (embedding table). | Extends the existing "CLIP is a translator" and "the U-Net never sees text" mental models. The "inventing a word" framing captures the core idea: the model already knows how to process any point in embedding space; you just need to find the right point for your concept. It also sets up the expressiveness limitation naturally: a single word can describe a noun or simple adjective, but not a complex compositional recipe. |
| Intuitive/Geometric | The learned embedding visualized as a point in CLIP's embedding space--positioned near "cat" and "fur" and "tabby" but distinct from all of them. Not on any existing token's position but in the continuous space between them. Connect to the EmbeddingSpaceExplorer experience from 4.1.3: the student saw semantic clusters and knows that positions between clusters are valid and meaningful. Textual inversion finds a point that was always valid in the space but had no word assigned to it. | The student has visceral experience with embedding space geometry from the interactive explorer. They dragged, hovered, and searched tokens in 2D projections. They know the space is continuous and that nearby means similar. Visualizing the learned embedding as a new point in this familiar space connects the abstract optimization to a concrete geometric intuition. The "space between existing tokens" framing addresses Misconception #5: the result is NOT the nearest existing word. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. Pseudo-token creation and embedding optimization (add a new token to the vocabulary, initialize its embedding, optimize it via the standard DDPM training loop while freezing all model weights)
  2. Textual inversion's expressiveness boundaries compared to LoRA (one 768-dim vector vs thousands of LoRA parameters; captures concepts expressible as a token-level descriptor but not compositional or spatial style patterns)

  Supporting concepts (not conceptually new, but new framings):
  - The two-stage CLIP text encoding pipeline (token embedding lookup -> transformer contextualizer) -- the student knows both stages but has not thought about them as separable stages where you can intervene at stage 1
  - Gradient flow through frozen components -- the student has done "freeze everything except X" but has not thought about gradients flowing through the entire CLIP encoder to reach X at the very start of the pipeline
  - The optimization landscape of embedding space -- connecting the "geometry encodes meaning" mental model to gradient-based optimization

- **Previous lesson's load:** img2img-and-inpainting was BUILD (2 new concepts: img2img as partial denoising, inpainting as per-step spatial masking).

- **This lesson's load:** STRETCH. The optimization target is conceptually surprising. Even though the training loop is the same DDPM loop the student has seen three times, the idea that you can optimize a single embedding vector while keeping the entire multi-hundred-million-parameter model frozen requires a genuine reframing of "what is trainable" in a neural network. The student's prior experience with optimization has always targeted model weights (gradient descent in Series 1, LoRA in 4.4 and 6.5). Targeting the input representation instead is a conceptual inversion.

- **Appropriate given trajectory:** Yes. The module plan specifies BUILD -> BUILD -> STRETCH. Two BUILD lessons precede this STRETCH. The student arrives comfortable and confident from reconfiguring familiar mechanisms. The STRETCH comes from the conceptual surprise, not from mechanical complexity--the training loop is identical to what they have done before.

### Connections to Prior Concepts

- **Token embeddings (4.1.3):** Direct application. The student learned "nn.Embedding is a lookup table" and "embeddings are learned parameters (requires_grad=True)." Textual inversion takes this to its logical conclusion: if individual embedding rows are trainable parameters, you can optimize a single row for a specific purpose. The student explicitly saw `embedding.weight[i]` and verified it was a trainable tensor. Now they optimize one such tensor for their cat.
- **Embedding space clustering (4.1.3):** The student explored the EmbeddingSpaceExplorer and saw that "geometry encodes meaning" -- similar tokens cluster nearby. Textual inversion finds a new point in this space where the meaning IS the target concept. The optimization navigates the same geometric space the student explored interactively.
- **CLIP shared embedding space (6.3.3):** The student learned that CLIP's space is continuous, meaningful, and geometrically organized. The core claim of textual inversion relies on this: a well-placed vector in CLIP's space can represent a concept the model has never seen. If the space were discontinuous or sparse, this would not work.
- **Rare-token identifier "sks" from subject LoRA (6.5.1):** In Lesson 1, the student learned that subject LoRAs use a rare token like "sks" to avoid polluting common words. Textual inversion takes a different approach: instead of repurposing a rare existing token, you create an entirely new one. The student can compare: LoRA + "sks" trains new weight pathways triggered by an existing token; textual inversion trains a new embedding for a new token while keeping all weight pathways unchanged.
- **"Freeze everything except X" pattern (6.5.1):** LoRA training froze the base U-Net and optimized only LoRA parameters. Textual inversion uses the identical pattern but with X = one embedding row instead of X = LoRA matrices. The student can see this as the same surgical training philosophy applied to a different target.
- **DDPM training loop (6.2.3, 6.2.5, 6.5.1):** The training loop is literally the same. Sample image, VAE encode, sample random t, noise, U-Net predicts noise, MSE loss. The only difference is what receives the gradient update. The student has implemented this loop from scratch (6.2.5) and adapted it for LoRA (6.5.1).
- **Cross-attention as the bridge from text to image (6.3.4):** The pseudo-token's embedding enters the U-Net through cross-attention, exactly as any other token does. The student knows that each spatial location attends to text tokens via cross-attention. The pseudo-token participates in this attention like any other token--it advertises its content via K/V, and spatial locations attend to it via Q. No special mechanism is needed.
- **"The U-Net never sees text" (6.4.1):** The student knows the U-Net only sees 77x768 floating-point tensors. It has no way to distinguish between an embedding that came from a "real" word and one that was optimized by textual inversion. To the U-Net, `<my-cat>` is just another column in the K/V matrices.

**Potentially misleading prior analogies:**
- **"Finetuning is a refinement, not a revolution" (4.4.4):** This analogy was about LoRA weight changes being low-rank refinements. It does NOT directly apply to textual inversion, which is not refining existing representations but creating a new one from scratch. The optimization finds a point in embedding space, which could be anywhere--it is not necessarily a small delta from an existing point. However, the initialization strategy (starting from a related word like "cat") does introduce a refinement framing that is partially applicable.

### Scope Boundaries

**This lesson IS about:**
- Creating a pseudo-token and optimizing its embedding vector in CLIP's text embedding space
- The two-stage pipeline (embedding lookup -> CLIP transformer) and where textual inversion intervenes
- The training loop: same DDPM loop, only one embedding vector receives gradients
- Initialization strategies (random, from related word)
- Expressiveness comparison with LoRA (one vector vs thousands of parameters)
- When to use textual inversion vs LoRA vs img2img
- The complete customization spectrum as a module summary

**This lesson is NOT about:**
- DreamBooth (full model or partial model fine-tuning for specific subjects, different technique)
- Multi-token textual inversion (using multiple pseudo-tokens for one concept; mentioned only)
- Hypernetworks or other embedding-space methods
- Training tricks specific to textual inversion (progressive training schedules, mixing real and synthetic images)
- CLIP's internal architecture details (transformer layers, attention heads)
- Reimplementing CLIP from scratch
- ControlNet or other structural conditioning (Series 7)
- Detailed mathematical analysis of the embedding optimization landscape
- Production deployment or optimization for speed

**Target depth:** Pseudo-token creation and embedding optimization at DEVELOPED. The student should be able to explain the mechanism (what is optimized, what is frozen, how gradients flow), trace the data path through the pipeline, and use the diffusers library to train a textual inversion embedding. Expressiveness comparison at DEVELOPED -- the student should be able to recommend textual inversion vs LoRA for a given use case. Not APPLIED because the notebook exercises are Guided and Supported, not fully independent technique selection from scratch.

### Lesson Outline

1. **Context + Constraints**
   - "You have two customization tools: LoRA changes the model's weights, img2img/inpainting changes the inference process. This lesson introduces the third and most surprising approach: changing what the model hears without changing the model at all."
   - Scope: NOT DreamBooth. NOT full fine-tuning. NOT CLIP reimplementation. Just one new embedding vector.
   - This is the FINAL lesson of Series 6 (Stable Diffusion). The module and series wrap up here.

2. **Recap** (brief, targeted)
   - Reactivate the two-stage CLIP text encoding pipeline: (1) tokenizer converts text to integer IDs, (2) nn.Embedding looks up initial vectors, (3) CLIP transformer applies self-attention to produce contextual embeddings, (4) contextual embeddings feed into U-Net cross-attention as K/V. One paragraph + brief diagram callback.
   - Reactivate "embeddings are learned parameters" from 4.1.3: each row of the embedding table is a trainable tensor. The student proved this by checking `embedding.weight.requires_grad`. One sentence.
   - Reactivate "CLIP's embedding space is continuous and meaningful" from 6.3.3: nearby vectors represent similar concepts. Geometry encodes meaning. One sentence.
   - Purpose: these three facts--the two-stage pipeline, embeddings as parameters, and the meaningful space--are the three premises from which textual inversion follows logically. The student should be able to derive the technique from these premises before being told.

3. **Hook** -- "What if you could invent a new word?"
   - Type: Problem + derivation challenge. Present the scenario: "You have five photos of your cat. You want to generate 'my cat wearing a spacesuit on Mars.' No prompt can describe your specific cat precisely enough. LoRA could work, but you would modify thousands of parameters across the entire U-Net. What if there were a simpler way?"
   - Derivation prompt (inside a `<details>` reveal): "You know three things: (1) Every word maps to a 768-dim vector via an embedding lookup. (2) These vectors are trainable parameters. (3) CLIP's embedding space is continuous and meaningful. From these three facts alone, can you derive a technique for teaching the model your cat?"
   - The student should be able to reason: "Create a new token, give it an embedding vector, and optimize that vector so the model generates my cat when it sees that token." If they get there, they have derived textual inversion before being told its name.
   - Reveal: "This technique is called textual inversion (Gal et al., 2022). You just reinvented it."

4. **Explain** -- The textual inversion mechanism
   - **The pseudo-token:** Create a new token `<my-cat>`, assign it the next available ID (e.g., 49409 for CLIP with 49408 entries). Extend the embedding table by one row. This row is the ONLY trainable parameter.
   - **Initialization:** Two strategies. (a) Random initialization -- the embedding starts at a random point in CLIP's space. Works but slower convergence. (b) Initialize from a related word -- start from the embedding of "cat" and optimize from there. Faster convergence because the optimization starts in the right neighborhood. Connect to "geometry encodes meaning": starting near a related word means starting in a semantically relevant region.
   - **The training loop:** Side-by-side ComparisonRow with the LoRA training loop from 6.5.1 (which itself was a side-by-side with the LLM LoRA training loop). Same DDPM steps: encode training image with frozen VAE, sample random t, noise to z_t, construct prompt "a photo of `<my-cat>`", encode prompt through tokenizer + frozen CLIP text encoder, U-Net predicts noise with cross-attention receiving the prompt embeddings, MSE loss on noise prediction, backprop. The ONLY difference from LoRA training: gradients flow all the way back through the frozen U-Net and frozen CLIP encoder to update the single embedding vector. Everything else is identical.
   - **Gradient flow diagram:** Visual showing the forward pass (embedding lookup -> CLIP transformer -> cross-attention -> U-Net -> noise prediction -> loss) and the backward pass (loss -> U-Net [frozen, gradients pass through] -> cross-attention [frozen] -> CLIP transformer [frozen] -> embedding lookup [ONLY the pseudo-token's row receives the update]). Color-coded: violet for the one trainable parameter, gray for everything frozen.
   - **Zero weight change verification:** After training, any prompt WITHOUT `<my-cat>` produces identical results to before training. The model is unchanged. Only the embedding vocabulary has grown by one entry.
   - **Pseudocode:** The 5-step setup (add token, resize embeddings, freeze everything, unfreeze one row, create optimizer with that one row) followed by the training loop (identical to LoRA training loop from 6.5.1 except the optimizer target).
   - **Parameter count comparison:** Textual inversion: 768 trainable parameters (one embedding vector). LoRA (r=4, 4 cross-attention projections per block, ~16 blocks): ~200,000-400,000 trainable parameters. Full U-Net fine-tuning: ~860M parameters. Three orders of magnitude difference between textual inversion and LoRA.

5. **Check** -- Predict-and-verify for mechanism understanding
   - "After textual inversion training, you generate an image with the prompt 'a beautiful sunset over the ocean' (no `<my-cat>` token). Has the output changed compared to before training?" (`<details>` reveal: No. The prompt does not contain the pseudo-token, so the new embedding row is never looked up. The tokenizer maps "a beautiful sunset over the ocean" to the same token IDs as before. The CLIP encoder processes the same input embeddings. The U-Net receives the same K/V tensors. Identical output.)
   - "You train `<my-cat>` on cat photos. Then you try the prompt 'a photo of `<my-cat>` playing piano.' Will the model know what a piano is, even though no training image showed a piano?" (`<details>` reveal: Yes. The model's knowledge of pianos, composition, and spatial relationships is in the U-Net's frozen weights. Textual inversion only taught the model what `<my-cat>` looks like. The U-Net already knows how to combine a subject with a scene. Cross-attention lets spatial locations attend to both `<my-cat>` and "piano" independently.)

6. **Explore** -- The two-stage pipeline distinction (interactive understanding)
   - TryThisBlock: "Trace the data flow for the prompt 'a photo of `<my-cat>` sitting on grass.' At each stage, what is frozen and what is the result of training?"
   - Walk through: tokenizer (frozen, just maps characters to IDs including the new one), embedding lookup (ONE row is the trained vector, rest frozen), CLIP transformer self-attention (frozen, but contextualizes the new embedding WITH surrounding tokens--critical point), output [77, 768] contextual embeddings, cross-attention K/V projections (frozen).
   - Key insight: textual inversion optimizes the embedding BEFORE the CLIP transformer processes it. The CLIP transformer then contextualizes it. This means the same initial embedding produces different contextual embeddings depending on the surrounding prompt. "a photo of `<my-cat>`" and "a sketch of `<my-cat>`" produce different K/V inputs to cross-attention because the CLIP transformer sees different contexts. This is NOT a limitation--it is a feature. The pseudo-token adapts to context, just like real words do.

7. **Elaborate** -- Expressiveness, limitations, and the complete spectrum
   - **Expressiveness comparison:** ComparisonRow (textual inversion vs LoRA):
     - Textual inversion: 768 parameters, trains in 3000-5000 steps, produces a single file (~4 KB), captures object identity and simple visual attributes, works well for "what does this thing look like?" Limited for complex spatial styles.
     - LoRA: 200K-400K parameters, trains in 500-2000 steps, produces a 2-50 MB file, captures style patterns including spatial composition, color palettes, rendering techniques. Works well for "how should images in this style look?"
   - **When to use which:** Three GradientCards:
     - Textual inversion: "I want to add MY specific cat/object/texture to any prompt." Small file, easy to share, no weight changes.
     - LoRA: "I want the model to generate in a specific style or capture a complex visual pattern." More parameters, more expressive, modifies weights.
     - Img2img/inpainting: "I want to edit or transform a specific existing image." No training at all, inference-time only.
   - **Limitations of textual inversion:**
     - Single concept per token (multi-token variants exist but are more complex; MENTIONED only)
     - Slow convergence compared to LoRA (optimizing in the high-dimensional CLIP space with gradients flowing through the entire frozen network is less efficient than LoRA's direct weight optimization)
     - The "one word" ceiling: textual inversion can encode what a concept LOOKS LIKE but not how to compose or light it. A single token participates in cross-attention at the token level, not at the architectural level.
   - **The learned embedding in geometric space:** After training, the embedding sits at a specific point in CLIP's 768-dim space. It is NOT at any existing vocabulary token's position. It is a new point that encodes the training concept. Callback to EmbeddingSpaceExplorer: the continuous space has infinitely more positions than the discrete vocabulary has entries. Textual inversion finds one of these unused positions.
   - **Multi-token textual inversion:** Brief mention (MENTIONED depth). Using 2-4 pseudo-tokens for one concept gives more capacity (2-4 vectors = 1536-3072 parameters). Each token can specialize. Not developed further.

8. **Check** -- Transfer questions
   - "A colleague wants to teach the model what their company logo looks like, so they can generate marketing images in various contexts ('our logo on a coffee mug,' 'our logo as a neon sign'). Would you recommend textual inversion or LoRA? Why?" (`<details>` reveal: Textual inversion. A logo is a specific visual object--exactly the kind of concept that maps to a single token descriptor. The colleague wants to use the logo in novel contexts, which means the model's existing compositional knowledge should be preserved. Textual inversion does not modify the model at all, so all compositional abilities remain intact. LoRA would work too but is overkill: modifying thousands of weights to teach one visual concept. Textual inversion's 768 parameters are sufficient and produce a tiny, easily shareable file.)
   - "You trained a textual inversion embedding for your cat using the prompt template 'a photo of `<my-cat>`.' Now you want to use it with a different Stable Diffusion model (say, SD v1.5 instead of SD v1.4). Will the embedding work?" (`<details>` reveal: It depends on whether both models use the same CLIP text encoder. SD v1.4 and v1.5 use the same CLIP (openai/clip-vit-large-patch14), so the embedding transfers. SD v2.0 uses a different CLIP encoder (OpenCLIP ViT-H), so the embedding would NOT transfer--it would be a vector in the wrong embedding space. This is analogous to speaking a word from one language in a conversation conducted in another language. The "word" only means something in the embedding space where it was trained.)

9. **Practice** -- Notebook exercises (Colab)
   - **Exercise 1 (Guided):** Explore the CLIP embedding table. Load a pretrained CLIP text encoder. Inspect the embedding table shape (49408 x 768). Look up the embeddings for "cat," "dog," "kitten." Compute cosine similarities between them. Add a new token `<my-concept>` and verify the embedding table grows to 49409 x 768. Inspect the new row (random values). Purpose: make the embedding table tangible and verify that adding a token is mechanically simple.
   - **Exercise 2 (Guided):** One textual inversion training step by hand. Load a training image, VAE encode it. Create the prompt "a photo of `<my-concept>`" and tokenize it. Run the prompt through the CLIP text encoder. Set up the diffusion training step (sample t, noise, U-Net prediction, MSE loss). Backprop and verify that ONLY the new embedding row has non-zero gradients. Print the gradient norm. Purpose: verify the "freeze everything except one vector" claim mechanically. Predict-before-run: "How many parameters will have non-zero gradients?"
   - **Exercise 3 (Supported):** Train a full textual inversion embedding. Use a provided set of 5-8 concept images. Train for 3000 steps using the diffusers `textual_inversion` training script or a simplified loop. Generate images with and without the pseudo-token. Compare results at different training checkpoints (500, 1500, 3000 steps). Purpose: hands-on training with observation of convergence. Key observation: early checkpoints produce vague results; later checkpoints produce specific concept identity.
   - **Exercise 4 (Independent):** Compare textual inversion vs LoRA. Take the same set of concept images. Train a textual inversion embedding AND a LoRA adapter (using the approach from Lesson 1's notebook). Generate from the same prompts using each. Compare: file size, training time, output quality for simple prompts (object in new context), output quality for complex prompts (object in specific style/composition). Write a brief comparison. Purpose: the student experiences the expressiveness difference firsthand rather than taking the lesson's word for it.
   - Exercises are cumulative: Ex1 grounds the embedding table, Ex2 verifies the mechanism, Ex3 trains a real embedding, Ex4 compares to LoRA.
   - Key reasoning to emphasize in solutions: WHY only one row gets gradients (everything else is frozen), WHY initialization matters (starting near "cat" vs random), WHY the learned embedding does not equal any existing word (continuous space, optimization finds a novel point), WHY textual inversion is less expressive than LoRA (one vector vs many projection matrices).

10. **Summarize**
    - Three key takeaways: (1) Textual inversion creates a new pseudo-token and optimizes its 768-dimensional embedding vector to represent a novel concept, while keeping the entire model (U-Net, VAE, CLIP encoder) completely frozen. (2) The optimized embedding enters the U-Net through the normal cross-attention pathway -- the U-Net cannot distinguish between "real" word embeddings and the optimized one. (3) Textual inversion is the most lightweight customization technique (768 parameters, ~4 KB file) but is less expressive than LoRA for complex styles.
    - Mental model echo: "Inventing a new word in CLIP's language -- finding the right point in embedding space where, if a word existed there, the model would understand it as your concept."
    - Module completion: "Three knobs on three different parts of the pipeline: LoRA changes the weights, img2img/inpainting changes the inference process, textual inversion changes the embeddings. Each is appropriate for different customization needs. You now understand all three."

11. **Next step** -- Series 6 conclusion
    - "You have traveled from 'what is a generative model?' through autoencoders, VAEs, diffusion theory, U-Net architecture, CLIP, cross-attention, classifier-free guidance, latent diffusion, samplers, and three customization techniques. You did not just learn to use Stable Diffusion -- you built the intuition for every component. When you adjust a parameter, you know what changes inside the pipeline. When a new technique is announced, you have the foundation to understand why it works."
    - Forward reference to Series 7: "Series 7 will explore what came after Stable Diffusion: ControlNet for structural control, SDXL for higher resolution, consistency models for faster generation, and flow matching as a new framework. Every one of these builds on what you now understand."

---

## Widget Assessment

**Widget needed:** No custom interactive widget required.

The lesson's core visual needs are:
1. Two-stage pipeline diagram with gradient flow arrows (embedding lookup -> CLIP transformer -> cross-attention -> U-Net) -- achievable with a static SVG or Mermaid diagram with color-coded frozen/trainable regions
2. ComparisonRow (textual inversion training loop vs LoRA training loop) -- existing component
3. GradientCards for the customization spectrum comparison -- existing component
4. Parameter count comparison -- simple text or table

The most impactful visual (the gradient flow diagram) does not require interactivity -- the key insight is structural (WHERE the trainable parameter sits), not dynamic. The notebook provides the hands-on interactive experience. The student's prior experience with the EmbeddingSpaceExplorer (4.1.3) provides the geometric intuition; the lesson can reference that experience without rebuilding it.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (module 6.5 record, module 6.3 record, module 4.1 record)
- [x] Depth match verified for each (all OK)
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises (each exercise builds on the previous)
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (2 positive + 1 verification + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (2 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-14 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: MAJOR REVISION

Critical findings exist. Must fix before this lesson is usable.

### Findings

#### [CRITICAL] -- Notebook Exercise 2 gradient verification will show wrong result

**Location:** Notebook cell-21 (Exercise 2, Step 6: Backprop and verify gradient flow)
**Issue:** The cell claims "ONLY the new embedding row has non-zero gradients" and prints gradient norms for all tokens in the prompt to verify this. However, this claim is incorrect. When `requires_grad_(True)` is set on the entire embedding weight tensor (cell-16), ALL token rows that participate in the forward pass receive gradients via autograd, not just the pseudo-token row. The CLIP transformer applies self-attention across all 77 tokens, so gradients from the loss flow through every contextual embedding position, through the CLIP transformer, and back to every initial embedding that was looked up. The SOT token, "a", "photo", "of", the pseudo-token, the EOT token, and the padding tokens all have non-zero gradients. The cell's commentary states "Rows with non-zero gradients: [49408]" but in reality, it will show multiple rows with non-zero gradients.

The "What Just Happened" markdown summary (cell-24) also states: "Only the pseudo-token's embedding row has non-zero gradients. All other embedding rows... have zero gradient updates." This is factually wrong. The predict-before-run question in cell-14 asks "How many parameters will have non-zero gradients?" and implies the answer is 768 (one row), which is also wrong.

**Student impact:** The student runs the cell and sees that MULTIPLE rows have non-zero gradients, directly contradicting both the lesson's claim and the notebook's printed commentary. This is deeply confusing because it appears to disprove the core mechanism. The student may conclude the lesson is wrong, or worse, that they misunderstand something fundamental. This is the single most important verification in the notebook and it produces a result that contradicts the narrative.

**Suggested fix:** Reframe the verification. The correct framing is: "All rows that participate in the forward pass receive gradients (the embedding table is one tensor with `requires_grad=True`), but in a real textual inversion training loop, we only UPDATE the pseudo-token's row." The verification should focus on two things: (1) the U-Net, VAE, and CLIP transformer weights have zero gradients (already verified in cell-22, which is correct), and (2) demonstrate the restore-original-embeddings pattern that ensures only the pseudo-token's row actually changes. Alternatively, show that the gradient magnitudes for the pseudo-token are significantly larger than for other tokens (since the loss signal is concentrated there), but do not claim other rows have zero gradients.

Additionally, the lesson TSX pseudocode (line 505) shows `token_embeds[new_token_id].requires_grad_(True)` which would set `requires_grad` on a single row slice. In PyTorch, setting `requires_grad_(True)` on a tensor slice does not work as expected because the slice is not a leaf tensor. This pseudocode is conceptually correct but mechanically misleading. The notebook correctly works around this by enabling grad on the whole table and restoring rows, but the lesson pseudocode suggests a simpler mechanism that does not actually work.

---

#### [IMPROVEMENT] -- Lesson pseudocode shows non-functional freeze/unfreeze pattern

**Location:** TSX lesson, "The Setup in Code" section (lines 499-518)
**Issue:** The pseudocode shows:
```python
token_embeds[new_token_id].requires_grad_(True)
optimizer = Adam([token_embeds[new_token_id]], lr=5e-4)
```
In PyTorch, `token_embeds[new_token_id]` is a tensor slice (not a leaf tensor). Calling `.requires_grad_(True)` on it does not make just that row trainable in the way the code implies. The slice is a view of the larger embedding tensor, and `requires_grad` is a property of the entire parameter tensor, not individual rows. Additionally, `Adam([token_embeds[new_token_id]])` would optimize a detached copy, not the actual row in the embedding table, unless handled carefully.

The notebook (cell-29) correctly addresses this by enabling grad on the full embedding table and restoring other rows after each step, but the lesson pseudocode shows a simpler pattern that would not work in practice.

**Student impact:** A student who tries to implement the lesson pseudocode directly would get unexpected behavior or errors. The discrepancy between the lesson's clean pseudocode and the notebook's more complex approach is unexplained.

**Suggested fix:** Either (a) add a brief note in the lesson that the pseudocode is simplified and the notebook shows the actual working pattern, or (b) show the real pattern (enable grad on the full table, mask updates by restoring original rows) in the lesson pseudocode. Option (a) is simpler and keeps the lesson focused on the concept rather than implementation details.

---

#### [IMPROVEMENT] -- Exercise 3 loads all components in float32, potential VRAM pressure on T4

**Location:** Notebook cell-26 (Exercise 3 component loading)
**Issue:** Exercise 3 loads the VAE, U-Net, CLIP text encoder, and dataset all in float32. The U-Net alone is ~3.4 GB in float32. Combined with VAE (~336 MB), text encoder (~492 MB), optimizer states for the embedding table (minimal), and forward/backward activations through the full U-Net + CLIP encoder, this could approach or exceed the ~15 GB available on a free T4 Colab instance. Training requires retaining activations for the entire backward pass through the frozen U-Net and CLIP encoder because gradients must flow to the embedding table.

**Student impact:** The student may encounter OOM errors during training, which is frustrating and difficult to debug. The VRAM tip at the top says to restart runtime, but does not explain that Exercise 3 might inherently exceed free-tier limits.

**Suggested fix:** Use mixed precision for the frozen components. Keep the embedding table in float32 (for gradient stability) but load the frozen VAE and U-Net in float16. Since they are frozen and no weight updates happen, float16 is safe. The text encoder must remain in float32 because gradients flow through it. Alternatively, add a note that if OOM occurs, reduce to float16 for VAE and U-Net while keeping text_encoder in float32.

---

#### [IMPROVEMENT] -- Narrative framing drifts from "specific object" to "style" without acknowledgment

**Location:** TSX lesson hook (Section 4) through notebook Exercise 3
**Issue:** The lesson's narrative consistently frames textual inversion around learning a specific visual object ("your cat"). The derivation challenge uses the cat example, the pseudocode uses `<my-cat>`, and the zero-weight-change verification uses `<my-cat>`. However, Exercise 3 in the notebook trains `<naruto-style>` initialized from "anime" -- a style concept, not an object. The lesson later explicitly states that textual inversion works poorly for complex styles (the "one word ceiling" section). This creates a tension: the notebook trains on exactly the kind of concept (style) that the lesson warns is a poor fit for textual inversion.

**Student impact:** The student may be confused about whether textual inversion is suited for styles or not. The lesson says "works well for specific visual objects, poorly for complex styles" but the notebook trains a style. If the results are weak (which they likely will be at 300 steps for a style), the student does not know whether this is expected behavior (textual inversion is bad at styles) or insufficient training.

**Suggested fix:** Either (a) change the notebook to train on a specific object concept (e.g., use a small set of consistent object images, or use a subset of a dataset filtered for one visual subject) to match the lesson's framing, or (b) keep the naruto-style training but explicitly frame it as demonstrating the limitation -- "Notice how 300 steps of textual inversion on a style dataset produces vague results. This previews the expressiveness limitation you will explore in Exercise 4." Option (b) is better because it turns the weakness into a learning moment.

---

#### [IMPROVEMENT] -- Missing `sklearn` in pip install (minor but blocks execution)

**Location:** Notebook cell-2 (Setup) and cell-9 (PCA import)
**Issue:** Cell-9 imports `from sklearn.decomposition import PCA`. While `scikit-learn` is pre-installed on Colab, this is not guaranteed for all environments and is not listed in the `pip install` cell. If a student runs this notebook locally or on a non-Colab environment, this import will fail.

**Student impact:** On Colab, no impact. On other environments, Exercise 1 breaks at the PCA visualization step.

**Suggested fix:** Add `scikit-learn` to the pip install line: `!pip install -q diffusers transformers accelerate peft datasets scikit-learn`. This is defensive and costs nothing.

---

#### [POLISH] -- Spaced em dashes in student-visible TSX prose

**Location:** TSX lesson, multiple locations
**Issue:** Three student-visible locations use a literal em dash character with surrounding spaces instead of the no-space `&mdash;` pattern:
1. Line 277: ComparisonRow item `'Faster convergence — less distance to travel'` (student-visible card text)
2. Line 618: SectionHeader subtitle `"Textual inversion optimizes stage 1 — CLIP contextualizes in stage 2"` (student-visible subtitle)

Line 508 (`# Training loop — identical to LoRA training from 6.5.1`) is inside a code comment within a CodeBlock, which is acceptable.

**Student impact:** Minor visual inconsistency with em dash formatting conventions.

**Suggested fix:** Replace the literal ` — ` with `&mdash;` (no spaces) in lines 277 and 618. For line 277, since it is inside a string in a ComparisonRow items array, use the Unicode em dash character `\u2014` without spaces: `'Faster convergence\u2014less distance to travel'`. For line 618, use `&mdash;` directly in the JSX: `"Textual inversion optimizes stage 1&mdash;CLIP contextualizes in stage 2"`.

---

#### [POLISH] -- Mermaid diagram uses emoji characters that may not render in all browsers

**Location:** TSX lesson, line 396-413 (gradient flow Mermaid diagram)
**Issue:** The Mermaid diagram node labels use emoji characters (fire emoji for trainable, snowflake for frozen). Mermaid rendering of emoji is inconsistent across browsers and rendering engines. Some render them well; others show empty squares or misaligned text.

**Student impact:** If emoji do not render, the diagram loses its at-a-glance frozen/trainable distinction. The color coding still works, so this is cosmetic.

**Suggested fix:** Replace emoji with text labels: instead of `(ONE row trainable [fire])` use `(ONE row TRAINABLE)` and instead of `(frozen [snowflake])` use `(frozen)`. The `classDef` colors already encode the distinction. Alternatively, test that the Mermaid renderer used by the app handles emoji correctly.

---

#### [POLISH] -- HTML comment em dashes use spaced format

**Location:** TSX lesson, lines 159, 223, 547, 612, 708, 910, 976, 1177
**Issue:** HTML comments in the TSX file use ` — ` (spaced em dash) in section separators like `Section 4: Hook — "What if you could invent a new word?"`. These are not student-visible but are inconsistent with the no-space convention.

**Student impact:** None (comments are not rendered).

**Suggested fix:** Low priority. These are invisible to the student. Fix only if doing a sweep of the file for other reasons.

### Review Notes

**What worked well:**
- The narrative arc is excellent. The "three premises" framing in the recap is pedagogically strong -- it sets up the student to derive textual inversion themselves before being told.
- The derivation challenge in the hook is the best part of the lesson. Having the student reason from premises to technique before being given the name is highly engaging.
- The side-by-side comparison with LoRA training is well executed. Making the training loop identical except for step 6 clearly communicates the key insight.
- The Mermaid gradient flow diagram effectively shows the forward/backward pass through frozen layers.
- All Row wrapping is correct throughout the lesson.
- The notebook's exercise progression (inspect -> verify -> train -> compare) is well-scaffolded and matches the planning document.
- The lesson follows the planning document's outline closely with no significant deviations.

**Systemic pattern:**
The critical finding is about a fundamental PyTorch behavior that the lesson gets conceptually right but mechanically wrong. In textual inversion, the *intention* is to update only one embedding row, but the *mechanism* requires enabling grad on the full embedding table and masking unwanted updates. The lesson's pseudocode and the notebook's verification both present the simplified version as if it were the real mechanism. The notebook needs to either show the real mechanism or reframe its verification to match what actually happens.

---

## Review -- 2026-02-14 (Iteration 2/3)

### Iteration 1 Fix Verification

| # | Finding | Severity | Status | Notes |
|---|---------|----------|--------|-------|
| 1 | Notebook Exercise 2 gradient verification will show wrong result | CRITICAL | FIXED | Cell-14 intro explicitly explains "the trick is not that only one row gets gradients -- it is that only one row gets updated." Cell-21 correctly shows multiple rows have non-zero gradients, then demonstrates the restore-original-rows pattern in step 6b. Cell-24 summary correctly states "Multiple embedding rows receive gradients, not just the pseudo-token." |
| 2 | Lesson pseudocode shows non-functional freeze/unfreeze pattern | IMPROVEMENT | FIXED | "Pseudocode vs Reality" GradientCard (lines 530-549) explicitly states lines 4-5 are "conceptually simplified" and explains the actual pattern (enable grad on full table, restore rows after each step, notebook shows working implementation). |
| 3 | Exercise 3 loads all components in float32, potential VRAM pressure | IMPROVEMENT | ACCEPTED | All components still load in float32. For textual inversion training, gradients must flow through the U-Net and CLIP encoder to reach the embedding table, making float32 the safest choice for correctness. The VRAM tip and cleanup-between-exercises pattern are in place. Accepted as a reasonable tradeoff between correctness and VRAM pressure. |
| 4 | Narrative framing drifts from "specific object" to "style" | IMPROVEMENT | FIXED | Cell-25 now explicitly states: "We are *intentionally* training on a style dataset here to give you firsthand experience of that limitation." Cell-35 "What Just Happened" says: "If your results look vague or inconsistent, that is *expected* -- this is the 'one word ceiling' from the lesson." The style training is reframed as a deliberate pedagogical choice. |
| 5 | Missing sklearn in pip install | IMPROVEMENT | FIXED | Cell-2 now includes `scikit-learn` in the pip install line. |
| 6 | Spaced em dashes in student-visible TSX prose | POLISH | FIXED | Lines 277 and 638 now use unspaced em dashes. |
| 7 | Mermaid diagram uses emoji characters | POLISH | FIXED | Diagram now uses text labels ("ONE row TRAINABLE", "frozen") instead of emoji. Color coding via classDef provides the visual distinction. |
| 8 | HTML comment em dashes use spaced format | POLISH | ACCEPTED | Comments are not student-visible. Low priority. Not worth a revision pass. |

**Summary:** 5 of 8 findings fully fixed. 1 finding (VRAM/float32) accepted as a reasonable tradeoff. 1 finding (HTML comment em dashes) accepted as not student-impacting. All CRITICAL and most IMPROVEMENT findings resolved.

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

No critical or improvement findings. One polish item remains (HTML comment em dashes, carried from iteration 1, not student-visible). The lesson is ready to ship.

### Findings

#### [POLISH] -- HTML comment em dashes still use spaced format

**Location:** TSX lesson, lines 159, 223, 508, 567, 632, 728, 930, 996, 1199
**Issue:** HTML comments in the TSX file use ` — ` (spaced em dash) in section separators like `Section 4: Hook — "What if you could invent a new word?"`. Line 508 is a code comment inside a CodeBlock (`# Training loop — identical to LoRA training from 6.5.1`). These are not student-visible (HTML comments are stripped, and the code comment is inside a syntax-highlighted code block where em dash formatting conventions do not apply).
**Student impact:** None.
**Suggested fix:** Low priority. Fix only if doing a sweep of the file for other reasons. The code comment on line 508 is inside a CodeBlock and follows natural code comment style, which is acceptable.

### Review Notes

**What worked well (carrying forward from iteration 1, plus new observations):**
- The narrative arc remains excellent. The "three premises" framing, derivation challenge, and "you just reinvented it" moment are the strongest pedagogical sequence in the lesson.
- The iteration 1 fixes significantly improved the lesson. The "Pseudocode vs Reality" GradientCard is a strong addition that turns a potential confusion point into a teaching moment. Students now learn that the conceptual simplification masks a real PyTorch subtlety, which is itself an important lesson about the gap between pseudocode and implementation.
- The notebook's Exercise 2 rewrite is thorough. The three predict-before-run questions in cell-14 now correctly prime the student for the actual behavior (multiple rows get gradients, CLIP transformer weights have no gradients, restore-original-rows pattern). The step 6b demonstration makes the mechanism tangible.
- The Exercise 3 reframing ("deliberate choice" framing) is pedagogically strong. Instead of the style training being an unacknowledged contradiction, it is now explicitly positioned as a preview of the expressiveness limitation that Exercise 4 will explore further. This turns a weakness into a learning moment.
- Exercise 4's solution is comprehensive and well-scaffolded with VRAM-conscious cleanup between LoRA training and TI generation.
- The lesson TSX and notebook are well-aligned in terminology and mental models. Both use "restore-original-rows pattern," "geometry encodes meaning," "three knobs on three different parts of the pipeline," and "the U-Net never sees text."
- All Row wrapping is correct throughout the lesson.
- All `<details>` summary elements have `cursor-pointer`.
- The notebook exercises match the planning document's specification (4 exercises: Guided, Guided, Supported, Independent).

**No systemic issues identified in this iteration.**
