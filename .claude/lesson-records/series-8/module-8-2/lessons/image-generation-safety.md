# Lesson 1: How Image Generation Safety Works (image-generation-safety) -- Planning Document

**Module:** 8.2 Safety & Content Moderation
**Position:** Lesson 1 of 1 (standalone)
**Type:** BUILD
**Slug:** image-generation-safety

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| CLIP dual-encoder architecture (image encoder + text encoder producing aligned embeddings in shared space) | INTRODUCED | clip (6.3.3), deepened in siglip-2 | Student knows the architecture, the shared embedding space, and cosine similarity for matching. Critically, understands that CLIP embeddings encode semantic meaning -- two images of similar content land near each other in embedding space. This is the foundation for the safety checker. |
| Cosine similarity between embeddings | DEVELOPED | clip (6.3.3), siglip-2 | Student has extensive experience computing and interpreting cosine similarity in embedding spaces. Used in contrastive loss, zero-shot classification, and similarity matching. |
| Classifier-free guidance (CFG) -- training with conditional and unconditional denoising, inference-time interpolation to steer generation | DEVELOPED | cfg-and-text-conditioning (6.3.4) | Student has deep understanding of how CFG works: train with both conditional and unconditional noise predictions, at inference extrapolate away from unconditional toward conditional. The formula: noise_pred = uncond + guidance_scale * (cond - uncond). This is directly relevant to Safe Latent Diffusion and understanding negative prompts as soft safety. |
| Negative prompts in diffusion models | DEVELOPED | cfg-and-text-conditioning (6.3.4) | Student understands negative prompts modify the unconditional direction in CFG. Replacing the unconditional prediction with a "negative" prediction steers away from unwanted content. |
| Fine-tuning pretrained models (modifying model weights for new objectives) | DEVELOPED | Multiple (transfer-learning 3.2.3, lora 6.5, etc.) | Extensive experience with fine-tuning from transfer learning, LoRA, and alignment lessons. Student understands both full fine-tuning and parameter-efficient approaches. Critical for understanding ESD concept erasure. |
| Cross-attention as the gateway for text conditioning in diffusion models | DEVELOPED | unet-architecture (6.3.2), text-conditioning-and-guidance (6.3.4) | Student knows that cross-attention layers are where text embeddings enter the U-Net. Text embeddings become keys and values, spatial features become queries. This is directly relevant to understanding why ESD-x targets cross-attention for style erasure. |
| Text classification / sentiment analysis (training a classifier on text to predict categories) | INTRODUCED | Multiple (implicit from transformer/LLM lessons) | Student understands transformer-based text classification conceptually from the LLM series. Has not built a text classifier specifically, but knows the pattern: encode text -> classification head -> category prediction. |
| CNN-based image classification (using a CNN to classify images into categories) | DEVELOPED | Multiple (Series 3) | Extensive experience from the CNN series. Student can explain feature extraction, classification heads, training on labeled data. Foundation for understanding NSFW image classifiers. |
| LoRA (low-rank adaptation of model weights) | DEVELOPED | lora (6.5) | Student understands parameter-efficient fine-tuning via low-rank decomposition. Relevant context for understanding efficient concept erasure methods. |
| Denoising process in diffusion models (iterative noise prediction and removal) | DEVELOPED | Multiple (Series 6-7) | Deep understanding of the full diffusion pipeline: forward noising process, reverse denoising, noise prediction networks, sampling. Critical for understanding where inference-time safety interventions happen. |

**Mental models and analogies already established:**
- "Two encoders, one shared space" (CLIP -- images and text in the same geometric space)
- "Classifier-free guidance = extrapolating away from unconditional toward conditional" (the CFG formula)
- "Cross-attention is the gateway for text conditioning" (where text enters the U-Net)
- "Architecture encodes assumptions about data" (from CNN module, extended in SAM)
- "Hire experienced, train specific" (transfer learning / fine-tuning)
- Temperature as a "sharpness knob" for distributions
- The student deeply understands embedding spaces as geometric spaces where proximity = semantic similarity

**What was explicitly NOT covered in prior lessons (relevant here):**
- Safety or content moderation as a topic (never addressed in the course)
- NSFW classifiers or safety checkers
- Keyword blocklists or prompt filtering
- Concept erasure from model weights (ESD, UCE)
- Safe Latent Diffusion or inference-time safety guidance
- How production systems (DALL-E 3, Midjourney) implement safety
- The idea of a multi-layered defense stack
- Adversarial attacks on safety systems

**Readiness assessment:** The student is well-prepared. Every technique in this lesson is built on concepts the student already has at DEVELOPED depth: CLIP embeddings power the safety checker, CFG powers Safe Latent Diffusion, fine-tuning powers concept erasure, CNN classification powers NSFW detectors, and cross-attention understanding explains why ESD-x works. The challenge is not any single technique but the integration -- seeing how they compose into a layered defense. BUILD is the right designation: familiar tools applied to a new problem domain.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how production image generation systems prevent harmful content through a multi-layered safety stack -- from prompt-level text classifiers through inference-time guidance to post-generation CLIP-based image classifiers and model-level concept erasure -- and why each layer exists because no single approach is sufficient.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| CLIP shared embedding space + cosine similarity | DEVELOPED | DEVELOPED | clip (6.3.3), siglip-2 | OK | The Stable Diffusion safety checker IS a CLIP cosine similarity threshold. Student must deeply understand embedding proximity = semantic similarity. |
| Classifier-free guidance formula | DEVELOPED | DEVELOPED | cfg-and-text-conditioning (6.3.4) | OK | SLD directly modifies the CFG formula by adding a safety guidance term. Student must know the formula to understand the modification. |
| Negative prompts | INTRODUCED | DEVELOPED | cfg-and-text-conditioning (6.3.4) | OK | Negative prompts are the simplest inference-time safety tool. INTRODUCED is sufficient; student has DEVELOPED. |
| Cross-attention in the U-Net | INTRODUCED | DEVELOPED | unet-architecture (6.3.2) | OK | Understanding why ESD-x targets cross-attention layers requires knowing what cross-attention does. INTRODUCED is sufficient. |
| Fine-tuning model weights | INTRODUCED | DEVELOPED | Multiple | OK | ESD is a fine-tuning method. Student needs to understand modifying weights for new objectives. |
| CNN-based image classification | INTRODUCED | DEVELOPED | Series 3 | OK | NSFW classifiers are image classifiers. INTRODUCED is sufficient to understand the pattern. |
| Text classification pattern | INTRODUCED | INTRODUCED | Implicit from LLM series | OK | Prompt classifiers are text classifiers. Student understands the pattern. |

### Gap Resolution

No gaps. All prerequisites are met at or above required depth. The student's extensive background in diffusion models, CLIP, and fine-tuning provides a strong foundation. No recap section needed -- connections to prior knowledge will be made inline as each technique is introduced.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Safety = one NSFW classifier at the end" | Most visible safety experience is seeing a black box replace a generated image. The student likely assumes a single gatekeeper model. | DALL-E 3 uses at minimum 4 layers: GPT-4 prompt rewriting, keyword blocklists, a bespoke output classifier, and training data filtering. A prompt like "a photograph of a nude statue in a museum" might pass the output classifier (artistic context) but be caught by prompt filtering. A prompt like "a beautiful sunset" with adversarial tokens appended might pass prompt filtering but produce unsafe content caught by the output classifier. Different attacks bypass different layers. | Hook section -- motivate the multi-layered approach by showing a single layer failing. Return to this throughout as each new layer is introduced. |
| "Model-level erasure (ESD) makes all other safety layers unnecessary" | ESD modifies the model weights so it literally cannot generate certain content. Seems like the definitive solution -- why add layers on top of a model that has been "fixed"? | ESD-u erasing "nudity" also degrades images of people in swimwear, medical imagery, and classical art. The erasure is not surgically precise. Additionally, concept erasure has been shown to be reversible with adversarial fine-tuning (just 1000 steps can recover erased concepts). A production system cannot rely solely on model-level erasure because: (1) erasure has collateral damage, (2) erasure can be reversed, (3) new harmful concepts emerge that were not in the original erasure set. | After the model-level section. Address explicitly with a "why not just erase everything?" discussion. |
| "Keyword blocklists are too simple to be useful -- just use AI classifiers for everything" | Keyword matching feels like the duct-tape approach compared to embedding-based classifiers. A student with deep ML knowledge might dismiss it as naive. | Keyword blocklists have near-zero latency, 100% recall for exact matches, are trivially auditable (you can read the list), and require no GPU. An embedding classifier might miss a prompt that uses an obscure slang term not in its training data, but a blocklist updated with that term catches it immediately. Blocklists and classifiers catch different failure modes. DALL-E 3 uses blocklists ALONGSIDE GPT-4 analysis. | Early in the prompt-level section. Frame blocklists as "fast, cheap, and predictable" rather than "simple." |
| "Safety filtering only blocks sexual content" | The most visible safety interventions are NSFW-related (black boxes replacing images). Stable Diffusion's safety checker explicitly only checks for sexual content. | Production systems filter for violence, gore, self-harm, hate symbols, real public figures, copyrighted characters, drug manufacturing, weapons, and child safety. OpenAI's categories include at least 10 distinct harm types. The Stable Diffusion safety checker's sexual-only scope is a known limitation, not the intended design. | When discussing real production systems, explicitly list the full taxonomy of harm categories. |
| "These techniques are unbreakable -- if a system has safety filtering, it is safe" | Safety systems sound robust when described technically. Embedding classifiers, model-level erasure, multi-layered stacks -- surely they work? | Every layer has been bypassed: keyword blocklists via misspellings and Unicode substitution, embedding classifiers via adversarial prompt engineering (SurrogatePrompt achieves 70%+ bypass rates), the SD safety checker via adding random noise to latents, ESD via adversarial fine-tuning recovery. Safety is an ongoing arms race, not a solved problem. | Final section. Honest assessment of the state of the art. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **SD safety checker walkthrough:** An image is generated, CLIP encodes it, cosine similarity is computed against 17 concept embeddings, threshold check determines block/pass | Positive | Shows the most concrete, fully-open-source safety system. Student can trace every step with real architecture details. | The SD safety checker is the only fully public implementation. It demonstrates CLIP embeddings applied to safety in a way the student can trace end-to-end. Every component (CLIP encoder, cosine similarity, thresholds) maps to concepts they already know. |
| **ESD concept erasure:** The standard SD model generates "a painting in the style of Van Gogh" -> starry-night-like output. ESD-x model with "Van Gogh" erased generates a generic painting with no Van Gogh characteristics. Same prompt, different model weights. | Positive | Shows how fine-tuning can remove specific concepts from model weights. The style erasure case is clearer and more demonstrable than NSFW erasure. | Style erasure is a clean, visual demonstration of concept erasure that avoids needing to show NSFW content. It demonstrates ESD-x (cross-attention targeting) in a way that connects directly to the student's understanding of how text conditioning enters the U-Net via cross-attention. |
| **DALL-E 3 safety pipeline:** User asks ChatGPT to "generate a photo of [public figure] committing a crime." GPT-4 rewrites/refuses the prompt before DALL-E 3 ever sees it. Versus: a user asks for "a sunset" but appends adversarial tokens -- GPT-4 passes the rewritten prompt, but the output classifier catches the problematic image. | Positive | Demonstrates the multi-layered approach where different layers catch different attacks. Shows why you need both input and output filtering. | This is the "real production system" example. It shows the student that the techniques from the lesson are not theoretical -- they are how actual products work. The two attack vectors (explicit prompt vs adversarial tokens) demonstrate why multiple layers exist. |
| **Negative prompt as soft safety:** Adding "nudity, violence, gore" as a negative prompt to CFG. This reduces but does not eliminate unsafe content. An adversarial prompt can still produce harmful content despite negative prompts. | Negative | Shows why inference-time soft guidance is not sufficient as the only safety measure. Negative prompts reduce probability but do not guarantee safety. | The student already deeply understands negative prompts and CFG. This negative example grounds the distinction between "probabilistic steering" and "hard filtering." It also connects SLD (which IS effective) to simple negative prompts (which are insufficient), showing the spectrum. |
| **Blocklist bypass:** The word "nude" is on a blocklist. A user types "nud3" or uses Unicode lookalikes. The blocklist misses it. An embedding classifier catches it because "nud3" embeds close to "nude" in the learned space. But an even more creative circumlocution bypasses even the embedding classifier. | Negative | Shows why no single layer is sufficient and why defense in depth matters. | This is the core negative example for the entire lesson's thesis (multi-layered defense). It makes the arms race concrete and visceral, progressing through three layers of attack and defense. |

---

## Phase 3: Design

### Narrative Arc

You have spent the last several series learning to build incredibly powerful generative models -- models that can produce photorealistic images from any text description. But power without control is a liability. These models learned from internet-scale data, which means they absorbed everything: art, photography, medical images, but also violence, pornography, hate imagery, and content involving real people without consent. The moment you deploy one of these models, you face a question that is as much engineering as ethics: how do you actually prevent it from generating content it should not? The answer is not one clever technique -- it is a layered defense stack where each layer catches what the others miss. This lesson walks through each layer of that stack, from the simplest (keyword blocklists) to the most sophisticated (rewriting model weights to forget concepts entirely), using the specific tools you already understand: CLIP embeddings, classifier-free guidance, fine-tuning, and cross-attention. By the end, you will be able to look at any image generation system and identify what safety layers it uses, what they catch, and what they miss.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual (architecture diagram)** | A "defense stack" diagram showing the full pipeline: prompt enters -> Layer 1 (blocklist) -> Layer 2 (text classifier / LLM rewriter) -> Layer 3 (inference-time guidance: SLD / negative prompts) -> Layer 4 (post-generation image classifier) -> output. Model-level erasure shown as modifying the generator itself. Arrows show where each attack type gets caught. | The core concept is a SYSTEM with layers, not a single technique. A visual pipeline is the only way to show the relationships between layers and where different attacks are intercepted. Without this, the student might remember individual techniques but not their composition. |
| **Symbolic / Code** | PyTorch-style pseudocode for: (1) the SD safety checker (CLIP encode -> cosine similarity -> threshold), (2) the SLD guidance modification to the CFG formula, (3) the ESD training loop (frozen model provides scores, edited model trains to oppose them). | Each of these is a concrete algorithm the student can trace. Pseudocode connects the theory to implementation, matching the course's standard depth. The student has seen CLIP code, CFG code, and fine-tuning code -- these are extensions, not new patterns. |
| **Concrete example (traced computation)** | Walk through the SD safety checker with a specific image: image -> CLIP encoder -> 768-dim embedding -> compute cosine similarity against concept embedding #7 -> get 0.83 -> threshold is 0.78 -> FLAGGED. Then the same flow for a safe image: similarity 0.31 -> threshold 0.78 -> PASS. | The student has traced CLIP embeddings and cosine similarity before. Tracing the safety checker with real numbers makes it concrete: "this is literally CLIP cosine similarity with a threshold, nothing magical." Demystifies the safety checker. |
| **Verbal / Analogy** | "Defense in depth" analogy from security: a castle has walls, a moat, guards, and a locked keep. Each layer can be breached, but an attacker must breach ALL of them. No single layer is trusted alone. Extend to: keyword blocklist = the wall (obvious, easy to see over), embedding classifier = the moat (harder to cross but possible with a bridge), output classifier = the guards (catch what got past the moat), model erasure = removing the weapons from inside the castle entirely. | The student needs a mental model for why layered defense exists. The castle/security analogy is widely understood and maps cleanly to the four safety layers. It also naturally introduces the idea that each layer has a different failure mode. |
| **Intuitive** | "Of course" moments at: (1) "Of course the safety checker uses CLIP -- it is literally a zero-shot classifier, and classifying 'is this image unsafe?' is just zero-shot classification." (2) "Of course ESD-x targets cross-attention -- cross-attention is where text conditioning enters, so erasing the text-concept association means erasing from cross-attention." (3) "Of course you need multiple layers -- each one has a blind spot that another one covers." | These connect new material to existing deep understanding. Each "of course" reinforces that safety techniques are not new inventions but applications of tools the student already knows. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. **The safety stack as a design pattern** (layered defense with heterogeneous techniques) -- genuinely new organizational concept
  2. **Concept erasure via fine-tuning** (ESD/UCE -- using CFG-style guidance at training time to push model weights away from a concept) -- new application of familiar techniques
  3. **CLIP-based image safety classification** (using embedding similarity against fixed concept vectors as a binary classifier) -- new application of CLIP, but architecturally simple
- **Previous lesson load:** BUILD (SAM 3 had 2-3 new concepts)
- **Assessment:** BUILD is appropriate. Each "new" concept is really a new APPLICATION of something the student already deeply understands (CLIP, CFG, fine-tuning). The cognitive load is in seeing how they compose, not in learning new math or architecture. The student's deep CLIP and CFG background means these techniques will feel like natural extensions.

### Connections to Prior Concepts

| Prior Concept | Connection | How |
|---------------|-----------|-----|
| CLIP embeddings + cosine similarity | SD safety checker | "The safety checker IS zero-shot CLIP classification. Same encoder, same cosine similarity, just comparing against concept embeddings instead of text labels." |
| Classifier-free guidance formula | Safe Latent Diffusion | "SLD adds a third term to the CFG formula. Where CFG steers toward the prompt, SLD simultaneously steers away from a safety concept. Same extrapolation idea, additional direction." |
| Cross-attention as text conditioning gateway | ESD-x targeting | "ESD-x modifies cross-attention weights because that is where the model learns 'when the text says Van Gogh, produce swirly brushstrokes.' Erase the association in cross-attention, erase the concept." |
| Fine-tuning / LoRA | Concept erasure | "ESD is fine-tuning with a reversed objective. Instead of teaching the model a new concept, you teach it to un-learn one. Same optimizer, same weight updates, opposite direction." |
| Negative prompts | Soft safety baseline | "You already use negative prompts to steer away from unwanted content. Safety-guided diffusion is the same idea, formalized and strengthened." |
| Zero-shot classification (CLIP) | Prompt classifiers | "A prompt text classifier is doing what CLIP does for images -- encoding text and comparing against known categories. The pattern is identical." |

**Potentially misleading analogies:** The "fine-tuning in reverse" framing for ESD could mislead students into thinking you can just negate the loss. The actual mechanism uses the frozen model's predictions as a reference and trains the edited model to produce the unconditional prediction when given the concept-conditioned prompt. This needs to be stated precisely.

### Scope Boundaries

**This lesson IS about:**
- How each layer of the safety stack works technically (prompt filtering, inference guidance, image classification, model-level erasure)
- Why layered defense is necessary (each layer's failure modes)
- How real production systems compose these layers (DALL-E 3, Midjourney, Stability AI)
- The tradeoffs of each approach (latency, precision, recall, collateral damage)
- INTRODUCED depth for each technique's mechanism, DEVELOPED depth for the overall safety stack pattern

**This lesson is NOT about:**
- Building a complete safety system from scratch (exercise scope, not lesson scope)
- The ethics/politics of what should be filtered (this is an engineering lesson)
- Adversarial attack methods in detail (mentioned as motivation for defense-in-depth, not taught as techniques)
- Mathematical proofs of erasure completeness or classifier optimality
- Training data filtering / data curation (related but a separate topic)
- Watermarking or provenance (adjacent but distinct problem)

**Depth targets:**
- Safety stack as a pattern: DEVELOPED (the student should be able to identify layers and explain why each exists)
- Individual techniques (SD safety checker, SLD, ESD): INTRODUCED (understand the mechanism well enough to explain it, but not implement from scratch)
- CLIP-based safety classification: INTRODUCED -> nearly DEVELOPED (the student already has the building blocks; this is connecting them)

### Lesson Outline

#### 1. Context + Constraints
What this lesson is about: the engineering of safety systems for image generation. What we are NOT doing: debating what should be censored, building a production system, or teaching adversarial attacks. The student has built powerful generative models; this lesson is about the control layer that makes them deployable.

#### 2. Hook (Real-world impact)
Open with a concrete scenario: you have trained a Stable Diffusion model and want to deploy it as an API. Within hours of launch, users discover it can generate: photorealistic violence, NSFW content, images of real public figures in compromising scenarios, and copyrighted characters. Your model can do ALL of this because it learned from internet data. You need to stop it. What do you build?

Present the "one classifier at the end" strawman and show why it fails: a single output classifier can be bypassed (adversarial latents), has false positives (blocking medical imagery), and cannot prevent the model from wasting compute generating images that will just be blocked. Motivate the multi-layered approach.

Introduce the "defense in depth" analogy (castle walls, moat, guards, locked keep). Show the full safety stack diagram. This becomes the lesson's organizing visual -- each subsequent section fills in one layer of the diagram.

#### 3. Explain: Layer 1 -- Prompt-Level Filtering
**Three sub-techniques, increasing sophistication:**

**(a) Keyword Blocklists:** The simplest layer. A list of banned terms; if the prompt contains one, reject immediately. Near-zero latency, 100% recall for exact matches, trivially auditable. Limitations: easily bypassed (misspellings, Unicode substitutions, circumlocutions). Frame as "the wall -- obvious, easy to see over, but stops casual attempts."

**(b) Text Embedding Classifiers:** Encode the prompt with a text encoder (could be BERT-family, could be CLIP's text encoder). Compare the embedding against known unsafe concept embeddings, or pass through a classification head trained on safe/unsafe prompts. DiffGuard (DistilBERT/RoBERTa-based, 67-125M params, fine-tuned on 250K prompts, 94% F1). Catches semantic meaning that blocklists miss -- "nud3" embeds near "nude." Limitations: still text-only, so cannot catch adversarial token sequences that look safe as text but produce unsafe images.

**(c) LLM-Based Prompt Analysis (DALL-E 3 approach):** GPT-4 rewrites the user's prompt before DALL-E 3 sees it. The LLM understands intent, context, and subtlety that keyword and embedding classifiers miss. Can refuse, rewrite, or sanitize. Trade-off: highest latency, highest accuracy, most expensive. Frame as "the smartest guard -- understands language, but slow and costly."

**Pseudocode:** Show a simple prompt classifier pipeline (encode -> classify -> block/pass).

#### 4. Check: Predict-and-Verify
Present three prompts. Ask the student to predict which layer catches each:
- "Generate a nude woman" (keyword blocklist)
- "Generate a n.u" + "d.e woman" (embedding classifier catches semantic similarity)
- "Generate a renaissance painting featuring classical figure studies" (ambiguous -- LLM analyzer would need to assess artistic intent vs exploitation)

#### 5. Explain: Layer 2 -- Inference-Level Safety

**(a) Negative Prompts as Soft Safety:** The student already knows negative prompts modify the unconditional prediction in CFG. Using negative prompts like "nudity, violence, gore, nsfw" steers generation away from unsafe content. Simple, no extra models needed. But probabilistic -- reduces probability, does not guarantee safety. Frame as "turning down the volume, not muting the channel."

**(b) Safe Latent Diffusion (SLD):** Formalizes and strengthens negative-prompt safety. Adds a dedicated safety guidance term to the denoising step:

```
noise_pred = uncond + guidance_scale * (cond - uncond) - sld_scale * (safety - uncond)
```

Where `safety` is the noise prediction conditioned on a safety concept text (e.g., "nudity, violence, harm..."). The third term actively pushes away from unsafe content at each denoising step. Has warmup steps (SLD kicks in after step 10 to avoid disrupting early structure), momentum (accumulated safety signal), and configurable strength (WEAK/MEDIUM/STRONG/MAX).

**Pseudocode:** Show the modified denoising step with the SLD term, compared to standard CFG.

**Connection to prior knowledge:** "You already know the CFG formula: `uncond + scale * (cond - uncond)`. SLD just adds one more term pushing in the opposite direction of a safety concept. Same geometric idea -- vector arithmetic in noise-prediction space."

#### 6. Explore: Interactive Widget -- The Safety Stack Simulator

**Widget concept:** A visual pipeline simulator where the student can toggle individual safety layers on/off and see what gets through. The widget shows:

- A set of ~8-10 example prompts (ranging from clearly safe to clearly unsafe to adversarial/ambiguous)
- The full safety pipeline as a horizontal flow diagram with checkpoints
- Each layer (blocklist, text classifier, SLD, output classifier) can be toggled on/off
- When a prompt enters the pipeline, it progresses through each active layer with a visual indicator (green checkmark = passed, red X = caught, yellow = modified)
- Key insight: with all layers on, unsafe prompts are caught. Turn off any single layer, and specific prompts get through.

The widget should make viscerally obvious WHY layered defense matters -- the student can experiment with removing layers and see which prompts sneak through.

**Implementation note:** This does not need real model inference. The results can be pre-computed for the fixed set of example prompts. The visualization is what matters, not live classification.

#### 7. Explain: Layer 3 -- Post-Generation Image Classification

**(a) The Stable Diffusion Safety Checker:** The most important concrete example because it is fully open-source.

Architecture: Uses CLIP ViT-L/14 as the image encoder. The generated image is encoded into CLIP embedding space (768-dim vector). This embedding is compared via cosine similarity against 17 fixed concept embeddings (the concepts are obfuscated -- only their embedding vectors are public, not the text descriptions). Each concept has a threshold. If any cosine similarity exceeds its threshold, the image is flagged and replaced with a black rectangle.

Two tiers: `special_care_embeds` (lower thresholds, more sensitive) and regular `concept_embeds`. An `adjustment` parameter globally shifts all thresholds.

**Traced computation:** Walk through a specific example with real dimensions. Image [3, 512, 512] -> CLIP ViT-L/14 encoder -> embedding [768] -> cosine_similarity(embedding, concept_7) = 0.83 -> threshold_7 = 0.78 -> 0.83 > 0.78 -> FLAGGED. Same image with a safe subject: cosine_similarity = 0.31 -> PASS.

**Pseudocode:** Show the complete safety checker: encode, compute similarities, threshold check.

**Key insight:** "This is literally zero-shot CLIP classification. You have already done this -- comparing an embedding against reference embeddings and thresholding on cosine similarity. The safety checker is the simplest possible application of CLIP."

**(b) Dedicated NSFW classifiers:** Beyond CLIP-based checking, some systems use purpose-trained image classifiers (ResNet/InceptionV3-based, like NudeNet). These are standard image classification models trained on labeled safe/unsafe datasets. Higher accuracy for specific categories (nudity detection) but require labeled training data and do not generalize to new harm categories without retraining.

**Known limitations of output classifiers:** Only catch what they are trained on (SD safety checker only checks sexual content, not violence). Can be fooled by adversarial perturbations. Add latency (CLIP encoding is not free). Generate false positives (medical imagery, classical art).

#### 8. Check: Explain-It-Back
Ask the student: "The SD safety checker uses 17 concept embeddings but their text descriptions are hidden. Why would you obfuscate the concepts? What would happen if the concept texts were public?"

Expected insight: Publishing the exact concepts would make it trivial to craft adversarial prompts that land just below the threshold for each concept, or to specifically target the gaps between concepts. Obfuscation is a form of security through obscurity -- imperfect, but it raises the cost of targeted attacks.

#### 9. Explain: Layer 4 -- Model-Level Concept Erasure

**The idea:** Instead of catching unsafe content after generation, remove the model's ability to generate it at all. Modify the model weights so that the concept of "nudity" (or "Van Gogh's style" or "a specific person's face") is no longer representable.

**(a) Erased Stable Diffusion (ESD):**

Core mechanism: Use classifier-free guidance logic at TRAINING time. The frozen pretrained model provides both conditional (concept-present) and unconditional noise predictions. The edited model is trained to produce, when given the concept prompt, a prediction that would GUIDE AWAY from the concept. Essentially: "when asked to generate [concept], generate the opposite of what you would have generated."

Two variants:
- **ESD-x:** Fine-tunes only cross-attention layers. Erases text-to-concept associations. Good for style removal ("Van Gogh") because the style IS a text-conditioned concept. Targeted, minimal collateral damage.
- **ESD-u:** Fine-tunes all non-cross-attention layers. Erases concepts globally, even without explicit text triggers. Better for NSFW removal because unsafe content can emerge without explicit prompting (e.g., from certain style descriptions). Broader erasure, more collateral damage.

**Connection to prior knowledge:** "ESD-x targets cross-attention because you know that cross-attention is where text conditioning enters the U-Net. The model learns 'Van Gogh' = swirly brushstrokes through cross-attention. Erase those weights, erase that association. ESD-u targets everything else because some concepts are not text-triggered -- they emerge from the model's learned distribution regardless of prompt."

**(b) Unified Concept Editing (UCE):** A closed-form solution (no gradient-based training) that edits cross-attention weights directly. Can debias, erase, and moderate multiple concepts simultaneously. Faster than ESD (no fine-tuning loop), but limited to cross-attention modifications.

**Tradeoffs of model-level erasure:**
- Collateral damage: erasing "nudity" also degrades medical imagery, swimwear, classical art
- Reversibility: adversarial fine-tuning can recover erased concepts (~1000 steps)
- Coverage: must anticipate every harmful concept in advance; new concepts require re-erasure
- Irreversibility of deployment: once you ship an erased model, you cannot add the concept back for legitimate uses

#### 10. Elaborate: How Real Systems Compose the Stack

**DALL-E 3 (OpenAI):** Keyword blocklists + GPT-4 prompt rewriting/refusal + bespoke CLIP-based output classifier + training data filtering. The GPT-4 layer is the most sophisticated -- it understands context, intent, and subtlety.

**Midjourney:** AI-driven content moderation that evaluates prompts holistically (not just keywords). Post-generation image analysis. Community reporting. Dynamic, regularly updated filter. No public architecture details.

**Stability AI (Stable Diffusion):** Open-source safety checker (CLIP-based, 17 concepts). SD 2.0 added training data filtering (removed NSFW from training set). Community-built tools (NudeNet, custom classifiers). Open-source means users CAN disable safety -- the safety stack is advisory, not enforced.

**Key insight:** Closed-source systems (DALL-E, Midjourney) can enforce all layers because they control the full pipeline. Open-source systems (SD) can only include layers as defaults that users can disable. This is a fundamental architectural constraint, not just a policy choice.

#### 11. Check: Transfer Question
"A startup is building a children's illustration generator. They want to use an open-source diffusion model. Design their safety stack -- which layers would you include and why? What is the biggest risk they should worry about?"

Expected reasoning: All four layers are needed. Extra-conservative thresholds on the output classifier. Consider ESD-u for the broadest concept erasure despite collateral damage (children's illustrations do not need anatomical accuracy). Biggest risk: the model generating subtly inappropriate content that passes all automated layers but is harmful in context (e.g., inappropriately sexualized poses that no single classifier catches). Human review layer may be necessary.

#### 12. Summarize
Echo the defense-in-depth mental model. Each layer of the safety stack catches what the others miss:
- **Prompt filtering** catches intent before any compute is spent
- **Inference guidance** steers the generation process itself
- **Output classification** catches what the model produced regardless of intent
- **Model erasure** removes the capability entirely from the weights

No single layer is sufficient. Safety is an ongoing engineering challenge, not a one-time solution. The same tools the student already knows (CLIP, CFG, fine-tuning, cross-attention) are the building blocks of every technique in the stack.

#### 13. Next Step
Mention that this lesson focused on the engineering of safety -- how to build these systems. Adjacent topics worth exploring: training data curation and filtering, watermarking and provenance tracking, adversarial robustness of safety classifiers, and the emerging area of constitutional approaches to generative model alignment (connecting back to constitutional AI from Series 5).

---

## Review -- 2026-02-22 (Iteration 1/3)

### Summary
- Critical: 2
- Improvement: 4
- Polish: 3

### Verdict: MAJOR REVISION

Critical findings require fixes before the lesson is usable. The lesson is well-structured and has strong narrative flow, but two issues would leave the student genuinely confused or holding a wrong mental model.

### Findings

### [CRITICAL] -- Check sections reveal answers immediately without hiding them

**Location:** Section 5 (Check: Which Layer Catches Which?), Section 9 (Check: Why Obfuscate the Concepts?), Section 12 (Check: Design a Safety Stack)
**Issue:** All three check sections present the answer directly below the question in a GradientCard. There is no `<details>` collapse, no click-to-reveal, no separation. The student reads the question and the answer simultaneously. The planning document explicitly calls these "Predict-and-Verify," "Explain-It-Back," and "Transfer Question" -- all of which require the student to THINK before seeing the answer. As built, these are not checks; they are just more exposition.
**Student impact:** The student never actually retrieves or applies knowledge. They read the question, their eye immediately falls to the bold "Answer:" text, and they passively absorb the explanation. The entire pedagogical purpose of the checks (active recall, self-assessment, transfer) is lost. This is especially damaging for the transfer question (children's illustration generator), which is the lesson's highest-order thinking exercise.
**Suggested fix:** Wrap each answer in a collapsible `<details><summary>` element, or use whatever reveal pattern the codebase has (e.g., a "Show answer" button). The question should be the only visible element until the student actively chooses to see the answer. At minimum, add a clear instruction like "Think about this before revealing the answer" and a visual break.

### [CRITICAL] -- Fifth misconception ("these techniques are unbreakable") not addressed

**Location:** Entire lesson
**Issue:** The planning document identifies five misconceptions. Misconception #5 -- "These techniques are unbreakable -- if a system has safety filtering, it is safe" -- is supposed to be addressed in a "final section" with an "honest assessment of the state of the art." The planned location says: "Every layer has been bypassed: keyword blocklists via misspellings and Unicode substitution, embedding classifiers via adversarial prompt engineering (SurrogatePrompt achieves 70%+ bypass rates), the SD safety checker via adding random noise to latents, ESD via adversarial fine-tuning recovery. Safety is an ongoing arms race, not a solved problem." However, the built lesson has no dedicated section addressing this. There are scattered mentions (reversibility of ESD, limitations of output classifiers, the aside about "arms race thinking"), but no systematic treatment. The summary's last bullet touches on it ("Every layer has been bypassed") but does not provide the concrete evidence the planning document specifies (SurrogatePrompt bypass rates, specific attack vectors per layer).
**Student impact:** The student could leave the lesson thinking that a well-designed safety stack is essentially bulletproof. The lesson teaches four layers of defense and shows them working in the widget -- the impression is of robustness. Without a concrete, evidence-backed section on how EACH layer has been bypassed in practice, the student forms an overconfident mental model. This is particularly dangerous for a safety-focused lesson.
**Suggested fix:** Add a dedicated section (between the "Real Systems" section and the final check, or between the final check and the summary) titled something like "The Arms Race: Every Layer Has Been Bypassed." Walk through specific attack vectors per layer with concrete numbers (SurrogatePrompt 70%+ bypass, ESD recovery in ~1000 fine-tuning steps, adversarial perturbations fooling CLIP-based classifiers). This directly fulfills the planning document's specification and corrects the overconfidence risk.

### [IMPROVEMENT] -- Ambiguous prompts in the widget treat SLD "modified" inconsistently with outcome logic

**Location:** SafetyStackSimulator widget -- `getFinalOutcome` function and scenario data
**Issue:** The `getFinalOutcome` function only checks for `'caught'` results to determine if a prompt is blocked. Results of `'modified'` are treated the same as `'pass'` for outcome determination. This means that the "museum statue" scenario (category: `ambiguous`) with the blocklist as the only layer returning `'caught'` correctly blocks when the blocklist is on. But when the blocklist is off, the text classifier returns `'modified'` (not `'caught'`), so the prompt passes through all remaining layers and gets delivered as "safe." However, the student might expect `'modified'` to mean "partially blocked" or "flagged for review." The distinction between `'modified'` and `'caught'` vs `'pass'` is never explained in the widget or the lesson text.
**Student impact:** When experimenting with the widget (which the lesson encourages), the student toggles layers and sees "MODIFIED" badges on some layers but the prompt still gets delivered as "Safe." This creates confusion: "Wait, the text classifier modified this prompt, but it still got through? What does 'modified' mean?" The concept of modification (like SLD steering the generation or a text classifier adjusting the prompt) vs hard blocking is a meaningful distinction that the lesson discusses for SLD but the widget never explains.
**Suggested fix:** Add a brief legend or tooltip to the widget explaining the three result states (PASS = no action, MODIFIED = content steered/adjusted but not blocked, BLOCKED = rejected). Alternatively, add a one-line explanation in the lesson text before the widget.

### [IMPROVEMENT] -- SLD warmup step direction appears inverted in pseudocode

**Location:** Section 6 (SLD pseudocode), line `if t < (T - warmup_steps):`
**Issue:** In diffusion models, timestep `t` typically counts DOWN from T (maximum noise) to 0 (clean image). The first ~10 steps are at HIGH t values (near T). The comment says "active after warmup" and the code says `t < (T - warmup_steps)`, which would be active for most of the generation (when t is less than, say, 990 out of 1000). This IS correct behavior (SLD is active for MOST steps, only skipping the first few high-noise steps), but the code structure is confusing. The variable naming and conditional direction make it look like SLD activates only at the very end. The lesson text says "kicks in after step ~10 to avoid disrupting early structure" -- "after step ~10" is ambiguous about whether we mean the 10th step of generation (t near T) or timestep value 10 (t near 0). The pseudocode and prose are technically consistent but could easily mislead a student who thinks "step 10" means "when t=10."
**Student impact:** A student reading the pseudocode would need to mentally track the direction of timestep counting, map "warmup steps" to "skip the first N steps of generation," and verify that `t < (T - warmup_steps)` means "after the first warmup_steps denoising steps." This is tractable but unnecessarily confusing for a lesson that is supposed to be BUILD-level (applying familiar concepts to a new domain). The student might think SLD only applies at the END of generation, which would be the opposite of the actual behavior.
**Suggested fix:** Add a comment to the pseudocode clarifying the timestep direction: `# t counts down from T to 0; warmup skips the first few noisy steps`. Alternatively, rewrite the condition with a more descriptive variable like `steps_completed = T - t` and `if steps_completed > warmup_steps:`.

### [IMPROVEMENT] -- The "museum statue" scenario reveals a false positive problem but the lesson never discusses it

**Location:** Widget scenario `museum-statue` and the lesson text generally
**Issue:** The widget includes a "museum statue" scenario where the keyword blocklist CATCHES a legitimate artistic prompt (false positive). This is a crucial aspect of safety systems -- false positives block legitimate content and degrade user experience. The planning document mentions false positives briefly under "Known Limitations of Output Classifiers" and in the misconception about blocklists, but neither the lesson text nor the widget draws explicit attention to the false positive problem as a design tension. The widget data demonstrates the issue, but the student has no guidance to notice or think about it.
**Student impact:** The student might use the widget, see the museum statue get blocked by the blocklist, and think "well, that is just how it works." They miss the deeper lesson: safety systems face a precision-recall tradeoff, and aggressive filtering blocks legitimate content. This is a core engineering insight for anyone building safety systems.
**Suggested fix:** Add a "TryThis" prompt in the widget aside (or a dedicated paragraph) that explicitly asks: "Notice that the museum statue prompt gets blocked by the keyword blocklist. This is a false positive -- legitimate content caught by an overly aggressive filter. What is the tradeoff between catching more unsafe content and blocking more safe content? How would you tune this for different applications?" This would make the widget's data pedagogically active rather than passive.

### [IMPROVEMENT] -- Missing explicit negative example for "concept erasure makes other layers unnecessary"

**Location:** Section 10 (Layer 4: Model-Level Concept Erasure) and the "Why Not Just Erase Everything?" subsection
**Issue:** The planning document identifies misconception #2: "Model-level erasure (ESD) makes all other safety layers unnecessary." It specifies a negative example: "ESD-u erasing 'nudity' also degrades images of people in swimwear, medical imagery, and classical art." The lesson DOES address this with four GradientCards listing problems (Collateral Damage, Reversibility, Coverage Gaps, Irreversibility of Deployment). However, the treatment is abstract -- it lists the problem categories without a concrete, traced-through negative example that demonstrates the failure. There is no specific "here is a prompt that works on the erased model because..." scenario. Compare this to the positive examples (Van Gogh style erasure), which are concrete and traceable.
**Student impact:** The student reads the four problems as a bullet list of caveats rather than experiencing a visceral demonstration of WHY erasure alone fails. The planning document's negative example (erasing "nudity" degrades swimwear/medical imagery) would be much more convincing as a concrete comparison (e.g., "Before ESD: 'a medical textbook diagram of human anatomy' produces clear educational content. After ESD-u erasing nudity: the same prompt produces distorted, unusable output").
**Suggested fix:** Add a concrete ComparisonRow or GradientCard showing a specific prompt that demonstrates collateral damage from concept erasure. Before/after ESD-u, showing how a legitimate use case (medical imagery, classical art) is degraded. This converts the abstract caveat into a concrete negative example.

### [POLISH] -- Missing cursor-pointer on interactive buttons in the widget

**Location:** SafetyStackSimulator widget -- `LayerToggle` and `PromptRow` components
**Issue:** Both `LayerToggle` and `PromptRow` use `<button>` elements which are interactive, but neither specifies `cursor-pointer` in the className. While browsers typically show a pointer cursor on `<button>` elements by default, Tailwind's preflight CSS resets the cursor on buttons to `cursor-default`. If the project uses Tailwind preflight (which it likely does), these buttons would show a default cursor instead of a pointer, making them look non-interactive.
**Student impact:** Minor -- the student might not immediately realize the layer toggles and prompt rows are clickable if the cursor does not change on hover. The toggle state indicators (filled circles, border highlights) provide some affordance, but cursor feedback is the primary interaction signal.
**Suggested fix:** Add `cursor-pointer` to both `LayerToggle`'s `<button>` className and `PromptRow`'s `<button>` className.

### [POLISH] -- Widget detail panel does not handle the "no layers active" edge case pedagogically

**Location:** SafetyStackSimulator widget -- when all layers are toggled off
**Issue:** When all four layers are toggled off, every unsafe prompt shows "LEAKED THROUGH" and the status banner says "N unsafe prompts leaked through your safety stack." This is technically correct but misses a pedagogical opportunity. The state "no layers active" is the baseline -- this IS the lesson's starting problem (deploying a model with no safety). A brief message like "This is what happens with no safety layers: every unsafe prompt gets through" would explicitly connect the widget to the lesson's hook.
**Student impact:** Minimal -- the student can figure this out. But the widget could be more pedagogically active by explicitly narrating the "zero defense" state.
**Suggested fix:** When all layers are off, show a special banner: "No safety layers active -- this is an unprotected model. Every prompt reaches the generator and every output is delivered."

### [POLISH] -- Summary block has 7 items (slightly heavy)

**Location:** Section 13 (Summary)
**Issue:** The SummaryBlock contains 7 items. While each item is individually clear, 7 summary points is on the heavier side for a lesson summary. The planning document's outline specifies 4 core points (one per layer) plus the meta-point about tools the student already knows. The built lesson adds a "no single layer is sufficient" and "safety is an ongoing challenge" item, bringing the total to 7. The first item (multi-layered defense stack) and the last item (ongoing challenge) somewhat overlap with item 6 (every technique is built on tools you already understand) in serving as "meta" observations.
**Student impact:** Minor -- the student skims a long summary rather than absorbing 4-5 crisp takeaways. No confusion, just diluted impact.
**Suggested fix:** Consider consolidating items 1 and 7 into a single framing point ("Image generation safety requires a multi-layered defense stack, and it is an ongoing engineering challenge -- not a one-time deployment"), reducing the total to 6. Alternatively, keep all 7 but visually group them: items 2-5 are per-layer summaries, items 1/6/7 are meta-insights.

### Review Notes

**What works well:**
- The narrative arc is strong. The lesson opens with a concrete, compelling problem (you deployed a model, users are exploiting it) and builds systematically through the four layers. The "defense in depth" castle analogy is used consistently and effectively.
- Every technique is explicitly connected to prior knowledge the student has. The "Of Course" moments are well-placed and genuinely connect safety techniques to CLIP, CFG, fine-tuning, and cross-attention.
- The Mermaid architecture diagram at the top serves as an effective organizing visual that the lesson fills in section by section.
- The interactive widget is well-designed with 8 diverse scenarios covering safe, unsafe, adversarial, and ambiguous prompts. The pre-computed results are plausible and pedagogically useful.
- The lesson respects its scope boundaries -- it stays focused on engineering rather than drifting into ethics, and it does not attempt to teach adversarial attacks.
- Code examples (pseudocode) are clear, well-commented, and at the right level of abstraction.
- The ESD explanation with two variants (ESD-x targeting cross-attention, ESD-u targeting everything else) is well-motivated and connects directly to the student's understanding of where text conditioning enters the U-Net.

**Systemic pattern:**
The most significant pattern is the check sections lacking answer hiding. This is a build pattern issue that may affect other lessons too -- wherever GradientCards are used for Q&A, the answer needs to be hidden behind a reveal mechanism. The lesson has three checks, and all three suffer from this same issue.

**Overall assessment:**
The lesson is close to ready. The content is accurate, well-structured, and deeply connected to prior knowledge. The two critical issues (unhidden check answers and missing misconception #5 treatment) are concrete and fixable. The improvement findings would each noticeably strengthen the lesson but are not blockers. Fix the criticals, address the improvements, and this lesson should pass on the next review.

---

## Review -- 2026-02-22 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings. Both critical issues from iteration 1 have been properly resolved. Two improvement findings remain -- one carried from iteration 1 (partially addressed but not fully resolved) and one new issue introduced by the fixes. Two minor polish items.

### Fix Verification (Iteration 1 Findings)

**[CRITICAL] Check sections reveal answers immediately -- RESOLVED.** All three check sections (Section 5: "Which Layer Catches Which?", Section 9: "Why Obfuscate the Concepts?", Section 12: "Design a Safety Stack") now use `<details><summary>Show answer</summary>` with a styled reveal pattern. The summaries have `cursor-pointer` and a hover transition. Answers are hidden behind a border-top separator within the details block. The student must actively click to see the answer. This is a clean fix.

**[CRITICAL] Fifth misconception ("these techniques are unbreakable") not addressed -- RESOLVED.** A dedicated "The Arms Race: Every Layer Has Been Bypassed" section (Section 11b) has been added between the "Real Systems" section and the final check. It includes four GradientCards, one per layer, each with a concrete "Bypassed via:" explanation. Specific evidence is present: SurrogatePrompt 70%+ bypass rates for text classifiers, ~1000 fine-tuning steps for ESD recovery, adversarial latent perturbations for output classifiers, and misspelling/Unicode bypasses for blocklists. The closing paragraph frames safety as an ongoing arms race. This directly fulfills the planning document's specification.

**[IMPROVEMENT] Widget legend for modified/blocked/pass states -- RESOLVED.** A three-state legend has been added to the widget (lines 548-561 of SafetyStackSimulator.tsx) showing colored swatches with labels: PASS (no action taken), MODIFIED (content steered but not blocked), BLOCKED (rejected). Clear and well-positioned.

**[IMPROVEMENT] SLD warmup pseudocode clarified -- RESOLVED.** The pseudocode now uses a `steps_completed = T - t` variable and the condition `if steps_completed > warmup_steps:`. Two clarifying comments are added: `# t counts DOWN from T (pure noise) to 0 (clean image).` and `# steps_completed counts UP from 0 (start) to T (done).` This eliminates the confusion about timestep direction.

**[IMPROVEMENT] False positive problem surfaced -- RESOLVED.** The TryThisBlock in the widget aside (lines 538-551 of the lesson) now explicitly calls out the museum statue false positive, asks the student to notice it, defines what a false positive is, and poses the precision-recall tradeoff question with a concrete application comparison (medical imaging platform vs children's app). This makes the widget data pedagogically active.

**[IMPROVEMENT] Collateral damage example for concept erasure -- RESOLVED.** A concrete ComparisonRow has been added (lines 957-976 of the lesson) showing "Before ESD-u" (clear anatomical illustration, appropriate for educational context) vs "After ESD-u (nudity erased)" (distorted, unusable output, anatomical features degraded). This converts the abstract caveat into a visceral negative example that the planning document specified.

**[POLISH] cursor-pointer on widget buttons -- RESOLVED.** Both `LayerToggle` (line 308) and `PromptRow` (line 345) now have `cursor-pointer` in their className.

**[POLISH] "No layers active" edge case -- RESOLVED.** When all layers are off, the widget now shows: "No safety layers active -- this is an unprotected model. Every prompt reaches the generator and every output is delivered." (lines 564-570). This explicitly connects the zero-defense state to the lesson's hook.

**[POLISH] Summary block has 7 items -- NOT ADDRESSED.** The summary still has 7 items. This was the lowest-priority finding and is acceptable to carry forward. See Polish finding below.

### Findings

### [IMPROVEMENT] -- The Arms Race section duplicates bypass information already presented in earlier sections

**Location:** Section 11b ("The Arms Race: Every Layer Has Been Bypassed") and Sections 4-10
**Issue:** The new Arms Race section addresses misconception #5 effectively, but some of its content now duplicates what the lesson already states in the individual layer sections. Specifically: (1) keyword blocklist bypasses via misspellings and Unicode substitution are already demonstrated in the hook section (line 224-226), the blocklist PhaseCard (line 224-226), and the widget scenarios. (2) ESD reversibility via fine-tuning in ~1000 steps is already stated in the "Reversibility" GradientCard (line 935-936). (3) Output classifier adversarial vulnerability is already mentioned in the "Known Limitations" WarningBlock (line 693). The Arms Race section repeats this information rather than adding new evidence beyond what was already scattered through the lesson.
**Student impact:** The student reads about blocklist bypasses three times (hook, Layer 1, Arms Race), about ESD reversibility twice (Layer 4 tradeoffs, Arms Race), and about classifier adversarial vulnerability twice (Layer 3 limitations, Arms Race). While some repetition reinforces, this level of redundancy may feel like padding. The student thinks "I already knew this -- didn't the lesson just tell me this?" The section's value should be in the SYSTEMATIC framing (every layer has been bypassed, this is an arms race) and any NEW evidence, not in re-stating facts already covered. The SurrogatePrompt reference (70%+ bypass rates for text classifiers) IS genuinely new -- the other three cards mostly repeat.
**Suggested fix:** Trim the Arms Race GradientCards to focus on what is NEW or not yet covered. For blocklists and output classifiers, a brief reference back ("As we saw earlier, blocklists are bypassed by...") followed by the new framing. Reserve the detailed "Bypassed via:" treatment for the two layers that do not already have detailed bypass discussion: text embedding classifiers (SurrogatePrompt) and perhaps a more specific output classifier bypass technique (the random noise addition to latents is mentioned but not well-explained). The section's primary contribution should be the FRAMING (arms race, not solved) and the SYSTEMATIC view (every single layer), not repeating the individual bypass details.

### [IMPROVEMENT] -- TryThisBlock items are dense and risk the student skipping the last bullet

**Location:** Section 7 (widget aside), TryThisBlock (lines 538-551)
**Issue:** The TryThisBlock in the widget aside has 5 bullet points. The first four are short, punchy experiment prompts (turn off blocklist, leave only output classifier, etc.). The fifth is a full paragraph -- roughly 4x longer than the others -- covering the false positive problem, defining false positives, posing the precision-recall tradeoff, and asking how you would tune differently for a medical platform vs a children's app. While the content is excellent (this was the fix for iteration 1's false positive finding), the sheer length of the fifth bullet relative to the others creates a visual imbalance. In a TryThisBlock where a student is scanning quickly for things to try, they are likely to read the first few short bullets, try the experiments, and never read the dense final bullet.
**Student impact:** The false positive insight -- which is one of the lesson's most important engineering takeaways -- is buried as the last and longest item in a list. The student who scans and skips misses the precision-recall tradeoff entirely. The student who reads it gets a great prompt but may feel overwhelmed by the wall of text in what should be a "try this" sidebar.
**Suggested fix:** Split the fifth bullet into two parts. Keep a short prompt in the TryThisBlock ("Notice the museum statue prompt gets **blocked** by the keyword blocklist -- that is a **false positive**. What is the tradeoff?"). Move the detailed elaboration (medical platform vs children's app comparison) either into a separate aside below the widget or into the lesson body text after the widget. Alternatively, promote the false positive discussion to a brief dedicated paragraph in the main content area before the widget, so the student encounters it as lesson content rather than a sidebar experiment.

### [POLISH] -- Summary block still has 7 items

**Location:** Section 13 (Summary)
**Issue:** Carried from iteration 1. The summary has 7 items where 5-6 would be tighter. Now that the Arms Race section explicitly covers the "safety is an ongoing challenge" point, the 7th summary item ("Safety is an ongoing engineering challenge, not a one-time deployment") is well-supported by lesson content, reducing the concern slightly. But 7 items is still on the heavier side.
**Student impact:** Minimal. The summary works; it is just slightly diluted.
**Suggested fix:** Same as iteration 1 -- consider merging items 1 and 7 into a single framing point. Not a blocker.

### [POLISH] -- Em dash spacing inconsistency in the Arms Race closing paragraph

**Location:** Section 11b, closing paragraph (line 1098)
**Issue:** The closing paragraph of the Arms Race section contains `<strong> ongoing engineering effort</strong>` with a leading space inside the `<strong>` tag. While this does not produce a visible em dash spacing issue, it creates an extra space before "ongoing" that renders as a slightly wider gap than intended. The pattern `an <strong> ongoing` produces a double space visually (one from the space after "an" and one from the space before "ongoing" inside the tag).
**Student impact:** Negligible. A very minor visual artifact.
**Suggested fix:** Remove the leading space inside the `<strong>` tag: `<strong>ongoing engineering effort</strong>`.

### Review Notes

**What improved since iteration 1:**
- The check sections now function as actual checks. The details/summary pattern forces active recall. This was the most important fix and it is well-executed.
- The Arms Race section is a substantial, well-organized addition that directly addresses the overconfidence risk. The four GradientCards with "Bypassed via:" headers are a clean structure. The SurrogatePrompt citation adds credibility.
- The widget is now significantly more self-explanatory with the legend and the "no layers active" banner.
- The SLD pseudocode is now unambiguous about timestep direction.
- The collateral damage ComparisonRow for ESD-u is a strong concrete negative example that grounds the abstract caveat.
- The false positive discussion in the TryThisBlock makes the widget data pedagogically active.
- The cursor-pointer fix makes the widget feel immediately interactive.

**What the fixes introduced:**
- The main new issue is content redundancy in the Arms Race section. The section serves its purpose (systematic framing of the arms race, addressing misconception #5) but duplicates ~60% of its content from earlier sections. This is not critical -- the framing alone justifies the section -- but trimming the duplicated content would make it sharper.
- The TryThisBlock density is a layout/pacing issue rather than a content issue. The false positive content is exactly right; it is just placed where it risks being skipped.

**Overall assessment:**
The lesson has improved substantially from iteration 1 to iteration 2. Both critical issues are resolved. The remaining findings are genuine improvements that would sharpen the lesson, but neither blocks learning. The Arms Race redundancy is the most significant remaining issue -- not because it is wrong, but because it could be tighter. The TryThisBlock density is a presentation issue with an easy fix. If these two improvements are addressed, the lesson should pass on iteration 3.

---

## Review -- 2026-02-22 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

No critical or improvement findings. The lesson is ready to ship.

### Fix Verification (Iteration 2 Findings)

**[IMPROVEMENT] Arms Race section duplicates bypass information -- RESOLVED.** The four individual GradientCards with "Bypassed via:" headers have been replaced with a single bulleted list (lines 1059-1064) that uses backreferences for the two layers already discussed ("as we saw in Layer 1," "as we saw in Layer 4") and reserves detailed treatment for the two layers that were NOT already discussed in depth: text classifiers (SurrogatePrompt 70%+ bypass rates -- genuinely new information) and output classifiers (CLIP embedding shift via crafted latent noise -- brief and non-duplicative). The section's primary contribution is now the FRAMING (arms race, ongoing effort, not bulletproof) and the SYSTEMATIC view (every layer has been bypassed), not repeating individual bypass details. The closing paragraph is clean, with the `<strong>` tag spacing issue from iteration 2 also fixed (`<strong>ongoing engineering effort</strong>` has no leading space). This is a much tighter section that serves its purpose without redundancy.

**[IMPROVEMENT] TryThisBlock items are dense -- RESOLVED.** The fix was well-executed: the detailed false positive elaboration (medical imaging platform vs children's app, precision-recall tradeoff explanation) has been moved OUT of the TryThisBlock and INTO the main content paragraph above the widget (lines 528-537). The TryThisBlock's fifth bullet (line 549) is now short and punchy: "Notice: the 'museum statue' gets **blocked** by the blocklist -- a **false positive**. What is the tradeoff?" This matches the length and tone of the other four bullets. The student encounters the detailed precision-recall discussion in the lesson body text (where they are reading carefully) and the TryThisBlock serves as a quick experiment prompt (where they are scanning for things to try). The pedagogical content is preserved; it is just better-placed.

**[POLISH] Summary block still has 7 items -- NOT ADDRESSED (carried).** See Polish finding below. Accepted as-is.

**[POLISH] Em dash spacing / `<strong>` tag spacing in Arms Race -- RESOLVED.** The Arms Race section was rewritten; the closing paragraph now reads `<strong>ongoing engineering effort</strong>` with no leading space (line 1068). Clean.

### Findings

### [POLISH] -- Summary block has 7 items (carried from iterations 1 and 2)

**Location:** Section 13 (Summary)
**Issue:** Carried across all three iterations. The summary has 7 items where 5-6 would be tighter. The content of each item is accurate and well-phrased. Items 2-5 are per-layer summaries, item 6 connects everything to prior knowledge, item 7 frames safety as ongoing. Items 1 and 7 both serve as meta-framing and could be consolidated. However, this has been reviewed twice and deliberately left as-is -- the 7 items are coherent and the summary is not confusing, just slightly heavy.
**Student impact:** Minimal. The student skims a slightly long summary rather than absorbing 5 crisp takeaways. No confusion or wrong mental models result.
**Suggested fix:** Optionally merge items 1 and 7 into a single framing point. Not a blocker. Acceptable as shipped.

### Review Notes

**Step 1 (Read as a Student) -- Summary:**

Student entering this lesson has DEVELOPED understanding of CLIP embeddings, cosine similarity, CFG, negative prompts, cross-attention, fine-tuning, and CNN-based image classification from modules 6-8. Reading sequentially:

- The hook lands well. The deployment scenario is concrete and the "one classifier at the end" strawman is immediately debunked with four specific failure modes. The student feels the problem before any solution is presented.
- Layer 1 (prompt filtering) progresses cleanly from simple (blocklist) to sophisticated (LLM analysis). The PhaseCard numbering creates a clear progression. The aside warning ("Don't Dismiss Blocklists") catches the likely dismissal from an ML-savvy student.
- The first check section works now. The three prompts are well-graded in difficulty, and the hidden answers force actual prediction. The "renaissance painting" ambiguity is particularly good -- it has no clean answer.
- Layer 2 (inference-time safety) connects cleanly to CFG. The SLD formula is presented as a one-term extension of the CFG formula the student already knows. The pseudocode is now unambiguous about timestep direction.
- The widget section flows naturally from the preceding content. The false positive discussion now appears in the body text before the widget, so the student encounters it while reading carefully. The TryThisBlock is now all short, scannable experiment prompts. The widget itself is well-designed with clear legends and useful feedback states.
- Layer 3 (output classification) is the strongest section. The SD safety checker walkthrough is the most concrete example in the lesson -- real architecture details, traced computation with specific numbers, and the "Of Course" moment connecting it to CLIP zero-shot classification.
- Layer 4 (concept erasure) builds well on the student's cross-attention understanding. ESD-x vs ESD-u is clearly motivated. The "Why Not Just Erase Everything?" subsection with four problems + the collateral damage ComparisonRow is effective.
- The Arms Race section now reads as a crisp, non-redundant capstone. The backreferences ("as we saw in Layer 1") avoid repetition. The SurrogatePrompt statistic and the latent noise technique add genuinely new information. The framing paragraph at the end ("arms race... attackers find bypasses, defenders patch them") gives the student the right mental model.
- The final check (children's illustration generator) is a strong transfer exercise. The hidden answer is well-reasoned.
- The summary is slightly long (7 items) but each item is clear.

No point where I would be genuinely confused, lost, or unable to continue. The lesson reads smoothly from start to finish.

**Step 2 (Check Against Plan) -- Summary:**

| Planned Element | Status |
|----------------|--------|
| Target concept (multi-layered safety stack) | Fully taught, correct framing |
| Prerequisites resolution (no gaps identified) | No gaps, inline connections throughout |
| Misconception 1 (safety = one classifier) | Addressed in hook section |
| Misconception 2 (erasure makes layers unnecessary) | Addressed in "Why Not Just Erase Everything?" with four problems + ComparisonRow |
| Misconception 3 (blocklists are naive) | Addressed in WarningBlock aside "Don't Dismiss Blocklists" |
| Misconception 4 (only blocks sexual content) | Addressed in WarningBlock "Not Just NSFW" (lines 701-709) |
| Misconception 5 (techniques are unbreakable) | Addressed in dedicated Arms Race section |
| Negative example: blocklist bypass | Present (PhaseCard + widget + check section) |
| Negative example: negative prompts as insufficient | Present (GradientCard "Negative Prompts as Soft Safety") |
| Positive example: SD safety checker walkthrough | Present with traced computation |
| Positive example: ESD style erasure | Present with ComparisonRow |
| Positive example: DALL-E 3 pipeline | Present in "Real Systems" section |
| Modalities: visual/architecture diagram | Mermaid diagram present |
| Modalities: symbolic/code | Three pseudocode blocks (prompt classifier, SLD, ESD training, safety checker) |
| Modalities: concrete/traced computation | SD safety checker with real numbers, SLD formula |
| Modalities: verbal/analogy | Castle defense-in-depth analogy |
| Modalities: intuitive | Three "Of Course" moments at planned locations |
| Narrative arc | Follows planned structure precisely |
| Lesson outline (13 sections) | All 13 sections present and in order |
| Scope boundaries | Stayed within scope -- no ethics drift, no adversarial attack teaching |

No undocumented deviations found.

**Step 3 (Pedagogical Principles) -- Summary:**

- **Motivation Rule:** Problem stated before solution. The hook presents the deployment problem before any technique.
- **Modality Rule:** 5+ modalities for the core concept (safety stack): verbal/analogy (castle), visual (Mermaid diagram), symbolic (four pseudocode blocks), concrete example (traced SD safety checker computation), intuitive (three "Of Course" moments). Passes.
- **Example Rules:** 3 positive examples (SD safety checker, ESD style erasure, DALL-E 3 pipeline) + 2 negative examples (blocklist bypass, negative prompts as insufficient). Passes.
- **Misconception Rule:** 5 misconceptions identified and addressed at planned locations. Each has a concrete example. Passes.
- **Ordering Rules:** Concrete before abstract (deployment scenario before techniques), problem before solution (hook before layers), parts before whole (individual layers before composition), simple before complex (blocklist before SLD before ESD). Passes.
- **Load Rule:** 2-3 genuinely new concepts (safety stack pattern, concept erasure, CLIP-based safety classification). Passes.
- **Connection Rule:** Every technique explicitly connected to prior knowledge (CLIP, CFG, fine-tuning, cross-attention). Six explicit connections in the planning document, all present in the built lesson. Passes.
- **Reinforcement Rule:** All prerequisite concepts (CLIP, CFG, cross-attention, fine-tuning) are from Series 6-7, within recent memory and explicitly reinforced through connections. Passes.
- **Interaction Design Rule:** Widget buttons have `cursor-pointer`. Detail/summary elements have `cursor-pointer`. Passes.
- **Writing Style Rule:** Em dashes consistently use `&mdash;` with no spaces. Checked throughout lesson prose and aside text. Passes.

**Step 4 (Examples) -- Summary:**

All examples are concrete with real numbers or specific scenarios. The SD safety checker walkthrough traces actual tensor dimensions and cosine similarity values. The ESD ComparisonRow contrasts specific prompts before/after erasure. The blocklist bypass example progresses from "nude" to "n.u.d.e" to Unicode substitution. The collateral damage ComparisonRow (medical textbook before/after ESD-u) is effective. No missing examples identified.

**Step 5 (Narrative and Flow) -- Summary:**

The hook is compelling -- a concrete deployment scenario that creates urgency. Sections connect through the organizing Mermaid diagram (each section fills in one layer). Transitions are explicit ("Even if a prompt passes all text-level filters, the generation process itself can be steered"). The Arms Race section provides the payoff/resolution. Pacing is even throughout -- no section is too dense or too thin. The summary closes with clear takeaways.

**Step 6 (Notebook) -- Summary:**

No notebook exists. The planning document does not define a Practice section with exercises. No notebook is expected. No finding.

**Step 7 and 8 -- Findings written above.**

**What works well (final assessment):**
- The lesson is accurate, well-structured, and deeply connected to prior knowledge across 5+ modalities.
- All five misconceptions from the planning document are addressed at their planned locations with concrete examples.
- The interactive widget with 8 scenarios, three result states, and a clear legend is an effective exploration tool.
- The three check sections now force active recall via details/summary.
- The Arms Race section (after iteration 2 fix) is crisp and non-redundant, with backreferences avoiding repetition.
- The false positive discussion is well-placed in the lesson body text, keeping the TryThisBlock scannable.
- Every technique is connected to tools the student already knows (CLIP, CFG, fine-tuning, cross-attention).
- The lesson respects its scope boundaries throughout.

**Iteration arc summary:**
- Iteration 1: 2 critical, 4 improvement, 3 polish. MAJOR REVISION.
- Iteration 2: 0 critical, 2 improvement, 2 polish. NEEDS REVISION.
- Iteration 3: 0 critical, 0 improvement, 1 polish. PASS.
