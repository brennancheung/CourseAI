# Module 8.2: Safety & Content Moderation -- Record

## Concept Index

| Concept | Depth | Lesson |
|---------|-------|--------|
| Safety stack as a design pattern (multi-layered defense with heterogeneous techniques: prompt filtering, inference guidance, output classification, model erasure) | DEVELOPED | image-generation-safety |
| Keyword blocklists for prompt filtering (exact match, near-zero latency, trivially auditable, trivially bypassed) | INTRODUCED | image-generation-safety |
| Text embedding classifiers for prompt filtering (encode prompt with DistilBERT/RoBERTa, classify against safe/unsafe clusters, catches semantic bypasses like "nud3") | INTRODUCED | image-generation-safety |
| LLM-based prompt analysis (GPT-4 rewrites/refuses prompts before the diffusion model sees them, highest accuracy/latency/cost) | INTRODUCED | image-generation-safety |
| Negative prompts as soft safety (using CFG negative direction to steer away from unsafe content, probabilistic not guaranteed) | INTRODUCED | image-generation-safety |
| Safe Latent Diffusion (SLD) -- adding a safety guidance term to the CFG denoising formula to push away from unsafe concepts at each step | INTRODUCED | image-generation-safety |
| SLD formula: noise_pred = uncond + s*(cond - uncond) - s_sld*(safety - uncond) | INTRODUCED | image-generation-safety |
| SLD warmup steps (safety term activates after ~10 denoising steps to avoid disrupting global structure) | MENTIONED | image-generation-safety |
| CLIP-based safety classification (SD safety checker: CLIP ViT-L/14 encodes image, cosine similarity against 17 obfuscated concept embeddings, threshold check) | INTRODUCED | image-generation-safety |
| SD safety checker two-tier system (special_care_embeds with lower thresholds, regular concept_embeds, global adjustment parameter) | MENTIONED | image-generation-safety |
| Concept obfuscation in safety systems (hiding concept texts, publishing only embedding vectors to raise cost of adversarial targeting) | INTRODUCED | image-generation-safety |
| Purpose-trained NSFW image classifiers (ResNet/InceptionV3-based like NudeNet, high accuracy for specific categories, no generalization without retraining) | MENTIONED | image-generation-safety |
| Erased Stable Diffusion (ESD) -- using CFG-style guidance at training time to teach a model to "forget" concepts | INTRODUCED | image-generation-safety |
| ESD-x variant (cross-attention only, erases text-to-concept associations, good for style removal, minimal collateral damage) | INTRODUCED | image-generation-safety |
| ESD-u variant (all non-cross-attention layers, erases concepts globally even without text triggers, better for NSFW, more collateral damage) | INTRODUCED | image-generation-safety |
| Unified Concept Editing (UCE) -- closed-form weight editing of cross-attention, no gradient-based training, can debias/erase/moderate multiple concepts simultaneously | MENTIONED | image-generation-safety |
| Collateral damage from concept erasure (erasing "nudity" degrades medical imagery, swimwear, classical art) | INTRODUCED | image-generation-safety |
| Reversibility of concept erasure (adversarial fine-tuning recovers erased concepts in ~1000 steps) | INTRODUCED | image-generation-safety |
| Safety as an ongoing arms race (every layer has been bypassed in practice, defense in depth raises cost but does not eliminate attacks) | INTRODUCED | image-generation-safety |
| False positives in safety systems (legitimate content blocked by aggressive filtering, precision-recall tradeoff depends on application) | INTRODUCED | image-generation-safety |

## Lesson Summaries

### Lesson 1: Image Generation Safety (image-generation-safety)

**Concepts taught:** The multi-layered safety stack for production image generation systems. Four layers: (1) prompt-level filtering (keyword blocklists, text embedding classifiers, LLM-based prompt analysis), (2) inference-time guidance (negative prompts as soft safety, Safe Latent Diffusion), (3) post-generation image classification (SD safety checker using CLIP ViT-L/14, dedicated NSFW classifiers), (4) model-level concept erasure (ESD-x for cross-attention, ESD-u for global erasure, UCE for closed-form editing). Also: how DALL-E 3, Midjourney, and Stability AI compose these layers; the arms race between attacks and defenses.

**Mental models established:**
- "Defense in depth" / castle analogy: walls (blocklist), moat (text classifier), guards (output classifier), removing weapons from inside (model erasure). Each layer can be breached, but an attacker must breach ALL of them.
- "Safety stack as a pipeline": prompt enters, flows through layers in order (cheapest first, most expensive last), gets blocked at the first layer that catches it.
- "Closed vs open source as an architectural constraint": closed-source systems enforce all layers; open-source systems can only include layers as defaults that users can disable.

**Analogies used:**
- Castle defense-in-depth (walls, moat, guards, locked keep) for the multi-layered safety stack
- "Turning down the volume, not muting the channel" for negative prompts as soft safety (probabilistic, not guaranteed)
- "Fine-tuning in reverse" for ESD (same optimizer, same weight updates, opposite direction) -- with explicit caveat that it is not just negating the loss

**"Of Course" moments:**
1. "The safety checker IS zero-shot CLIP classification" -- encoding an image and comparing against reference embeddings via cosine similarity
2. "ESD-x targets cross-attention because cross-attention is where text conditioning enters the U-Net" -- erase the association in cross-attention, erase the concept
3. "SLD is the same geometric idea as CFG" -- vector arithmetic in noise-prediction space, one more term pushing away from a safety concept

**What is NOT covered:**
- Building a complete safety system from scratch
- Ethics/politics of what should be filtered (explicitly scoped out -- engineering lesson only)
- Adversarial attack methods in detail (mentioned as motivation, not taught as techniques)
- Training data filtering / data curation
- Watermarking or provenance tracking
- Mathematical proofs of erasure completeness or classifier optimality

**How concepts were taught:**
- The safety stack is introduced via a deployment scenario hook (you shipped a model, users are exploiting it), then taught layer by layer in order of cost/complexity
- Each layer connected to prior knowledge: CLIP for the safety checker, CFG formula for SLD, fine-tuning for ESD, cross-attention understanding for ESD-x targeting
- SD safety checker traced with specific numbers (cosine similarity 0.83 vs threshold 0.78, dimensions [3, 512, 512] -> [768])
- SLD taught as a one-term extension of the CFG formula the student already knows
- ESD taught with the Van Gogh style erasure example (concrete, visual, avoids needing NSFW content)
- Interactive widget (SafetyStackSimulator) with 8 pre-computed scenarios lets student toggle layers on/off to see what leaks through
- Three check sections with hidden answers: predict-and-verify (which layer catches which prompt), explain-it-back (why obfuscate concepts), transfer (design a safety stack for a children's app)
- Dedicated "Arms Race" section addresses overconfidence misconception with specific bypass evidence (SurrogatePrompt 70%+ bypass rates)
- False positive problem surfaced through the "museum statue" widget scenario and explicit discussion of precision-recall tradeoffs

**Review history:** PASS after 3 iterations (2 critical -> 0, 4 improvement -> 0, 1 polish carry). Critical fixes: hidden check answers via details/summary, dedicated Arms Race section for misconception #5. Key improvements: SLD pseudocode timestep clarification, widget legend, false positive discussion, collateral damage ComparisonRow, Arms Race section deduplication.
