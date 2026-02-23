# Module 8.2: Safety & Content Moderation

## Module Goal

The student can explain how production image generation systems prevent harmful content at every layer of the pipeline -- from prompt filtering through inference-time guidance to post-generation classification -- and understands the tradeoffs between each approach.

## Narrative Arc

Image generation models learn from internet-scale data, which means they learn everything -- including content you would never want them to produce. This module asks: once you have a powerful generative model, how do you actually prevent it from generating harmful content? The answer is not one technique but a multi-layered defense stack, where each layer catches different things and has different failure modes. The student progresses from the simplest approach (keyword blocklists) through increasingly sophisticated techniques (embedding classifiers, inference-time guidance, model-level concept erasure), understanding not just what each technique does but why a single layer is never enough.

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| image-generation-safety | The multi-layered safety stack for image generation (prompt-level, inference-level, image-level, model-level) | BUILD | First and currently only lesson. BUILD because the student already has deep knowledge of diffusion models, CLIP embeddings, and classifier-free guidance -- this lesson applies those existing concepts to a new problem domain rather than introducing fundamentally new theory. |

## Topic Allocation

- **Lesson 1 (image-generation-safety):** The complete safety stack. Prompt-level filtering (keyword blocklists, text classifiers, LLM-based prompt rewriting). Inference-level techniques (Safe Latent Diffusion, negative prompts as soft censorship). Image-level post-generation classification (Stable Diffusion safety checker using CLIP embeddings, dedicated NSFW classifiers). Model-level concept erasure (ESD, Unified Concept Editing). How real systems (DALL-E 3, Midjourney, Stability AI) combine these layers. Why no single layer is sufficient.

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| image-generation-safety | BUILD | All techniques build on concepts the student already has (CLIP embeddings, classifier-free guidance, fine-tuning, cosine similarity). The new ideas are how these combine into a safety stack, not new mathematical foundations. |

## Module-Level Misconceptions

- Students may think safety = one classifier at the end (miss the multi-layered nature)
- Students may think model-level erasure (ESD) solves everything and makes other layers unnecessary
- Students may think keyword blocklists are naive/useless (they are limited but still a valuable first line)
- Students may think safety filtering is a solved problem rather than an active arms race with adversarial bypasses
