# Module 8.3: Architecture Analysis

## Module Goal

The student can reverse-engineer production AI systems by reasoning from observable behavior, disclosed fragments, and published precedents to construct plausible architectural hypotheses -- developing the skill of informed speculation that working ML engineers use when evaluating undocumented systems.

## Narrative Arc

The AI industry increasingly ships production systems without comprehensive technical papers. Engineers need to evaluate, compare, and reason about these systems despite incomplete information. This module teaches the meta-skill of architecture analysis: observing behavior, gathering disclosed fragments, mapping to known precedents, and constructing falsifiable hypotheses about how something works. It is not about memorizing architectures -- it is about building the reasoning toolkit to analyze any system.

The arc begins with a concrete case study (Nano Banana Pro / Gemini 3 Pro Image) that forces the student to synthesize knowledge from across the entire course -- transformers, diffusion, autoregressive generation, visual tokenization, conditioning mechanisms -- into a coherent analysis of a system where the ground truth is not fully available.

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| nano-banana-pro | Autoregressive image generation as an alternative paradigm to diffusion, analyzed through the lens of a production system | STRETCH | First lesson in module; requires synthesizing knowledge from Series 4 (autoregressive transformers), Series 6-7 (diffusion pipeline), and Series 8 (vision-language models). STRETCH because it introduces the autoregressive image generation paradigm as a genuinely new concept while also requiring architectural reasoning across the full course. |

Future lessons could analyze other undocumented or partially-documented systems as they emerge (e.g., reasoning model internals, multimodal model routing, production inference optimization stacks).

## Rough Topic Allocation

- **nano-banana-pro:** Autoregressive vs diffusion for image generation (the fundamental paradigm difference), discrete visual tokenization (ViT-VQGAN / codebook approach), reasoning-guided synthesis (mandatory thinking step as composition planning), GemPix 2 as likely tokenizer+decoder component, why autoregressive excels at text rendering, hybrid autoregressive+diffusion possibilities (HART, BLIP3o-NEXT)

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| nano-banana-pro | STRETCH | Introduces autoregressive image generation as a genuinely new paradigm + requires cross-course synthesis. Two new concepts: discrete visual tokenization for images and autoregressive image token generation. The analytical reasoning framework (speculation from evidence) is also new but lightweight. |

## Module-Level Misconceptions

- Students may assume autoregressive image generation is inherently inferior to diffusion because diffusion dominated the course. In reality, the two paradigms have different strengths and the field is converging toward hybrid approaches.
- Students may think "we cannot know anything" about undocumented systems. The point is that substantial architectural reasoning is possible from observable behavior + disclosed fragments + published precedents.
- Students may conflate Google's marketing language with technical precision. "Reasoning-guided synthesis" and "GemPix 2" are branding -- the lesson must separate what these likely mean technically from what they literally say.
