import { CurriculumNode } from './types'

/**
 * Vision & Vision-Language Models
 *
 * Module 8.1: Vision & Vision-Language Models
 * 1. SigLIP 2
 * 2. SAM 3 (planned)
 */
const visionAndVisionLanguage: CurriculumNode = {
  slug: 'vision-and-vision-language',
  title: 'Vision & Vision-Language Models',
  children: [
    {
      slug: 'siglip-2',
      title: 'SigLIP 2',
      description:
        'How SigLIP replaces CLIP\u2019s softmax cross-entropy loss with sigmoid-based per-pair binary classification, removing the batch-size dependency that makes CLIP expensive to train, and how SigLIP 2 adds training methodology improvements.',
      duration: '30 min',
      category: 'Vision & Vision-Language Models',
      objectives: [
        'Explain why CLIP\u2019s softmax cross-entropy loss creates a structural dependence on large batch sizes via the softmax denominator',
        'Describe how SigLIP\u2019s sigmoid loss scores each image-text pair independently as a binary classification, removing batch-size coupling',
        'Trace the sigmoid loss computation on a similarity matrix and verify that each cell\u2019s loss is self-contained',
        'Explain SigLIP 2\u2019s training improvements: multi-stage training, self-distillation, and multi-resolution processing',
        'Articulate why contrastive pretraining produces vision encoders suitable for VLMs like PaliGemma',
      ],
      skills: [
        'sigmoid-contrastive-loss',
        'batch-size-independence',
        'siglip-training-recipe',
        'self-distillation',
        'vlm-vision-encoder',
      ],
      prerequisites: ['clip'],
    },
    {
      slug: 'sam-3',
      title: 'SAM 3',
      description:
        'How the Segment Anything Model brings the \u201cfoundation model\u201d approach to image segmentation. Traces the prompt encoding math (Fourier positional encoding), mask decoder internals (three attention operations per layer with tensor shapes), loss function (focal loss, dice loss, minimum-loss assignment), SAM 2\u2019s memory mechanism, and SAM 3\u2019s concept-level text prompting.',
      duration: '50 min',
      category: 'Vision & Vision-Language Models',
      objectives: [
        'Explain what image segmentation is and how it differs from classification and detection',
        'Describe SAM\u2019s three-component architecture (ViT encoder + prompt encoder + mask decoder) and each component\u2019s role',
        'Trace how a point prompt is encoded: Fourier positional encoding with 128 frequency bands, projection, and learned type embeddings',
        'Explain the mask decoder\u2019s three attention operations per layer (self-attention, token-to-image cross-attention, image-to-token cross-attention) with tensor dimension traces',
        'Derive the focal loss and dice loss formulas, trace them on concrete pixel values, and explain minimum-loss assignment for multi-mask training',
        'Describe SAM 2\u2019s memory mechanism: memory encoder, memory bank, memory attention as cross-attention with tensor shapes',
        'Trace the additive evolution from SAM 1 (images) to SAM 2 (video) to SAM 3 (concept-level segmentation)',
      ],
      skills: [
        'image-segmentation',
        'promptable-segmentation',
        'sam-architecture',
        'fourier-positional-encoding',
        'focal-loss',
        'dice-loss',
        'data-engine',
        'foundation-model-pattern',
      ],
      prerequisites: ['siglip-2'],
    },
  ],
}

/**
 * Safety & Content Moderation
 *
 * Module 8.2: Safety & Content Moderation
 * 1. Image Generation Safety
 */
const safetyAndContentModeration: CurriculumNode = {
  slug: 'safety-and-content-moderation',
  title: 'Safety & Content Moderation',
  children: [
    {
      slug: 'image-generation-safety',
      title: 'Image Generation Safety',
      description:
        'How production image generation systems prevent harmful content through a multi-layered defense stack\u2014from keyword blocklists through inference-time guidance to model-level concept erasure.',
      duration: '35 min',
      category: 'Safety & Content Moderation',
      objectives: [
        'Explain the multi-layered safety stack pattern and why no single technique is sufficient',
        'Describe how prompt-level filtering works: keyword blocklists, text embedding classifiers, and LLM-based prompt analysis',
        'Trace how Safe Latent Diffusion adds a safety guidance term to the CFG formula to steer generation away from unsafe content',
        'Explain how the Stable Diffusion safety checker uses CLIP cosine similarity against 17 concept embeddings with thresholds',
        'Describe how Erased Stable Diffusion (ESD) uses CFG-style guidance at training time to remove concepts from model weights',
        'Compare how DALL-E 3, Midjourney, and Stability AI compose their safety stacks differently',
      ],
      skills: [
        'safety-stack-pattern',
        'prompt-filtering',
        'safe-latent-diffusion',
        'clip-safety-checker',
        'concept-erasure',
        'defense-in-depth',
      ],
      prerequisites: ['clip', 'text-conditioning-and-guidance'],
    },
  ],
}

/**
 * Architecture Analysis
 *
 * Module 8.3: Architecture Analysis
 * 1. Nano Banana Pro (Gemini 3 Pro Image)
 */
const architectureAnalysis: CurriculumNode = {
  slug: 'architecture-analysis',
  title: 'Architecture Analysis',
  children: [
    {
      slug: 'nano-banana-pro',
      title: 'Nano Banana Pro',
      description:
        'How Google\u2019s Gemini 3 Pro image generator likely works\u2014and why autoregressive generation is a viable alternative to diffusion that excels at text rendering. Constructs an informed architectural hypothesis from observable behavior, disclosed fragments, and published precedents.',
      duration: '35 min',
      category: 'Architecture Analysis',
      objectives: [
        'Explain how discrete visual tokenization (VQ-VAE/ViT-VQGAN) maps images to integer tokens, bridging from continuous VAE latent spaces',
        'Describe autoregressive image generation: the same predict-next-token loop as GPT, applied to visual tokens',
        'Articulate why autoregressive generation inherently excels at text rendering while diffusion struggles',
        'Construct a plausible architectural hypothesis for Nano Banana Pro from disclosed facts, observable behavior, and published precedents',
        'Explain why the mandatory thinking step is architecturally motivated for autoregressive generation',
      ],
      skills: [
        'discrete-visual-tokenization',
        'autoregressive-image-generation',
        'architectural-analysis',
        'vq-vae-codebook',
        'paradigm-comparison',
      ],
      prerequisites: [
        'building-nanogpt',
        'variational-autoencoders',
        'sampling-and-generation',
        'text-conditioning-and-guidance',
        'diffusion-transformers',
      ],
    },
  ],
}

/**
 * Image Generation Landscape
 *
 * Module 8.4: Image Generation Landscape
 * 1. Open Weight Image Generation Models
 */
const imageGenerationLandscape: CurriculumNode = {
  slug: 'image-generation-landscape',
  title: 'Image Generation Landscape',
  children: [
    {
      slug: 'open-weight-image-gen',
      title: 'The Open Weight Image Generation Landscape',
      description:
        'A comprehensive survey of every major open weight image generation model from 2022 to 2025\u2014organized by architecture, text encoding, and training objective into a navigable taxonomy with a reference comparison table.',
      duration: '45-60 min',
      category: 'Image Generation Landscape',
      objectives: [
        'Place any open weight image generation model into the architectural taxonomy by identifying its backbone type (U-Net, DiT, MMDiT, S3-DiT)',
        'Trace the three independent evolution lines: backbone (U-Net \u2192 DiT \u2192 MMDiT \u2192 S3-DiT), text encoding (CLIP \u2192 multi-encoder \u2192 LLM), training objective (DDPM \u2192 flow matching \u2192 distillation)',
        'Identify the key innovation each model contributed to the field and which innovations propagated to other models',
        'Use the five-question framework to place new model announcements into the landscape',
        'Explain why ecosystem maturity can matter as much as architectural quality for practical adoption',
      ],
      skills: [
        'model-landscape-navigation',
        'architecture-taxonomy',
        'innovation-attribution',
        'model-lineage-tracing',
        'ecosystem-evaluation',
      ],
      prerequisites: [
        'sdxl',
        'diffusion-transformers',
        'sd3-and-flux',
        'z-image',
        'flow-matching',
        'consistency-models',
      ],
    },
  ],
}

/**
 * Preference Optimization Deep Dives
 *
 * Module 8.5: Preference Optimization Deep Dives
 * 1. Direct Preference Optimization
 */
const preferenceOptimizationDeepDives: CurriculumNode = {
  slug: 'preference-optimization-deep-dives',
  title: 'Preference Optimization Deep Dives',
  children: [
    {
      slug: 'direct-preference-optimization',
      title: 'Direct Preference Optimization',
      description:
        'Deriving the DPO loss from first principles\u2014starting from the RLHF objective, finding the closed-form optimal policy, and substituting into the Bradley-Terry preference model to eliminate the reward model entirely.',
      duration: '40 min',
      category: 'Preference Optimization Deep Dives',
      objectives: [
        'State the Bradley-Terry preference model and explain how it converts reward differences into preference probabilities',
        'Write the RLHF objective (KL-constrained reward maximization) and explain each term',
        'Derive the closed-form optimal policy and explain why the reference model appears in it',
        'Trace the substitution that produces the DPO loss and explain why it is not an approximation',
        'Walk through a numerical DPO training step with specific log-probabilities',
        'Explain why the reference model is structurally essential (not optional regularization)',
        'Describe the implicit reward model insight\u2014any policy paired with a reference defines a reward',
      ],
      skills: [
        'bradley-terry-model',
        'dpo-derivation',
        'dpo-loss-function',
        'implicit-reward-model',
        'preference-optimization-math',
      ],
      prerequisites: ['rlhf-and-alignment', 'alignment-techniques-landscape'],
      exercise: {
        constraints: [
          'Focus on understanding the derivation before implementing',
          'Use small models (GPT-2 small) for tractable training',
          'Exercises are cumulative\u2014each builds on the previous',
        ],
        steps: [
          'Verify the DPO loss formula by hand on pre-computed log-probabilities',
          'Implement the DPO loss function in PyTorch',
          'Train a small model with DPO on preference pairs',
          'Extract and analyze the implicit reward model',
        ],
      },
    },
  ],
}

/**
 * Special Topics
 *
 * Series 8: Standalone deep dives into interesting models, techniques,
 * and ideas that don't belong to a structured series.
 */
export const specialTopics: CurriculumNode = {
  slug: 'special-topics',
  title: 'Special Topics',
  icon: 'Telescope',
  description:
    'Standalone deep dives into modern vision models, vision-language models, and other interesting topics\u2014self-contained lessons that build on the full course',
  children: [
    visionAndVisionLanguage,
    safetyAndContentModeration,
    architectureAnalysis,
    imageGenerationLandscape,
    preferenceOptimizationDeepDives,
  ],
}
