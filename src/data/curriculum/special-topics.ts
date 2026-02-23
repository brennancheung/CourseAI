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
  children: [visionAndVisionLanguage, safetyAndContentModeration],
}
