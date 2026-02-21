import { CurriculumNode } from './types'

/**
 * Controllable Generation
 *
 * Module 7.1: Controllable Generation
 * 1. ControlNet
 * 2. ControlNet in Practice
 * 3. IP-Adapter
 */
const controllableGeneration: CurriculumNode = {
  slug: 'controllable-generation',
  title: 'Controllable Generation',
  children: [
    {
      slug: 'controlnet',
      title: 'ControlNet',
      description:
        'How ControlNet adds spatial conditioning to a frozen Stable Diffusion model by cloning the encoder, training only the clone on spatial maps, and connecting it via zero convolutions that guarantee the original model starts unchanged.',
      duration: '35 min',
      category: 'Controllable Generation',
      objectives: [
        'Explain how ControlNet clones the U-Net encoder and trains the copy on spatial maps while keeping the original model frozen',
        'Describe zero convolutions as 1×1 convs initialized to zero that guarantee the frozen model starts unchanged',
        'Trace the ControlNet forward pass showing additive connections at each resolution level',
        'Articulate how ControlNet (WHERE) coexists with text conditioning (WHAT) and timestep conditioning (WHEN)',
        'Distinguish ControlNet from LoRA: same frozen-model principle, different scale and mechanism',
      ],
      skills: [
        'controlnet-architecture',
        'zero-convolution',
        'trainable-encoder-copy',
        'spatial-conditioning',
        'multi-resolution-additive-connections',
      ],
      prerequisites: ['textual-inversion'],
      exercise: {
        constraints: [
          'Architecture understanding only—no preprocessing spatial maps',
          'No conditioning scale parameter or control-creativity tradeoff',
          'No stacking multiple ControlNets',
          'No IP-Adapter or image-based conditioning',
          'No training a ControlNet from scratch',
        ],
        steps: [
          'Inspect the ControlNet architecture: load a pre-trained ControlNet, print parameter counts, identify trainable vs frozen',
          'Verify the zero-initialization property: create a zero conv, pass features through it, confirm output is all zeros',
          'Trace the forward pass: inspect feature map shapes at each resolution from both frozen encoder and ControlNet copy',
          'Compare ControlNet vs vanilla SD: generate with and without spatial conditioning, vary text with fixed spatial map',
        ],
      },
    },
    {
      slug: 'controlnet-in-practice',
      title: 'ControlNet in Practice',
      description:
        'Turn photographs into spatial control signals with Canny edge detection, MiDaS depth estimation, and OpenPose skeleton extraction. Tune the conditioning scale tradeoff and stack multiple ControlNets for combined structural control.',
      duration: '40 min',
      category: 'Controllable Generation',
      objectives: [
        'Preprocess images into spatial maps using Canny, MiDaS depth, and OpenPose—choosing the right preprocessor for each use case',
        'Tune the conditioning_scale parameter to balance spatial precision against creative freedom',
        'Stack multiple ControlNets with per-ControlNet conditioning scales for complementary spatial control',
        'Diagnose common ControlNet failures: bad preprocessing, over-constrained scales, conflicting spatial maps',
      ],
      skills: [
        'canny-preprocessing',
        'depth-preprocessing',
        'openpose-preprocessing',
        'conditioning-scale-tuning',
        'multi-controlnet-stacking',
        'controlnet-failure-diagnosis',
      ],
      prerequisites: ['controlnet'],
      exercise: {
        constraints: [
          'Preprocessors used as black-box tools—no internal algorithm details',
          'No ControlNet architecture internals (covered in previous lesson)',
          'No training a ControlNet from scratch',
          'No IP-Adapter or image-based conditioning',
        ],
        steps: [
          'Extract Canny edges with three threshold pairs and compare ControlNet output quality',
          'Run three preprocessors (Canny, depth, OpenPose) on the same image and compare control types',
          'Sweep conditioning_scale from 0.3 to 2.0 and identify the control-creativity tradeoff',
          'Stack Canny + depth ControlNets and compare against single-ControlNet results',
          'Choose your own image and preprocessor(s) for an independent creative composition',
        ],
      },
    },
    {
      slug: 'ip-adapter',
      title: 'IP-Adapter',
      description:
        'How IP-Adapter adds image-based semantic conditioning to a frozen Stable Diffusion model by injecting a parallel set of K/V projections for CLIP image embeddings alongside the existing text K/V projections in cross-attention—decoupled cross-attention.',
      duration: '35 min',
      category: 'Controllable Generation',
      objectives: [
        'Explain how decoupled cross-attention adds a parallel K/V pathway for CLIP image embeddings alongside the existing text K/V path',
        'Trace the data flow: shared Q from spatial features, separate K/V from text and image embeddings, weighted addition of attention outputs',
        'Reason about why the parallel path preserves text conditioning—the text K/V projections are untouched',
        'Predict IP-Adapter behavior at different scale values (0 = text only, 1.0 = strong image influence)',
        'Distinguish IP-Adapter from LoRA (modifies existing weights vs adds new pathway) and ControlNet (spatial vs semantic conditioning)',
      ],
      skills: [
        'decoupled-cross-attention',
        'ip-adapter-architecture',
        'image-conditioning',
        'ip-adapter-scale-tuning',
        'controllable-generation-taxonomy',
      ],
      prerequisites: ['controlnet-in-practice'],
      exercise: {
        constraints: [
          'Architecture understanding and practical usage—no implementing from scratch',
          'No IP-Adapter training procedure in detail',
          'No IP-Adapter Plus, Face ID, or other variants (mentioned only)',
          'No CLIP image encoder internals (ViT architecture used as black box)',
        ],
        steps: [
          'Load IP-Adapter and generate with a reference image, comparing scale=0 vs scale=0.6',
          'Sweep the scale parameter (0.0, 0.3, 0.5, 0.7, 1.0) and observe text-dominant to image-dominant transition',
          'Test text-image coexistence: same reference image with three different text prompts',
          'Compose IP-Adapter with ControlNet: reference image for style + edge map for structure',
        ],
      },
    },
  ],
}

/**
 * The Score-Based Perspective
 *
 * Module 7.2: The Score-Based Perspective
 * 1. Score Functions & SDEs
 */
const scoreBasedPerspective: CurriculumNode = {
  slug: 'score-based-perspective',
  title: 'The Score-Based Perspective',
  children: [
    {
      slug: 'score-functions-and-sdes',
      title: 'Score Functions & SDEs',
      description:
        'What the noise prediction network has been secretly learning all along—the score function (gradient of log probability)—and how it connects DDPM to a continuous-time SDE/ODE framework that unifies diffusion models.',
      duration: '30 min',
      category: 'The Score-Based Perspective',
      objectives: [
        'Define the score function as the gradient of log probability and compute it for simple distributions',
        'Explain the score-noise equivalence: epsilon_theta is a scaled version of the score function',
        'Describe how the DDPM forward process generalizes to a continuous-time SDE',
        'Distinguish the reverse SDE (stochastic, DDPM-like) from the probability flow ODE (deterministic, DDIM-like)',
        'Connect score-based vocabulary to the DDPM/DDIM/DPM-Solver mechanics already learned',
      ],
      skills: [
        'score-function',
        'score-noise-equivalence',
        'forward-sde',
        'reverse-sde',
        'probability-flow-ode',
      ],
      prerequisites: ['ip-adapter'],
      exercise: {
        constraints: [
          'Conceptual/theoretical lesson—no new model architectures or tools',
          'No Ito calculus, stochastic integration, or Fokker-Planck equations',
          'No score matching training objective (DDPM training already does this implicitly)',
          'No flow matching (next lesson)',
        ],
        steps: [
          'Compute score functions analytically for 1D Gaussians and Gaussian mixtures, plot alongside PDFs',
          'Visualize the score as a 2D vector field for a Gaussian mixture, observe smoothing at different noise levels',
          'Verify the score-noise equivalence with a pre-trained diffusion model: compare noise prediction to implied score',
          'Compare SDE vs ODE sampling trajectories: stochastic (wiggly, diverse) vs deterministic (smooth, repeatable)',
        ],
      },
    },
    {
      slug: 'flow-matching',
      title: 'Flow Matching',
      description:
        'What if we designed the generation trajectory to be straight? Flow matching replaces curved diffusion paths with straight-line interpolation and velocity prediction—the training objective behind SD3 and Flux.',
      duration: '30 min',
      category: 'The Score-Based Perspective',
      objectives: [
        'Explain why diffusion ODE trajectories curve and how curvature limits ODE solver step count',
        'Describe conditional flow matching: straight-line interpolation, constant velocity, and the MSE velocity prediction loss',
        'Work through a flow matching training step with concrete numbers',
        'Relate velocity prediction to noise prediction and score prediction as three parameterizations of the same vector field',
        'Explain rectified flow as iterative trajectory straightening at an intuition level',
        'Connect flow matching to SD3 and Flux as the reason for fewer sampling steps',
      ],
      skills: [
        'flow-matching-interpolation',
        'velocity-prediction',
        'conditional-flow-matching',
        'rectified-flow',
        'parameterization-equivalence',
      ],
      prerequisites: ['score-functions-and-sdes'],
      exercise: {
        constraints: [
          'No optimal transport formulations (Lipman et al. 2023)',
          'No DiT architecture (Module 7.4)',
          'No consistency models (Module 7.3)',
          'No training from scratch beyond toy 2D examples',
        ],
        steps: [
          'Implement flow matching vs DDPM interpolation in 1D, plot paths and velocity profiles',
          'Apply Euler method on curved vs straight 2D trajectories, compare accuracy at different step counts',
          'Train a flow matching model on 2D data (two-moons), generate samples via Euler ODE solving',
          'Compare flow matching vs DDPM: sample quality at varying step counts (1, 5, 10, 20, 50)',
        ],
      },
    },
  ],
}

/**
 * Fast Generation
 *
 * Module 7.3: Fast Generation
 * 1. Consistency Models
 * 2. Latent Consistency & Turbo
 * 3. The Speed Landscape
 */
const fastGeneration: CurriculumNode = {
  slug: 'fast-generation',
  title: 'Fast Generation',
  children: [
    {
      slug: 'consistency-models',
      title: 'Consistency Models',
      description:
        'What if you could collapse an entire ODE trajectory into a single function evaluation? Consistency models use the self-consistency property of deterministic ODEs as a training objective, bypassing multi-step ODE solving entirely.',
      duration: '35 min',
      category: 'Fast Generation',
      objectives: [
        'Explain the self-consistency property: any point on the same ODE trajectory maps to the same clean endpoint',
        'Describe the consistency function f(x_t, t) = x_0 and the boundary condition f(x, epsilon) = x',
        'Distinguish consistency models from ODE solvers with fewer steps—no direction, no stepping, a direct mapping',
        'Explain consistency distillation: using a pretrained teacher to provide ODE trajectory estimates for training',
        'Compare consistency distillation (with teacher) to consistency training (without teacher) and their quality-speed tradeoffs',
        'Describe multi-step consistency as a quality-speed tradeoff between 1-step and multi-step diffusion',
      ],
      skills: [
        'self-consistency-property',
        'consistency-function',
        'consistency-distillation',
        'consistency-training',
        'multi-step-consistency',
        'ema-target-network',
      ],
      prerequisites: ['flow-matching'],
      exercise: {
        constraints: [
          'Toy 2D data only—no production-scale training',
          'No latent consistency models or LCM-LoRA (next lesson)',
          'No adversarial diffusion distillation (next lesson)',
          'No full mathematical derivations of the consistency training loss',
          'No architecture details (skip connections, EMA schedule specifics)',
        ],
        steps: [
          'Visualize self-consistency on ODE trajectories: verify multiple points on one trajectory reach the same endpoint',
          'Compare one-step ODE, one-step DDIM, and the true ODE endpoint—see why single-step methods fail',
          'Train a toy consistency model via distillation on 2D two-moons data, generate 1-step samples',
          'Compare multi-step consistency (1, 2, 4, 8 steps) to teacher ODE solving (1, 5, 10, 20, 50 steps)',
        ],
      },
    },
    {
      slug: 'latent-consistency-and-turbo',
      title: 'Latent Consistency & Turbo',
      description:
        'Two production-ready approaches to 1-4 step generation: Latent Consistency Models apply consistency distillation to SD/SDXL latent space with LCM-LoRA for plug-and-play acceleration, while SDXL Turbo adds a discriminator for sharper single-step results via adversarial diffusion distillation.',
      duration: '35 min',
      category: 'Fast Generation',
      objectives: [
        'Explain how LCM applies consistency distillation to SD/SDXL latent diffusion—same procedure, latent-space tensors',
        'Describe LCM-LoRA as a LoRA adapter that captures "how to generate fast" and generalizes across compatible checkpoints',
        'Distinguish LCM-LoRA from style LoRAs: different training data, different loss, different purpose',
        'Explain the discriminator concept and its role in adversarial diffusion distillation (ADD)',
        'Compare ADD vs consistency distillation: different teacher signals, different failure modes (sharpness vs softness)',
        'Choose between LCM-LoRA and SDXL Turbo based on universality, quality, and flexibility requirements',
      ],
      skills: [
        'latent-consistency-models',
        'lcm-lora',
        'lcm-lora-universality',
        'adversarial-diffusion-distillation',
        'discriminator-concept',
        'hybrid-loss',
        'speed-approach-selection',
      ],
      prerequisites: ['consistency-models'],
      exercise: {
        constraints: [
          'LCM-LoRA usage only—no training from scratch',
          'No discriminator architecture or full ADD loss derivation',
          'No GAN theory beyond what is needed for ADD',
          'No SDXL Turbo in exercises (requires dedicated model)',
          'No SDXL architecture details (Module 7.4)',
        ],
        steps: [
          'Load SD v1.5 + LCM-LoRA, generate at 50/4/4-with-LoRA steps, compare quality and timing',
          'Apply the same LCM-LoRA to a community fine-tune, verify universality',
          'Sweep step counts (1, 2, 4, 8) and guidance scales (1.0, 1.5, 2.0, 4.0) to find practical sweet spots',
          'Compose LCM-LoRA with a style LoRA, compare to style LoRA alone at 50 steps',
        ],
      },
    },
    {
      slug: 'the-speed-landscape',
      title: 'The Speed Landscape',
      description:
        'A decision framework for choosing the right acceleration approach: organize DPM-Solver++, flow matching, LCM, LCM-LoRA, and SDXL Turbo by speed, quality, flexibility, and composability—the workshop manual for every tool you have learned.',
      duration: '30 min',
      category: 'Fast Generation',
      objectives: [
        'Evaluate acceleration approaches across four dimensions: speed, quality, flexibility, and composability',
        'Apply the decision framework to concrete generation scenarios reaching different conclusions',
        'Identify which acceleration levels compose (across levels) and which conflict (within levels)',
        'Explain the nonlinear quality-speed curve: most speedup is free, the last steps cost the most',
        'Choose the right approach for a given scenario using the complete taxonomy',
      ],
      skills: [
        'acceleration-taxonomy',
        'speed-decision-framework',
        'composability-analysis',
        'quality-speed-tradeoff',
      ],
      prerequisites: ['latent-consistency-and-turbo'],
      exercise: {
        constraints: [
          'Analysis and comparison only—no model training',
          'No SDXL Turbo exercises (requires dedicated model and significant VRAM)',
          'No new technical concepts—pure synthesis of prior knowledge',
          'No production deployment optimization',
        ],
        steps: [
          'Load SD 1.5, generate the same image with DDPM (50 steps), DPM-Solver++ (20 steps), and DPM-Solver++ (10 steps)—compare quality and time',
          'Load the same model + LCM-LoRA, generate at 4 and 8 steps, compare with DPM-Solver++ at 20 steps',
          'Combine LCM-LoRA + style LoRA at 4 steps, compare with each approach alone—verify composability',
          'Given three scenarios, write which approach to use and why using the four-dimensional framework, then verify one choice',
        ],
      },
    },
  ],
}

/**
 * Next-Generation Architectures
 *
 * Module 7.4: Next-Generation Architectures
 * 1. SDXL
 * 2. Diffusion Transformers (DiT)
 * 3. SD3 & Flux
 */
const nextGenerationArchitectures: CurriculumNode = {
  slug: 'next-generation-architectures',
  title: 'Next-Generation Architectures',
  children: [
    {
      slug: 'sdxl',
      title: 'SDXL',
      description:
        'The U-Net pushed to its limit—dual text encoders for richer conditioning, micro-conditioning to handle multi-resolution training data, and a refiner model for fine detail. Every improvement is about what goes IN to the U-Net, not a new architecture.',
      duration: '35 min',
      category: 'Next-Generation Architectures',
      objectives: [
        'Explain how SDXL concatenates CLIP ViT-L [77, 768] and OpenCLIP ViT-bigG [77, 1280] into [77, 2048] for richer cross-attention conditioning',
        'Distinguish SDXL dual encoder concatenation from IP-Adapter decoupled cross-attention—same goal, different mechanism',
        'Describe micro-conditioning (original_size, crop_top_left, target_size) as a solution to multi-resolution training artifacts',
        'Explain the refiner model as img2img with a specialized second U-Net for fine detail',
        'Articulate why SDXL represents the U-Net architecture ceiling, setting up the paradigm shift to transformers',
      ],
      skills: [
        'sdxl-dual-text-encoders',
        'sdxl-micro-conditioning',
        'sdxl-refiner-pipeline',
        'sdxl-architecture-scaling',
        'unet-ceiling',
      ],
      prerequisites: ['the-speed-landscape'],
      exercise: {
        constraints: [
          'Pipeline inspection and generation—no training from scratch',
          'No SDXL Turbo or LCM-LoRA for SDXL (already covered in Module 7.3)',
          'No ControlNet or IP-Adapter for SDXL (same concepts from Module 7.1)',
          'No every internal U-Net detail (channel counts, attention placement)',
          'No DiT architecture (next lesson)',
        ],
        steps: [
          'Inspect SDXL dual text encoders: print model class names, parameter counts, output shapes—verify [77, 768] and [77, 1280]',
          'Generate with SDXL base at 1024x1024, compare to SD v1.5 at 512x512, vary guidance_scale',
          'Explore micro-conditioning: generate with original_size (1024, 1024) vs (256, 256), crop offsets (0,0) vs (512,512)',
          'Set up the base + refiner two-stage pipeline, compare base-only vs base+refiner, vary the handoff point',
        ],
      },
    },
    {
      slug: 'diffusion-transformers',
      title: 'Diffusion Transformers (DiT)',
      description:
        'Replace the U-Net with a standard vision transformer on latent patches. Patchify the latent into tokens, process with transformer blocks conditioned via adaLN-Zero, and scale with the same two-knob recipe as GPT. Two knowledge threads—transformers and latent diffusion—converge.',
      duration: '40 min',
      category: 'Next-Generation Architectures',
      objectives: [
        'Explain the patchify operation: split a latent tensor into non-overlapping patches, flatten, and project to d_model dimensions—the image equivalent of tokenization',
        'Trace tensor shapes through the full DiT pipeline: noisy latent → patchify → N transformer blocks → unpatchify → predicted noise',
        'Describe adaLN-Zero conditioning: scale (γ), shift (β), and gate (α) on LayerNorm, with zero-initialized gate making each block start as an identity function',
        'Compare DiT architecture to the U-Net: no convolutions, no encoder-decoder hierarchy, no skip connections, attention at every layer',
        'Articulate why DiT scales more predictably than U-Nets—the two-knob recipe (d_model and N) vs ad hoc U-Net scaling decisions',
      ],
      skills: [
        'dit-patchify',
        'dit-unpatchify',
        'adaln-zero',
        'vit-on-latent-patches',
        'dit-scaling-laws',
        'transformer-vs-unet-tradeoff',
      ],
      prerequisites: ['sdxl'],
      exercise: {
        constraints: [
          'Architecture understanding and pretrained inference—no DiT training from scratch',
          'No text conditioning (DiT is class-conditional; text conditioning is next lesson)',
          'No ViT pretraining on classification tasks (DiT trains from scratch on diffusion)',
          'No DiT variants (U-ViT, MDT)—only the original DiT from Peebles & Xie 2023',
          'No SD3/Flux/MMDiT architecture (next lesson)',
        ],
        steps: [
          'Implement patchify and unpatchify from scratch, verify shapes at every step and round-trip consistency',
          'Implement one adaLN-Zero conditioning step, verify the identity property at α=0',
          'Load a pretrained DiT model, inspect architecture, count parameters, compare to U-Net layer summary',
          'Generate class-conditional ImageNet images with DiT-XL/2, vary sampling steps and guidance scale',
        ],
      },
    },
    {
      slug: 'sd3-and-flux',
      title: 'SD3 & Flux',
      description:
        'The current frontier—combine DiT with joint text-image attention (MMDiT), T5-XXL text encoding, and flow matching. Every component traces to a lesson you have already completed. The convergence of transformers, latent diffusion, and flow matching.',
      duration: '40 min',
      category: 'Next-Generation Architectures',
      objectives: [
        'Explain how MMDiT replaces cross-attention with joint self-attention: concatenate text and image tokens, run standard self-attention with modality-specific Q/K/V projections',
        'Describe why T5-XXL complements rather than replaces CLIP—different training produces different text understanding',
        'Trace the full SD3 pipeline end-to-end, annotating which component comes from which lesson',
        'Compare joint attention to cross-attention: one bidirectional attention operation vs two unidirectional operations',
        'Articulate the convergence: every component of SD3/Flux traces to a concept built earlier in the course',
      ],
      skills: [
        'mmdit-joint-attention',
        'modality-specific-projections',
        'triple-text-encoder',
        't5-xxl-text-encoder',
        'rectified-flow-in-practice',
        'logit-normal-sampling',
        'sd3-flux-pipeline',
      ],
      prerequisites: ['diffusion-transformers'],
      exercise: {
        constraints: [
          'Pipeline inspection and generation—no training from scratch',
          'No implementing MMDiT from scratch (too much architecture code)',
          'No every Flux variant (dev/schnell/pro)—mentioned for vocabulary only',
          'No video extensions, multimodal extensions, or ControlNet/IP-Adapter for SD3/Flux',
          'No detailed SD3 vs Flux architecture comparison—treated as same family',
        ],
        steps: [
          'Inspect the SD3 triple text encoder setup: print model class names, parameter counts, verify embedding shapes',
          'Visualize joint attention: extract attention weights from one MMDiT block, identify four quadrants (text-text, text-image, image-text, image-image)',
          'Generate images with SD3, vary inference steps (10, 20, 30, 50) to observe the flow matching payoff',
          'Trace the full SD3 pipeline end-to-end, annotating each step with the lesson that covered it',
        ],
      },
    },
  ],
}

/**
 * Post-SD Advances
 *
 * Series 7: Capstone series exploring what the field built on top of
 * Stable Diffusion. Assumes deep knowledge of the SD architecture
 * from Series 6.
 */
export const postSdAdvances: CurriculumNode = {
  slug: 'post-sd-advances',
  title: 'Post-SD Advances',
  icon: 'Sparkles',
  description:
    'Controllable generation, efficient architectures, and the frontier beyond Stable Diffusion—ControlNet, consistency models, flow matching, and more',
  children: [
    controllableGeneration,
    scoreBasedPerspective,
    fastGeneration,
    nextGenerationArchitectures,
  ],
}
