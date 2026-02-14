import { CurriculumNode } from './types'

/**
 * Generative Foundations
 *
 * Module 6.1: Generative Foundations
 * 1. From Classification to Generation
 * 2. Autoencoders
 * 3. Variational Autoencoders
 * 4. Exploring Latent Spaces
 */
const generativeFoundations: CurriculumNode = {
  slug: 'generative-foundations',
  title: 'Generative Foundations',
  children: [
    {
      slug: 'from-classification-to-generation',
      title: 'From Classification to Generation',
      description:
        'The conceptual shift from discriminative models (learning decision boundaries) to generative models (learning the data distribution itself).',
      duration: '20 min',
      category: 'Generative Foundations',
      objectives: [
        'Articulate the difference between discriminative and generative models',
        'Understand that generation means sampling from a learned probability distribution',
        'Connect language model generation (familiar) to image generation (new)',
        'Explain why generative models learn structure, not memorized instances',
      ],
      skills: [
        'discriminative-vs-generative',
        'probability-distribution-over-data',
        'sampling-as-generation',
        'generative-framing',
      ],
      prerequisites: ['decoder-only-transformers'],
      exercise: {
        constraints: [
          'Conceptual distinction only\u2014no specific generative architectures',
          'No probability density functions, likelihood, or formal probability',
          'No training procedures or loss functions for generative models',
          'No code or implementation\u2014conceptual only',
        ],
        steps: [
          'Recall language model generation as a bridge to generative modeling',
          'Understand discriminative models as learning P(y|x)\u2014decision boundaries',
          'See why inverting a classifier cannot produce generation (many-to-one problem)',
          'Understand generative models as learning P(x)\u2014the data distribution',
          'Interact with the discriminative vs generative widget to see both paradigms on the same data',
          'Understand why generation is not memorization (the dimensionality argument)',
        ],
      },
    },
    {
      slug: 'autoencoders',
      title: 'Autoencoders',
      description:
        'Force a neural network through a tiny bottleneck and train it to reconstruct its input. What it learns to keep is what matters\u2014and that compressed representation is the foundation for generative models.',
      duration: '30 min',
      category: 'Generative Foundations',
      objectives: [
        'Understand the encoder-decoder (hourglass) architecture and the role of the bottleneck',
        'Recognize that reconstruction loss measures how well the compressed representation preserves what matters',
        'See that the autoencoder is NOT a generative model\u2014random latent codes produce garbage',
        'Build and train an autoencoder on Fashion-MNIST in a Colab notebook',
      ],
      skills: [
        'encoder-decoder-architecture',
        'bottleneck-latent-representation',
        'reconstruction-loss',
        'conv-transpose-upsampling',
        'self-supervised-learning',
      ],
      prerequisites: ['from-classification-to-generation'],
      exercise: {
        constraints: [
          'No variational autoencoders, KL divergence, or probabilistic encoding',
          'No sampling from the latent space for generation\u2014the autoencoder cannot do this',
          'No denoising autoencoders, sparse autoencoders, or other variants',
          'No latent space interpolation or arithmetic\u2014that comes later',
        ],
        steps: [
          'Understand the encoder as a CNN compressing to a bottleneck vector',
          'Understand the decoder as the reverse path using ConvTranspose2d',
          'See reconstruction loss as MSE where the target IS the input',
          'Explore the bottleneck size tradeoff with the interactive widget',
          'Read the PyTorch implementation and recognize familiar patterns',
          'Build and train an autoencoder in the Colab notebook',
          'Confirm the autoencoder fails at generation (random noise \u2192 garbage)',
        ],
      },
    },
    {
      slug: 'variational-autoencoders',
      title: 'Variational Autoencoders',
      description:
        'Encode to a distribution, not a point. Add KL divergence as a regularizer to keep the latent space organized and sampleable\u2014turning the autoencoder into your first true generative model.',
      duration: '35 min',
      category: 'Generative Foundations',
      objectives: [
        'Understand why autoencoder latent spaces have gaps that prevent generation',
        'See how encoding to a distribution (mean + variance) fills those gaps',
        'Understand KL divergence as a regularizer that keeps the latent space organized',
        'Implement a VAE by converting your autoencoder (three changes)',
        'Generate your first novel images by sampling from the latent space',
      ],
      skills: [
        'distributional-encoding',
        'kl-divergence-regularizer',
        'reparameterization-trick',
        'vae-loss-function',
        'reconstruction-regularization-tradeoff',
      ],
      prerequisites: ['autoencoders'],
      exercise: {
        constraints: [
          'No full ELBO derivation or variational inference theory',
          'No conditional VAEs, beta-VAE theory, or disentangled representations',
          'No comparing VAEs to GANs or other generative architectures',
          'No latent space interpolation, arithmetic, or generation experiments\u2014that is the next lesson',
        ],
        steps: [
          'Understand why the autoencoder\u2019s latent space has gaps',
          'See how encoding to a distribution fills the gaps',
          'Understand the reparameterization trick at intuition level',
          'See KL divergence as a regularizer that prevents distribution collapse',
          'Explore the autoencoder vs VAE latent space with the interactive widget',
          'Read the PyTorch VAE code and identify the three changes',
          'Convert the autoencoder to a VAE in the Colab notebook',
          'Generate novel images by sampling from N(0,1)',
        ],
      },
    },
    {
      slug: 'exploring-latent-spaces',
      title: 'Exploring Latent Spaces',
      description:
        'Sample, interpolate, and do arithmetic in the latent space. Create images that have never existed\u2014then see the quality ceiling that motivates diffusion models.',
      duration: '25 min',
      category: 'Generative Foundations',
      objectives: [
        'Generate novel images by sampling from the VAE\u2019s latent space',
        'Interpolate between two images in latent space and see coherent transitions',
        'Perform latent arithmetic to transfer attributes between encoded items',
        'Visualize latent space structure with t-SNE',
        'Recognize VAE quality limitations and connect them to the reconstruction-vs-KL tradeoff',
      ],
      skills: [
        'latent-space-sampling',
        'latent-interpolation',
        'latent-arithmetic',
        'tsne-visualization',
        'vae-quality-limitations',
      ],
      prerequisites: ['variational-autoencoders'],
      exercise: {
        constraints: [
          'No new mathematical theory\u2014this is a CONSOLIDATE lesson',
          'No training or modifying the VAE\u2014that was the previous lesson',
          'No GANs, diffusion, or other generative architectures',
          'No t-SNE/UMAP algorithmic details',
          'No disentangled representations or beta-VAE theory',
        ],
        steps: [
          'Sample random z vectors from N(0,1) and decode them into a grid of novel images',
          'Interpolate between two images in latent space and observe coherent transitions',
          'Compare pixel-space interpolation (ghostly overlay) with latent-space interpolation',
          'Perform latent arithmetic to transfer attributes between categories',
          'Visualize the latent space with t-SNE and interpret the cluster structure',
          'Compare VAE output quality to original images and Stable Diffusion outputs',
        ],
      },
    },
  ],
}

/**
 * Diffusion Models
 *
 * Module 6.2: Diffusion
 * 1. The Diffusion Idea
 */
const diffusion: CurriculumNode = {
  slug: 'diffusion',
  title: 'Diffusion Models',
  children: [
    {
      slug: 'the-diffusion-idea',
      title: 'The Diffusion Idea',
      description:
        'Destruction is easy. Creation from scratch is impossibly hard. But undoing one small step of destruction is learnable\u2014and that is the entire idea behind diffusion models.',
      duration: '20 min',
      category: 'Diffusion',
      objectives: [
        'Explain why breaking image generation into many small denoising steps makes the problem learnable',
        'Distinguish the forward process (adding noise) from the reverse process (learned denoising)',
        'Connect diffusion to familiar building blocks: conv layers, MSE loss, backprop',
        'Recognize that diffusion generates coarse-to-fine, mirroring CNN feature hierarchies',
      ],
      skills: [
        'forward-process-intuition',
        'reverse-process-intuition',
        'small-steps-insight',
        'diffusion-building-blocks-connection',
      ],
      prerequisites: ['exploring-latent-spaces'],
      exercise: {
        constraints: [
          'Intuition only\u2014no mathematical formulation, noise schedules, or alpha_bar',
          'No training objective or loss function derivation',
          'No sampling algorithm or code',
          'No U-Net or specific architecture details',
          'No score matching, SDEs, or continuous-time formulations',
        ],
        steps: [
          'See the quality gap between VAE samples and diffusion outputs',
          'Understand the forward process as gradual noise addition (ink in water analogy)',
          'See why one-shot reversal fails (pure noise is underdetermined)',
          'Grasp the key insight: small denoising steps are learnable, chained together they produce creation',
          'Explore the interactive noise widget to see what different noise levels look like',
          'Understand that denoising at different noise levels captures different image scales',
          'Connect diffusion building blocks to prior knowledge (conv layers, MSE, backprop)',
        ],
      },
    },
    {
      slug: 'the-forward-process',
      title: 'The Forward Process',
      description:
        'The math behind noise addition\u2014noise schedules, alpha-bar, and a closed-form formula that lets you jump to any timestep in one step.',
      duration: '30 min',
      category: 'Diffusion',
      objectives: [
        'Understand the two Gaussian properties needed for the derivation (addition, variance scaling)',
        'Explain the variance-preserving formulation and why each coefficient exists',
        'Define alpha-bar as the cumulative signal fraction and interpret the noise schedule curve',
        'Derive and use the closed-form formula q(x_t|x_0) to jump to any noise level directly',
      ],
      skills: [
        'gaussian-addition-property',
        'variance-preserving-formulation',
        'noise-schedule',
        'alpha-bar',
        'forward-process-closed-form',
      ],
      prerequisites: ['the-diffusion-idea'],
      exercise: {
        constraints: [
          'Forward process math only\u2014no reverse process or denoising',
          'No training objective or loss function',
          'No sampling algorithm or code implementation',
          'No U-Net or denoising architecture',
          'No score matching, SDEs, or continuous-time formulations',
        ],
        steps: [
          'Learn two key Gaussian properties: addition and variance scaling',
          'See why naive noise addition fails (variance-exploding negative example)',
          'Understand the variance-preserving single-step formula',
          'Define the noise schedule beta_t and understand it as a design choice',
          'Build from beta to alpha to alpha-bar (progressive simplification)',
          'Explore the alpha-bar curve interactively and see its effect on images',
          'Derive the closed-form shortcut by unrolling two steps',
          'Verify the shortcut with a 1D pixel walkthrough',
          'Understand why Gaussian noise specifically enables this derivation',
          'Compare linear vs cosine noise schedules',
        ],
      },
    },
    {
      slug: 'learning-to-denoise',
      title: 'Learning to Denoise',
      description:
        'The DDPM training objective\u2014predict the noise, compare with MSE loss, update weights. After all the forward process math, the training objective is surprisingly simple.',
      duration: '25 min',
      category: 'Diffusion',
      objectives: [
        'Explain why the model predicts noise (\u03B5) rather than the clean image (x\u2080)',
        'Trace the complete DDPM training algorithm step by step',
        'Recognize that the DDPM loss is the same MSE formula from Series 1',
        'Connect MSE loss across three contexts: regression, reconstruction, and noise prediction',
      ],
      skills: [
        'ddpm-training-objective',
        'noise-prediction-rationale',
        'ddpm-training-algorithm',
        'mse-loss-generalization',
      ],
      prerequisites: ['the-forward-process'],
      exercise: {
        constraints: [
          'Training objective and algorithm only\u2014no reverse process or sampling',
          'No U-Net architecture or timestep conditioning mechanism',
          'No code implementation\u2014that is the capstone lesson',
          'No score matching, SDEs, or full variational lower bound derivation',
        ],
        steps: [
          'See the DDPM loss formula and recognize it as MSE',
          'Understand why predicting noise gives a more consistent target than predicting the clean image',
          'See the algebraic equivalence: predicted noise can recover x\u2080',
          'Trace the 7-step training algorithm with concrete values',
          'Understand that training samples one random timestep per image, not a sequence',
          'Compare MSE loss across three contexts: regression, autoencoder, DDPM',
          'Recognize that one network handles all timesteps via conditioning',
        ],
      },
    },
    {
      slug: 'sampling-and-generation',
      title: 'Sampling and Generation',
      description:
        'The DDPM reverse process\u2014start from pure noise, iteratively denoise, and generate an image that has never existed. Trace the sampling algorithm step by step.',
      duration: '25 min',
      category: 'Diffusion',
      objectives: [
        'Explain why one-shot denoising fails and iterative denoising is necessary',
        'Trace the reverse step formula term by term and explain what each part does',
        'Connect stochastic noise injection to temperature in language models',
        'Walk through the complete sampling algorithm from t=T to t=0',
        'Visualize the coarse-to-fine denoising trajectory and explain why early steps matter most',
      ],
      skills: [
        'ddpm-reverse-step-formula',
        'stochastic-sampling',
        'sampling-algorithm',
        'coarse-to-fine-generation',
        'sampling-computational-cost',
      ],
      prerequisites: ['learning-to-denoise'],
      exercise: {
        constraints: [
          'Understand the sampling algorithm\u2014no code implementation',
          'No DDIM or accelerated samplers\u2014those come later',
          'No classifier-free guidance or conditional generation',
          'No U-Net architecture internals',
          'No score matching, SDEs, or continuous-time formulations',
        ],
        steps: [
          'See why one-shot denoising produces blurry results (negative example)',
          'Learn the reverse step formula and break it down term by term',
          'Connect the formula to the forward process and reparameterization trick',
          'Understand why noise is added back during sampling (temperature analogy)',
          'Explore the interactive denoising trajectory widget',
          'Trace the full sampling algorithm as pseudocode',
          'Walk through a concrete numerical example at t=500',
        ],
      },
    },
    {
      slug: 'build-a-diffusion-model',
      title: 'Build a Diffusion Model',
      description:
        'Implement the complete DDPM pipeline from scratch\u2014forward process, training loop, and sampling\u2014on real image data. Generate images from pure noise and experience the cost of pixel-space diffusion.',
      duration: '90 min',
      category: 'Diffusion',
      objectives: [
        'Implement the forward process (noise schedule, alpha-bar, closed-form formula) in PyTorch',
        'Build a minimal denoising network with skip connections and timestep conditioning',
        'Write the full DDPM training loop and train on MNIST',
        'Implement the sampling algorithm and generate images from pure noise',
        'Experience the computational cost of 1,000-step pixel-space sampling firsthand',
      ],
      skills: [
        'forward-process-implementation',
        'simple-unet-architecture',
        'ddpm-training-implementation',
        'ddpm-sampling-implementation',
        'pixel-space-diffusion-limitations',
      ],
      prerequisites: ['sampling-and-generation'],
      exercise: {
        constraints: [
          'Minimal architecture only\u2014no attention, group norm, or sinusoidal embeddings',
          'MNIST (28\u00d728) only\u2014no high-resolution images',
          'No conditional generation or classifier-free guidance',
          'No DDIM or accelerated samplers',
          'No new theoretical concepts\u2014pure implementation of existing knowledge',
        ],
        steps: [
          'Implement the noise schedule and compute alpha-bar',
          'Implement the forward process as a q_sample() function',
          'Read and understand the minimal U-Net architecture',
          'Fill in the DDPM training loop with diffusion-specific steps',
          'Train on MNIST and observe the loss curve',
          'Implement the sampling loop with the reverse step formula',
          'Generate a grid of images and measure the sampling time',
          'Compare VAE vs diffusion generation quality and speed',
        ],
      },
    },
  ],
}

/**
 * Architecture & Conditioning
 *
 * Module 6.3: Architecture & Conditioning
 * 1. U-Net Architecture
 * 2. Conditioning the U-Net
 * (more lessons to follow: CLIP, Text Conditioning & Guidance)
 */
const architectureAndConditioning: CurriculumNode = {
  slug: 'architecture-and-conditioning',
  title: 'Architecture & Conditioning',
  children: [
    {
      slug: 'unet-architecture',
      title: 'The U-Net Architecture',
      description:
        'Why the denoising network is shaped like an hourglass with side doors\u2014and why every piece of that shape is essential for multi-scale denoising.',
      duration: '25 min',
      category: 'Architecture & Conditioning',
      objectives: [
        'Explain why a stack of same-resolution conv layers fails at heavy noise (limited receptive field)',
        'Trace the encoder-decoder architecture with tensor dimensions at each level',
        'Articulate why skip connections are essential (not optional) for pixel-precise denoising',
        'Connect the multi-resolution structure to the coarse-to-fine denoising progression',
        'Distinguish the U-Net from a plain autoencoder and explain why both paths are needed',
      ],
      skills: [
        'unet-encoder-decoder',
        'unet-skip-connections',
        'multi-resolution-denoising',
        'bottleneck-global-context',
        'residual-blocks-in-unet',
      ],
      prerequisites: ['build-a-diffusion-model'],
      exercise: {
        constraints: [
          'Spatial architecture only\u2014no timestep conditioning',
          'No attention layers (self-attention, cross-attention)\u2014mentioned only',
          'No text conditioning or guided generation',
          'No implementation\u2014conceptual/theoretical only',
          'No group normalization details\u2014mentioned as a component',
        ],
        steps: [
          'See why same-resolution conv stacks fail at heavy noise',
          'Trace the encoder path as a CNN feature hierarchy with specific dimensions',
          'Understand the bottleneck as global context (same principle as autoencoders)',
          'See what happens without skip connections (blurry autoencoder problem)',
          'Understand skip connections as essential information bypass',
          'Map multi-resolution processing to coarse-to-fine denoising',
          'Read pseudocode for the U-Net forward pass',
          'Predict the effect of removing skip connections at different noise levels',
        ],
      },
    },
    {
      slug: 'conditioning-the-unet',
      title: 'Conditioning the U-Net',
      description:
        'How sinusoidal embeddings and adaptive normalization let a single U-Net handle every noise level\u2014from pure static to nearly clean.',
      duration: '30 min',
      category: 'Architecture & Conditioning',
      objectives: [
        'Connect sinusoidal timestep embedding to positional encoding from transformers (same formula, different input)',
        'Explain why sinusoidal encoding is superior to the simple linear projection from the capstone',
        'Understand adaptive group normalization as timestep-dependent gamma and beta after standard normalization',
        'Articulate why conditioning must happen at every residual block, not just the input or bottleneck',
        'Trace the complete U-Net forward pass with timestep conditioning',
      ],
      skills: [
        'sinusoidal-timestep-embedding',
        'adaptive-group-normalization',
        'group-normalization',
        'global-conditioning-pattern',
        'film-conditioning',
      ],
      prerequisites: ['unet-architecture'],
      exercise: {
        constraints: [
          'Timestep conditioning only\u2014no text conditioning or cross-attention',
          'No CLIP or text embeddings',
          'No classifier-free guidance',
          'No full implementation\u2014conceptual BUILD lesson; implementation is the Module 6.4 capstone',
          'Group normalization introduced, not developed in detail',
        ],
        steps: [
          'Understand group normalization as the middle ground between batch norm and layer norm',
          'See the sinusoidal timestep formula and connect it to positional encoding',
          'Compare simple linear projection vs sinusoidal embedding with concrete examples',
          'Understand the MLP refinement step (sinusoidal encoding -> MLP -> timestep embedding)',
          'See adaptive group normalization as the minimal delta from standard normalization',
          'Walk through concrete gamma(t) and beta(t) values at different timesteps',
          'Understand per-block projection as the "learned lens" pattern from attention',
          'See why input-only conditioning fails (dilution through conv layers)',
          'Trace the complete U-Net forward pass with timestep conditioning',
          'Implement sinusoidal encoding and adaptive group norm in notebook exercises',
        ],
      },
    },
    {
      slug: 'clip',
      title: 'CLIP',
      description:
        'How contrastive learning trains two separate encoders to put matching text and images near each other in a shared embedding space\u2014giving the U-Net a way to understand what to generate.',
      duration: '30 min',
      category: 'Architecture & Conditioning',
      objectives: [
        'Explain contrastive learning as a training paradigm: push matching pairs together, non-matching pairs apart',
        'Describe CLIP\u2019s dual-encoder architecture and why both encoders must be trained together',
        'Understand the shared embedding space and why it enables text-image comparison',
        'Read the contrastive loss formula and connect each part to cross-entropy',
        'Explain zero-shot classification as an emergent property of the shared space',
        'Identify what CLIP does and does not understand (typographic attacks, spatial reasoning, counting)',
      ],
      skills: [
        'contrastive-learning',
        'clip-dual-encoder',
        'shared-embedding-space',
        'contrastive-loss-function',
        'zero-shot-transfer',
      ],
      prerequisites: ['conditioning-the-unet'],
      exercise: {
        constraints: [
          'CLIP as a standalone concept\u2014no connection to the U-Net yet',
          'No cross-attention or classifier-free guidance',
          'No CLIP implementation from scratch\u2014understanding the concept is the goal',
          'No Vision Transformer architecture details\u2014mentioned, not developed',
          'No CLIP variants (SigLIP, OpenCLIP) or fine-tuning',
        ],
        steps: [
          'Compute a 4\u00d74 cosine similarity matrix from pre-computed embeddings',
          'Identify the diagonal as the correct matches and connect to cross-entropy labels',
          'Explore a pretrained CLIP model and visualize the similarity matrix as a heatmap',
          'Perform zero-shot classification on CIFAR-10 using text prompts',
          'Probe CLIP\u2019s limitations with spatial, counting, and adversarial examples',
        ],
      },
    },
    {
      slug: 'text-conditioning-and-guidance',
      title: 'Text Conditioning & Guidance',
      description:
        'How cross-attention injects CLIP text embeddings into the U-Net\u2014and how classifier-free guidance amplifies their influence at inference time.',
      duration: '30 min',
      category: 'Architecture & Conditioning',
      objectives: [
        'Explain cross-attention as the same QKV formula with Q from spatial features, K/V from text embeddings',
        'Trace the per-spatial-location nature of cross-attention (different locations attend to different words)',
        'Identify where cross-attention layers sit in the U-Net (interleaved at 16\u00d716 and 32\u00d732)',
        'Describe the CFG training trick (random text dropout) and inference formula',
        'Explain the guidance scale tradeoff (diversity vs text fidelity vs image quality)',
        'Contrast global conditioning (timestep via adaptive norm) with spatially-varying conditioning (text via cross-attention)',
      ],
      skills: [
        'cross-attention-mechanism',
        'spatially-varying-conditioning',
        'classifier-free-guidance',
        'guidance-scale-tradeoff',
        'unet-block-ordering',
      ],
      prerequisites: ['clip'],
      exercise: {
        constraints: [
          'Cross-attention mechanism and CFG only\u2014no implementation from scratch (that is the Module 6.4 capstone)',
          'No latent diffusion or VAE encoder-decoder\u2014next lesson',
          'No negative prompts or prompt engineering techniques',
          'No CLIP architecture details\u2014covered in the previous lesson',
          'No alternative text conditioning approaches (T5 in Imagen)',
        ],
        steps: [
          'Modify self-attention to cross-attention by changing the source of K and V',
          'Predict and verify the non-square attention weight matrix shape',
          'Visualize cross-attention weights as a heatmap showing per-spatial-location patterns',
          'Implement the CFG formula and test at multiple guidance scales',
          'Generate images at different guidance scales and identify the quality/fidelity tradeoff',
        ],
      },
    },
    {
      slug: 'from-pixels-to-latents',
      title: 'From Pixels to Latents',
      description:
        'The diffusion algorithm you built is identical in latent space\u2014the VAE compresses, diffusion denoises, and Stable Diffusion is born.',
      duration: '25 min',
      category: 'Architecture & Conditioning',
      objectives: [
        'Explain that latent diffusion runs the same DDPM algorithm in the VAE\u2019s compressed latent space',
        'Trace the complete encode-diffuse-decode pipeline with specific tensor dimensions (512\u00d7512\u00d73 \u2192 64\u00d764\u00d74 \u2192 512\u00d7512\u00d73)',
        'Describe the frozen-VAE pattern and explain why the VAE is trained separately and never modified',
        'Compute the 48\u00d7 compression ratio and explain why latent space makes diffusion practical',
        'Identify every component of Stable Diffusion and explain what problem each solves',
      ],
      skills: [
        'latent-diffusion-architecture',
        'frozen-vae-pattern',
        'encode-diffuse-decode-pipeline',
        'compression-ratio-computation',
        'stable-diffusion-component-map',
      ],
      prerequisites: ['text-conditioning-and-guidance'],
      exercise: {
        constraints: [
          'No new algorithms or math\u2014this is a CONSOLIDATE lesson',
          'No implementing latent diffusion from scratch\u2014that is the Module 6.4 capstone',
          'No DDIM or accelerated samplers',
          'No perceptual loss or adversarial training details\u2014mentioned only',
          'No SD v1 vs v2 vs XL differences, LoRA, or fine-tuning',
        ],
        steps: [
          'Explore SD\u2019s VAE encoder and decoder with a pre-trained model',
          'Visualize the 4-channel latent representation and interpolate between encoded images',
          'Compute the compression ratio and estimate computational cost savings',
          'Trace the full encode-diffuse-decode pipeline step by step',
          'Identify which parts of the pipeline are identical to the Module 6.2 implementation',
        ],
      },
    },
  ],
}

/**
 * Stable Diffusion
 *
 * Module 6.4: Stable Diffusion
 * 1. Stable Diffusion Architecture (CONSOLIDATE — assembles all components from 6.1–6.3)
 * 2. Samplers and Efficiency (planned)
 * 3. Generate with Stable Diffusion (planned)
 */
const stableDiffusionModule: CurriculumNode = {
  slug: 'stable-diffusion-module',
  title: 'Stable Diffusion',
  children: [
    {
      slug: 'stable-diffusion-architecture',
      title: 'The Stable Diffusion Pipeline',
      description:
        'Every component you built across 14 lessons, assembled into one system. Trace the complete pipeline from text prompt to generated image with real tensor shapes at every stage.',
      duration: '30 min',
      category: 'Stable Diffusion',
      objectives: [
        'Trace the complete Stable Diffusion inference pipeline from text prompt to pixel image with tensor shapes at every handoff',
        'Explain how CLIP, the conditioned U-Net, and the VAE work together as independently trained components',
        'Describe the denoising loop with all conditioning mechanisms firing simultaneously (timestep, cross-attention, CFG)',
        'Contrast the training pipeline (VAE encoder used) with the inference pipeline (VAE encoder not used)',
        'Explain negative prompts as a direct application of the CFG formula',
      ],
      skills: [
        'sd-pipeline-trace',
        'component-modularity',
        'training-vs-inference-pipeline',
        'negative-prompts',
        'sd-tensor-shapes',
      ],
      prerequisites: ['from-pixels-to-latents'],
      exercise: {
        constraints: [
          'No new concepts\u2014this is a CONSOLIDATE lesson assembling prior knowledge',
          'No implementing the pipeline from scratch\u2014notebook uses diffusers',
          'No samplers beyond DDPM (DDIM, Euler, DPM-Solver are next lesson)',
          'No SD v1 vs v2 vs XL differences, LoRA, or fine-tuning',
        ],
        steps: [
          'Load and inspect the three SD components separately (CLIP, VAE, U-Net) and verify parameter counts',
          'Trace the CLIP stage: tokenize a prompt, inspect token IDs and output embedding shape',
          'Trace one denoising step: embed timestep, two U-Net passes for CFG, scheduler step',
          'Execute the complete pipeline manually and verify every tensor shape matches the lesson predictions',
        ],
      },
    },
    {
      slug: 'samplers-and-efficiency',
      title: 'Samplers and Efficiency',
      description:
        'The model predicts noise. The sampler decides what to do with that prediction. Same weights, different walkers\u2014from 1000 steps to 20.',
      duration: '35 min',
      category: 'Stable Diffusion',
      objectives: [
        'Explain why DDPM requires many steps (Markov chain assumption, adjacent-step formula)',
        'Describe DDIM\u2019s predict-x\u2080-then-jump mechanism and why it enables step-skipping',
        'Connect the ODE perspective to gradient descent (Euler\u2019s method bridge)',
        'Explain why higher-order solvers (DPM-Solver) achieve good quality at 20 steps',
        'Choose a sampler with understanding: DPM-Solver++ (default), DDIM (reproducibility), Euler (debugging)',
      ],
      skills: [
        'ddim-predict-and-leap',
        'ode-perspective-diffusion',
        'euler-method-connection',
        'higher-order-solvers',
        'sampler-selection',
        'deterministic-vs-stochastic-sampling',
      ],
      prerequisites: ['stable-diffusion-architecture'],
      exercise: {
        constraints: [
          'No deriving DDIM or DPM-Solver from first principles',
          'No score-based models or SDE/ODE duality in full rigor',
          'No implementing samplers from scratch\u2014notebook uses diffusers schedulers',
          'No ancestral sampling, Karras samplers, or UniPC',
          'No training-based acceleration (distillation, consistency models)',
        ],
        steps: [
          'Same model, three samplers: DDPM at 1000 steps, DDIM at 50, DPM-Solver at 20. Compare quality and timing',
          'DDIM determinism: generate 3 images with the same seed and verify pixel-level identity',
          'Step count exploration: generate at 5, 10, 20, 50, 100, 200 steps with DPM-Solver. Find the sweet spot',
          'Inspect DDIM intermediates at 10 steps: VAE-decode each intermediate latent and observe coarse-to-fine progression',
        ],
      },
    },
    {
      slug: 'generate-with-stable-diffusion',
      title: 'Generate with Stable Diffusion',
      description:
        'Every parameter in the diffusers API maps to a concept you built from scratch. Use the real library and understand every parameter\u2014this is you driving a machine you built.',
      duration: '30 min',
      category: 'Stable Diffusion',
      objectives: [
        'Use the diffusers StableDiffusionPipeline to generate images with full parameter comprehension',
        'Map every API parameter to its underlying concept and source lesson',
        'Predict the effect of parameter changes before running the code, then verify',
        'Diagnose common failure modes by reasoning from the underlying mechanisms',
        'Apply a systematic experimental workflow for parameter exploration',
      ],
      skills: [
        'diffusers-pipeline-usage',
        'parameter-concept-mapping',
        'guidance-scale-tuning',
        'sampler-selection-applied',
        'negative-prompt-usage',
        'seed-reproducibility',
        'systematic-experimentation',
      ],
      prerequisites: ['samplers-and-efficiency'],
      exercise: {
        constraints: [
          'No new mathematical formulas or derivations\u2014this is CONSOLIDATE',
          'No fine-tuning, LoRA, or customization (Module 6.5)',
          'No img2img, inpainting, or ControlNet',
          'No prompt engineering as a deep topic',
          'No SD v1 vs v2 vs XL differences',
          'No advanced pipeline configurations (custom pipelines, callbacks, attention processors)',
        ],
        steps: [
          'Load StableDiffusionPipeline and generate your first image with default parameters',
          'Guidance scale sweep: generate at guidance_scale = 1, 3, 7.5, 15, 25 with fixed seed and prompt',
          'Step count and scheduler comparison: generate at 5, 10, 20, 50, 100 steps with DPM-Solver++, then swap to DDIM and Euler',
          'Design your own controlled experiment: pick a parameter, form a hypothesis, generate, and write a conclusion',
        ],
      },
    },
  ],
}

/**
 * Customization & Fine-Tuning
 *
 * Module 6.5: Customization
 * 1. LoRA Fine-Tuning
 * 2. Img2Img and Inpainting (planned)
 * 3. Textual Inversion (planned)
 */
const customization: CurriculumNode = {
  slug: 'customization',
  title: 'Customization & Fine-Tuning',
  children: [
    {
      slug: 'lora-finetuning',
      title: 'LoRA Fine-Tuning for Diffusion Models',
      description:
        'Same detour, different highway. Apply LoRA to the Stable Diffusion U-Net for style and subject customization\u2014without retraining 860M parameters.',
      duration: '35 min',
      category: 'Customization',
      objectives: [
        'Explain why cross-attention projections are the primary LoRA target in the diffusion U-Net',
        'Trace one complete diffusion LoRA training step end to end with tensor shapes',
        'Compare the diffusion LoRA training loop to the LLM LoRA training loop and identify what transfers and what changes',
        'Distinguish style LoRA data strategies from subject LoRA data strategies',
        'Describe multiple LoRA composition and its practical limitations',
      ],
      skills: [
        'diffusion-lora-target-layers',
        'diffusion-lora-training-loop',
        'style-vs-subject-lora',
        'lora-composition',
        'diffusion-lora-hyperparameters',
      ],
      prerequisites: ['generate-with-stable-diffusion'],
      exercise: {
        constraints: [
          'Not reimplementing LoRA from scratch\u2014done in LoRA and Quantization',
          'Not DreamBooth, textual inversion, or other fine-tuning techniques',
          'Not ControlNet or structural conditioning',
          'Not SD v1 vs v2 vs XL LoRA differences',
          'Not hyperparameter optimization beyond practical guidance',
        ],
        steps: [
          'Inspect LoRA target layers: list cross-attention projections, compute param counts for rank-4 vs rank-16',
          'Trace one LoRA training step: VAE encode, sample timestep, add noise, U-Net forward, MSE loss, verify gradient flow',
          'Train a style LoRA with diffusers + PEFT on a small dataset, compare with and without LoRA',
          'LoRA composition experiment: load two adapters, apply individually and together, experiment with alpha scaling',
        ],
      },
    },
  ],
}

/**
 * Stable Diffusion & Image Generation
 *
 * Series 6: From generative foundations through diffusion models
 * to understanding Stable Diffusion.
 */
export const stableDiffusion: CurriculumNode = {
  slug: 'stable-diffusion',
  title: 'Stable Diffusion & Image Generation',
  icon: 'Image',
  description:
    'Generative foundations, diffusion models, and Stable Diffusion\u2014from autoencoders to text-to-image generation',
  children: [
    generativeFoundations,
    diffusion,
    architectureAndConditioning,
    stableDiffusionModule,
    customization,
  ],
}
