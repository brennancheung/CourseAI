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
 * (more lessons to follow: Timestep Conditioning, Self-Attention, Cross-Attention, Classifier-Free Guidance)
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
  children: [generativeFoundations, diffusion, architectureAndConditioning],
}
