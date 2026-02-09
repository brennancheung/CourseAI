import { CurriculumNode } from './types'

/**
 * The Learning Problem - introductory lessons
 */
const theLearningProblem: CurriculumNode = {
  slug: 'the-learning-problem',
  title: 'The Learning Problem',
  children: [
    {
      slug: 'core-concepts',
      title: 'Core Concepts',
      children: [
        {
          slug: 'what-is-learning',
          title: 'What is Learning?',
          description: 'Understanding machine learning as function approximation',
          duration: '15 min',
          category: 'Fundamentals',
          objectives: [
            'Understand what "learning" means for a machine',
            'See why generalization beats memorization',
            'Grasp the bias-variance tradeoff intuitively',
            'Learn why we need train/val/test splits',
          ],
          skills: ['ml-intuition', 'generalization'],
          exercise: {
            constraints: [
              'Focus on intuition, not math',
              'No code yet — just concepts',
              'Build mental model first',
            ],
            steps: [
              'Understand what "learning" means for a machine',
              'See why generalization beats memorization',
              'Grasp the bias-variance tradeoff intuitively',
              'Learn why we need train/val/test splits',
            ],
          },
        },
        {
          slug: 'linear-regression',
          title: 'Linear Regression',
          description: 'The simplest model: fitting a line to data',
          duration: '20 min',
          category: 'Fundamentals',
          objectives: [
            'Understand what a linear model is',
            'Learn what parameters (slope, intercept) do',
            'Explore fitting interactively',
            'See why we need a way to measure "goodness" of fit',
          ],
          skills: ['linear-models', 'parameters'],
          prerequisites: ['what-is-learning'],
          exercise: {
            constraints: [
              'Focus on intuition first',
              'Math is secondary to understanding',
              'Play with the interactive widget',
            ],
            steps: [
              'Understand what a linear model is',
              'Learn what parameters (slope, intercept) do',
              'Explore fitting interactively',
              'See why we need a way to measure "goodness" of fit',
            ],
          },
        },
        {
          slug: 'loss-functions',
          title: 'Loss Functions',
          description: 'Measuring how wrong our predictions are',
          duration: '20 min',
          category: 'Fundamentals',
          objectives: [
            'Understand residuals (prediction errors)',
            'Learn why we square the errors (MSE)',
            'See the loss as a landscape',
            'Understand why we need to minimize loss',
          ],
          skills: ['loss-functions', 'mse'],
          prerequisites: ['linear-regression'],
          exercise: {
            constraints: [
              'Understand the intuition first',
              'Then the math',
              'Play with both visualizations',
            ],
            steps: [
              'Understand residuals (prediction errors)',
              'Learn why we square the errors (MSE)',
              'See the loss as a landscape',
              'Understand why we need to minimize loss',
            ],
          },
        },
      ],
    },
    {
      slug: 'optimization',
      title: 'Optimization',
      children: [
        {
          slug: 'gradient-descent',
          title: 'Gradient Descent',
          description: 'Following the slope to find the minimum',
          duration: '20 min',
          category: 'Fundamentals',
          objectives: [
            'Understand the ball-rolling-downhill analogy',
            'Learn what the gradient tells us',
            'See the update rule in action',
            'Experiment with different learning rates',
          ],
          skills: ['gradient-descent', 'optimization'],
          prerequisites: ['loss-functions'],
          exercise: {
            constraints: [
              'Focus on intuition first',
              'Math is secondary',
              'Watch the animation many times',
            ],
            steps: [
              'Understand the ball-rolling-downhill analogy',
              'Learn what the gradient tells us',
              'See the update rule in action',
              'Experiment with different learning rates',
            ],
          },
        },
        {
          slug: 'learning-rate',
          title: 'Learning Rate',
          description: 'The most important hyperparameter',
          duration: '15 min',
          category: 'Fundamentals',
          objectives: [
            'See what happens with learning rate too small',
            'See what happens with learning rate too large',
            'Understand overshooting and divergence',
            'Preview learning rate schedules',
          ],
          skills: ['learning-rate', 'hyperparameters'],
          prerequisites: ['gradient-descent'],
          exercise: {
            constraints: [
              'Focus on developing intuition',
              'Experiment with different values',
              'Watch the behaviors carefully',
            ],
            steps: [
              'See what happens with learning rate too small',
              'See what happens with learning rate too large',
              'Understand overshooting and divergence',
              'Preview learning rate schedules',
            ],
          },
        },
      ],
    },
    {
      slug: 'implementation',
      title: 'Implementation',
      children: [
        {
          slug: 'implementing-linear-regression',
          title: 'Linear Regression from Scratch',
          description: 'Build it yourself in Python',
          duration: '30 min',
          category: 'Fundamentals',
          objectives: [
            'Understand the training loop structure',
            'Derive the gradient formulas',
            'Implement in Python',
            'Watch the model learn',
          ],
          skills: ['python', 'numpy', 'implementation'],
          prerequisites: ['learning-rate'],
          exercise: {
            constraints: [
              'Pure Python + NumPy only',
              'No sklearn or PyTorch',
              'Compute every gradient by hand',
            ],
            steps: [
              'Understand the training loop structure',
              'Derive the gradient formulas',
              'Implement in Python',
              'Watch the model learn',
            ],
          },
        },
      ],
    },
  ],
}

/**
 * From Linear to Neural - transition to neural networks
 *
 * Flat structure - lessons tell the story through their order:
 * 1. What are neurons and neural networks (still linear)
 * 2. Why linear networks fail (XOR problem)
 * 3. Add activation functions (the fix, XOR solved)
 * 4. Reference guide to all activations
 */
const fromLinearToNeural: CurriculumNode = {
  slug: 'from-linear-to-neural',
  title: 'From Linear to Neural',
  children: [
    {
      slug: 'neuron-basics',
      title: 'Neurons and Neural Networks',
      description: 'What neurons compute and how they connect into networks',
      duration: '12 min',
      category: 'From Linear to Neural',
      objectives: [
        'See what a single neuron computes',
        'Connect it to linear regression',
        'Understand layers and networks',
      ],
      skills: ['neurons', 'neural-networks', 'layers'],
      prerequisites: ['implementing-linear-regression'],
      exercise: {
        constraints: [
          'Builds on Module 1.1 concepts',
          'No activation functions yet',
          'Focus on structure, not training',
        ],
        steps: [
          'See what a single neuron computes',
          'Connect it to linear regression',
          'Understand layers and networks',
        ],
      },
    },
    {
      slug: 'limits-of-linearity',
      title: 'The Limits of Linearity',
      description: 'Why linear networks fail on XOR',
      duration: '15 min',
      category: 'From Linear to Neural',
      objectives: [
        'Understand XOR as a classification problem',
        'Try (and fail) to separate XOR with a line',
        "See why linear networks can't solve this",
      ],
      skills: ['linear-models', 'decision-boundaries', 'xor-problem'],
      prerequisites: ['neuron-basics'],
      exercise: {
        constraints: [
          'No math required',
          'Just intuition and experimentation',
          'Try to beat the impossible',
        ],
        steps: [
          'Understand XOR as a classification problem',
          'Try (and fail) to separate XOR with a line',
          "See why linear networks can't solve this",
        ],
      },
    },
    {
      slug: 'activation-functions',
      title: 'Activation Functions',
      description: 'The missing ingredient that solves XOR',
      duration: '20 min',
      category: 'From Linear to Neural',
      objectives: [
        'Add activation functions to our neurons',
        'See how this solves XOR',
        'Explore sigmoid, ReLU, and others',
      ],
      skills: ['activation-functions', 'nonlinearity', 'relu', 'sigmoid'],
      prerequisites: ['limits-of-linearity'],
      exercise: {
        constraints: [
          'Focus on why, then what',
          'See XOR get solved visually',
          'Explore different activation shapes',
        ],
        steps: [
          'Add activation functions to our neurons',
          'See how this solves XOR',
          'Explore sigmoid, ReLU, and others',
        ],
      },
    },
    {
      slug: 'activation-functions-deep-dive',
      title: 'Activation Functions Reference',
      description: 'Visual guide to ReLU, sigmoid, tanh, GELU, and more',
      duration: '15 min',
      category: 'From Linear to Neural',
      objectives: [
        'See how each activation transforms inputs',
        'Understand the basic properties',
        'Learn the simple decision guide',
      ],
      skills: ['relu', 'sigmoid', 'tanh', 'gelu', 'activation-functions'],
      prerequisites: ['activation-functions'],
      exercise: {
        constraints: [
          'Reference material — no memorization needed',
          'Focus on shapes and intuition',
          'The "why" comes later (after backprop)',
        ],
        steps: [
          'See how each activation transforms inputs',
          'Understand the basic properties',
          'Learn the simple decision guide',
        ],
      },
    },
  ],
}

/**
 * Training Neural Networks - how networks actually learn
 *
 * Covers:
 * 1. Backpropagation (the chain rule, gradient flow)
 * Future: Batching, optimizers, regularization
 */
const trainingNeuralNetworks: CurriculumNode = {
  slug: 'training-neural-networks',
  title: 'Training Neural Networks',
  children: [
    {
      slug: 'backpropagation',
      title: 'Backpropagation',
      description: 'How neural networks compute gradients and learn',
      duration: '25 min',
      category: 'Training Neural Networks',
      objectives: [
        'Understand why we need gradients for every parameter',
        'Review the chain rule from calculus',
        'See how gradients flow backward through the network',
        'Walk through a concrete example',
      ],
      skills: ['backpropagation', 'chain-rule', 'gradients'],
      prerequisites: ['activation-functions'],
      exercise: {
        constraints: [
          'Focus on intuition first',
          'The chain rule is the key idea',
          "We'll work through a concrete example",
        ],
        steps: [
          'Understand why we need gradients for every parameter',
          'Review the chain rule from calculus',
          'See how gradients flow backward through the network',
          'Walk through a concrete example',
        ],
      },
    },
    {
      slug: 'backprop-worked-example',
      title: 'Backprop by the Numbers',
      description: 'Trace every gradient through a real neural network with actual numbers',
      duration: '20 min',
      category: 'Training Neural Networks',
      objectives: [
        'Compute forward pass values through a 2-layer network',
        'Trace gradients backward using the chain rule with real numbers',
        'Verify gradients using numerical differentiation',
        'Complete a full training step and watch the loss drop',
      ],
      skills: ['backpropagation', 'gradients', 'numerical-verification'],
      prerequisites: ['backpropagation'],
      exercise: {
        constraints: [
          '2-layer network, 1 neuron per layer, 4 parameters',
          'Real numbers at every step',
          'No matrices — focus on the chain rule mechanics',
        ],
        steps: [
          'Trace the forward pass with concrete values',
          'Compute all 4 gradients via the backward pass',
          'Verify one gradient numerically',
          'Apply the update rule and confirm the loss decreases',
        ],
      },
    },
    {
      slug: 'computational-graphs',
      title: 'Computational Graphs',
      description: 'A visual notation that makes gradient bookkeeping automatic',
      duration: '20 min',
      category: 'Training Neural Networks',
      objectives: [
        'Read and trace computational graphs',
        'Trace forward values left-to-right and gradients right-to-left',
        'Apply the fan-out rule (sum gradients when paths split)',
        'Understand how this connects to PyTorch autograd',
      ],
      skills: ['computational-graphs', 'autograd', 'chain-rule'],
      prerequisites: ['backprop-worked-example'],
      exercise: {
        constraints: [
          'Same math as previous lessons — new visual notation',
          'No code — just the graph framework',
          'Focus on reading graphs, not building them from scratch',
        ],
        steps: [
          'Trace a simple graph: f(x) = (x+1)\u00B2',
          'Map the lesson-2 network onto a computational graph',
          'Learn the fan-out rule with f(x) = x\u00B7(x+1)',
          'Connect graph notation to automatic differentiation',
        ],
      },
    },
    {
      slug: 'batching-and-sgd',
      title: 'Batching and SGD',
      description: 'From one data point at a time to training on real datasets',
      duration: '25 min',
      category: 'Training Neural Networks',
      objectives: [
        'Understand why full-batch gradient descent is too slow for real datasets',
        'Learn how mini-batches approximate the full gradient',
        'Trace the full SGD algorithm: epochs, batches, and updates',
        'See how gradient noise can help escape bad minima',
      ],
      skills: ['mini-batches', 'sgd', 'epochs', 'gradient-noise'],
      prerequisites: ['computational-graphs'],
      exercise: {
        constraints: [
          'No optimizers beyond vanilla SGD — momentum and Adam come next',
          'Focus on mechanics: batching, epochs, and tradeoffs',
          'No GPU parallelism or data loading details',
        ],
        steps: [
          'Understand the scale problem with full-batch gradient descent',
          'See mini-batches as polling: random samples give useful estimates',
          'Trace through: 1000 examples, batch size 50, 20 updates per epoch',
          'Explore how batch size affects optimization paths interactively',
        ],
      },
    },
    {
      slug: 'optimizers',
      title: 'Optimizers',
      description: 'Why vanilla SGD struggles in ravines—and how momentum, RMSProp, and Adam fix it',
      duration: '25 min',
      category: 'Training Neural Networks',
      objectives: [
        'Understand why vanilla SGD struggles on elongated loss landscapes',
        'Learn how momentum smooths gradient direction with EMA',
        'See how RMSProp adapts learning rates per parameter',
        'Understand Adam as momentum + RMSProp combined',
        'Know when SGD might beat Adam',
      ],
      skills: ['momentum', 'adam', 'rmsprop', 'optimizers', 'ema'],
      prerequisites: ['batching-and-sgd'],
      exercise: {
        constraints: [
          'Focus on 3 optimizers: Momentum, RMSProp, Adam',
          'No code implementation — that comes in PyTorch series',
          'No convergence proofs or theoretical guarantees',
        ],
        steps: [
          'See the ravine problem that motivates better optimizers',
          'Learn EMA as a tool for smoothing noisy signals',
          'Understand momentum, RMSProp, and Adam formulas with intuition',
          'Compare optimizer trajectories interactively on the ravine landscape',
        ],
      },
    },
    {
      slug: 'training-dynamics',
      title: 'Training Dynamics',
      description: 'Why deep networks fail to train—and the three ideas that fix it',
      duration: '25 min',
      category: 'Training Neural Networks',
      objectives: [
        'Understand vanishing and exploding gradients as products of local derivatives',
        'Learn why Xavier and He initialization preserve signal through many layers',
        'See how batch normalization stabilizes gradient flow during training',
        'Know the modern baseline: ReLU + He init + batch norm',
      ],
      skills: ['vanishing-gradients', 'exploding-gradients', 'weight-initialization', 'batch-normalization'],
      prerequisites: ['optimizers'],
      exercise: {
        constraints: [
          'No code implementation — that comes in the PyTorch series',
          'Skip connections mentioned but not developed',
          'No full mathematical derivations of initialization formulas',
        ],
        steps: [
          'See why 10-layer sigmoid networks fail to train',
          'Trace gradient magnitudes layer by layer',
          'Compare initialization strategies interactively',
          'Observe how batch normalization stabilizes deep networks',
        ],
      },
    },
    {
      slug: 'overfitting-and-regularization',
      title: 'Overfitting and Regularization',
      description: 'Diagnose overfitting from training curves and apply three techniques to prevent it',
      duration: '25 min',
      category: 'Training Neural Networks',
      objectives: [
        'Diagnose overfitting from training curves (train vs validation loss)',
        'Apply dropout to force robust feature learning',
        'Use weight decay (L2 regularization) to keep functions smooth',
        'Know when to stop training with early stopping and patience',
      ],
      skills: ['overfitting', 'regularization', 'dropout', 'weight-decay', 'early-stopping'],
      prerequisites: ['training-dynamics'],
      exercise: {
        constraints: [
          'Three regularization techniques: dropout, weight decay, early stopping',
          'No PyTorch implementation \u2014 that comes in the next series',
          'No hyperparameter tuning strategies',
        ],
        steps: [
          'Read training curves to diagnose overfitting (the scissors pattern)',
          'See how dropout, weight decay, and early stopping close the gap',
          'Compare techniques interactively with the RegularizationExplorer',
          'Know the priority order: early stopping > AdamW > dropout',
        ],
      },
    },
  ],
}

/**
 * Foundations - everything a beginner needs
 */
export const foundations: CurriculumNode = {
  slug: 'foundations',
  title: 'Foundations',
  icon: 'Layers',
  description: 'Core concepts every ML practitioner needs',
  children: [
    theLearningProblem,
    fromLinearToNeural,
    trainingNeuralNetworks,
  ],
}
