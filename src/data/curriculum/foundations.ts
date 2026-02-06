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
    // Future: add more foundation topics
    // mathForML,
    // practicalSkills,
  ],
}
