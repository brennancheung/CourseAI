import { CurriculumNode } from './types'

/**
 * Convolutions - the core operation and architecture
 *
 * Module 3.1: Convolutions
 * 1. What Convolutions Compute
 * 2. Building CNN Architecture (planned)
 * 3. Training Your First CNN (planned)
 */
const convolutions: CurriculumNode = {
  slug: 'convolutions',
  title: 'Convolutions',
  children: [
    {
      slug: 'what-convolutions-compute',
      title: 'What Convolutions Compute',
      description:
        'The core operation: slide a small filter across an image, computing weighted sums at each position',
      duration: '25 min',
      category: 'Convolutions',
      objectives: [
        'Compute what a convolutional filter produces on a 2D grid',
        'Understand feature maps as spatial pattern-detection output',
        'Explain why weight sharing and locality beat dense layers for images',
        'Predict filter output for edge detection examples',
      ],
      skills: [
        'convolutions',
        'feature-maps',
        'edge-detection',
        'weight-sharing',
        'spatial-locality',
      ],
      prerequisites: ['overfitting-and-regularization'],
      exercise: {
        constraints: [
          'Convolution operation and intuition only',
          'No pooling, stride, padding, or full CNN architecture',
          'No training or nn.Conv2d in depth',
        ],
        steps: [
          'Understand the flat vector problem with images',
          'Compute the convolution operation step by step',
          'See how different filters detect different features',
          'Explore weight sharing and parameter efficiency',
          'Experiment with the ConvolutionExplorer widget',
        ],
      },
    },
    {
      slug: 'building-a-cnn',
      title: 'Building a CNN',
      description:
        'Assemble convolutions into a full architecture with pooling, stride, padding, and the classic conv-pool-fc pattern',
      duration: '25 min',
      category: 'Convolutions',
      objectives: [
        'Explain how max pooling shrinks spatial dimensions while preserving features',
        'Compute output dimensions using the general formula with stride and padding',
        'Trace data shapes through a complete conv-pool-fc architecture',
        'Read a CNN architecture in PyTorch and identify each layer\'s role',
      ],
      skills: [
        'pooling',
        'stride',
        'padding',
        'cnn-architecture',
        'dimension-tracking',
      ],
      prerequisites: ['what-convolutions-compute'],
      exercise: {
        constraints: [
          'Architecture assembly and dimension tracking only',
          'No training a CNN or comparing accuracy',
          'No advanced architectures (ResNet, VGG, etc.)',
        ],
        steps: [
          'Understand why feature maps need to shrink',
          'Compute max pooling on concrete feature maps',
          'See how stride and padding control output dimensions',
          'Learn the general output size formula',
          'Build a CNN layer by layer in the dimension calculator',
          'Trace data through the conv-pool-fc pattern',
        ],
      },
    },
    {
      slug: 'mnist-cnn-project',
      title: 'MNIST CNN Project',
      description:
        'Build a CNN for MNIST in PyTorch, train it, and see it beat the dense network with fewer parameters',
      duration: '30 min',
      category: 'Convolutions',
      objectives: [
        'Implement a CNN architecture in PyTorch using nn.Conv2d and nn.MaxPool2d',
        'Train the CNN on MNIST and compare accuracy to the dense network',
        'Explain why the CNN wins with fewer parameters (weight sharing, spatial locality)',
        'Articulate the core insight: architecture encodes assumptions about data',
      ],
      skills: [
        'cnn-implementation',
        'model-comparison',
        'parameter-counting',
        'weight-sharing',
        'architecture-design',
      ],
      prerequisites: ['building-a-cnn'],
      exercise: {
        constraints: [
          'No new concepts — assembly and comparison only',
          'No hyperparameter tuning or advanced techniques',
          'Focus on understanding why architecture matters, not achieving highest accuracy',
        ],
        steps: [
          'Review the CNN architecture from Building a CNN',
          'Fill in the CNN class skeleton in the Colab notebook',
          'Verify dimensions with a test forward pass',
          'Train the CNN and compare accuracy to the dense network',
          'Compare parameter counts and explain the difference',
        ],
      },
    },
  ],
}

/**
 * Modern Architectures - how CNNs evolved from LeNet to ResNet
 *
 * Module 3.2: Modern Architectures
 * 1. Architecture Evolution (LeNet -> AlexNet -> VGG)
 * 2. ResNets (planned)
 * 3. Transfer Learning (planned)
 */
const modernArchitectures: CurriculumNode = {
  slug: 'modern-architectures',
  title: 'Modern Architectures',
  children: [
    {
      slug: 'architecture-evolution',
      title: 'Architecture Evolution',
      description:
        'How CNNs evolved from LeNet to AlexNet to VGG\u2014and why going deeper works',
      duration: '25 min',
      category: 'Modern Architectures',
      objectives: [
        'Trace the architectural progression from LeNet to AlexNet to VGG',
        'Explain why deeper networks learn richer hierarchical features',
        'Calculate the receptive field and parameter efficiency of stacked 3x3 filters vs larger filters',
        'Identify the degradation problem that limits naive depth scaling',
      ],
      skills: [
        'architecture-evolution',
        'receptive-field',
        'parameter-efficiency',
        'lenet',
        'alexnet',
        'vgg',
      ],
      prerequisites: ['mnist-cnn-project'],
      exercise: {
        constraints: [
          'Conceptual understanding only\u2014no implementation',
          'LeNet/AlexNet/VGG comparison only\u2014no ResNets',
          'No training these architectures\u2014focus on design principles',
        ],
        steps: [
          'Map your MNIST CNN to LeNet architecture',
          'Identify AlexNet\u2019s innovations (ReLU, dropout, GPU scale)',
          'Calculate receptive field for stacked 3x3 convolutions',
          'Compare parameter counts: 3x3+3x3 vs 5x5, 3x3+3x3+3x3 vs 7x7',
          'Explore architectures in the comparison widget',
          'Understand the degradation problem and why deeper is not always better',
        ],
      },
    },
    {
      slug: 'resnets',
      title: 'ResNets and Skip Connections',
      description:
        'Why deeper networks fail to train\u2014and the elegant residual connection that made 152-layer networks possible',
      duration: '30 min',
      category: 'Modern Architectures',
      objectives: [
        'Explain why the degradation problem occurs (identity mapping argument)',
        'Describe how residual connections solve it (F(x) + x formulation)',
        'Implement a ResNet block in PyTorch with Conv-BN-ReLU + shortcut',
        'Use batch normalization in a CNN (nn.BatchNorm2d, train/eval mode)',
        'Compare a plain deep CNN to a ResNet on CIFAR-10',
      ],
      skills: [
        'resnet',
        'skip-connections',
        'residual-learning',
        'batch-norm-cnn',
        'identity-shortcut',
        'projection-shortcut',
        'global-average-pooling',
      ],
      prerequisites: ['architecture-evolution'],
      exercise: {
        constraints: [
          'Basic ResNet blocks only\u2014no bottleneck architecture',
          'No ResNet variants (ResNeXt, DenseNet, WideResNet)',
          'No training tricks (LR scheduling, data augmentation, warmup)',
        ],
        steps: [
          'Understand the degradation problem and identity mapping argument',
          'Learn the residual connection formulation: H(x) = F(x) + x',
          'Explore plain vs residual blocks in the interactive widget',
          'Implement a ResidualBlock class in PyTorch',
          'Build a small ResNet and train on CIFAR-10',
          'Compare accuracy to a plain network of similar depth',
        ],
      },
    },
    {
      slug: 'transfer-learning',
      title: 'Transfer Learning',
      description:
        'Take a pretrained ResNet and adapt it to new tasks in minutes\u2014the practical payoff of CNN architectures',
      duration: '25 min',
      category: 'Modern Architectures',
      objectives: [
        'Load a pretrained ResNet via torchvision.models and inspect its architecture',
        'Perform feature extraction: freeze the backbone and replace the classification head',
        'Perform fine-tuning: selectively unfreeze layers with differential learning rates',
        'Choose between feature extraction and fine-tuning based on dataset size and domain similarity',
        'Apply data augmentation transforms to small datasets',
      ],
      skills: [
        'transfer-learning',
        'pretrained-models',
        'feature-extraction',
        'fine-tuning',
        'data-augmentation',
        'torchvision-models',
      ],
      prerequisites: ['resnets'],
      exercise: {
        constraints: [
          'ResNet-18 only\u2014no EfficientNet, ViT, or other architectures',
          'Image classification only\u2014no detection, segmentation, or generation',
          'No knowledge distillation or self-supervised pretraining',
          'No advanced augmentation (Mixup, CutMix, AutoAugment)',
        ],
        steps: [
          'Understand why training from scratch fails on small datasets',
          'Load a pretrained ResNet-18 and inspect model.fc',
          'Freeze the backbone and replace the classification head',
          'Train the head on a small dataset (feature extraction)',
          'Unfreeze the last stage and fine-tune with differential learning rates',
          'Compare all three approaches: from scratch, feature extraction, fine-tuning',
        ],
      },
    },
  ],
}

/**
 * Seeing What CNNs See - visualization and interpretability
 *
 * Module 3.3: Seeing What CNNs See
 * 1. Visualizing Features (filter viz, activation maps, Grad-CAM)
 * 2. Fine-tuning with Visualization (planned)
 */
const seeingWhatCnnsSee: CurriculumNode = {
  slug: 'seeing-what-cnns-see',
  title: 'Seeing What CNNs See',
  children: [
    {
      slug: 'visualizing-features',
      title: 'Visualizing Features',
      description:
        'Open the black box\u2014see the filters a CNN learned, watch activations at different layers, and use Grad-CAM to ask why the model made a specific prediction',
      duration: '30 min',
      category: 'Seeing What CNNs See',
      objectives: [
        'Visualize conv1 filter weights as images and interpret them as edge/color detectors',
        'Use register_forward_hook to capture activation maps at any layer without modifying the model',
        'Observe the feature hierarchy in action: edges (conv1) to textures (layer2) to abstract patterns (layer4)',
        'Implement Grad-CAM to produce class-specific spatial heatmaps',
        'Identify shortcut learning using Grad-CAM as a debugging tool',
      ],
      skills: [
        'filter-visualization',
        'activation-maps',
        'pytorch-hooks',
        'grad-cam',
        'model-interpretability',
        'shortcut-learning',
      ],
      prerequisites: ['transfer-learning'],
      exercise: {
        constraints: [
          'Three visualization techniques only\u2014no deep dream, saliency maps, or model-agnostic methods',
          'PyTorch hooks introduced for activation capture only\u2014no custom hook patterns',
          'Grad-CAM only\u2014no Grad-CAM++ or other attribution variants',
          'Interpretation and debugging focus\u2014no training or fine-tuning',
        ],
        steps: [
          'Visualize all 64 conv1 filters as a grid of 7x7 images',
          'Register forward hooks and capture activation maps at conv1, layer2, and layer4',
          'Interpret the feature hierarchy: concrete edges to abstract patterns',
          'Implement Grad-CAM step by step (forward hook, backward hook, gradient weighting)',
          'Compare Grad-CAM heatmaps for different classes on the same image',
          'Investigate a shortcut learning example with Grad-CAM',
        ],
      },
    },
    {
      slug: 'transfer-learning-project',
      title: 'Project: Transfer Learning',
      description:
        'Fine-tune a pretrained CNN on a small flower dataset and use Grad-CAM to verify the model learned the right features\u2014not shortcuts',
      duration: '45 min',
      category: 'Seeing What CNNs See',
      objectives: [
        'Execute the complete transfer learning practitioner workflow on a new dataset',
        'Fine-tune a pretrained ResNet-18 via feature extraction and compare to fine-tuning',
        'Use Grad-CAM to validate model reasoning beyond accuracy metrics',
        'Diagnose whether a model exhibits shortcut learning on your own fine-tuned model',
        'Compare feature extraction and fine-tuning approaches with both accuracy and visualization',
      ],
      skills: [
        'transfer-learning-workflow',
        'grad-cam-debugging',
        'fine-tuning-practice',
        'shortcut-learning-detection',
        'model-validation',
      ],
      prerequisites: ['visualizing-features'],
      exercise: {
        constraints: [
          'No new concepts\u2014capstone project integrating prior skills',
          'ResNet-18 only\u2014no architecture selection or comparison',
          'No hyperparameter search or advanced augmentation',
          'Focus on workflow execution and interpretation, not optimization',
        ],
        steps: [
          'Explore the Oxford Flowers dataset (class distribution, sample images)',
          'Perform feature extraction: freeze backbone, train classification head',
          'Run Grad-CAM on correctly classified images to validate model focus',
          'Fine-tune with unfrozen layer4 and differential learning rates',
          'Compare accuracy and Grad-CAM heatmaps for both approaches',
        ],
      },
    },
  ],
}

/**
 * CNNs - Convolutional Neural Networks
 */
export const cnns: CurriculumNode = {
  slug: 'cnns',
  title: 'CNNs',
  icon: 'Grid3x3',
  description: 'Convolutional neural networks for visual understanding',
  children: [convolutions, modernArchitectures, seeingWhatCnnsSee],
}
