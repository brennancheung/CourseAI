import { CurriculumNode } from './types'

/**
 * PyTorch Core — hands-on PyTorch fundamentals
 *
 * Module 2.1: PyTorch Core
 * 1. Tensors
 * 2. Autograd
 * 3. nn.Module
 * 4. Training Loop
 */
const pytorchCore: CurriculumNode = {
  slug: 'pytorch-core',
  title: 'PyTorch Core',
  children: [
    {
      slug: 'tensors',
      title: 'Tensors',
      description: "PyTorch's core data structure — NumPy arrays that can ride the GPU",
      duration: '25 min',
      category: 'PyTorch Core',
      objectives: [
        'Create tensors from data, random values, and NumPy arrays',
        'Understand tensor attributes: shape, dtype, device',
        'Reshape tensors and perform matrix operations',
        'Move tensors between CPU and GPU',
        'Convert between NumPy arrays and PyTorch tensors',
      ],
      skills: ['pytorch', 'tensors', 'gpu', 'numpy-interop'],
      prerequisites: ['implementing-linear-regression'],
      exercise: {
        constraints: [
          'Tensors only — no autograd or requires_grad',
          'No nn.Module or layers',
          'No training loops — data structure first',
        ],
        steps: [
          'Translate NumPy code to PyTorch tensors',
          'Check shape, dtype, and device for each tensor',
          'Move data to GPU and verify the device changed',
          'Compute the forward pass: y_hat = X @ w + b',
          'Reshape a batch of 28x28 images into 784-element vectors',
        ],
      },
    },
    {
      slug: 'autograd',
      title: 'Autograd',
      description: "PyTorch computes gradients for you — using the exact algorithm you already know",
      duration: '25 min',
      category: 'PyTorch Core',
      objectives: [
        'Use requires_grad to tell PyTorch to track operations',
        'Call backward() to compute gradients automatically',
        'Read gradient results from .grad attributes',
        'Handle gradient accumulation with zero_grad()',
        'Use torch.no_grad() to disable tracking during updates',
        'Use .detach() to sever tensors from the computational graph',
      ],
      skills: ['pytorch', 'autograd', 'gradients', 'backpropagation'],
      prerequisites: ['tensors'],
      exercise: {
        constraints: [
          'Autograd only — no nn.Module or layer abstractions',
          'No optimizers like Adam or SGD objects',
          'No full training loops — one manual update step as preview',
        ],
        steps: [
          'Compute gradients for a polynomial function, verify by hand',
          'Reproduce the backprop-worked-example network and compare .grad values',
          'Demonstrate the accumulation trap — predict, run, fix with zero_grad()',
          'Write a single manual training step with no_grad() and zero_grad()',
          '(Stretch) Use detach() to stop gradient flow and verify .grad = None',
        ],
      },
    },
    {
      slug: 'nn-module',
      title: 'nn.Module',
      description: 'Package neurons, layers, and networks into reusable building blocks with automatic parameter management',
      duration: '20 min',
      category: 'PyTorch Core',
      objectives: [
        'Use nn.Linear as a layer that computes y = x @ W.T + b',
        'Define custom nn.Module subclasses with __init__ and forward()',
        'Collect all learnable parameters with model.parameters()',
        'Use nn.Sequential for simple layer stacks',
        'Understand why custom forward() is needed for skip connections',
      ],
      skills: ['pytorch', 'nn-module', 'nn-linear', 'nn-sequential'],
      prerequisites: ['autograd'],
      exercise: {
        constraints: [
          'nn.Module and nn.Linear only — no optimizers',
          'No loss function objects like nn.MSELoss',
          'No full training loops — verify forward + backward only',
        ],
        steps: [
          'Create and inspect nn.Linear layers — print weight shapes, verify parameter counts',
          'Verify nn.Linear IS w*x + b — manually compute x @ weight.T + bias and compare',
          'Build a 2-layer nn.Module subclass with __init__ and forward()',
          'Convert to nn.Sequential — rewrite the same network, verify same output',
          'Linear collapse experiment — build Sequential with and without activations',
          '(Stretch) Build a skip-connection Module with a residual path',
        ],
      },
    },
    {
      slug: 'training-loop',
      title: 'The Training Loop',
      description: 'Assemble tensors, autograd, and nn.Module into a complete PyTorch training loop — the same pattern you already know',
      duration: '20 min',
      category: 'PyTorch Core',
      objectives: [
        'Use nn.MSELoss as a loss function object that wraps the MSE formula',
        'Use torch.optim.SGD and torch.optim.Adam as optimizer objects',
        'Write a complete training loop: forward, loss, backward, step',
        'Swap optimizers with a one-line change',
        'Identify and fix the gradient accumulation bug in a training loop',
      ],
      skills: ['pytorch', 'training-loop', 'nn-mseloss', 'torch-optim', 'sgd', 'adam'],
      prerequisites: ['nn-module'],
      exercise: {
        constraints: [
          'No DataLoader or Dataset objects — raw tensors for data',
          'No validation loops or train/val split in code',
          'No model.train() / model.eval() in practice',
          'No learning rate schedulers or gradient clipping',
          'No GPU training — CPU only',
        ],
        steps: [
          'Verify nn.MSELoss matches manual MSE — compute both, compare values',
          'Verify optimizer.step() matches manual update — single SGD step, compare parameters',
          'Train linear regression in PyTorch — complete loop on y = 3x - 2 data',
          'Swap SGD for Adam — change one line, observe convergence difference',
          'Diagnose the accumulation bug — given broken loop (no zero_grad), predict, run, fix',
          '(Stretch) Train a 2-layer network on nonlinear y = x^2 data with nn.Sequential and ReLU',
        ],
      },
    },
  ],
}

/**
 * Real Data — connecting PyTorch to real datasets
 *
 * Module 2.2: Real Data
 * 1. Datasets and DataLoaders
 * 2. (planned) MNIST Project
 * 3. (planned) Evaluation and Metrics
 */
const realData: CurriculumNode = {
  slug: 'real-data',
  title: 'Real Data',
  children: [
    {
      slug: 'datasets-and-dataloaders',
      title: 'Datasets and DataLoaders',
      description: "Connect your training loop to real data with PyTorch's two-layer data abstraction",
      duration: '25 min',
      category: 'Real Data',
      objectives: [
        'Implement a custom Dataset with __len__ and __getitem__',
        'Use DataLoader for batching, shuffling, and iteration',
        'Apply transforms (ToTensor, Normalize) with Compose',
        'Load MNIST with torchvision.datasets and inspect batch shapes',
        'Integrate DataLoader into an existing training loop',
      ],
      skills: ['pytorch', 'dataset', 'dataloader', 'transforms', 'torchvision'],
      prerequisites: ['training-loop'],
      exercise: {
        constraints: [
          'No training a model to convergence on a real dataset',
          'No cross-entropy loss or softmax — deferred to MNIST project',
          'No data augmentation strategies (RandomFlip, RandomCrop)',
          'No advanced DataLoader options (num_workers, pin_memory, custom collate)',
        ],
        steps: [
          'Implement a custom Dataset for y=2x+1 data, wrap in DataLoader, iterate and print shapes',
          'Load MNIST with torchvision.datasets, apply transforms, inspect batches',
          'Integrate DataLoader into the training loop from The Training Loop',
          'Experiment with batch sizes (1, 32, 256, full-batch) — measure iterations per epoch',
          'Write a custom Dataset for a CSV file, apply transforms, train with DataLoader',
        ],
      },
    },
    {
      slug: 'mnist-project',
      title: 'MNIST Project',
      description: 'Build, train, and evaluate your first real classifier — a network that reads handwritten digits',
      duration: '35 min',
      category: 'Real Data',
      objectives: [
        'Build a fully-connected classifier for MNIST using nn.Module',
        'Understand softmax as the generalization of sigmoid to multiple classes',
        'Use nn.CrossEntropyLoss for classification (not MSE)',
        'Measure accuracy with torch.argmax and compare to loss',
        'Add dropout, batch norm, and weight decay for regularization',
        'Use model.train() / model.eval() correctly with regularization layers',
      ],
      skills: ['pytorch', 'classification', 'cross-entropy', 'softmax', 'accuracy', 'dropout', 'batch-norm', 'mnist'],
      prerequisites: ['datasets-and-dataloaders'],
      exercise: {
        constraints: [
          'Fully-connected network only — no convolutional layers',
          'No hyperparameter tuning — reasonable defaults only',
          'No learning rate schedulers or gradient clipping',
          'No saving/loading models',
          'Celebrate 97–98% accuracy as appropriate for this architecture',
        ],
        steps: [
          'Load MNIST with torchvision, create train and test DataLoaders',
          'Build MNISTClassifier — Flatten + 3 linear layers with ReLU',
          'Write the training loop with cross-entropy loss and accuracy tracking',
          'Evaluate on the test set with torch.no_grad() and model.eval()',
          'Visualize correct and incorrect predictions with confidence scores',
          'Build ImprovedMNISTClassifier with BatchNorm, Dropout, and weight decay',
        ],
      },
    },
    {
      slug: 'debugging-and-visualization',
      title: 'Debugging and Visualization',
      description: 'Three diagnostic instruments for when PyTorch training goes wrong — and a systematic workflow to use them',
      duration: '25 min',
      category: 'Real Data',
      objectives: [
        'Use torchinfo.summary() to inspect model architecture, shapes, and parameter counts',
        'Write a gradient magnitude checking function using model.named_parameters()',
        'Diagnose vanishing and exploding gradients from per-layer gradient norms',
        'Set up TensorBoard logging with SummaryWriter for training metrics',
        'Compare multiple training runs in TensorBoard',
        'Apply a systematic debugging checklist: torchinfo, gradient check, TensorBoard',
      ],
      skills: ['pytorch', 'debugging', 'torchinfo', 'tensorboard', 'gradient-checking'],
      prerequisites: ['mnist-project'],
      exercise: {
        constraints: [
          'Three tools only — torchinfo, gradient checking, TensorBoard basics',
          'No advanced TensorBoard features (histograms, embeddings, profiler)',
          'No Weights & Biases, MLflow, or other experiment tracking',
          'No performance profiling or GPU debugging',
        ],
        steps: [
          'Run torchinfo on MNIST model and verify parameter counts',
          'Introduce a shape bug, use torchinfo to diagnose, fix it',
          'Write log_gradient_norms() and compare healthy vs unhealthy models',
          'Add TensorBoard logging to the MNIST training loop',
          'Train 3 runs with different learning rates and compare in TensorBoard',
          'Debug a broken training script with 3 intentional bugs using the checklist',
        ],
      },
    },
  ],
}

/**
 * Practical Patterns — real-world PyTorch workflows
 *
 * Module 2.3: Practical Patterns
 * 1. Saving, Loading, and Checkpoints
 * 2. (planned) GPU Training
 * 3. (planned) Data Augmentation
 */
const practicalPatterns: CurriculumNode = {
  slug: 'practical-patterns',
  title: 'Practical Patterns',
  children: [
    {
      slug: 'saving-and-loading',
      title: 'Saving, Loading, and Checkpoints',
      description: 'Make your trained models durable -- save state, resume interrupted training, and never lose progress',
      duration: '20 min',
      category: 'Practical Patterns',
      objectives: [
        'Understand state_dict as a dictionary of named tensors (model and optimizer)',
        'Save and load model weights using torch.save() and load_state_dict()',
        'Implement checkpointing to resume interrupted training',
        'Explain why state_dict is preferred over torch.save(model)',
        'Use map_location for cross-device loading',
      ],
      skills: ['pytorch', 'state-dict', 'checkpointing', 'save-load', 'model-persistence'],
      prerequisites: ['debugging-and-visualization'],
      exercise: {
        constraints: [
          'state_dict save/load and checkpoint patterns only',
          'No TorchScript, ONNX export, or model compilation',
          'No distributed checkpointing or multi-GPU saving',
          'No experiment management tools (MLflow, W&B)',
        ],
        steps: [
          'Save and load a trained MNIST model, verify predictions match with torch.allclose()',
          'Add checkpointing to a training loop -- save every 5 epochs and save best model by validation loss',
          'Simulate a training crash -- train, checkpoint, resume, verify continuous loss curve',
          'Implement full early stopping with patience counter and checkpoint restore',
        ],
      },
    },
    {
      slug: 'gpu-training',
      title: 'GPU Training',
      description: 'Move your training loop to GPU and squeeze more speed with mixed precision -- without changing the heartbeat',
      duration: '25 min',
      category: 'Practical Patterns',
      objectives: [
        'Write device-aware training code that runs identically on CPU and GPU',
        'Handle device mismatch errors in a real training loop',
        'Know when GPU training helps and when it does not',
        'Use torch.amp.autocast and GradScaler for mixed precision',
        'Save and load device-aware checkpoints with map_location',
      ],
      skills: ['pytorch', 'gpu', 'cuda', 'mixed-precision', 'device-management'],
      prerequisites: ['saving-and-loading'],
      exercise: {
        constraints: [
          'Single-GPU training only -- no multi-GPU or distributed training',
          'No CUDA programming, kernels, or streams',
          'No memory management or OOM debugging',
          'No torch.compile or custom CUDA extensions',
          'Mixed precision basics only -- no bfloat16 deep dive',
        ],
        steps: [
          'Move the MNIST model and training loop to GPU, time it, compare to CPU',
          'Add device-aware checkpointing -- save during GPU training, load on CPU with map_location',
          'Add mixed precision (autocast + GradScaler) to the GPU training loop, compare speed',
          'Write a complete portable training script: device detection, GPU, mixed precision, device-aware checkpoints',
        ],
      },
    },
  ],
}

/**
 * PyTorch — practical implementation skills
 */
export const pytorch: CurriculumNode = {
  slug: 'pytorch',
  title: 'PyTorch',
  icon: 'Flame',
  description: 'Practical PyTorch implementation skills',
  children: [
    pytorchCore,
    realData,
    practicalPatterns,
  ],
}
