# Series 3: CNNs — Summary

**Status:** In progress (Modules 3.1-3.2: complete, Module 3.3: planned)

## Series Goal

Understand convolutional networks — how architecture encodes spatial assumptions, how the field evolved, and how to use pretrained models. Hands-on throughout with PyTorch. Needed for understanding U-Net in Stable Diffusion (Series 6).

## Modules

| Module | Title | Lessons | Status |
|--------|-------|---------|--------|
| 3.1 | Convolutions | 3 | Complete |
| 3.2 | Modern Architectures | 3 | Complete |
| 3.3 | Seeing What CNNs See | 2 | Planned |

## Rolled-Up Concept List

| Concept | Depth | Module | Lesson |
|---------|-------|--------|--------|
| Convolution as sliding filter (multiply-and-sum over local region) | DEVELOPED | 3.1 | what-convolutions-compute |
| Feature map (output of convolution, spatial answer key) | DEVELOPED | 3.1 | what-convolutions-compute |
| Edge detection filters (vertical, horizontal, blur) | DEVELOPED | 3.1 | what-convolutions-compute |
| Output size formula (N - F + 1) | DEVELOPED | 3.1 | what-convolutions-compute |
| Spatial structure / locality | INTRODUCED | 3.1 | what-convolutions-compute |
| Weight sharing | INTRODUCED | 3.1 | what-convolutions-compute |
| Multiple filters = multiple feature maps | INTRODUCED | 3.1 | what-convolutions-compute |
| Receptive field | INTRODUCED | 3.1 | what-convolutions-compute |
| Learned filters (not hand-designed) | INTRODUCED | 3.1 | what-convolutions-compute |
| Hierarchical feature composition | DEVELOPED | 3.1 | building-a-cnn (upgraded from MENTIONED in L1) |
| Max pooling | DEVELOPED | 3.1 | building-a-cnn |
| Average pooling | INTRODUCED | 3.1 | building-a-cnn |
| Stride | DEVELOPED | 3.1 | building-a-cnn |
| Padding | DEVELOPED | 3.1 | building-a-cnn |
| General output size formula: floor((N - F + 2P) / S) + 1 | DEVELOPED | 3.1 | building-a-cnn |
| Conv-pool-fc architecture pattern | APPLIED | 3.1 | building-a-cnn -> mnist-cnn-project (upgraded from DEVELOPED) |
| Receptive field growth through stacking | INTRODUCED | 3.1 | building-a-cnn |
| nn.Conv2d / nn.MaxPool2d API | APPLIED | 3.1 | building-a-cnn -> mnist-cnn-project (upgraded from INTRODUCED) |
| Flatten transition (spatial -> flat) | INTRODUCED | 3.1 | building-a-cnn |
| Stride=2 conv as pooling replacement | INTRODUCED | 3.1 | building-a-cnn |
| CNN vs dense network comparison | DEVELOPED | 3.1 | mnist-cnn-project |
| Architecture encodes assumptions about data | DEVELOPED | 3.1 | mnist-cnn-project |
| Spatial invariance via weight sharing + pooling | INTRODUCED | 3.1 | mnist-cnn-project |
| LeNet-5 architecture (conv-pool-conv-pool-fc with sigmoid + average pooling) | INTRODUCED | 3.2 | architecture-evolution |
| AlexNet innovations (ReLU, dropout, GPU training, scale) | INTRODUCED | 3.2 | architecture-evolution |
| VGG-16 architecture and 3x3 philosophy | DEVELOPED | 3.2 | architecture-evolution |
| Effective receptive field of stacked small filters (RF = n(k-1) + 1) | DEVELOPED | 3.2 | architecture-evolution |
| Parameter efficiency of small vs large filters (3x3+3x3 vs 5x5) | DEVELOPED | 3.2 | architecture-evolution |
| Architecture evolution as problem-driven innovation | INTRODUCED | 3.2 | architecture-evolution |
| RGB / multi-channel input convolution | INTRODUCED | 3.2 | architecture-evolution |
| Degradation problem (deeper networks worse on training data) | DEVELOPED | 3.2 | architecture-evolution -> resnets |
| Residual connection / skip connection (F(x) + x) | DEVELOPED | 3.2 | resnets |
| Batch normalization in CNN practice (nn.BatchNorm2d, Conv-BN-ReLU) | DEVELOPED | 3.2 | resnets |
| model.train() vs model.eval() for batch norm | DEVELOPED | 3.2 | resnets |
| Identity shortcut (x + F(x) when dimensions match) | DEVELOPED | 3.2 | resnets |
| Projection shortcut (1x1 conv for dimension matching) | INTRODUCED | 3.2 | resnets |
| Global average pooling (replacing FC layers) | INTRODUCED | 3.2 | resnets |
| ResNet basic block structure (Conv-BN-ReLU-Conv-BN + shortcut + ReLU) | DEVELOPED | 3.2 | resnets |
| Full ResNet architecture variants (ResNet-18 through ResNet-152) | INTRODUCED | 3.2 | resnets |
| Transfer learning (reusing pretrained weights for new tasks) | DEVELOPED | 3.2 | transfer-learning |
| Pretrained models / torchvision.models API | DEVELOPED | 3.2 | transfer-learning |
| Feature extraction (freeze backbone, replace and train head) | DEVELOPED | 3.2 | transfer-learning |
| Fine-tuning (selectively unfreeze layers with differential LR) | INTRODUCED | 3.2 | transfer-learning |
| Differential learning rates (parameter groups in optimizer) | INTRODUCED | 3.2 | transfer-learning |
| Data augmentation transforms (RandomHorizontalFlip, RandomResizedCrop, ColorJitter) | INTRODUCED | 3.2 | transfer-learning |
| Cross-entropy loss for multi-class classification (nn.CrossEntropyLoss) | INTRODUCED | 3.2 | transfer-learning |
| Feature transferability spectrum (early layers universal, later task-specific) | DEVELOPED | 3.2 | transfer-learning |
| Transfer learning decision framework (dataset size x domain similarity) | INTRODUCED | 3.2 | transfer-learning |

## Key Mental Models Carried Forward

| Model/Analogy | Established In |
|---------------|---------------|
| "A filter is a pattern detector" | what-convolutions-compute (3.1) |
| "Feature map is a spatial answer key" | what-convolutions-compute (3.1) |
| "Same weighted sum, different scope (local vs all)" | what-convolutions-compute (3.1) |
| "A CNN is a series of zoom-outs" | building-a-cnn (3.1) |
| "Spatial shrinks, channels grow, then flatten" | building-a-cnn (3.1) |
| "Pooling preserves feature presence, not exact position" | building-a-cnn (3.1) |
| "Architecture encodes assumptions about data" | mnist-cnn-project (3.1) |
| "The dense network flattens pixels; the CNN flattens features" | mnist-cnn-project (3.1) |
| "Same experiment, one variable" (controlled comparison) | mnist-cnn-project (3.1) |
| "Depth buys hierarchy and receptive field, but each step must be earned" | architecture-evolution (3.2) |
| "Given a receptive field budget, spend it on many small filters" (VGG insight) | architecture-evolution (3.2) |
| "Architecture design encodes assumptions about the problem" (extended from 3.1) | architecture-evolution (3.2) |
| "A residual block starts from identity and learns to deviate" | resnets (3.2) |
| "Editing a document, not writing from scratch" (residual learning) | resnets (3.2) |
| "Skip connection = direct phone line" (extends telephone game from 1.3) | resnets (3.2) |
| "Hire experienced, train specific" (transfer learning) | transfer-learning (3.2) |
| Feature extraction vs fine-tuning as a spectrum | transfer-learning (3.2) |
| Transfer learning is the default, not a workaround | transfer-learning (3.2) |

## What This Series Does NOT Cover

- 1D convolutions (text, audio) -- mentioned briefly
- 3D convolutions (video, medical imaging)
- Object detection architectures (YOLO, Mask R-CNN)
- Segmentation architectures
- Generative models (GANs, VAEs) -- separate series
- U-Net architecture -- covered in Stable Diffusion series where it is motivated
