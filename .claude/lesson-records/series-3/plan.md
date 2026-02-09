# Series 3: CNNs — Plan

**Status:** In progress
**Prerequisites:** Series 2 (PyTorch)

## Series Goal

Understand convolutional networks — how architecture encodes spatial assumptions, how the field evolved, and how to use pretrained models. Hands-on throughout with PyTorch. Needed for understanding U-Net in Stable Diffusion (Series 6).

## Modules and Lessons

### Module 3.1: Convolutions

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 1 | what-convolutions-compute | What Convolutions Compute | Yes | Filters, feature maps, edge detection by hand, `nn.Conv2d` |
| 2 | building-a-cnn | Building a CNN | Yes | Pooling, stride, padding, the conv → pool → fc pattern |
| 3 | channels | Channels In, Channels Out | Yes | RGB → feature maps → deeper feature maps; how `in_channels`/`out_channels` work |
| 4 | mnist-cnn-project | Project: MNIST with a CNN | Yes | Beat the dense network from Series 2, see the difference |

### Module 3.2: Modern Architectures

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 5 | architecture-evolution | Architecture Evolution | No | LeNet → AlexNet → VGG: deeper = better, but why? |
| 6 | resnets | ResNets and Skip Connections | Yes | The degradation problem, residual connections, batch norm revisited |
| 7 | transfer-learning | Transfer Learning | Yes | Pretrained models, feature extraction vs fine-tuning, `torchvision.models` |

### Module 3.3: Seeing What CNNs See

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 7 | visualizing-features | Visualizing Features | Yes | Filter visualization, activation maps, Grad-CAM |
| 8 | transfer-learning-project | Project: Transfer Learning | Yes | Fine-tune a pretrained model on a small custom dataset |

## Scope Boundaries

**In scope:**
- 2D convolutions for image data
- Classic and modern architectures (LeNet through ResNet)
- Transfer learning and fine-tuning
- Feature visualization and interpretability

**Out of scope:**
- 1D convolutions (text, audio) — covered briefly if relevant to LLMs
- 3D convolutions (video, medical)
- Object detection, segmentation architectures (YOLO, Mask R-CNN)
- Generative models (GANs, VAEs) — separate series
- U-Net architecture — covered in Stable Diffusion series where it's motivated

## Notebook Convention

Same as Series 2: `{series}-{module}-{lesson}-{slug}.ipynb`

Example: `3-1-1-what-convolutions-compute.ipynb`

## Connections to Earlier Series

| CNN Concept | Earlier Concept | Source |
|------------|----------------|--------|
| Convolution as learned filters | Weights as learnable parameters (1.1) | Foundations |
| Feature hierarchies (edges → shapes → objects) | Layers and networks (1.2.1) | Foundations |
| Skip connections | Vanishing gradients, batch norm (1.3.6) | Foundations |
| `nn.Conv2d`, training loop | All of Series 2 | PyTorch |
