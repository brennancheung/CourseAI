# Module 3.2: Modern Architectures — Plan

## Module Goal

The student can trace how CNN architectures evolved from LeNet to ResNet, explain why depth improves performance (hierarchical features, increasing receptive field), identify the degradation problem that deeper networks face, and understand how residual connections solve it — preparing them to use pretrained models in practice.

## Narrative Arc

**Beginning (the problem):** The student just built a 2-conv-layer CNN for MNIST and saw it beat a dense network. But MNIST is easy — 28x28, grayscale, centered digits. Real images (ImageNet: 224x224, RGB, cluttered scenes) are orders of magnitude harder. The student's tiny CNN would fail catastrophically. How did the field get from toy CNNs to networks that classify 1000 categories with superhuman accuracy? The answer is depth — but depth is not free. Each step deeper bought real capabilities but introduced new problems that required architectural innovation to solve.

**Middle (building the tools):** We trace the historical arc: LeNet (1998) proved CNNs work on simple tasks. AlexNet (2012) scaled depth and added ReLU + dropout + GPU training to crush ImageNet. VGG (2014) showed that a disciplined pattern of small 3x3 filters stacked deep beats larger filters — establishing "deeper = better." But then the field hit a wall: networks deeper than ~20 layers performed WORSE than shallower ones, even on training data. This wasn't overfitting — it was a new problem. ResNets (2015) solved it with skip connections, enabling 152-layer networks. Along the way, the student sees batch normalization in practice and learns transfer learning as the practical payoff.

**End (the payoff):** The student understands that modern deep learning is built on a small number of architectural innovations, each solving a specific problem. They can use pretrained models (the real practical skill) because they understand what those models learned and why the architecture works. This sets up the U-Net architecture needed for Stable Diffusion.

## Lesson Sequence

| # | Slug | Core Concepts | Type | Rationale for Position |
|---|------|--------------|------|----------------------|
| 1 | architecture-evolution | LeNet/AlexNet/VGG comparison, depth as feature hierarchy, 3x3 stacking, parameter efficiency | STRETCH | Must come first: establishes the historical trajectory and the "deeper = better" principle; introduces the architectures that ResNets respond to |
| 2 | resnets | Degradation problem, residual connections, batch norm in practice, identity shortcut | STRETCH | Requires understanding of "deeper = better" from L1 to motivate the degradation problem; skip connections are the architectural innovation |
| 3 | transfer-learning | Pretrained models, feature extraction vs fine-tuning, torchvision.models | BUILD | Practical payoff: uses the architectures from L1-L2; lower cognitive load, applies understanding to real-world task |

## Rough Topic Allocation

| Lesson | Topics | Notes |
|--------|--------|-------|
| architecture-evolution | LeNet architecture, AlexNet innovations (ReLU, dropout, GPU), VGG's 3x3 philosophy, effective receptive field, parameter efficiency of small filters, depth as hierarchical feature extraction | No notebook — interactive web lesson. Comparative architecture exploration. |
| resnets | The degradation problem (not overfitting), identity mapping baseline, residual connections, batch norm's role in deep networks, ResNet architecture variants | Notebook: implement a ResNet block, train on CIFAR-10 or similar |
| transfer-learning | Feature extraction (freeze backbone, train head), fine-tuning (unfreeze some layers), torchvision.models API, when to use which strategy, practical tips for small datasets | Notebook: fine-tune a pretrained ResNet on a small custom dataset |

## Cognitive Load Trajectory

```
Lesson 1 (architecture-evolution):  STRETCH  — New architectural concepts, historical reasoning, parameter analysis
Lesson 2 (resnets):                 STRETCH  — New architectural innovation (skip connections), deep theory
Lesson 3 (transfer-learning):       BUILD    — Applies architectures practically, lower conceptual novelty
```

Two STRETCH lessons back-to-back is a concern. Mitigating factors:
1. L1 is conceptual/historical (no code), so the cognitive load is different in kind from L2 which is implementation-focused
2. L1 ends with a clear understanding of "deeper = better" which directly motivates L2's "but deeper breaks" — the narrative momentum carries
3. L3 is a significant step down in difficulty, providing recovery

If L2 feels too heavy during lesson planning, consider splitting the degradation problem motivation into L1's ending (as a cliffhanger) to reduce L2's new concept count.

## Module-Level Misconceptions

- **"Deeper always means better"** — True up to a point, then the degradation problem kicks in. The entire ResNet story is about making depth work.
- **"AlexNet was just a bigger LeNet"** — AlexNet introduced ReLU, dropout, GPU training, and data augmentation. The innovations matter as much as the scale.
- **"3x3 is just an arbitrary choice"** — VGG showed it is principled: two 3x3 filters have the same receptive field as one 5x5 but fewer parameters and more nonlinearity.
- **"Skip connections are just a trick"** — They solve a fundamental problem (degradation) by providing an identity mapping baseline that deeper layers can refine rather than learn from scratch.
- **"Transfer learning only works on similar datasets"** — Early layers learn general features (edges, textures) useful across nearly all image tasks. Only later layers are task-specific.
- **"The degradation problem is just overfitting"** — It manifests as worse TRAINING accuracy in deeper networks, ruling out overfitting as the cause.
