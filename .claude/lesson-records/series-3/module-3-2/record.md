# Module 3.2: Modern Architectures — Record

**Goal:** The student can trace how CNN architectures evolved from LeNet to ResNet, explain why depth improves performance (hierarchical features, increasing receptive field), identify the degradation problem that deeper networks face, and understand how residual connections solve it — preparing them to use pretrained models in practice.
**Status:** Complete (3 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| LeNet-5 architecture (conv-pool-conv-pool-fc with sigmoid + average pooling) | INTRODUCED | architecture-evolution | Mapped to student's own MNIST CNN — "you already built a modernized LeNet." Diagram with dimensions (32x32x1 through 10 classes, ~62K params). Compared to student's MNIST CNN: same pattern, different details (sigmoid vs ReLU, average vs max pooling). |
| AlexNet innovations (ReLU, dropout, GPU training, scale) | INTRODUCED | architecture-evolution | Each innovation taught as solving a specific problem: ReLU solves vanishing gradients, dropout solves overfitting at 60M-parameter scale, GPUs enable ImageNet-scale training. Misconception addressed: GPUs were necessary but not sufficient. |
| VGG-16 architecture and 3x3 philosophy | DEVELOPED | architecture-evolution | Repeating block pattern: 2-3 3x3 convs followed by max pool, repeated 5 times. 16 weight layers but only two building blocks. "Spatial shrinks, channels grow, then flatten" pattern applied more deeply and systematically. Vgg16BlockDiagram shows proportional spatial/channel tradeoff. |
| Effective receptive field of stacked small filters (quantitative) | DEVELOPED | architecture-evolution | Formula: RF = n(k-1) + 1. Two 3x3 convs give RF=5 (same as single 5x5). Three 3x3 convs give RF=7 (same as single 7x7). ReceptiveFieldDiagram SVG showing 5x5 input -> 3x3 intermediate -> 1x1 output with color-coded grids. Extends the qualitative "zoom out" analogy from building-a-cnn to quantitative computation. |
| Parameter efficiency of small vs large filters (3x3+3x3 vs 5x5) | DEVELOPED | architecture-evolution | Two 3x3 convs: 18C^2 params, 2 nonlinearities, RF=5. One 5x5 conv: 25C^2 params, 1 nonlinearity, RF=5. 28% fewer parameters for 3x3 stacking. Three 3x3 convs vs one 7x7: 27C^2 vs 49C^2 (45% fewer). More nonlinearity = more expressive power. FilterSwapExplorer widget lets student toggle VGG blocks between 3x3 stacking and equivalent large filters. |
| Architecture evolution as problem-driven innovation | INTRODUCED | architecture-evolution | Meta-pattern: each generation solved the previous generation's limitation. LeNet proved CNNs work (limited by sigmoid). AlexNet solved vanishing gradients + overfitting (limited by ad-hoc filter sizes). VGG found the design principle (limited by depth). Explicitly addressed misconception that evolution was just "adding more of the same layers." |
| RGB / multi-channel input convolution | INTRODUCED | architecture-evolution | Brief gap-resolution section. Conv2d(3, 64, 3) means 64 filters, each 3x3x3 = 27 weights. First layer in_channels=3 (RGB), subsequent layers in_channels = previous out_channels. Extends student's Conv2d(1, 32, 3) experience from MNIST. |
| Degradation problem (deeper networks performing worse on training data) | MENTIONED | architecture-evolution | Cliffhanger for resnets lesson. 56-layer network has higher training error than 20-layer — NOT overfitting (overfitting would show high training accuracy, low test accuracy). Something prevents deeper networks from learning what shallower ones can. |
| Degradation problem (cause: optimizer cannot find identity mappings in plain layers) | DEVELOPED | resnets | Full explanation: weights initialized near zero produce near-zero output (not identity). The identity function is a specific non-trivial weight configuration. Numerical example: learning H(x)=5.1 from 0.0 vs learning F(x)=0.1 from 0.0. Explicitly distinguished from overfitting and vanishing gradients. |
| Residual connection / skip connection (F(x) + x formulation) | DEVELOPED | resnets | Core insight: learn the residual (correction from identity) instead of the full mapping. If F(x)=0, block is identity — safe default. Taught via identity mapping thought experiment (20-layer + 36 identity layers), numerical example (x=5.0, H(x)=5.1), block diagram (plain vs residual side-by-side), "editing vs writing" analogy. Interactive widget (ResNetBlockExplorer) for toggling skip connection. |
| Batch normalization in CNN practice (nn.BatchNorm2d, Conv-BN-ReLU pattern) | DEVELOPED | resnets | Extended from INTRODUCED (training-dynamics 1.3) to practical CNN usage. nn.BatchNorm2d(channels) normalizes per channel with learned gamma/beta. Conv-BN-ReLU as the standard building block. Misconception addressed: BN is not just repeated input normalization — it has learned parameters that can undo normalization. |
| model.train() vs model.eval() mode switching for batch norm | DEVELOPED | resnets | Training: per-batch statistics, updates running averages. Eval: stored running averages, deterministic output. ComparisonRow visualization. Practical rule: always switch before training/inference. Extended from MENTIONED in training-loop (2.1). |
| Identity shortcut (x + F(x) when dimensions match) | DEVELOPED | resnets | The default shortcut in most ResNet blocks. No extra parameters. Directly adds input to conv path output. |
| Projection shortcut (1x1 conv for dimension matching) | INTRODUCED | resnets | Used at stage transitions when spatial size or channel count changes. 1x1 conv = per-pixel linear transformation that adjusts channels. With stride, also downsamples spatially. Code example: Conv2d(64, 128, 1, stride=2) + BatchNorm2d(128). |
| 1x1 convolution (per-pixel channel transformation) | INTRODUCED | resnets | Introduced in context of projection shortcuts. Does not look at spatial neighbors; linearly combines channels at each position. Used to match dimensions in ResNet blocks. |
| Global average pooling (replacing FC layers) | INTRODUCED | resnets | Average each channel's spatial grid into one number. nn.AdaptiveAvgPool2d(1). Replaces VGG's expensive FC layers: 103M params -> ~512K params (200x reduction). Motivated by VGG's FC parameter dominance from architecture-evolution. |
| ResNet basic block structure (Conv-BN-ReLU-Conv-BN + shortcut + ReLU) | DEVELOPED | resnets | Standard 2-conv residual block used in ResNet-18/34. ReLU comes after the addition. Compared to VGG block: "same computation, plus one line for the shortcut." Misconception addressed: skip wraps 2+ layers, not individual layers — single-layer skip would collapse to near-identity. |
| Full ResNet architecture variants (ResNet-18 through ResNet-152) | INTRODUCED | resnets | Table showing layers, block type (basic vs bottleneck), params, top-5 accuracy. Bottleneck blocks (1x1-3x3-1x1 for ResNet-50+) MENTIONED but not developed. Key insight: ResNet-152 has fewer params than VGG-16 (60M vs 138M) despite being 8x deeper. |
| Gradient highway through skip connections (derivative = 1.0 on skip path) | INTRODUCED | resnets | Presented as Perspective 2 alongside "easier optimization" (Perspective 1). Explicitly warned that gradient highway is a partial explanation — BN alone would fix gradient flow if that were the whole story. The deeper insight is the residual learning formulation. Telephone game analogy extended: skip connection = direct phone line bypassing the chain. |
| Transfer learning (reusing pretrained model weights for new tasks) | DEVELOPED | transfer-learning | Core concept of the lesson. Motivated by the "small dataset problem" — 1,500 images vs 11M parameters. Taught via "hiring an experienced employee" analogy. Two strategies: feature extraction (freeze backbone, replace head) and fine-tuning (unfreeze some layers with differential LR). |
| Pretrained models / torchvision.models API | DEVELOPED | transfer-learning | Loading pretrained ResNet-18 with `models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`. Demystified: it is the same nn.Module pattern — print it, access model.fc, replace components. ImageNet normalization (mean/std) and 224x224 input size as fixed requirements. |
| Feature extraction (freeze backbone, replace and train classification head) | DEVELOPED | transfer-learning | Three-step pattern: (1) load pretrained, (2) freeze all params with requires_grad=False, (3) replace model.fc with nn.Linear(512, num_classes). Connected to requires_grad from autograd lesson. Only head parameters optimized — fast and memory-efficient. |
| Fine-tuning (selectively unfreeze layers with differential learning rates) | INTRODUCED | transfer-learning | Unfreeze later stages (e.g., layer4) alongside the new head. Key practical skill: parameter groups in optimizer with different learning rates (e.g., head at 1e-3, layer4 at 1e-5). Motivated by domain differences — pretrained features may need adaptation. Low LR prevents catastrophic forgetting. |
| Differential learning rates (parameter groups in optimizer) | INTRODUCED | transfer-learning | Pass list of dicts to optimizer, each with own 'params' and 'lr'. Same optimizer API (.step(), .zero_grad()), just parameter groups. Prevents destroying pretrained features while allowing adaptation. |
| Data augmentation transforms (RandomHorizontalFlip, RandomResizedCrop, ColorJitter) | INTRODUCED | transfer-learning | Extends existing transforms knowledge (Compose, ToTensor, Normalize from 2.2). Random augmentations prevent overfitting on small datasets by ensuring model never sees same image twice. Connected to overfitting/regularization from 1.3 — augmentation as data-level regularization. Train-only (validation uses deterministic transforms). |
| Cross-entropy loss for multi-class classification (nn.CrossEntropyLoss) | INTRODUCED | transfer-learning | Brief gap resolution: combines log-softmax + negative log-likelihood. Standard loss for classification tasks. Bridges from nn.MSELoss (regression) to classification. Training loop unchanged — just swap the loss function. |
| Feature transferability spectrum (early layers universal, later layers task-specific) | DEVELOPED | transfer-learning | conv1/layer1: highly universal (edges, textures). layer2/layer3: moderately universal (shapes, parts). layer4: increasingly task-specific. fc: always replaced. Directly addresses "only works on similar datasets" misconception — early features are domain-agnostic (edges in cats, tumors, satellite images). |
| Transfer learning decision framework (dataset size x domain similarity) | INTRODUCED | transfer-learning | 2x2 matrix: small/large dataset x similar/different domain determines strategy. Practical rules: always start with feature extraction, fine-tune only if needed AND enough data. Data augmentation supplements small datasets. |
| BatchNorm behavior when freezing (requires_grad vs eval mode distinction) | INTRODUCED | transfer-learning | requires_grad=False stops gradient computation but does NOT stop BN from updating running statistics in train mode. model.eval() is what freezes BN to use stored running averages. Practical note in aside — model.train() during training loop works fine for feature extraction in practice. |

## Per-Lesson Summaries

### architecture-evolution
**Status:** Built
**Cognitive load type:** STRETCH
**Widget:** ArchitectureComparisonExplorer — Tabbed architecture browser (LeNet-5, AlexNet, VGG-16). Shows layer-by-layer pipeline table with output shapes, parameter counts (per-layer and cumulative), and receptive field growth. Metrics bar shows total params, weight layers, conv vs FC param split. Quick comparison sidebar across all three architectures. VGG-16 tab includes FilterSwapExplorer: toggle individual VGG blocks between 3x3 stacking and equivalent large filter, see parameter increase and nonlinearity loss in real time.

**What was taught:**
- The progression LeNet (1998) -> AlexNet (2012) -> VGG (2014) as problem-driven architecture evolution
- Why depth helps: hierarchical features, larger receptive fields, more nonlinearity
- VGG's 3x3 stacking insight: same receptive field as large filters, fewer parameters, more nonlinearities
- Quantitative receptive field computation for stacked convolutions (RF = n(k-1) + 1)
- RGB/multi-channel convolution as a small extension of single-channel experience
- The degradation problem as a cliffhanger for the next lesson

**How concepts were taught:**
- **Hook (MNIST vs ImageNet comparison):** ComparisonRow contrasting student's 2-layer MNIST CNN (28x28, 10 classes, ~62K params) with ImageNet (224x224 RGB, 1000 classes, 138M-param winner). Bridge: "you basically built a 1998 architecture."
- **RGB gap resolution:** Brief section (3 paragraphs) explaining Conv2d(3, 64, 3) means 3x3x3 = 27 weights per filter. "Everything else works the same."
- **LeNet architecture diagram:** StageRow format with dimensions and annotations. Direct mapping to student's MNIST CNN: "same pattern." Comprehension check: "what would you change about LeNet?" (Expected: ReLU, max pooling.) Telephone game analogy refreshed in parenthetical.
- **AlexNet innovations as problem-solution pairs:** Four GradientCards (ReLU, Dropout, GPU, Scale), each framed as problem -> solution. WarningBlock addresses "GPUs made it work" misconception. Architecture diagram in StageRow format with mixed filter sizes annotated.
- **Predict-and-verify check:** "Why larger filters early, smaller later?" Student predicts before reveal.
- **VGG 3x3 insight (core concept, 5 modalities):**
  - Verbal: explanation of "only 3x3 filters, stacked deep"
  - Visual: ReceptiveFieldDiagram (inline SVG showing 5x5 -> 3x3 -> 1x1 layer structure) + Vgg16BlockDiagram (proportional block diagram showing spatial/channel tradeoff)
  - Symbolic: RF = n(k-1) + 1 formula with three worked examples
  - Concrete: 18C^2 vs 25C^2 parameter comparison in a grid layout
  - Interactive: FilterSwapExplorer (toggle VGG blocks between 3x3 stacking and large filters)
- **Transfer question:** "Replace a 7x7 with 3x3 stacking — how many, parameter count, nonlinearities?"
- **Architecture Comparison Explorer widget:** TryThisBlock with guided experiments (compare param counts, find RF > 7, notice FC param dominance, use FilterSwapExplorer).
- **Pattern of Innovation section:** Meta-level narrative connecting all three architectures. Explicit callout that innovations were qualitative, not just quantitative.
- **Degradation cliffhanger:** 56-layer worse than 20-layer on TRAINING data. Distinguished from overfitting. Sets up ResNets lesson.

**Mental models established:**
- "Depth buys hierarchy and receptive field, but each step must be earned with the right innovations" (central mental model of the lesson)
- "Given a receptive field budget, spend it on many small filters rather than fewer large ones" (VGG's core insight)
- Architecture design encodes assumptions about the problem (extends "architecture encodes assumptions about data" from Module 3.1)

**Analogies used:**
- "You already built a LeNet" — mapping student's MNIST CNN to first-generation architecture
- "Telephone game" refreshed from training-dynamics: with ReLU, the message arrives intact (gradient flows without decay)
- "Zoom out" extended: stacking convs without pooling is a gentler zoom that expands receptive field while maintaining spatial resolution
- Each architecture generation "earned" its depth through specific innovations

**What was NOT covered (scope boundaries):**
- ResNets or skip connections (next lesson)
- GoogLeNet/Inception (mentioned in plan but explicitly excluded from scope)
- Implementing or training these architectures (conceptual understanding only, no notebook)
- Data augmentation
- Batch normalization mechanics (already INTRODUCED; deferred to resnets lesson)
- Vision Transformers or architectures beyond VGG
- Global average pooling (mentioned briefly in Module 3.1, not developed here)

**Misconceptions addressed:**
1. "GPUs made AlexNet work" — GPUs necessary but not sufficient; sigmoid network on GPUs would still fail. WarningBlock explicitly addresses this.
2. "Same receptive field = same computation" — Two 3x3 convs have a ReLU between them, making them more expressive than a single 5x5. WarningBlock: "same RF does NOT mean same computation."
3. "Deeper always means better" — Degradation problem: 56-layer network has higher TRAINING error than 20-layer. Not overfitting. WarningBlock distinguishes degradation from overfitting.
4. "Architecture evolution was about adding more of the same layers" — Explicitly called out in Pattern of Innovation section: "each generation didn't just add more layers." AlexNet changed activation, pooling, regularization. VGG changed filter philosophy entirely.
5. "Modern architectures are too complex to understand" — VGG-16 has only two building blocks (3x3 conv and 2x2 pool). "Everything else is repetition."

### resnets
**Status:** Built
**Cognitive load type:** STRETCH
**Widget:** ResNetBlockExplorer — Simplified single-weight block simulation. Toggle between plain and residual modes. Slider adjusts conv weight (w). Shows block diagram with live values, stat badges (input, F(x), output), gradient flow bars (conv path, skip path, total). At w=0: residual mode shows identity (output=x), plain mode shows signal loss (output=0). Gradient bars show skip path always contributes 1.0.

**What was taught:**
- The degradation problem fully explained: optimizer cannot find identity mappings in plain layers because weights near zero produce near-zero output, not identity
- Residual connections as the solution: F(x) + x formulation makes identity the default behavior
- Batch normalization in CNN practice: nn.BatchNorm2d, Conv-BN-ReLU pattern, train/eval mode switching
- The full ResNet basic block: Conv-BN-ReLU-Conv-BN + shortcut + final ReLU
- Identity vs projection shortcuts (1x1 conv for dimension matching)
- Global average pooling as VGG FC-layer replacement
- The ResNet family (ResNet-18 through ResNet-152, basic vs bottleneck blocks)

**How concepts were taught:**
- **Hook (mystery resolution):** Restated the degradation cliffhanger from architecture-evolution. Explicitly ruled out overfitting (training error is higher, not just test) and vanishing gradients (BN + He init already applied). Builds urgency: "Something else is wrong."
- **Identity mapping argument (6 modalities for the core concept):**
  - Verbal: Thought experiment — bolt 36 identity layers onto a working 20-layer net. Should be at least as good, but the optimizer cannot find identity in plain layers.
  - Concrete: Numerical example x=5.0, H(x)=5.1. Plain layer must learn 5.1 from starting point 0.0. Residual layer must learn F(x)=0.1 from starting point 0.0, output = 0.0 + 5.0 = 5.0, only 0.1 away from target.
  - Visual: Inline SVG block diagram comparing plain block (Conv-BN-ReLU-Conv-BN) vs residual block (same + green dashed skip connection with + node).
  - Symbolic: H(x) = F(x) + x, rearranging to F(x) = H(x) - x. BlockMath/InlineMath rendering.
  - Intuitive/Analogy: "Editing a document vs writing from scratch." GradientCard with the mapping: input = draft, conv layers = proposed changes, output = edited version.
  - Interactive: ResNetBlockExplorer widget with guided TryThisBlock experiments.
- **Predict-and-verify checks:** Two checks — (1) "What if conv weights are all zero?" in residual vs plain block, (2) "Can you use identity shortcut when dimensions change?" transfer question.
- **Batch norm in practice:** Bridged from concept (training-dynamics) to CNN API. Code block showing Conv-BN-ReLU pattern with F.relu(bn(conv(x))). ComparisonRow for train vs eval mode. Misconception addressed: BN is a trainable layer with learned gamma/beta, not just preprocessing.
- **Dimension mismatch:** ComparisonRow for identity vs projection shortcuts. Code example for 1x1 conv projection. 1x1 conv explained as "per-pixel linear transformation that changes channel count."
- **Global average pooling:** Motivated by VGG's 103M FC params. VGG vs ResNet ending comparison (monospace). nn.AdaptiveAvgPool2d(1) API.
- **Why it works (two perspectives):** GradientCards for (1) easier optimization / smoother loss landscape, (2) gradient highway (derivative = 1.0 on skip path). WarningBlock explicitly states gradient highway is incomplete explanation — the deeper insight is residual learning formulation.
- **ResNet family table:** 5 variants with layers, block type, params, top-5 accuracy. Bottleneck blocks mentioned but not developed. Key stat: ResNet-152 fewer params than VGG-16 despite 8x deeper.
- **Colab notebook:** Scaffolded implementation — student implements ResidualBlock class and small ResNet, trains on CIFAR-10, compares to plain network at same depth.

**Mental models established:**
- "A residual block starts from identity and learns to deviate — making 'do nothing' the easiest path, not the hardest" (central mental model)
- "Editing a document, not writing from scratch" (residual learning analogy)
- "Skip connection = direct phone line bypassing the telephone chain" (extends telephone game from training-dynamics)
- "Architecture design encodes assumptions about the problem" extended: ResNet encodes the assumption that most transformations are small refinements (residuals)

**Analogies used:**
- "Editing a document vs writing from scratch" — plain block = blank page, residual block = editing a draft. F(x) = changes, x = draft, F(x)+x = edited version.
- "Direct phone line" — extends the telephone game analogy from training-dynamics. Skip connection bypasses the chain of whispered messages.
- "Take exactly what you know from VGG, add one line" — ResNet block compared to VGG block.
- "LEGO bricks" pattern refreshed — ResidualBlock as nn.Module subclass with __init__ + forward().

**What was NOT covered (scope boundaries):**
- Bottleneck blocks (1x1-3x3-1x1 for ResNet-50+) — mentioned but not developed
- ResNet variants (ResNeXt, DenseNet, WideResNet)
- Pre-activation ResNets (BN-ReLU-Conv ordering)
- Training tricks (LR scheduling, data augmentation, warmup)
- Loss landscape smoothing theory (mentioned but not proved)
- Stochastic depth or other ResNet regularization techniques
- Vision Transformers or post-ResNet architectures

**Misconceptions addressed:**
1. "Degradation is overfitting or vanishing gradients" — Explicitly ruled out both in Hook section. Overfitting disproof: TRAINING error is higher. Vanishing gradient disproof: BN + He init applied, problem persists. WarningBlock: "Not Just Gradient Flow."
2. "Skip connections just help gradients flow (gradient highway only)" — Addressed in "Why It Works" section. WarningBlock: "Gradient Highway is Incomplete." The fundamental insight is residual learning formulation, not just gradient flow.
3. "Skip connections on every individual layer" — WarningBlock "Why Two Convs, Not One?" in the Full ResNet Block section. Single-layer skip collapses to near-identity with no incentive to learn.
4. "Skip connections make the network less powerful" — "At least as good" argument: F(x)=0 means identity (harmless), not garbage. Check 1 reinforces: zero-weight residual block = identity, zero-weight plain block = lost signal.
5. "Batch normalization is just repeated input normalization" — WarningBlock in BN section: BN has learned gamma/beta that can undo normalization. It is a trainable layer, not preprocessing.

### transfer-learning
**Status:** Built
**Cognitive load type:** BUILD
**Widget:** None (notebook is the interactive component)

**What was taught:**
- Transfer learning as a strategy: reuse pretrained model weights for new tasks with small datasets
- The torchvision.models API: loading pretrained ResNet-18, inspecting and modifying the nn.Module
- Feature extraction: freeze backbone (requires_grad=False on all params), replace model.fc, train only the head
- Fine-tuning: selectively unfreeze later layers (layer4) with differential learning rates via parameter groups
- Data augmentation transforms for small datasets (RandomHorizontalFlip, RandomResizedCrop, ColorJitter)
- Cross-entropy loss as the standard classification loss (brief gap resolution from nn.MSELoss)
- Feature transferability spectrum: early layers universal, later layers task-specific, fc always replaced
- Decision framework: dataset size x domain similarity determines feature extraction vs fine-tuning
- BatchNorm subtlety: requires_grad=False does not freeze BN running statistics — model.eval() does

**How concepts were taught:**
- **Hook (small dataset problem):** ComparisonRow contrasting training from scratch (99% train, ~35% val, massive overfitting) vs transfer learning (95% train, 85%+ val) on same data (1,500 images of cats/dogs/horses from CIFAR-10 subset). "Same model, same data — the only difference is the starting point."
- **Why transfer learning works (connects to hierarchical features):** "Early and middle features are not specific to ImageNet." Edge detectors, texture recognizers are useful for any image domain. TransferDiagram inline SVG showing frozen backbone (blue, with layer labels and feature descriptions: edges, textures, patterns, object parts, high-level features) and trainable head (green, AdaptiveAvgPool + Linear(512, N)).
- **"Hiring an experienced employee" analogy (GradientCard):** Pretrained backbone = experienced hire with general skills. New head = job-specific training. Feature extraction = "just learn our product catalog." Fine-tuning = "also adjust your general skills for our industry."
- **torchvision.models API (code-first, demystification):** Load with `models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, print it, access model.conv1, model.layer1, model.fc. Explicitly named the "pretrained models are sealed black boxes" misconception and disproved it: "same nn.Module pattern you have been building with."
- **Predict-and-verify check:** "What is model.fc?" Expected: Linear(512, 1000). Follow-up: "If your task has 3 classes, what changes?" Expected: replace with nn.Linear(512, 3).
- **Feature extraction (3-step pattern):** Numbered steps with code: (1) load pretrained, (2) freeze all params, (3) replace head. Connected requires_grad to autograd lesson. Noted that new layer has requires_grad=True by default.
- **Cross-entropy gap resolution:** Two paragraphs in the feature extraction section explaining nn.CrossEntropyLoss as combining log-softmax + NLL. "The training loop itself is unchanged — just swap the loss function."
- **Data augmentation (positioned before notebook):** Extended existing transforms knowledge. Code block showing train_transform (with random augments) vs val_transform (deterministic). Connected to overfitting from 1.3: "augmentation is a form of data-level regularization."
- **Notebook (Colab, scaffolded):** Feature extraction + fine-tuning + from-scratch comparison on CIFAR-10 subset. Data loading and training loops provided. Student sets up pretrained model and compares approaches.
- **Why early layers transfer across domains (elaboration):** Gabor-like edge detectors respond to vertical edges regardless of domain. Transferability spectrum: conv1/layer1 (universal) through layer4 (task-specific) to fc (always replaced). Addressed "only works on similar datasets" misconception with medical imaging, satellite, art examples.
- **Fine-tuning with differential LR:** Code pattern for parameter groups in optimizer. Connected to uniform optimizer interface from training-loop (2.1). Explained why low LR for pretrained layers: prevents destroying features that took millions of images to learn.
- **Debugging transfer question (check 2):** Scenario — 200 images, fine-tune ALL layers with lr=0.01. Expected diagnosis: LR too high, destroys pretrained features. Recommendation: start with feature extraction, possibly fine-tune layer4 with lr=1e-5.
- **Decision framework table:** 2x2 matrix (small/large x similar/different domain). Practical rules of thumb: always start with feature extraction, fine-tune only if needed.
- **Looking ahead:** Transfer learning as the foundation for fine-tuning LLMs (GPT) and diffusion models (Stable Diffusion). "Transfer learning is the default way practitioners use deep learning."
- **ModuleCompleteBlock:** Module 3.2 achievements listed, next module: 3.3 (Visualizing What CNNs Learn).

**Mental models established:**
- "Hire experienced, train specific" — pretrained backbone = experienced employee, new head = job-specific training (central analogy)
- Feature extraction vs fine-tuning as a spectrum, not a binary choice (freeze all -> freeze early + fine-tune later -> fine-tune all)
- "Start with the simplest strategy, only add complexity if needed" (practical decision-making heuristic)
- Transfer learning as the default way to use deep learning in practice (not a workaround for small datasets)

**Analogies used:**
- "Hiring an experienced employee" — feature extraction = "learn our product catalog," fine-tuning = "adjust your skills for our industry"
- "Same LEGO bricks" — pretrained model is the same nn.Module pattern, not a sealed unit (refreshed from nn-module)
- "Same heartbeat" — training loop unchanged, just parameter groups (refreshed from training-loop 2.1)

**What was NOT covered (scope boundaries):**
- Training on ImageNet from scratch
- EfficientNet, ViT, or architectures beyond ResNet
- Object detection, segmentation, or tasks beyond classification
- Knowledge distillation, self-supervised pretraining
- Learning rate schedulers (mentioned as practical tip, not developed)
- Advanced augmentation (Mixup, CutMix, AutoAugment)
- ONNX export, model deployment, optimization
- Multiple GPU training

**Misconceptions addressed:**
1. "Transfer learning only works on similar datasets" — Early-layer features (edges, textures, color gradients) are universal to all natural images. ImageNet models successfully used for medical imaging, satellite, art, industrial defect detection. WarningBlock with explicit disproof.
2. "You need to retrain the entire network for a new task" — ComparisonRow in hook: same model, same data, pretrained vs scratch gives 85%+ vs ~35% val accuracy. The pretrained features carry enormous value.
3. "Fine-tuning means training the whole model with a lower learning rate" — Distinguished from feature extraction. Differential LR shown explicitly. Check 2 scenario: lr=0.01 on all layers destroys features. Parameter groups are the practical skill.
4. "Pretrained models are black boxes you cannot modify" — Explicitly named: "You might assume a pretrained model is a sealed unit." Disproved by printing model, accessing model.fc, replacing components. Connected to nn.Module LEGO bricks analogy.
5. "More data always beats transfer learning" — WarningBlock: even with 50K images, pretrained weights converge faster and often achieve better accuracy. Transfer learning gives a head start, not just a workaround.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "Depth buys hierarchy and receptive field, but each step must be earned" | architecture-evolution | |
| "Given a receptive field budget, spend it on many small filters" (VGG insight) | architecture-evolution | |
| "Architecture design encodes assumptions about the problem" (extended from 3.1) | architecture-evolution | |
| "Telephone game" gradient flow (refreshed from 1.3) | architecture-evolution | |
| "Zoom out" extended to conv stacking without pooling | architecture-evolution | |
| "A residual block starts from identity and learns to deviate" | resnets | |
| "Editing a document, not writing from scratch" (residual learning) | resnets | |
| "Skip connection = direct phone line" (extends telephone game from 1.3) | resnets | |
| "Take what you know from VGG, add one line" (ResNet = VGG + shortcut) | resnets | |
| "Hire experienced, train specific" (transfer learning = reuse pretrained backbone) | transfer-learning | |
| Feature extraction vs fine-tuning as a spectrum (freeze all -> partial -> full) | transfer-learning | |
| "Start with the simplest strategy, add complexity only if needed" | transfer-learning | |
| Transfer learning is the default, not a workaround | transfer-learning | |
