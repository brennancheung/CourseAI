# Diagram & Visualization Inventory

Complete inventory of all diagrams and visualizations in the CourseAI codebase.

## Mermaid Diagrams

Rendered via `src/components/widgets/MermaidDiagram.tsx` (client component, dark theme, lazy-loaded).

All diagrams use `graph LR` (left-right) or `graph TD` (top-down) flowcharts. No sequence, state, or class diagrams.

| Lesson File | Module | Diagram Concept |
|-------------|--------|-----------------|
| `src/components/lessons/module-2-1/AutogradLesson.tsx` | 2.1 | Computational graph with gradient flow (2-layer network) |
| `src/components/lessons/module-2-1/NnModuleLesson.tsx` | 2.1 | nn.Module architecture + residual connection loops |
| `src/components/lessons/module-2-2/DatasetsAndDataloadersLesson.tsx` | 2.2 | Data pipeline: Raw Data -> Dataset -> DataLoader -> Training Loop |
| `src/components/lessons/module-4-1/WhatIsALanguageModelLesson.tsx` | 4.1 | Autoregressive sampling loop: Context -> LM -> Probability -> Sample -> Append |
| `src/components/lessons/module-6-1/AutoencodersLesson.tsx` | 6.1 | Hourglass architecture: Input(28x28) -> Encoder -> Bottleneck(32-dim) -> Decoder -> Output(28x28) |
| `src/components/lessons/module-6-2/LearningToDenoiseLesson.tsx` | 6.2 | Forward process: x0 + t + e -> closed-form -> x_t |
| `src/components/lessons/module-6-3/UnetArchitectureLesson.tsx` | 6.3 | Full U-Net encoder-decoder with skip connections |
| `src/components/lessons/module-6-3/ClipLesson.tsx` | 6.3 | CLIP dual-encoder: Image + Caption -> separate encoders -> shared 512-dim space |
| `src/components/lessons/module-6-3/FromPixelsToLatentsLesson.tsx` | 6.3 | Pixel vs latent diffusion comparison (512x512x3 vs 64x64x4) |
| `src/components/lessons/module-6-3/ConditioningTheUnetLesson.tsx` | 6.3 | Residual block with AdaGN: Conv -> AdaGN(timestep) -> Activation -> Conv |
| `src/components/lessons/module-6-3/TextConditioningAndGuidanceLesson.tsx` | 6.3 | Block flow: Residual -> Self-Attention -> Cross-Attention -> Residual |
| `src/components/lessons/module-6-4/StableDiffusionArchitectureLesson.tsx` | 6.4 | Full SD pipeline: Text -> CLIP(123M) -> U-Net(860M) -> VAE(84M) -> Image |
| `src/components/lessons/module-6-5/LoraFinetuningLesson.tsx` | 6.5 | LoRA injection: Frozen U-Net blocks + trainable LoRA blocks |
| `src/components/lessons/module-6-5/TextualInversionLesson.tsx` | 6.5 | Trainable embedding row -> CLIP Transformer -> Cross-Attention -> U-Net |
| `src/components/lessons/module-7-1/ControlNetLesson.tsx` | 7.1 | Trainable encoder copy + zero convolution for spatial conditioning |
| `src/components/lessons/module-7-1/ControlNetInPracticeLesson.tsx` | 7.1 | End-to-end: Photo -> Preprocessor -> Spatial Map -> ControlNet -> SD -> Image |
| `src/components/lessons/module-7-1/IPAdapterLesson.tsx` | 7.1 | IP-Adapter: Spatial Features + Image Embeddings -> shared cross-attention |
| `src/components/lessons/module-8-1/Sam3Lesson.tsx` | 8.1 | SAM architecture: Image -> ViT Encoder -> Embedding; SAM v1 -> v2 -> v3 evolution |
| `src/components/lessons/module-8-3/NanoBananaProLesson.tsx` | 8.3 | VQ-VAE: 256x256 -> Encoder -> 32x32 Vectors -> Codebook -> 1024 Tokens |

## Inline SVG Diagrams (in widgets)

Custom SVG elements rendered directly in React components. These are interactive, not static.

| Widget File | Diagram Type | Concept |
|-------------|-------------|---------|
| `src/components/widgets/ComputationalGraphExplorer.tsx` | DAG / computation graph | Forward/backward pass through computational graphs with gradient flow |
| `src/components/widgets/ResNetBlockExplorer.tsx` | Architecture diagram | ResNet block: conv layers, skip connection, add operation, gradient flow paths |
| `src/components/widgets/SGDExplorer.tsx` | 2D contour heatmap + line chart | SGD trajectory on loss surface with two minima + training loss curve |
| `src/components/widgets/OptimizerExplorer.tsx` | 2D contour heatmap + line chart | SGD vs Momentum vs RMSProp vs Adam on ravine-shaped loss landscape |
| `src/components/widgets/LossSurfaceExplorer.tsx` | 2D contour heatmap | MSE loss surface over slope/intercept params + data fitting view |
| `src/components/widgets/TrainingDynamicsExplorer.tsx` | Bar chart + line chart | Gradient magnitude distribution (log scale) + training loss over epochs |
| `src/components/widgets/RegularizationExplorer.tsx` | Dual line chart + function plot | Train vs validation loss (overfitting) + learned function vs true function |
| `src/components/widgets/PositionalEncodingHeatmap.tsx` | Heatmap matrix | Sinusoidal positional encoding: rows=positions, cols=dimensions |
| `src/components/widgets/NormalizationComparisonWidget.tsx` | Grid visualization | Batch Norm vs Layer Norm vs Group Norm statistics over mini-batch dimensions |
| `src/components/widgets/XORTransformationWidget.tsx` | Side-by-side scatter plots | How hidden layer transforms non-linearly separable data into separable space |
| `src/components/widgets/GenerativeVsDiscriminativeWidget.tsx` | Scatter + decision boundary | Two Gaussian clusters with decision boundary: generative vs discriminative |
| `src/components/widgets/VaeLatentSpaceWidget.tsx` | 2D latent space | VAE encoded points, gaussian clouds, sampled points, grid coordinates |
| `src/components/widgets/DiffusionNoiseWidget.tsx` | Pixel grid | 28x28 pixel grid showing image degradation across diffusion noise levels |
| `src/components/widgets/AlphaBarCurveWidget.tsx` | Line chart + pixel grid | Alpha bar schedule curve (signal/noise decay) + pixel snapshots at timesteps |
| `src/components/widgets/DenoisingTrajectoryWidget.tsx` | Timeline + pixel grids | Denoising progression from pure noise (t=1000) to clean image (t=0) |
| `src/components/widgets/TrainingStepSimulator.tsx` | Pixel grid | Noisy image progression: x0, xt, predicted noise, actual noise |
| `src/components/widgets/AutoencoderBottleneckWidget.tsx` | Pixel grids | Original vs reconstructed images at different bottleneck sizes |
| `src/components/widgets/AutogradExplorer.tsx` | Arrow icons | Flow direction arrows between computational graph nodes |
| `src/components/lessons/module-8-2/SafetyStackDiagram.tsx` | Vertical flowchart | Four-layer AI safety defense: prompt filter -> inference guidance -> output classifier -> model erasure |

## Recharts (data charts)

| File | Chart Type | Concept |
|------|-----------|---------|
| `src/components/widgets/TemperatureExplorer.tsx` | Bar chart | Token probability distribution with temperature sampling |
| `src/components/widgets/EmbeddingSpaceExplorer.tsx` | Scatter chart | 2D PCA projection of GPT-2 token embeddings by semantic cluster |
| `src/components/lessons/module-7-3/03-TheSpeedLandscapeLesson.tsx` | Line chart | Quality vs steps tradeoff in diffusion models (Free/Cheap/Expensive regions) |
| `src/components/lessons/module-4-1/WhatIsALanguageModelLesson.tsx` | Bar chart | Next-token probability distribution ("The cat sat on the ___") |
| `src/components/lessons/module-2-3/SavingAndLoadingLesson.tsx` | Line chart | Training loss: resume with vs without optimizer state |
| `src/components/lessons/module-2-2/DebuggingAndVisualizationLesson.tsx` | Line chart | Learning rate comparison (TensorBoard mockup) |
| `src/components/ui/chart.tsx` | Base wrapper | ChartContainer, ChartTooltip, ChartLegend (shared infrastructure) |

## Mafs (interactive math)

| File | Concept |
|------|---------|
| `src/components/widgets/LearningRateExplorer.tsx` | Gradient descent on L(x)=x^2 — four scenarios: too small, just right, too large, divergent |

## Canvas/Konva Visualizations (non-SVG, non-Mermaid)

Interactive widgets using react-konva or HTML5 Canvas. Included for completeness.

| Widget File | Concept |
|-------------|---------|
| `src/components/widgets/ActivationFunctionExplorer.tsx` | Activation functions: sigmoid, tanh, relu, leaky-relu, gelu, swish |
| `src/components/widgets/GradientDescentExplorer.tsx` | Animated gradient descent on 1D loss curve |
| `src/components/widgets/SingleNeuronExplorer.tsx` | Single neuron: y = w1*x1 + w2*x2 + b |
| `src/components/widgets/LinearFitExplorer.tsx` | Interactive linear regression fitting |
| `src/components/widgets/OverfittingWidget.tsx` | Training vs validation loss curves |
| `src/components/widgets/BackpropFlowExplorer.tsx` | Animated step-through backpropagation (11-step epoch cycle) |
| `src/components/widgets/XORClassifierExplorer.tsx` | XOR problem decision boundaries |
| `src/components/widgets/ConvolutionExplorer.tsx` | CNN convolution: filter sliding + feature map generation |
| `src/components/widgets/TrainingLoopExplorer.tsx` | Training loop dynamics and metric progression |
| `src/components/widgets/AttentionMatrixWidget.tsx` | Token-to-token attention weights heatmap |
| `src/components/widgets/NetworkDiagramWidget.tsx` | Neural network topology with weight visualization |
| `src/components/widgets/BpeVisualizer.tsx` | Byte-pair encoding merge algorithm step-by-step |
| `src/components/widgets/ArchitectureDiagram/ArchitectureDiagram.tsx` | Zoomable architecture diagrams with interactive nodes and edges |
| `src/components/widgets/ArchitectureComparisonExplorer.tsx` | CNN architecture evolution: LeNet-5 -> AlexNet -> VGG-16 |
| `src/components/widgets/SafetyStackSimulator.tsx` | Safety mechanism layering simulator |

## Canvas Infrastructure

| File | Purpose |
|------|---------|
| `src/components/canvas/MathCanvas.tsx` | 2D math coordinate system with transformations |
| `src/components/canvas/ZoomableCanvas.tsx` | Pan/zoom-enabled 2D canvas |
| `src/components/canvas/primitives/Axis.tsx` | X/Y axes with arrows and labels |
| `src/components/canvas/primitives/Grid.tsx` | 2D grid with configurable spacing |
| `src/components/canvas/primitives/Curve.tsx` | Function curve plotting y = f(x) |
| `src/components/canvas/primitives/Ball.tsx` | Draggable point/circle with labels |

## Standalone Diagram Files

| File | Type | Concept |
|------|------|---------|
| `docs/diagrams/safety-stack.html` | HTML + embedded SVG | AI safety stack: 4-layer defense flowchart (prompt filter, inference guidance, output classifier, model erasure) |

## Installed but Unused

These visualization libraries are in `package.json` but have zero imports in the codebase:

- **@visx/visx** (v3.12.0) — Custom 2D visualizations
- **@react-three/fiber** / **@react-three/drei** / **three** — 3D visualizations
