# Module 3.1: Convolutions — Record

**Goal:** The student can explain what a convolutional layer computes, why spatial structure matters for images, and build a working CNN in PyTorch that outperforms a dense network on image data.
**Status:** Complete (3 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Convolution as sliding filter (multiply-and-sum over local region) | DEVELOPED | what-convolutions-compute | Core operation: overlay 3x3 filter on input patch, multiply element-wise, sum. Student computed by hand and explored interactively. |
| Feature map (output of convolution) | DEVELOPED | what-convolutions-compute | The spatial grid of filter responses. Taught that output(i,j) corresponds to input region starting at (i,j) — spatial layout preserved. |
| Edge detection filters (vertical, horizontal) | DEVELOPED | what-convolutions-compute | Hand-crafted [-1,0,1] filters used to build intuition. Student saw vertical edge filter respond strongly at edges, near-zero on uniform input. Blur filter also introduced (all 1s = averaging). |
| Output size formula (N - F + 1) | DEVELOPED | what-convolutions-compute | For filter size F on input size N. Tested in comprehension check (4x4 input, 2x2 filter -> 3x3 output). |
| Spatial structure / locality | INTRODUCED | what-convolutions-compute | Motivated by "flat vector problem" — flattening images destroys spatial relationships. Conv filters only look at local neighborhoods where useful information is. |
| Weight sharing | INTRODUCED | what-convolutions-compute | Same 9 filter weights used at every position. Detects the same pattern everywhere. Contrasted with dense layers (must learn each position separately). Parameter count comparison: 288 vs 25,088. |
| Multiple filters = multiple feature maps | INTRODUCED | what-convolutions-compute | Each filter asks a different question at each location. Output of conv layer is a stack of feature maps, one per filter. Analogous to multiple neurons in a dense layer. |
| Receptive field (what each output position "sees") | INTRODUCED | what-convolutions-compute | Output at (0,0) computed from only the 9 pixels in the top-left 3x3 region. Knows nothing about distant pixels. Locality is the point. |
| Hierarchical feature composition (edges -> corners -> shapes -> objects) | MENTIONED -> DEVELOPED | what-convolutions-compute -> building-a-cnn | Planted in L1. Developed in L2: the conv-pool pattern is what enables this hierarchy. Each conv-pool stage expands the receptive field, forcing the network to represent increasingly abstract features. |
| Learned filters (not hand-designed) | INTRODUCED | what-convolutions-compute | Trained CNNs (AlexNet, VGGNet) rediscover edge and texture detectors via backprop. Training loop unchanged: forward, loss, backward, update. |
| Max pooling (take max of each region to shrink spatial dimensions) | DEVELOPED | building-a-cnn | Concrete 4x4 -> 2x2 worked example (2x2 window, stride 2). Before/after edge-detection feature map showed edge presence preserved. "Zoom out" analogy — each pooling step steps back from the image. Comprehension check: student computes pooling on a new 4x4 grid. |
| Average pooling (mean of each region) | INTRODUCED | building-a-cnn | Shown in contrast with max pooling on same input. ComparisonRow: max keeps strongest response ("is the feature present?"), average smooths ("how strong on average?"). Aside notes global average pooling used in modern architectures like ResNet. |
| Stride (how far the filter jumps between positions) | DEVELOPED | building-a-cnn | Stride=1 visits every position, stride=2 skips every other. Concrete example: 6x6 input, 3x3 filter, stride 1 -> 4x4, stride 2 -> 2x2. Stride > 1 roughly halves spatial dimensions. Connected to pooling: stride=2 conv can replace separate pooling layer. |
| Padding (adding zeros around input border) | DEVELOPED | building-a-cnn | Motivated by the "shrinking problem" — 5 conv layers without padding: 32 -> 30 -> 28 -> 26 -> 24 -> 22. Padding=1 with 3x3 filter preserves spatial dimensions ("same" padding). Rule: padding = floor(F/2) gives same output size. Border information loss addressed. |
| General output size formula: floor((N - F + 2P) / S) + 1 | DEVELOPED | building-a-cnn | Presented AFTER concrete examples of stride and padding separately. Verified against 4 examples including combined stride=2 + padding=1. Extends the N - F + 1 formula from L1 (special case where S=1, P=0). |
| Conv-pool-fc architecture pattern | DEVELOPED | building-a-cnn | Full dimension tracking through MNIST CNN: 28x28x1 -> Conv(32) -> 28x28x32 -> Pool -> 14x14x32 -> Conv(64) -> 14x14x64 -> Pool -> 7x7x64 -> Flatten(3136) -> FC(128) -> FC(10). Pattern: spatial dimensions shrink, channels grow, then flatten for classification. |
| Receptive field growth through stacking conv-pool stages | INTRODUCED | building-a-cnn | 3x3 filter on pooled output sees roughly 6x6 region of original input because each pooled position summarizes a 2x2 area. This is how the network goes from edges to objects. Qualitative understanding, not exact computation. |
| nn.Conv2d / nn.MaxPool2d API (PyTorch) | INTRODUCED | building-a-cnn | Arguments mapped to the output size formula: Conv2d(in_channels, out_channels, kernel_size, stride, padding). MaxPool2d(kernel_size) with stride defaulting to kernel_size. Reading comprehension level — student can read and identify parameters, not expected to write from memory. |
| Flatten transition (spatial grid -> flat vector) | INTRODUCED | building-a-cnn | 7x7x64 becomes 3,136-element vector. Where the network switches from "where are the features?" (spatial) to "what do the features mean?" (flat). Feature maps are intermediate, not final output. |
| Stride=2 convolution as pooling replacement | INTRODUCED | building-a-cnn | Side-by-side comparison: Conv(32, 3x3, stride=2, pad=1) on 28x28 -> 14x14x32 vs Conv(32, 3x3, stride=1, pad=1) + MaxPool(2x2) -> 14x14x32. Same dimensions, different mechanism. Modern architectures use this. |
| nn.Conv2d / nn.MaxPool2d API (PyTorch) | INTRODUCED -> APPLIED | building-a-cnn -> mnist-cnn-project | Upgraded from reading comprehension (L2) to writing the code (L3). Student fills in Conv2d/MaxPool2d layers in a scaffolded CNN class, specifying in_channels, out_channels, kernel_size, padding arguments. |
| Conv-pool-fc architecture pattern | DEVELOPED -> APPLIED | building-a-cnn -> mnist-cnn-project | Upgraded from tracing dimensions through the pattern (L2) to implementing end-to-end in PyTorch (L3). Student builds the full conv-relu-pool-conv-relu-pool-flatten-fc-relu-fc pipeline as a working nn.Module. |
| CNN vs dense network comparison | DEVELOPED | mnist-cnn-project | New concept at DEVELOPED. Student compares accuracy (~97% dense vs ~99%+ CNN), parameter counts (~110K dense vs ~421K CNN total, but 100K dense first layer vs 18.8K CNN conv stack), and training curves on same data/epochs/optimizer. Understands architecture is the independent variable. |
| Architecture encodes assumptions about data | DEVELOPED | mnist-cnn-project | The core insight of the module. CNN assumes spatial structure exists (locality + weight sharing). Dense network makes no spatial assumption and treats each pixel position independently. Matching architecture to data structure is the key design principle. |
| Spatial invariance via weight sharing + pooling | INTRODUCED | mnist-cnn-project | Shifting a digit 2px changes ~750 of 784 flat vector values — the dense network sees a drastically different input. The CNN handles this because weight sharing detects the same pattern everywhere and pooling absorbs small spatial shifts. |

## Per-Lesson Summaries

### what-convolutions-compute
**Status:** Built
**Cognitive load type:** STRETCH
**Widget:** ConvolutionExplorer — 7x7 input grids with preset patterns (vertical edge, horizontal edge, diagonal, uniform, corner), 3x3 preset filters (vertical edge, horizontal edge, blur, sharpen). Step-through and auto-animate modes. Shows element-wise multiplication and sum at each position. Output feature map fills progressively with color-coded values.

**What was taught:**
- The convolution operation mechanically: slide a 3x3 filter across input, multiply-and-sum at each position, producing a smaller feature map
- Edge detection as the primary example: vertical edge filter [-1,0,1] responds strongly at vertical transitions, weakly elsewhere
- Why convolutions beat dense layers: locality (9 weights vs 784) and weight sharing (same filter everywhere)
- Multiple filters produce multiple feature maps (one per filter)
- Filter values are learned via backprop, not hand-designed

**How concepts were taught:**
- **Flat vector problem hook:** Started from MNIST experience — flattening 28x28 to 784 destroys spatial relationships. ComparisonRow: dense (784 weights, no locality) vs conv (9 weights, spatial locality).
- **Worked example with numbers:** 3x3 input patch from a vertical edge image, vertical edge filter, full element-wise multiplication chain = 3. Shown inline before the widget.
- **Interactive widget (ConvolutionExplorer):** Placed immediately after the worked example. Student selects input pattern + filter, steps through or animates. ComputationDetail panel shows multiplication at each position.
- **Formula after concrete:** output(i,j) = sum of filter(m,n) * input(i+m, j+n) with hardcoded 3x3 bounds (appropriate for this stage).
- **Real-photograph bridge:** Descriptive paragraph about applying edge filters to a photograph of a building. References AlexNet/VGGNet first-layer learned filters.

**Mental models established:**
- "A filter is a pattern detector — it asks 'does this local region look like my pattern?' at every position"
- "The feature map is a spatial answer key — output position (i,j) tells you about the input neighborhood at (i,j)"
- "A convolutional layer asks a fixed set of questions at every location; the answers are the feature maps"

**Analogies used:**
- Dense layer vs convolution: same operation (weighted sum), different scope (all inputs vs local neighborhood)
- Multiple filters as "questions the layer asks about each neighborhood"
- 28x28 filter on 28x28 image IS a dense layer (extreme case argument for why small filters matter)

**What was NOT covered (scope boundaries):**
- Pooling, stride, padding (Lesson 2)
- Full CNN architecture (Lesson 2)
- Training a CNN or nn.Conv2d in depth (Lesson 2-3)
- Backprop through conv layers (relies on existing knowledge)
- 1D or 3D convolutions
- Multiple input channels (RGB)

**Misconceptions addressed:**
1. "Convolution is just a smaller dense layer" — No: weight sharing means same weights everywhere, not fewer connections
2. "Filter values need to be hand-designed" — No: networks learn them via backprop
3. "Convolution changes the training process" — No: same forward-loss-backward-update loop
4. "Each output pixel is influenced by entire input" — No: only sees its 3x3 neighborhood (receptive field)
5. "Bigger filters are always better" — No: extreme case is a dense layer; small filters + stacking is better (teased for Lesson 2)

### building-a-cnn
**Status:** Built
**Cognitive load type:** BUILD
**Widget:** CnnDimensionCalculator — Layer-by-layer CNN architecture builder. Student adds Conv2d, MaxPool2d, Flatten, Linear layers with configurable parameters. Real-time shape computation at each stage. Proportional pipeline visualization (blocks sized by spatial dimensions and channel count using sqrt scaling). LeNet preset. Warnings for invalid configurations (spatial after flatten, linear before flatten, dimensions reduced to 0). Color-coded by layer type.

**What was taught:**
- Max pooling: take the maximum value in each region (2x2 window, stride 2) to shrink spatial dimensions while preserving feature presence
- Average pooling: take the mean instead of max — smoother, used in some architectures (briefly, at INTRODUCED level)
- Stride: controls how far the filter jumps between positions; stride > 1 reduces output size
- Padding: adds zeros around the border to prevent spatial shrinking and preserve edge information
- The general output size formula: floor((N - F + 2P) / S) + 1
- The conv-pool-fc architecture pattern: convolutions extract spatial features, pooling shrinks dimensions, flatten collapses spatial structure, fully-connected layers classify
- Data shape transforms: spatial dimensions shrink while channel count grows, then flatten for classification

**How concepts were taught:**
- **Hook (motivation before solution):** 32 filters on 28x28 MNIST = 21,632 values, more than the 784 pixels you started with. "Something needs to shrink."
- **Max pooling worked example:** 4x4 input, 2x2 window, stride 2. Four windows, four max operations. Before/after edge-detection feature map showing edge preserved after pooling.
- **"Zoom out" analogy for pooling:** Each pooling step is like stepping back from an image — close up you see pixels and edges, step back and you see shapes, step back further and you see the whole object.
- **Shrinking problem (negative example):** 5 conv layers without padding: 32 -> 30 -> 28 -> 26 -> 24 -> 22. Border pixels systematically underrepresented.
- **General formula after concrete examples:** Formula presented after stride and padding are taught separately with worked examples. Verified against 4 examples including combined stride=2 + padding=1.
- **Full dimension tracking:** 28x28x1 through 8 stages to 10-class output using StageRow component with color-coded dots and annotations. The lesson's centerpiece.
- **Stride=2 vs pooling comparison:** Side-by-side on 28x28 input, both yielding 14x14x32. Placed after the general formula so padding is taught first.
- **Conv-Conv-Conv-Pool alternative architecture:** Concrete dimension tracking showing non-alternating pattern works. Addresses misconception about mandatory pooling after every conv.
- **Interactive widget (CnnDimensionCalculator):** Student builds architectures and watches shapes transform in real time. Proportional pipeline visualization (blocks sized by dimensions). TryThisBlock experiments: remove padding, replace pool with stride=2 conv, stack convs before pool.
- **PyTorch API:** nn.Conv2d and nn.MaxPool2d arguments mapped to the formula. Reading comprehension level.

**Mental models established:**
- "A CNN is a series of zoom-outs" — each conv detects patterns at the current scale, each pool zooms out so the next conv detects at a larger scale
- "Spatial shrinks, channels grow, then flatten" — the data shape signature of every CNN
- "Pooling preserves feature presence, not exact position" — approximate location is enough for recognition

**Analogies used:**
- Pooling as "zooming out" from an image
- "Same output dimensions, different mechanism" for stride=2 conv vs pooling
- Flatten as the transition from "where are the features?" to "what do the features mean?"

**What was NOT covered (scope boundaries):**
- Training a CNN (Lesson 3)
- Comparing CNN vs dense accuracy (Lesson 3)
- Architecture design choices in depth — how many layers, how many filters (Lesson 3)
- Backprop through pooling or conv layers
- Global average pooling (mentioned briefly in aside, not developed)
- Batch normalization, dropout in CNNs
- Advanced architectures (ResNet, VGG) beyond brief mentions
- Multiple input channels / RGB images
- 1D or 3D convolutions

**Misconceptions addressed:**
1. "Pooling destroys information" — Before/after edge-detection feature map: edge still clearly visible after pooling. Feature presence preserved, exact position traded for computational efficiency.
2. "Stride and padding are just implementation details" — Without padding, 5 layers lose 10 pixels. Border information systematically underrepresented. Padding determines whether border information survives deep networks.
3. "The output of a CNN is a feature map" — CNN classifying digits outputs a 10-element vector. The flatten + FC layers transition from spatial to flat. Feature maps are intermediate, not final.
4. "You need pooling after every conv layer" — Concrete Conv-Conv-Conv-Pool architecture shown with traced dimensions. Modern architectures also use stride=2 convolutions instead of pooling.
5. "A bigger feature map (more spatial resolution) is always better" — Reducing dimensions via pooling means each filter application covers more of the original image. Keeping full resolution traps the network in local feature detection.

### mnist-cnn-project
**Status:** Built
**Cognitive load type:** CONSOLIDATE
**Notebook:** `notebooks/3-1-3-mnist-cnn-project.ipynb` — scaffolded Colab notebook. Student fills in CNN class (layers in `__init__`, forward pass in `forward()`). Data loading, dense baseline, training loop, evaluation, and comparison code provided.

**What was taught:**
- No new concepts (CONSOLIDATE). Depth upgrades only:
  - nn.Conv2d / nn.MaxPool2d: INTRODUCED -> APPLIED (writing the code, not just reading)
  - Conv-pool-fc architecture: DEVELOPED -> APPLIED (implementing end-to-end)
  - CNN vs dense comparison: NEW at DEVELOPED (quantitative evidence that architecture matters)
- Architecture encodes assumptions about data — the core insight of the entire module
- Spatial invariance: weight sharing + pooling give the CNN tolerance to small spatial shifts that the dense network cannot have

**How concepts were taught:**
- **Architecture recap:** StageRow component showing 10-stage pipeline (28x28x1 through 10-class output) — 30-second priming from L2, not re-teaching
- **Controlled experiment framing (hook):** Same data, same optimizer, same epochs, same loss function. Architecture is the only variable. "Can the CNN do better?"
- **API recap (bridge INTRODUCED -> APPLIED):** Annotated code block mapping nn.Conv2d arguments to the dimension formula. One sentence connecting ReLU to prior knowledge: "Same ReLU you have used before."
- **Scaffolded Colab notebook:** Student fills in CNN class TODOs while boilerplate is provided. Focus on CNN-specific code only.
- **Dimension verification check:** Predict-and-verify exercise before training. Student traces shapes through 6 stages and verifies with `model(torch.randn(1, 1, 28, 28))`.
- **Side-by-side comparison:** ComparisonRow showing dense (~97%, ~110K params total, ~100K first layer) vs CNN (~99%+, ~421K params total, ~19K conv stack). Reframed as feature extraction efficiency, not total parameter count.
- **Explain-it-back check:** Student answers "why does the CNN win?" before reading the explanation. Expected: spatial locality, weight sharing, feature hierarchy, dense weakness with shifted inputs.
- **Shifting experiment (negative example):** Code snippet showing ~750 of 784 flat vector values change when a digit shifts 2px. Dense network sees a drastically different input; CNN handles it via weight sharing + pooling.
- **Parameter count arithmetic:** Dense first layer (784x128 = 100,352 params) vs entire CNN conv stack (320 + 18,496 = 18,816 params). The CNN's feature extraction is 5x more parameter-efficient. Total CNN params are higher (~421K) due to FC layers — honestly acknowledged.
- **Flatten misconception:** "The dense network flattens pixels; the CNN flattens features." Flattening after conv-pool stages is different from flattening raw pixels because the 3136 values represent learned abstractions, not raw data.

**Mental models established:**
- "Architecture encodes assumptions about data" — matching architecture to data structure is the key design principle
- "The dense network flattens pixels; the CNN flattens features" — both flatten, but the CNN earned the right to flatten by extracting spatial features first
- "Same experiment, one variable" — controlled comparison as a way to isolate architectural impact

**Analogies used:**
- Controlled experiment: same data, same optimizer, same epochs — architecture is the independent variable
- Dense network treats a shifted digit as a "completely different input" (the flat vector problem made concrete with the student's own model)

**What was NOT covered (scope boundaries):**
- No new concepts, layers, or techniques introduced
- Hyperparameter tuning or architecture search
- Batch normalization, dropout, or regularization in the CNN context
- Multiple input channels / RGB images
- Advanced architectures (ResNet, VGG) beyond brief mentions in the next step
- Data augmentation or learning rate scheduling
- Backprop through conv/pool layers (relies on existing autograd knowledge)

**Misconceptions addressed:**
1. "The CNN is better because it has more parameters" — The CNN actually has MORE total parameters (~421K vs ~110K), but its conv stack (18.8K) outperforms the dense first layer (100K) at feature extraction. Weight sharing is the reason.
2. "The CNN just needs more training time" — Both trained for the same epochs. The dense network plateaus around 97-98% even with more epochs. The gap is structural, not about patience.
3. "I need a complex architecture to beat the dense network" — The simplest possible CNN (2 conv, 2 pool, 2 FC) beats the dense network. No batch norm, no dropout, no scheduling.
4. "ReLU activations between conv layers are something new" — Same ReLU from Foundations, applied independently to each value in the feature map.
5. "The flatten step in the CNN loses important information" — By flatten time, the 7x7x64 values are abstract learned features, not raw pixels. The CNN earned the right to flatten.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "A filter is a pattern detector" | what-convolutions-compute | |
| "Feature map is a spatial answer key" | what-convolutions-compute | |
| "Same weighted sum, different scope (local vs all)" | what-convolutions-compute | |
| "Multiple filters = multiple questions at each location" | what-convolutions-compute | |
| "28x28 filter = dense layer" (extreme case) | what-convolutions-compute | |
| "A CNN is a series of zoom-outs" | building-a-cnn | |
| "Spatial shrinks, channels grow, then flatten" (CNN shape signature) | building-a-cnn | |
| "Pooling preserves feature presence, not exact position" | building-a-cnn | |
| "Architecture encodes assumptions about data" | mnist-cnn-project | |
| "The dense network flattens pixels; the CNN flattens features" | mnist-cnn-project | |
| "Same experiment, one variable" (controlled comparison) | mnist-cnn-project | |
