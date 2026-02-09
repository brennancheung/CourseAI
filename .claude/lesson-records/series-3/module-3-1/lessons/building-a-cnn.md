# Lesson Plan: Building a CNN

**Series:** 3 (CNNs) | **Module:** 3.1 (Convolutions) | **Lesson:** 2 of 3
**Slug:** `building-a-cnn`
**Cognitive load type:** BUILD

---

## Phase 1: Orient — Student State

The student completed Lesson 1 of this module (what-convolutions-compute), a STRETCH lesson that established the core convolution operation. Before that, they completed all of Series 1 (Foundations, 17 lessons covering neural networks, backprop, optimization, regularization) and are assumed to have completed Series 2 (PyTorch, practical implementation). They are coming off high cognitive load from Lesson 1 and this BUILD lesson should feel like assembling pieces they already understand rather than learning something fundamentally new.

### Relevant Concepts the Student Has

| Concept | Depth | Source | How It's Relevant Here |
|---------|-------|--------|----------------------|
| Convolution as sliding filter (multiply-and-sum over local region) | DEVELOPED | what-convolutions-compute (3.1 L1) | The core operation this lesson builds on top of. Student can compute it by hand and interacted with it in the ConvolutionExplorer widget. |
| Feature map (output of convolution) | DEVELOPED | what-convolutions-compute (3.1 L1) | Student knows feature maps are spatial grids of filter responses. This lesson introduces pooling which operates ON feature maps. |
| Edge detection filters | DEVELOPED | what-convolutions-compute (3.1 L1) | Student computed edge filter outputs by hand. Provides concrete grounding for understanding what pooling does to feature maps. |
| Output size formula (N - F + 1) | DEVELOPED | what-convolutions-compute (3.1 L1) | The "shrinking" problem this lesson solves with padding. Student knows convolution reduces spatial dimensions. |
| Spatial structure / locality | INTRODUCED | what-convolutions-compute (3.1 L1) | Motivation for why convolutions exist. This lesson extends it: pooling exploits spatial redundancy; stride controls how much the filter overlaps. |
| Weight sharing | INTRODUCED | what-convolutions-compute (3.1 L1) | Student can explain why same weights everywhere is powerful. Not directly extended this lesson, but reinforced when discussing the full architecture. |
| Multiple filters = multiple feature maps | INTRODUCED | what-convolutions-compute (3.1 L1) | Student knows a conv layer outputs a stack of feature maps. This lesson introduces the concept of channels flowing through the architecture. |
| Receptive field | INTRODUCED | what-convolutions-compute (3.1 L1) | Student knows each output position sees only a local neighborhood. Stacking layers increases receptive field — key insight for why the conv-pool pattern works. |
| Hierarchical feature composition (edges -> shapes -> objects) | MENTIONED | what-convolutions-compute (3.1 L1) | Planted as a teaser. This lesson develops it: the conv-pool pattern enables this hierarchy. |
| Learned filters (not hand-designed) | INTRODUCED | what-convolutions-compute (3.1 L1) | Student knows filters are learned via backprop. Reinforced when discussing nn.Conv2d. |
| Training loop (forward-loss-backward-update) | DEVELOPED | module 1.1-1.3 | Same loop applies to CNNs. No re-teaching needed. |
| nn.Module, forward(), parameters | DEVELOPED (assumed) | series-2 | Student can read and write PyTorch module code. Needed for nn.Conv2d and nn.MaxPool2d. |

### Mental Models Already Established

- "A filter is a pattern detector — it asks 'does this local region look like my pattern?' at every position"
- "The feature map is a spatial answer key — output position (i,j) tells you about the input neighborhood at (i,j)"
- "A convolutional layer asks a fixed set of questions at every location; the answers are the feature maps"
- "Same weighted sum as a neuron, but only over a local neighborhood"
- "A 28x28 filter on a 28x28 image IS a dense layer" (extreme case argument)

### What Was NOT Covered

- Pooling (any form — max, average, global)
- Stride (filter jumping instead of sliding one pixel at a time)
- Padding (adding zeros around the border)
- How to stack conv layers into an architecture
- The conv-pool-fc pattern (convolution -> pooling -> fully connected)
- nn.Conv2d or nn.MaxPool2d API
- Multiple input channels (RGB images, or the channel dimension between conv layers)
- How receptive field grows through stacking layers

### Readiness Assessment

The student is well-prepared. Every concept this lesson introduces (pooling, stride, padding, the CNN architecture pattern) builds directly on the convolution operation and feature maps from Lesson 1. The student knows what a feature map looks like, knows it shrinks (N - F + 1), knows each position sees a local region, and knows multiple filters produce stacks of feature maps. The three new concepts — pooling (simplifying feature maps), stride/padding (controlling dimensions), and the conv-pool-fc pattern (assembling it all) — are engineering decisions that sit on top of the conceptual foundation from Lesson 1.

**Activation energy concern:** This is a BUILD lesson following a STRETCH lesson. The student should feel: "I understand convolutions; now I am assembling them into something real." The opening should validate what they learned in Lesson 1, then quickly introduce a concrete problem (the feature maps are too big / information is redundant) that motivates pooling.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how pooling, stride, and padding control spatial dimensions in a CNN, and trace data through the conv-pool-fc architecture pattern that powers real convolutional networks.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Convolution as sliding filter | DEVELOPED | DEVELOPED | what-convolutions-compute (3.1 L1) | OK | Must be fluent with the operation to understand stride (which modifies the slide) and padding (which modifies the input before sliding) |
| Feature map | DEVELOPED | DEVELOPED | what-convolutions-compute (3.1 L1) | OK | Pooling operates on feature maps. Student must know what they are, how they are produced, and that they are spatial. |
| Output size formula (N - F + 1) | DEVELOPED | DEVELOPED | what-convolutions-compute (3.1 L1) | OK | This lesson generalizes the formula to include stride and padding. Student must have the base formula. |
| Spatial locality | INTRODUCED | INTRODUCED | what-convolutions-compute (3.1 L1) | OK | Motivation for pooling (nearby values are redundant) and for the architecture pattern (early layers detect local features, later layers see larger regions). Only need recognition-level here — lesson builds on it, does not require application. |
| Multiple filters = stack of feature maps | INTRODUCED | INTRODUCED | what-convolutions-compute (3.1 L1) | OK | The architecture has conv layers producing N feature maps, which become the input channels for the next conv layer. Student needs to understand the "stack" concept at recognition level. |
| Receptive field | INTRODUCED | INTRODUCED | what-convolutions-compute (3.1 L1) | OK | Stacking conv + pool layers increases the receptive field. Student needs to know what a receptive field is, which they do from Lesson 1. |
| Dense (fully connected) layers | DEVELOPED | DEVELOPED | module 1.2 | OK | The "fc" part of conv-pool-fc. Student has built dense networks. No re-teaching needed. |
| nn.Module / PyTorch basics | DEVELOPED (assumed) | DEVELOPED (assumed) | series-2 | OK | Needed for reading nn.Conv2d and nn.MaxPool2d code. Student has written modules in PyTorch. |

All prerequisites are met. No gaps require dedicated sections.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Pooling is just throwing away information — it should hurt accuracy" | Pooling literally discards values (max pooling keeps 1 of 4). Intuitively, throwing away 75% of data seems destructive. | Apply max pooling to a feature map from an edge detector: the edge is still clearly visible in the pooled output. The spatial location is approximate rather than exact, but the feature presence is preserved. A 2x2 pooled region saying "there is a strong edge somewhere in this 2x2 area" is almost as useful as knowing the exact pixel. Meanwhile, the spatial dimensions are halved, making the next layer 4x faster to compute. | During the pooling section, immediately after showing the max pooling computation. Show a before/after feature map. |
| "Stride and padding are just implementation details — they do not matter conceptually" | These sound like engineering knobs that do not change what the network does. The student might want to skip them as minor technicalities. | Without padding, a 32x32 input with a 3x3 filter becomes 30x30 after one conv layer. After 5 conv layers: 32 -> 30 -> 28 -> 26 -> 24 -> 22. The image shrinks by 10 pixels. With "same" padding, it stays 32x32 at every layer. Padding is not cosmetic — it determines whether border information survives through deep networks. Stride=2 halves the spatial dimensions (like pooling) but inside the convolution itself, changing the output fundamentally. | During the stride/padding section. Use a concrete dimension-tracking example through multiple layers. |
| "The output of a CNN is a feature map" | The student just learned that conv layers produce feature maps. They might think the CNN as a whole outputs a spatial grid. | A CNN classifying digits outputs a 10-element vector (one probability per class), not a feature map. The transition from spatial feature maps to a flat classification vector is exactly what the fully-connected layers at the end do. Show the shape at each stage: 28x28x1 -> 14x14x32 -> 7x7x64 -> 3136 -> 128 -> 10. The feature maps are intermediate, not final. | During the conv-pool-fc architecture section, when introducing the "flatten + FC" step. |
| "You need pooling after every conv layer" | The classic diagrams always show conv-pool-conv-pool. The student might think this is a rigid rule. | Modern architectures (ResNet, for example) use stride=2 convolutions instead of pooling in many places. Even in the classic LeNet pattern, the ratio of conv to pool layers is a design choice. You can stack multiple conv layers before pooling. The only rule is that spatial dimensions need to shrink at some point so the network can aggregate information. | In the architecture section, as a brief callout when presenting the conv-pool-fc pattern. Frame it as "the classic pattern" rather than "the only pattern." |
| "A bigger feature map (more spatial resolution) is always better" | More data = better, right? Higher resolution captures more detail. | Higher resolution means more computation, more memory, and each subsequent layer sees less spatial context per filter application. A 3x3 filter on a 224x224 feature map sees only a tiny fraction of the image. Reducing spatial dimensions via pooling or stride means each filter application covers more of the original image in later layers — this is HOW the network goes from detecting edges to detecting objects. Keeping full resolution would trap the network in local feature detection forever. | During the "why pooling matters" explanation, connecting back to receptive field growth. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Max pooling on a 4x4 feature map with 2x2 window | Positive | First concrete computation: show exactly how max pooling takes a 4x4 grid and produces a 2x2 grid. Shows the mechanical operation. | Small enough to compute fully (4 windows, 4 max operations). Student can see the entire input and output at once. The 4x4 -> 2x2 halving is the most common pooling configuration. |
| Max pooling on an edge-detection feature map — feature preserved | Positive | Shows that pooling preserves important information (edge presence) while reducing spatial size. Answers "why is throwing away values OK?" | Directly connects to what the student learned in Lesson 1. Uses the same edge-detection context they are comfortable with. The "before and after" makes the information preservation concrete. |
| Average pooling on the same feature map — smoothing effect | Positive | Shows a different pooling strategy and its different behavior. Average pooling blurs rather than selecting max. Confirms pooling is a design choice, not a single operation. | Contrast with max pooling. Student sees that max pooling preserves peak responses (edges stay sharp) while average pooling smooths them. Builds intuition for why max pooling is more common for feature detection. |
| Dimension tracking through a full CNN: 28x28x1 -> conv -> pool -> conv -> pool -> flatten -> fc -> output | Positive (stretch) | The "aha" example: trace shapes through the entire architecture. This is the payoff — seeing how all the pieces fit together. | Uses MNIST dimensions (28x28) the student already knows. Makes the abstract architecture concrete by showing exact shapes at every stage. The flatten step (7x7x64 -> 3136) is where spatial -> flat happens. |
| Conv layer with stride=2 vs conv + separate pooling — same output dimensions, different tradeoffs | Positive | Shows that stride inside the convolution can replace separate pooling. Demonstrates that these are engineering choices, not separate concepts. | Prevents the misconception that pooling is the ONLY way to reduce dimensions. Shows flexibility in architecture design. |
| Convolution WITHOUT padding on a small input, repeated 5 times — shrinks to almost nothing | Negative | Shows what goes wrong without padding: the feature map shrinks with every layer until there is almost nothing left. Border pixels only contribute to one output position while center pixels contribute to many — information at the borders is systematically underrepresented. | Makes padding feel necessary rather than optional. The student sees the problem before getting the solution. |

---

## Phase 3: Design

### Narrative Arc

You finished Lesson 1 knowing what a single convolution computes: a small filter slides across an image, computing weighted sums at each position, producing a feature map that shows where the filter's pattern was found. But a single convolution is not a CNN — it is one building block. Right now you have one piece of the puzzle: you can detect edges (or textures, or gradients) at every location. But what do you do with a 5x5 feature map full of edge responses? It has the same spatial dimensions as the input (roughly). If you want to classify the image — "is this a 3 or a 7?" — you need to go from a spatial grid of local features to a single decision. And if you want to detect not just edges but corners, shapes, and eventually whole objects, you need to stack convolutions so later layers can combine the features from earlier ones. This lesson is about assembling the pieces: pooling to shrink spatial dimensions and build spatial tolerance, stride and padding to control how dimensions change, and the classic conv-pool-fc architecture pattern that takes raw pixels and transforms them step by step into a classification. By the end, you will be able to trace data through a complete CNN and explain what happens at every stage — which is exactly what you need to build one yourself in Lesson 3.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | 4x4 feature map with max pooling (2x2 window, stride 2): show every window, every max selection, produce 2x2 output. Also: dimension tracking through a complete CNN with specific numbers (28x28x1 -> ... -> 10). | Pooling and dimension changes are mechanical — students must see specific numbers to internalize how dimensions change. Abstract "halves the size" is insufficient without seeing it happen. |
| Visual/Interactive | An interactive widget that lets the student configure a CNN architecture (number of conv layers, filter sizes, pooling yes/no, stride, padding) and see the shape of data at every stage. A "dimension calculator" that shows the flow from input to output. | The architecture is a pipeline. Seeing the shapes transform step by step makes the pattern visible. Interactive exploration lets the student ask "what if" questions: what if I remove pooling? What if I use stride=2? What if I skip padding? |
| Symbolic | The generalized output size formula: output = floor((N - F + 2P) / S) + 1, where P=padding, S=stride. Shown AFTER concrete examples, not before. | Students need the general formula to compute dimensions in their own architectures. But it must come after they have seen enough concrete examples that the formula feels like a summary, not a surprise. |
| Verbal/Analogy | Pooling as "zooming out" — each pooling step is like stepping back from an image. Close up you see individual pixels and edges. Step back and you see shapes and regions. Step back further and you see the whole object. The conv-pool stack is a series of zoom-outs, each level detecting increasingly abstract patterns. | This is the key intuition for why the conv-pool pattern works. Without it, pooling seems like arbitrary information destruction. The "zoom out" metaphor connects to everyday experience and explains why later layers have larger effective receptive fields. |
| Geometric/Spatial | A diagram showing the "data pipeline" of a CNN: rectangles getting shorter and wider (spatial dimensions shrink, channel count grows) flowing from left to right, ending in a thin tall vector. The classic CNN architecture diagram. | This is the canonical way CNNs are visualized in papers and textbooks. The student needs to be able to read these diagrams. The shape change (wide+shallow -> narrow+deep -> flat vector) is the visual signature of a CNN. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3 new concepts at target depths:
  1. Pooling (max pooling primarily, average pooling briefly) — target DEVELOPED
  2. Stride and padding — target DEVELOPED
  3. The conv-pool-fc architecture pattern — target DEVELOPED
- **What was the load of the previous lesson?** STRETCH (what-convolutions-compute). The student learned the convolution operation, feature maps, edge detection, weight sharing, spatial locality — a heavy conceptual load.
- **Is this lesson's load appropriate?** Yes. BUILD after STRETCH is the planned trajectory. The three new concepts here are engineering extensions of the foundation from Lesson 1, not fundamentally new ideas. Pooling is "take the max of a region" (trivially mechanical). Stride and padding modify the existing output size formula. The architecture pattern is assembling known pieces. Cognitive effort goes into seeing how pieces fit together, not learning new abstractions.

### Connections to Prior Concepts

| New Concept | Prior Concept | Connection |
|-------------|--------------|------------|
| Max pooling (take max of a 2x2 region) | Convolution (weighted sum of a 3x3 region) | "Pooling is even simpler than convolution — it just takes the max (or average) of a small region. Same sliding-window idea, but instead of multiply-and-sum, just find the largest value." |
| Stride (filter jumps by S positions) | Convolution sliding one position at a time | "In Lesson 1, the filter moved one pixel at a time. Stride=2 means it jumps two pixels. Fewer positions visited = smaller output. This is why pooling with stride=2 halves the spatial dimensions." |
| Padding (adding zeros around the border) | Output size shrinks: N - F + 1 | "You saw that a 7x7 input with a 3x3 filter gives a 5x5 output — it shrinks by 2. Padding adds a border of zeros so the output stays the same size (or shrinks less). It is the fix for the shrinking problem." |
| Conv-pool-fc pattern | Dense (fully connected) layers from Series 1 + conv from Lesson 1 | "You already know dense layers (every input connected to every output) and conv layers (local connections, shared weights). A CNN uses conv layers to extract spatial features, pooling to shrink dimensions, then dense layers at the end to make a decision. You already know both halves — this lesson connects them." |
| Receptive field growth through stacking | Receptive field = 3x3 local region from Lesson 1 | "In Lesson 1, each output position saw a 3x3 region. Stack two conv layers and the second layer's output sees a 5x5 region of the original input. Add pooling and it grows even faster. This is how the network goes from seeing edges to seeing objects." |

**Potentially misleading prior analogies:**
- The "pattern detector" analogy from Lesson 1 applies to individual conv layers but not to the full architecture. Need to extend it: "Early layers detect simple patterns (edges). Later layers combine those patterns into complex features (corners, shapes, objects). The hierarchy IS the pattern detection — each level asks questions about the answers from the previous level."
- The "feature map is a spatial answer key" analogy still holds for individual layers but the student needs to understand that after flatten, the spatial structure is gone — by design. The transition from spatial to flat is not a loss; it is the point where the network switches from "where are the features?" to "what do the features mean?"

### Scope Boundaries

**This lesson IS about:**
- What pooling does (max pooling primary, average pooling secondary), computed concretely
- How stride and padding modify output dimensions
- The generalized output size formula with stride and padding
- The conv-pool-fc architecture pattern — tracing data shapes through a full CNN
- How receptive field grows through stacking layers and pooling
- nn.Conv2d and nn.MaxPool2d at a reading-comprehension level (show code, explain arguments)

**This lesson is NOT about:**
- Training a CNN (Lesson 3)
- Comparing CNN vs dense accuracy (Lesson 3)
- Architecture design choices (how many layers, how many filters) in depth (Lesson 3)
- Backprop through pooling or conv layers (relies on existing backprop knowledge)
- Global average pooling (modern technique, out of scope for this module)
- Batch normalization, dropout in CNNs (out of scope)
- Advanced architectures (ResNet, VGG, etc.) beyond brief mentions
- Multiple input channels / RGB images (brief mention, not developed)

**Target depths:**
- Pooling (max pooling): DEVELOPED — student can compute it and explain why it is useful
- Stride: DEVELOPED — student can compute output size with stride and explain the tradeoff
- Padding: DEVELOPED — student can compute output size with padding and explain when/why to use it
- Conv-pool-fc architecture pattern: DEVELOPED — student can trace shapes through the pipeline and explain each stage's role
- Average pooling: INTRODUCED — student recognizes it and can contrast with max pooling
- Receptive field growth: INTRODUCED — student can explain why stacking increases receptive field (not compute exact sizes)
- nn.Conv2d / nn.MaxPool2d: INTRODUCED — student can read the code and identify key arguments

### Lesson Outline

**1. Context + Constraints**
What this lesson is about: assembling convolutions into a full CNN architecture — with pooling, stride, padding, and the classic conv-pool-fc pattern. What we are NOT doing: training a CNN or comparing accuracy — that is Lesson 3. Connection to the arc: "Lesson 1 gave you the building block. This lesson assembles the architecture. Lesson 3 builds and trains it."

**2. Hook — The "So What?" Problem (puzzle)**
Type: Challenge preview. "You can compute what a convolution produces. But consider this: your edge-detection filter on a 28x28 MNIST image produces a 26x26 feature map. If you use 32 filters, you have 32 feature maps of 26x26 = 21,632 values. That is MORE than the 784 pixels you started with. And you want to classify this as a digit (one of 10 classes). How do you get from 21,632 spatial feature values to a 10-class prediction? And if you stack another conv layer, the computation gets even more expensive. Something needs to shrink." This motivates pooling and the architecture.

**3. Explain: Pooling — Shrinking with Purpose**
- Start with the problem: feature maps are spatially redundant. If there is a strong vertical edge at position (5,5), there is probably also a strong response at (5,6). We do not need pixel-perfect spatial precision — approximate location is enough.
- **Max pooling:** Concrete example first. Take a 4x4 feature map (with meaningful values from an edge-detection example). Apply 2x2 max pooling with stride 2. Walk through each of the four 2x2 windows: take the max. Produce a 2x2 output. Four numbers summarize what sixteen numbers said.
- **Before/after comparison:** Show the original feature map (from Lesson 1's edge detection) and the pooled version side by side. The edge is still visible — its exact position is slightly fuzzier, but its presence is clear.
- **Average pooling:** Same 4x4 input, average instead of max. Show the difference: averages are smoother, maxes preserve peak responses. "Max pooling asks: 'is the feature present in this region?' Average pooling asks: 'how strong is the feature on average in this region?'"
- **The "zoom out" analogy:** Each pooling step is like stepping back from the image. Details blur but the overall picture becomes clearer. This is how the network goes from seeing edges to seeing shapes.
- **Spatial invariance:** Max pooling makes the network slightly tolerant of small shifts. An edge at (5,5) or (5,6) produces the same pooled value. This is desirable for recognition — a "3" shifted one pixel right is still a "3."

**4. Check 1 — Compute the Pool**
Give a 6x6 feature map. Ask: "Apply 2x2 max pooling with stride 2. What is the output size? What is the value at position (1,0)?" Tests mechanical understanding of pooling and dimension change.

**5. Explain: Stride and Padding — Controlling Dimensions**
- **Stride:** "In Lesson 1, the filter moved one pixel at a time (stride=1). What if it jumped two pixels (stride=2)? It visits fewer positions, producing a smaller output." Show with a tiny example: 6x6 input, 3x3 filter, stride 1 -> 4x4 output. Same input, stride 2 -> 2x2 output. The formula: output = floor((N - F) / S) + 1.
- **Padding:** Start from the problem. "A 3x3 filter on a 7x7 input gives 5x5. Another conv layer: 5x5 -> 3x3. Another: 3x3 -> 1x1. Three layers and you are down to a single pixel." Show this shrinking chain. "Padding fixes this: add a border of zeros around the input before convolving." Show: 7x7 + padding=1 -> 9x9, then 3x3 filter gives 7x7. Output same size as input. "This is called 'same' padding."
- **Negative example:** The shrinking chain without padding. Five conv layers on a 32x32 input: 32 -> 30 -> 28 -> 26 -> 24 -> 22. Ten pixels lost. Border information is systematically underrepresented because border pixels participate in fewer convolution windows.
- **The general formula:** output = floor((N - F + 2P) / S) + 1. Show it AFTER the concrete examples. Walk through it with the examples they just saw to verify it matches.
- **Brief PyTorch connection:** Show nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1). Point out how the arguments map to the formula. Do not deep-dive into the API — just reading comprehension.

**6. Check 2 — Predict the Shape**
"A 28x28 input goes through nn.Conv2d(1, 32, 3, padding=1). What is the output shape?" Then: "That output goes through nn.MaxPool2d(2). What is the shape now?" Tests stride, padding, and pooling together on a familiar (MNIST) input size.

**7. Explore — Interactive Widget: CNN Dimension Calculator**
An interactive widget where the student builds a CNN layer by layer and sees the data shape transform at each stage. They can add conv layers (choose filter size, stride, padding, number of filters), pooling layers (max or average, window size), and a flatten + FC layer at the end. The widget shows a visual pipeline of shapes flowing left to right, with each layer labeled and the dimensions annotated. Key interactions:
- Add / remove layers
- Configure each layer's parameters
- See the output shape update in real time
- See a warning if dimensions become too small (< 1) or if the flatten produces an unexpectedly large vector
- Preset architectures: "LeNet-style" (conv-pool-conv-pool-fc-fc) and "Custom"

**8. Explain: The Conv-Pool-FC Pattern**
- Now bring it all together. "Most CNNs follow a simple pattern: convolution layers to extract features, pooling layers to shrink spatial dimensions, then fully-connected (dense) layers at the end to classify."
- **Dimension tracking example:** Trace MNIST through a complete CNN:
  - Input: 28x28x1
  - Conv(32 filters, 3x3, padding=1): 28x28x32
  - MaxPool(2x2): 14x14x32
  - Conv(64 filters, 3x3, padding=1): 14x14x64
  - MaxPool(2x2): 7x7x64
  - Flatten: 3136
  - FC(128): 128
  - FC(10): 10
- **The shape diagram:** Show the classic CNN architecture visualization — rectangles getting shorter but deeper (spatial shrinks, channels grow), then a sudden transition to a thin tall vector (flatten + FC).
- **Why this pattern:** Early conv layers detect local features (edges, textures). Pooling shrinks spatial dimensions so the next conv layer's 3x3 filter covers more of the original image. Later conv layers detect combinations of earlier features (corners, shapes). By the time we flatten, the spatial dimensions are small (7x7) but the channel count is rich (64 features). The FC layers take this compressed representation and make a decision.
- **Receptive field growth:** "In Lesson 1, a single 3x3 filter saw a 3x3 region. After one pool + one conv, the second conv layer's filter sees an effective 6x6 region of the original input (because pooling halved the resolution). This is how the network goes from local features to global understanding — without using the huge filters we showed are impractical."
- **Brief mention:** Modern architectures sometimes use stride=2 conv layers instead of separate pooling. The conv-pool-fc pattern is the classic starting point, not the only option.

**9. Check 3 — Explain the Architecture**
Show a CNN architecture in PyTorch code (a simple nn.Sequential with Conv2d, ReLU, MaxPool2d layers, Flatten, Linear). Ask: "What is the output shape after the second MaxPool2d? Why does the model Flatten before the Linear layer? What would happen if you removed all padding?" Tests whether the student can read an architecture, trace dimensions, and explain the reasoning behind each choice.

**10. Elaborate: Why This Architecture Works**
- Connect back to the hierarchy teaser from Lesson 1: "We mentioned that later layers detect more complex patterns. Now you can see WHY — each conv-pool stage expands the receptive field and compresses spatial dimensions, forcing the network to represent information in increasingly abstract terms."
- Brief mention of real architectures: LeNet (1998) used this exact conv-pool-fc pattern on handwritten digits. It was one of the first successful CNNs. The pattern remains the backbone even of complex modern architectures.
- Misconception address: "You do NOT need pooling after every conv layer. Many architectures stack 2-3 conv layers before pooling. And modern networks often use stride=2 convolutions instead of separate pooling. The principle is: spatial dimensions need to shrink at some point so the network can see the big picture."

**11. Summarize**
Key takeaways:
- Max pooling takes the maximum value in each region, shrinking spatial dimensions while preserving feature presence
- Stride controls how far the filter jumps; padding adds zeros around the border to control output size
- The general formula: output = floor((N - F + 2P) / S) + 1
- The conv-pool-fc pattern: convolutions extract spatial features, pooling shrinks dimensions (growing receptive field), fully-connected layers classify
- Data shape transforms: spatial dimensions shrink, channel count grows, then flatten for classification

Echo the mental model: "A CNN is a series of zoom-outs. Each conv layer detects patterns at the current scale. Each pooling step zooms out, so the next conv layer detects patterns at a larger scale. By the end, the network has gone from pixels to edges to shapes to a classification decision."

**12. Next Step**
"You now know all the pieces of a CNN: convolution, pooling, stride, padding, and the conv-pool-fc architecture. Next: you will build one from scratch in PyTorch, train it on MNIST, and see it beat the dense network you built in Series 2. That is the payoff for everything in this module."

### Widget Specification: CNN Dimension Calculator

**Purpose:** Let the student build a CNN architecture layer by layer and see how data shapes transform at each stage. The emphasis is on understanding the dimension flow — how spatial dimensions shrink and channels grow through the pipeline.

**Key interactions:**
- Add layers: Conv2d (configurable filter_size, stride, padding, out_channels), MaxPool2d (configurable kernel_size, stride), Flatten, Linear (configurable out_features)
- Remove layers
- See the data shape after each layer, updated in real time
- Visual pipeline: rectangles proportional to spatial dimensions and channel depth
- Warnings: "dimensions too small" (< 1), "flatten produces very large vector" (> 10000)
- Preset: "LeNet-style" populates a working architecture for MNIST

**NOT in scope for this widget:**
- Actual computation (no weights, no forward pass, no training)
- Activation functions (mentioned in text, not in widget)
- Batch normalization, dropout
- Real images flowing through the network

**Visualization approach:** A horizontal pipeline of labeled blocks. Each block shows [H x W x C] dimensions. Blocks are connected by arrows. Conv blocks are one color, pool blocks another, FC blocks another. When dimensions are invalid, the block turns red with a warning.

**Library:** Custom component with CSS/Tailwind. The visualization is shapes and text, not charts. No external visualization library needed.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (what-convolutions-compute record, module 3.1 record)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (5 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 3 DEVELOPED + 2 INTRODUCED + 1 INTRODUCED concepts (within bounds for BUILD)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 5
- Polish: 3

### Verdict: MAJOR REVISION

One critical finding: the planned fifth misconception ("you need pooling after every conv layer") is addressed only as a floating callout in the elaboration section rather than with a proper negative example as specified in the plan. Five improvement findings affect the lesson's pedagogical completeness: a missing planned example (stride=2 conv vs separate pooling), the widget missing the planned "proportional rectangle" spatial visualization, the receptive field growth explanation lacking a concrete computation, the general dimension formula being shown without first grounding all three parameters in separate concrete examples, and Check 1 testing a slightly different operation than what was just taught. Three polish items round out the findings.

### Findings

#### [CRITICAL] — Planned misconception "you need pooling after every conv layer" lacks a concrete negative example

**Location:** Lines 975-998 (the violet callout box in "Why This Architecture Works" section)
**Issue:** The plan's misconception table specifies: misconception 4 is "You need pooling after every conv layer," with a negative example that shows "Modern architectures (ResNet) use stride=2 convolutions instead of pooling in many places. Even in the classic LeNet pattern, the ratio of conv to pool layers is a design choice. You can stack multiple conv layers before pooling." The built lesson addresses this with a styled callout box (lines 978-988) that says "You do NOT need pooling after every conv layer" and mentions ResNet and stride=2 convolutions. However, this is a declarative statement, not a negative example. The plan called for the student to see a concrete architecture that stacks 2-3 conv layers before pooling and compare it to the alternating conv-pool pattern. The callout tells the student what to think instead of showing them an architecture where pooling-after-every-conv is NOT used. The TryThisBlock aside for the widget (line 758-760) suggests "Stack 3 conv layers before the first pool" as an experiment, which partially addresses this, but the experiment is optional and disconnected from the misconception callout.
**Student impact:** The student reads "you don't need pooling after every conv layer" as an assertion and may accept it or may not internalize it. Without seeing a concrete example where a non-alternating pattern works, the canonical conv-pool-conv-pool from the dimension tracking example (the only full architecture shown) will dominate their mental model. The misconception persists because the lesson's primary example reinforces the alternating pattern.
**Suggested fix:** Either (a) add a second, brief dimension-tracking example immediately after the callout showing a 3-conv-then-pool architecture (e.g., Conv-Conv-Conv-Pool-Flatten-FC) with dimensions traced, or (b) move the relevant TryThisBlock experiment ("Stack 3 conv layers before the first pool") into the main content as a guided exploration immediately after the callout, with explicit dimension annotations showing it works.

#### [IMPROVEMENT] — Planned example "stride=2 conv vs separate pooling" is missing

**Location:** Missing from the lesson entirely
**Issue:** The plan's examples table includes Example 5: "Conv layer with stride=2 vs conv + separate pooling — same output dimensions, different tradeoffs." This was a positive example specifically designed to show that stride inside the convolution can replace separate pooling, demonstrating that pooling is not the only way to reduce dimensions. The built lesson mentions stride=2 as equivalent to pooling in the "Stride vs Pooling" aside (lines 500-506) and in the callout box (lines 978-988), but never shows a concrete side-by-side comparison with actual dimensions. The student is told stride=2 convolutions can replace pooling but never sees the two architectures compared with traced dimensions.
**Student impact:** The student understands stride=2 reduces dimensions (the stride section teaches this well with the 6x6 example), but does not see the direct substitution demonstrated. The connection between "stride in a convolution" and "replacing a pooling layer" remains abstract. The TryThisBlock in the widget aside suggests this experiment, but optional exploration is weaker than a taught example.
**Suggested fix:** Add a brief concrete comparison in the stride section or in the conv-pool-fc section. For example: "Compare: Conv(32, 3x3, stride=2, pad=1) on 28x28 gives 14x14x32. Conv(32, 3x3, stride=1, pad=1) + MaxPool(2x2) also gives 14x14x32. Same dimensions, different mechanism." This does not need to be a full section, just 3-4 sentences with concrete numbers.

#### [IMPROVEMENT] — Widget lacks the planned "proportional rectangle" spatial visualization

**Location:** CnnDimensionCalculator widget, PipelineVisualization component (lines 332-391)
**Issue:** The plan specified under modalities: "A diagram showing the 'data pipeline' of a CNN: rectangles getting shorter and wider (spatial dimensions shrink, channel count grows) flowing from left to right, ending in a thin tall vector. The classic CNN architecture diagram." The plan also stated under widget spec: "Visual pipeline: rectangles proportional to spatial dimensions and channel depth." The built widget shows a horizontal pipeline of labeled blocks, but all blocks are the same size (min-w-[60px], same padding). The shape information is shown as text labels (e.g., "28x28x32", "14x14x32") inside uniformly-sized blocks. The blocks do NOT get shorter/wider proportional to the actual spatial dimensions and channel depth. The "classic CNN architecture diagram" where the student can visually see the spatial-shrinks-channels-grow pattern through block proportions is absent.
**Student impact:** The student gets the dimension information textually but misses the visual/spatial modality for the data flow. The planned geometric/spatial modality ("rectangles getting shorter and wider") was intended to give the student a visual imprint of the CNN's data transformation. Without it, the "spatial shrinks, channels grow" pattern must be inferred from reading numbers rather than seen in the shape of the visualization.
**Suggested fix:** Modify the PipelineVisualization to make block heights proportional to spatial dimensions and block widths proportional to channel count (or vice versa), even approximately. This does not need to be pixel-perfect; even a rough proportional scaling (large blocks for 28x28, medium for 14x14, small for 7x7, thin for flat) would convey the pattern visually. Alternatively, add a separate static "architecture diagram" below the widget that shows the proportional rectangle visualization for the current configuration.

#### [IMPROVEMENT] — Receptive field growth explanation lacks concrete computation

**Location:** Lines 847-855 (ConceptBlock "Receptive Field Growth" aside)
**Issue:** The plan specified under connections: "In Lesson 1, a single 3x3 filter saw a 3x3 region. After one pool + one conv, the second conv layer's filter sees an effective 6x6 region of the original input (because pooling halved the resolution). This is how the network goes from edges to objects." The built lesson's aside (lines 849-854) says essentially the same thing but presents it only in an aside, not in the main content. The main content (lines 836-843) says "Later conv layers detect combinations of earlier features" and "Pooling shrinks spatial dimensions so the next conv layer's 3x3 filter covers more of the original image" but never works through the specific receptive field arithmetic. The plan's target depth for receptive field growth is INTRODUCED, which only requires the student to "explain why stacking increases receptive field (not compute exact sizes)." So the aside placement is technically sufficient for depth level, but the main content statement is vague enough that a student might not connect "covers more of the original image" to a specific geometric fact.
**Student impact:** Moderate. The student understands that later layers "see more" but does not have a concrete number to anchor this understanding. "Covers more of the original image" is hand-wavy compared to "a 3x3 filter after a 2x2 pool effectively sees a 6x6 region of the original input." The aside provides this specificity, but asides are optional reading.
**Suggested fix:** Move the concrete example (3x3 filter after 2x2 pool = 6x6 effective receptive field) from the aside into the main content, keeping it brief: one sentence with the specific numbers. The aside can retain the broader framing about edges-to-objects.

#### [IMPROVEMENT] — General dimension formula presented before stride and padding are grounded together in a concrete example

**Location:** Lines 580-628 (the general formula section)
**Issue:** The ordering of concepts in the stride/padding section is: (1) stride explained with a concrete example (lines 478-506), (2) padding explained with a concrete example (lines 509-577), (3) the general formula combining both (lines 580-628). This ordering is mostly correct (parts before whole), but the concrete examples for stride and padding are taught separately, and the first time the student sees stride AND padding together is in the general formula itself. The three verification examples (lines 608-617) show stride=1/no padding, stride=2/no padding, and stride=1/padding=1, but never stride + padding together. The student does not see a worked example where both stride > 1 AND padding > 0 are in play simultaneously before being given the formula.
**Student impact:** Low-moderate. The formula is presented after the parts, which is correct. But the student may struggle to apply the formula when BOTH stride and padding are non-default, because they have only seen each in isolation. Check 2 (lines 670-717) tests stride=1/padding=1 (no stride variation) and then pooling (stride=2, no padding), so even the checks do not exercise both parameters together.
**Suggested fix:** Add one verification example to the formula section that uses both stride and padding: e.g., "28x28 input, 3x3 filter, stride=2, padding=1: (28 - 3 + 2) / 2 + 1 = 14." This also primes the student for the architecture section where Conv(3x3, stride=1, pad=1) is used throughout.

#### [IMPROVEMENT] — Check 1 tests max pooling on a 6x6 grid but the worked example used a 4x4 grid

**Location:** Lines 408-455 (Check 1: "Compute the Pool")
**Issue:** The worked example in the pooling section uses a 4x4 grid with 2x2 max pooling, producing a 2x2 output. Check 1 immediately asks about a 6x6 grid with 2x2 max pooling. The student has seen one worked example at one size and is now asked to apply the operation at a different size. This is fine for testing generalization, but question 2 asks: "If the input has value 8 at position (2,3) and values 1, 4, 3 at the other three positions in its 2x2 window, what is the output value at position (1,1)?" The student must map position (2,3) in a 6x6 grid to pooling window (1,1) in the output. This requires understanding that the second pooling window in each dimension starts at row 2 (or column 2), which was shown for the 4x4 case but not for the 6x6 case. The position mapping (input position to output window) is a new cognitive step that was not explicitly taught.
**Student impact:** The student may get the output size correct (3x3) by dividing 6/2, but may struggle with question 2 because they need to figure out which 2x2 window maps to output position (1,1). This is a fair test if the student understands the general window-mapping principle, but the lesson only showed one specific case (the 4x4 example) where the mapping was worked through explicitly.
**Suggested fix:** Either (a) add a brief note before or in the check explaining how to find which window maps to a given output position: "Output position (r,c) corresponds to the input window starting at (r*stride, c*stride)," or (b) simplify question 2 to not require the position mapping: "What is the maximum value in the top-left 2x2 window of a 6x6 grid?" Both approaches reduce the untaught cognitive step.

#### [POLISH] — Average pooling comparison section has no aside

**Location:** Lines 357-403 (average pooling comparison and ComparisonRow)
**Issue:** This is the only substantial content section in the lesson without any aside content. Every other section has a TipBlock, InsightBlock, WarningBlock, or ConceptBlock in the aside. The average pooling section—which introduces a new concept (average pooling at INTRODUCED depth) and includes a ComparisonRow—has the Row component without a Row.Aside, so the DefaultAside renders (empty column). This is structurally correct (the Row component handles it), but pedagogically, a brief aside could reinforce why max pooling is more common or when average pooling appears (e.g., "Global Average Pooling is used at the end of modern architectures like ResNet to replace the flatten + FC pattern entirely").
**Student impact:** Negligible. The content is self-contained. But the visual pattern of "every section has sidebar content" is broken here, which may make the section feel less polished.
**Suggested fix:** Add a brief TipBlock or ConceptBlock aside. Possible content: "Max pooling is the default choice for feature detection layers. Average pooling shows up most often at the very end of modern architectures (global average pooling) or in specialized contexts. When in doubt, use max pooling."

#### [POLISH] — Widget `idCounter` is a module-level mutable variable

**Location:** CnnDimensionCalculator.tsx, line 143 (`let idCounter = 0`)
**Issue:** The `idCounter` variable is a module-level `let` that increments every time `nextId()` is called. In React's development mode with Strict Mode, components render twice, which means `makeLeNetLayers` (called in the `useState` initializer) will produce different IDs on the double-render. More importantly, if the component is unmounted and remounted, the counter does not reset, so IDs will continue incrementing (e.g., "layer-8", "layer-9" instead of "layer-1", "layer-2"). This does not cause functional bugs because the IDs are used only as React keys and internal identifiers, but it is not idiomatic React.
**Student impact:** None. This is a code hygiene issue.
**Suggested fix:** Use `useRef` for the counter inside the component, or use `crypto.randomUUID()` / `Math.random().toString(36)` for unique IDs that do not depend on global mutable state.

#### [POLISH] — The `_widthOverride` prop in CnnDimensionCalculator is unused

**Location:** CnnDimensionCalculator.tsx, lines 406-411
**Issue:** The component accepts `width` and `height` props (via `CnnDimensionCalculatorProps`) but immediately destructures `width` as `_widthOverride` and ignores `height` entirely. The underscore prefix follows the convention for intentionally unused variables, but if the props serve no purpose, they should be removed from the type definition. If they were planned for responsive behavior, they are incomplete.
**Student impact:** None. Code cleanliness issue.
**Suggested fix:** Either implement the width/height responsive behavior or remove the unused props from the type definition.

### Review Notes

**What works well:**
- **The hook is strong.** The "So What?" problem is well-motivated. Starting from "32 filters on 28x28 = 21,632 values, more than the 784 pixels you started with" is a concrete, surprising fact that creates genuine need for pooling. Problem before solution is followed correctly.
- **Max pooling is taught excellently.** The 4x4 worked example with explicit window-by-window max computation is concrete, the before/after edge-detection feature map example addresses the "pooling destroys information" misconception elegantly, and the ComparisonRow between max and average pooling is clear.
- **The stride/padding section follows the ordering rules well.** Both concepts are introduced with a problem first (stride: "what if it jumped two pixels?", padding: "what happens without padding?"), then the solution. The "shrinking problem" visual showing 5 layers without padding is an effective negative example.
- **The general formula is shown AFTER concrete examples**, and then verified against the examples the student already saw. This is good concrete-before-abstract ordering.
- **The dimension-tracking example (28x28 through the full CNN to 10)** is the lesson's centerpiece and it works. The StageRow component with color-coded dots and annotations makes the pipeline readable. The "spatial shrinks, channels grow" pattern is clearly articulated.
- **Connection to prior knowledge is consistent.** The lesson explicitly references "What Convolutions Compute" when building on the output size formula, when explaining stride (filter moved one pixel at a time in Lesson 1), and when discussing receptive field growth. The connection to dense layers from Series 1 ("you already know both halves") is made at the right moment.
- **The interactive widget (CnnDimensionCalculator) is well-designed.** Layer adding/removal, real-time shape computation, error detection (spatial after flatten, linear before flatten), pipeline visualization, presets, and clear color coding per layer type. The TryThisBlock experiments are well-chosen and build genuine "what if" curiosity.
- **Scope boundaries are respected.** The lesson does not drift into training, accuracy comparisons, or deep architecture design. Forward pointers to Lesson 3 are brief and appropriate.
- **All Row layout patterns are correct.** Every content section uses `<Row>/<Row.Content>` with optional `<Row.Aside>`. No manual flex layouts.
- **Code conventions are followed.** No `switch` statements, no `any` types, no `else if`/`else` patterns. Em dashes have no surrounding spaces.
- **The summary is comprehensive and well-structured.** Five key takeaways covering all core concepts, followed by a mental model echo that ties everything together with the "zoom out" analogy.

**Structural patterns:**
- The lesson follows the plan's outline faithfully. All 12 planned sections are present in the correct order. The deviation in examples (missing stride-vs-pooling comparison) and misconception treatment (callout instead of negative example) are the main divergences from the plan.
- The lesson is well-paced for a BUILD lesson. After the dense STRETCH of Lesson 1, this lesson's concepts (pooling, stride, padding) are mechanically simple and the lesson correctly treats them as engineering extensions of the foundation, not fundamentally new abstractions. The cognitive load feels appropriate.

**Recommendation for revision:** Fix the CRITICAL item first (add a concrete negative example for the "pooling after every conv" misconception). Then address the five IMPROVEMENT items, prioritizing the missing stride-vs-pooling comparison and the widget spatial visualization. Re-review after fixes.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 5

### Verdict: NEEDS REVISION

All critical and improvement findings from Iteration 1 have been addressed. The lesson is now pedagogically complete: all planned examples are present, all misconceptions have concrete negative examples, the widget has proportional rectangle visualization, the receptive field computation is in the main content, the formula section includes a combined stride+padding example, and Check 1 now uses the same grid size as the worked example. Three new improvement findings surfaced in this pass: the stride=2 comparison uses padding before padding is taught (ordering violation), the receptive field arithmetic presents an oversimplified calculation as precise, and the average pooling section lacks an aside for a concept at INTRODUCED depth. Five polish items round out the findings.

### Findings

#### [IMPROVEMENT] — Stride=2 conv comparison placed before padding is taught, but uses padding=1

**Location:** Lines 514-557 (Stride=2 conv vs separate pooling comparison)
**Issue:** The stride=2 conv vs pooling comparison is placed in the stride section, immediately after the stride GradientCard (lines 485-512). This comparison uses "Conv(32, 3x3, stride=2, pad=1)" and shows the formula "(28 - 3 + 2) / 2 + 1 = 14x14x32." The student has not yet been taught padding -- that section starts at line 559. The comparison uses padding=1 in the stride=2 conv option, and padding=1 + MaxPool in Option B. The student encounters "pad=1" and the formula term "+2" (from 2*P) without having been introduced to what padding is or why it is used. The general formula has not been presented yet either (it appears at line 639). The student must accept "pad=1" on faith here and cannot verify the computation.
**Student impact:** A student reading sequentially will see "pad=1" for the first time in this comparison and will not understand what it means or why it is there. The formula "(28 - 3 + 2) / 2 + 1 = 14" uses the "+2" padding term that has not been explained. The student can follow the conclusion ("same output dimensions") but cannot verify the arithmetic. This violates the "parts before whole" ordering principle -- padding is being used before it is taught.
**Suggested fix:** Either (a) move the stride=2 vs pooling comparison to after the padding section (perhaps after the general formula, where the student has all the tools to verify it), or (b) use a no-padding example for the comparison: "Conv(32, 3x3, stride=2) on a 7x7 input gives (7-3)/2 + 1 = 3x3x32. Conv(32, 3x3, stride=1) on 7x7 gives 5x5x32, then MaxPool(2x2) gives... wait, 5 is not evenly divisible by 2." Option (a) is cleaner -- relocating the comparison preserves its impact while respecting prerequisite order.

#### [IMPROVEMENT] — Receptive field arithmetic "3x2 = 6" is misleadingly precise

**Location:** Lines 899-906 (the "Why does this pattern work?" section, second paragraph)
**Issue:** The main content says: "A 3x3 filter on the pooled output then covers 3x2 = 6 pixels in each dimension -- an effective 6x6 region of the original input." The calculation "3x2 = 6" implies a simple general rule (multiply filter size by pool size) that does not hold in the general case. If there was a 3x3 conv before the pool, the effective receptive field of the second conv is actually 7x7 (not 6x6), because the first conv already expanded each position's view to 3x3. The lesson's context (a conv-pool-conv pipeline) would actually produce a 7x7 receptive field, not 6x6. The "3x2 = 6" formula is correct only if pooling is applied directly to the raw input with no preceding convolution. For INTRODUCED depth (explain why, not compute exact sizes), the right intuition is more important than the right number, but presenting a specific arithmetic step ("3x2 = 6") that gives the wrong answer for the architecture being discussed creates a false anchor.
**Student impact:** Low-moderate. The student gets the right directional intuition (later layers see larger regions) but may anchor on a specific formula ("multiply by pool size") that does not generalize. If they try to compute receptive fields for real architectures, they will get incorrect answers.
**Suggested fix:** Soften the language to avoid the specific arithmetic. Replace "covers 3x2 = 6 pixels in each dimension -- an effective 6x6 region" with "effectively sees a much larger region of the original input -- roughly 6x6 -- because each pooled position already summarizes a 2x2 area." Dropping the "3x2 = 6" avoids implying a general formula while preserving the intuition.

#### [IMPROVEMENT] — Average pooling section has no aside (elevated from Iteration 1 POLISH)

**Location:** Lines 357-403 (average pooling comparison and ComparisonRow)
**Issue:** This was flagged as POLISH in Iteration 1. On fresh review, the absence of an aside here is more impactful than initially assessed. The average pooling section introduces a new concept at INTRODUCED depth (average pooling). Every other section that introduces a concept has supporting aside content. The ComparisonRow between max and average pooling is a key pedagogical moment -- the student is contrasting two approaches. An aside here could address a natural question: "When would I actually use average pooling?" Without it, the student sees max vs average as a comparison but has no guidance on when each applies. The ComparisonRow mentions "Used in some architectures (often at the end)" for average pooling, but this is vague.
**Student impact:** The student learns what average pooling is and how it differs from max pooling, but lacks guidance on when each is used in practice. This leaves a "so what?" gap for average pooling specifically -- the student may wonder why it was introduced if max pooling is the default.
**Suggested fix:** Add a TipBlock or ConceptBlock aside. Content: "Max pooling is the default for feature detection. Average pooling appears most often as global average pooling at the end of modern architectures (like ResNet), where it replaces the flatten + FC pattern entirely. For now, default to max pooling."

#### [POLISH] — Widget `idCounter` is a module-level mutable variable (carried from Iteration 1)

**Location:** CnnDimensionCalculator.tsx, line 143 (`let idCounter = 0`)
**Issue:** Same as Iteration 1. The `idCounter` variable is module-level mutable state. In React Strict Mode (development), double-renders produce different IDs. Not a functional bug since IDs are only used as keys, but not idiomatic React.
**Student impact:** None.
**Suggested fix:** Use `crypto.randomUUID()` or move the counter into a `useRef`.

#### [POLISH] — Widget `_widthOverride` prop is unused (carried from Iteration 1)

**Location:** CnnDimensionCalculator.tsx, lines 467-472
**Issue:** Same as Iteration 1. The `CnnDimensionCalculatorProps` type defines `width` and `height` but neither is used. The `width` is destructured as `_widthOverride` and ignored.
**Student impact:** None.
**Suggested fix:** Remove the unused props from the type.

#### [POLISH] — Check 3 does not have an aside

**Location:** Lines 952-1007 (Check 3: "Read an Architecture")
**Issue:** Check 3 is a substantial challenge with 4 questions including a PyTorch code block, but has no Row.Aside. Other checks in the lesson (Check 1 at line 453, Check 2 at line 769) both have helpful TipBlock asides. Check 3 is the most complex check in the lesson (reading real PyTorch code, computing dimensions through 4 layers, and reasoning about removing padding), and would benefit from a contextual tip -- for example, reminding the student that for 5x5 filters, padding=2 gives "same" output size (extending the floor(F/2) rule introduced earlier).
**Student impact:** Negligible. The check is self-contained. But the aside could reduce friction for students who get stuck.
**Suggested fix:** Add a TipBlock aside with a hint like "Remember: for a 5x5 filter, padding=2 preserves spatial dimensions (the floor(F/2) rule from earlier)."

#### [POLISH] — Stride=2 conv comparison section has no aside

**Location:** Lines 514-557 (Stride=2 conv vs separate pooling comparison)
**Issue:** This section compares two approaches for dimension reduction with concrete numbers, but has no aside. A brief aside could reinforce the tradeoff: stride=2 convolutions are more parameter-efficient (one layer instead of two) but max pooling provides explicit invariance to small translations.
**Student impact:** Negligible. The main content explains the comparison clearly.
**Suggested fix:** Add a brief TipBlock or ConceptBlock aside about the tradeoff, or leave as-is since the comparison section is short.

#### [POLISH] — The `as` cast in pool type select

**Location:** CnnDimensionCalculator.tsx, line 294 (`e.target.value as 'max' | 'avg'`)
**Issue:** The pool type select uses a type assertion (`as 'max' | 'avg'`). Since the only options in the select are "max" and "avg", this is safe at runtime, but type assertions are generally discouraged. A safer pattern would be to validate the value before using it.
**Student impact:** None. Code style issue only.
**Suggested fix:** Create a helper function that validates and returns the pool type, or accept this as a common React pattern for select elements with known values.

### Review Notes

**What was fixed from Iteration 1:**
- **CRITICAL fix (pooling-after-every-conv misconception):** Now addressed with a full Conv-Conv-Conv-Pool dimension-tracking example at lines 1043-1085 using the same StageRow component as the main architecture example. The student sees a concrete alternative architecture with traced dimensions, proving that non-alternating patterns work. This is a strong fix -- it shows rather than tells.
- **Stride=2 vs pooling comparison:** Now present as a concrete side-by-side at lines 514-557 with actual numbers on a 28x28 input. Both options yield 14x14x32. Clear and effective.
- **Widget proportional rectangles:** The PipelineVisualization now uses `scaleDimension()` with sqrt scaling, making blocks physically taller for larger spatial dimensions and wider for more channels. Flat layers are tall and narrow. This delivers the planned geometric/spatial modality.
- **Receptive field in main content:** The concrete 3x3-after-pool = 6x6 calculation is now in the main content paragraph (lines 899-906) rather than only in the aside.
- **Combined stride+padding formula verification:** A fourth example "stride=2, padding=1: (28 - 3 + 2) / 2 + 1 = 14" is now at line 669.
- **Check 1 grid size:** Now uses a 4x4 grid (same as the worked example) instead of a 6x6 grid, and question 2 explicitly identifies "the top-right 2x2 window."

**What works well (reconfirmed):**
- The hook is compelling: 21,632 values from 784 pixels creates genuine surprise and need.
- Max pooling worked example is thorough (4x4 input, 4 explicit windows, before/after edge map).
- The general formula is presented after concrete examples and verified against them.
- The dimension-tracking centerpiece (28x28x1 through to 10) with StageRow is readable and color-coded.
- The widget is now both functionally complete and visually informative with proportional blocks.
- Connection to prior knowledge is strong throughout (explicit references to "What Convolutions Compute").
- All Row layout patterns are correct. No manual flex layouts.
- No `switch` statements, no `any` types, no `else if`/`else`. Em dashes have no surrounding spaces.
- Scope boundaries are respected; lesson does not drift into training or accuracy comparisons.
- The Conv-Conv-Conv-Pool alternative architecture (new in this iteration) effectively addresses the misconception about mandatory pooling after every conv.

**New concern identified:**
The stride=2 comparison using padding=1 before padding is taught is the most significant new finding. It is an ordering violation (using a concept before it is introduced) that could confuse a sequential reader. The fix is straightforward -- relocate the comparison to after the padding section or the general formula section.

**Recommendation:** Fix the three IMPROVEMENT items: (1) move the stride=2 vs pooling comparison to after the padding section, (2) soften the receptive field "3x2 = 6" arithmetic to an approximation, (3) add an aside to the average pooling section. Polish items can be fixed opportunistically. One more review pass after.

---

## Review — 2026-02-09 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 3

### Verdict: PASS

All three improvement findings from Iteration 2 have been correctly fixed. The stride=2 vs pooling comparison is now placed after the general formula section (line 654), so padding is fully taught before the comparison uses it. The receptive field arithmetic now uses "roughly 6x6 region" with a qualitative explanation ("each of the 9 pooled positions it reads covers a 2x2 patch") rather than the misleading "3x2 = 6" formula. The average pooling section now has a TipBlock aside ("When to Use Which") that contextualizes when each pooling type is used. No critical or improvement-level findings remain. Three polish items carried from prior iterations remain.

### Findings

#### [POLISH] — Widget `idCounter` is a module-level mutable variable (carried from Iterations 1 and 2)

**Location:** CnnDimensionCalculator.tsx, line 143 (`let idCounter = 0`)
**Issue:** The `idCounter` variable is module-level mutable state. In React Strict Mode (development), double-renders produce different IDs. If the component is unmounted and remounted, IDs continue incrementing from where they left off. Not a functional bug since IDs are only used as React keys and internal identifiers.
**Student impact:** None. Code hygiene issue.
**Suggested fix:** Use `crypto.randomUUID()` or move the counter into a `useRef`.

#### [POLISH] — Widget `_widthOverride` prop is unused (carried from Iterations 1 and 2)

**Location:** CnnDimensionCalculator.tsx, lines 467-472
**Issue:** The `CnnDimensionCalculatorProps` type defines `width` and `height` but neither is used. The `width` is destructured as `_widthOverride` and ignored; `height` is not referenced.
**Student impact:** None. Code cleanliness issue.
**Suggested fix:** Remove the unused props from the type definition.

#### [POLISH] — Check 3 does not have an aside (carried from Iteration 2)

**Location:** Lines 961-1017 (Check 3: "Read an Architecture")
**Issue:** Check 3 is the most complex check in the lesson (reading PyTorch code, computing through 4 layers, reasoning about removing padding), but has no Row.Aside. Other checks in the lesson both have helpful TipBlock asides.
**Student impact:** Negligible. The check is self-contained.
**Suggested fix:** Add a TipBlock aside with a hint like "Remember: for a 5x5 filter, padding=2 preserves spatial dimensions."

### Review Notes

**Iteration 2 fixes verified:**
- **Stride=2 comparison ordering:** The comparison now appears at line 654, after the general formula (which is at line 594). The padding section (starting at line 523) is well before both. The student encounters padding -> formula -> stride=2 comparison, which correctly teaches parts before using them together. The formula verification examples now include "stride=2, padding=1: (28 - 3 + 2) / 2 + 1 = 14" as the fourth example (line 634), so the combined-parameter case is exercised before the comparison uses it. Clean fix.
- **Receptive field arithmetic:** The main content (lines 909-916) now reads: "A 3x3 filter on the pooled output therefore responds to a roughly 6x6 region of the original input -- each of the 9 pooled positions it reads covers a 2x2 patch." The qualitative explanation ("each pooled position covers a 2x2 patch") gives the student an intuitive grounding without implying a general multiplicative formula. This correctly conveys directional understanding at the INTRODUCED depth level.
- **Average pooling aside:** The TipBlock at lines 404-410 ("When to Use Which") provides clear guidance: max pooling is the default, average pooling appears as global average pooling at the end of modern architectures like ResNet. This answers the natural "when would I use this?" question and gives the student practical guidance at the INTRODUCED depth level.

**No new issues introduced by the fixes.** The relocation of the stride=2 comparison fits naturally after the formula section -- the transition "In fact, a stride=2 convolution can replace a separate pooling layer entirely" (line 659) flows logically from the formula verification. The average pooling aside content is consistent with the ComparisonRow text. The receptive field rewording reads naturally in context.

**Overall lesson quality:**
- **Pedagogically complete.** All planned examples, misconceptions, modalities, and scope boundaries are present and correctly implemented.
- **Well-ordered.** Concrete before abstract throughout. Problem before solution for both pooling (feature maps are too big) and padding (shrinking problem). Parts before whole (pooling, stride, padding taught individually before the architecture combines them).
- **Good cognitive load.** As a BUILD lesson following a STRETCH lesson, the new concepts (pooling, stride, padding) are mechanically simple extensions of Lesson 1's foundation. The lesson correctly treats them as engineering decisions, not new abstractions.
- **Strong interactive component.** The CnnDimensionCalculator with proportional pipeline visualization, real-time shape computation, error detection, and guided experiments in the TryThisBlock delivers both the geometric/spatial and interactive modalities effectively.
- **Code conventions fully respected.** No switch statements, no `any` types, no `else if`/`else`, no spaced em dashes. All Row layout patterns correct. TypeScript compiles with no errors, lint passes.

This lesson is ready to ship. Proceed to Phase 5 (Record) to update the module record with the building-a-cnn summary.
