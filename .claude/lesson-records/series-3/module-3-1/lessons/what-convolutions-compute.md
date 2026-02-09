# Lesson Plan: What Convolutions Compute

**Series:** 3 (CNNs) | **Module:** 3.1 (Convolutions) | **Lesson:** 1 of 3
**Slug:** `what-convolutions-compute`
**Cognitive load type:** STRETCH

---

## Phase 1: Orient — Student State

This is the first lesson in Series 3. The student has completed Series 1 (Foundations, 17 lessons) and is assumed to have completed Series 2 (PyTorch, 10 lessons). They have deep conceptual understanding of neural networks and practical PyTorch fluency.

### Relevant Concepts the Student Has

| Concept | Depth | Source | How It's Relevant Here |
|---------|-------|--------|----------------------|
| Neuron = weighted sum + bias | DEVELOPED | module-1-2 neuron-basics | Convolution IS a weighted sum over a local region — same operation, different connectivity |
| Weights as learnable parameters | DEVELOPED | module-1-1 linear-regression | Filters are weights that the network learns, not hand-designed |
| Layers and networks | INTRODUCED | module-1-2 neuron-basics | CNNs are networks with a specific layer type; builds on existing "stacking layers" mental model |
| Training loop (forward-loss-backward-update) | DEVELOPED | module-1-1 implementing-linear-regression | Same training loop applies; nothing changes about HOW the network learns |
| Backpropagation | DEVELOPED | module-1-3 backpropagation | Gradients flow through conv layers just like dense layers; no new training algorithm |
| Space transformation (hidden layers move points) | DEVELOPED | module-1-2 activation-functions-deep-dive | Feature maps ARE space transformations — each filter creates a new representation |
| `nn.Module`, `forward()`, parameters | DEVELOPED (assumed) | series-2 nn-module | Student can read and write PyTorch module code |
| `loss.backward()`, optimizer.step() | DEVELOPED (assumed) | series-2 autograd, training-loop | Student has trained models in PyTorch |
| MNIST dense network | APPLIED (assumed) | series-2 mnist-project | Student built a dense network for MNIST — the baseline this series will beat |

### Mental Models Already Established

- "Parameters are knobs the model learns" -- directly applicable to filter weights
- "Networks transform space" -- feature maps are a spatial transformation
- "ML is function approximation" -- CNNs approximate functions that have spatial structure
- "The complete training recipe" -- still applies; only the architecture changes

### What Was NOT Covered

- Any notion of spatial structure or locality in data
- 2D operations (all prior work used 1D or flat vectors)
- The idea that architecture encodes assumptions about data
- Why treating an image as a flat vector loses information
- Convolutions, filters, kernels, feature maps (entirely new vocabulary)

### Readiness Assessment

The student is well-prepared. They understand weighted sums, learnable parameters, and PyTorch. The key bridge is: a dense layer computes a weighted sum over ALL inputs, a convolutional layer computes a weighted sum over a LOCAL neighborhood and slides that computation across the input. Every piece of the convolution operation maps to something they already know — the novelty is the spatial constraint and weight sharing.

**Activation energy concern:** This is the first lesson in a new series. The student finished Foundations and PyTorch. There may be a gap since their last session. The opening needs to reconnect to familiar ground quickly and make the new material feel like a natural next step, not a jarring topic switch.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to compute what a convolutional filter produces when applied to a 2D grid, and explain why this spatial operation detects features that dense layers cannot.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Weighted sum (w*x + b) | DEVELOPED | DEVELOPED | neuron-basics (1.2) | OK | Convolution is a weighted sum over a local patch; student must be able to compute these fluently |
| Learnable parameters | DEVELOPED | DEVELOPED | linear-regression (1.1) | OK | Filter values are learned via backprop, not hand-designed; student already knows parameters are learned |
| Layers stacking | INTRODUCED | INTRODUCED | neuron-basics (1.2) | OK | Need to understand that conv layers stack; existing "layers" mental model sufficient |
| PyTorch nn.Module | DEVELOPED | DEVELOPED (assumed) | nn-module (2.1) | OK | Will show nn.Conv2d; student should be comfortable with module API |
| 2D arrays / grids | INTRODUCED | NOT EXPLICIT | (general programming) | OK | Student is a software engineer; comfortable with 2D arrays; no teaching needed |
| Matrix multiplication | INTRODUCED | INTRODUCED | (implicit in 1.2) | OK | Not required for this lesson — convolutions are introduced via element-wise multiply + sum, not matmul |

All prerequisites are met. No gaps require dedicated sections.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Convolution is just a smaller dense layer" | Both compute weighted sums; the student's only frame for weighted sums is dense neurons | Show parameter counts: a 3x3 filter has 9 parameters regardless of image size; a dense layer connecting a 224x224 image has 50,176 weights PER neuron. Also: same filter everywhere means it can detect a vertical edge at ANY position, unlike a dense layer that must learn "vertical edge at position (10,10)" separately from "vertical edge at position (50,50)." | During the weight sharing section, after establishing what filters compute |
| "The filter values need to be chosen carefully by hand" | The lesson teaches edge detection with specific hand-crafted filter values like [-1, 0, 1] | State explicitly: these hand-crafted filters are for building intuition. In a real CNN, the network learns filter values via backprop — just like it learned weights in dense networks. The learned filters often end up looking like edge detectors, but the network discovers this on its own. | Immediately after the hand-computed edge detection examples |
| "Convolution changes the fundamental training process" | New operation might seem to require a new learning algorithm | The training loop is identical: forward pass (now includes convolutions), compute loss, backward pass (backprop through conv layers), update parameters. Nothing about training changes — only the architecture. | In the "what stays the same" framing early in the lesson |
| "Each pixel in the output is influenced by the entire input" | Dense layers connect every input to every output; student might assume convolutions work similarly | Show the receptive field: a 3x3 filter at position (5,5) only sees the 9 pixels in its 3x3 neighborhood. The output pixel at (5,5) knows nothing about pixel (0,0). This locality is the whole point. | During the filter sliding computation, emphasize what each output "sees" |
| "Bigger filters are always better because they see more context" | Intuition that more information should help | A 28x28 filter on a 28x28 image IS a dense layer — it sees everything. The power of small filters comes from locality (nearby pixels are more related than distant ones) and composition (stack two 3x3 layers to get a 5x5 receptive field with fewer parameters and nonlinearity between them). | Scope boundary / teaser for Lesson 2 and Module 3.2 |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| 5x5 grid with a vertical stripe, 3x3 vertical edge filter | Positive | First concrete computation: multiply-and-sum on a tiny grid. Shows the filter "lighting up" where the edge is. | Small enough to compute by hand (9 multiplications), large enough to slide across multiple positions. Vertical edge is visually intuitive. |
| Same 5x5 grid, horizontal edge filter | Positive | Shows a DIFFERENT filter produces a DIFFERENT feature map from the same input. Confirms generalization: filters detect different things. | Same input, different filter, different output — proves the filter determines what is detected, not the image. |
| Same 5x5 grid, vertical edge filter, but image has NO vertical edges (uniform or horizontal stripe) | Negative | Shows what happens when the feature is absent: the feature map is all zeros (or near-zero). The filter is a detector that can return "not found." | Demonstrates that filters are selective: they respond strongly to their target pattern and weakly to everything else. |
| Real photograph edge detection (visual only, not hand-computed) | Positive (stretch) | Shows the same operation at scale: a vertical edge filter applied to a real photo produces a feature map highlighting vertical edges everywhere. Bridges from toy example to real use. | Motivates why this toy operation matters at scale. The student sees edge detection is not an academic exercise — it is what the first layer of every CNN does. |

---

## Phase 3: Design

### Narrative Arc

You have spent two series learning how neural networks work: how they compute predictions, how they learn from errors, how to train them in PyTorch. Every network you have built so far treats its input as a flat list of numbers. For MNIST, you flattened a 28x28 image into a 784-length vector and fed it to a dense layer. This works — you got decent accuracy. But think about what that flattening destroys. In the image, pixel (0,0) is next to pixel (0,1) — they are part of the same local region, maybe part of the same stroke. In the flat vector, pixel (0,0) is at index 0 and pixel (0,1) is at index 1, but pixel (1,0) is at index 28. The dense layer has no idea which pixels are neighbors. It treats a pixel at the top-left the same as a pixel at the bottom-right. And worse, every dense neuron connects to every pixel, so a pattern the network learns to detect at one location tells it nothing about the same pattern at another location. Convolutions solve both problems with one elegant idea: instead of connecting to all pixels, use a small filter that slides across the image, computing the same weighted sum at every position. This lesson is about understanding exactly what that operation computes, and why it is so powerful for data with spatial structure.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | 5x5 grid with numbers, 3x3 filter, hand-computed output step by step | Convolutions are mechanical: multiply-and-sum. Students must compute one by hand to internalize the operation. Abstract description is insufficient. |
| Visual | Animated/interactive filter sliding across a grid, highlighting the active region and producing the output | The sliding window is inherently spatial and sequential. Static text cannot convey "the filter moves." Animation/interaction makes the spatial nature tangible. |
| Symbolic | The convolution formula: output(i,j) = sum over (m,n) of filter(m,n) * input(i+m, j+n) | Connects the hand-computed example to general notation. Student needs this to read documentation and papers. |
| Intuitive/Analogy | "The filter is a pattern detector — it asks 'does this local region look like my pattern?' at every location" | Gives the student a one-sentence mental model for what convolutions DO, beyond the mechanics of how they compute. |
| Geometric/Spatial | Side-by-side: input image, filter overlay, output feature map. The feature map is a "heat map" of where the pattern was found. | Feature maps are spatial — their values have positions that correspond to input positions. This spatial correspondence is the key insight. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3 at DEVELOPED (convolution as sliding filter, feature maps, edge detection filters) + 3 at INTRODUCED (spatial structure/locality, weight sharing, multiple filters). This is at the upper bound but acceptable for a STRETCH lesson that opens a new series — the INTRODUCED concepts are lightweight (they are "why" framing for the DEVELOPED "how" concepts).
- **Previous lesson load:** N/A (first lesson in series; last Foundations lesson was CONSOLIDATE — overfitting-and-regularization). If Series 2 is complete, the last lesson was a CONSOLIDATE project. Either way, the student is coming off low cognitive load.
- **Appropriate:** Yes. STRETCH after CONSOLIDATE is ideal. The student has rested from high-load material and is ready for something genuinely new.

### Connections to Prior Concepts

| New Concept | Prior Concept | Connection |
|-------------|--------------|------------|
| Convolution (weighted sum over local region) | Neuron = weighted sum + bias | "A conv filter does the same thing a neuron does — multiply inputs by weights and sum — but only over a small neighborhood instead of all inputs" |
| Filter weights are learned | Parameters are learnable knobs | "These filter values are not hand-designed. The network learns them via backprop, just like it learned w and b in your linear regression" |
| Feature map | Space transformation | "Each filter creates a new representation of the input — a feature map. This is the same idea as hidden layers 'transforming space,' but now the transformation preserves spatial layout" |
| Spatial locality | Polling analogy (batching-and-sgd) | Lightweight connection: "Just as a random sample of data points estimates the full gradient, a small patch of pixels captures the local pattern. You don't need to see the whole image to detect an edge." |

**Potentially misleading prior analogies:** The "networks transform space" analogy from Module 1.2 involved rearranging points in abstract feature space. Feature maps are also transformations but they preserve spatial layout (the output is still a 2D grid with spatial meaning). Need to note this distinction explicitly.

### Scope Boundaries

**This lesson IS about:**
- What the convolution operation computes (mechanically)
- What filters detect (edge detection as primary example)
- Why spatial locality and weight sharing matter
- Feature maps as the output of convolution

**This lesson is NOT about:**
- Pooling, stride, or padding (Lesson 2)
- How to build a full CNN architecture (Lesson 2)
- Training a CNN or comparing to dense networks (Lesson 3)
- `nn.Conv2d` in depth (mentioned/shown briefly, developed in Lesson 2)
- Backprop through conv layers (relies on existing backprop knowledge; no special treatment needed)
- 1D or 3D convolutions
- Multiple input channels (e.g., RGB) — planted briefly, developed in Lesson 2

**Target depth:** Convolution operation at DEVELOPED (student can compute it and explain what it does). Feature maps at DEVELOPED. Edge detection at DEVELOPED. Weight sharing and spatial locality at INTRODUCED (student can explain why they matter but hasn't practiced with them).

### Lesson Outline

**1. Context + Constraints**
What this lesson is about (convolutions: what they compute and why). What we are NOT doing (building CNNs, training, pooling — that is Lessons 2 and 3). Estimated time. Connection to the series arc: "This module gives you the building blocks of convolutional networks. Today: the core operation. Next: assembling it into architecture. Then: building one that works."

**2. Hook — The Flat Vector Problem (before/after + puzzle)**
Show the MNIST "8" image as a 2D grid — the student recognizes it. Now show it flattened to a 784-length vector — it is unrecognizable noise. Ask: "Your dense network from Series 2 saw THIS (the flat vector). It never saw the 2D image. And yet it got ~97% accuracy. What if we could build a network that actually sees spatial structure? Could it do better?" This reconnects to their PyTorch MNIST project and creates the motivation for the entire series.

**3. Explain: The Convolution Operation**
- Start with a 5x5 grid of numbers (a tiny "image")
- Introduce a 3x3 filter (just 9 numbers arranged in a grid)
- Walk through the computation: overlay the filter on the top-left 3x3 region, multiply element-wise, sum. That sum is one output value.
- Slide the filter one position right. Repeat. Continue until the filter has visited every valid position.
- The result: a 3x3 output grid (the feature map).
- Interactive widget: ConvolutionExplorer. The student sees the filter slide, the active region highlighted, the multiplication happening, and the output being filled in one cell at a time.
- Symbolic: write the formula output(i,j) = sum of filter(m,n) * input(i+m, j+n)
- Key callout: this is a weighted sum — the same operation as a neuron, but LOCAL.

**4. Check 1 — Predict and Verify**
Give a 4x4 input and a 2x2 filter. Ask: "What is the output at position (0,0)? What is the size of the full output?" Student predicts, then reveal the answer. (Tests: can they do the multiply-sum? Do they understand output size shrinks?)

**5. Explain: What Filters Detect — Edge Detection**
- Show a 5x5 image with a clear vertical edge (left half bright, right half dark)
- Apply a vertical edge filter: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
- Walk through 2-3 positions: away from the edge (output near 0), right on the edge (output large). The feature map "lights up" at the edge.
- Apply a horizontal edge filter to the same image: output near zero everywhere (no horizontal edges). The filter is selective.
- Negative example: apply the vertical edge filter to a uniform (constant) image. Output is exactly 0 everywhere. The detector says "nothing here."
- Stretch example (visual only): show a real photograph and its vertical-edge feature map side by side. "This is what the first layer of a CNN sees."
- Mental model: "A filter is a pattern detector. It asks 'does this neighborhood look like my pattern?' at every position. The feature map is the answer at every position."

**6. Check 2 — Spot the Pattern**
Show three small feature maps (outputs of different filters on the same input). Ask: "Which filter produced each feature map?" (One highlights vertical edges, one horizontal edges, one responds to uniform regions.) Tests whether the student understands the filter-to-feature relationship.

**7. Explain: Why This Beats Dense — Weight Sharing and Locality**
- **Locality:** A dense neuron connecting to a 28x28 image has 784 weights. A 3x3 conv filter has 9 weights. The filter only looks at neighbors — which is where the useful information is for images.
- **Weight sharing:** The same 9 weights are used at every position. If the filter learns to detect a vertical edge, it detects vertical edges EVERYWHERE. A dense network would need to learn "vertical edge at (10,10)" and "vertical edge at (50,50)" separately.
- **Parameter count:** A dense layer for 28x28 with 32 neurons: 784 * 32 = 25,088 parameters. A conv layer with 32 filters of size 3x3: 32 * 9 = 288 parameters. Orders of magnitude fewer.
- Misconception address: "A convolution is NOT just a smaller dense layer. It is the same filter applied everywhere. This is weight sharing — a single set of weights used at every spatial position."

**8. Explain: Multiple Filters = Multiple Features**
- One filter detects vertical edges. Another detects horizontal edges. A third might detect diagonal edges.
- Each filter produces its own feature map. Stack them: the output of a conv layer is a STACK of feature maps, one per filter.
- "The number of filters is how many questions the layer asks about each neighborhood."
- Briefly mention: this is analogous to having multiple neurons in a dense layer. More filters = richer representation.
- Plant for Lesson 2: "We will see how to control the number of filters and how to stack conv layers in the next lesson."

**9. Explore — Interactive Widget: ConvolutionExplorer**
The main interactive element. Features:
- Editable input grid (small, e.g., 6x6 or 7x7, with preset patterns: vertical edge, horizontal edge, diagonal, corner, uniform)
- Filter selector: preset filters (vertical edge, horizontal edge, diagonal, blur/average, sharpen) OR custom editable filter
- Step-by-step mode: filter moves one position at a time, showing multiplication and sum
- Auto-play mode: animates the sliding window
- Output feature map fills in as the filter slides
- Highlight: active input region, filter overlay, current output cell
- Color scale: output values shown as heatmap (positive = bright, negative = dark, zero = neutral)

**10. Elaborate: What the Network Actually Learns**
- "We hand-designed these filters for edge detection. In a real CNN, the network LEARNS filter values via backprop — the same backward pass you already know."
- "Researchers have visualized the first-layer filters of trained CNNs. They typically learn edge detectors, color detectors, and texture detectors — but they discover these patterns on their own."
- "Later layers combine first-layer features into more complex patterns: edges become corners become shapes become objects."
- This is INTRODUCED depth — plant the seed for Module 3.3 (Seeing What CNNs See).

**11. Summarize**
Key takeaways:
- A convolution slides a small filter across the input, computing a weighted sum at each position
- The output (feature map) shows where the filter's pattern was detected
- Weight sharing means the same filter works at every position — fewer parameters, position-invariant detection
- Multiple filters detect multiple features; stack of feature maps = layer output
- Filter values are LEARNED, not hand-designed

Echo the mental model: "A convolutional layer asks a fixed set of questions ('is there a vertical edge here? a horizontal edge? a diagonal?') at every location in the image. The answers, arranged spatially, are the feature maps."

**12. Next Step**
"You now know what a single convolution computes. Next: how to assemble convolutions into a full CNN architecture — with pooling to shrink spatial dimensions, stride and padding to control sizes, and the classic conv-pool-fc pattern that powers everything from LeNet to modern networks."

### Widget Specification: ConvolutionExplorer

**Purpose:** Let the student see and interact with the convolution operation — watch the filter slide, see each multiply-and-sum, understand the output feature map.

**Key interactions:**
- Select input pattern (presets + editable grid)
- Select filter (presets + editable)
- Step through positions one at a time OR auto-animate
- See the multiplication and sum at each position
- See the full feature map build up

**NOT in scope for this widget:**
- Padding or stride (Lesson 2)
- Multiple filters simultaneously (show one at a time)
- Training or backprop through the conv layer
- Real images (keep it to small numeric grids for clarity)

**Visualization library:** Likely a custom component with CSS grid. The grid sizes are small (5x5 to 7x7 input, 3x3 filter). No charting library needed. Color-coded cells with hover details.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 3 DEVELOPED + 3 INTRODUCED concepts (within bounds for STRETCH)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 2
- Improvement: 5
- Polish: 3

### Verdict: MAJOR REVISION

Two critical findings must be addressed before this lesson is usable: the interactive widget is placed too late in the lesson (after all explanation is complete), and a planned example (real photograph edge detection) was dropped without replacement.

### Findings

#### [CRITICAL] — Interactive widget placement defeats its pedagogical purpose

**Location:** Section 9 (ConvolutionExplorer), lines 556-591
**Issue:** The ConvolutionExplorer widget is placed AFTER all conceptual explanation is complete (sections 3-8). The plan specified it should appear during Section 3 ("Explain: The Convolution Operation") to let the student see the sliding computation AS the concept is being taught. Instead, the lesson explains the full convolution operation textually (Section 3), does a comprehension check (Section 4), explains edge detection (Section 5), does another check (Section 6), explains weight sharing and locality (Section 7), explains multiple filters (Section 8), and THEN finally shows the widget (Section 9). By the time the student reaches the widget, they have already read ~500 words of explanation about what the operation does. The widget becomes a "nice to have" recap rather than the primary learning tool for the mechanical operation.
**Student impact:** The student must build their mental model of the sliding operation entirely from text and a static formula before they can see it animated. This is a modality gap during the critical "how does this work" section. Students who struggle with spatial reasoning from text alone will be lost well before the widget appears.
**Suggested fix:** Move the ConvolutionExplorer to immediately after the step-by-step algorithm description in Section 3 (after line 212, before the formula). The student reads the steps, then immediately sees them in action. The formula can follow the widget. Keep the "Try This" experiments in the aside. The later sections on edge detection and weight sharing can reference the widget with prompts like "go back to the explorer and try..." or the widget could appear a second time.

#### [CRITICAL] — Planned real-photograph example dropped without replacement

**Location:** Missing from the lesson entirely
**Issue:** The plan specified Example 4: "Real photograph edge detection (visual only, not hand-computed) — Shows the same operation at scale." This example was categorized as "Positive (stretch)" and its stated purpose was to bridge from the toy 7x7 grid to real use, so the student sees edge detection is not academic. The built lesson contains zero real-world visual examples. The only visual grounding is the 7x7 numeric grids in the widget and the inline 3x3 filter display.
**Student impact:** Without seeing the operation at real scale, the student may think convolutions are a toy operation on tiny grids. The "so what?" gap remains open. The plan explicitly identified this as the bridge to "why this matters"—dropping it leaves the lesson feeling abstract and disconnected from the real CNNs the student will build.
**Suggested fix:** Add a static visual (image or illustration) after Section 5 showing a photograph alongside its vertical-edge feature map. This does not need to be interactive—a side-by-side static image is sufficient. Even a description with a placeholder for a future image would be better than nothing. Alternatively, add a note in the lesson text describing what a real edge-detection output looks like, with a concrete "imagine applying this to a photo of a building" description.

#### [IMPROVEMENT] — Misconception 4 ("each pixel influenced by entire input") not explicitly addressed

**Location:** Sections 3 and 7
**Issue:** The plan identified misconception 4: "Each pixel in the output is influenced by the entire input." The plan specified addressing it "during the filter sliding computation, emphasize what each output 'sees.'" The built lesson shows the computation and mentions locality, but never explicitly states: "The output pixel at position (i,j) knows NOTHING about distant pixels—it only sees its 3x3 neighborhood." The receptive field concept is implied but never made concrete with a negative example (e.g., "the output at (0,0) has no information about input pixel (6,6)").
**Student impact:** Students coming from dense layers may still assume the output has some awareness of the full input. Without an explicit statement that each output position has a limited "field of view," this misconception can persist.
**Suggested fix:** Add 1-2 sentences in Section 3 or in the widget aside that explicitly states what each output pixel cannot see. Example: "The output at position (0,0) was computed from ONLY the 9 input pixels in the top-left 3x3 region. It knows nothing about the bottom-right corner. This locality is the entire point."

#### [IMPROVEMENT] — Misconception 5 ("bigger filters are always better") not addressed at all

**Location:** Missing from lesson
**Issue:** The plan identified misconception 5 and marked it as a "scope boundary / teaser for Lesson 2." However, the built lesson never addresses it—not even as a planted question. The student could leave thinking "why not just use a bigger filter to see more?" without any acknowledgment that this is a valid question with a non-obvious answer.
**Student impact:** Low impact for this lesson specifically, but it is a missed opportunity to plant curiosity for Lesson 2 and to prevent a misconception from solidifying.
**Suggested fix:** Add a brief aside or a sentence in the "Why This Beats Dense Layers" section: "You might wonder: why not use a bigger filter? A 28x28 filter on a 28x28 image IS a dense layer. Small filters + stacking is better—we will see why in the next lesson."

#### [IMPROVEMENT] — No worked example with specific numbers shown in lesson text

**Location:** Sections 3-5
**Issue:** The plan specified "5x5 grid with a vertical stripe, 3x3 vertical edge filter" as Example 1, to be "hand-computed" with the student walking through specific multiply-and-sum operations. The built lesson describes the algorithm as steps (Section 3, lines 193-207) and defers the actual computation to the widget. There is no inline worked example where the text shows specific numbers: "Input region is [[0,0,1],[0,0,1],[0,0,1]], filter is [[-1,0,1],[-1,0,1],[-1,0,1]], multiply element-wise: 0*-1 + 0*0 + 1*1 + ... = 3." The widget's ComputationDetail panel does show this, but it requires interaction to see.
**Student impact:** Students who prefer reading through examples before interacting will not get a grounded numerical example. The step-by-step list in Section 3 is procedural but abstract—no actual numbers appear until the widget or the check in Section 4 (which is a test, not a teaching example). This violates the "concrete before abstract" ordering rule—the abstract algorithm is given first, then the student is expected to apply it in a check.
**Suggested fix:** Add a concrete 3x3 or 4x4 worked example with specific numbers inline in Section 3, BEFORE the formula. Show the input patch, the filter, the element-wise products, and the sum. Then say "the widget below lets you see this for every position."

#### [IMPROVEMENT] — Check 2 ("Spot the Pattern") uses blur filter without teaching it

**Location:** Section 6, lines 387-430
**Issue:** The check presents three feature maps and asks the student to identify which filter produced each. The answer for feature map C is "blur filter (averages all neighbors, no edge selectivity)." However, the lesson text never introduces or explains the blur/averaging filter. The student has only been taught edge detection filters (vertical and horizontal). The blur filter appears in the widget presets but is not explained in the lesson prose. The student is being tested on a concept they were not taught.
**Student impact:** The student may guess "C" correctly by elimination (A is vertical, B is horizontal, so C must be the remaining one), but they will not understand WHY the blur filter produces uniform output. This is a fairness issue in the check—it tests untaught material.
**Suggested fix:** Either (a) replace the blur filter answer with something the student has been taught (e.g., diagonal edge filter), or (b) add a brief explanation of the blur/averaging filter before this check: "A filter with all 1s averages the neighborhood—it smooths the image rather than detecting edges."

#### [IMPROVEMENT] — Feature map spatial correspondence not made explicit

**Location:** Sections 3 and 5
**Issue:** The plan specified a geometric/spatial modality: "Side-by-side: input image, filter overlay, output feature map. The feature map is a 'heat map' of where the pattern was found." The plan also noted: "Feature maps are spatial—their values have positions that correspond to input positions. This spatial correspondence is the key insight." The built lesson mentions that the feature map shows "where the pattern was detected" but never explicitly states the spatial correspondence: output position (i,j) corresponds to the input region starting at (i,j). The widget shows this implicitly through highlighting, but the lesson text does not articulate it.
**Student impact:** The student may not realize that the feature map is spatially aligned with the input—that output position (2,3) tells you about the input region centered around (2,3). This is important for understanding how later CNN layers compose features spatially.
**Suggested fix:** Add a sentence or callout after the formula or after the widget: "Each position in the feature map corresponds to the same position in the input. Output(2,3) tells you whether the pattern was present in the input region starting at row 2, column 3. The feature map is a spatial 'answer key.'"

#### [POLISH] — Output size formula given in aside, not in main content

**Location:** Section 3, line 241-247 (TipBlock in aside)
**Issue:** The output size formula (N - F + 1) is presented only in the aside. The main content mentions "A 7x7 input with a 3x3 filter produces a 5x5 feature map" but does not give the general formula. The check in Section 4 tests output size (4x4 input, 2x2 filter), but the student may not have read the aside. Then the aside in Section 4 gives the formula AGAIN. This is fine pedagogically but the formula appearing only in asides means a student who skips asides will be tested on something they never saw in the main content.
**Student impact:** Minimal—most students will read asides. But the formula is important enough to deserve a sentence in the main content.
**Suggested fix:** Add one sentence in the main content after "A 7x7 input with a 3x3 filter produces a 5x5 feature map": "In general, a filter of size F on an input of size N produces an output of size N - F + 1."

#### [POLISH] — Widget placed inside ExercisePanel may confuse purpose

**Location:** Section 9, lines 562-567
**Issue:** The ConvolutionExplorer is wrapped in an ExercisePanel component. But this widget is not an exercise—it is an exploration/demonstration tool. The ExercisePanel framing may set the wrong expectation (the student might think they need to "solve" something rather than explore).
**Student impact:** Minor confusion about whether they need to produce an answer or just experiment.
**Suggested fix:** If there is a more appropriate wrapper component (e.g., a generic panel without "Exercise" semantics), use that instead. If ExercisePanel is the standard wrapper for all widgets, this is fine—just ensure the subtitle makes it clear this is exploration, not a graded exercise. The current subtitle "Select different inputs and filters, then step through or animate" is adequate.

#### [POLISH] — Aside text "What You Already Know" lists concepts vaguely

**Location:** Section 1, lines 74-78 (TipBlock)
**Issue:** The aside says "You understand weighted sums, learnable parameters, and PyTorch." This is very generic. It could more precisely connect to the specific prior knowledge that will be relevant: "You have built dense networks where each neuron connects to every input and computes a weighted sum. Convolutions use the same weighted sum—but only over a small local region."
**Student impact:** Negligible. The aside is functional but misses an opportunity to prime the connection between dense layers and convolutions before the lesson begins.
**Suggested fix:** Make the aside text more specific: reference the dense layer + weighted sum connection that will be the bridge to convolutions.

### Review Notes

**What works well:**
- The hook ("The Flat Vector Problem") is strong. Starting from the student's MNIST experience and showing what flattening destroys is an effective motivation. Problem before solution is followed correctly.
- The ComparisonRow between dense and convolutional layers is clear and well-structured.
- The ConvolutionExplorer widget is well-implemented: step-by-step mode, auto-play, computation detail panel, preset inputs and filters, color coding. The interaction design is solid (cursor-pointer on buttons, clear controls).
- Em dash formatting is consistent (no spaces).
- Scope boundaries are clearly stated and respected—the lesson does not drift into pooling, stride, or padding.
- The connection to prior knowledge ("same weighted sum as a neuron, but local") is made explicitly and repeatedly.
- The summary section effectively echoes the key mental models.
- The Row layout pattern is consistently used throughout.

**Structural pattern:** The two critical findings share a root cause—the builder appears to have followed the outline section numbers literally but reordered when the widget should appear. The plan put the widget in Section 3 (during explanation) AND Section 9 (as a standalone exploration section). The built lesson collapsed these into one Section 9 placement, which breaks the pedagogical flow.

**Recommendation for revision:** Fix the two CRITICAL items first (widget placement and real-photograph example). Then address the IMPROVEMENT items. Re-review after fixes.

---

## Review — 2026-02-09 (Iteration 2/3)

### Iteration 1 Resolution Status

| # | Finding | Severity | Status | Notes |
|---|---------|----------|--------|-------|
| 1 | Widget placement defeats pedagogical purpose | CRITICAL | RESOLVED | Widget moved to Section 3, immediately after the worked example and before the formula. Student now sees the operation animated right when they need it. |
| 2 | Real-photograph example dropped | CRITICAL | RESOLVED | A "real-photograph bridge" paragraph added after the edge detection section (lines 476-506). Describes applying filters to a building photograph and includes a callout about trained CNN first-layer filters (AlexNet, VGGNet). No actual image, but the descriptive bridge is effective and avoids the need for static assets. |
| 3 | Misconception 4 (receptive field) not explicit | IMPROVEMENT | RESOLVED | A WarningBlock "Receptive Field" callout added as an aside to the worked example (lines 277-283). Explicitly states: "The output at position (0,0) was computed from ONLY the 9 input pixels in the top-left 3x3 region. It knows nothing about the bottom-right corner." |
| 4 | Misconception 5 (bigger filters) not addressed | IMPROVEMENT | NOT ADDRESSED | Still absent from the lesson. No aside, no planted question, no teaser for Lesson 2. |
| 5 | No inline worked example with specific numbers | IMPROVEMENT | RESOLVED | A full worked example with a 3x3 input patch, vertical edge filter, element-wise multiplication, and sum = 3 is now inline in Section 3 (lines 229-283). Concrete numbers before the formula. |
| 6 | Check 2 uses blur filter without teaching it | IMPROVEMENT | RESOLVED | A sentence added before Check 2 (line 521-522): "A filter where every value is 1 (or 1/9) computes the average of the neighborhood—it blurs the image instead of highlighting sharp transitions." |
| 7 | Feature map spatial correspondence not explicit | IMPROVEMENT | NOT ADDRESSED | The lesson text still does not explicitly state that output position (i,j) corresponds to the input region at (i,j). The summary mentions "a spatial answer key" but this correspondence is never articulated in the main explanation. |
| 8 | Output size formula only in aside | POLISH | RESOLVED | The general formula N - F + 1 now appears in the main content (line 212-214): "In general, a filter of size F on an input of size N produces an output of size N - F + 1." |
| 9 | Widget inside ExercisePanel | POLISH | UNCHANGED | Still uses ExercisePanel. The subtitle makes purpose clear. Acceptable. |
| 10 | "What You Already Know" aside is vague | POLISH | NOT ADDRESSED | Still says "You understand weighted sums, learnable parameters, and PyTorch." |

**Resolution summary:** 2/2 CRITICAL resolved. 3/5 IMPROVEMENT resolved. 1/3 POLISH resolved.

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings remain. The two structural problems from iteration 1 (widget placement and real-photograph bridge) are fixed and work well. Four improvement-level findings need attention, two carried forward from iteration 1 and two new.

### Findings

#### [IMPROVEMENT] — Misconception 5 ("bigger filters are always better") still not addressed

**Location:** Missing from lesson (carried from iteration 1, finding #4)
**Issue:** The plan identified this misconception and marked it for a "scope boundary / teaser" treatment. The revised lesson still does not address it. The student learns that 3x3 filters are used, sees them work, and is told they are better than dense layers. But no mention is made of why you would not just use a bigger filter. The key insight ("a 28x28 filter on a 28x28 image IS a dense layer") would solidify the locality argument and plant curiosity for Lesson 2.
**Student impact:** The student may leave wondering "why 3x3 specifically?" without realizing this is an important design choice. Low severity but a missed pedagogical opportunity.
**Suggested fix:** Add 1-2 sentences in the "Why This Beats Dense Layers" section or as an aside: "You might wonder: why not use a bigger filter to capture more context? Consider the extreme case: a 28x28 filter on a 28x28 image connects to every pixel—that is just a dense layer. The power of small filters comes from locality and composition. We will see how stacking small filters gives you large receptive fields with far fewer parameters in the next lesson."

#### [IMPROVEMENT] — Feature map spatial correspondence still not made explicit

**Location:** Sections 3 and 5 (carried from iteration 1, finding #7)
**Issue:** The plan specified the geometric/spatial modality: "The feature map is a 'heat map' of where the pattern was found" and "Feature maps are spatial—their values have positions that correspond to input positions." The lesson says the feature map shows "where the pattern was detected" (summary item 2) and "a spatial answer key" (closing mental model), but never explicitly articulates the 1-to-1 positional mapping: output(i,j) tells you about the input neighborhood starting at (i,j). The formula on line 327 shows this symbolically, but the verbal/intuitive modality for this spatial correspondence is missing.
**Student impact:** The student understands the feature map conceptually but may not grasp that the output has a precise spatial meaning—that position (2,3) in the feature map corresponds to the input patch at (2,3). This matters for understanding how later layers compose spatial features.
**Suggested fix:** Add one sentence after the formula or after the widget: "Notice that each output position corresponds to a specific input region: output(2,3) tells you whether the filter's pattern was present in the input starting at row 2, column 3. The feature map preserves spatial layout."

#### [IMPROVEMENT] — Real-photograph bridge paragraph is redundant with "Not Hand-Designed" warning

**Location:** Lines 464-506 (real-photograph bridge) and lines 464-472 (WarningBlock "Not Hand-Designed")
**Issue:** The "Not Hand-Designed" warning aside (lines 464-471) says: "In a real CNN, the network learns filter values via backprop... The learned filters often end up looking like edge detectors, but the network discovers this on its own." The real-photograph bridge section immediately following (lines 476-496) makes a very similar point: "Researchers have visualized the filters learned by AlexNet, VGGNet... The first layer consistently learns edge detectors... They look remarkably similar to the hand-crafted filters you just used—but the network discovered them entirely on its own through gradient descent." Then Section 10 ("What the Network Actually Learns," lines 696-731) repeats this again: "Researchers have visualized the first-layer filters of trained CNNs. They typically learn edge detectors, color detectors, and texture detectors—but the network discovers these patterns on its own." The same idea—"networks learn filters that look like hand-crafted edge detectors"—appears three separate times in the lesson.
**Student impact:** Repetition is not harmful but it dilutes the lesson's energy. Each repetition adds ~50-80 words. The student reads essentially the same insight three times, which makes the lesson feel like it is padding rather than progressing. The third time they encounter "the network discovers edge detectors on its own" has no new information.
**Suggested fix:** Keep ONE strong version. The real-photograph bridge paragraph is the best location because it grounds the claim in real networks (AlexNet, VGGNet). Move the "Not Hand-Designed" aside's content into the bridge paragraph. Then in Section 10 ("What the Network Actually Learns"), skip the re-explanation of first-layer filter visualization and focus solely on the hierarchical composition insight (edges become corners become shapes become objects), which IS new information.

#### [IMPROVEMENT] — Worked example uses a 4x4 input description but shows a 3x3 patch

**Location:** Section 3, lines 229-283 (worked example)
**Issue:** The worked example text says "Imagine a 4x4 input with a vertical edge" (line 234), but the visual immediately shows only a 3x3 grid of values (the patch at position (0,0)). The student may be confused about whether they are looking at the 4x4 input or the 3x3 patch. The label says "Input patch at position (0,0)" which is correct, but the setup text promises a 4x4 input that is never shown. The student has to imagine a 4x4 grid and then immediately see only a 3x3 subset of it, which is a cognitive jump.
**Student impact:** Moderate confusion. The student might wonder "where is the 4x4 grid?" or not realize the 3x3 visual is a patch extracted from a larger grid. The worked example describes the input but only shows the extracted patch, which is an abstraction leap for a concept that should be maximally concrete.
**Suggested fix:** Either (a) show the full 4x4 input grid first with the 3x3 patch highlighted, then show the extracted patch alongside the filter, or (b) simplify to a 3x3 input with a 2x2 filter so the entire input is visible and the computation can be shown without the "patch extraction" indirection. Option (a) is better pedagogically because it reinforces the "sliding window extracts patches" concept.

#### [POLISH] — "What You Already Know" aside remains vague

**Location:** Section 1, lines 74-78 (TipBlock)
**Issue:** (Carried from iteration 1, finding #10) The aside still says "You understand weighted sums, learnable parameters, and PyTorch. Everything in this lesson builds on those foundations." This misses the chance to prime the dense-to-conv bridge.
**Student impact:** Negligible.
**Suggested fix:** Rewrite to: "You have built dense networks where each neuron connects to every input and computes a weighted sum. Convolutions use the same weighted sum—but only over a small local neighborhood. That single change is what this lesson is about."

#### [POLISH] — "TryThisBlock" experiments numbered with text, not proper ordered list

**Location:** Section 3, lines 297-315 (TryThisBlock aside)
**Issue:** The experiments are listed as `<li>1. ...`, `<li>2. ...` etc. inside an unordered list (`<ul>`). The numbers are part of the text content rather than rendered by an `<ol>`. This produces correct visual output (the numbers show) but is semantically incorrect HTML. A screen reader would announce "bullet: 1. Select..." rather than "item 1: Select..."
**Student impact:** Negligible for sighted users. Minor accessibility issue.
**Suggested fix:** Change `<ul>` to `<ol>` and remove the text numbers, or keep the `<ul>` and remove the numbers (since bullet points do not need them).

#### [POLISH] — Closing mental model echo is a standalone styled div, not a block component

**Location:** Lines 776-787 (closing mental model)
**Issue:** The closing mental model echo is implemented as a raw styled `<div>` with inline Tailwind classes (`bg-violet-500/5 border border-violet-500/20 rounded-lg`). Every other callout-style block in the lesson uses a block component (InsightBlock, TipBlock, ConceptBlock, etc.). This is the only raw styled block in the lesson. It works visually but breaks the pattern.
**Student impact:** None. This is a code consistency issue.
**Suggested fix:** Use an InsightBlock or ConceptBlock component if an appropriate one exists for "echo" content. If no component fits, this is fine as-is.

### Review Notes

**What was fixed well:**
- The widget relocation is the biggest improvement. The ConvolutionExplorer now appears in Section 3 right after the inline worked example and before the formula. This is exactly the right pedagogical position: the student reads the steps, sees a concrete worked example with numbers, then interacts with the widget, and only then sees the formal notation. The flow is now concrete -> interactive -> symbolic, which is excellent.
- The inline worked example is well-crafted. It shows the input patch, the filter (with color-coded cells), the full multiplication chain, and the result. The color coding (rose for negative filter values, violet for positive) matches the widget's color scheme, providing visual consistency.
- The receptive field callout is precisely worded and well-placed as a WarningBlock aside to the worked example.
- The real-photograph bridge paragraph does a reasonable job of grounding the toy examples in real-world scale, even without a static image. The AlexNet/VGGNet reference gives it credibility.
- The blur filter is now introduced before Check 2, resolving the fairness issue.

**Remaining patterns:**
- The main structural issue remaining is content redundancy: the "networks learn filters that look like hand-crafted edge detectors" insight appears three times. This is the most impactful improvement-level fix because it would tighten the lesson noticeably.
- Two iteration 1 findings (misconception 5, feature map spatial correspondence) were not addressed. Both are worth fixing but neither is critical.
- The worked example input-size confusion (says 4x4, shows 3x3 patch) is a new issue introduced by the fix. It needs attention because the worked example is now a central part of the lesson's flow.

**Overall assessment:** The lesson has improved substantially from iteration 1. The two critical structural problems are resolved. The flow through Section 3 is now strong: procedural steps -> concrete worked example -> interactive widget -> formal formula. The remaining issues are all improvement or polish level. One more revision pass should bring this to PASS.

---

## Review — 2026-02-09 (Iteration 3/3)

### Iteration 2 Resolution Status

| # | Finding | Severity | Status | Notes |
|---|---------|----------|--------|-------|
| 1 | Misconception 5 ("bigger filters") not addressed | IMPROVEMENT | RESOLVED | Lines 640-645 now address this with "a 28x28 filter on a 28x28 image connects to every pixel. That is a dense layer" argument, plus a forward pointer to Lesson 2 on stacking small filters. |
| 2 | Feature map spatial correspondence not explicit | IMPROVEMENT | RESOLVED | Lines 337-339 now explicitly state: "output position (i,j) tells you whether the filter's pattern was found in the input region starting at (i,j). The feature map preserves spatial layout — the top-left of the output corresponds to the top-left of the input." |
| 3 | Real-photograph bridge redundant with triple repetition | IMPROVEMENT | RESOLVED | The "networks learn edge detectors" insight now appears once in the bridge paragraph (lines 479-501). Section 10 opens with "You saw earlier..." as a brief callback and moves directly to the NEW insight about hierarchical composition. No more triple-redundancy. |
| 4 | Worked example says "4x4 input" but shows 3x3 patch | IMPROVEMENT | RESOLVED | Text now says "Here is a 3x3 region from an input image with a vertical edge" (line 237), correctly framing it as a patch from a larger image without claiming a specific input size. |
| 5 | "What You Already Know" aside remains vague | POLISH | RESOLVED | Lines 76-80 now read: "You have built dense networks where each neuron computes a weighted sum over every input, with learnable parameters updated by the training loop. Convolutions use the same weighted sum — but only over a small local neighborhood. That single change is what this lesson is about." Specific, primes the dense-to-conv bridge. |
| 6 | TryThisBlock uses `<ul>` with text numbers | POLISH | RESOLVED | Now uses `<ol>` with `list-decimal list-inside` (lines 300-315). Semantically correct ordered list. |
| 7 | Closing mental model echo is raw styled div | POLISH | RESOLVED | Now uses `<InsightBlock>` component (lines 782-789). Consistent with all other callout blocks in the lesson. |

**Resolution summary:** 4/4 IMPROVEMENT resolved. 3/3 POLISH resolved. All 7 iteration 2 findings addressed.

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All critical and improvement findings from iterations 1 and 2 are resolved. The lesson is pedagogically sound, well-structured, and ready to ship. One minor polish observation noted below for the record.

### Findings

#### [POLISH] — Convolution formula hardcodes 3x3 upper bounds

**Location:** Section 3, formula block (line 330)
**Issue:** The BlockMath formula uses `\sum_{m=0}^{2}\sum_{n=0}^{2}` which hardcodes the upper bound for a 3x3 filter. The surrounding text gives the general formula N - F + 1 for output size, but the symbolic representation is specific to F=3. A more general formula would use `\sum_{m=0}^{F-1}\sum_{n=0}^{F-1}`, but this adds abstraction that may not serve the student at this stage.
**Student impact:** Negligible. The student sees the concrete formula immediately after a concrete 3x3 worked example. The hardcoded bounds reinforce the specific case they just computed. Generalizing the formula would add cognitive load without pedagogical benefit at this point.
**Suggested fix:** No fix needed. The concrete formula is pedagogically appropriate here. If a future lesson needs the general formula, it can introduce it then.

### Review Notes

**What was fixed well in iteration 3:**
- The misconception 5 treatment is elegant. Rather than adding a standalone section, it is woven naturally into the "Why This Beats Dense Layers" section as a rhetorical question: "You might wonder: why not use a bigger filter to capture more context?" The extreme-case argument (28x28 filter = dense layer) is concise and convincing. The forward pointer to Lesson 2 on stacking small filters creates curiosity without overloading.
- The spatial correspondence fix is precise. One sentence after the formula articulates the 1-to-1 positional mapping and explicitly uses the phrase "preserves spatial layout." This fills the planned geometric/spatial modality gap without adding verbosity.
- The redundancy elimination is clean. Section 10 now focuses entirely on hierarchical composition (the genuinely new insight), using "You saw earlier..." to avoid repeating the first-layer filter visualization material. The lesson's energy no longer sags from repetition.
- The worked example reframing ("a 3x3 region from an input image") is natural and avoids the earlier confusion about input size.
- The "What You Already Know" aside is now one of the strongest asides in the lesson — it primes the exact connection the lesson will build on.

**Overall lesson quality:**
- **Narrative arc:** Strong. The flat-vector problem creates genuine motivation. The progression from "what's wrong with dense layers for images" through "here is the operation" to "here is why it works" is logical and engaging.
- **Pedagogical flow:** The Section 3 sequence (procedural steps -> worked example with numbers -> interactive widget -> formal formula) is textbook concrete-before-abstract ordering. The widget placement, which was the critical finding in iteration 1, is now in exactly the right position.
- **Modalities:** Five distinct modalities are present (verbal/analogy, visual/interactive, symbolic/formula, geometric/spatial, concrete/worked example). Each genuinely adds a different perspective on the core concept.
- **Misconceptions:** All five planned misconceptions are addressed at appropriate locations in the lesson. No misconception is left unaddressed.
- **Interaction design:** Widget buttons, preset selectors, and detail/summary elements all have appropriate cursor styles. The widget supports both step-through and auto-animate modes. The ComputationDetail panel shows the full multiplication chain at each position.
- **Scope discipline:** The lesson stays strictly within its stated scope. No drift into pooling, stride, padding, or training. Forward pointers to Lesson 2 are brief and appropriate.
- **Connection to prior knowledge:** Every new concept is explicitly tied to something the student already knows. The "same weighted sum, different scope" bridge is established early and reinforced throughout.

**Across all three iterations:**
- Iteration 1: 2 critical, 5 improvement, 3 polish -> MAJOR REVISION
- Iteration 2: 0 critical, 4 improvement, 3 polish -> NEEDS REVISION
- Iteration 3: 0 critical, 0 improvement, 1 polish -> PASS

The review loop worked as designed. The two critical structural issues (widget placement, missing example) were caught in iteration 1 and fixed. Iteration 2 caught content-level issues (missing misconception, spatial correspondence gap, redundancy, worked example confusion) and polish items. Iteration 3 confirmed all fixes and found no new issues of substance. The lesson is ready to ship.
