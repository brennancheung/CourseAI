# Lesson: Architecture Evolution

**Module:** 3.2 — Modern Architectures
**Position:** Lesson 1 of 3
**Slug:** `architecture-evolution`
**Type:** Interactive web lesson (no notebook)

---

## Phase 1: Student State

### Relevant Concepts (from records)

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Convolution as sliding filter | DEVELOPED | what-convolutions-compute (3.1) | Core operation fully understood; hand-computed and widget-explored |
| Feature map (output of convolution) | DEVELOPED | what-convolutions-compute (3.1) | Spatial grid of filter responses; knows output(i,j) corresponds to input region at (i,j) |
| Edge detection filters | DEVELOPED | what-convolutions-compute (3.1) | Hand-crafted filters as intuition builders; knows networks learn their own |
| Conv-pool-fc architecture pattern | APPLIED | mnist-cnn-project (3.1) | Built end-to-end CNN; "spatial shrinks, channels grow, then flatten" |
| Output size formula: floor((N-F+2P)/S)+1 | DEVELOPED | building-a-cnn (3.1) | Verified against multiple examples |
| Max pooling | DEVELOPED | building-a-cnn (3.1) | "Zoom out" analogy; preserves feature presence, not exact position |
| Stride and padding | DEVELOPED | building-a-cnn (3.1) | Engineering knobs; padding prevents shrinking, stride controls step size |
| Hierarchical feature composition (edges -> shapes -> objects) | DEVELOPED | building-a-cnn (3.1) | Planted in L1, developed in L2; each conv-pool stage expands receptive field |
| Receptive field (what output position "sees") | INTRODUCED | what-convolutions-compute (3.1) | Qualitative understanding; 3x3 filter sees 3x3 region; stacking increases RF |
| Architecture encodes assumptions about data | DEVELOPED | mnist-cnn-project (3.1) | Core insight of Module 3.1; CNN assumes spatial structure |
| Weight sharing | INTRODUCED | what-convolutions-compute (3.1) | Same filter weights at every position; parameter efficiency |
| nn.Conv2d / nn.MaxPool2d API | APPLIED | mnist-cnn-project (3.1) | Wrote the code; can specify in/out channels, kernel_size, padding |
| ReLU as modern default activation | DEVELOPED | activation-functions (1.2) | max(0,x); knows it replaced sigmoid; full neuron formula |
| Vanishing gradients (quantitative) | DEVELOPED | training-dynamics (1.3) | 0.25^N decay; sigmoid vs ReLU; telephone game analogy |
| Batch normalization | INTRODUCED | training-dynamics (1.3) | Normalize activations between layers; learned gamma/beta; stabilizes training; allows deeper networks |
| Dropout | DEVELOPED | overfitting-and-regularization (1.3) | Randomly silence neurons during training; creates implicit ensemble |
| Xavier/He initialization | DEVELOPED | training-dynamics (1.3) | Preserve signal variance; He for ReLU |
| Skip connections / ResNets | MENTIONED | training-dynamics (1.3) | Teased as enabling 152-layer networks; deferred |
| nn.Module subclass pattern | DEVELOPED | nn-module (2.1) | __init__ + forward(); LEGO bricks analogy |
| Learned filters (via backprop) | INTRODUCED | what-convolutions-compute (3.1) | AlexNet/VGGNet rediscover edge detectors; training loop unchanged |

### Established Mental Models

- "A filter is a pattern detector" (what-convolutions-compute)
- "Feature map is a spatial answer key" (what-convolutions-compute)
- "A CNN is a series of zoom-outs" (building-a-cnn)
- "Spatial shrinks, channels grow, then flatten" (building-a-cnn)
- "Architecture encodes assumptions about data" (mnist-cnn-project)
- "Telephone game for gradient flow" (training-dynamics)
- "ReLU + He init + batch norm = modern baseline" (training-dynamics)

### What Was NOT Covered (Relevant Gaps)

- **Multiple input channels / RGB images**: All CNN work used single-channel MNIST. The student has not worked with 3-channel inputs, though `in_channels` parameter was used in Conv2d.
- **ImageNet or large-scale image classification**: All examples used MNIST (28x28 grayscale, 10 classes).
- **Historical evolution of architectures**: LeNet, AlexNet, VGG only briefly name-dropped. No analysis of what changed between them.
- **Receptive field computation for stacked layers**: Only qualitative understanding that stacking increases RF. No formula or quantitative analysis.
- **Global average pooling**: Mentioned briefly in an aside in building-a-cnn, not developed.
- **Data augmentation**: Not covered anywhere.

### Readiness Assessment

The student is well-prepared. They have a solid understanding of what convolutions compute, how CNNs are structured, and why architecture matters. The key gap is that all their experience is with a single toy architecture on MNIST. This lesson extends their understanding to real-world scale by examining how the field evolved through a sequence of progressively deeper architectures. The prerequisite concepts (conv, pooling, stride, padding, ReLU, vanishing gradients, batch norm) are all at DEVELOPED or higher depth.

---

## Phase 2: Analyze

### Target Concept

**This lesson teaches the student to explain why stacking more convolutional layers improves performance (deeper feature hierarchies, larger effective receptive field, more nonlinearity) and how the field discovered this through the progression from LeNet to AlexNet to VGG.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Convolution as sliding filter | DEVELOPED | DEVELOPED | what-convolutions-compute | OK | Need to reason about what multiple stacked conv layers compute |
| Conv-pool-fc pattern | DEVELOPED | APPLIED | mnist-cnn-project | OK | Exceeds requirement; student built this |
| Output size formula | DEVELOPED | DEVELOPED | building-a-cnn | OK | Need for parameter counting and dimension reasoning |
| Receptive field (qualitative) | INTRODUCED | INTRODUCED | what-convolutions-compute | OK | We develop receptive field quantitatively in this lesson as a new concept |
| Hierarchical features (edges->shapes->objects) | INTRODUCED | DEVELOPED | building-a-cnn | OK | Core to understanding why depth helps |
| ReLU activation | INTRODUCED | DEVELOPED | activation-functions | OK | Needed to understand AlexNet's innovation over sigmoid |
| Vanishing gradients | INTRODUCED | DEVELOPED | training-dynamics | OK | Needed to understand why sigmoid limited depth and ReLU enabled it |
| Dropout | INTRODUCED | DEVELOPED | overfitting-and-regularization | OK | Needed to understand AlexNet's innovations |
| Batch normalization | MENTIONED | INTRODUCED | training-dynamics | OK | Mentioned in VGG context; not deeply required here (developed in resnets lesson) |
| Weight sharing / parameter efficiency | INTRODUCED | INTRODUCED | what-convolutions-compute | OK | Needed for 3x3 vs 5x5 parameter comparison |
| Multiple input channels (RGB) | INTRODUCED | NOT TAUGHT | - | GAP | LeNet/AlexNet/VGG all process RGB images; student only used single-channel |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Multiple input channels (RGB) | Small | Brief recap section (2-3 paragraphs + visual). Student knows `in_channels` parameter from Conv2d and knows "multiple filters = multiple feature maps." Extending to 3-channel input is a small step: 3x3x3 filter instead of 3x3x1. Show dimension: Conv2d(3, 64, 3) means 64 filters, each 3x3x3 = 27 weights per filter. Include a visual of RGB channels being processed. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example | Where to Address |
|---------------|----------------------|-----------------|-----------------|
| "Deeper always means better — just stack more layers" | The lesson's narrative arc says "deeper = better" and the history shows increasingly deep networks winning. Natural extrapolation. | Networks deeper than ~20 layers (pre-ResNet) performed WORSE on both training AND test. If it were just overfitting, training accuracy would still be high. Tease this as cliffhanger for resnets lesson. | End of lesson, after VGG section |
| "AlexNet succeeded because GPUs made bigger networks possible" | GPUs are a salient and easily understood factor. True that GPUs helped, but insufficient explanation. | A large network with sigmoid activations and no dropout would still fail — the architectural innovations (ReLU, dropout, local response normalization) were essential. GPU was necessary but not sufficient. | AlexNet section |
| "A single 5x5 filter and two stacked 3x3 filters do the same thing" | Both cover a 5x5 receptive field, so they seem equivalent. The student may confuse "same receptive field" with "same computation." | Two 3x3 filters have 2 nonlinearities (ReLU between them); one 5x5 has only 1. Two 3x3 filters compute 18 parameters; one 5x5 computes 25. Same receptive field, different computational power and efficiency. | VGG section (core concept) |
| "Architecture evolution was about adding more of the same layers" | The student's only architecture experience is stacking conv-pool pairs. Natural assumption that evolution = more pairs. | Each generation changed WHAT was stacked: LeNet used sigmoid + average pooling + 5x5 filters; AlexNet switched to ReLU + max pooling + mixed filter sizes + dropout; VGG standardized on 3x3 only. The innovations were qualitative, not just quantitative. | Throughout — each architecture section highlights what changed and why |
| "Modern architectures are too complex to understand" | The student may feel intimidated by names like "VGG-16" or "152-layer ResNet." | VGG-16 is literally just 3x3 conv and 2x2 pool repeated in a pattern. The core ideas are simple — the depth is just repetition of a simple pattern. | VGG section — show the repeating block structure |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| LeNet-5 architecture diagram with dimensions | Positive | Show the original CNN pattern the student already knows — conv, pool, conv, pool, FC. Familiar ground. | Connects directly to the student's MNIST CNN from Module 3.1. Similar structure, similar scale. Shows "you already understand the first generation." |
| AlexNet architecture with annotated innovations | Positive | Show how each innovation (ReLU, dropout, multiple GPUs, larger scale) addressed a specific problem. | Demonstrates that progress is problem-driven, not random. Each change has a reason. |
| VGG-16 block structure with 3x3 stacking analysis | Positive | The core concept: two 3x3 convs = same receptive field as one 5x5 but fewer params and more nonlinearity. | This is THE key insight of the lesson. VGG proved that disciplined small-filter stacking beats ad-hoc large filters. |
| 3x3+3x3 vs 5x5 receptive field comparison | Positive (detailed) | Concrete numerical comparison: receptive field, parameter count, nonlinearities. | Makes the abstract principle concrete with actual numbers the student can verify. |
| "Just make it deeper" failure (pre-ResNet) | Negative | Show that blindly adding layers eventually makes things worse. Training accuracy degrades. | Prevents the "deeper always better" misconception. Sets up the resnets lesson. Critical for narrative arc. |
| Student's MNIST CNN vs LeNet comparison | Positive (bridge) | Map the student's own architecture to LeNet to show they already built something similar. | Low activation energy — "you've already done this." Grounds the historical narrative in personal experience. |

---

## Phase 3: Design

### Narrative Arc

You just built a CNN that crushes MNIST — 99% accuracy, better than a dense network, and you understand why. But MNIST is a toy problem: 28x28 grayscale images of centered, isolated digits. Real-world images are 224x224, have three color channels, contain cluttered scenes with multiple objects, and need to distinguish among 1000 categories. Your 2-layer CNN would be lost. So how did the field get from toy-scale CNNs to networks that match human performance on ImageNet? The answer turns out to be surprisingly simple in principle — make the network deeper — but each step deeper required solving a real problem. LeNet (1998) proved CNNs work. AlexNet (2012) scaled them up with ReLU and dropout. VGG (2014) discovered that a disciplined pattern of tiny 3x3 filters stacked deep beats larger filters — fewer parameters, more nonlinearity, same receptive field. Each generation is a response to the previous generation's limitations, and understanding this progression gives you the mental framework to evaluate any architecture you encounter.

### Modalities Planned

| Modality | What Specifically | Why This Modality |
|----------|------------------|-------------------|
| Visual (architecture diagrams) | Side-by-side architecture diagrams for LeNet, AlexNet, VGG showing layer types, dimensions, and data flow. Color-coded by layer type. | Architecture comparison is inherently spatial — seeing the structures side-by-side makes the depth progression visceral in a way text cannot. |
| Concrete example (parameter counting) | Exact parameter counts for 3x3+3x3 vs 5x5, for each architecture's total parameters, for the student's MNIST CNN vs LeNet. | Numbers make abstract claims ("more efficient") concrete and verifiable. The student can check the math themselves. |
| Interactive (Architecture Comparison Explorer) | Widget where student can select architectures (LeNet/AlexNet/VGG-16) and see layer-by-layer dimension tracking, parameter counts per layer, cumulative parameter count, and receptive field growth. | The student already used the CnnDimensionCalculator in building-a-cnn. This extends that experience to real architectures. Interactivity lets them explore "what if" questions (what if VGG used 5x5 instead?). |
| Verbal/Analogy | "Each generation solved the previous generation's limitation" as a narrative thread. LeNet = proof of concept, AlexNet = scale + ReLU + regularization, VGG = principled depth. | The historical narrative gives structure and memorability to what could otherwise be a dry list of architectures. |
| Symbolic (receptive field formula) | RF_l = RF_{l-1} + (k-1) * product of strides up to layer l-1. Applied concretely: two 3x3 convs give RF=5, three give RF=7. | Quantifies the receptive field growth that the student previously only understood qualitatively. Connects to the "zoom out" analogy. |
| Intuitive | "You already built a LeNet" — mapping the student's MNIST CNN to LeNet-5. The progression from their work to the state of the art is a series of comprehensible steps, not a mysterious leap. | Reduces intimidation. Connects new content to lived experience. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3
  1. Effective receptive field of stacked small filters (quantitative RF computation for stacked convs)
  2. Parameter efficiency of small vs large filters (3x3+3x3 vs 5x5 tradeoff)
  3. Architecture evolution as problem-driven innovation (LeNet -> AlexNet -> VGG as a connected narrative)
- **Previous lesson load:** CONSOLIDATE (mnist-cnn-project was a project with no new concepts)
- **This lesson's load:** STRETCH — three genuinely new concepts, but heavily supported by existing knowledge (conv, pooling, ReLU, vanishing gradients, dropout all at DEVELOPED+). The conceptual novelty is in composition and comparison, not in learning new operations.
- **Appropriate?** Yes. After a CONSOLIDATE lesson, a STRETCH is well-timed. The student has momentum from their MNIST success.

### Connections to Prior Concepts

| Prior Concept | Connection in This Lesson |
|---------------|--------------------------|
| Conv-pool-fc pattern (APPLIED) | LeNet IS this pattern. "You already built a LeNet." |
| "Spatial shrinks, channels grow, then flatten" | Visible in all three architectures — the pattern is universal |
| ReLU as modern default (DEVELOPED) | AlexNet's switch from sigmoid to ReLU was a breakthrough — connects to vanishing gradient understanding |
| Vanishing gradients (DEVELOPED) | Explains WHY sigmoid limited depth and ReLU enabled it. The telephone game analogy applies directly. |
| Dropout (DEVELOPED) | AlexNet introduced dropout — the student already knows how it works and why |
| Receptive field (INTRODUCED) | Upgraded to quantitative: stacking 3x3 convs grows RF predictably |
| "Architecture encodes assumptions about data" (DEVELOPED) | Extended: architecture DESIGN encodes assumptions about the PROBLEM (simple vs complex features, small vs large objects) |
| Output size formula (DEVELOPED) | Used for parameter counting and dimension tracking through architectures |
| Batch normalization (INTRODUCED) | Referenced as part of the modern training recipe; deepened in resnets lesson |

**Extending prior analogies:**
- "A CNN is a series of zoom-outs" extends to: deeper networks zoom out further, seeing larger patterns
- "Telephone game" extends to: with ReLU + careful design, the message survives more relays

**Potentially misleading analogies:** None identified. The zoom-out analogy naturally extends to deeper networks.

### Scope Boundaries

**This lesson IS about:**
- The architectural progression LeNet -> AlexNet -> VGG
- WHY depth helps (hierarchical features, receptive field, nonlinearity)
- The 3x3 stacking insight (parameter efficiency + more nonlinearity for same RF)
- Brief treatment of RGB/multi-channel input (gap resolution)
- The "deeper = better" principle and its limits (tease degradation problem)

**This lesson is NOT about:**
- ResNets or skip connections (next lesson)
- GoogLeNet/Inception (mentioned as contemporary of VGG, not developed)
- Implementation in PyTorch (no notebook; widget handles exploration)
- Training these architectures (too expensive; conceptual understanding is the goal)
- Data augmentation details
- Batch normalization mechanics (already INTRODUCED; deepened in resnets)
- ImageNet competition history beyond what motivates architecture choices
- Modern architectures beyond VGG (EfficientNet, Vision Transformers, etc.)

**Depth targets:**
- LeNet architecture: INTRODUCED (recognize structure, map to student's own CNN)
- AlexNet innovations: INTRODUCED (explain what changed and why)
- VGG architecture and 3x3 philosophy: DEVELOPED (analyze parameter efficiency, explain receptive field math)
- Receptive field (quantitative for stacked convs): DEVELOPED (compute for stacked 3x3 layers)
- Parameter efficiency of small filters: DEVELOPED (calculate and compare)

### Lesson Outline

1. **Context + Constraints** — This lesson traces how CNNs evolved from LeNet to VGG. We are building understanding of architectural design principles, not implementing these networks. By the end, you will understand why VGG uses only 3x3 filters and why that matters. We are NOT covering ResNets (next lesson) or training these architectures.

2. **Hook (before/after + bridge to personal experience)** — "Your MNIST CNN has 2 conv layers and handles 10 digit classes at 99%. ImageNet has 1000 classes of real-world objects at 224x224 RGB. The winning network in 2012 had 8 layers. By 2014, the winners had 19. How did the field figure out that going deeper was the answer — and what does 'deeper' actually buy you?" Side-by-side: student's CNN vs ImageNet challenge stats. Map student's architecture to LeNet: "Congratulations, you basically built a 1998 architecture."

3. **Recap: RGB / Multi-channel Input** — Brief section resolving the multi-channel gap. "Your MNIST CNN used Conv2d(1, 32, 3). For RGB images, that becomes Conv2d(3, 64, 3). Each filter is now 3x3x3 = 27 weights instead of 3x3x1 = 9. Everything else works the same." Include a simple visual showing 3-channel input being processed by a single filter. Keep to 2-3 paragraphs.

4. **LeNet (1998): The Proof of Concept** — Architecture diagram with dimensions. 5x5 filters, average pooling, sigmoid/tanh activations, 2 conv layers, ~60K parameters. Map directly to student's MNIST CNN: same pattern (conv-pool-conv-pool-fc), similar scale. Key point: this worked for handwritten digits and zip codes, but the architecture choices (sigmoid, avg pooling, 5x5) reflect the era's constraints and understanding. "Comprehension check: what would you change about LeNet based on what you know?" (Expected: ReLU instead of sigmoid, max pooling instead of average.)

5. **AlexNet (2012): The Breakthrough** — Architecture diagram. The innovations that mattered: (1) ReLU instead of sigmoid — connect to vanishing gradient understanding, (2) Dropout — connect to regularization knowledge, (3) GPU training for scale, (4) 8 layers deep. NOT just "bigger LeNet" — each change solved a specific problem. Misconception addressed: GPU alone did not make AlexNet work; the architectural innovations were essential. Show parameter count (~60M) vs LeNet (~60K) — 1000x scale-up.

6. **Check 1: Predict-and-verify** — "AlexNet uses 11x11 filters in the first layer and 3x3 filters in later layers. Why might larger filters be useful early and smaller filters later?" Student predicts, then reveal: early layers need to cover enough input to detect basic features in 224x224 images; later layers operate on already-processed feature maps where 3x3 is sufficient.

7. **VGG (2014): The 3x3 Insight** — The core concept of the lesson. VGG asked: "What if we ONLY use 3x3 filters and go deeper?" Architecture diagram showing the block pattern: 2-3 conv layers (all 3x3) followed by max pool, repeated 5 times. 16-19 layers total. The key analysis:
   - **Receptive field equivalence:** Two 3x3 convs see a 5x5 region of the input. Three 3x3 convs see a 7x7 region. Worked example with diagram showing how the second 3x3 filter's receptive field expands through the first.
   - **Parameter comparison:** Two 3x3 convs = 2 * (3*3*C*C) = 18C^2 parameters. One 5x5 conv = 25C^2 parameters. 28% fewer parameters for the same receptive field.
   - **More nonlinearity:** Two 3x3 convs have 2 ReLU activations between them. One 5x5 has only 1. More nonlinearity = more expressive power from the same receptive field.
   - "VGG proved: given a receptive field budget, spend it on many small filters rather than fewer large ones."

8. **Check 2: Transfer question** — "A colleague proposes using a single 7x7 filter. How many stacked 3x3 filters give the same receptive field? Compare the parameter counts." (Answer: three 3x3 = RF of 7; 3*9C^2 = 27C^2 vs 49C^2 — 45% fewer parameters, 3 nonlinearities vs 1.)

9. **Explore: Architecture Comparison Explorer** — Interactive widget. Student selects architecture (LeNet / AlexNet / VGG-16), sees:
   - Layer-by-layer pipeline with dimensions (extends CnnDimensionCalculator experience)
   - Parameter count per layer and cumulative
   - Receptive field growth through layers
   - Sidebar comparing architectures on key metrics (total params, depth, min filter size, year, ImageNet accuracy)
   TryThisBlock experiments:
   - Compare VGG-16's total parameter count to AlexNet's
   - Find where in VGG-16 the receptive field exceeds 7x7 (when 3 stacked 3x3 convs without pooling)
   - Notice that most of VGG's parameters are in the FC layers, not the conv layers

10. **Elaborate: The Pattern of Innovation** — Pull back to the meta-level. Each architecture generation responded to the previous one's limitations: LeNet was limited by sigmoid + small scale. AlexNet solved vanishing gradients (ReLU) and overfitting (dropout) but used ad-hoc filter sizes. VGG showed that principled simplicity (all 3x3) + depth beats ad-hoc complexity. This pattern — identify the bottleneck, design the solution — is how all architecture innovation works. Connect to "architecture encodes assumptions": deeper networks assume features are hierarchical and compositional.

11. **Elaborate: But Deeper Has Limits** — The cliffhanger. "If deeper is better, why stop at 19 layers? VGG researchers and others tried going to 30, 50, 100 layers. Something unexpected happened: deeper networks performed WORSE. Not just on test data — on training data too. This is NOT overfitting (overfitting means high training accuracy, low test accuracy). This is something else entirely." Name it: the degradation problem. Show the famous training accuracy curve (deeper network with higher training error). "The next lesson explains what causes this and the elegant solution that enabled 152-layer networks."

12. **Summarize** — Key takeaways:
    - Deeper networks learn richer hierarchical features and have larger effective receptive fields
    - VGG's insight: stacking small (3x3) filters beats using large filters — fewer parameters, more nonlinearity, same receptive field
    - Architecture evolution is problem-driven: each generation solves the previous one's limitation
    - But depth has a limit — the degradation problem (next lesson)
    Mental model: "Depth buys hierarchy and receptive field, but each step must be earned with the right innovations."

13. **Next Step** — "We know deeper is better but deeper eventually breaks. The next lesson tackles the degradation problem head-on and introduces the residual connection — the innovation that took networks from 19 layers to 152."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (RGB gap identified and resolved)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (RGB: brief recap section)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned (6: visual, concrete, interactive, verbal, symbolic, intuitive)
- [x] At least 2 positive examples + 1 negative (6 examples: 5 positive + 1 negative)
- [x] At least 3 misconceptions identified (5 misconceptions with negative examples)
- [x] Cognitive load <= 3 new concepts (3: quantitative RF, parameter efficiency, architecture evolution as narrative)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 5
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings that would leave the student truly lost, but the one critical finding (missing visual modality for receptive field stacking) weakens the core concept delivery, and several improvement findings would meaningfully strengthen the lesson.

### Findings

#### [CRITICAL] — Missing visual modality for receptive field equivalence (core concept)

**Location:** "Receptive Field Equivalence" section (lines 549-604)
**Issue:** The core insight of this lesson is that stacking small 3x3 filters builds large receptive fields. The planning document explicitly calls for a "visual/diagram" modality and specifically mentions "Worked example with diagram showing how the second 3x3 filter's receptive field expands through the first." The built lesson teaches this entirely through text and a formula. There is no diagram showing how a 3x3 filter on top of a 3x3 filter sees a 5x5 region of the original input. The overlap explanation ("they overlap by 2 positions") is abstract and hard to visualize without a spatial diagram.
**Student impact:** The receptive field equivalence is the single most important concept in the lesson. Without a visual showing the expanding receptive field through stacked layers, the student must construct the spatial reasoning entirely from text. This is precisely the kind of concept that benefits enormously from a visual modality (highlighting the input region that each output position "sees" through successive layers). The text explanation is correct but relies on the student mentally simulating a 2D spatial process from a 1D description.
**Suggested fix:** Add a small inline SVG or styled diagram showing: (a) a 1D or 2D grid with a 3x3 window highlighted, (b) a second 3x3 window on top of that, with the corresponding 5x5 region in the original input highlighted. Color-coded to show the expansion. This does not need to be a full interactive widget; a static visual showing the geometric stacking would suffice.

#### [IMPROVEMENT] — Widget is exploration-only, no manipulation or "what if" discovery

**Location:** ArchitectureComparisonExplorer widget (lines 765-806 in lesson, full widget file)
**Issue:** The planning document says "Interactivity lets them explore 'what if' questions (what if VGG used 5x5 instead?)." The built widget is a read-only architecture browser: the student selects one of three architectures and views a static table of layers. There is no ability to modify filter sizes, add/remove layers, or compare "what if" scenarios. The widget is essentially a formatted data viewer. This contrasts with the CnnDimensionCalculator from building-a-cnn, which let the student build architectures interactively. The TryThisBlock experiments ask the student to find information in the tables, but these are reading comprehension tasks, not genuine exploration.
**Student impact:** The student clicks through three tabs and reads numbers. The interactive value is low compared to what they experienced with the CnnDimensionCalculator. The "what if VGG used 5x5" question from the plan cannot be answered with this widget. The student may disengage because there is no genuine manipulation or discovery.
**Suggested fix:** Consider adding a mode or separate section where the student can toggle between "3x3 stacking" and "equivalent single large filter" for a VGG block, showing the parameter count and nonlinearity count change in real time. Even a simple toggle (e.g., "Replace Block 3's three 3x3 convs with one 7x7 conv") would make the core insight interactive rather than just read.

#### [IMPROVEMENT] — Fourth misconception ("evolution was about adding more of the same") not explicitly addressed

**Location:** Throughout the lesson
**Issue:** The planning document identifies the misconception "Architecture evolution was about adding more of the same layers" and says it should be addressed "Throughout" each architecture section by highlighting what changed and why. In the built lesson, each architecture section does describe innovations, but the misconception is never explicitly called out or disproved. There is no moment where the lesson says "you might think each generation just added more of the same layers, but actually..." The student could still walk away thinking the pattern was purely quantitative (more layers, more parameters) rather than qualitative (different activations, different filter philosophies, different regularization).
**Student impact:** This is a subtle misconception that the lesson's narrative partially addresses through its structure, but without explicit confrontation the student may not update their mental model. The "Pattern of Innovation" section (lines 810-867) comes closest but frames it as positive observation rather than misconception correction.
**Suggested fix:** Add a brief aside or inline callout near the AlexNet section or the "Pattern of Innovation" section that explicitly names this misconception: "Notice that each generation didn't just add more layers. AlexNet changed the activation function, the pooling type, and added regularization. VGG changed the filter size philosophy. The innovations were qualitative, not just quantitative."

#### [IMPROVEMENT] — No visual diagram for any of the three architectures

**Location:** LeNet section (lines 270-283), AlexNet section (lines 438-455), VGG section (lines 677-693)
**Issue:** The planning document calls for "side-by-side architecture diagrams" as the primary visual modality. The built lesson uses text-based StageRow components for all three architectures. These are essentially text tables with colored dots. While clear and well-organized, they are not spatial diagrams showing the network's shape (the progressive narrowing and channel expansion). A true architecture diagram would show the spatial dimensions visually (blocks shrinking in width/height but growing in depth), making the "spatial shrinks, channels grow" pattern visceral rather than requiring the student to parse text.
**Student impact:** The student reads numbers in a text list rather than seeing the architecture's shape. The "zoom-out" metaphor from Building a CNN would be reinforced by seeing the spatial dimensions physically shrink in a diagram. The StageRow format is accurate but does not leverage the visual/spatial modality the plan calls for.
**Suggested fix:** For at least VGG-16 (the core architecture), consider adding a proportional block diagram where block width represents spatial dimensions and block color/depth represents channels. This was done in the CnnDimensionCalculator widget (which had "proportional pipeline visualization"). Even a simple CSS-based proportional representation would add genuine visual modality.

#### [IMPROVEMENT] — The fifth misconception ("modern architectures too complex to understand") not addressed

**Location:** VGG section
**Issue:** The planning document identifies the misconception that "Modern architectures are too complex to understand" and plans to address it in the VGG section by showing the repeating block structure. While the lesson does show VGG's repeating pattern and says "one of the simplest to understand because it follows a rigid template" (line 675), it does not explicitly name or disprove the intimidation factor. The student who sees "16 weight layers" and "138M parameters" may still feel intimidated.
**Student impact:** The lesson implicitly addresses this by showing simplicity, but an explicit acknowledgment of the intimidation ("16 layers sounds complex, but look at the pattern: it is the same two operations repeated") would be more effective for an ADHD learner who might otherwise disengage when seeing large numbers.
**Suggested fix:** Add one sentence near line 675 explicitly addressing the intimidation: "16 weight layers sounds complex, but VGG-16 only has two building blocks: a 3x3 conv and a 2x2 pool. Everything else is repetition."

#### [IMPROVEMENT] — Comprehension check after LeNet references "Training Dynamics" without recap

**Location:** LeNet comprehension check, line 315-316
**Issue:** The reveal text says "sigmoid causes vanishing gradients (remember the telephone game from Training Dynamics)." The student learned the telephone game analogy and vanishing gradients at DEVELOPED depth in Training Dynamics (module 1.3, lesson 6). That was at least 4 lessons ago (training-dynamics, overfitting-and-regularization, what-convolutions-compute, building-a-cnn, mnist-cnn-project = 5 lessons ago). Per the Reinforcement Rule, concepts INTRODUCED more than 3 lessons ago need reinforcement. While vanishing gradients is at DEVELOPED depth (not INTRODUCED), the specific telephone game analogy may be fading. The lesson drops the reference without any brief recap of what the telephone game is.
**Student impact:** If the student has forgotten the telephone game analogy, the reference is a dead link in their mental model. They may skim past it rather than connecting to their prior understanding.
**Suggested fix:** Add a parenthetical: "sigmoid causes vanishing gradients (remember the telephone game? each layer slightly shrinks the gradient signal, and after many layers almost nothing gets through)." One sentence of context is sufficient to reactivate the memory.

#### [POLISH] — VGG-16 receptive field annotations may be incorrect

**Location:** VGG-16 architecture diagram, line 684
**Issue:** The annotation says "RF=40 after block" for Block 3 (three 3x3 convs on 56x56 input after two 2x2 max pools). The widget data shows RF=40 at the end of Block 3's third conv (line 107-108 of the widget), and RF=44 after the subsequent MaxPool. The lesson's StageRow shows "RF=40 after block" on the last conv of Block 3, which is technically before the pool. This is not wrong but could be confusing since "after block" could mean "after the pool that ends the block." Similarly, Block 5 shows "RF=212 after block" on the last conv, but the pool brings it to 212 as well (per the widget data). Recommend verifying that the RF values in the StageRow annotations match the widget data consistently and clarifying whether "after block" means after the convs or after the pool.
**Student impact:** Minor confusion if the student cross-references the lesson text with the widget and finds different numbers for the same layer. Low impact since the RF concept is taught through the formula, not through these annotations.
**Suggested fix:** Verify RF values match the widget exactly, and clarify that "after block" means after the final conv in the block (before pool).

#### [POLISH] — "Training Dynamics" capitalized as a lesson name in body text

**Location:** Line 315
**Issue:** "Training Dynamics" is capitalized as if it is a title, but there is no link or explicit lesson reference syntax. Other lesson references in the lesson use a more natural phrasing (e.g., "In Building a CNN, you learned..." on line 558, or "You already know this from Overfitting and Regularization" on line 391). The capitalization is inconsistent: sometimes it reads like a proper noun ("from Training Dynamics") and sometimes more naturally. This is minor but the capitalized form without a link may look odd to the student.
**Student impact:** Negligible. The student understands these are lesson references.
**Suggested fix:** Ensure consistent phrasing for lesson references. Either always capitalize as proper nouns or use natural phrasing with "the" (e.g., "from the training dynamics lesson").

### Review Notes

**What works well:**
- The narrative arc is strong. The progression from "you built a LeNet" through AlexNet to VGG feels natural and motivated. The student is grounded in their own experience at every step.
- The hook (MNIST vs ImageNet comparison) is compelling and makes the scale gap visceral.
- The RGB recap section is well-scoped: just enough to fill the gap without overteaching.
- The misconception about GPUs being sufficient (WarningBlock in the AlexNet section) is well-placed and well-argued.
- The "3x3 vs 5x5" parameter comparison is concrete and verifiable. The student can check the math.
- The comprehension checks (predict-and-verify and transfer question) are well-designed and appropriately placed.
- The cliffhanger about the degradation problem is effective and sets up the next lesson cleanly.
- The summary and mental model are concise and capture the essential takeaways.
- Em dashes are correctly formatted throughout (no spaces).
- Cursor styles are correct on all interactive elements.
- The lesson stays within its stated scope boundaries.

**Patterns to watch:**
- The widget is the weakest element relative to the plan. The plan envisioned genuine interactivity ("what if VGG used 5x5?") but the built widget is a static data viewer. This is an investment question: a more interactive widget would significantly strengthen the lesson but requires more implementation effort.
- The lesson relies heavily on the StageRow text format for architecture visualization. This works but misses the "architecture diagrams" visual modality from the plan. The CnnDimensionCalculator in building-a-cnn had proportional pipeline visualization; extending that pattern to this lesson's architectures would be high-value.
- The lesson addresses 3 of 5 planned misconceptions explicitly (GPUs misconception, same RF = same computation, deeper always better). The other 2 (evolution = more of the same, modern architectures too complex) are addressed implicitly through narrative but would benefit from explicit callouts.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 3

### Verdict: NEEDS REVISION

The critical finding from Iteration 1 (missing receptive field visual) has been resolved with the ReceptiveFieldDiagram. All five planned misconceptions are now explicitly addressed. The FilterSwapExplorer and Vgg16BlockDiagram are meaningful additions. No critical findings remain, but two improvement items would strengthen the lesson noticeably.

### Findings

#### [IMPROVEMENT] — ReLU gradient description is misleading ("no vanishing" while gradient can be 0)

**Location:** AlexNet section, ReLU GradientCard (line 601-604)
**Issue:** The card states "Gradient is either 0 or 1 — no vanishing." This is internally contradictory. A gradient of exactly 0 is the most extreme form of vanishing. The student learned vanishing gradients at DEVELOPED depth and knows "gradient shrinks toward zero" is the problem. Telling them ReLU's gradient is "0 or 1" and then saying "no vanishing" invites the student to wonder: "Wait, 0 IS vanishing — how is that better?" The intended message is that ReLU does not *multiplicatively decay* the gradient through many layers the way sigmoid does (where 0.25^N compounds), but the phrasing conflates "no gradual decay" with "no zero gradients."
**Student impact:** A careful student who remembers the vanishing gradient mechanics would notice the contradiction. At best they ignore it; at worst they form an incorrect model that ReLU has no gradient issues at all, which will conflict if they later encounter dying ReLU. An ADHD learner may also snag on the inconsistency and lose momentum.
**Suggested fix:** Rephrase to: "For positive inputs, the gradient is exactly 1 — no shrinking, no compounding decay through layers. Unlike sigmoid's 0.25 multiplier at each layer, ReLU passes the gradient through unchanged." This preserves the core message while being technically accurate and connecting explicitly to the telephone game analogy.

#### [IMPROVEMENT] — FilterSwapExplorer renders regardless of selected architecture

**Location:** ArchitectureComparisonExplorer widget (line 521)
**Issue:** The `FilterSwapExplorer` component always renders inside the widget, even when the student has selected LeNet or AlexNet. The section is titled "What If VGG Used Larger Filters?" and operates on VGG block data. When the student is viewing LeNet's layers in the table above and sees a VGG-specific "what if" section below, the context is disconnected. The student may wonder why they're seeing VGG content while browsing LeNet.
**Student impact:** Minor disorientation. The student might not understand the relevance until they switch to VGG. For an ADHD learner, an unexplained context switch can break engagement flow.
**Suggested fix:** Conditionally render `FilterSwapExplorer` only when `selectedId === 'vgg16'`, or add a brief transition sentence when it appears below a non-VGG architecture: "The comparison below applies to VGG-16's block structure."

#### [POLISH] — ReceptiveFieldDiagram does not visually demonstrate overlap

**Location:** Receptive Field Equivalence section (line 807, ReceptiveFieldDiagram component)
**Issue:** The diagram shows three grids (5x5 input, 3x3 middle, 1x1 output) with dashed connection lines between them. While this correctly depicts the layer structure, it does not visually show *why* 3x3 applied to 3x3 covers a 5x5 region rather than a 9x9 region. The critical insight is that the nine 3x3 windows of the first conv *overlap* in the input, and this overlap is what makes the total coverage 5x5 instead of 9x9. The diagram highlights all cells uniformly without differentiating which input cells are covered by which middle positions or showing the overlapping regions.
**Student impact:** The text explanation and caption handle this adequately ("they overlap by 2 positions"), so the student will understand the concept. But the diagram functions more as a structural illustration ("here are the three layers") than as a proof of the RF expansion. It does not independently convey the overlap insight.
**Suggested fix:** Consider highlighting a specific middle cell (e.g., the center of the 3x3) and drawing lines to the 3x3 region of the input it covers, then highlighting an adjacent middle cell and showing how its 3x3 input region overlaps with the first. This would visually prove the 5x5 coverage. Alternatively, if this is too complex for a static SVG, the current version is acceptable since the text carries the explanation.

#### [POLISH] — Vgg16BlockDiagram FC block shows misleading spatial label

**Location:** Vgg16BlockDiagram component (line 265, 285)
**Issue:** The FC block displays "1x1" as its spatial dimension label and "1000ch" as its channel count. The diagram description says "width = spatial size." FC layers have no spatial dimensions — they operate on a flattened vector. Showing "1x1" with "1000ch" implies the FC layers are 1x1 convolutions with 1000 channels, which is a different operation. A student who later encounters actual 1x1 convolutions (common in Inception/ResNet bottleneck blocks) could be confused by this false association.
**Student impact:** Low impact in this lesson, but creates a potentially misleading precedent. The student might remember "VGG's FC was 1x1" and misapply that when they encounter actual 1x1 convolutions later.
**Suggested fix:** Change the FC block's spatial label from "1x1" to "flat" or remove the spatial annotation entirely for the FC block. Change "1000ch" to "1000 classes" to distinguish from channel counts.

#### [POLISH] — FC block height overflows container in Vgg16BlockDiagram

**Location:** Vgg16BlockDiagram component (line 290)
**Issue:** The FC block has `channels: 1000` and the height formula is `Math.max((block.channels / 512) * 72, 8)` which computes to approximately 140px. The container has `h-28` (112px). The FC block bar would exceed the container height. This is a cosmetic rendering issue — the block would overflow or be clipped depending on CSS overflow behavior.
**Student impact:** The diagram may look visually broken with the FC bar extending beyond its container. Minor visual distraction.
**Suggested fix:** Cap the FC block's channel value at 512 for height calculation purposes, or adjust the formula to normalize against the max channel count across all blocks. Alternatively, set `channels: 512` for the FC block since the height dimension is not meaningful for FC layers anyway.

### Review Notes

**What improved since Iteration 1:**
- The ReceptiveFieldDiagram resolves the critical finding. The visual shows the layer structure clearly, with color-coded grids and labeled stages. While it could be stronger at showing the overlap mechanism, it provides a genuine visual modality for the core concept that was entirely missing before.
- The FilterSwapExplorer adds meaningful interactivity to the widget. Students can now toggle VGG blocks between 3x3 stacking and equivalent large filters, seeing parameter increases and nonlinearity losses in real time. This directly addresses the "exploration-only" finding.
- The Vgg16BlockDiagram adds the proportional visual modality for VGG's architecture, making "spatial shrinks, channels grow" visceral rather than just textual.
- The vanishing gradient recap in the LeNet comprehension check (line 539-540) now includes the parenthetical explanation of the telephone game, reactivating the memory as suggested.
- All five planned misconceptions are now explicitly addressed with concrete callouts.
- RF annotations now say "after convs" (not "after block"), and RF=196 replaces the previous RF=212 error.

**What works well:**
- The narrative arc remains the lesson's strongest feature. The progression from "you built this" (LeNet) through problem-driven innovation (AlexNet) to principled design (VGG) is engaging and well-motivated.
- The parameter comparison (18C^2 vs 25C^2) is concrete, verifiable, and placed at the right moment.
- The two comprehension checks (predict-and-verify and transfer question) are appropriately challenging and well-positioned.
- The degradation problem cliffhanger is well-crafted and cleanly sets up the ResNets lesson.
- Em dashes correctly formatted throughout (no spaces in student-facing text).
- All interactive elements have cursor-pointer.
- The lesson stays within its stated scope boundaries.

**Remaining concern:**
- The two IMPROVEMENT findings are both in the widget area. The ReLU gradient description is a content accuracy issue in the lesson prose that is worth fixing before shipping. The FilterSwapExplorer rendering condition is a minor UX issue. Neither blocks learning, but the ReLU description could create a subtle misconception.

---

## Review — 2026-02-09 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All critical and improvement findings from Iterations 1 and 2 have been resolved. The two remaining items are minor polish-level observations that do not affect the student's learning experience.

### Iteration 2 Resolution Check

Before new findings, verifying all prior findings are resolved:

| Iteration 2 Finding | Status | Evidence |
|---------------------|--------|----------|
| [IMPROVEMENT] ReLU gradient description misleading | RESOLVED | Line 603-608: Now reads "For positive inputs, the gradient is exactly 1 — no shrinking, no compounding decay through layers. Unlike sigmoid's 0.25 multiplier at each layer, ReLU passes the gradient through unchanged." Technically accurate, connects to telephone game without the "0 or 1 — no vanishing" contradiction. |
| [IMPROVEMENT] FilterSwapExplorer renders regardless of selected architecture | RESOLVED | Widget line 521: `{selectedId === 'vgg16' && <FilterSwapExplorer />}` — conditionally rendered only when VGG-16 is selected. |
| [POLISH] ReceptiveFieldDiagram does not visually demonstrate overlap | ACCEPTED (as-is) | The diagram with its caption ("Each of those 9 positions sees a 3x3 input region. With overlap, the total input coverage is 5x5.") plus the text explanation provides adequate coverage. Enhancing the overlap visualization was optional per the review note. |
| [POLISH] Vgg16BlockDiagram FC block misleading labels | RESOLVED | Line 285: FC shows "flat" (not "1x1"). Line 296: FC shows "1000 classes" (not "1000ch"). Line 265: channels set to 512 so height formula stays within container. |
| [POLISH] FC block height overflow | RESOLVED | FC block has `channels: 512`, computing to 72px height, well within the `h-28` (112px) container. |

### Findings

#### [POLISH] — ReceptiveFieldDiagram SVG key prop uses non-unique pattern

**Location:** ReceptiveFieldDiagram component, line 123
**Issue:** The `key` prop for grid cells is `${x}-${y}-${fill}`. Since all input cells share the same fill color and all middle cells share the same fill color, the key is effectively `${x}-${y}` within each grid. This works correctly because x and y positions are unique within each grid, and the cells are rendered in separate arrays (`inputCells` and `midCells`). However, the inclusion of `fill` in the key pattern suggests the intent was to disambiguate cells with the same position but different fills, which is not actually needed here. This is a minor code clarity issue.
**Student impact:** None. React rendering works correctly.
**Suggested fix:** Simplify to `key={`cell-${r}-${c}`}` within each grid loop, or leave as-is since it functions correctly.

#### [POLISH] — VGG-16 StageRow RF annotation says "RF=5 after convs" for Block 1 but widget shows RF=5 at second conv

**Location:** VGG-16 architecture diagram, line 922
**Issue:** The StageRow for Block 1 reads `"Block 1: 2x Conv(64, 3x3)"` with annotation `"RF=5 after convs"` on a single row. In the widget, the two convolutions are shown separately: first conv has RF=3, second has RF=5. The lesson's compressed single-row representation correctly states the final RF after both convs (5), but does not show the intermediate RF=3 after the first conv. This is a presentation choice (space vs detail), not an error — the RF=5 value is accurate for after both convs in the block.
**Student impact:** Negligible. The student sees RF=5 on the single-row version and can discover the per-layer breakdown in the widget. No confusion risk since the annotation says "after convs" (plural).
**Suggested fix:** No change needed. The compressed representation is appropriate for an architecture overview, and the widget provides the per-layer detail.

### Review Notes

**What was fixed since Iteration 2:**
- The ReLU gradient description is now technically accurate and pedagogically sound. It avoids the "0 or 1" framing entirely, instead contrasting "gradient is exactly 1" (for positive inputs) with sigmoid's "0.25 multiplier at each layer." This connects directly to the telephone game analogy and does not invite confusion about dying ReLU or zero gradients.
- The FilterSwapExplorer now renders only when VGG-16 is selected, eliminating the context-switch disorientation for students browsing LeNet or AlexNet.
- The Vgg16BlockDiagram FC block now shows "flat" and "1000 classes," preventing any false association with 1x1 convolutions. The height overflow is fixed.

**What works well (full lesson assessment):**
- **Narrative arc:** The progression from "you built a LeNet" (MNIST CNN) through AlexNet's problem-driven innovations to VGG's principled design philosophy is engaging and well-motivated. The student is anchored in their own experience throughout.
- **Hook:** The MNIST vs ImageNet ComparisonRow makes the scale gap visceral. The bridge ("you basically built a 1998 architecture") is effective for reducing intimidation.
- **RGB recap:** Well-scoped gap resolution. Just enough to fill the prerequisite without overteaching.
- **Misconception coverage:** All 5 planned misconceptions are explicitly addressed: (1) GPUs misconception in WarningBlock, (2) same RF = same computation in WarningBlock, (3) deeper always better as degradation cliffhanger, (4) evolution as "more of the same" explicitly called out in Pattern of Innovation section, (5) "too complex" addressed with the "only two building blocks" framing.
- **Core concept delivery:** The 3x3 stacking insight is taught through 5 distinct modalities: verbal explanation, ReceptiveFieldDiagram (visual), RF formula (symbolic), parameter comparison table (concrete numerical), and FilterSwapExplorer (interactive). This exceeds the 3-modality minimum.
- **Comprehension checks:** Both the predict-and-verify (Why mixed filter sizes?) and transfer question (7x7 replacement) are well-placed and appropriately challenging.
- **Interactive widget:** The ArchitectureComparisonExplorer provides genuine exploration value with the layer table, metrics bar, comparison sidebar, and the VGG-specific FilterSwapExplorer toggle. The TryThisBlock experiments guide discovery without being prescriptive.
- **Degradation cliffhanger:** The distinction between degradation and overfitting is clearly drawn (training error, not just test error), setting up the ResNets lesson effectively.
- **Writing quality:** Em dashes correctly formatted throughout (no spaces). Cursor styles present on all interactive elements. Lesson stays within stated scope boundaries.
- **Ordering principles:** Concrete before abstract (LeNet comparison before VGG theory), problem before solution (why depth? before how), simple before complex (LeNet before AlexNet before VGG), familiar before unfamiliar (student's own CNN as the starting point).
- **Cognitive load:** Three new concepts (quantitative RF, parameter efficiency, architecture evolution as narrative), well within the 3-concept limit. Each is heavily supported by existing knowledge at DEVELOPED+ depth.

**Overall:** This lesson is ready to ship. The narrative is strong, the core concepts are well-taught through multiple modalities, all planned misconceptions are addressed, and the two remaining polish items are cosmetic code-level observations with zero student impact.

---

## What Was Actually Built

Implementation closely followed the design with these notable additions and minor deviations:

### Additions (not in original design)
- **ReceptiveFieldDiagram (inline SVG):** Added after Review Iteration 1 flagged missing visual modality for the core concept. Static SVG showing 5x5 input -> 3x3 intermediate -> 1x1 output with color-coded grids and connection lines.
- **FilterSwapExplorer:** Added after Review Iteration 1 flagged the widget as exploration-only. Lets student toggle VGG blocks between 3x3 stacking and equivalent large filters, showing parameter increase and nonlinearity loss in real time. Renders only when VGG-16 is selected.
- **Vgg16BlockDiagram:** Proportional block diagram showing VGG-16's spatial/channel tradeoff visually. Blocks shrink in width (spatial) and grow in height (channels).
- **Explicit misconception callouts:** All 5 planned misconceptions are now explicitly addressed with WarningBlocks or inline callouts (Iterations 1-2 found 2 were only implicitly addressed).

### Deviations from design
- **No separate architecture diagrams:** The plan called for "side-by-side architecture diagrams." The built lesson uses StageRow text-based format with color-coded dots for all three architectures, plus the Vgg16BlockDiagram for VGG-16 specifically. Full visual architecture diagrams were replaced by the ArchitectureComparisonExplorer widget which serves the comparison purpose more interactively.
- **RGB recap slightly shorter than planned:** Design called for 2-3 paragraphs + visual. Built as 3 paragraphs + dimension example (StageRow) but no separate RGB visual diagram. The text explanation was sufficient given the student's existing Conv2d knowledge.
- **No famous training accuracy curve for degradation problem:** Design mentioned "Show the famous training accuracy curve (deeper network with higher training error)." Built as a descriptive callout box with text explanation rather than a chart. The cliffhanger works well without the chart; the chart would be better placed in the resnets lesson where the degradation problem is fully developed.

### Depth targets achieved
- LeNet architecture: INTRODUCED (as planned)
- AlexNet innovations: INTRODUCED (as planned)
- VGG architecture and 3x3 philosophy: DEVELOPED (as planned)
- Receptive field (quantitative for stacked convs): DEVELOPED (as planned)
- Parameter efficiency of small filters: DEVELOPED (as planned)
- RGB / multi-channel input: INTRODUCED (gap resolution, as planned)
- Degradation problem: MENTIONED (cliffhanger, as planned)
