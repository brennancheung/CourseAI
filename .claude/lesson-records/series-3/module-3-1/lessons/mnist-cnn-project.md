# Lesson Plan: MNIST CNN Project

**Series:** 3 (CNNs) | **Module:** 3.1 (Convolutions) | **Lesson:** 3 of 3
**Slug:** `mnist-cnn-project`
**Cognitive load type:** CONSOLIDATE

---

## Phase 1: Orient -- Student State

The student has completed both prior lessons in this module: what-convolutions-compute (STRETCH) and building-a-cnn (BUILD). They also completed all of Series 1 (Foundations, 17 lessons) and are assumed to have completed Series 2 (PyTorch), including the MNIST dense network project. This is a CONSOLIDATE lesson -- no new concepts are introduced. The student assembles everything from Lessons 1 and 2 into a working CNN, trains it on MNIST, and sees it outperform the dense network they built in Series 2. This is the payoff for the entire module.

### Relevant Concepts the Student Has

| Concept | Depth | Source | How It's Relevant Here |
|---------|-------|--------|----------------------|
| Convolution as sliding filter (multiply-and-sum over local region) | DEVELOPED | what-convolutions-compute (3.1 L1) | The core operation inside nn.Conv2d layers. Student computed by hand and interacted with the ConvolutionExplorer widget. |
| Feature map (output of convolution) | DEVELOPED | what-convolutions-compute (3.1 L1) | Each Conv2d layer produces feature maps. Student understands these as spatial pattern-detection output. |
| Weight sharing and spatial locality | INTRODUCED | what-convolutions-compute (3.1 L1) | Why CNNs use fewer parameters than dense networks for images. Directly relevant to the parameter count comparison. |
| Max pooling (take max of each region to shrink spatial dimensions) | DEVELOPED | building-a-cnn (3.1 L2) | Used in the CNN architecture. Student can compute it and explain why it preserves feature presence. |
| Stride and padding | DEVELOPED | building-a-cnn (3.1 L2) | Engineering knobs for controlling output dimensions. Student can compute output sizes with the general formula. |
| General output size formula: floor((N - F + 2P) / S) + 1 | DEVELOPED | building-a-cnn (3.1 L2) | Student can verify dimensions at each layer of the CNN they will build. |
| Conv-pool-fc architecture pattern | DEVELOPED | building-a-cnn (3.1 L2) | The architecture this lesson implements. Student traced 28x28x1 through conv-pool-conv-pool-flatten-fc-fc to 10 outputs. |
| nn.Conv2d / nn.MaxPool2d API | INTRODUCED | building-a-cnn (3.1 L2) | Student can read the API and identify parameters. This lesson requires writing it, which is a depth upgrade from INTRODUCED to APPLIED. |
| Flatten transition (spatial grid to flat vector) | INTRODUCED | building-a-cnn (3.1 L2) | The 7x7x64 -> 3136 flatten step. Student understands why spatial structure ends here. |
| Training loop (forward-loss-backward-update) | DEVELOPED | module 1.1, series-2 | Same training loop applies to CNNs. No changes to the training algorithm. |
| nn.Module, forward(), parameters | DEVELOPED (assumed) | series-2 nn-module | Student can write PyTorch modules. Required for implementing the CNN class. |
| Optimizer (Adam) | DEVELOPED | module 1.3 optimizers + series-2 | Student has used optim.Adam in training loops. |
| DataLoader, batching, epochs | DEVELOPED (assumed) | series-2 datasets-and-dataloaders + module 1.3 batching-and-sgd | Student has loaded MNIST data and iterated through batches. |
| MNIST dense network | APPLIED (assumed) | series-2 mnist-project | The baseline this CNN will beat. Student built a dense network achieving ~97% accuracy on MNIST. |
| Overfitting / training curves | DEVELOPED | module 1.3 overfitting-and-regularization | Student can read train vs val loss divergence. Relevant for evaluating the CNN's training. |
| Dropout | DEVELOPED | module 1.3 overfitting-and-regularization | Available as a regularization technique in the CNN, but not the focus of this lesson. |

### Mental Models Already Established

- "A filter is a pattern detector -- it asks 'does this local region look like my pattern?' at every position"
- "The feature map is a spatial answer key -- output position (i,j) tells you about the input neighborhood at (i,j)"
- "A CNN is a series of zoom-outs" -- each conv detects patterns at the current scale, each pool zooms out
- "Spatial shrinks, channels grow, then flatten" -- the CNN data shape signature
- "Pooling preserves feature presence, not exact position"
- "Same weighted sum as a neuron, but only over a local neighborhood"
- "Parameters are knobs the model learns"
- "Training loop = forward -> loss -> backward -> update" (unchanged for CNNs)
- "requires_grad = press Record, backward() = press Rewind" (autograd mental model)

### What Was NOT Covered

- Actually training a CNN (this lesson's content)
- Comparing CNN accuracy vs dense network accuracy (this lesson's payoff)
- Architecture design choices -- how many layers, how many filters, filter size tradeoffs (touched on lightly here)
- Multiple input channels / RGB images (out of scope for this module)
- Backprop through conv or pooling layers (relies on existing backprop + autograd knowledge)
- Batch normalization in CNNs (out of scope)
- Advanced architectures (ResNet, VGG) beyond brief mentions

### Readiness Assessment

The student is fully prepared. Every concept needed for this lesson has been taught to at least INTRODUCED depth, and most are at DEVELOPED. The student has:

1. Computed convolutions by hand and explored them interactively (L1)
2. Traced data shapes through a complete CNN architecture with specific numbers (L2)
3. Read nn.Conv2d and nn.MaxPool2d API code (L2)
4. Built and trained a dense MNIST network in PyTorch (Series 2)
5. Used the CnnDimensionCalculator widget to build architectures and verify shapes (L2)

The gap between "reading API code" (INTRODUCED, L2) and "writing a CNN class" (APPLIED, this lesson) is small because the student has written nn.Module subclasses before (Series 2) and has seen the exact layer sequence they will implement. The architecture is not new -- they traced it step by step in L2. The training loop is not new -- they used it in Series 2. This lesson connects two things the student already knows: "the CNN architecture from L2" and "the PyTorch training workflow from Series 2."

**Activation energy concern:** This is a project lesson, which can feel daunting ("now I have to do it all myself"). The opening should frame it as assembly, not invention: "You already know every piece. Today you put them together." The Colab notebook should be scaffolded so the student is never staring at a blank cell.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to build, train, and evaluate a CNN for MNIST in PyTorch, seeing the accuracy improvement over a dense network and understanding that the improvement comes from architecture, not more parameters.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Conv-pool-fc architecture pattern | DEVELOPED | DEVELOPED | building-a-cnn (3.1 L2) | OK | Student must know the architecture they are implementing. They traced it with exact shapes in L2. |
| nn.Conv2d / nn.MaxPool2d API | APPLIED | INTRODUCED | building-a-cnn (3.1 L2) | GAP | Student can read the API (INTRODUCED) but this lesson requires writing it (APPLIED). Gap is small -- bridged by scaffolded code in the notebook. |
| nn.Module, forward() | APPLIED | DEVELOPED (assumed) | series-2 nn-module | OK | Student has written nn.Module subclasses. Writing a CNN module is the same pattern with different layer types. |
| Training loop (DataLoader, optimizer, loss, backward) | APPLIED | APPLIED (assumed) | series-2 training-loop | OK | Student has written complete training loops in PyTorch. Identical for CNNs. |
| MNIST data loading | APPLIED | APPLIED (assumed) | series-2 mnist-project | OK | Student has loaded MNIST with torchvision. Same code reused here. |
| Convolution operation (intuition) | DEVELOPED | DEVELOPED | what-convolutions-compute (3.1 L1) | OK | Student understands what the conv layers are doing. No re-teaching needed. |
| Pooling (max pooling) | DEVELOPED | DEVELOPED | building-a-cnn (3.1 L2) | OK | Student can explain what pooling does and compute it. |
| Output size formula | DEVELOPED | DEVELOPED | building-a-cnn (3.1 L2) | OK | Student can verify dimensions at each layer. Used for the dimension annotation exercise. |
| Flatten transition | INTRODUCED | INTRODUCED | building-a-cnn (3.1 L2) | OK | Student understands the spatial-to-flat transition. Only need recognition level here. |
| Overfitting / training curves | DEVELOPED | DEVELOPED | module 1.3 overfitting-and-regularization | OK | Student can read train/val curves to evaluate model performance. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| nn.Conv2d / nn.MaxPool2d: INTRODUCED -> APPLIED | Small | The student can already read the API. The gap is from reading to writing. Resolution: provide the architecture in scaffolded form in the Colab notebook -- the student fills in specific parameters (kernel_size, padding, out_channels) rather than writing from scratch. A brief "API Recap" section (3-4 sentences + one code block) maps the arguments to the dimension formula they already know. This bridges the gap without a dedicated section. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The CNN is better because it has more parameters" | More parameters = more capacity = better accuracy (learned in Foundations). The CNN might seem "bigger." | Count the parameters: dense network for MNIST has ~120K parameters (784*128 + 128*64 + 64*10). The CNN has ~60K parameters (32*9 + 64*32*9 + 3136*128 + 128*10). The CNN has FEWER parameters yet achieves higher accuracy. The improvement comes from architecture (locality + weight sharing), not capacity. | During the comparison section, immediately after showing both accuracy numbers. Make the parameter count concrete and surprising. |
| "The CNN just needs more training time / epochs to do well" | If both networks had enough time, they would converge to the same accuracy. The CNN is faster but not fundamentally better. | Train both for the same number of epochs (e.g., 5). The dense network plateaus around 97-98%. The CNN reaches 99%+. More training time does not close the gap -- the dense network cannot represent spatial features the way a CNN can. A 28x28 image shifted one pixel to the right is an entirely different input vector to a dense network but nearly identical to a CNN (weight sharing + pooling). | In the comparison section, after showing accuracy at the same epoch count. Frame as "it is not about patience -- it is about what the architecture can represent." |
| "I need a complex architecture to beat the dense network" | The student may think they need many layers, advanced techniques (batch norm, skip connections), or careful hyperparameter tuning. | A simple 2-conv-layer CNN (conv-pool-conv-pool-fc-fc) with default hyperparameters beats the dense network. No batch norm, no dropout, no learning rate scheduling. The architecture advantage is so large that a simple CNN dominates. | In the framing/hook. Set expectations: "The simplest possible CNN will beat your dense network. No tricks required." |
| "ReLU activations between conv layers are something new" | The student has used ReLU in dense networks but might think convolutions need different activation functions. | The same ReLU from Series 1 applies identically. Conv2d computes a linear operation (weighted sum); ReLU applies the same nonlinearity afterward. Nothing changes about activation functions -- they work on each value in the feature map independently. | Brief mention when building the model. One sentence: "Same ReLU you have used before -- applied to every value in the feature map." |
| "The flatten step loses important information" | Flattening collapses spatial structure into a 1D vector. The student learned in L1 that flattening images is the problem CNNs solve. Seeing flatten in the CNN may feel contradictory. | By the time we flatten, spatial dimensions are 7x7 -- a 49-position summary of the entire 28x28 image, each position representing an abstract feature, not a raw pixel. This is fundamentally different from flattening the raw 28x28 image into 784 raw pixel values. The CNN earned the right to flatten by extracting meaningful spatial features first. The dense network flattened pixels; the CNN flattens features. | In the architecture walkthrough, when the student reaches the flatten layer. Connect back to the "flat vector problem" hook from L1: "This is the same flatten -- but now each value represents a learned feature, not a pixel." |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Building the CNN class: conv-pool-conv-pool-flatten-fc-fc | Positive | The main deliverable. Student implements the exact architecture they traced in L2 with real PyTorch code. | Uses the same architecture and exact same dimensions (28x28x1 -> 28x28x32 -> 14x14x32 -> 14x14x64 -> 7x7x64 -> 3136 -> 128 -> 10) they traced in L2. No surprises -- the code implements the diagram they already understand. |
| Side-by-side accuracy comparison: dense ~97% vs CNN ~99%+ | Positive | The payoff. Student sees the CNN dramatically outperform their dense network on the same data, same training budget. | MNIST is familiar (they built the dense network on it). The accuracy gap is large enough to be unambiguous (2-3%) but both are high enough that the student feels pride in both results. The comparison is fair: same data, same epochs, same optimizer. |
| Parameter count comparison: dense ~120K vs CNN ~60K | Positive (surprise) | Disproves the "more parameters = better" misconception. The CNN wins with fewer parameters. | The numbers are concrete and countable. The student can verify them by summing up the layer sizes. The surprise ("fewer parameters AND better accuracy?") drives the key insight home. |
| The dense network on a shifted image (negative) | Negative | Shows WHY the dense network is worse. A digit shifted a few pixels is nearly identical to a human but a completely different input vector. The dense network has no mechanism to handle this. | This is the spatial invariance argument made concrete. The student saw the "flat vector problem" in L1 -- this is the quantitative proof. The CNN handles this via weight sharing and pooling, which they already understand. |

---

## Phase 3: Design

### Narrative Arc

You have spent two lessons understanding what convolutions compute and how to assemble them into a CNN architecture. You can trace data through a conv-pool-conv-pool-flatten-fc pipeline and explain what happens at every stage. But you have not yet built or trained one. In Series 2, you built a dense network for MNIST -- your first real PyTorch project. That network flattened each 28x28 image into a 784-element vector and got around 97% accuracy, which felt impressive. But you now know what flattening destroys: spatial relationships, neighborhood structure, the fact that a vertical edge at any position is the same feature. Today you build a CNN for MNIST using the exact architecture you traced in Lesson 2, train it with the same PyTorch workflow you already know, and compare it to your dense network. The result will be unambiguous: the CNN wins. And it wins with fewer parameters. The lesson is not about learning new concepts -- it is about seeing the payoff. Architecture matters. How you connect layers matters as much as how many neurons you have. That is the core insight of this entire module.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example (code) | The full CNN implementation in PyTorch: nn.Module subclass with Conv2d, ReLU, MaxPool2d, Flatten, Linear layers. The student writes (fills in) the code in a Colab notebook. | This is a project lesson -- code IS the primary modality. The student needs to go from "I understand the architecture" to "I can implement it." The code makes every concept from L1 and L2 tangible. |
| Visual (comparison) | Side-by-side accuracy curves: dense network vs CNN over the same training epochs. Training loss and validation accuracy for both models on the same plot. | The visual comparison is the payoff. Seeing the CNN curve pull ahead of the dense curve makes the architecture advantage visceral, not just intellectual. Training curves also reinforce the diagnostic skill from module 1.3. |
| Symbolic (parameter count) | Concrete parameter count arithmetic: dense = 784*128 + 128 + 128*64 + 64 + 64*10 + 10 = ~110K. CNN = 1*32*9 + 32 + 32*64*9 + 64 + 3136*128 + 128 + 128*10 + 10 = ~420K. (Note: actual counts depend on the specific architecture; the exact numbers will be computed in the notebook with `sum(p.numel() for p in model.parameters())`.) | Parameter count is the quantitative evidence that disproves the "more parameters" misconception. The student needs to see specific numbers, not hand-waving. The code-computed count is more trustworthy than manual arithmetic. |
| Verbal/Analogy | "The dense network flattens pixels; the CNN flattens features." "Architecture encodes assumptions about data -- the CNN assumes spatial structure exists, and MNIST proves it right." | These verbal framings compress the lesson into memorable one-liners that capture the core insight. They connect to established mental models (the "flat vector problem" from L1, "parameters are knobs" from Series 1). |
| Concrete example (negative) | Shifting a digit image by 2 pixels and showing the dense network's confusion vs the CNN's resilience. Not a formal experiment -- a conceptual demonstration with an explanation of why (weight sharing means the CNN uses the same filter everywhere; the dense network treats shifted pixels as entirely new features). | This makes the spatial invariance advantage concrete and personal. The student's own dense network has this weakness. The CNN they just built does not. |

### Cognitive Load Assessment

- **New concepts in this lesson:** Zero. This is a CONSOLIDATE lesson. Every concept used here has been taught in prior lessons. The "new" skill is assembly -- combining nn.Conv2d (INTRODUCED in L2) with nn.Module patterns (DEVELOPED in Series 2) and the training loop (APPLIED in Series 2). Writing nn.Conv2d code is a depth upgrade from INTRODUCED to APPLIED, not a new concept.
- **What was the load of the previous lesson?** BUILD (building-a-cnn). The student learned pooling, stride, padding, and the conv-pool-fc pattern.
- **Is this lesson's load appropriate?** Yes. CONSOLIDATE after BUILD is the planned trajectory (STRETCH-BUILD-CONSOLIDATE). The student should feel: "I know all of this -- now I am putting it together." Activation energy is low because the architecture is familiar, the training loop is familiar, and the data is familiar. The only cognitive effort is connecting pieces they already have.

### Connections to Prior Concepts

| New Application | Prior Concept | Connection |
|-----------------|--------------|------------|
| Writing nn.Conv2d with specific arguments | Reading nn.Conv2d API (L2, INTRODUCED) | "In Lesson 2, you read Conv2d code and identified the arguments. Now you write it. Same arguments, same formula for output size." |
| CNN training loop | Dense network training loop (Series 2) | "The training loop is identical. Forward pass, compute loss, backward, optimizer step. The only difference is what happens inside model(x) -- and that is the architecture you traced in Lesson 2." |
| CNN on MNIST | Dense network on MNIST (Series 2) | "Same data, same task, same evaluation metric. The only change is the model architecture. This is a controlled experiment -- architecture is the independent variable." |
| Parameter count comparison | Weights as learnable parameters (Series 1) | "You know that each parameter is a number the model adjusts during training. Now count them: the CNN has fewer, yet does better. Weight sharing is why." |
| Flatten in the CNN | Flatten as 'the problem' (L1 hook) | "The flat vector problem from Lesson 1 was about flattening raw pixels. The flatten here happens after two rounds of convolution and pooling -- you are flattening abstract features, not raw data." |

**Potentially misleading prior analogies:**
- The "flat vector problem" hook from L1 framed flattening as destructive. The student might be confused by seeing flatten in the CNN. Explicitly address this: flattening features is not the same as flattening pixels.
- The "CNN is a series of zoom-outs" analogy from L2 applies to the architecture but not to training. Do not extend it to the training process.

### Scope Boundaries

**This lesson IS about:**
- Implementing the CNN architecture from L2 in PyTorch code (filling in a scaffolded Colab notebook)
- Training the CNN on MNIST using the standard PyTorch training loop
- Comparing CNN accuracy to dense network accuracy on the same data and training budget
- Comparing parameter counts to show the CNN wins with fewer parameters
- Understanding WHY the CNN wins: architecture encodes spatial assumptions that match the data

**This lesson is NOT about:**
- Introducing new concepts (no new operations, layers, or techniques)
- Hyperparameter tuning or architecture search
- Batch normalization, dropout, or other regularization in the CNN context
- Multiple input channels / RGB images
- Advanced architectures (ResNet, VGG, etc.)
- Data augmentation
- Learning rate scheduling
- The mechanics of backprop through conv/pool layers
- Deployment, inference optimization, or model export

**Target depths (upgrades from prior lessons):**
- nn.Conv2d / nn.MaxPool2d: INTRODUCED -> APPLIED (student writes the code, not just reads it)
- Conv-pool-fc architecture: DEVELOPED -> APPLIED (student implements it end-to-end)
- CNN vs dense comparison: NEW at DEVELOPED (student understands why architecture matters, with evidence)

### Lesson Outline

**1. Context + Constraints**
What this lesson is about: building and training a CNN for MNIST, then comparing it to the dense network from Series 2. What we are NOT doing: introducing new concepts, tuning hyperparameters, or using advanced techniques. Frame: "You already know every piece. This lesson puts them together."

**2. Recap -- The Architecture You Will Build**
Brief visual recap of the architecture from L2: 28x28x1 -> Conv(32, 3x3, pad=1) -> ReLU -> MaxPool(2x2) -> Conv(64, 3x3, pad=1) -> ReLU -> MaxPool(2x2) -> Flatten -> FC(128) -> ReLU -> FC(10). Annotate with shapes at each stage. This is a 30-second refresher, not a re-teach. Purpose: prime the student so the code they are about to write maps directly to this diagram.

**3. Hook -- The Challenge (before/after preview)**
Type: Challenge preview. "Your dense network from Series 2 got ~97% accuracy on MNIST. Can this CNN do better? And if it can -- why? It uses the same data, the same optimizer, the same number of training epochs. The only thing that changes is the architecture. Let us find out."

This hook works because:
- It connects to something the student built and is proud of (their MNIST dense network)
- It sets up a controlled experiment (architecture as the only variable)
- It creates genuine curiosity ("can it do better? why?")
- It frames the lesson as a test of the ideas from L1 and L2, not new content

**4. Explain -- API Recap (bridge from INTRODUCED to APPLIED)**
3-4 sentences + one annotated code block. Map nn.Conv2d arguments to the dimension formula:
- `nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)` -- "1 input channel (grayscale), 32 filters, 3x3 filter size, padding=1 to preserve spatial dimensions"
- `nn.MaxPool2d(kernel_size=2)` -- "2x2 window, stride defaults to kernel_size (2), halves spatial dimensions"
- `nn.Flatten()` -- "collapses spatial dimensions into a flat vector"

This is not new -- it is the reading comprehension from L2 with one additional step: "now you will type these lines yourself."

**5. Explore -- Build the CNN (Colab Notebook, guided)**
The student works in a Colab notebook. The notebook is scaffolded:
- Data loading code is provided (same MNIST loading from Series 2)
- The CNN class skeleton is provided with `TODO` comments where the student fills in layers
- A dimension annotation exercise: after writing each layer, predict the output shape and verify with a test forward pass
- The training loop is provided (identical to their Series 2 training loop)
- An evaluation cell computes test accuracy

The student's task: fill in the `__init__` and `forward` methods of the CNN class. This is the right level of scaffolding for a CONSOLIDATE lesson -- the student focuses on the CNN-specific code without boilerplate friction.

**6. Check 1 -- Dimension Verification**
Type: predict-and-verify. Before running the training loop, the student traces through their model:
- "What is the shape after the first Conv2d? After the first MaxPool2d? After Flatten? After the final Linear?"
- They verify by running `model(torch.randn(1, 1, 28, 28))` and checking intermediate shapes (using hooks or print statements in forward()).
- This connects directly to the dimension-tracking skill from L2 and ensures the architecture is correct before training.

**7. Explore -- Train and Compare**
The student trains the CNN (the training loop is provided). Then they compare:
- **Accuracy:** Dense ~97% vs CNN ~99%+ on the test set
- **Parameter count:** `sum(p.numel() for p in model.parameters())` for both models
- **Training curves:** Loss vs epoch for both models (plotting code provided)

This is the payoff moment. The comparison must be concrete and fair:
- Same data (MNIST)
- Same number of epochs (5)
- Same optimizer (Adam, same learning rate)
- Same loss function (cross-entropy)

**8. Check 2 -- Explain the Difference**
Type: explain-it-back. The student answers: "The CNN has [fewer/more] parameters than the dense network, yet achieves [higher/lower] accuracy. Why?"

Expected answer touches on: spatial locality (filters see neighborhoods, not all 784 pixels), weight sharing (same filter everywhere, position-invariant), feature hierarchy (edges -> shapes -> digits through conv-pool stages), and the dense network treats shifted digits as entirely different inputs.

**9. Elaborate -- Why Architecture Matters**
- **The shifting experiment (conceptual):** Take a test digit, shift it 2 pixels right. The 784-element flat vector changes at every position (every pixel shifts). To the dense network, this is a drastically different input. To the CNN, the same filters detect the same features at slightly different positions, and pooling absorbs the shift. Weight sharing and pooling together give the CNN spatial invariance that the dense network cannot have.
- **Parameter count surprise:** Walk through the arithmetic. The dense network's first layer alone (784 * 128 = 100K parameters) is larger than the CNN's entire conv stack. Weight sharing is the reason: 32 filters of 3x3 = 288 parameters detect features everywhere. The dense network needs separate weights for every spatial position.
- **The core insight:** "Architecture encodes assumptions about data. The CNN assumes spatial structure exists -- that nearby pixels are related and that the same patterns appear at different positions. MNIST proves this assumption correct. The dense network makes no such assumption and must learn spatial relationships from scratch, wasting parameters on what the CNN gets for free."

**10. Summarize**
Key takeaways:
- A simple CNN (2 conv layers, 2 pool layers, 2 FC layers) beats a dense network on MNIST
- The CNN wins with fewer parameters -- weight sharing means less to learn
- The training loop is identical -- only the architecture changed
- Architecture encodes assumptions about data; matching architecture to data structure is the key insight
- "The dense network flattens pixels; the CNN flattens features"

Echo the module arc: "You started this module by learning what a single convolution computes. Then you assembled convolutions into an architecture with pooling, stride, and padding. Now you have built one, trained it, and proved that architecture matters. That is the foundation for everything that follows in CNNs."

**11. Next Step**
"You have built your first CNN and seen it outperform a dense network. The natural question: can we go deeper? Better architectures have pushed CNN accuracy much further -- from LeNet's 99% on digits to human-level performance on real photographs. Next, we explore how CNN architectures evolved and what architectural innovations made deeper, more powerful networks possible."

### Notebook Specification

**Filename:** `3-1-3-mnist-cnn-project.ipynb`

**Structure:**
1. **Setup cell:** Imports (torch, torchvision, matplotlib). Device setup.
2. **Data loading cell:** MNIST download, transforms (ToTensor, Normalize), DataLoaders. Provided complete -- identical to Series 2.
3. **Dense network baseline cell:** The student's dense network from Series 2, provided as reference. Pre-trained results loaded or re-trained quickly (1-2 minutes).
4. **CNN class cell:** Skeleton with TODO comments. Student fills in layers in `__init__` and the forward pass in `forward()`.
5. **Dimension verification cell:** Test forward pass with random input, print shapes at each stage.
6. **Training loop cell:** Provided complete. Trains for 5 epochs. Prints loss and accuracy per epoch.
7. **Evaluation cell:** Test accuracy computation. Side-by-side comparison print.
8. **Parameter count cell:** Count parameters for both models. Print comparison.
9. **Training curves cell:** Plot loss curves for both models on the same axes.
10. **Reflection cell:** Markdown cell with the "explain the difference" prompt.

**Scaffolding level:** Guided. The student writes the CNN architecture (the novel part) but gets the data loading, training loop, and evaluation for free (the familiar parts). This is appropriate for CONSOLIDATE -- the student should spend cognitive effort on the CNN, not on boilerplate they have already written.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (module 3.1 record, module 2.1 record, module 1.3 record)
- [x] Depth match verified for each
- [x] No untaught concepts remain (nn.Conv2d gap is small and has resolution plan)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (one small gap: API recap section)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 0 new concepts (CONSOLIDATE)
- [x] Every application connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings that would leave the student lost, but one critical issue (the Colab notebook does not exist) makes the lesson uncompletable in its current form, and four improvement findings would make the lesson significantly stronger if addressed.

### Findings

#### [CRITICAL] — Colab notebook does not exist

**Location:** Section 5 (Build the CNN), Colab link
**Issue:** The lesson links to `https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/3-1-3-mnist-cnn-project.ipynb` but this notebook file does not exist in the repository. The entire hands-on portion of the lesson depends on this notebook. Without it, the student cannot do the project.
**Student impact:** The student clicks the "Open in Google Colab" link and gets a 404 or error page. The lesson becomes purely reading with no implementation. For a CONSOLIDATE/project lesson, this is a showstopper -- the entire point is assembly and practice.
**Suggested fix:** Create the notebook `notebooks/3-1-3-mnist-cnn-project.ipynb` following the notebook specification in Phase 3 of the planning document (10 cells: setup, data loading, dense baseline, CNN skeleton with TODOs, dimension verification, training loop, evaluation, parameter count, training curves, reflection).

#### [IMPROVEMENT] — Missing visual modality for the comparison

**Location:** Section 7 (Train and Compare)
**Issue:** The planning document specifies "Visual (comparison): Side-by-side accuracy curves: dense network vs CNN over the same training epochs. Training loss and validation accuracy for both models on the same plot." The built lesson only includes a static `ComparisonRow` component with bullet points and a code block for parameter counting. There is no training curve visualization in the lesson itself. The planning doc says this visual "is the payoff" and "makes the architecture advantage visceral." The notebook presumably generates these plots, but the lesson page itself has no visual representation of the training curves.
**Student impact:** The student reads about the comparison but does not see it in the lesson material. They would only see it in the Colab notebook (if it existed). The lesson page itself -- which is the primary reading experience -- lacks this powerful visual. The ComparisonRow is informative but static and textual. A training curve showing the CNN pulling ahead of the dense network would be far more compelling.
**Suggested fix:** Either (a) add a static image/figure showing representative training curves for both models (even a stylized illustration would work), or (b) accept that this modality lives in the notebook and explicitly tell the student to look at the training curve plot they generated ("Look at the training curves you plotted in the notebook -- the CNN's loss drops faster and its accuracy plateaus higher"). Option (b) is lighter and appropriate for a CONSOLIDATE lesson where the notebook is the primary artifact.

#### [IMPROVEMENT] — "ReLU is something new" misconception not clearly addressed as a misconception

**Location:** Section 4 (API Recap), aside TipBlock
**Issue:** The planning document identifies "ReLU activations between conv layers are something new" as a misconception. The lesson addresses it in a TipBlock aside ("Same ReLU you have used since Foundations..."). However, the main body text never mentions ReLU at all in the API Recap section. The aside is supplementary content that the student may not read carefully. The misconception resolution is entirely in the margin, not in the main flow.
**Student impact:** A student who skims the main body and skips the aside could arrive at the CNN skeleton code with `# Conv1 -> ReLU -> Pool1` comments and wonder whether ReLU works differently in a CNN context. The aside placement is adequate for a minor misconception in a CONSOLIDATE lesson, but the main text in the API Recap section discusses Conv2d, MaxPool2d, and Flatten without even mentioning ReLU. A single sentence in the main body would close this gap.
**Suggested fix:** Add one sentence to the main body of the API Recap section, e.g., after the code block: "ReLU is the same activation function you have used before -- it applies independently to each value in the feature map, adding nonlinearity after the linear convolution."

#### [IMPROVEMENT] — Parameter count numbers inconsistent with planning document

**Location:** Section 7 (ComparisonRow) and Section 9 (parameter arithmetic)
**Issue:** The planning document states "dense network for MNIST has ~120K parameters" and later "~110K" in different places. The built lesson uses "~110K" for the dense network and "~60K" for the CNN. However, when you add up the CNN parameter arithmetic shown in the lesson (conv stack: 18,816 + FC layers: 3136*128 + 128 + 128*10 + 10 = 402,506 + 18,816 = ~421K), the total CNN parameter count is actually much higher than 60K. The bulk of the CNN's parameters are in the FC layers (specifically the 3136 -> 128 layer = 401,408 parameters), which the lesson's arithmetic section conspicuously omits. The lesson only shows the conv stack parameters (18,816) and compares them to the dense network's first layer (100,352), which is a valid and interesting comparison, but then the ComparisonRow claims "Total: ~60K parameters" for the CNN, which is wrong.
**Student impact:** A student who does the math will notice the discrepancy. The 3136 -> 128 FC layer alone has ~400K parameters. The CNN total is approximately 421K, not 60K. If the student verifies with `sum(p.numel() for p in model.parameters())` in the notebook, they will get a number much larger than 60K, contradicting the lesson text. This undermines trust and creates confusion about the "fewer parameters" claim. The CNN actually has MORE total parameters than the dense network -- the parameter efficiency story is specifically about the conv layers, not the whole model.
**Suggested fix:** Fix the numbers to be accurate. The CNN with this architecture (Conv(1->32,3,pad=1), Pool, Conv(32->64,3,pad=1), Pool, FC(3136->128), FC(128->10)) has approximately 421K total parameters, NOT 60K. The interesting comparison is between how each model processes the spatial information: the dense network's first layer alone (100K params) does what the CNN's entire conv stack (18.8K params) does more effectively. Reframe the comparison: the CNN's convolutional feature extraction is dramatically more parameter-efficient than the dense approach, even though the FC classifier layers add parameters. Alternatively, use a smaller FC layer (e.g., 3136 -> 64 -> 10) to actually make the total parameter count lower, and update the architecture throughout. The key insight ("weight sharing = efficiency in the feature extraction stage") is valid and powerful -- it just needs accurate numbers to support it.

#### [IMPROVEMENT] — The "shifting experiment" is purely verbal with no visual or concrete aid

**Location:** Section 9 (Why Architecture Matters)
**Issue:** The planning document describes a "shifting experiment (conceptual)" where the student imagines shifting a digit 2 pixels right and seeing how the flat vector changes vs. how the CNN handles it. The built lesson describes this purely in prose (three paragraphs). For a concept this important -- it is the core negative example that explains WHY the dense network is worse -- there is no visual, no code, no diagram, and no concrete numbers. The student must imagine what happens, which is harder than seeing it.
**Student impact:** The student reads about the shifting problem but may not fully internalize it without a concrete visual. A simple before/after showing a digit and its shifted version, or even a short code snippet showing the pixel vectors differ at every position, would make this much more concrete. The prose explanation is clear, but this is the key insight of the lesson and deserves a stronger modality.
**Suggested fix:** Add a minimal visual aid. Options: (a) A simple illustration showing a "3" digit and the same "3" shifted 2px right, with annotation that the 784-element vectors differ at every position. (b) A code snippet: `original = image.flatten(); shifted = shift(image, 2).flatten(); print((original != shifted).sum())` showing that nearly all 784 values change. (c) A conceptual diagram showing the CNN's filter detecting the same edge at both positions (weight sharing). Any of these would strengthen the lesson's most important argument.

#### [POLISH] — Dense network parameter count in ComparisonRow says "~110K" but the arithmetic section says first layer alone is 100,352

**Location:** Section 7 (ComparisonRow) vs Section 9 (parameter arithmetic)
**Issue:** The ComparisonRow shows "Total: ~110K parameters" for the dense network. The arithmetic section shows the first layer alone is 100,352 parameters. A dense network with 784->128->64->10 architecture would be approximately 784*128+128+128*64+64+64*10+10 = 109,386. The "~110K" figure is approximately correct, but this is minor -- the issue is that the student might wonder how the other layers only contribute ~10K if the first layer is already 100K. A brief note would help.
**Student impact:** Very minor. The student might briefly wonder about the arithmetic but would likely move on.
**Suggested fix:** No action needed, or optionally add a brief "(784x128 + 128x64 + 64x10 + biases)" annotation in the ComparisonRow.

#### [POLISH] — Section 9 uses "Let us" instead of "Let's" or natural contractions

**Location:** Section 3 (The Challenge), line "Let us find out."
**Issue:** The text uses "Let us find out" which reads slightly formally for a lesson that otherwise uses contractions freely ("you are," "that is," "it is"). The lesson inconsistently avoids contractions -- sometimes using "that is" (formal) and other times using conversational phrasing. This is a minor stylistic inconsistency.
**Student impact:** Negligible. The tone is slightly uneven but not distracting.
**Suggested fix:** Either commit to no contractions throughout (consistent formality) or use contractions consistently. The current lesson mostly uses uncontracted forms, which is fine as a stylistic choice -- just make it consistent.

### Review Notes

**What works well:**
- The lesson nails the CONSOLIDATE energy. It repeatedly and explicitly tells the student "nothing is new here -- you are assembling pieces you already have." This is exactly right for reducing activation energy on a project lesson.
- The recap section (Section 2) with the StageRow component is an excellent priming device. The student sees the exact architecture they traced in L2 and can immediately map it to the code they are about to write.
- The hook (Section 3) is genuinely compelling. "Same data, same optimizer, same epochs -- the only variable is architecture" sets up a controlled experiment that the student can understand and care about.
- The "Explain the Difference" check (Section 8) is well-placed and well-structured. The student articulates the key insight before the lesson provides the detailed explanation.
- The flatten misconception (Section after Section 9) is addressed clearly and at the right moment. "The dense network flattens pixels; the CNN flattens features" is a memorable framing.
- All scope boundaries from the planning document are respected. The lesson does not drift into batch norm, dropout, augmentation, or advanced architectures.
- The module arc echo and ModuleCompleteBlock provide satisfying closure.

**Systemic concern -- the parameter count error:**
The most important issue is the parameter count discrepancy (Improvement finding #3). The CNN's FC layers dominate its parameter count, making the "fewer parameters" claim factually incorrect for this architecture. This needs to be resolved carefully because it is central to the lesson's narrative. The correct framing is that the CNN's *feature extraction* (conv layers) is vastly more parameter-efficient than the dense approach, even though the overall model may have comparable or more parameters due to the FC classifier. This is still a powerful insight -- it just needs accurate numbers.

**The missing notebook is the blocker:**
The lesson is well-structured as a reading experience, but it is a project lesson. Without the notebook, the student has nothing to do. This must be created before the lesson can ship.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All critical and improvement findings from iteration 1 have been resolved. The notebook now exists with all 10 required sections. Parameter counts are accurate and honestly framed (conv stack efficiency vs total parameters). The shifting experiment has a concrete code snippet. ReLU is addressed in the main body. Training curves are referenced via the notebook. Only two minor polish issues remain.

### Findings

#### [POLISH] — Inconsistent rounding of conv stack parameter count

**Location:** ComparisonRow (line 544) vs InsightBlock (line 554), parameter section (line 747), and summary (line 844)
**Issue:** The ComparisonRow says "~19K params" for the conv stack, but the InsightBlock says "18K parameters," the elaboration section says "18K parameters outperform 100K," and the summary says "18K conv parameters." The actual total is 18,816, which rounds more naturally to ~19K. Three places say "18K" and one says "~19K."
**Student impact:** Negligible. The student might notice the slight inconsistency if reading carefully, but both 18K and 19K communicate the same order-of-magnitude point. The exact figure (18,816) is shown in the arithmetic breakdown.
**Suggested fix:** Standardize to "~19K" everywhere, or use "18.8K" for precision. The ComparisonRow already uses "~19K" which is the most natural rounding of 18,816.

#### [POLISH] — Double hyphen instead of em dash in NextStepBlock description

**Location:** NextStepBlock description prop (line 910)
**Issue:** The description text uses `--` (double hyphen) in two places: "can we go deeper? Better architectures have pushed CNN accuracy much further -- from LeNet's 99%..." and "...what innovations made deeper, more powerful networks possible." The rest of the lesson consistently uses `&mdash;` (HTML entity) for em dashes in JSX text. The `description` prop renders as plain text, so `--` would appear as a literal double hyphen rather than an em dash.
**Student impact:** Negligible. The double hyphen reads fine, but it is a stylistic inconsistency with the rest of the lesson.
**Suggested fix:** Replace `--` with `\u2014` (Unicode em dash) in the description string, or restructure the sentence to avoid needing an em dash.

### Review Notes

**Resolution of iteration 1 findings:**

1. **[CRITICAL] Notebook does not exist** — RESOLVED. The notebook `notebooks/3-1-3-mnist-cnn-project.ipynb` now exists with all 10 required sections: setup, data loading, dense baseline, CNN skeleton with TODOs, dimension verification, training loop, evaluation comparison, parameter breakdown, training curves, and reflection. The notebook is well-scaffolded — the student fills in the CNN class while data loading, training, and evaluation are provided. The parameter breakdown cell includes a dedicated "KEY COMPARISON: Feature Extraction" section that highlights conv stack vs dense first layer, consistent with the lesson's reframed narrative.

2. **[IMPROVEMENT] Missing visual modality for comparison** — RESOLVED. The lesson now explicitly references the notebook's training curves: "Look at the training curves you plotted in the notebook — the CNN's loss drops faster and its accuracy plateaus higher." The notebook includes a dedicated training curves cell (cell-17) that plots loss and accuracy for both models side-by-side. For a CONSOLIDATE lesson where the notebook is the primary artifact, this is the right approach — the visual lives where the student is working.

3. **[IMPROVEMENT] ReLU misconception not addressed in main body** — RESOLVED. The API Recap section now includes: "ReLU is the same activation function you have used before — it applies independently to each value in the feature map, adding nonlinearity after the linear convolution." This addresses the misconception in the main flow, not just the aside. Additionally, the architecture recap (Section 2) explicitly includes ReLU steps (steps 3, 6, and 9) with the annotation "Same ReLU you know."

4. **[IMPROVEMENT] Parameter count numbers inconsistent** — RESOLVED. The ComparisonRow now correctly shows: Dense total ~110K, CNN total ~421K. The comparison is reframed as feature extraction efficiency: "Feature extraction (first layer): ~100K params" vs "Feature extraction (conv stack): ~19K params." The code block and arithmetic section explicitly acknowledge "The CNN has MORE total parameters (the FC layers add bulk), but its conv feature extraction uses far fewer than the dense network's first layer alone." This is accurate, honest, and pedagogically powerful — the student sees that the architectural advantage is specifically about how spatial features are extracted, not about total model size.

5. **[IMPROVEMENT] Shifting experiment purely verbal** — RESOLVED. The shifting experiment now includes a concrete code snippet: `original = image.view(-1); shifted = shift(image, 2).view(-1); (original != shifted).sum()` showing ~750 of 784 values differ. The code with comment annotations ("To the dense network: almost entirely different input / To the CNN: same filters detect same features, pooling absorbs the shift") makes the argument concrete and checkable.

6. **[POLISH] Dense parameter annotation** — Not explicitly changed, but the reframing makes this moot. The parameter breakdown is now detailed enough that the student can trace all the numbers.

7. **[POLISH] "Let us" formality** — No longer present in the lesson text.

**What continues to work well:**
- The CONSOLIDATE energy is strong throughout. The lesson never wavers from "you know all of this — now assemble it."
- The notebook scaffolding level is appropriate: the student writes the CNN class (the novel part) while boilerplate is provided.
- The reframed parameter comparison (conv stack vs dense first layer) is more nuanced and more interesting than the original "fewer parameters" claim. The student learns something genuinely surprising: the CNN's feature extraction is 5x more parameter-efficient, but the FC classifier adds bulk. This sets up future lessons on architectures that reduce the FC layer (e.g., global average pooling in modern architectures).
- The module arc echo and ModuleCompleteBlock provide satisfying closure for the entire 3-lesson module.

**Overall assessment:** The lesson is ready to ship. Both polish items are trivial and can be fixed inline without re-review.
