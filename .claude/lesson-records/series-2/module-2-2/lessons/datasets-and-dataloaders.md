# Lesson Plan: Datasets and DataLoaders

**Module:** 2.2 — Real Data
**Position:** Lesson 1 of 3
**Slug:** `datasets-and-dataloaders`
**Type:** BUILD

---

## Phase 1: Orient — Student State

The student has completed all of Series 1 (Foundations) and Module 2.1 (PyTorch Core). They can write a complete PyTorch training loop with tensors, autograd, nn.Module, and optimizers. Their current gap: they have only trained on synthetic data (hand-crafted tensors), never on a real dataset.

**Relevant concepts the student has:**

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Mini-batch SGD | DEVELOPED | batching-and-sgd (1.3.4) | Polling analogy: random sample estimates full gradient; spectrum from batch=1 to batch=ALL; common batch sizes 32-256 |
| Epochs | DEVELOPED | batching-and-sgd (1.3.4) | One pass through all data; iterations per epoch = N/B |
| Batch size as hyperparameter | DEVELOPED | batching-and-sgd (1.3.4) | Tradeoff between gradient accuracy and update frequency |
| Complete PyTorch training loop | DEVELOPED | training-loop (2.1.4) | forward -> loss -> backward -> update with model(x), criterion, optimizer.step() |
| PyTorch tensor creation/manipulation | DEVELOPED | tensors (2.1.1) | torch.tensor(), shapes, dtypes, device, broadcasting |
| Tensor reshaping (view/reshape) | INTRODUCED | tensors (2.1.1) | view() vs reshape(), -1 auto-dimension |
| nn.Module subclass pattern | DEVELOPED | nn-module (2.1.3) | __init__ + forward(), model.parameters(), LEGO bricks analogy |
| nn.Linear, nn.Sequential, nn.ReLU | DEVELOPED/INTRODUCED | nn-module (2.1.3) | Layer building blocks |
| Gradient noise as beneficial | INTRODUCED | batching-and-sgd (1.3.4) | Noisy gradients help escape sharp minima |
| NumPy-PyTorch interop | DEVELOPED | tensors (2.1.1) | torch.from_numpy(), .numpy(), shared memory |

**Mental models already established:**
- "Polling analogy: random sample estimates the whole" — mini-batch gradients approximate full gradients (from 1.3.4)
- "The ball is still rolling downhill, but now the hill is shaking" — gradient noise from mini-batches
- "Same heartbeat, new instruments" — training loop pattern is fixed, tools change
- "Shape, dtype, device — check these first" — debugging trinity

**What was explicitly NOT covered:**
- How to load data from files or datasets (all data was hand-crafted in code)
- Data preprocessing / transforms (normalization, augmentation)
- The Dataset/DataLoader abstractions
- How batching actually works in code (the concept was taught, the implementation was deferred)
- `torchvision` and its datasets
- Multi-class classification, softmax, cross-entropy (deferred to mnist-project)

**Readiness assessment:** The student is well-prepared. They understand WHY batching matters (from 1.3.4), they can write a training loop (from 2.1.4), and they know tensor manipulation (from 2.1.1). The only gap is the PyTorch API for feeding data — which is exactly what this lesson teaches. No prerequisite gaps.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to load, transform, and batch datasets using PyTorch's Dataset and DataLoader abstractions, connecting the batching theory from Series 1 to practical data pipelines.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Mini-batch SGD | INTRODUCED | DEVELOPED | batching-and-sgd (1.3.4) | OK | Student needs to understand WHY we batch; they have this at DEVELOPED with the polling analogy and batch size tradeoffs |
| Epochs | INTRODUCED | DEVELOPED | batching-and-sgd (1.3.4) | OK | Student needs epoch concept for training loop integration |
| Complete PyTorch training loop | DEVELOPED | DEVELOPED | training-loop (2.1.4) | OK | DataLoader plugs into the existing training loop; student must have the loop to integrate with |
| Tensor creation and manipulation | INTRODUCED | DEVELOPED | tensors (2.1.1) | OK | Data comes out of DataLoader as tensors; student needs to understand shapes and dtypes |
| Tensor reshaping | INTRODUCED | INTRODUCED | tensors (2.1.1) | OK | Transforms may reshape data; student needs view/reshape basics |
| nn.Module pattern | INTRODUCED | DEVELOPED | nn-module (2.1.3) | OK | Model receives batched data from DataLoader; student needs model-building skills |
| Python iterators/for-loops | INTRODUCED | assumed (SWE background) | N/A | OK | DataLoader is an iterator; student is a software engineer, this is basic Python |
| Python classes (__init__, __len__, __getitem__) | INTRODUCED | assumed (SWE background) | N/A | OK | Dataset is a class with dunder methods; student is a software engineer |

**Gap resolution:** No gaps. All prerequisites are met at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "DataLoader is just a for loop over sliced tensors" | The student could write `for i in range(0, N, batch_size): batch = X[i:i+batch_size]` and get batches. Why use DataLoader? | Show the naive for-loop approach: it doesn't shuffle between epochs (biased gradient estimates), doesn't handle the last incomplete batch, doesn't parallelize loading, and requires manually tracking indices. DataLoader handles all of this. | After introducing naive batching, before DataLoader |
| "I need to load the entire dataset into memory as one tensor" | In Module 2.1, all data was a single tensor (X, y). Student assumes real datasets work the same way. | MNIST is 60,000 images. ImageNet is 1.2 million images at high resolution. Loading everything into memory would crash. Dataset's __getitem__ loads one sample at a time — the DataLoader batches them. | During Dataset introduction — motivate lazy loading |
| "Shuffling is just a nice-to-have" | The student knows mini-batch SGD uses random samples, but may think order doesn't matter much in practice. | If data is sorted by class (all 0s, then all 1s, ...), each batch contains only one class. The model learns to predict that class, then "forgets" when it sees the next class. Shuffling prevents this catastrophic pattern. Show training curves: sorted data oscillates wildly, shuffled data converges. | When explaining shuffle=True parameter |
| "Transforms modify the original data" | A software engineer might think of transforms as in-place mutations (like list.sort()). | Transforms are applied on-the-fly in __getitem__. The original data on disk is unchanged. Calling dataset[0] twice with random augmentation can return different results. This is by design — augmentation creates variety. | During transforms section |
| "batch_size should always be as large as possible for speed" | Larger batches = fewer iterations = faster training, right? | From 1.3.4, the student already knows the tradeoff: larger batches give less noisy gradients but fewer updates. Reinforce: very large batches also hit GPU memory limits. The practical constraint is often memory, not theory. | When discussing batch_size parameter, connecting to 1.3.4 |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Manual for-loop batching of the training loop's y=2x+1 data | Positive (baseline) | Show the naive approach the student would write, then reveal its limitations | Connects to the exact data from training-loop (2.1.4); makes DataLoader's value tangible by showing what it replaces |
| Custom Dataset for y=2x+1 with DataLoader | Positive | Demonstrate the full Dataset/DataLoader pattern on familiar data | Same data as the naive example — shows the pattern is about organization, not new math; student sees 1:1 correspondence |
| torchvision.datasets.MNIST with transforms | Positive (real-world) | Show how a real dataset works: download, transform pipeline, DataLoader wrapping | This is the actual dataset the student will use in the next lesson (mnist-project); builds anticipation; transforms pipeline is new |
| Sorted-by-class dataset (no shuffling) | Negative | Demonstrate why shuffle=True matters — without it, training on class-sorted data oscillates | Concretely disproves "shuffling is optional"; the training curve tells the story visually |
| Dataset where __getitem__ returns wrong types/shapes | Negative | Show what happens when the Dataset contract is violated — DataLoader crashes at batch collation | Builds debugging instinct; the error message is confusing without understanding that DataLoader collates individual samples into batches |

### Gap Resolution

No gaps identified. All prerequisites are at sufficient depth.

---

## Phase 3: Design

### Narrative Arc

The student just finished Module 2.1 by writing a complete training loop — but on fake data. Twenty points, generated in two lines of code. That works for learning the API, but it's not how real ML works. Real datasets have thousands to millions of samples. They come as files on disk, not tensors in memory. They need preprocessing — normalization, reshaping, sometimes augmentation. And the training loop needs to see different random subsets each epoch (the student already knows why from the SGD lesson: random sampling gives unbiased gradient estimates). The question this lesson answers is: "How do I actually feed real data into my training loop?" PyTorch's answer is two abstractions — Dataset (how to access one sample) and DataLoader (how to batch, shuffle, and iterate over samples) — and once the student sees them, the training loop from Module 2.1 works on any dataset with zero changes.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example** | Rewrite the y=2x+1 training loop from 2.1.4 to use DataLoader; same data, same model, same loss curve | The student needs to see that DataLoader plugs into the EXISTING training loop. Using the same data eliminates confounding variables — the only change is how data is fed. |
| **Visual (diagram)** | Flow diagram: Raw Data -> Dataset (one sample) -> DataLoader (batches) -> Training Loop. Show the DataLoader as a "batching machine" that sits between data and the loop. | The architecture of the data pipeline is spatial — what connects to what. A diagram makes the two-layer abstraction (Dataset wraps data, DataLoader wraps Dataset) concrete. |
| **Symbolic (code)** | Dataset class with __init__, __len__, __getitem__; DataLoader constructor with batch_size, shuffle; integration into training loop | This is a code-first lesson. The API IS the concept. Code is the primary modality. |
| **Verbal/Analogy** | "Dataset is a menu (tells you what's available and how to get one item). DataLoader is the kitchen (batches orders, shuffles the queue, serves plates efficiently)." | Software engineers think in terms of interfaces and consumers. The menu/kitchen analogy maps to their existing mental model: Dataset = data access interface, DataLoader = consumer that handles logistics. |
| **Intuitive (connection)** | Explicit bridge from the polling analogy (1.3.4) to shuffle=True: "Remember the polling analogy? Random sampling gives unbiased estimates. shuffle=True is HOW you get random sampling in practice." | The student already has the "why" of random batching. This lesson provides the "how." Making the connection explicit reinforces both concepts. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. `torch.utils.data.Dataset` abstraction (__getitem__ + __len__)
  2. `torch.utils.data.DataLoader` (batching, shuffling, iteration)
  3. `torchvision.transforms` pipeline (arguably just API knowledge, not a new concept)
- **Previous lesson load:** CONSOLIDATE (training-loop was integration, no new theory)
- **This lesson's load:** BUILD — appropriate. The underlying concept (mini-batch SGD) is already at DEVELOPED depth. The new learning is the API pattern, not the theory. Two genuinely new abstractions (Dataset, DataLoader) plus one API pattern (transforms) is within the 2-3 new concept limit.

### Connections to Prior Concepts

| Prior Concept | Connection | How to Make It Explicit |
|---------------|-----------|----------------------|
| Mini-batch SGD (1.3.4) | DataLoader implements what the student learned conceptually: random subsets, configurable batch size, epoch iteration | "Remember the polling analogy? shuffle=True is how you get that random sampling." |
| Epochs (1.3.4) | One full pass through the DataLoader = one epoch; iterations per epoch = len(dataset) / batch_size, exactly the formula from 1.3.4 | Show the math: "60,000 samples / 64 batch size = 937 iterations per epoch. Sound familiar?" |
| Training loop pattern (2.1.4) | DataLoader replaces the hand-crafted tensor; the loop body (forward/loss/backward/update) stays identical | Side-by-side: old loop with raw tensors vs new loop with DataLoader. Only the data-feeding line changes. |
| Tensor shapes (2.1.1) | DataLoader outputs tensors with a batch dimension prepended; shapes matter for model input | Show batch tensor shape: [batch_size, features] for tabular, [batch_size, channels, H, W] for images |
| "Shape, dtype, device — check these first" (2.1.1) | Debugging DataLoader output: print shape, dtype, value range after transforms | Reinforce the debugging trinity: first thing to do with a new DataLoader is inspect one batch |

**Potentially misleading analogies:** The "Same heartbeat, new instruments" metaphor from training-loop still applies — the loop body doesn't change, only the data feeding changes. No existing analogies are misleading here.

### Scope Boundaries

**This lesson IS about:**
- The Dataset abstraction (what it is, why it exists, how to implement one)
- The DataLoader abstraction (batching, shuffling, iterating)
- Basic transforms (ToTensor, Normalize) and the Compose pipeline
- torchvision.datasets as a source of pre-built datasets
- Connecting DataLoader to the existing training loop

**This lesson is NOT about:**
- Training a model to convergence on a real dataset (that's mnist-project)
- Cross-entropy loss or softmax (deferred to mnist-project)
- Data augmentation strategies (RandomFlip, RandomCrop, etc.) — mentioned but not developed
- Custom collate functions or advanced DataLoader options (num_workers, pin_memory)
- Writing your own image loading code (pillow, etc.)
- Train/val/test splitting in code (the concept is known from 1.1; implementation kept simple here)
- Multi-class classification (deferred to mnist-project)

**Target depth:**
- Dataset: DEVELOPED (student can implement a custom Dataset)
- DataLoader: DEVELOPED (student can configure and use DataLoader in a training loop)
- Transforms: INTRODUCED (student understands the pattern and can use basic transforms; not designing custom ones)
- torchvision.datasets: INTRODUCED (student can load a built-in dataset; not exploring the full catalog)

### Lesson Outline

1. **Context + Constraints** (~2 min read)
   - What: How to feed real data into a PyTorch training loop
   - What NOT: We won't train a real model to convergence here (that's next lesson). We're setting up the plumbing.
   - Connection: "You have the training loop from last module. Now we connect it to real data."

2. **Hook: The Scale Problem** (~3 min)
   - Type: Before/after contrast
   - Show the training loop from 2.1.4 — it works, but the "data" is 20 points in two tensors. Then: "MNIST has 60,000 images. ImageNet has 1.2 million. Your approach of `X = torch.tensor(...)` doesn't scale." Show what happens if you try to create a tensor of 60,000 28x28 images (it works for MNIST — memory isn't the issue — but the batching, shuffling, and epoch logic is a mess without the right abstraction).
   - The real hook: "What if the training loop body didn't have to change at all? What if you could swap the data source and keep everything else?"

3. **Explain: The Naive Approach** (~4 min)
   - Show manual for-loop batching on the y=2x+1 data from 2.1.4
   - It works! But enumerate the problems: no shuffling between epochs, manual index math, no handling of the last incomplete batch, no parallelism
   - "You could fix all of these one by one. Or..."

4. **Explain: Dataset — One Sample at a Time** (~6 min)
   - The Dataset contract: `__len__` (how many samples) and `__getitem__` (get one sample by index)
   - Implement `SimpleDataset` for y=2x+1 data — the student recognizes this data
   - Emphasize: Dataset doesn't batch. It doesn't shuffle. It just says "I have N items, here's item i."
   - Menu analogy: "A Dataset is a menu — it tells you what's available and how to get one item."
   - **Modalities:** code (primary), verbal/analogy (menu)

5. **Explain: DataLoader — The Batching Machine** (~6 min)
   - DataLoader wraps a Dataset and handles batching, shuffling, and iteration
   - Show: `DataLoader(dataset, batch_size=8, shuffle=True)`
   - Iterate and print shapes — the batch dimension appears automatically
   - Connect to 1.3.4: "Remember `iterations per epoch = N/B`? Watch: 100 samples / 8 batch size = 13 iterations. The last batch has 4 samples (100 - 12*8)."
   - Kitchen analogy: "DataLoader is the kitchen — it batches orders, shuffles the queue, serves plates."
   - **Modalities:** code (primary), verbal/analogy (kitchen), concrete example (count iterations), visual (pipeline diagram)

6. **Check 1: Predict-and-Verify** (~2 min)
   - Given a Dataset with 200 samples and DataLoader with batch_size=32, shuffle=True:
     - How many iterations per epoch? (7, with last batch having 200 - 6*32 = 8 samples)
     - Will two consecutive epochs iterate in the same order? (No — shuffle=True)
     - What is the shape of each batch if each sample is a tensor of shape [5]? ([32, 5] for full batches, [8, 5] for the last)

7. **Explain: Plugging DataLoader into the Training Loop** (~5 min)
   - Side-by-side: the training loop from 2.1.4 (raw tensors) vs the same loop with DataLoader
   - Highlight: the loop body is IDENTICAL. Only the data-feeding line changes.
   - Run both — same convergence, same loss curve
   - "The heartbeat doesn't change. The DataLoader is a new instrument."
   - **Modalities:** code side-by-side (primary), connection to prior (heartbeat metaphor)

8. **Explain: Shuffling Matters (Negative Example)** (~4 min)
   - Create a dataset sorted by target value (or class label if using simple classification)
   - Train with shuffle=False vs shuffle=True
   - Show training curves: sorted data has erratic loss, shuffled data converges smoothly
   - Connect to 1.3.4: "This is the polling analogy in action. Biased samples give biased gradients."
   - **Modalities:** visual (training curves), concrete example (sorted vs shuffled), connection to prior (polling)

9. **Explain: Transforms — Preprocessing on the Fly** (~5 min)
   - Introduce torchvision.transforms: ToTensor(), Normalize(), Compose()
   - Show MNIST loading: `torchvision.datasets.MNIST(transform=transforms.Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))`
   - Explain: transforms run in __getitem__, not upfront. Original data on disk is unchanged.
   - Inspect one sample: shape [1, 28, 28], dtype float32, values in normalized range
   - Reinforce debugging trinity: "Shape, dtype, device — check these first on any new data."
   - **Modalities:** code (primary), intuitive (on-the-fly preprocessing), connection to prior (debugging trinity)

10. **Explore: MNIST Data Inspection** (~3 min)
    - Load MNIST with DataLoader, inspect one batch
    - Print shapes, visualize a few images (matplotlib grid), check label distribution
    - "This is the data you'll train on in the next lesson."
    - Build anticipation without starting the project

11. **Check 2: Transfer Question** (~2 min)
    - "A colleague loads a custom image dataset as a list of PIL images and a list of labels. They convert everything to tensors upfront, then loop with manual slicing. Their training works but is slow and uses a lot of memory. What would you suggest?"
    - Expected answer: Implement a custom Dataset that loads images lazily in __getitem__, use transforms for ToTensor/Normalize, wrap in DataLoader for batching and shuffling.

12. **Elaborate: The Dataset Contract (Negative Example)** (~3 min)
    - Show a broken Dataset where __getitem__ returns inconsistent shapes or types
    - DataLoader crashes at collation time with a confusing error
    - Explain: DataLoader tries to stack individual samples into a batch tensor. If shapes don't match, it fails.
    - "The error message says 'stack expects each tensor to be equal size'. Now you know why."

13. **Practice: Colab Notebook** (~15-20 min student time)
    - Exercise 1 (guided): Implement a custom Dataset for the y=2x+1 data, wrap in DataLoader, iterate and print shapes
    - Exercise 2 (guided): Load MNIST with torchvision.datasets, apply transforms, inspect batches
    - Exercise 3 (supported): Integrate DataLoader into the training loop from 2.1.4 — train on y=2x+1 but with proper batching
    - Exercise 4 (supported): Experiment with batch sizes (1, 32, 256, full-batch) — measure iterations per epoch and observe loss curves
    - Exercise 5 (independent): Write a custom Dataset for a simple CSV file (provided), apply basic transforms, train a model with DataLoader

14. **Summarize** (~2 min)
    - Dataset = how to access one sample (__len__ + __getitem__)
    - DataLoader = how to batch, shuffle, and iterate (wraps Dataset)
    - Transforms = preprocessing applied per-sample in __getitem__
    - The training loop body doesn't change — DataLoader replaces the data source

15. **Next Step**
    - "You have the plumbing. Next lesson: we use it. MNIST — 60,000 handwritten digits. You'll build a model, train it, and see real predictions."

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 5
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues — the student will not be lost or form a wrong mental model. But five improvement-level findings mean the lesson is noticeably weaker than it should be. The biggest gaps: a planned misconception about batch size was dropped, MNIST images are never visualized (the student is told "this is the data you'll train on" but never sees a digit), the shuffling negative example lacks the training curve comparison the plan called for, and the lazy loading pattern is never demonstrated despite being the answer to a planned misconception. Fix these and the lesson goes from good to strong.

### Findings

### [IMPROVEMENT] — Missing misconception: "batch_size should always be as large as possible"

**Location:** Omitted entirely (planned for "when discussing batch_size parameter")
**Issue:** The planning document identified this as misconception #5 with a concrete plan to address it when discussing the batch_size parameter, connecting back to the 1.3.4 tradeoff and adding the practical GPU memory constraint. The built lesson never mentions this. The student configures batch_size=8, batch_size=32, and batch_size=64 without any discussion of how to choose batch size or what happens when it is too large.
**Student impact:** The student knows the theoretical tradeoff from 1.3.4, but this lesson was supposed to ground it in practice (memory limits, practical guidance). Without it, the student may default to "bigger is faster" thinking when working with real datasets. The Colab Exercise 4 asks the student to experiment with batch sizes (1, 32, 256, full-batch), but the lesson itself provides no framework for interpreting what they will observe.
**Suggested fix:** Add a brief paragraph or aside after the DataLoader introduction (Section 5) or after the full integrated code (Section 7) connecting batch_size to the tradeoff from 1.3.4 and adding the practical constraint: larger batches require more GPU memory. A TipBlock aside like "Batch size is a tradeoff: larger batches give cleaner gradients but use more memory and update less often. Common practice: start with 32 or 64 and adjust based on your GPU's memory." This does not need a full section — an aside or a brief paragraph would suffice.

### [IMPROVEMENT] — MNIST images are never visualized

**Location:** Section 10 (MNIST — Your First Real Dataset)
**Issue:** The planning document explicitly stated: "Load MNIST with DataLoader, inspect one batch. Print shapes, visualize a few images (matplotlib grid), check label distribution." The built lesson prints shapes and value ranges but never shows what an MNIST digit looks like. The student is told "this is the data you'll train on" without ever seeing a single image.
**Student impact:** For a visual dataset, seeing the data is pedagogically important. The student has been working with abstract tensors for the entire course. MNIST is their first image dataset. Seeing the 28x28 pixels as an actual handwritten digit creates a concrete mental image of what the model will learn to classify. Without it, MNIST remains abstract — a tensor of shape [64, 1, 28, 28] with no visual grounding.
**Suggested fix:** Add a CodeBlock showing a simple matplotlib visualization (2-3 lines: `plt.imshow(images[0].squeeze(), cmap='gray'); plt.title(f"Label: {labels[0]}")`) or a small grid of 8-10 images with their labels. This is a code-centric lesson and the visualization is code — it fits naturally. Even a mention like "In your Colab notebook, you'll visualize these images" would be better than nothing, but actually showing the code inline is stronger.

### [IMPROVEMENT] — Shuffling negative example lacks training curve comparison

**Location:** Section 8 (Shuffling Matters)
**Issue:** The planning document stated: "Show training curves: sorted data has erratic loss, shuffled data converges smoothly." The built lesson explains the problem with sorted data and connects to the polling analogy, but never shows a training curve comparison. The argument is entirely verbal — "the gradient estimates are biased" — without visual evidence.
**Student impact:** The student has seen training curves in the overfitting lesson (1.3.6) and in the training loop lesson (2.1.4). A visual comparison of erratic-loss vs smooth-convergence would be the strongest modality for this point. Without it, the lesson asserts that shuffling matters but does not demonstrate it visually. The student is asked to take it on faith rather than seeing the evidence.
**Suggested fix:** Add a code block showing a training run comparison (even if the student would run it in Colab), or describe what the curves look like ("without shuffling, the loss zigzags wildly; with shuffling, it descends smoothly"). Given this is a code-centric lesson without custom widgets, a descriptive paragraph referencing the Colab exercise where the student will see it firsthand is acceptable. Best option: add a note pointing to Exercise 4 in the Colab notebook where the student will experiment with batch sizes and can also test shuffle=True vs shuffle=False.

### [IMPROVEMENT] — Lazy loading pattern never demonstrated

**Location:** Sections 4-5 (Dataset / DataLoader) and the Transfer Question
**Issue:** The planning document identified "I need to load the entire dataset into memory as one tensor" as misconception #2, with a planned response: "Dataset's __getitem__ loads one sample at a time — the DataLoader batches them." However, the lesson's own SimpleDataset stores ALL data in memory in __init__ (self.x and self.y). The MNIST example uses torchvision.datasets which handles this internally. The transfer question's answer mentions "stores file paths (not the images themselves) and loads each image lazily in __getitem__" — but this pattern is never taught in the lesson itself. The student is expected to produce an answer the lesson never demonstrated.
**Student impact:** The student learns that Dataset has __getitem__ but sees it used only for indexing into pre-loaded tensors. They never see the pattern of loading from disk in __getitem__. When the transfer question expects them to suggest lazy loading, they are reaching beyond what was taught. The answer is provided in the collapsible, but it introduces a pattern the student has no experience with.
**Suggested fix:** Two options: (1) Add a brief paragraph in the Dataset section explaining that __getitem__ CAN load data lazily from disk (one image at a time) and that this is how large datasets work — even if you don't build a full example. This plants the seed. (2) After the SimpleDataset example, show a 5-line skeleton of a lazy Dataset (stores file paths in __init__, loads image in __getitem__) with a comment like "# We won't build this now, but this is the pattern for large datasets." This connects the misconception to a concrete code pattern without requiring a full implementation.

### [IMPROVEMENT] — `__new__` hack in sorted data example

**Location:** Section 8 (Shuffling Matters), sorted_data_problem.py code block
**Issue:** The code uses `dataset_sorted = SimpleDataset.__new__(SimpleDataset)` to create a Dataset instance without calling `__init__`. This is a Python meta-programming pattern that bypasses normal construction. While a software engineer may understand `__new__`, it is an unusual pattern that could confuse the student into thinking this is a standard way to create Datasets. The lesson does not explain why `__new__` is used or that this is a hack for demonstration purposes.
**Student impact:** The student may wonder: "Is `__new__` something I should use when creating Datasets?" The answer is no — this is a convenience for the example. The cognitive load of parsing an unfamiliar construction pattern distracts from the actual point (shuffling matters).
**Suggested fix:** Replace with a simpler approach: either create a new `SortedDataset` class that takes x and y as constructor arguments, or simply create the sorted tensors and wrap them in a `TensorDataset` (which has not been taught but is simpler than `__new__`). Simplest fix: define a 4-line inline Dataset class that accepts pre-created tensors, or just use the sorted tensors directly without wrapping in a Dataset (the point is about shuffling, not about the Dataset class).

### [POLISH] — "PIL objects" referenced without context

**Location:** Section 9 (Transforms), first paragraph
**Issue:** The text says "Images come as PIL objects or NumPy arrays — not tensors." PIL has not been mentioned in the course before. A software engineer likely knows PIL/Pillow, but within the course context, this term appears without establishment.
**Student impact:** Minor — the student probably knows what PIL is. But the term appears in a list alongside "NumPy arrays," which the student has used extensively. PIL has less grounding. The student would not be lost, but might briefly wonder.
**Suggested fix:** Change "PIL objects" to "PIL images (from Python's Pillow library)" on first mention, or simply "image files" to avoid the unnecessary specificity. Since transforms.ToTensor() is shown converting PIL images, a brief parenthetical is sufficient.

### [POLISH] — Long stretch between check points

**Location:** Between Check 1 (Section 6, ~line 424) and Check 2 (Section 11, ~line 839)
**Issue:** The student reads through five substantial sections (Integration, Full code, Shuffling, Transforms, MNIST) with no interactive check point. This is approximately 400 lines of lesson content / ~8 minutes of reading without a pause to verify understanding.
**Student impact:** For an ADHD-friendly design, sustained attention over 8 minutes of dense reading without a check point is risky. The student may lose focus or drift. Engagement drops when there is no prompt to actively process what they are reading.
**Suggested fix:** Consider adding a brief check after the Transforms section or the MNIST inspection section. A simple predict-and-verify question like "If your DataLoader yields images of shape [64, 1, 28, 28], what is the batch size? What does the 1 represent?" would break up the reading and reinforce the new content. Alternatively, move Check 2 earlier (after MNIST inspection) and place the broken-shapes negative example and practice section after it.

### [POLISH] — Hook says "100 data points" but plan says "20"

**Location:** Section 2 (Hook), line 106
**Issue:** The hook says "you trained a model on y=2x+1 with 100 data points." The planning document's hook description says "the 'data' is 20 points in two tensors." The code block shows `torch.randn(100, 1)`. The lesson is internally consistent (100 points throughout), but differs from the plan.
**Student impact:** None — the lesson is self-consistent. The student sees 100 points in the code and 100 points in the text.
**Suggested fix:** No action needed in the lesson. This is a plan-vs-build deviation to note. The plan could be updated to match the build, or leave as-is since 100 is a better number for demonstrating incomplete last batches (100/8 = 12.5).

### Review Notes

**What works well:**
- The narrative arc is strong. Problem-before-solution is maintained throughout. The student feels the need for Dataset/DataLoader before seeing the solution.
- The menu/kitchen analogy is clean and maps well to the software engineer's mental model of interfaces and consumers.
- The connections to prior lessons (polling analogy, debugging trinity, same heartbeat) are explicit and well-placed. This lesson does not assume the student remembers — it reminds them at the exact moment the connection matters.
- The side-by-side comparison (old loop vs new loop) is the lesson's central payoff and it delivers. The student can see that the loop body is identical.
- Scope boundaries are well-enforced. The lesson does not drift into cross-entropy, model architecture, or training MNIST to convergence.
- The two checks (predict-and-verify + transfer question) test different cognitive levels. Good.

**Patterns to watch:**
- The lesson introduces 3 concepts (Dataset, DataLoader, transforms) and addresses 3 misconceptions clearly, with 2 more planned but underserved. For a BUILD lesson, this is acceptable but leaves room for improvement.
- The absence of any visualization (no matplotlib, no image grid) in a lesson about image data is the most notable gap. This is the student's first encounter with image data and they never see what it looks like.
- The `__new__` usage is a one-off issue, but it reflects a broader pattern: code examples should use standard patterns the student would actually write, not clever shortcuts for demonstration convenience.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 2

### Verdict: NEEDS REVISION

All 5 improvement findings and 2 of 3 polish findings from iteration 1 have landed correctly. The lazy loading skeleton, MNIST visualization, training curve comparison, batch_size TipBlock, and SortedDataset class are all present and well-integrated. One new improvement finding: the subtitle of the Hook section contradicts the body text. Two polish items: a code comment uses a literal em dash instead of `--`, and there is a minor redundancy in the flow between the MNIST preview paragraph and the next section.

### Iteration 1 Fix Verification

| Finding | Status |
|---------|--------|
| Missing batch_size misconception | FIXED — TipBlock aside in DataLoader section (lines 406-411) |
| MNIST images never visualized | FIXED — matplotlib grid code block + descriptive prose (lines 904-926) |
| Shuffling lacks training curve comparison | FIXED — shuffle_comparison.py code block (lines 712-745) + prose (lines 748-752) |
| Lazy loading pattern never demonstrated | FIXED — LazyImageDataset skeleton (lines 298-312) with "we will not build this now" note |
| `__new__` hack in sorted data | FIXED — clean SortedDataset class (lines 670-691) |
| "PIL objects" without context | FIXED — parenthetical "(from Python's Pillow library)" (line 774) |
| Long stretch between checkpoints | FIXED — Quick Check added after MNIST inspection (lines 953-993) |
| Hook says 100 vs plan says 20 | Noted as no-action; lesson is self-consistent at 100 |

### Findings

### [IMPROVEMENT] — Subtitle contradicts body text in Hook section

**Location:** Section 2 (The Scale Problem), line 101 vs line 106
**Issue:** The subtitle reads "Your training loop works -- but only on 20 data points." The very next paragraph says "you trained a model on y = 2x + 1 with 100 data points." The code block shows `torch.randn(100, 1)`. The subtitle says 20; everything else says 100.
**Student impact:** The student reads "20 data points" in the subtitle, then immediately reads "100 data points" in the body. This is a factual contradiction in consecutive lines. The student would notice and wonder which is correct. It is a small thing, but it undermines the lesson's credibility at the very moment it needs to establish trust (the hook). The rest of the lesson consistently uses 100.
**Suggested fix:** Change the subtitle from "Your training loop works -- but only on 20 data points" to "Your training loop works -- but only on 100 data points." Alternatively, drop the specific number: "Your training loop works -- but only on a handful of data points."

### [POLISH] — Inconsistent comment style in lazy loading code block

**Location:** Section 4, line 301, `lazy_dataset_skeleton.py` code block
**Issue:** The code comment reads `# Lazy loading pattern — for large datasets` using a literal em dash character (`—`). Every other code comment in the lesson uses double dashes (`--`), e.g., `# Naive batching -- manual index math`, `# From The Training Loop -- all data in memory as tensors`, `# --- Dataset: same y = 2x + 1 data ---`. The em dash in a code comment is stylistically inconsistent.
**Student impact:** Negligible. The student is unlikely to notice. But consistency in code style signals care.
**Suggested fix:** Change `—` to `--` in the code comment to match the rest of the lesson's code comments.

### [POLISH] — Redundant MNIST preview paragraph

**Location:** Lines 940-951 (preview paragraph) and lines 1198-1217 (What Comes Next section)
**Issue:** The paragraph after MNIST visualization (lines 940-950) says "This is the data you will train on in the next lesson. You now have the plumbing — a Dataset with transforms, wrapped in a DataLoader that produces batched tensors. The model, loss, and training loop come next." The "What Comes Next" section (lines 1205-1216) repeats very similar content: "You have the plumbing. A Dataset that loads MNIST images, transforms that convert them to normalized tensors, and a DataLoader that batches and shuffles them. Next lesson: you use it." The two passages say the same thing in nearly the same words, about 250 lines apart.
**Student impact:** Minor. The student may experience a vague sense of "didn't I just read this?" The redundancy is not harmful but slightly weakens the conclusion's impact — the payoff of the summary should feel fresh, not like a repeat.
**Suggested fix:** Shorten the earlier preview paragraph (lines 940-950) to just the forward-looking hook: "This is the data you will train on in the next lesson." Remove the plumbing summary from the preview and let the formal summary section (Section 14) and the "What Comes Next" section (Section 15) carry that weight. Alternatively, make the earlier paragraph more observational ("These are the digits your model will learn to classify") and save the structural summary for the end.

### Review Notes

**What works well:**
- All iteration 1 fixes landed cleanly. The lesson is noticeably stronger. The lazy loading skeleton, MNIST visualization, training curve comparison, and batch_size aside all fit naturally into the flow without feeling bolted on.
- The new Quick Check after MNIST (batch dimension questions) is well-placed and tests a genuinely useful skill — reading image tensor shapes. The two questions (batch size and channel dimension) are the exact two things a student needs to parse when they first see `[64, 1, 28, 28]`.
- The SortedDataset class is a significant improvement over the `__new__` hack. It is clean, readable, and uses the standard Dataset pattern the student just learned.
- The lesson now addresses all 5 planned misconceptions at appropriate points.
- The narrative arc remains intact after the additions. Each fix integrates smoothly.

**Overall assessment:**
This lesson is close to PASS. The one improvement finding (subtitle contradiction) is a quick fix. The two polish items are minor. One more iteration should be sufficient.

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
- [x] At least 3 modalities planned for the core concept, each with rationale
- [x] At least 2 positive examples + 1 negative example, each with stated purpose
- [x] At least 3 misconceptions identified with negative examples
- [x] Cognitive load <= 3 new concepts (Dataset, DataLoader, transforms)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated
