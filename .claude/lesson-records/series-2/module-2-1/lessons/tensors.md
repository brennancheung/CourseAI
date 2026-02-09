# Lesson: tensors

**Module:** 2.1 PyTorch Core
**Position:** 1 of 4
**Type:** BUILD
**Colab Notebook:** `2-1-1-tensors.ipynb`

---

## Phase 1: Orient (Student State)

The student is a software engineer who completed all 17 lessons of Series 1. They implemented linear regression from scratch in NumPy (~15 lines). They are comfortable with:

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| NumPy array operations | APPLIED | implementing-linear-regression (1.1.5) | Built a working model with NumPy; comfortable with array creation, indexing, math ops |
| Linear model y-hat = wx + b | DEVELOPED | linear-regression (1.1.3) | Parameters as learnable knobs; formula is second nature |
| MSE loss function | DEVELOPED | loss-functions (1.1.4) | "Wrongness score"; computed manually and in NumPy |
| Training loop | DEVELOPED | implementing-linear-regression (1.1.5) | Forward -> loss -> backward -> update; implemented in NumPy |
| Matrix multiplication (implicit) | APPLIED | implementing-linear-regression (1.1.5) | Used `np.dot()` and broadcasting in NumPy implementation |
| Broadcasting | APPLIED | implementing-linear-regression (1.1.5) | Used in NumPy linear regression; understands shape compatibility |

**Mental models established:**
- "Parameters are knobs the model learns" (applies to tensors as containers for parameters)
- NumPy as the baseline for numerical computing (tensors should feel like a superset)

**Not covered that's relevant:**
- GPU computing concepts (CUDA, device management) — entirely new
- PyTorch-specific dtype system — new API
- Tensor memory layout and views — new concept
- `torch.no_grad()` — deferred to autograd lesson

**Readiness assessment:** The student is very well prepared. They have hands-on NumPy experience and understand the mathematical operations tensors will perform. The new learning is the PyTorch API and GPU concept, not the underlying math.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to create, manipulate, and move PyTorch tensors, understanding them as GPU-capable NumPy arrays that form the foundation of everything in PyTorch.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| NumPy array creation | APPLIED | APPLIED | implementing-linear-regression | OK | Student created arrays, did math with them; tensors are the same API |
| Array shapes and indexing | APPLIED | APPLIED | implementing-linear-regression | OK | Used shapes in NumPy implementation |
| Broadcasting rules | DEVELOPED | APPLIED | implementing-linear-regression | OK | Exceeds requirement; used broadcasting in practice |
| Floating point dtypes | INTRODUCED | INTRODUCED | loss-functions (implicit) | OK | Used float64 in NumPy; will introduce float32 as PyTorch default |
| GPU computing | INTRODUCED | MISSING | — | MISSING | Never discussed; entirely new |

**Gap resolution:**
- GPU computing (MISSING, small gap): The student doesn't need deep GPU knowledge for this lesson. A brief section explaining "GPU = massively parallel processor, good at the exact math neural networks need" is sufficient. 2-3 paragraphs + a simple diagram showing CPU vs GPU. This is INTRODUCED depth only — they just need to know WHY to move tensors to GPU and HOW (`.to('cuda')`), not how GPUs work internally.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Tensors are just NumPy arrays with a different name" | The API is nearly identical for basic operations | Show something NumPy can't do: `tensor.to('cuda')` moves computation to GPU; `tensor.requires_grad = True` enables autograd. NumPy has neither. "Same interface, different engine." | After the NumPy comparison section |
| "I should always use float64 for precision" | NumPy defaults to float64; more precision = better | In deep learning, float32 is the default because (1) gradients are approximate anyway (mini-batch noise >> float precision), (2) float64 uses 2x memory and is 2x slower on GPU, (3) float16 is even used for training large models. Show: same training result with float32 vs float64, but float32 is 2x faster. | During dtype section |
| "GPU is always faster" | GPU = fast is the common narrative | Small tensors (< ~1000 elements): CPU is faster because GPU transfer overhead dominates. Show timing: 10-element tensor multiply is faster on CPU. GPU wins at scale (10,000+ elements). | After GPU section |
| "`.numpy()` and back is free" | It looks like a simple conversion | Tensors with `requires_grad=True` can't be converted to NumPy directly (breaks the computation graph). Must `.detach().numpy()`. This foreshadows autograd lesson. | End of NumPy interop section |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Create the same linear regression data in NumPy vs PyTorch | Positive | Show API similarity; lowest activation energy ("you already know this") | Student literally did this in 1.1.5; seeing the same code in PyTorch makes it feel familiar |
| Reshape a batch of images (28x28 -> 784) | Positive | Practical shape manipulation; previews MNIST data shape | Real-world operation they'll do in Module 2.2; demonstrates `view`/`reshape` |
| Small tensor on GPU vs CPU timing | Negative | GPU isn't always faster; small tensors have transfer overhead | Disproves "GPU always wins" misconception; teaches when to use GPU |
| Broadcasting: add bias vector to batch of predictions | Positive (stretch) | Connects to neural network operations they'll write | Directly maps to the bias term in y = Wx + b they already know |

### Widget Consideration

**Decision: No custom interactive widget for this lesson.**

Rationale: This lesson's primary learning is API fluency — reading and writing PyTorch code. The modality IS code. An interactive shape explorer or tensor visualizer would add engagement but risk replacing the hands-on coding that IS the point. The Colab notebook (`2-1-1-tensors.ipynb`) serves the interactive role: the student types real PyTorch code, sees real outputs, and builds muscle memory.

In-lesson interactivity comes from:
- Side-by-side code comparisons (NumPy vs PyTorch) with collapsible reveals
- Predict-and-verify checks where the student guesses output before seeing it
- The Colab notebook exercises at the end

If a widget is added later, the best candidate would be a **tensor shape calculator** — input operation (reshape, matmul, broadcast) and tensor shapes, output the result shape or an error explanation. This would help with the #1 source of PyTorch bugs (shape mismatches) without replacing coding practice.

---

## Phase 3: Design

### Narrative Arc

You've already built a linear regression model from scratch in NumPy. You understand every line: create arrays, multiply, compute loss, update parameters. The problem is: NumPy maxes out on your CPU. Real neural networks have millions of parameters and train on millions of examples. You need two things NumPy doesn't have: GPU acceleration and automatic differentiation. PyTorch tensors solve the first problem — they're NumPy arrays that can run on a GPU, doing thousands of operations in parallel. This lesson is about making the switch: everything you know about NumPy still applies, but now your data can move to a GPU with a single line of code.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | Side-by-side NumPy vs PyTorch code for the SAME linear regression data | Tensors are best understood by seeing "this is what you already know, in a new syntax" — concrete comparison eliminates abstraction |
| Symbolic | Code snippets showing `torch.tensor()`, `.shape`, `.dtype`, `.to()`, `@` operator | Tensors are a CODE concept — the student needs to read and write these fluently; code IS the modality |
| Visual | Shape diagram showing tensor dimensions (scalar, vector, matrix, batch) | Shapes are the #1 source of bugs; a visual showing 0D/1D/2D/3D tensor with labeled dimensions builds spatial intuition |
| Intuitive | "NumPy arrays that can ride the GPU bus" — tensors are containers that know where they live (CPU or GPU) and can move between them | The mental model for device management needs to be simple and sticky |

### Cognitive Load Assessment

- **New concepts:** 3 (PyTorch tensor API, GPU device management, PyTorch dtypes)
- **Previous lesson load:** N/A (first lesson in module; last Series 1 lesson was CONSOLIDATE)
- **This lesson's load:** BUILD — most of the content is mapping existing NumPy knowledge to PyTorch syntax. The only genuinely new idea is GPU placement.
- **Appropriate:** Yes. Coming off a CONSOLIDATE capstone, a BUILD lesson is the right re-entry point. The student is starting a new series and needs a confidence boost, not a challenge.

### Connections to Prior Concepts

- **NumPy arrays (1.1.5):** Direct mapping. "Every NumPy operation you used has a PyTorch equivalent." The student's implementing-linear-regression code is the anchor.
- **Linear model y = wx + b (1.1.3):** The running example. "Let's create the same data and parameters, but as tensors."
- **Training loop (1.1.5):** Foreshadow. "By lesson 4, you'll rewrite your entire NumPy training loop in PyTorch."

**Prior analogies to extend:**
- "Parameters are knobs" — tensors are the containers that hold those knobs

**Prior analogies that could mislead:**
- None significant. NumPy -> PyTorch is a clean mapping.

### Scope Boundaries

**This lesson IS about:**
- Creating tensors (from data, from NumPy, random, zeros/ones)
- Tensor attributes: shape, dtype, device
- Basic operations: arithmetic, matrix multiply, reshaping
- GPU: moving tensors to/from GPU, why it matters
- NumPy interop: `.numpy()`, `torch.from_numpy()`
- INTRODUCED depth for GPU, DEVELOPED depth for tensor operations

**This lesson is NOT about:**
- Autograd / `requires_grad` (next lesson)
- `torch.no_grad()` (next lesson)
- nn.Module or layers (lesson 3)
- Training loops (lesson 4)
- Advanced indexing, scatter/gather operations
- Memory management, pinned memory, CUDA streams
- Distributed tensors

### Lesson Outline

1. **Context + Constraints** — "This is the first lesson in PyTorch Core. By the end of this module, you'll train a neural network. This lesson: tensors — the data structure everything is built on. We're NOT doing gradients or training yet."

2. **Hook** — Side-by-side: the student's NumPy linear regression data creation (from 1.1.5) next to the identical code in PyTorch. "Spot the differences." (There are almost none.) This is a recognition hook — low activation energy, builds confidence.

3. **Explain: Tensor Basics** — Creating tensors: `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()`. Attributes: `.shape`, `.dtype`, `.device`. Compare each to NumPy equivalent. Code examples are primary — not sidebars.

4. **Check 1** — "Given this NumPy code, rewrite it in PyTorch." (Predict-and-verify with collapsible answer.)

5. **Explain: Shapes and Operations** — Reshaping (`view`, `reshape`), arithmetic operations, matrix multiply (`@` operator, `torch.matmul`). Broadcasting rules (same as NumPy). Shape diagram for 0D through 3D tensors.

6. **Explain: dtypes** — float32 as PyTorch default (vs NumPy's float64). Why: GPU efficiency, gradient precision is approximate anyway. Brief dtype table (float32, float64, int64, bool).

7. **Explain: GPU** — What is a GPU (brief, 2-3 paragraphs). `.to('cuda')` / `.to('cpu')`. Device management pattern: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Show timing comparison: small tensor (CPU wins) vs large tensor (GPU wins).

8. **Check 2** — "Your colleague says GPU is always faster. What would you tell them?" (Transfer question — apply the timing insight.)

9. **Explain: NumPy Interop** — `.numpy()`, `torch.from_numpy()`. Shared memory (modifying one modifies the other). The `.detach().numpy()` pattern (brief mention — "we'll see why in the next lesson").

10. **Practice** — "Open the Colab notebook. Complete exercises: (1) Create training data as tensors, (2) Move to GPU, (3) Implement the forward pass `y_hat = X @ w + b` with tensors." Guided scaffolding — the notebook has starter code.

11. **Summarize** — Key takeaway: "Tensors are NumPy arrays that know where they live. Everything you know about NumPy works on tensors. The new superpower is `.to(device)`." Next: "Now that you have the data structure, the next lesson teaches PyTorch's killer feature: automatic gradient computation."

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings that would leave the student completely lost, but one finding is borderline critical (Check 1 uses untaught API) and three improvement findings would meaningfully strengthen the lesson. Another pass is warranted after fixes.

### Findings

#### [CRITICAL] — Check 1 tests `torch.cat` before it is taught

**Location:** Section 4 (Check Your Understanding)
**Issue:** The exercise asks the student to translate `np.concatenate` to PyTorch. The answer uses `torch.cat` with the `dim` keyword argument. Neither `torch.cat` nor the `axis`->`dim` naming convention has been taught anywhere in the lesson before this point. The "Creating Tensors" section covers `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`, and `torch.from_numpy` — but not `torch.cat`. The student is being tested on something they haven't learned.
**Student impact:** The student attempts the exercise, cannot figure out the PyTorch equivalent of `np.concatenate`, opens the answer, and sees an unfamiliar function. Instead of feeling "I got this!" (the intended BUILD-lesson emotion), they feel "I should have known this?" This undermines the low-activation-energy design of the lesson.
**Suggested fix:** Either (a) teach `torch.cat` and the `axis`->`dim` convention in the "Creating Tensors" or "Shapes and Operations" section before Check 1, or (b) replace Check 1 with an exercise that only uses taught operations (e.g., "Create training data X and y as tensors, print their shape/dtype/device" — testing the three attributes). Option (b) is simpler and better tests what was actually taught. If `torch.cat` is important, move Check 1 to after the "Shapes and Operations" section and add `torch.cat` there.

---

#### [IMPROVEMENT] — Missing planned negative example: GPU vs CPU timing code

**Location:** Section 7 (GPU: The Reason You Switched)
**Issue:** The planning document explicitly calls for a "Small tensor on GPU vs CPU timing" negative example with actual timing code. The built lesson only has a prose callout box ("GPU is not always faster") asserting that small tensors are faster on CPU without showing it empirically. The plan's misconception table says "Show timing: 10-element tensor multiply is faster on CPU." This was not implemented.
**Student impact:** The student reads "GPU is not always faster" as an assertion they should believe, rather than something they can verify. For a software engineer, seeing actual timing numbers (even approximate) is far more convincing than being told. The assertion may not stick without empirical backing.
**Suggested fix:** Add a concrete code example showing a timing comparison. This could be a code block with `time.time()` or `torch.cuda.Event` showing that a 10-element multiply is faster on CPU, while a 10,000,000-element multiply is dramatically faster on GPU. Even simulated output (with a note like "approximate results, try it yourself in the notebook") would be more persuasive than prose alone. Alternatively, make this a Colab exercise so the student runs it themselves.

---

#### [IMPROVEMENT] — `view` vs `reshape` aside introduces "contiguous in memory" without explanation

**Location:** Section 5 (Shapes and Operations), ConceptBlock aside
**Issue:** The aside says "`view()` requires the tensor to be contiguous in memory (fails if not)." The student has no concept of memory contiguity for tensors. The term "contiguous" is introduced without any explanation of what it means or when a tensor would become non-contiguous. This is a jargon drop.
**Student impact:** The student reads "contiguous in memory" and either (a) ignores it because it is meaningless to them, or (b) worries about a concept they cannot reason about. Neither outcome is productive. The aside's practical advice ("when in doubt, use `reshape()`") is good, but the unexplained jargon undermines it.
**Suggested fix:** Either (a) simplify the aside to remove the jargon: "`view()` is slightly faster but can fail after certain operations like `transpose()`. `reshape()` always works. Use `reshape()` unless you have a specific reason to use `view()`." Or (b) add one sentence of explanation: "A tensor is contiguous when its data is stored in a single, unbroken block of memory. Some operations like `transpose()` rearrange the logical view without moving the data, making it non-contiguous." Option (a) is simpler and sufficient for INTRODUCED depth.

---

#### [IMPROVEMENT] — `torch.zeros(3, 4)` vs `np.zeros((3, 4))` argument style difference not called out

**Location:** Section 3 (Creating Tensors)
**Issue:** NumPy shape-creation functions take a tuple: `np.zeros((3, 4))`. PyTorch takes positional arguments: `torch.zeros(3, 4)`. This is a real API difference that causes errors for students switching from NumPy. The lesson shows the PyTorch style but doesn't flag that it differs from NumPy's tuple-based approach. The hook section says "the differences are cosmetic" — this specific difference may surprise the student.
**Student impact:** The student writes `torch.zeros((3, 4))` (NumPy habit), which actually works in PyTorch (it creates a 2-element 1D tensor) but produces the wrong result silently. This is a subtle bug the student could carry into their Colab exercises. Alternatively, they notice the difference and wonder why it changed.
**Suggested fix:** Add a brief note in the creation section or as a TipBlock aside: "Notice: PyTorch takes shape dimensions as separate arguments (`torch.zeros(3, 4)`), not as a tuple like NumPy (`np.zeros((3, 4))`). PyTorch also accepts tuples, but the convention is positional args." One sentence is enough.

---

#### [POLISH] — No explicit transition between "Creating Tensors" and "Check 1"

**Location:** Between sections 3 and 4
**Issue:** The lesson moves from "Creating Tensors" directly to "Check Your Understanding" without a transitional sentence. The check tests not just creation but also concatenation and shape prediction, which goes beyond what was taught. Even if Check 1 is revised (per the Critical finding), a brief transition like "Let's see if you can translate NumPy code to PyTorch using what you just learned" within the flow would help.
**Student impact:** Minor. The student can infer the purpose of the check block. But explicit transitions improve flow.
**Suggested fix:** This is likely handled automatically if Check 1 is revised per the Critical finding. If the check is moved to after "Shapes and Operations," the transition would be natural.

---

#### [POLISH] — `torch.matmul` mentioned in plan but not in lesson

**Location:** Section 5 (Shapes and Operations)
**Issue:** The plan outline says "matrix multiply (`@` operator, `torch.matmul`)." The lesson only shows the `@` operator and never mentions `torch.matmul`. While `@` is the preferred syntax, the student will encounter `torch.matmul` in documentation and tutorials.
**Student impact:** Minimal. The student can discover `torch.matmul` when they encounter it. But a one-line mention would help completeness.
**Suggested fix:** Add a brief note in the matrix multiplication section or as a comment in the code: "The `@` operator is shorthand for `torch.matmul()`. They do the same thing; `@` is more readable." One line is sufficient.

### Review Notes

**What works well:**
- The hook (side-by-side NumPy vs PyTorch) is excellent. It's the right entry point for a BUILD lesson: low activation energy, high recognition, builds confidence immediately.
- The "measuring a rough sketch with a micrometer" analogy for float32 vs float64 is memorable and well-placed.
- The lesson stays within scope boundaries. No scope creep into autograd or nn.Module. The constraints block at the top sets clear expectations.
- The NumPy interop section handles the shared memory surprise well, with a clear callout and practical advice.
- The summary captures the right mental models at the right grain size.
- Code examples are first-class content (not relegated to sidebars), which matches the lesson type. This is a code-learning lesson and the code IS the modality.

**Pattern to watch:**
- The lesson could benefit from more "predict before you see" moments. The Check blocks are the only predict-and-verify points. In a code-heavy lesson, asking "what do you think this outputs?" before showing the output is a cheap, effective way to keep the student active. Consider adding 1-2 inline "what would you expect?" prompts before key code outputs (e.g., before the shared memory surprise in NumPy interop).

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All 6 findings from iteration 1 were addressed correctly. The lesson is effective, well-structured, and ready to ship. One minor polish item noted below.

### Iteration 1 Fix Verification

| Finding | Status | Notes |
|---------|--------|-------|
| CRITICAL: Check 1 tests `torch.cat` before taught | Fixed | Replaced with forward-pass translation exercise (X @ w + b). Tests only taught operations. Predict-and-verify format works well. |
| IMPROVEMENT: Missing GPU timing code | Fixed | Full timing comparison added (`timing_comparison.py`). Shows 10-element (CPU ~12x faster) vs 10M-element (GPU ~350x faster) with `torch.cuda.synchronize()`. Realistic numbers, runnable code. |
| IMPROVEMENT: `view` vs `reshape` jargon | Fixed | Simplified to: "`view()` is slightly faster but can fail after certain operations like `transpose()`. `reshape()` always works." No unexplained jargon. |
| IMPROVEMENT: `torch.zeros` argument style | Fixed | WarningBlock aside added explaining NumPy tuple style vs PyTorch positional args. Clear and concise. |
| POLISH: No transition before Check 1 | Fixed | Transition sentence added: "Let's see if you can translate NumPy to PyTorch using what you just learned." |
| POLISH: `torch.matmul` not mentioned | Fixed | Parenthetical added in matrix multiply section: "The `@` operator is shorthand for `torch.matmul()`—they do the same thing." |

### Findings

#### [POLISH] — NumPy interop shared memory demo could use a "predict" prompt

**Location:** Section 9 (NumPy Interop), the `numpy_interop.py` code block
**Issue:** The code shows `np_array[0] = 99.0` followed by `print(tensor)` revealing `tensor([99., 2., 3.])`. This is a surprising result (shared memory means modifying the NumPy array changes the tensor). The iteration 1 review notes flagged this pattern generally: "the lesson could benefit from more predict-before-you-see moments." This specific example is the best candidate for an inline "what would you expect?" prompt because the surprise is pedagogically valuable—it's a misconception moment (students expect the tensor to be independent of the array).
**Student impact:** Minimal. The lesson works fine as-is. But pausing the student to predict before the reveal would make the shared-memory lesson stickier. A one-line prompt ("What do you think `tensor` contains after modifying `np_array[0]`?") before the print output would activate prediction, which improves retention.
**Suggested fix:** Add a brief inline prompt before the shared memory reveal, or restructure the code block to have the student predict the output before showing it. One sentence is sufficient. This is a small enhancement, not a requirement.

### Review Notes

**What works well (building on iteration 1 observations):**
- All iteration 1 fixes were implemented cleanly. No over-corrections or regressions.
- The new Check 1 exercise is well-designed: it tests the three core attributes (shape, dtype, device), uses only taught operations, and naturally surfaces the float32 vs float64 difference as a preview of the dtypes section.
- The timing comparison code (`timing_comparison.py`) is the strongest improvement. It transforms the "GPU isn't always faster" assertion from something the student believes on authority to something they can verify empirically. The `torch.cuda.synchronize()` calls are correct, and the code is copy-pasteable into the Colab notebook.
- The view/reshape aside is now clean and practical—gives the student a rule ("use `reshape()` unless you have a reason") without requiring them to understand memory layout.
- The `torch.zeros` WarningBlock is placed well as an aside, so it doesn't interrupt the main flow but catches the NumPy habit.

**Overall assessment:**
This lesson achieves what a BUILD lesson should: the student enters with familiar knowledge (NumPy), maps it to new syntax (PyTorch), encounters one genuinely new idea (GPU placement), and leaves with confidence that they can work with tensors. The emotional arc—"I already know this" to "oh, this is the same but better"—is intact. The scope is disciplined (no autograd, no nn.Module, no training loops). The lesson is ready for use.
