# Module 1.1: The Learning Problem — Record

**Goal:** Teach what ML is, how to measure model quality, and how to optimize — from concept to implementation.
**Status:** Complete (6 lessons)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| ML as function approximation | INTRODUCED | what-is-learning | Core framing: "unknown function → approximate from examples" |
| Generalization vs memorization | INTRODUCED | what-is-learning | Study analogy: cramming vs understanding |
| Bias-variance tradeoff (intuitive) | INTRODUCED | what-is-learning | Underfitting/overfitting as "Goldilocks" — no math |
| Train/val/test splits | INTRODUCED | what-is-learning | Textbook/practice-test/real-exam analogy |
| Linear model y = mx + b → y-hat = wx + b | DEVELOPED | linear-regression | Familiar algebra → ML notation |
| Parameters (weight, bias) | DEVELOPED | linear-regression | "The knobs the model learns" |
| Fitting = finding good parameters | INTRODUCED | linear-regression | Intuitive only — no quantitative measure yet |
| ML notation (y-hat, w, b) | INTRODUCED | linear-regression | y-hat = prediction, y = truth, hat = predicted |
| y-hat vs y distinction | INTRODUCED | linear-regression | Difference = error (previews residuals) |
| Residuals (y - y-hat) | DEVELOPED | loss-functions | Formula + visual (red lines in widget) |
| MSE loss function | DEVELOPED | loss-functions | Full formula with KaTeX, squaring rationale |
| Loss landscape | INTRODUCED | loss-functions | Bowl shape, valley = minimum, convex for linear regression |
| Gradient (slope, steepest ascent) | DEVELOPED | gradient-descent | "Ball rolling downhill" analogy |
| Derivatives (calculus) | INTRODUCED | gradient-descent | Positive/negative/zero interpretation, nabla notation |
| Gradient descent update rule | DEVELOPED | gradient-descent | θ_new = θ_old - α∇L, term-by-term breakdown |
| Learning rate (α) | DEVELOPED | learning-rate | Step size, Goldilocks zone, failure modes |
| Hyperparameters | INTRODUCED | learning-rate | "Values YOU choose, not the model" |
| LR failure: oscillation | DEVELOPED | learning-rate | Overshooting minimum, bouncing |
| LR failure: divergence | DEVELOPED | learning-rate | Loss increases, parameters grow unbounded |
| LR schedules | MENTIONED | learning-rate | Step/exponential/cosine/warmup — preview only |
| Convergence | INTRODUCED | gradient-descent | Gradient near zero = flat spot |
| Convexity (bowl shape, one minimum) | MENTIONED | gradient-descent, loss-functions | Noted for linear regression; non-convex for NNs |
| Training loop (forward → loss → backward → update) | DEVELOPED | implementing-linear-regression | 6-step StepList, "heartbeat of training" |
| Gradient formulas for linear reg (dL/dw, dL/db) | DEVELOPED | implementing-linear-regression | Full formulas with KaTeX, -2 factor |
| Forward/backward pass terminology | INTRODUCED | implementing-linear-regression | Used in training loop description |
| NumPy implementation of linear regression | APPLIED | implementing-linear-regression | ~15 lines of code, Colab notebook exercise |
| Chain rule | MENTIONED | implementing-linear-regression | Referenced in TipBlock for gradient derivation |

## Per-Lesson Summaries

### what-is-learning
ML as function approximation. Generalization vs memorization (study analogy). Bias-variance tradeoff (Goldilocks — intuitive only). Post-widget "What Goes Wrong" section maps each fit to memorization framework: underfitting = hasn't learned, good fit = generalization, overfitting = memorization in disguise. Addresses "lower loss = always better" misconception with stock trading example (WarningBlock aside). Train/val/test splits (exam analogy). Interactive: OverfittingWidget. NOT covered: any specific ML algorithms, loss functions, math.

### linear-regression
Linear model as simplest function approximator. Parameters = weight + bias. Fitting = making the line close to data (intuitive). Introduced ML notation (y-hat, w, b). Interactive: LinearFitExplorer (residuals OFF). Ends with open question: "How do we measure goodness?" NOT covered: residuals, loss, optimization.

### loss-functions
Residuals = actual - predicted. MSE = average squared residual ("wrongness score"). Loss landscape as a bowl where training = finding the valley. Interactive: LinearFitExplorer (residuals ON) + LossSurfaceExplorer. NOT covered: how to find the minimum (next lesson), other loss functions.

### gradient-descent
Gradient = direction of steepest ascent (ball-on-hill analogy). Update rule: θ_new = θ_old - α∇L. Learning rate preview (too big/small). Interactive: GradientDescentExplorer with step button and LR slider. NOT covered: multi-parameter gradients, implementation, schedules.

### learning-rate
Deepens LR from preview to full exploration. Hyperparameter concept. Two failure modes: oscillation and divergence. Interactive: LearningRateExplorer (comparison mode + interactive mode). LR schedules previewed (deferred to later). NOT covered: adaptive LR (Adam), per-parameter LR, implementation.

### implementing-linear-regression
Capstone: integrates all concepts into code. Training loop as universal pattern (forward → loss → backward → update). Gradient formulas for w and b. ~15 lines of Python/NumPy. Interactive: TrainingLoopExplorer. Colab notebook exercise. ModuleCompleteBlock. NOT covered: multi-feature regression, batching, stopping criteria.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "Ball rolling downhill" | gradient-descent | learning-rate, implementing-linear-regression |
| "Loss landscape = bowl with valley" | loss-functions | gradient-descent, learning-rate |
| "Training = forward → loss → backward → update" | implementing-linear-regression | (Used in all future modules) |
| "Parameters = knobs the model learns" | linear-regression | gradient-descent, implementing-linear-regression |
| "Goldilocks" (model complexity / LR) | what-is-learning (bias-variance) | learning-rate |
| "Perfect fit trap" — lower loss ≠ always better | what-is-learning (What Goes Wrong) | (Relevant to regularization, early stopping) |
| Stock trading overfitting example | what-is-learning (WarningBlock aside) | (Available to reference in future overfitting discussions) |
| "Wrongness score" for loss | loss-functions | gradient-descent |
