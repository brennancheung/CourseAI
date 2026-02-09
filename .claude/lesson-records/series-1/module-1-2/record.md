# Module 1.2: From Linear to Neural — Record

**Goal:** Show why linear models are insufficient, add nonlinearity, demonstrate solving XOR.
**Status:** Complete (4 lessons)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Neuron = weighted sum + bias | DEVELOPED | neuron-basics | Multi-input linear regression; formula + widget |
| Layer = group of neurons | INTRODUCED | neuron-basics | Same inputs, different weight sets; matrix form mentioned |
| Network = stacked layers | INTRODUCED | neuron-basics | Output of one layer → input of next |
| Hidden layers | INTRODUCED | neuron-basics | "Don't observe directly; learn internal representations" |
| "Deep" = many hidden layers | INTRODUCED | neuron-basics | Origin of "deep learning" |
| Linear collapse | DEVELOPED | neuron-basics | Proof: W2(W1x + b1) + b2 = Wx + b. 100 layers = 1 layer |
| Linear separability | INTRODUCED | limits-of-linearity | Can a single line/hyperplane separate two classes? |
| Decision boundary | INTRODUCED | limits-of-linearity | The line between classes; one side = class 0, other = class 1 |
| XOR problem | DEVELOPED | limits-of-linearity | Truth table, diagonal pattern, interactive failure |
| "Not linearly separable" | DEVELOPED | limits-of-linearity | XOR as canonical example; impossible regardless of parameters |
| Classification (framing) | INTRODUCED | limits-of-linearity | Turned XOR into a classification problem |
| Activation function (concept) | DEVELOPED | activation-functions | σ(w·x + b) — nonlinear function after linear combination |
| "Full neuron" formula | DEVELOPED | activation-functions | output = σ(w·x + b), linear then activation |
| Two-line collapse proof | DEVELOPED | activation-functions | Why two linear neurons still give one line |
| Activation creates thresholds | DEVELOPED | activation-functions | ReLU zero/positive behavior = uncollapsible threshold |
| "AND logic" via thresholds | DEVELOPED | activation-functions | Output neuron detects "above line 1 AND below line 2" |
| Space transformation | DEVELOPED | activation-functions | Hidden layer MOVES points to new positions; one line suffices |
| Sigmoid formula and shape | DEVELOPED | activation-functions, activation-functions-deep-dive | σ(x) = 1/(1+e^-x), range (0,1), S-curve |
| ReLU formula and shape | DEVELOPED | activation-functions, activation-functions-deep-dive | max(0,x), range [0,∞), hinge at zero |
| Sigmoid: output-layer use for probability | INTRODUCED | activation-functions | Not for hidden layers anymore |
| ReLU as modern default | INTRODUCED | activation-functions | Won over sigmoid in 2012 deep learning revolution |
| Tanh formula and shape | INTRODUCED | activation-functions-deep-dive | Range (-1,1), zero-centered, used in RNNs |
| Leaky ReLU | INTRODUCED | activation-functions-deep-dive | max(0.01x, x), fix for dying ReLU |
| GELU | INTRODUCED | activation-functions-deep-dive | x·Φ(x), smooth, used in transformers |
| Swish | INTRODUCED | activation-functions-deep-dive | x·σ(x), smooth, used in vision |
| Vanishing gradients | MENTIONED | activation-functions | Small derivatives multiply to near-zero; why sigmoid is bad for deep networks |
| Dying ReLU | MENTIONED | activation-functions | Neurons stuck at 0 permanently |
| Desirable activation properties | INTRODUCED | activation-functions | Nonlinear, differentiable, computationally cheap |

## Per-Lesson Summaries

### neuron-basics
Neuron = multi-input linear regression (weighted sum + bias). Connected to Module 1.1 linear regression. Layer = group of neurons, Network = stacked layers. KEY INSIGHT: linear collapse — W2(W1x + b1) + b2 = Wx + b. Interactive: SingleNeuronExplorer + NetworkDiagramWidget. NOT covered: activation functions (deferred), training, matrix multiplication details.

### limits-of-linearity
XOR as the canonical non-linearly-separable problem. Truth table → plotted as 4 corners → diagonal pattern. Interactive: XORClassifierExplorer (student tries and fails). Decision boundaries: one straight line = one straight cut. Real-world examples (images, language). Previews activation functions as the solution ("two neurons, two lines" hint). NOT covered: how to solve XOR, activation functions, training.

### activation-functions
THE payoff lesson. Proves two-line collapse (two linear neurons = one line). Shows how ReLU thresholding creates uncollapsible regions. "AND logic" via thresholds. Properties of good activations. Sigmoid vs ReLU with ComparisonRows. KEY VISUAL: XORTransformationWidget — hidden layer moves points, one line now suffices. NOT covered: tanh, GELU, Swish, backprop through activations, training.

### activation-functions-deep-dive
Reference guide: sigmoid, tanh, ReLU, Leaky ReLU, GELU, Swish. Each gets formula + range + shape + use. Three instances of ActivationFunctionExplorer. Decision guide: ReLU (default), GELU (transformers), sigmoid (output binary), tanh (RNNs). "Don't overthink" — pick reasonable choice and move on. NOT covered: why some work better (deferred to after backprop), training comparisons.

## Key Mental Models and Analogies

| Model/Analogy | Established In | Used Again In |
|---------------|---------------|---------------|
| "Neuron = multi-input linear regression" | neuron-basics | activation-functions |
| "100-layer linear = 1-layer linear" | neuron-basics | limits-of-linearity, activation-functions |
| "Networks TRANSFORM the space" | activation-functions | (key for future representation learning) |
| "Thresholds enable AND logic" | activation-functions | (key for understanding hidden representations) |
| "ReLU = max(0,x), default choice" | activation-functions | activation-functions-deep-dive |
