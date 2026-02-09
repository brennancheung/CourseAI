# Series 1: Foundations — Summary

**Status:** Complete (all 3 modules, 17 lessons)

## Series Goal

Build the complete conceptual and practical foundation for deep learning: from "what is ML?" through linear models, neural network architecture, activation functions, backpropagation, optimization, and regularization. The series completes the full arc from "what is learning?" (lesson 1) to "how to make sure networks learn the right things" (lesson 17).

## Rolled-Up Concept List

### From Module 1.1: The Learning Problem

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| ML as function approximation | INTRODUCED | "Unknown function → approximate from examples" |
| Generalization vs memorization | INTRODUCED | Study analogy: cramming vs understanding |
| Bias-variance tradeoff | INTRODUCED | Intuitive only (Goldilocks — too simple/too complex) |
| Train/val/test splits | INTRODUCED | Exam analogy: textbook/practice/real |
| Linear model y-hat = wx + b | DEVELOPED | Parameters = learnable knobs |
| MSE loss function | DEVELOPED | "Wrongness score" — formula + interactive |
| Loss landscape | INTRODUCED | Bowl shape, valley = minimum |
| Gradient descent | DEVELOPED | Ball-on-hill, update rule θ_new = θ_old - α∇L |
| Learning rate | DEVELOPED | Goldilocks zone, oscillation/divergence failure modes |
| Training loop | DEVELOPED | Forward → loss → backward → update (universal pattern) |
| From-scratch implementation | APPLIED | NumPy linear regression, ~15 lines |

### From Module 1.2: From Linear to Neural

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Neuron = weighted sum + bias | DEVELOPED | "Multi-input linear regression" |
| Layers and networks | INTRODUCED | Groups of neurons, stacked, hidden layers |
| Linear collapse | DEVELOPED | Proof: stacking linear = still one linear transform |
| Linear separability | INTRODUCED | Can one line/hyperplane separate classes? |
| XOR impossibility | DEVELOPED | Diagonal pattern, interactive failure |
| Activation functions | DEVELOPED | σ(w·x + b) — nonlinear after linear |
| ReLU | DEVELOPED | max(0,x), modern default, fast |
| Sigmoid | DEVELOPED | 1/(1+e^-x), output-layer for probability |
| Thresholds / AND logic | DEVELOPED | How activation enables solving XOR |
| Space transformation | DEVELOPED | Hidden layers MOVE points, not just draw more lines |
| Tanh, Leaky ReLU, GELU, Swish | INTRODUCED | Reference-level: shapes, ranges, when to use |

### From Module 1.3: Training Neural Networks

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Chain rule | DEVELOPED | dy/dx = dy/dg · dg/dx; effects multiply |
| Backpropagation | DEVELOPED | Forward + backward pass = all gradients efficiently |
| Vanishing gradients | DEVELOPED | 0.25^N layer-by-layer decay; sigmoid derivative max=0.25; telephone game analogy |
| Exploding gradients | DEVELOPED | Mirror of vanishing: large factors multiply to infinity; NaN symptom |
| Computational graph notation | DEVELOPED | Operations = nodes, data flow = edges; visual chain rule |
| Fan-out gradient summation | DEVELOPED | When a value feeds multiple paths, sum the gradients |
| Automatic differentiation | INTRODUCED | PyTorch builds graph during forward, walks it backward for loss.backward() |
| Mini-batch gradient computation | DEVELOPED | Polling analogy: random sample estimates full gradient; same formula, fewer data points |
| Stochastic gradient descent (SGD) | DEVELOPED | Mini-batch SGD is the default; spectrum from batch=1 to batch=ALL |
| Epochs | DEVELOPED | One pass through all data; iterations per epoch = N/B |
| Gradient noise as beneficial | INTRODUCED | Noisy gradients help escape sharp minima; "the hill is shaking" |
| Batch size as hyperparameter | DEVELOPED | Common values 32-256; tradeoff between accuracy and update frequency |
| Sharp vs wide minima | INTRODUCED | Wide minima generalize better; noise helps find them |
| Xavier initialization | DEVELOPED | Var(w) = 1/n_in; preserves signal variance; for sigmoid/tanh |
| He initialization | DEVELOPED | Var(w) = 2/n_in; accounts for ReLU zeroing ~50% of neurons |
| Batch normalization | INTRODUCED | Normalize activations between layers during training; learned gamma/beta |
| Overfitting / generalization (operationalized) | DEVELOPED | From INTRODUCED (1.1); training curve diagnostic, model capacity framing, "scissors" pattern |
| Training curves as diagnostic tool | DEVELOPED | Plot train + val loss; three phases (learning, sweet spot, overfitting); gap = overfitting |
| Dropout | DEVELOPED | Randomly silence neurons during training; implicit ensemble; p=0.5 default |
| Weight decay / L2 regularization | DEVELOPED | Penalty on large weights; modified update rule with decay factor; AdamW as default |
| Early stopping | DEVELOPED | Monitor validation loss; patience hyperparameter; save best model weights |
| Regularization (general concept) | INTRODUCED | Constrains models to prevent memorization; increases training loss to improve validation |
| Model capacity / expressiveness | INTRODUCED | More parameters = more capacity; parameter-to-data ratio matters |
| Gradient clipping | MENTIONED | Cap gradient magnitude as safety net for exploding gradients |
| Skip connections / ResNets | MENTIONED | Teased for future module; enabled 152-layer networks |
| AdamW | MENTIONED | Adam + weight decay; practical default optimizer; deferred to PyTorch |

## Key Mental Models Carried Forward

1. **"ML is function approximation"** — The foundational frame for everything
2. **"Training loop = forward → loss → backward → update"** — Universal pattern
3. **"Parameters are knobs the model learns"** — Applies from linear regression to GPT
4. **"Ball rolling downhill"** — Physical intuition for gradient descent
5. **"Loss landscape = bowl/surface to minimize"** — Geometric optimization frame
6. **"Networks transform space"** — How hidden layers enable solving complex problems
7. **"Effects multiply through the chain"** — Chain rule as the engine of backprop
8. **"Local × Local × Local"** — Each layer computes independently; chain rule composes
9. **"The graph IS the chain rule, drawn instead of written"** — Computational graphs are notation, not a new algorithm
10. **"Fan-out = sum the gradients"** — When a value feeds multiple paths, total gradient is the sum
11. **"No path = no gradient = doesn't learn"** — Graph structure determines which parameters get gradients
12. **"Polling analogy: random sample estimates the whole"** — Mini-batch gradient approximates full gradient, like polling 50 people estimates a city
13. **"The ball is still rolling downhill, but now the hill is shaking"** — Extends ball-on-hill; mini-batch noise = helpful shaking
14. **"Noise as a feature, not a bug"** — Gradient noise escapes sharp minima into wide ones that generalize better
15. **"Telephone game for gradient flow"** — Each layer passes a message (gradient); fidelity lost = vanishing, amplified = exploding
16. **"Products of local derivatives: the stability question"** — Vanishing and exploding are the same problem; product is only stable when each factor is near 1.0
17. **"Each layer should preserve the signal"** — The design principle behind Xavier/He initialization
18. **"Flatline = vanishing, NaN = exploding"** — Diagnostic symptom guide for training failures
19. **"ReLU + He init + batch norm = modern baseline"** — The practical starting point for any new deep network
20. **"The scissors pattern"** — When train and val loss diverge, the model is overfitting; every regularization technique aims to keep scissors closed
21. **"Training and validation are different landscapes"** — Reframes ball-on-hill; ball can reach bottom of training while climbing UP validation
22. **"Regularization increases training loss — that is the point"** — Training loss alone was never the goal; higher training loss with lower validation loss = better model
23. **"The complete training recipe"** — Xavier/He init + batch norm + AdamW + dropout (if needed) + early stopping; the modern baseline

## What This Series Does NOT Cover

- Multi-class classification (softmax, cross-entropy)
- L1 regularization / Lasso (mentioned as alternative to L2)
- Data augmentation as regularization
- Skip connections / residual networks (mentioned only)
- Convolutional and recurrent architectures
- Transformers and attention
- PyTorch implementation (AdamW named but not implemented; all techniques conceptual only)
- Hyperparameter tuning strategies (grid search, random search, etc.)
- Any specific application domain
