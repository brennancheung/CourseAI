# CourseAI Curriculum

## Learner Context

- Previous deep learning course (pre-transformer era, GAN days)
- High school calculus officially; familiar with more through self-study
- Understands neural network concepts but rusty on details
- Strong programming background
- Goal: Deep intuition, not just surface understanding

## Curriculum Structure

```
Series 1: Fundamentals          (Foundation for everything)
Series 2: Transformers & LLMs   (Deep dive, the main event)
Series 3: Diffusion Models      (Stable Diffusion & beyond)
```

---

# Series 1: Fundamentals

**Goal:** Rebuild intuition for how neural networks actually work. Not just "what" but "why it works."

---

## Module 1.1: The Learning Problem

### 1.1.1 What is Learning?
- Function approximation as the core idea
- Generalization vs memorization
- The bias-variance tradeoff (intuitive explanation)
- Training, validation, test — why three splits?

### 1.1.2 Linear Regression from Scratch
- The simplest "model" — a line through points
- Parameters: slope and intercept
- What does "fitting" mean?
- **Interactive:** Draggable line fitting to data points

### 1.1.3 Loss Functions: Measuring "Wrongness"
- Mean Squared Error (MSE)
  - Formula: $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
  - **Interactive:** Visualize residuals, see MSE change as line moves
- Why squared? (Penalizes large errors, differentiable)
- Loss landscape visualization
  - **Interactive:** 3D surface plot of loss vs parameters

### 1.1.4 Gradient Descent: Following the Slope
- Intuition: Ball rolling downhill
- The gradient points uphill, we go opposite
- Update rule: $\theta_{new} = \theta_{old} - \alpha \nabla L$
  - **Interactive:** Break down each part of equation
- **Interactive:** Animated ball rolling on loss surface

### 1.1.5 Learning Rate Deep Dive
- Too big: Overshooting, divergence
- Too small: Slow convergence, stuck in local minima
- **Interactive:** Slider to adjust LR, watch convergence behavior
- Learning rate schedules (preview)

### 1.1.6 Implementing Linear Regression
- Pure Python/NumPy implementation
- Computing gradients by hand
- Training loop anatomy
- **Colab Notebook:** Linear regression from scratch

---

## Module 1.2: From Linear to Neural

### 1.2.1 The Limits of Linearity
- Linear models can only draw straight decision boundaries
- XOR problem — impossible with one line
- **Interactive:** Try to separate XOR with a line (impossible)

### 1.2.2 Adding Nonlinearity: Activation Functions
- The key insight: compose linear + nonlinear
- A neuron: $output = \sigma(wx + b)$

### 1.2.3 Activation Functions Deep Dive

#### Sigmoid
- Formula: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Output range: (0, 1)
- **Interactive:** Plot with adjustable input, see output
- Pros: Smooth, interpretable as probability
- Cons: Vanishing gradients, not zero-centered
- Gradient: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
  - **Interactive:** Show gradient approaching 0 at extremes

#### Tanh
- Formula: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- Output range: (-1, 1)
- Zero-centered (better than sigmoid)
- Still has vanishing gradient problem
- **Interactive:** Compare sigmoid vs tanh side-by-side

#### ReLU (Rectified Linear Unit)
- Formula: $\text{ReLU}(x) = \max(0, x)$
- **Interactive:** Simple plot, show the "hinge"
- Pros: Fast, no vanishing gradient for positive values
- Cons: "Dying ReLU" — neurons stuck at 0
- Why ReLU won: Computational efficiency + works well

#### Leaky ReLU
- Formula: $\text{LeakyReLU}(x) = \max(\alpha x, x)$ where $\alpha \approx 0.01$
- Fixes dying ReLU problem
- **Interactive:** Adjust α, see effect

#### GELU (Gaussian Error Linear Unit)
- Formula: $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is standard normal CDF
- Used in transformers (BERT, GPT)
- Smooth approximation to ReLU
- **Interactive:** Compare GELU vs ReLU

#### Swish / SiLU
- Formula: $\text{Swish}(x) = x \cdot \sigma(x)$
- Self-gated activation
- Used in many modern architectures

#### Activation Function Comparison
- **Interactive:** All activations on one plot, toggle visibility
- When to use which (decision guide)

### 1.2.4 Multi-Layer Networks
- Stacking layers: input → hidden → output
- Universal approximation theorem (intuition)
- Depth vs width tradeoffs
- **Interactive:** Build network layer by layer, see decision boundary evolve

### 1.2.5 The Forward Pass
- Data flowing through the network
- Matrix multiplication view
- **Interactive:** Step through forward pass, see activations at each layer

---

## Module 1.3: Backpropagation

### 1.3.1 The Chain Rule Refresher
- Composite functions: $f(g(x))$
- Chain rule: $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$
- **Interactive:** Step-by-step derivative of nested functions
- Multiple variables: Partial derivatives

### 1.3.2 Backprop Intuition
- Error signals flowing backward
- Each layer asks: "How much did I contribute to the error?"
- Credit assignment problem
- **Interactive:** Animated gradient flow through network

### 1.3.3 Backprop Math: Simple Example
- 2-layer network, work through every derivative
- Input → Hidden → Output
- Loss with respect to output weights
- Loss with respect to hidden weights
- **Interactive:** Fill in the blanks in derivative chain

### 1.3.4 Backprop Math: General Case
- Notation: $\frac{\partial L}{\partial W^{[l]}}$
- The backward pass algorithm
- Why it's efficient (reusing computations)

### 1.3.5 Computational Graphs
- Operations as nodes, data flows on edges
- How PyTorch/TensorFlow track operations
- Automatic differentiation
- **Interactive:** Build a computational graph, see gradients compute

### 1.3.6 Vanishing and Exploding Gradients
- Why deep networks are hard to train
- Gradient magnitude through many layers
- **Interactive:** Visualize gradient magnitude at each layer
- Solutions preview: Better activations, skip connections, normalization

### 1.3.7 Implementing Backprop from Scratch
- Pure Python/NumPy
- No autograd — compute every gradient
- **Colab Notebook:** Backprop from scratch on tiny network

---

## Module 1.4: Optimization Algorithms

### 1.4.1 Vanilla SGD (Stochastic Gradient Descent)
- Update rule: $\theta = \theta - \alpha \nabla L$
- Stochastic: Using mini-batches instead of full dataset
- Why stochastic? Noise helps escape local minima
- **Interactive:** SGD path on loss surface

### 1.4.2 Momentum
- Intuition: Ball with momentum, builds up speed
- Update rule:
  - $v = \beta v + \nabla L$
  - $\theta = \theta - \alpha v$
- Typical $\beta = 0.9$
- **Interactive:** Compare SGD vs Momentum paths

### 1.4.3 Nesterov Accelerated Gradient (NAG)
- "Look ahead" before computing gradient
- Smarter momentum
- Update rule and intuition

### 1.4.4 AdaGrad
- Adaptive learning rates per parameter
- Parameters that update often get smaller learning rates
- Problem: Learning rate decays to zero

### 1.4.5 RMSprop
- Fix AdaGrad's decaying learning rate
- Exponential moving average of squared gradients
- Update rule breakdown

### 1.4.6 Adam (Adaptive Moment Estimation)
- Combines momentum + RMSprop
- The most popular optimizer
- Update rules:
  - $m = \beta_1 m + (1-\beta_1)\nabla L$ (first moment)
  - $v = \beta_2 v + (1-\beta_2)(\nabla L)^2$ (second moment)
  - Bias correction: $\hat{m} = \frac{m}{1-\beta_1^t}$, $\hat{v} = \frac{v}{1-\beta_2^t}$
  - $\theta = \theta - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$
- **Interactive:** Break down Adam equation, visualize each component
- Default hyperparameters: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

### 1.4.7 AdamW (Adam with Weight Decay)
- Decoupled weight decay
- Why it matters (L2 regularization done right)
- The default for transformers

### 1.4.8 Optimizer Comparison
- **Interactive:** All optimizers on same loss surface, race visualization
- When to use which (decision guide)
- Learning rate sensitivity by optimizer

---

## Module 1.5: Regularization & Generalization

### 1.5.1 Overfitting Visualized
- Training loss goes down, validation loss goes up
- **Interactive:** Train model, watch curves diverge
- Memorization vs learning

### 1.5.2 L2 Regularization (Weight Decay)
- Add penalty for large weights: $L_{total} = L + \lambda \sum w^2$
- Intuition: Prefer simpler models
- Effect on weight distribution
- **Interactive:** Adjust λ, see effect on weights and fit

### 1.5.3 L1 Regularization
- Add penalty: $L_{total} = L + \lambda \sum |w|$
- Encourages sparsity (some weights go to exactly 0)
- L1 vs L2 comparison
- **Interactive:** Compare L1 vs L2 weight distributions

### 1.5.4 Dropout
- Randomly "turn off" neurons during training
- Each forward pass uses different subnetwork
- Dropout rate: Typically 0.1-0.5
- **Interactive:** Visualize dropout masks changing each pass
- Why it works: Ensemble of networks, prevents co-adaptation
- Dropout at test time: Scale activations or use inverted dropout
- Where to apply dropout (between layers, attention)

### 1.5.5 Batch Normalization
- Normalize activations within a batch
- Formula: $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
- Learnable scale (γ) and shift (β)
- Benefits: Faster training, some regularization
- **Interactive:** See activation distributions before/after BatchNorm

### 1.5.6 Layer Normalization
- Normalize across features, not batch
- Critical for transformers (batch size independence)
- BatchNorm vs LayerNorm comparison
- **Interactive:** Visualize difference in normalization dimensions

### 1.5.7 Early Stopping
- Stop training when validation loss stops improving
- Patience hyperparameter
- **Interactive:** Training curve with early stopping point marked

### 1.5.8 Data Augmentation
- Create more training data through transformations
- Image: Flip, rotate, crop, color jitter
- Text: Paraphrase, back-translation
- Why it helps generalization

---

## Module 1.6: Weight Initialization

### 1.6.1 Why Initialization Matters
- Bad init → Vanishing/exploding activations from the start
- **Interactive:** Train same network with different inits

### 1.6.2 Xavier/Glorot Initialization
- For sigmoid/tanh activations
- Formula: $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$
- Intuition: Keep variance stable through layers

### 1.6.3 He Initialization
- For ReLU activations
- Formula: $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$
- Why different from Xavier (ReLU kills half the values)

### 1.6.4 Initialization Comparison
- **Interactive:** Compare activation distributions through layers

---

## Module 1.7: Practical Training

### 1.7.1 Batching and Mini-Batches
- Batch size tradeoffs
- Small batch: Noisy gradients, regularization effect
- Large batch: Stable gradients, faster computation, worse generalization
- Typical sizes: 32, 64, 128, 256

### 1.7.2 Learning Rate Schedules
- Step decay
- Exponential decay
- Cosine annealing
- Warmup + decay
- **Interactive:** Visualize different schedules

### 1.7.3 Gradient Clipping
- Prevent exploding gradients
- Clip by value vs clip by norm
- Common in RNNs and transformers

### 1.7.4 Mixed Precision Training
- FP16 vs FP32
- Speedup with minimal accuracy loss
- Loss scaling for stability

### 1.7.5 MNIST: Putting It All Together
- Classic handwritten digit recognition
- Build and train from scratch
- **Colab Notebook:** Full MNIST training pipeline

---

## Module 1.8: CNNs (Convolutional Neural Networks)

### 1.8.1 Convolution Intuition
- Sliding window over image
- Kernels as feature detectors (edges, corners)
- **Interactive:** Animated kernel sliding, show output feature map

### 1.8.2 Convolution Math
- Formula: $(I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) K(m,n)$
- **Interactive:** Step through convolution computation

### 1.8.3 CNN Building Blocks

#### Convolutional Layer
- Kernel size, stride, padding
- **Interactive:** Adjust parameters, see output size change
- Output size formula: $\frac{W - K + 2P}{S} + 1$

#### Pooling
- Max pooling vs average pooling
- Downsampling, translation invariance
- **Interactive:** Visualize pooling operation

#### Padding
- Valid vs same padding
- Why padding matters (edge information)

### 1.8.4 CNN Architectures Overview
- LeNet (1998) — the original
- AlexNet (2012) — deep learning takes off
- VGG (2014) — deeper, simpler
- ResNet (2015) — skip connections enable very deep networks
- Architecture diagrams for each

### 1.8.5 Skip Connections / Residual Learning
- The key insight: Learn residuals, not mappings
- Formula: $y = F(x) + x$
- Why it helps: Gradient highway, easier optimization
- **Interactive:** Gradient flow with/without skip connections

### 1.8.6 CIFAR-10 Classifier
- 10 classes of small images
- **Colab Notebook:** Build and train CNN

---

# Series 2: Transformers & LLMs

**Goal:** Deep understanding of attention and transformers. This is the main event.

---

## Module 2.1: Why Attention?

### 2.1.1 The Problem with Sequences
- RNNs process sequentially (slow, hard to parallelize)
- Long-range dependencies are hard
- Information bottleneck in encoder-decoder

### 2.1.2 The Attention Intuition
- "Look at relevant parts of the input"
- Database lookup analogy
  - Query: What am I looking for?
  - Keys: What do I have?
  - Values: What do I return?
- **Interactive:** Simple key-value lookup visualization

### 2.1.3 Attention Before Transformers
- Bahdanau attention (2014)
- Luong attention
- How attention helped sequence-to-sequence

---

## Module 2.2: Attention Mechanics

### 2.2.1 Scaled Dot-Product Attention

The core formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Interactive breakdown:**
- $Q$ (Query matrix): What each position is looking for
- $K$ (Key matrix): What each position offers
- $K^T$ (Transpose): Flip for matrix multiplication
- $QK^T$: Similarity scores between all query-key pairs
- $\sqrt{d_k}$: Scaling factor (why we need this)
- $\text{softmax}$: Convert scores to probabilities
- $V$ (Value matrix): The actual content to retrieve
- Final multiplication: Weighted sum of values

### 2.2.2 Why Scale by √d_k?
- Dot products grow with dimension
- Large values → softmax becomes spiky
- Scaling keeps gradients healthy
- **Interactive:** Compare attention weights with/without scaling

### 2.2.3 Attention Weights Visualization
- Heatmap of which tokens attend to which
- **Interactive:** Input a sentence, see attention pattern

### 2.2.4 Implementing Attention from Scratch
- Matrix dimensions step by step
- Pure Python/NumPy implementation
- **Colab Notebook:** Attention from scratch

---

## Module 2.3: Self-Attention

### 2.3.1 Self-Attention Concept
- Query, Key, Value all come from same sequence
- Each token attends to all tokens (including itself)
- **Interactive:** Token attending to all other tokens

### 2.3.2 Computing Q, K, V
- Learned projection matrices: $W_Q$, $W_K$, $W_V$
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- **Interactive:** Visualize projections

### 2.3.3 Multi-Head Attention
- Multiple attention "heads" in parallel
- Each head can learn different patterns
- Formula:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h)W^O$$
where $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- **Interactive:** Compare what different heads learn

### 2.3.4 Why Multiple Heads?
- Different heads capture different relationships
- Some heads: Syntactic patterns
- Some heads: Semantic patterns
- Some heads: Positional patterns
- **Interactive:** Visualize different head attention patterns

### 2.3.5 Masked Self-Attention (Causal)
- For autoregressive models (GPT-style)
- Can only attend to previous tokens
- Mask: Upper triangle of -∞
- **Interactive:** Toggle mask, see attention pattern change

---

## Module 2.4: Transformer Architecture

### 2.4.1 Positional Encoding
- Attention is permutation invariant (no position info)
- Must inject position information

#### Sinusoidal Positional Encoding
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$
- **Interactive:** Visualize positional encoding patterns
- Why sin/cos? Can extrapolate to longer sequences

#### Learned Positional Embeddings
- Just learn a position embedding table
- Used in BERT, GPT

#### Rotary Position Embeddings (RoPE)
- Encode position in rotation of embedding space
- Used in modern LLMs (LLaMA, etc.)
- Better length generalization

### 2.4.2 Layer Normalization
- Normalize across features: $\hat{x} = \frac{x - \mu}{\sigma}$
- Pre-norm vs post-norm (modern models use pre-norm)
- Why LayerNorm not BatchNorm for transformers

### 2.4.3 Feed-Forward Network (FFN)
- Two linear layers with activation
$$FFN(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$
- Expansion ratio: Usually 4x hidden size
- The "thinking" or "memory" part of transformer

### 2.4.4 Residual Connections
- Add input to output of each sublayer
- $output = LayerNorm(x + Sublayer(x))$
- Enables gradient flow in deep networks
- **Interactive:** Gradient flow visualization

### 2.4.5 The Full Transformer Block
```
Input
  ↓
LayerNorm → Multi-Head Attention → + (residual)
  ↓
LayerNorm → Feed-Forward → + (residual)
  ↓
Output
```
- **Interactive:** Full block diagram, click to explore each part

### 2.4.6 Encoder vs Decoder

#### Encoder (BERT-style)
- Bidirectional attention (see all tokens)
- Good for understanding (classification, NER)

#### Decoder (GPT-style)
- Causal/masked attention (only see previous)
- Good for generation

#### Encoder-Decoder (T5-style)
- Encoder processes input
- Decoder generates output with cross-attention
- Good for seq2seq (translation)

### 2.4.7 Building a Tiny Transformer
- From scratch implementation
- Train on simple task (copy, reverse)
- **Colab Notebook:** Tiny transformer from scratch

---

## Module 2.5: Language Modeling

### 2.5.1 Tokenization

#### Word-level
- Simple but huge vocabulary
- Can't handle unknown words

#### Character-level
- Small vocabulary
- Sequences too long

#### Subword Tokenization
- Best of both worlds
- BPE (Byte Pair Encoding)
- WordPiece (BERT)
- SentencePiece (language agnostic)
- **Interactive:** Tokenize text with different methods

### 2.5.2 Token Embeddings
- Lookup table: Token ID → vector
- Learned during training
- **Interactive:** Visualize embedding space (t-SNE/UMAP)

### 2.5.3 The Language Modeling Objective
- Next token prediction
- Loss: Cross-entropy over vocabulary
- Teacher forcing during training

### 2.5.4 Sampling Strategies

#### Greedy Decoding
- Always pick most likely token
- Repetitive, boring output

#### Temperature Sampling
- Divide logits by temperature before softmax
- $T < 1$: More confident (sharper)
- $T > 1$: More random (flatter)
- **Interactive:** Adjust temperature, see output change

#### Top-k Sampling
- Only sample from top k tokens
- **Interactive:** Adjust k, see distribution

#### Top-p (Nucleus) Sampling
- Sample from smallest set with cumulative probability ≥ p
- More dynamic than top-k
- **Interactive:** Adjust p, see cutoff

#### Combining Strategies
- Typical: Temperature 0.7-1.0 + top-p 0.9-0.95

### 2.5.5 Beam Search
- Keep top-k sequences at each step
- Trade-off: Quality vs diversity
- Used more for translation than chat

---

## Module 2.6: Training LLMs

### 2.6.1 Pretraining
- Self-supervised on massive text
- Next token prediction (GPT)
- Masked language modeling (BERT)
- Compute requirements and scaling

### 2.6.2 Scaling Laws
- Chinchilla optimal: 20 tokens per parameter
- Loss predictable from compute/data/params
- **Interactive:** Scaling law curves

### 2.6.3 Fine-Tuning
- Adapt pretrained model to specific task
- Full fine-tuning vs parameter-efficient

### 2.6.4 Instruction Tuning
- Fine-tune to follow instructions
- Dataset: (instruction, response) pairs
- Makes models actually useful

### 2.6.5 RLHF (Reinforcement Learning from Human Feedback)
- Train reward model on human preferences
- Optimize policy with PPO
- Steps:
  1. Supervised fine-tuning
  2. Train reward model
  3. PPO optimization
- **Interactive:** Visualize reward model preferences

### 2.6.6 DPO (Direct Preference Optimization)
- Skip the reward model
- Directly optimize on preferences
- Simpler than RLHF, similar results

### 2.6.7 Constitutional AI
- AI providing its own feedback
- Self-improvement loop

---

## Module 2.7: Parameter-Efficient Fine-Tuning

### 2.7.1 Why Parameter-Efficient?
- Full fine-tuning is expensive
- Storage: One copy per task
- Compute: All parameters updated

### 2.7.2 LoRA (Low-Rank Adaptation)
- Key insight: Weight updates are low-rank
- Instead of updating $W$, learn $W + BA$ where $B$ and $A$ are small
- $B$: (d × r), $A$: (r × d) where r << d
- **Interactive:** Visualize low-rank decomposition
- Rank selection (typically 8-64)
- Which layers to adapt (typically attention)

### 2.7.3 QLoRA
- LoRA + 4-bit quantization
- Fine-tune huge models on consumer GPUs
- NF4 quantization

### 2.7.4 Adapters
- Small bottleneck layers inserted into transformer
- Only train adapter parameters

### 2.7.5 Prompt Tuning / Prefix Tuning
- Learn continuous "soft prompts"
- Prepend learned embeddings

### 2.7.6 Choosing a Method
- LoRA: Best general choice
- QLoRA: When memory constrained
- Full fine-tune: When you have compute and need best quality

---

## Module 2.8: Modern LLM Advances

### 2.8.1 Extending Context Length
- Attention is O(n²) in sequence length
- RoPE extrapolation tricks
- ALiBi (Attention with Linear Biases)
- Sliding window attention

### 2.8.2 Efficient Attention
- Flash Attention (memory-efficient, faster)
- Sparse attention patterns
- Linear attention approximations

### 2.8.3 Mixture of Experts (MoE)
- Multiple FFN "experts", route tokens to subset
- More parameters, same compute
- Mixtral, Switch Transformer

### 2.8.4 Chain of Thought
- Prompting models to reason step-by-step
- Emergent in large models
- Zero-shot CoT: "Let's think step by step"

### 2.8.5 Tool Use / Function Calling
- Models calling external tools
- JSON structured output
- ReAct pattern

### 2.8.6 Multimodal Models
- Vision-language models (GPT-4V, LLaVA)
- Image tokenization / encoders
- Cross-attention between modalities

### 2.8.7 Reasoning Models
- Models that "think longer" (o1-style)
- Test-time compute scaling
- Verification and self-correction

---

# Series 3: Diffusion Models

**Goal:** Understand how Stable Diffusion works and the rapid advances since.

---

## Module 3.1: Generative Models Overview

### 3.1.1 What is Generative Modeling?
- Learning the data distribution
- Sampling new examples

### 3.1.2 Generative Model Families
- Autoregressive (GPT for images: PixelCNN)
- VAEs (Variational Autoencoders)
- Flow-based models
- Diffusion models
- Comparison: Quality, speed, diversity

---

## Module 3.2: Diffusion Intuition

### 3.2.1 The Core Idea
- Forward: Gradually add noise until pure noise
- Reverse: Learn to denoise step by step
- **Interactive:** Watch image dissolve into noise and back

### 3.2.2 The Forward Process (Adding Noise)
- Each step adds a bit of Gaussian noise
- After enough steps: Pure noise
- Mathematically defined process (no learning)
- **Interactive:** Slider through noise levels

### 3.2.3 The Reverse Process (Denoising)
- Learn to predict and remove noise
- Iterative refinement
- Start from pure noise, end with image
- **Interactive:** Step through denoising

### 3.2.4 Why Diffusion Works
- Breaking generation into easy steps
- Each step is a small denoising task
- Accumulated small steps = big transformation

---

## Module 3.3: DDPM (Denoising Diffusion Probabilistic Models)

### 3.3.1 Forward Process Math
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$
- **Interactive:** Break down each term
- $\beta_t$: Noise schedule
- Closed form for any timestep:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

### 3.3.2 Noise Schedules
- Linear schedule (original DDPM)
- Cosine schedule (better for high resolution)
- **Interactive:** Compare schedules visually

### 3.3.3 Training Objective
- Predict the noise that was added
- Simple MSE loss:
$$L = \mathbb{E}_{t,x_0,\epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$$
- **Interactive:** Visualize noise prediction

### 3.3.4 The U-Net Architecture
- Encoder-decoder with skip connections
- Downsampling → bottleneck → upsampling
- Time embedding injection
- Attention layers
- **Interactive:** U-Net architecture diagram

### 3.3.5 Sampling (Inference)
- Start with pure noise
- Iteratively denoise
- DDPM sampling formula
- Slow: Typically 1000 steps

### 3.3.6 Training a Tiny Diffusion Model
- **Colab Notebook:** Train on simple 2D or small image dataset

---

## Module 3.4: Faster Sampling

### 3.4.1 DDIM (Denoising Diffusion Implicit Models)
- Deterministic sampling (given same noise → same image)
- Can skip steps (e.g., 50 steps instead of 1000)
- Trade-off: Speed vs quality

### 3.4.2 Other Fast Samplers
- DPM-Solver
- UniPC
- Euler, Heun methods
- Comparison and when to use each

---

## Module 3.5: Latent Diffusion / Stable Diffusion

### 3.5.1 The Problem with Pixel Space
- High resolution = huge computational cost
- 512×512×3 = 786,432 dimensions

### 3.5.2 The Latent Space Solution
- Compress image with VAE first
- Do diffusion in latent space (much smaller)
- Decode back to pixels at the end
- **Interactive:** Pixel vs latent comparison

### 3.5.3 The VAE (Variational Autoencoder)
- Encoder: Image → Latent
- Decoder: Latent → Image
- Typically 8× downsampling (64×64 latent for 512×512 image)
- Trained separately, frozen during diffusion training

### 3.5.4 Text Conditioning

#### CLIP Text Encoder
- Text → embeddings understood by image models
- Pretrained on image-text pairs
- Rich semantic understanding

#### Cross-Attention
- Text embeddings as Keys and Values
- Image features as Queries
- "Inject" text information into image generation
- **Interactive:** Visualize cross-attention between text and image regions

### 3.5.5 Classifier-Free Guidance (CFG)
- Train model with and without conditioning (dropout)
- At inference: Amplify the conditional signal
$$\epsilon_{guided} = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond})$$
- $s$ = guidance scale (typically 7-15)
- **Interactive:** Adjust CFG, see quality vs diversity trade-off

### 3.5.6 Stable Diffusion Full Pipeline
```
Text → CLIP Encoder → Text Embeddings
                            ↓
Random Noise → U-Net (× N steps) → Latent
                            ↓
            Latent → VAE Decoder → Image
```
- **Interactive:** Full pipeline walkthrough

### 3.5.7 Negative Prompts
- What to avoid in generation
- CFG with negative conditioning

---

## Module 3.6: Fine-Tuning Diffusion Models

### 3.6.1 Full Fine-Tuning
- Update all U-Net parameters
- Expensive, risk of catastrophic forgetting
- When to use: Large datasets, significant style change

### 3.6.2 Textual Inversion
- Learn new "word" embeddings
- Freeze model, only train embeddings
- Good for specific objects/styles with few images

### 3.6.3 DreamBooth
- Fine-tune U-Net + text encoder
- Special token + class noun ("a [V] dog")
- Prior preservation loss
- 3-5 images of subject needed

### 3.6.4 LoRA for Diffusion
- Same low-rank adaptation as LLMs
- Apply to attention layers in U-Net
- Much smaller file sizes
- Easy to combine multiple LoRAs

### 3.6.5 Training a LoRA
- Dataset preparation
- Hyperparameters (rank, learning rate, steps)
- **Colab Notebook:** Train a LoRA on custom images

---

## Module 3.7: Controlled Generation

### 3.7.1 ControlNet
- Add spatial conditioning (pose, edges, depth)
- Locked copy of U-Net encoder
- Trainable copy + zero convolutions
- **Interactive:** Different control types visualization

### 3.7.2 Control Types
- Canny edges
- OpenPose (human poses)
- Depth maps
- Segmentation maps
- Scribbles/sketches

### 3.7.3 IP-Adapter
- Image prompt instead of (or with) text
- Use image as style/subject reference

### 3.7.4 Inpainting
- Fill in masked regions
- Conditioned on surrounding context

### 3.7.5 Outpainting
- Extend image beyond original boundaries

### 3.7.6 Image-to-Image
- Start from existing image + noise
- Control strength of transformation

---

## Module 3.8: Advanced Diffusion

### 3.8.1 SDXL
- Larger U-Net, two text encoders
- Refiner model for detail
- Better prompt following

### 3.8.2 Consistency Models
- Distill diffusion into one/few step generation
- Trade training for inference speed

### 3.8.3 Rectified Flow / Flow Matching
- Simpler training objective
- Straight paths in latent space
- Basis for newer models

### 3.8.4 DiT (Diffusion Transformers)
- Replace U-Net with transformer
- Better scaling properties
- Used in Sora, etc.

### 3.8.5 Video Diffusion
- Extending to temporal dimension
- 3D U-Net or temporal attention
- Sora, Runway, etc.

---

## Module 3.9: Flux Deep Dive (Optional)

*In-depth analysis of the Flux architecture*

### 3.9.1 Flux Overview
- Architecture differences from SD
- Key innovations

### 3.9.2 Architecture Walkthrough
- Component by component analysis
- **Colab Notebook:** Explore Flux code in transformers library

### 3.9.3 Training Approach
- How Flux was trained
- Differences from previous models

### 3.9.4 Using Flux
- Generation examples
- Fine-tuning considerations

---

# Future Topics

*Ideas for additional content — expand based on interest*

## GANs (Generative Adversarial Networks)
- Historical significance
- Generator vs Discriminator
- GAN training dynamics
- Mode collapse
- Key architectures: DCGAN, StyleGAN
- Why diffusion "won"

## Vision Transformers (ViT)
- Patch embedding
- Applying transformers to images
- MAE (Masked Autoencoders)
- DINO, DINOv2

## Audio/Speech
- Whisper architecture
- Text-to-speech models
- Music generation

## Reinforcement Learning
- RL basics
- Policy gradient
- PPO in depth
- RLHF implementation details

## Mechanistic Interpretability
- What do neurons represent?
- Probing classifiers
- Activation patching
- Circuit analysis

## Quantization
- INT8, INT4, NF4
- GPTQ, AWQ, GGML
- When and why to quantize

## Inference Optimization
- KV cache
- Speculative decoding
- Continuous batching
- vLLM, TensorRT-LLM
