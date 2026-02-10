# Series 2: PyTorch — Plan

**Status:** Complete
**Prerequisites:** Series 1 (Foundations)

## Series Goal

Bridge theory to practice. The student understands backprop, optimizers, and training dynamics conceptually — now they build and train real models in PyTorch. Every lesson has a Colab notebook.

## Modules and Lessons

### Module 2.1: PyTorch Core

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 1 | tensors | Tensors | Yes | NumPy → Tensor, GPU, dtype, shapes, broadcasting |
| 2 | autograd | Autograd | Yes | `requires_grad`, `loss.backward()`, connects to computational graphs (1.3) |
| 3 | nn-module | nn.Module | Yes | Parameters, layers, `forward()`, building a network |
| 4 | training-loop | Training Loop in PyTorch | Yes | Reimplement Foundations linear regression + a neural net with `optim.Adam` |

### Module 2.2: Real Data

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 5 | datasets-and-dataloaders | Datasets and DataLoaders | Yes | `Dataset`, `DataLoader`, batching (connects to SGD lesson), transforms |
| 6 | mnist-project | Project: MNIST | Yes | End-to-end: load data, build model, train, evaluate. First real result. |
| 7 | debugging-and-visualization | Debugging and Visualization | Yes | Shape errors, gradient checking, TensorBoard/logging, `torchinfo` |

### Module 2.3: Practical Patterns

| # | Slug | Title | Notebook | Description |
|---|------|-------|----------|-------------|
| 8 | saving-and-loading | Saving, Loading, and Checkpoints | Yes | `state_dict`, resuming training, model export |
| 9 | gpu-training | GPU Training | Yes | `.to(device)`, mixed precision basics, when it matters |
| 10 | fashion-mnist-project | Project: Fashion-MNIST | Yes | Pull it all together, beat a baseline, experiment |

## Scope Boundaries

**In scope:**
- Core PyTorch APIs (tensors, autograd, nn, optim, data)
- Training loops, evaluation, saving/loading
- Basic debugging and visualization
- GPU basics

**Out of scope:**
- Distributed training
- Custom autograd functions
- TorchScript / model compilation
- Domain-specific libraries (torchvision models, torchaudio, etc. — covered in later series)
- Hyperparameter tuning frameworks

## Notebook Convention

Notebooks stored in `notebooks/` with naming pattern: `{series}-{module}-{lesson}-{slug}.ipynb`

Example: `2-1-1-tensors.ipynb`

Linked from lessons via: `https://colab.research.google.com/github/{user}/{repo}/blob/main/notebooks/{filename}.ipynb`

## Connections to Foundations

| PyTorch Concept | Foundations Concept | Lesson |
|----------------|--------------------|----|
| `tensor.backward()` | Computational graphs (1.3.3) | 2 |
| `optim.Adam` | Optimizers (1.3.5) | 4 |
| `DataLoader(batch_size=32)` | Batching and SGD (1.3.4) | 5 |
| `nn.BatchNorm1d` | Batch normalization (1.3.6) | 6 |
| `nn.Dropout` | Regularization (1.3.7) | 6 |
