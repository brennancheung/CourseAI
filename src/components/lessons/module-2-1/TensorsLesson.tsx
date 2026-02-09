'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  ConceptBlock,
  SummaryBlock,
  ConstraintBlock,
  ComparisonRow,
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

/**
 * Tensors — Lesson 1 of Module 2.1 (PyTorch Core)
 *
 * Teaches the student to:
 * - Create and manipulate PyTorch tensors
 * - Understand tensor attributes: shape, dtype, device
 * - Move tensors between CPU and GPU
 * - Convert between NumPy arrays and tensors
 *
 * Target depths:
 * - PyTorch tensor API: DEVELOPED
 * - GPU device management: INTRODUCED
 * - PyTorch dtypes: INTRODUCED
 * - NumPy interop: DEVELOPED
 *
 * NO custom interactive widget — the Colab notebook IS the interactive element.
 * Code examples are first-class, not sidebars.
 */

export function TensorsLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Tensors"
            description="PyTorch's core data structure — NumPy arrays that can ride the GPU."
            category="PyTorch Core"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Create, manipulate, and move PyTorch tensors. By the end, you can
            rewrite any NumPy code from the Foundations series in PyTorch&mdash;and
            run it on a GPU with a single line change.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="New Series">
            This is the first lesson in PyTorch Core. Everything here builds on
            the NumPy skills you developed in Linear Regression from Scratch.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'Tensors only — no autograd or requires_grad (that\'s the next lesson)',
              'No nn.Module or layers — those come in lesson 3',
              'No training loops — we\'re learning the data structure first',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook — Side-by-side NumPy vs PyTorch */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Spot the Differences"
            subtitle="Your NumPy code, translated to PyTorch"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Linear Regression from Scratch, you wrote NumPy code to create
              training data and compute predictions. Here is that same code in
              PyTorch. See if you can spot what changed:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">NumPy</p>
                <CodeBlock
                  language="python"
                  code={`import numpy as np

# Create training data
X = np.random.randn(100, 1)
y = 2 * X + 0.5 + np.random.randn(100, 1) * 0.1

# Initialize parameters
w = np.zeros((1, 1))
b = np.zeros(1)

# Forward pass
y_pred = X @ w + b`}
                />
              </div>
              <div>
                <p className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">PyTorch</p>
                <CodeBlock
                  language="python"
                  code={`import torch

# Create training data
X = torch.randn(100, 1)
y = 2 * X + 0.5 + torch.randn(100, 1) * 0.1

# Initialize parameters
w = torch.zeros(1, 1)
b = torch.zeros(1)

# Forward pass
y_pred = X @ w + b`}
                />
              </div>
            </div>

            <p className="text-muted-foreground">
              The differences are cosmetic: <code className="text-sm bg-muted px-1.5 py-0.5 rounded">np</code> becomes <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch</code>,
              and the function names are slightly different (<code className="text-sm bg-muted px-1.5 py-0.5 rounded">randn</code> instead
              of <code className="text-sm bg-muted px-1.5 py-0.5 rounded">random.randn</code>). The <code className="text-sm bg-muted px-1.5 py-0.5 rounded">@</code> operator,
              broadcasting, and shapes all work exactly the same way.
            </p>
            <p className="text-muted-foreground">
              So why bother switching? Because tensors can do something NumPy arrays
              cannot: <strong>move to a GPU</strong>. Add one line&mdash;<code className="text-sm bg-muted px-1.5 py-0.5 rounded">X = X.to(&apos;cuda&apos;)</code>&mdash;and
              your matrix multiplications run on thousands of parallel cores instead
              of your CPU. That is the superpower.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Interface, Different Engine">
            PyTorch intentionally mirrors NumPy&apos;s API. If you know NumPy, you
            already know 80% of tensor operations. The new 20% is device
            management and autograd (next lesson).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Explain: Tensor Basics */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Creating Tensors"
            subtitle="Every way to make a tensor"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              There are four common ways to create tensors. Each has a NumPy
              equivalent you already know:
            </p>

            <CodeBlock
              language="python"
              filename="tensor_creation.py"
              code={`import torch

# 1. From raw data (like np.array)
t = torch.tensor([1.0, 2.0, 3.0])

# 2. Zeros and ones (like np.zeros, np.ones)
zeros = torch.zeros(3, 4)    # 3x4 matrix of zeros
ones = torch.ones(2, 3)      # 2x3 matrix of ones

# 3. Random values (like np.random.randn)
rand_normal = torch.randn(5, 3)   # standard normal
rand_uniform = torch.rand(5, 3)   # uniform [0, 1)

# 4. From a NumPy array (bridge between worlds)
import numpy as np
np_array = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(np_array)`}
            />

            <p className="text-muted-foreground">
              Every tensor has three attributes you need to know:
            </p>

            <CodeBlock
              language="python"
              code={`t = torch.randn(3, 4)

t.shape    # torch.Size([3, 4]) — dimensions
t.dtype    # torch.float32      — data type
t.device   # device(type='cpu') — where it lives`}
            />

            <p className="text-muted-foreground">
              These three attributes&mdash;shape, dtype, device&mdash;determine
              everything about how a tensor behaves. When something goes wrong in
              PyTorch, check these first. Shape mismatches are the most common bug;
              device mismatches are the most confusing.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Shape Is King">
            The single most useful debugging skill in PyTorch: print
            the <code className="text-xs">.shape</code> of every tensor. Shape
            mismatches cause 90% of beginner errors. When confused, print shapes.
          </TipBlock>
          <WarningBlock title="Gotcha: Shape Arguments">
            NumPy takes a tuple: <code className="text-xs">np.zeros((3, 4))</code>.
            PyTorch takes separate args: <code className="text-xs">torch.zeros(3, 4)</code>.
            PyTorch also accepts tuples, but the convention is positional arguments.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 4. Check 1 — Predict and Verify */}
      <Row>
        <Row.Content>
          <p className="text-muted-foreground">
            Let&apos;s see if you can translate NumPy to PyTorch using what you just
            learned.
          </p>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              Translate this NumPy code to PyTorch. Before opening the answer,
              predict the shape, dtype, and device of the result.
            </p>
            <CodeBlock
              language="python"
              code={`import numpy as np
X = np.random.randn(50, 3)
w = np.zeros((3, 1))
b = np.ones(1)
y_hat = X @ w + b
print(y_hat.shape, y_hat.dtype)  # ?`}
            />
            <details className="group mt-4">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <CodeBlock
                  language="python"
                  code={`import torch
X = torch.randn(50, 3)
w = torch.zeros(3, 1)
b = torch.ones(1)
y_hat = X @ w + b
print(y_hat.shape, y_hat.dtype)
# torch.Size([50, 1])  torch.float32`}
                />
                <p className="text-muted-foreground">
                  The shape is <code className="text-xs bg-muted px-1 rounded">[50, 1]</code> (50
                  samples, 1 prediction each). The dtype
                  is <code className="text-xs bg-muted px-1 rounded">float32</code>&mdash;PyTorch&apos;s
                  default, not NumPy&apos;s <code className="text-xs bg-muted px-1 rounded">float64</code>.
                  The device
                  is <code className="text-xs bg-muted px-1 rounded">cpu</code> (we haven&apos;t moved
                  anything to GPU yet).
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 5. Explain: Shapes and Operations */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Shapes and Operations"
            subtitle="Reshaping, arithmetic, and matrix multiply"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Tensors come in different dimensions. Here is the vocabulary:
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border bg-card p-4 space-y-2">
                <p className="font-semibold text-sm">0D: Scalar</p>
                <CodeBlock language="python" code={`loss = torch.tensor(0.5)
# shape: torch.Size([])  — no dimensions`} />
              </div>
              <div className="rounded-lg border bg-card p-4 space-y-2">
                <p className="font-semibold text-sm">1D: Vector</p>
                <CodeBlock language="python" code={`bias = torch.tensor([1.0, 2.0, 3.0])
# shape: torch.Size([3])`} />
              </div>
              <div className="rounded-lg border bg-card p-4 space-y-2">
                <p className="font-semibold text-sm">2D: Matrix</p>
                <CodeBlock language="python" code={`weights = torch.randn(3, 4)
# shape: torch.Size([3, 4])`} />
              </div>
              <div className="rounded-lg border bg-card p-4 space-y-2">
                <p className="font-semibold text-sm">3D: Batch of matrices</p>
                <CodeBlock language="python" code={`batch = torch.randn(32, 28, 28)
# shape: torch.Size([32, 28, 28])`} />
              </div>
            </div>

            <p className="text-muted-foreground">
              <strong>Reshaping</strong> changes the dimensions without changing the data.
              The most common operation: flattening an image into a vector for a neural
              network.
            </p>

            <CodeBlock
              language="python"
              filename="reshaping.py"
              code={`# A batch of 32 images, each 28x28 pixels
images = torch.randn(32, 28, 28)

# Flatten each image into a 784-element vector
flat = images.view(32, -1)    # -1 means "figure it out"
print(flat.shape)             # torch.Size([32, 784])

# reshape() works the same way
flat = images.reshape(32, -1) # torch.Size([32, 784])`}
            />

            <p className="text-muted-foreground">
              <strong>Arithmetic</strong> works element-wise, just like NumPy:
            </p>

            <CodeBlock
              language="python"
              code={`a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

a + b    # tensor([5., 7., 9.])
a * b    # tensor([4., 10., 18.])  — element-wise!
a ** 2   # tensor([1., 4., 9.])`}
            />

            <p className="text-muted-foreground">
              <strong>Matrix multiplication</strong> uses the <code className="text-sm bg-muted px-1.5 py-0.5 rounded">@</code> operator&mdash;the
              same one you used in your NumPy linear regression. (The <code className="text-sm bg-muted px-1.5 py-0.5 rounded">@</code> operator
              is shorthand for <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.matmul()</code>&mdash;they
              do the same thing.)
            </p>

            <CodeBlock
              language="python"
              code={`# The forward pass: y_hat = X @ w + b
X = torch.randn(100, 3)   # 100 samples, 3 features
w = torch.randn(3, 1)     # 3 weights
b = torch.zeros(1)        # 1 bias

y_hat = X @ w + b          # shape: (100, 1)
# Broadcasting adds b (shape [1]) to every row`}
            />

            <p className="text-muted-foreground">
              Broadcasting works identically to NumPy: the bias
              vector <code className="text-sm bg-muted px-1.5 py-0.5 rounded">b</code> with
              shape <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[1]</code> is
              automatically expanded to match the <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[100, 1]</code> result
              of <code className="text-sm bg-muted px-1.5 py-0.5 rounded">X @ w</code>. Same
              rules, same behavior.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="view vs reshape">
            <code className="text-xs">view()</code> is slightly faster but can fail after
            certain operations like <code className="text-xs">transpose()</code>.{' '}
            <code className="text-xs">reshape()</code> always works. Use{' '}
            <code className="text-xs">reshape()</code> unless you have a specific reason
            to use <code className="text-xs">view()</code>.
          </ConceptBlock>
          <WarningBlock title="* Is Not Matmul">
            <code className="text-xs">a * b</code> is element-wise multiplication.
            <code className="text-xs"> a @ b</code> is matrix multiplication. Confusing them
            is a silent bug&mdash;the code runs but gives wrong answers.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 6. Explain: dtypes */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Data Types"
            subtitle="Why PyTorch defaults to float32, not float64"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              NumPy defaults to <code className="text-sm bg-muted px-1.5 py-0.5 rounded">float64</code> (double
              precision). PyTorch defaults to <code className="text-sm bg-muted px-1.5 py-0.5 rounded">float32</code> (single
              precision). This is not an accident.
            </p>
            <p className="text-muted-foreground">
              In deep learning, gradients come from random mini-batches&mdash;they are
              approximate by design. Using 64-bit precision to store an approximate
              gradient is like measuring a rough sketch with a micrometer. You get
              2x the memory cost and 2x the computation time for precision you
              cannot actually use.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-muted-foreground border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 pr-4 font-semibold">dtype</th>
                    <th className="text-left py-2 pr-4 font-semibold">Bits</th>
                    <th className="text-left py-2 pr-4 font-semibold">When to Use</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4"><code className="text-xs bg-muted px-1 rounded">torch.float32</code></td>
                    <td className="py-2 pr-4">32</td>
                    <td className="py-2 pr-4">Default for everything. Model parameters, data, gradients.</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4"><code className="text-xs bg-muted px-1 rounded">torch.float64</code></td>
                    <td className="py-2 pr-4">64</td>
                    <td className="py-2 pr-4">Scientific computing. Rarely needed in deep learning.</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4"><code className="text-xs bg-muted px-1 rounded">torch.float16</code></td>
                    <td className="py-2 pr-4">16</td>
                    <td className="py-2 pr-4">Mixed precision training (saves GPU memory on large models).</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4"><code className="text-xs bg-muted px-1 rounded">torch.int64</code></td>
                    <td className="py-2 pr-4">64</td>
                    <td className="py-2 pr-4">Indices, class labels, token IDs.</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4"><code className="text-xs bg-muted px-1 rounded">torch.bool</code></td>
                    <td className="py-2 pr-4">8</td>
                    <td className="py-2 pr-4">Masks (attention masks, padding masks).</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <CodeBlock
              language="python"
              code={`# PyTorch defaults to float32
t = torch.tensor([1.0, 2.0, 3.0])
print(t.dtype)  # torch.float32

# Force float64 (rarely needed)
t64 = torch.tensor([1.0, 2.0], dtype=torch.float64)

# Convert between dtypes
t32 = t64.float()   # float64 -> float32
t16 = t32.half()    # float32 -> float16

# Integer tensor (for labels/indices)
labels = torch.tensor([0, 1, 2, 1, 0])
print(labels.dtype)  # torch.int64`}
            />

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5">
              <p className="text-amber-400 font-semibold mb-2">
                float32 is not &ldquo;less precise.&rdquo; It is right-sized.
              </p>
              <p className="text-sm text-muted-foreground">
                float32 has ~7 decimal digits of precision. Mini-batch gradients
                have far less signal than that. Using float64 is paying double for
                storage you cannot use. GPUs are optimized for float32 (and float16)&mdash;float64
                runs 2&ndash;32x slower depending on the hardware.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="dtype Mismatch Bug">
            If you create a tensor from a Python int list, it defaults
            to <code className="text-xs">int64</code>. If you try to multiply it with
            a <code className="text-xs">float32</code> tensor, you get an error. Fix
            it: <code className="text-xs">torch.tensor([1, 2, 3], dtype=torch.float32)</code> or <code className="text-xs">.float()</code>.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 7. Explain: GPU */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="GPU: The Reason You Switched"
            subtitle="Moving tensors to massively parallel hardware"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your CPU has 8&ndash;16 cores. A modern GPU has thousands. A CPU core
              is fast at sequential work (do this, then this, then this). A GPU core
              is slower individually, but there are so many that it wins overwhelmingly
              at parallel work (multiply these 10,000 numbers all at once).
            </p>
            <p className="text-muted-foreground">
              Neural network training is almost entirely parallel math: matrix
              multiplications, element-wise operations, reductions. This is exactly
              what GPUs are built for. A training run that takes 10 hours on CPU
              might take 20 minutes on GPU.
            </p>

            <ComparisonRow
              left={{
                title: 'CPU',
                color: 'blue',
                items: [
                  '8–16 powerful cores',
                  'Fast at sequential logic',
                  'Good for: small tensors, data loading, preprocessing',
                  'Always available',
                ],
              }}
              right={{
                title: 'GPU (CUDA)',
                color: 'emerald',
                items: [
                  'Thousands of simple cores',
                  'Fast at parallel math',
                  'Good for: large matrix multiplications, training',
                  'Requires NVIDIA GPU + CUDA',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Moving a tensor to the GPU is one line:
            </p>

            <CodeBlock
              language="python"
              filename="device_management.py"
              code={`# The standard device pattern
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)  # 'cuda' if you have a GPU, 'cpu' otherwise

# Move tensors to the device
X = torch.randn(1000, 784)
X = X.to(device)
print(X.device)  # device(type='cuda', index=0)

# Create tensors directly on the device
w = torch.randn(784, 128, device=device)

# Move back to CPU (e.g., for NumPy conversion)
X_cpu = X.cpu()`}
            />

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5">
              <p className="text-amber-400 font-semibold mb-2">
                GPU is not always faster.
              </p>
              <p className="text-sm text-muted-foreground">
                Moving data between CPU and GPU takes time (the &ldquo;transfer
                overhead&rdquo;). For small tensors, this overhead is larger than
                the computation itself. A 10-element multiplication is faster on CPU.
                GPU wins when tensors are large enough (roughly 10,000+ elements)
                that the parallel speedup exceeds the transfer cost.
              </p>
            </div>

            <CodeBlock
              language="python"
              filename="timing_comparison.py"
              code={`import time

# Small tensor: CPU wins
a_cpu = torch.randn(10)
b_cpu = torch.randn(10)

start = time.time()
for _ in range(10000):
    _ = a_cpu * b_cpu
cpu_small = time.time() - start

a_gpu = a_cpu.to('cuda')
b_gpu = b_cpu.to('cuda')
torch.cuda.synchronize()

start = time.time()
for _ in range(10000):
    _ = a_gpu * b_gpu
torch.cuda.synchronize()
gpu_small = time.time() - start

print(f"Small (10 elements):  CPU {cpu_small:.3f}s  GPU {gpu_small:.3f}s")
# Typical result:           CPU 0.008s   GPU 0.097s  ← CPU is ~12x faster

# Large tensor: GPU wins overwhelmingly
a_cpu = torch.randn(10_000_000)
b_cpu = torch.randn(10_000_000)

start = time.time()
for _ in range(100):
    _ = a_cpu * b_cpu
cpu_large = time.time() - start

a_gpu = a_cpu.to('cuda')
b_gpu = b_cpu.to('cuda')
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    _ = a_gpu * b_gpu
torch.cuda.synchronize()
gpu_large = time.time() - start

print(f"Large (10M elements): CPU {cpu_large:.3f}s  GPU {gpu_large:.3f}s")
# Typical result:           CPU 1.420s   GPU 0.004s  ← GPU is ~350x faster`}
            />

            <p className="text-muted-foreground text-sm">
              Try this yourself in the Colab notebook&mdash;your exact numbers will
              differ, but the pattern is consistent: small tensors lose to transfer
              overhead, large tensors win massively from parallelism.
            </p>

            <CodeBlock
              language="python"
              filename="device_rule.py"
              code={`# THE RULE: all tensors in an operation must be on the same device

X_gpu = torch.randn(100, 3, device='cuda')
w_cpu = torch.randn(3, 1)

# This CRASHES:
# y = X_gpu @ w_cpu
# RuntimeError: Expected all tensors to be on the same device

# Fix: move w to the same device as X
w_gpu = w_cpu.to(X_gpu.device)
y = X_gpu @ w_gpu  # works!`}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Tensors Know Where They Live">
            Every tensor has a <code className="text-xs">.device</code> attribute.
            Think of it as an address: CPU memory or GPU memory. Operations only
            work between tensors at the same address. The <code className="text-xs">.to(device)</code> method
            moves a tensor between addresses.
          </InsightBlock>
          <TipBlock title="No GPU? No Problem">
            The Colab notebook runs on a free GPU. Locally, all the code in this
            lesson works on CPU&mdash;just skip the <code className="text-xs">.to(&apos;cuda&apos;)</code> calls.
            The tensor API is identical on both devices.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 8. Check 2 — Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              Your colleague says: &ldquo;I moved everything to the GPU but my
              code got <em>slower</em>. GPUs are supposed to be faster!&rdquo; Their
              code multiplies two 10-element vectors in a loop, 100,000 times. What
              would you tell them?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  <strong>GPU transfer overhead dominates for small tensors.</strong> Each
                  iteration sends 10 numbers to the GPU, multiplies them, and sends the
                  result back. The transfer takes longer than the multiplication itself.
                  Solutions: (1) batch the work into a single large matrix
                  operation, or (2) keep these small operations on CPU. GPU wins
                  when tensors are large enough that the parallel speedup exceeds
                  the transfer cost.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 9. Explain: NumPy Interop */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="NumPy Interop"
            subtitle="Moving between NumPy and PyTorch"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You already have NumPy code from the Foundations series. You do not
              need to rewrite everything from scratch. PyTorch can share memory
              with NumPy:
            </p>

            <CodeBlock
              language="python"
              filename="numpy_interop.py"
              code={`import numpy as np
import torch

# NumPy -> PyTorch (shared memory!)
np_array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_array)

# Modify one, both change:
np_array[0] = 99.0
print(tensor)  # tensor([99.,  2.,  3.])

# PyTorch -> NumPy (also shared memory)
tensor2 = torch.tensor([4.0, 5.0, 6.0])
np_array2 = tensor2.numpy()

# Modify one, both change:
tensor2[0] = 88.0
print(np_array2)  # [88.  5.  6.]`}
            />

            <div className="rounded-md border border-violet-500/30 bg-violet-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-violet-400">Shared memory means zero-copy:</strong>{' '}
              <code className="text-xs bg-muted px-1 rounded">torch.from_numpy()</code> and <code className="text-xs bg-muted px-1 rounded">.numpy()</code> do
              not copy data. The tensor and the array point to the same memory. This
              is fast, but changes to one affect the other. If you want an independent
              copy, use <code className="text-xs bg-muted px-1 rounded">torch.tensor(np_array)</code> (note: <code className="text-xs bg-muted px-1 rounded">torch.tensor</code>,
              not <code className="text-xs bg-muted px-1 rounded">torch.from_numpy</code>)&mdash;this always copies.
            </div>

            <p className="text-muted-foreground">
              There is one gotcha you will hit in the next lesson:
            </p>

            <CodeBlock
              language="python"
              code={`# GPU tensors can't convert to NumPy directly
t_gpu = torch.tensor([1.0, 2.0], device='cuda')
# t_gpu.numpy()  # ERROR: can't convert CUDA tensor

# Fix: move to CPU first
t_cpu = t_gpu.cpu().numpy()  # works!

# Tensors tracking gradients also need .detach() first
# (we'll cover this in the next lesson on autograd)
# t_with_grad.detach().cpu().numpy()`}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Shared Memory Surprise">
            <code className="text-xs">torch.from_numpy()</code> shares memory. If you
            modify the NumPy array, the tensor changes too. This is efficient but
            can cause silent bugs if you are not expecting it. When in
            doubt, <code className="text-xs">torch.tensor()</code> makes a safe copy.
          </WarningBlock>
          <TipBlock title=".detach().cpu().numpy()">
            This three-method chain is the most common pattern for getting data out
            of PyTorch for plotting or analysis. You will see it everywhere. The
            next lesson explains why <code className="text-xs">.detach()</code> is needed.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Practice — Colab */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Try It Yourself"
            subtitle="Build muscle memory with real PyTorch code"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Reading code is not the same as writing it. Open the notebook and
                complete these exercises:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>Create training data as PyTorch tensors (translate your NumPy code)</li>
                <li>Check <code className="text-xs bg-muted px-1 rounded">.shape</code>, <code className="text-xs bg-muted px-1 rounded">.dtype</code>, and <code className="text-xs bg-muted px-1 rounded">.device</code> for each tensor</li>
                <li>Move your data to GPU and verify the device changed</li>
                <li>Compute the forward pass: <code className="text-xs bg-muted px-1 rounded">y_hat = X @ w + b</code></li>
                <li>Reshape a batch of 28x28 images into 784-element vectors</li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-1-1-tensors.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                The notebook has starter code and hints. A free GPU is available
                in Colab&mdash;no local setup needed.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Save a Copy">
            Colab opens in read-only mode. Click &ldquo;Copy to Drive&rdquo; to
            save your own version that you can edit and experiment with.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 11. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Tensors are NumPy arrays that know where they live',
                description:
                  'Same API for creation, arithmetic, reshaping, and broadcasting. The new superpower is .to(device) — move computation to a GPU with one line.',
              },
              {
                headline: 'Three attributes: shape, dtype, device',
                description:
                  'When debugging PyTorch code, check these first. Shape mismatches are the most common error. Device mismatches are the most confusing.',
              },
              {
                headline: 'float32 is the default, and that\'s intentional',
                description:
                  'Mini-batch gradients are approximate — float64 precision is wasted. GPUs are optimized for float32 (and float16). Don\'t "upgrade" to float64.',
              },
              {
                headline: 'GPU wins at scale, CPU wins for small operations',
                description:
                  'Transfer overhead means small tensors are faster on CPU. Move to GPU when tensors are large (10,000+ elements) and you\'re doing heavy math.',
              },
              {
                headline: 'NumPy interop is seamless but shares memory',
                description:
                  'torch.from_numpy() and .numpy() share memory (zero-copy). torch.tensor() makes an independent copy. GPU tensors need .cpu() before .numpy().',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/autograd"
            title="Ready for PyTorch's killer feature?"
            description="You have the data structure. Next: automatic gradient computation — no more deriving gradients by hand."
            buttonText="Continue to Autograd"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
