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
  GradientCard,
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

/**
 * GPU Training -- Lesson 2 of Module 2.3 (Practical Patterns)
 *
 * STRETCH lesson: teaches the student to write device-aware training code
 * that runs identically on CPU and GPU, and to use mixed precision to
 * accelerate training when it matters.
 *
 * New concepts (2-3):
 *   1. Device-aware training loop pattern (DEVELOPED from INTRODUCED)
 *   2. Mixed precision with torch.amp.autocast + GradScaler (INTRODUCED)
 *   3. "When does GPU help?" timing-based decision (DEVELOPED refinement)
 *
 * Central insight: GPU training is the same training loop with 3 lines of
 * device placement added. "Same heartbeat, new instruments."
 */

export function GpuTrainingLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="GPU Training"
            description="Move your training loop to GPU and squeeze more speed with mixed precision&mdash;without changing the heartbeat."
            category="Practical Patterns"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Write device-aware training code that runs on whatever hardware is
            available&mdash;CPU or GPU&mdash;with no code changes. Understand
            when GPU actually helps, handle the device mismatch error in a
            real training loop, and use mixed precision to accelerate training
            when it matters. By the end, you have a portable training pattern
            that carries forward to every future project.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="STRETCH Lesson">
            This lesson coordinates device placement across model, data, and
            training loop&mdash;a higher-order pattern than anything you have
            done before. Coming after the BUILD lesson on saving and loading,
            you have had a lower-load session to recover.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 1. Context + Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'Single-GPU training only -- no multi-GPU or distributed training',
              'No CUDA programming, kernels, or streams',
              'No memory management or OOM debugging',
              'No torch.compile or custom CUDA extensions',
              'No profiling tools -- just wall-clock timing',
              'Mixed precision basics only -- no bfloat16 deep dive',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook -- The Speed Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Speed Problem"
            subtitle="Your MNIST model trained in 60 seconds. That is about to stop being enough."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your MNIST model from the MNIST project trained in about a minute
              on CPU. That was fine&mdash;fast enough to iterate, fast enough to
              experiment. But that is about to change. CNNs, transformers, and
              diffusion models are all ahead. A model that takes 2 minutes on
              GPU might take 2 hours on CPU, or never finish at all.
            </p>

            <p className="text-muted-foreground">
              But here is the thing most tutorials skip:{' '}
              <strong>GPU is not always faster.</strong> Let&rsquo;s look at two
              timing comparisons:
            </p>

            <ComparisonRow
              left={{
                title: 'Small Model, Tiny Data',
                color: 'amber',
                items: [
                  '2-layer network, 500 parameters',
                  '100 training samples, batch_size=32',
                  'CPU: 0.8 seconds',
                  'GPU: 2.1 seconds',
                  'CPU wins -- transfer overhead dominates',
                ],
              }}
              right={{
                title: 'MNIST Model, Full Data',
                color: 'emerald',
                items: [
                  '3-layer network, 235K parameters',
                  '60,000 training samples, batch_size=64',
                  'CPU: ~60 seconds',
                  'GPU: ~12 seconds',
                  'GPU wins -- parallel compute dominates',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Remember the mental model from Tensors: <strong>&ldquo;GPU
              wins at scale, CPU wins for small operations.&rdquo;</strong> That
              was about individual tensor operations. The same principle applies
              to training: GPU wins when there is enough work to keep thousands
              of cores busy. A tiny model on a tiny dataset does not have enough
              work&mdash;the time spent copying data to the GPU overwhelms any
              speedup from parallel computation.
            </p>

            <p className="text-muted-foreground">
              Two questions for this lesson: <strong>how do you move your
              existing training code to GPU</strong> (it is easier than you
              think), and <strong>how do you squeeze even more speed out of
              the GPU with mixed precision?</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Universally Faster">
            GPU training is not a free win. Transfer overhead is real. If
            your model trains in under 30 seconds on CPU, GPU may not help.
            If training takes minutes to hours, GPU will likely give 3&ndash;10x
            speedup.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Recap: Device Fundamentals */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Recap: Device Fundamentals"
            subtitle="Activating what you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You learned device management in Tensors, about six lessons ago.
              Quick refresher before we build on it:
            </p>

            <CodeBlock
              language="python"
              filename="device_pattern.py"
              code={`# The portable device pattern -- same as Tensors lesson
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# .to(device) moves a tensor to the specified device
x = torch.randn(3, 3)
x = x.to(device)  # now on GPU (if available)

# All tensors in an operation must be on the same device
y = torch.randn(3, 3).to(device)
z = x + y  # works -- both on the same device`}
            />

            <p className="text-muted-foreground">
              Three things to remember: (1) the device detection pattern gives
              you a variable that works on any machine, (2){' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.to(device)</code>{' '}
              moves tensors between CPU and GPU, and (3) all tensors in an
              operation must be on the same device. That last point is about to
              become very important.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Debugging Trinity">
            From Tensors: when something goes wrong, check{' '}
            <strong>shape, dtype, device</strong>. Device is already part
            of your debugging mental model.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 4. Explain: The Device Mismatch Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Device Mismatch Problem"
            subtitle="Moving only the model is not enough"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Take your existing CPU training loop and add one line to move the
              model to GPU:
            </p>

            <CodeBlock
              language="python"
              filename="incomplete_gpu.py"
              code={`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTClassifier().to(device)  # model is on GPU

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for images, labels in train_loader:
    # images and labels come from DataLoader -- they are on CPU!
    outputs = model(images)  # RuntimeError!
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
            />

            <div className="rounded-lg border border-rose-500/30 bg-rose-500/5 p-4 font-mono text-xs text-rose-400 overflow-x-auto">
              <p>RuntimeError: Expected all tensors to be on the same device,</p>
              <p>but found at least two devices, cuda:0 and cpu!</p>
            </div>

            <p className="text-muted-foreground">
              You have seen this error before&mdash;in Tensors, on individual
              tensors. Now it appears in a real training loop. The model is on
              GPU, but the DataLoader yields CPU tensors every iteration. The
              forward pass tries to multiply GPU weights by CPU inputs, and
              PyTorch refuses.
            </p>

            <p className="text-muted-foreground">
              <strong>The fix:</strong> move each batch of data to the device
              inside the loop.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="DataLoader Always Yields CPU">
            DataLoader creates new CPU tensors each iteration. You must move
            each batch to the device <strong>inside</strong> the loop, not
            before it.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 5. Explain: The Device-Aware Training Loop */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Device-Aware Training Loop"
            subtitle="Three lines of change. That is all."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is your CPU training loop and the GPU training loop, side by
              side. Look for the differences:
            </p>

            <ComparisonRow
              left={{
                title: 'CPU Training Loop',
                color: 'amber',
                items: [
                  'model = MNISTClassifier()',
                  '# (no device line needed)',
                  'for images, labels in train_loader:',
                  '    outputs = model(images)',
                  '    loss = criterion(outputs, labels)',
                  '    optimizer.zero_grad()',
                  '    loss.backward()',
                  '    optimizer.step()',
                ],
              }}
              right={{
                title: 'GPU Training Loop',
                color: 'emerald',
                items: [
                  'model = MNISTClassifier().to(device)',
                  '# device = torch.device(...)',
                  'for images, labels in train_loader:',
                  '    images = images.to(device)',
                  '    labels = labels.to(device)',
                  '    outputs = model(images)',
                  '    loss = criterion(outputs, labels)',
                  '    ... (same heartbeat)',
                ],
              }}
            />

            <p className="text-muted-foreground">
              <strong>Three lines changed:</strong> (1) move the model to the
              device, (2) move inputs to the device inside the loop, (3) move
              targets to the device inside the loop. The forward-loss-backward-update
              heartbeat is identical. No special GPU layers, no different APIs,
              no separate training function.
            </p>

            <p className="text-muted-foreground">
              Here is the complete, copy-ready pattern:
            </p>

            <CodeBlock
              language="python"
              filename="gpu_training_loop.py"
              code={`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

model = MNISTClassifier().to(device)  # Line 1: model to device
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)     # Line 2: inputs to device
        labels = labels.to(device)     # Line 3: targets to device

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`}
            />

            <p className="text-muted-foreground">
              That moment of &ldquo;that is all?&rdquo; is the point.{' '}
              <strong>Same heartbeat, new instruments.</strong> GPU placement is
              just one more instrument that slots into the existing training
              loop pattern. You do not need to rewrite anything.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Heartbeat">
            The training loop pattern from The Training Loop is universal.
            Forward, loss, backward, step&mdash;always the same rhythm.
            GPU placement does not change the structure, just where the
            computation happens.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 6. Check 1 -- Predict-and-Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">
              Check 1: Predict-and-Verify
            </h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                You move your model to GPU with{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">model.to(device)</code>{' '}
                but forget to move the data. What error do you get, and on which
                line of the training loop?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      <strong>RuntimeError on the forward pass line</strong>{' '}
                      (<code className="text-xs bg-muted px-1 py-0.5 rounded">outputs = model(images)</code>).
                      The model&rsquo;s weights are on GPU, the images are on
                      CPU. PyTorch cannot multiply tensors on different devices.
                    </p>
                  </div>
                </div>
              </details>

              <p className="text-muted-foreground text-sm mt-4">
                <strong>Follow-up:</strong> Your colleague puts{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">images.to(device)</code>{' '}
                <em>outside</em> the training loop, before it starts. Why does
                this not work?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      <strong>DataLoader yields new CPU tensors each
                      iteration.</strong> Moving one batch to GPU before the
                      loop starts only moves that one batch. The next call
                      to the DataLoader produces fresh CPU tensors. You must
                      move each batch inside the loop.
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 7. Explain: When Does GPU Help? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When Does GPU Help?"
            subtitle="The crossover depends on model size and data size"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Return to the timing comparison from the hook. GPU is not a
              universal speedup&mdash;it is a tradeoff between parallel compute
              and transfer overhead. Three factors determine where the crossover
              happens:
            </p>

            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Model Size" color="blue">
                <p className="text-sm">
                  More parameters means more parallel computation. A 500-parameter
                  model barely uses the GPU&rsquo;s thousands of cores. A 235K-parameter
                  model keeps them busy.
                </p>
              </GradientCard>
              <GradientCard title="Batch Size" color="cyan">
                <p className="text-sm">
                  Larger batches give the GPU more work to parallelize per
                  iteration. A batch of 4 is too little. A batch of 64 or 128
                  starts to amortize the transfer cost.
                </p>
              </GradientCard>
              <GradientCard title="Transfer Overhead" color="amber">
                <p className="text-sm">
                  Every batch moves from CPU to GPU memory each iteration.
                  If the compute per batch is small, the transfer time dominates
                  and the GPU is waiting, not computing.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              <strong>Practical guideline:</strong> if your model trains in
              under 30 seconds on CPU, GPU probably does not help. If training
              takes minutes to hours, GPU will likely give 3&ndash;10x speedup.
              This is a heuristic, not a law&mdash;when in doubt, time both
              and compare.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Assembly Line">
            Think of the training loop as an assembly line. Moving to GPU
            swaps in faster workers for the compute-heavy steps. The
            assembly line itself (the loop structure) does not change.
            If the product is tiny, the overhead of swapping workers
            costs more than the speed gain.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 8. Explain: Device-Aware Checkpoints */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Device-Aware Checkpoints"
            subtitle="Saving on GPU, loading on CPU (and vice versa)"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Saving, Loading, and Checkpoints you learned the checkpoint
              pattern: bundle model state, optimizer state, epoch, and loss into
              one dictionary. Now your tensors might be on GPU. What happens
              when you save on GPU and try to load on a CPU-only machine?
            </p>

            <p className="text-muted-foreground">
              By default,{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.load()</code>{' '}
              tries to restore tensors to the device they were saved on. If you
              saved on{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">cuda:0</code>{' '}
              and load on a machine without a GPU, it fails. You previewed the
              fix in the last lesson&mdash;now let&rsquo;s use it for real:
            </p>

            <CodeBlock
              language="python"
              filename="device_aware_checkpoint.py"
              code={`# SAVE: checkpoint during GPU training
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, 'checkpoint.pth')

# LOAD: portable pattern -- works on any machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoint.pth',
                        map_location=device,
                        weights_only=False)

model = MNISTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1`}
            />

            <p className="text-muted-foreground">
              The key is{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">map_location=device</code>.
              This tells{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.load()</code>{' '}
              to remap all tensors to whatever device is available right now,
              regardless of where they were when saved. Saved on GPU, loading on
              CPU? The tensors move to CPU. Saved on CPU, loading on GPU? They
              move to GPU. Your checkpoint files are portable.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Portable Pattern">
            Always use{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">map_location=device</code>{' '}
            when loading checkpoints. Your files work on any machine, regardless
            of what hardware saved them.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 9. Explore: GPU Training in Practice */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="GPU Training in Practice"
            subtitle="The complete pattern, ready to type into Colab"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the full device-aware training script&mdash;your MNIST
              model with GPU training, timing, and device-aware checkpointing.
              This is the pattern you carry forward:
            </p>

            <CodeBlock
              language="python"
              filename="complete_gpu_training.py"
              code={`import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Device detection ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True,
                            transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# --- Model, optimizer, loss ---
model = MNISTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# --- Training with timing ---
num_epochs = 10
best_loss = float('inf')

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}  loss: {avg_loss:.4f}")

    # --- Device-aware checkpoint ---
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, 'best_checkpoint.pth')

elapsed = time.time() - start_time
print(f"Total training time: {elapsed:.1f}s on {device}")`}
            />

            <p className="text-muted-foreground">
              Notice how everything fits together: the device pattern from
              Tensors, the training loop from The Training Loop, the DataLoader
              from Datasets and DataLoaders, the checkpoint pattern from
              Saving, Loading, and Checkpoints&mdash;and exactly three new lines
              for GPU placement. Same heartbeat, new instruments.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Everything Connects">
            This script uses concepts from five previous lessons. GPU training
            is not a new skill&mdash;it is the coordination of existing skills
            with device awareness added.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 10. Check 2 -- Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">
              Check 2: Transfer Question
            </h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                A colleague says their GPU training is <em>slower</em> than CPU.
                Their model has 500 parameters and their dataset has 200 samples
                with batch_size=32. What would you tell them?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      <strong>The model and data are too small for GPU to
                      overcome transfer overhead.</strong> With only 500
                      parameters and 200 samples (about 7 batches per epoch),
                      the GPU barely has time to spin up before the epoch
                      ends. The cost of moving 7 batches to GPU exceeds the
                      speedup from parallel computation. Stick with CPU until
                      the model or dataset scales up.
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 11. Explain: Why Mixed Precision? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Mixed Precision?"
            subtitle="GPU training is faster. Can we make it even faster?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              GPU training gave you a 5x speedup. But there is more speed
              sitting on the table. To understand how, revisit the precision
              analogy from Tensors.
            </p>

            <p className="text-muted-foreground">
              Remember &ldquo;measuring a rough sketch with a
              micrometer&rdquo;? Float64 is too precise for approximate
              mini-batch gradients&mdash;float32 is the right tool. Now extend
              the analogy one step further:
            </p>

            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="float64 (Micrometer)" color="rose">
                <p className="text-sm">
                  64 bits. Too precise for training. Wasted memory and compute.
                  You already stopped using this.
                </p>
              </GradientCard>
              <GradientCard title="float32 (Ruler)" color="blue">
                <p className="text-sm">
                  32 bits. The default. Right-sized for gradient updates and
                  weight accumulation. This is what you have been using.
                </p>
              </GradientCard>
              <GradientCard title="float16 (Tape Measure)" color="emerald">
                <p className="text-sm">
                  16 bits. Half the memory, and modern GPUs process it 2&ndash;4x
                  faster with dedicated hardware (Tensor Cores). Good enough for
                  the forward pass&mdash;but not for everything.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Float16 uses half the memory and modern GPUs have hardware that
              processes it 2&ndash;4x faster. But there is a catch.{' '}
              <strong>You cannot use float16 for everything.</strong>
            </p>

            <CodeBlock
              language="python"
              filename="float16_underflow.py"
              code={`# The problem with full float16 training
gradient = torch.tensor(0.00001, dtype=torch.float32)

# Convert to float16
gradient_fp16 = gradient.half()
print(f"float32: {gradient.item()}")   # 1e-05
print(f"float16: {gradient_fp16.item()}")  # 9.999e-06 -- close enough

# But smaller gradients underflow to zero
tiny_gradient = torch.tensor(0.000001, dtype=torch.float32)
tiny_fp16 = tiny_gradient.half()
print(f"float32: {tiny_gradient.item()}")  # 1e-06
print(f"float16: {tiny_fp16.item()}")      # 9.537e-07 -- still okay

# Even smaller -- float16 minimum is ~6e-08
very_tiny = torch.tensor(1e-08, dtype=torch.float32)
very_tiny_fp16 = very_tiny.half()
print(f"float32: {very_tiny.item()}")      # 1e-08
print(f"float16: {very_tiny_fp16.item()}")  # 0.0 -- UNDERFLOW!`}
            />

            <p className="text-muted-foreground">
              Small gradients round to zero in float16. When gradients are zero,
              the optimizer cannot update weights, and the model stops learning.
              This is <strong>gradient underflow</strong>, and it is why full
              float16 training breaks.
            </p>

            <p className="text-muted-foreground">
              The solution: use float16 where it is safe (the forward pass,
              where values are typically large) and float32 where precision
              matters (gradient accumulation and weight updates, where values
              can be tiny). This is why it is called <strong>mixed</strong>{' '}
              precision&mdash;different operations use different precisions.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Full Float16">
            &ldquo;Mixed precision&rdquo; does not mean &ldquo;use float16
            everywhere.&rdquo; The forward pass runs in float16 for speed.
            Gradient accumulation and weight updates stay in float32 for
            stability. <strong>Mixed</strong> is the key word.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 12. Explain: torch.amp (Automatic Mixed Precision) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="torch.amp: Automatic Mixed Precision"
            subtitle="PyTorch handles the precision switching for you"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You could manually cast every tensor to the right dtype at the
              right time&mdash;float16 for the forward pass, float32 for the
              backward pass, careful dtype management throughout. That would be
              error-prone and tedious. Instead, PyTorch automates it:
            </p>

            <CodeBlock
              language="python"
              filename="mixed_precision_loop.py"
              code={`# Standard GPU training loop with mixed precision added
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# NEW: Create a GradScaler
scaler = torch.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # NEW: autocast wraps the forward pass
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # NEW: scaler handles the backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()   # scale loss up before backward
        scaler.step(optimizer)           # unscale gradients, then step
        scaler.update()                  # adjust scale factor`}
            />

            <p className="text-muted-foreground">
              Two new tools, four changed lines:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong><code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.amp.autocast(device_type=&apos;cuda&apos;)</code></strong>{' '}
                wraps the forward pass. Inside the context manager, PyTorch
                automatically chooses float16 where safe and float32 where
                needed. You do not pick&mdash;it picks for you.
              </li>
              <li>
                <strong><code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.amp.GradScaler()</code></strong>{' '}
                solves the underflow problem. It scales the loss <em>up</em>{' '}
                before{' '}
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code>{' '}
                (so small gradients in float16 do not round to zero), then
                scales gradients back <em>down</em> before the optimizer step
                (so weight updates are correct).
              </li>
            </ul>

            <p className="text-muted-foreground">
              The structure of the loop is the same. Forward, loss, backward,
              step&mdash;still the same heartbeat. Autocast and GradScaler wrap
              around it, handling the precision switching automatically. This is
              the same pattern as autograd: <strong>not magic&mdash;automation.</strong>{' '}
              You <em>could</em> manage precision manually, just as you{' '}
              <em>could</em> compute gradients by hand. The API automates the
              tedious parts.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Magic&mdash;Automation">
            Autocast chooses float16 where safe, float32 where needed.
            GradScaler prevents underflow by scaling the loss up before
            backward and down before the optimizer step. Same pattern
            as autograd: automation of a correct-but-tedious manual process.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 13. Elaborate: When Mixed Precision Helps */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When Mixed Precision Helps"
            subtitle="Not always, but easy to try"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Mixed precision helps most when:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Your GPU has Tensor Cores</strong>&mdash;most NVIDIA
                GPUs from the last few generations (RTX 20xx+, T4, A100, etc.)
                have dedicated hardware for fast float16 math.
              </li>
              <li>
                <strong>The model is large enough</strong> that memory or compute
                is the bottleneck, not data loading.
              </li>
              <li>
                <strong>The forward pass dominates training time</strong>&mdash;this
                is where autocast applies float16, so that is where the speedup
                comes from.
              </li>
            </ul>

            <p className="text-muted-foreground">
              It helps <em>less</em> for small models, when data loading is the
              bottleneck, or on older GPUs without Tensor Cores. On CPU, autocast
              uses bfloat16 instead of float16&mdash;a different format with
              the same range as float32 but lower precision. We will not develop
              bfloat16 here; just know the option exists.
            </p>

            <p className="text-muted-foreground">
              <strong>Practical guideline:</strong> try it. If training gets
              faster without accuracy loss, keep it. If accuracy drops, remove
              it. The code change is 4 lines&mdash;easy to add, easy to revert.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Easy Experiment">
            Mixed precision is a 4-line change. Add it, compare training speed
            and final accuracy. If speed improves and accuracy holds, keep it.
            If not, revert. Low-risk experiment.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 14. Check 3 -- Predict-and-Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">
              Check 3: Predict-and-Verify
            </h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                Why does GradScaler multiply the loss by a large number before{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">backward()</code>?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      <strong>To prevent float16 underflow.</strong> Float16 has
                      a smaller representable range than float32. Small gradients
                      that would be non-zero in float32 can round to zero in
                      float16. Scaling the loss up before backward means the
                      resulting gradients are proportionally larger&mdash;large
                      enough to stay non-zero in float16. The scaler then divides
                      them back to the correct magnitude before the optimizer
                      step.
                    </p>
                    <p>
                      This tests understanding of <em>why</em> mixed precision
                      is &ldquo;mixed&rdquo;&mdash;float16 alone is not precise
                      enough for small gradient values.
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 15. Practice -- Colab Notebook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="Practice GPU training and mixed precision hands-on"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The Colab notebook walks you through each pattern using your
                MNIST model from previous lessons:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>(Guided)</strong> Move the MNIST model and training
                  loop to GPU. Time it. Compare to CPU training time.
                </li>
                <li>
                  <strong>(Supported)</strong> Add device-aware checkpointing.
                  Save during GPU training, load and verify on CPU using{' '}
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">map_location</code>.
                </li>
                <li>
                  <strong>(Supported)</strong> Add mixed precision (autocast +
                  GradScaler) to the GPU training loop. Compare training speed
                  with and without.
                </li>
                <li>
                  <strong>(Independent)</strong> Write a complete, portable
                  training script that: detects device, uses GPU if available,
                  uses mixed precision if on GPU, checkpoints with device
                  portability. This is the &ldquo;production-ready&rdquo; pattern
                  you carry forward.
                </li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-3-2-gpu-training.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Exercise 1 is guided with complete starter code. Exercises
                2&ndash;3 are supported with templates. Exercise 4 is
                independent&mdash;you implement the full portable training
                pattern from scratch.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="GPU Required">
            Exercises 1&ndash;4 need a GPU. In Colab, go to Runtime &rarr;
            Change runtime type &rarr; T4 GPU. The free tier usually provides
            one.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 16. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'GPU training = same loop + 3 lines of device placement',
                description:
                  'Move the model to the device once, and move each batch of inputs and targets inside the loop. The forward-loss-backward-step heartbeat is identical.',
              },
              {
                headline: 'All tensors must be on the same device',
                description:
                  'Model weights, inputs, and targets must all be on the same device. The device mismatch RuntimeError tells you when they are not.',
              },
              {
                headline: 'GPU wins at scale, not always',
                description:
                  'Small models on small data can be slower on GPU because transfer overhead dominates. GPU shines when there is enough work to keep thousands of cores busy.',
              },
              {
                headline: 'Mixed precision: float16 for speed, float32 for precision',
                description:
                  'autocast runs the forward pass in float16 where safe. GradScaler prevents gradient underflow by scaling the loss before backward. 4 lines of change, same structure.',
              },
              {
                headline: 'Portable checkpoints use map_location',
                description:
                  'Always load checkpoints with map_location=device. Your files work on any machine regardless of what hardware saved them.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 17. Next step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/fashion-mnist-project"
            title="Fashion-MNIST Project"
            description="You now have the full practical toolkit: build, train, save, load, checkpoint, GPU, mixed precision. The next lesson puts it all together&mdash;an independent project where you make your own design decisions."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
