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
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * Datasets and DataLoaders -- Lesson 1 of Module 2.2 (Real Data)
 *
 * BUILD lesson: teaches the student to load, transform, and batch datasets
 * using PyTorch's Dataset and DataLoader abstractions. Connects the batching
 * theory from Series 1 (polling analogy, mini-batch SGD) to practical data
 * pipelines.
 *
 * New concepts (2-3):
 *   1. torch.utils.data.Dataset (__getitem__ + __len__)
 *   2. torch.utils.data.DataLoader (batching, shuffling, iteration)
 *   3. torchvision.transforms pipeline (API knowledge)
 *
 * Target depths:
 * - Dataset: DEVELOPED
 * - DataLoader: DEVELOPED
 * - Transforms: INTRODUCED
 * - torchvision.datasets: INTRODUCED
 *
 * Central insight: Dataset says "I have N items, here's item i."
 * DataLoader wraps a Dataset and handles batching, shuffling, and iteration.
 * The training loop body does not change -- only the data-feeding line.
 */

export function DatasetsAndDataloadersLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Datasets and DataLoaders"
            description="Connect your training loop to real data with PyTorch's two-layer data abstraction."
            category="Real Data"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Load, transform, and batch data using{' '}
            <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.utils.data.Dataset</code> and{' '}
            <code className="text-sm bg-muted px-1.5 py-0.5 rounded">DataLoader</code>. By the end,
            you can plug any dataset into the training loop you wrote in The Training Loop&mdash;without
            changing the loop body.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Theory to Practice">
            You already know WHY we batch (from Batching and SGD). This lesson
            shows you HOW&mdash;the PyTorch API that makes mini-batch SGD a
            concrete <code className="text-xs">for</code> loop.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'No training a model to convergence on a real dataset -- that is the next lesson',
              'No cross-entropy loss or softmax -- deferred to the MNIST project',
              'No data augmentation strategies (RandomFlip, RandomCrop) -- mentioned only',
              'No advanced DataLoader options (num_workers, pin_memory, custom collate)',
              'No train/val/test splitting in code -- concept known from Series 1',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook: The Scale Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Scale Problem"
            subtitle="Your training loop works—but only on 100 data points"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In The Training Loop, you trained a model on <InlineMath math="y = 2x + 1" /> with
              100 data points. The data was two tensors, created in two lines:
            </p>

            <CodeBlock
              language="python"
              filename="from_training_loop.py"
              code={`# From The Training Loop -- all data in memory as tensors
x = torch.randn(100, 1)
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)`}
            />

            <p className="text-muted-foreground">
              This works for learning the API. It does not work for real datasets.
              MNIST has 60,000 images. ImageNet has 1.2 million. You cannot hand-craft
              60,000 tensor entries.
            </p>

            <p className="text-muted-foreground">
              But memory is not even the main problem. Think about what your training loop
              needs to do each epoch:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Split the data into batches of size <InlineMath math="B" /></li>
              <li>Shuffle the order every epoch (remember the polling analogy&mdash;random samples give unbiased estimates)</li>
              <li>Handle the last batch when <InlineMath math="N" /> is not divisible by <InlineMath math="B" /></li>
              <li>Feed each batch as a properly shaped tensor</li>
            </ul>

            <p className="text-muted-foreground">
              What if the training loop body did not have to change at all? What if you
              could swap the data source and keep everything else?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Real Question">
            The question is not &ldquo;how do I load data?&rdquo; It is &ldquo;how do I feed
            data into a training loop that already works?&rdquo; PyTorch has a clean answer.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. The Naive Approach */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Naive Approach"
            subtitle="You could write this yourself -- but should you?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Given what you know, you might batch data by slicing tensors in a for loop:
            </p>

            <CodeBlock
              language="python"
              filename="naive_batching.py"
              code={`# Naive batching -- manual index math
x = torch.randn(100, 1)
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)
batch_size = 8

for epoch in range(num_epochs):
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`}
            />

            <p className="text-muted-foreground">
              This works! But it has four problems:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>No shuffling between epochs.</strong> Every epoch sees the same
                batch order. From Batching and SGD: biased sampling gives biased gradient
                estimates.
              </li>
              <li>
                <strong>Manual index math.</strong> Off-by-one errors, tracking
                the start index, computing the number of batches.
              </li>
              <li>
                <strong>Last batch is silently smaller.</strong> When 100 is not divisible by 8,
                the last batch has 4 samples. The loop does not tell you.
              </li>
              <li>
                <strong>No separation between data access and training logic.</strong> The
                batching code is tangled with the training code.
              </li>
            </ul>

            <p className="text-muted-foreground">
              You could fix all of these one by one. Or you could use an abstraction
              that handles all of them&mdash;and separates concerns cleanly.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Biased Batches">
            Without shuffling, if data happens to be sorted (e.g., all class-0 samples
            first, then class-1), each batch contains only one class. The model
            &ldquo;forgets&rdquo; the previous class with each new batch.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 4. Dataset -- One Sample at a Time */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Dataset -- One Sample at a Time"
            subtitle="The interface that says 'I have N items, here's item i'"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              PyTorch splits the data problem into two layers. The first
              is <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.utils.data.Dataset</code>.
              A Dataset answers two questions:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__len__</code>&mdash;How
                many samples do you have?
              </li>
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code>&mdash;Give
                me sample number <em>i</em>.
              </li>
            </ul>

            <p className="text-muted-foreground">
              That is it. No batching. No shuffling. Just &ldquo;how many?&rdquo; and
              &ldquo;give me one.&rdquo;
            </p>

            <CodeBlock
              language="python"
              filename="simple_dataset.py"
              code={`from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """Dataset for the y = 2x + 1 data from The Training Loop."""

    def __init__(self, num_samples=100):
        torch.manual_seed(42)
        self.x = torch.randn(num_samples, 1)
        self.y = 2 * self.x + 1 + 0.1 * torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Use it:
dataset = SimpleDataset(100)
print(len(dataset))        # 100
x_0, y_0 = dataset[0]     # one sample
print(x_0.shape, y_0.shape)  # torch.Size([1]) torch.Size([1])`}
            />

            <p className="text-muted-foreground">
              Think of a Dataset as a <strong>menu</strong>. It tells you what is available
              (100 items) and how to get one item (by index). It does not decide the order.
              It does not group items into courses. It just serves one dish at a time.
            </p>

            <p className="text-muted-foreground">
              If you are a software engineer, this is a familiar pattern: a Dataset is a
              read-only collection with index access. Python&rsquo;s{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__len__</code> and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code> are
              dunder methods you have used before&mdash;just applied to data samples.
            </p>

            <p className="text-muted-foreground">
              Our <code className="text-sm bg-muted px-1.5 py-0.5 rounded">SimpleDataset</code> stores
              all data in memory. That works for 100 points, but not for 1.2 million images.
              The key insight: <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code> can
              load data from disk <strong>on demand</strong>&mdash;storing only file paths
              in memory, not the data itself:
            </p>

            <CodeBlock
              language="python"
              filename="lazy_dataset_skeleton.py"
              code={`# Lazy loading pattern -- for large datasets
class LazyImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.paths = image_paths  # just strings, not images
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_image(self.paths[idx])  # load from disk HERE
        return image, self.labels[idx]`}
            />

            <p className="text-muted-foreground text-sm">
              We will not build this now, but this is the pattern for large datasets.
              The DataLoader calls <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code> for
              each sample in the batch&mdash;only one batch of images lives in memory at a time.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Dataset Contract">
            <code className="text-xs">__len__</code>: returns an integer (number of samples).{' '}
            <code className="text-xs">__getitem__</code>: takes an index, returns one sample
            (typically a tuple of input and target tensors).
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 5. DataLoader -- The Batching Machine */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="DataLoader -- The Batching Machine"
            subtitle="Wraps a Dataset and handles batching, shuffling, and iteration"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The second layer is{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.utils.data.DataLoader</code>.
              It wraps a Dataset and handles everything the naive for-loop could not:
            </p>

            <CodeBlock
              language="python"
              filename="dataloader_basics.py"
              code={`from torch.utils.data import DataLoader

dataset = SimpleDataset(100)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Iterate over one epoch:
for batch_idx, (x_batch, y_batch) in enumerate(loader):
    print(f"Batch {batch_idx}: x shape = {x_batch.shape}, y shape = {y_batch.shape}")

# Output:
# Batch 0: x shape = torch.Size([8, 1]), y shape = torch.Size([8, 1])
# Batch 1: x shape = torch.Size([8, 1]), y shape = torch.Size([8, 1])
# ...
# Batch 12: x shape = torch.Size([4, 1]), y shape = torch.Size([4, 1])`}
            />

            <p className="text-muted-foreground">
              Notice: 13 batches. Twelve full batches of 8, one final batch of 4.
              That is <InlineMath math="100 / 8 = 12.5" />, rounded up to 13.
              Sound familiar? From Batching and SGD:{' '}
              <InlineMath math="\text{iterations per epoch} = \lceil N / B \rceil" />.
            </p>

            <p className="text-muted-foreground">
              Also notice what DataLoader does automatically:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Shuffles</strong> the order each epoch (<code className="text-sm bg-muted px-1.5 py-0.5 rounded">shuffle=True</code>)
              </li>
              <li>
                <strong>Batches</strong> individual samples into tensors with a batch dimension
              </li>
              <li>
                <strong>Handles the last batch</strong> automatically (smaller if <InlineMath math="N" /> is not divisible by <InlineMath math="B" />)
              </li>
              <li>
                <strong>Yields</strong> batches as an iterator&mdash;use it in a plain{' '}
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">for</code> loop
              </li>
            </ul>

            <p className="text-muted-foreground">
              If Dataset is the <strong>menu</strong>, DataLoader is the{' '}
              <strong>kitchen</strong>. It takes orders from the menu, groups them into
              batches, shuffles the queue, and serves plates efficiently. The menu does
              not know about batch sizes. The kitchen does not know how to prepare a
              single dish. Each has its own job.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Menu and Kitchen">
            <strong>Dataset</strong> = menu (what is available, how to get one item).{' '}
            <strong>DataLoader</strong> = kitchen (batches orders, shuffles the queue,
            serves plates). Clean separation of concerns.
          </InsightBlock>
          <TipBlock title="Choosing batch_size">
            Batch size is a tradeoff you already know from Batching and SGD: larger
            batches give cleaner gradient estimates but update parameters less often.
            There is also a practical constraint&mdash;larger batches use more GPU memory.
            Common starting point: 32 or 64, then adjust based on your hardware.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Pipeline diagram */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The data pipeline in one picture:
            </p>
            <MermaidDiagram chart={`
              graph LR
                A["Raw Data<br/>(files, arrays, etc.)"] --> B["Dataset<br/>__getitem__(i) returns one sample"]
                B --> C["DataLoader<br/>batches, shuffles, iterates"]
                C --> D["Training Loop<br/>forward / loss / backward / step"]
            `} />
            <p className="text-muted-foreground text-sm">
              Each layer has a single responsibility. Change your data source? Write a
              new Dataset. Change your batch size? Reconfigure the DataLoader. The
              training loop does not care.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Shuffle connection to 1.3.4 */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the polling analogy from Batching and SGD? Random sampling gives
              unbiased gradient estimates.{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">shuffle=True</code> is{' '}
              <strong>how you get random sampling in practice</strong>. Each epoch, the
              DataLoader shuffles the indices and draws non-overlapping batches from the
              new order. Different batches every epoch&mdash;unbiased gradient estimates.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Polling Analogy, Revisited">
            From Batching and SGD: a random sample of 32 gives a noisy but unbiased
            estimate of the full gradient.{' '}
            <code className="text-xs">shuffle=True</code> makes each batch a
            random sample.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 6. Check 1: Predict-and-Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                You have a Dataset with <strong>200 samples</strong> and create a DataLoader
                with <code className="text-sm bg-muted px-1.5 py-0.5 rounded">batch_size=32, shuffle=True</code>.
              </p>

              <p className="text-muted-foreground text-sm">
                <strong>1.</strong> How many iterations per epoch?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    <InlineMath math="\lceil 200 / 32 \rceil = 7" /> iterations. Six full batches
                    of 32 (<InlineMath math="6 \times 32 = 192" />), then one final batch of 8
                    (<InlineMath math="200 - 192 = 8" />).
                  </p>
                </div>
              </details>

              <p className="text-muted-foreground text-sm">
                <strong>2.</strong> Will two consecutive epochs iterate in the same order?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    No. <code className="text-xs bg-muted px-1 rounded">shuffle=True</code> re-shuffles
                    the indices at the start of every epoch. Different batches, different order.
                  </p>
                </div>
              </details>

              <p className="text-muted-foreground text-sm">
                <strong>3.</strong> Each sample is a tensor of shape{' '}
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[5]</code>. What is
                the shape of each batch?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    <code className="text-xs bg-muted px-1 rounded">[32, 5]</code> for the six full
                    batches, <code className="text-xs bg-muted px-1 rounded">[8, 5]</code> for the
                    last batch. DataLoader stacks individual samples along a new batch dimension.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 7. Plugging DataLoader into the Training Loop */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Plugging Into the Training Loop"
            subtitle="The loop body stays identical -- only the data source changes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the payoff. Take the training loop from The Training Loop and
              replace the raw tensor data with a DataLoader. The loop body does not change:
            </p>

            <ComparisonRow
              left={{
                title: 'Before: Raw Tensors (The Training Loop)',
                color: 'amber',
                items: [
                  'x = torch.randn(100, 1)',
                  'y = 2 * x + 1 + noise',
                  'for epoch in range(100):',
                  '    y_hat = model(x)  # all data at once',
                  '    loss = criterion(y_hat, y)',
                  '    optimizer.zero_grad()',
                  '    loss.backward()',
                  '    optimizer.step()',
                ],
              }}
              right={{
                title: 'After: DataLoader (This Lesson)',
                color: 'blue',
                items: [
                  'dataset = SimpleDataset(100)',
                  'loader = DataLoader(dataset, batch_size=8, shuffle=True)',
                  'for epoch in range(100):',
                  '    for x_batch, y_batch in loader:',
                  '        y_hat = model(x_batch)',
                  '        loss = criterion(y_hat, y_batch)',
                  '        optimizer.zero_grad()',
                  '        loss.backward()',
                  '        optimizer.step()',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The four-line loop body is <strong>identical</strong>. The only change is an
              extra inner loop over DataLoader batches. The &ldquo;heartbeat&rdquo; from
              The Training Loop&mdash;forward, loss, backward, update&mdash;does not change.
              DataLoader is a new instrument, not a new song.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Heartbeat">
            The training loop pattern&mdash;forward, loss, backward, update&mdash;is
            always the same. DataLoader just changes how data arrives. The heartbeat
            does not change.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Full integrated code */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The complete integrated code:
            </p>

            <CodeBlock
              language="python"
              filename="training_with_dataloader.py"
              code={`import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Dataset: same y = 2x + 1 data ---
class SimpleDataset(Dataset):
    def __init__(self, num_samples=100):
        torch.manual_seed(42)
        self.x = torch.randn(num_samples, 1)
        self.y = 2 * self.x + 1 + 0.1 * torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- DataLoader: batching + shuffling ---
dataset = SimpleDataset(100)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- Model, loss, optimizer (unchanged from The Training Loop) ---
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# --- Training loop with DataLoader ---
for epoch in range(100):
    epoch_loss = 0.0
    for x_batch, y_batch in loader:
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 20 == 0:
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:3d}: avg loss = {avg_loss:.4f}")

print(f"\\nLearned weight: {model.weight.item():.4f}")  # ~2.0
print(f"Learned bias:   {model.bias.item():.4f}")       # ~1.0`}
            />

            <p className="text-muted-foreground">
              Same convergence. Same result. The model learns <InlineMath math="y = 2x + 1" /> just
              as well&mdash;but now with proper mini-batch SGD, shuffled every epoch, with
              automatic batch handling.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 8. Shuffling Matters (Negative Example) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Shuffling Matters"
            subtitle="A concrete demonstration of why shuffle=True is not optional"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              &ldquo;Shuffling is just nice to have, right?&rdquo; Let us test that.
              Imagine a dataset where the samples are sorted by target value&mdash;all
              low values first, then high values:
            </p>

            <CodeBlock
              language="python"
              filename="sorted_data_problem.py"
              code={`# A Dataset whose samples are sorted by target value
class SortedDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(-5, 5, 200).unsqueeze(1)
        self.y = 2 * self.x + 1  # sorted: targets go from -9 to 11

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

sorted_data = SortedDataset()

# Train WITHOUT shuffling
loader_no_shuffle = DataLoader(sorted_data, batch_size=32, shuffle=False)
# Early batches: all targets near -9
# Late batches: all targets near +11

# Train WITH shuffling
loader_shuffled = DataLoader(sorted_data, batch_size=32, shuffle=True)
# Each batch: targets spread across the full range`}
            />

            <p className="text-muted-foreground">
              Without shuffling, each batch contains samples from a narrow range of
              target values. The model optimizes for the current range, then sees a
              completely different range in the next batch. The gradient estimates are
              biased&mdash;they represent one slice of the data, not the whole picture.
            </p>

            <p className="text-muted-foreground">
              With shuffling, each batch contains a random mix of target values. The
              gradient estimates are noisy but <strong>unbiased</strong>&mdash;they
              represent the full data distribution. This is the polling analogy from
              Batching and SGD in action: a random sample of 32 gives a better estimate
              than a carefully selected 32.
            </p>

            <p className="text-muted-foreground">
              The training curves tell the story. Run both and plot loss over epochs:
            </p>

            <CodeBlock
              language="python"
              filename="shuffle_comparison.py"
              code={`import matplotlib.pyplot as plt

def train_and_record(loader, num_epochs=50):
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    return losses

sorted_data = SortedDataset()
losses_no_shuffle = train_and_record(
    DataLoader(sorted_data, batch_size=32, shuffle=False))
losses_shuffled = train_and_record(
    DataLoader(sorted_data, batch_size=32, shuffle=True))

plt.plot(losses_no_shuffle, label='shuffle=False (sorted)')
plt.plot(losses_shuffled, label='shuffle=True')
plt.xlabel('Epoch'); plt.ylabel('Avg Loss'); plt.legend()
plt.title('Shuffling matters: sorted data oscillates, shuffled converges')
plt.show()`}
            />

            <p className="text-muted-foreground">
              Without shuffling, the loss zigzags&mdash;the model adjusts to one region
              of the data, then overcorrects for the next. With shuffling, the loss
              descends smoothly. Try this in the Colab notebook to see the curves yourself.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Sorted Data Is Catastrophic">
            If your data is sorted by class or target, each batch is biased toward
            one region of the data. Gradients point in the wrong direction.{' '}
            <code className="text-xs">shuffle=True</code> is not a nice-to-have&mdash;it
            is essential for convergence.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 9. Transforms -- Preprocessing on the Fly */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transforms -- Preprocessing on the Fly"
            subtitle="Convert, normalize, and compose data transformations"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Real datasets need preprocessing. Images come as PIL images (from Python&rsquo;s Pillow library) or NumPy arrays&mdash;not
              tensors. Pixel values range from 0 to 255&mdash;not the normalized range models prefer.
              PyTorch handles this with <strong>transforms</strong>: functions that run inside{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code>,
              applied to each sample as it is loaded.
            </p>

            <CodeBlock
              language="python"
              filename="transforms_example.py"
              code={`from torchvision import transforms

# A transform pipeline: compose multiple transforms into one
transform = transforms.Compose([
    transforms.ToTensor(),          # PIL image -> tensor, scales to [0, 1]
    transforms.Normalize(
        mean=(0.1307,),             # MNIST mean
        std=(0.3081,)               # MNIST std
    ),
])

# Each call to dataset[i] runs the transform on-the-fly
# The original data on disk is unchanged`}
            />

            <p className="text-muted-foreground">
              Key point: transforms run in{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code>,
              not upfront. The original data on disk is <strong>never modified</strong>.
              If you call <code className="text-sm bg-muted px-1.5 py-0.5 rounded">dataset[0]</code> twice
              with a random augmentation transform, you can get different results each time.
              This is by design&mdash;augmentation creates variety.
            </p>

            <p className="text-muted-foreground">
              The two most common transforms:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">ToTensor()</code>&mdash;converts
                a PIL image or NumPy array to a float tensor, scales pixel values from [0, 255] to [0, 1]
              </li>
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">Normalize(mean, std)</code>&mdash;subtracts
                the mean and divides by the standard deviation, centering the data
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Transforms Are Lazy">
            Transforms do not modify the original data. They run per-sample
            inside <code className="text-xs">__getitem__</code>. The raw files on
            disk stay untouched. This is why you can use random augmentations&mdash;each
            access can produce a different result.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 10. MNIST Data Inspection */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="MNIST -- Your First Real Dataset"
            subtitle="60,000 handwritten digits, loaded in three lines"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torchvision.datasets</code> provides
              pre-built Dataset classes for common benchmarks. MNIST&mdash;60,000 handwritten
              digit images&mdash;is the classic:
            </p>

            <CodeBlock
              language="python"
              filename="load_mnist.py"
              code={`from torchvision import datasets, transforms

# Define the transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),                    # PIL -> tensor [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # center the data
])

# Download and load MNIST
train_dataset = datasets.MNIST(
    root='./data',          # where to store downloaded files
    train=True,             # training set (60,000 images)
    download=True,          # download if not already present
    transform=transform     # apply our pipeline
)

print(f"Dataset size: {len(train_dataset)}")  # 60000`}
            />

            <p className="text-muted-foreground">
              Now wrap it in a DataLoader and inspect one batch:
            </p>

            <CodeBlock
              language="python"
              filename="inspect_mnist.py"
              code={`train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Grab one batch
images, labels = next(iter(train_loader))

print(f"Batch images shape: {images.shape}")  # [64, 1, 28, 28]
print(f"Batch labels shape: {labels.shape}")  # [64]
print(f"Image dtype: {images.dtype}")         # torch.float32
print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")
# After normalization: approximately [-0.42, 2.82]`}
            />

            <p className="text-muted-foreground">
              The batch shape is <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[64, 1, 28, 28]</code>&mdash;64
              images, 1 channel (grayscale), 28 pixels tall, 28 pixels wide. Labels are
              integers from 0 to 9.
            </p>

            <p className="text-muted-foreground">
              Apply the debugging trinity from Tensors: <strong>shape, dtype,
              device</strong>. First thing you do with any new data source is print
              these three for one batch. It catches most data pipeline bugs immediately.
            </p>

            <p className="text-muted-foreground">
              But tensors are abstract. What do these digits actually look like? Visualize a
              sample grid:
            </p>

            <CodeBlock
              language="python"
              filename="visualize_mnist.py"
              code={`import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')
plt.suptitle("Sample MNIST Digits")
plt.tight_layout()
plt.show()`}
            />

            <p className="text-muted-foreground">
              These are the handwritten digits your model will learn to classify in the
              next lesson. Each image is 28x28 pixels&mdash;small, grayscale, and
              surprisingly varied. Some threes look like eights. Some ones are
              slanted. This is the messiness of real data.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Debugging Trinity">
            Shape, dtype, device&mdash;check these first on any new DataLoader.
            For MNIST: shape <code className="text-xs">[B, 1, 28, 28]</code>,
            dtype <code className="text-xs">float32</code>,
            values in the normalized range. If any of these are wrong, the problem
            is in your transforms.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Quick check: MNIST batch dimensions */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Quick Check</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                Your DataLoader yields a batch of MNIST images with shape{' '}
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[64, 1, 28, 28]</code>.
              </p>

              <p className="text-muted-foreground text-sm">
                <strong>1.</strong> What is the batch size?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    64&mdash;the first dimension is always the batch size.
                  </p>
                </div>
              </details>

              <p className="text-muted-foreground text-sm">
                <strong>2.</strong> What does the 1 represent?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    One color channel&mdash;MNIST is grayscale. An RGB image would have 3 channels.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 11. Check 2: Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                A colleague loads a custom image dataset as a list of PIL images and a list
                of labels. They convert everything to tensors upfront (one giant tensor),
                then loop with manual index slicing. Training works but is slow and
                uses a lot of memory. What would you suggest?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <p className="text-muted-foreground">
                    Implement a custom <code className="text-xs bg-muted px-1 rounded">Dataset</code> that
                    stores file paths (not the images themselves) and loads each image lazily
                    in <code className="text-xs bg-muted px-1 rounded">__getitem__</code>. Use{' '}
                    <code className="text-xs bg-muted px-1 rounded">transforms.Compose</code> for{' '}
                    <code className="text-xs bg-muted px-1 rounded">ToTensor()</code> and{' '}
                    <code className="text-xs bg-muted px-1 rounded">Normalize()</code>. Wrap
                    in a <code className="text-xs bg-muted px-1 rounded">DataLoader</code> with{' '}
                    <code className="text-xs bg-muted px-1 rounded">shuffle=True</code> for
                    automatic batching and shuffling. This loads only one batch at a time
                    instead of the entire dataset, cutting memory usage dramatically.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 12. The Dataset Contract (Negative Example) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When the Contract Breaks"
            subtitle="What happens when __getitem__ returns inconsistent shapes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DataLoader works by calling{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code> on
              multiple indices, then <strong>stacking</strong> the results into a single
              batch tensor. This means every sample must have the same shape. What happens
              if they do not?
            </p>

            <CodeBlock
              language="python"
              filename="broken_dataset.py"
              code={`class BrokenDataset(Dataset):
    def __init__(self):
        self.data = [
            torch.randn(5),    # shape [5]
            torch.randn(5),    # shape [5]
            torch.randn(3),    # shape [3] -- inconsistent!
            torch.randn(5),    # shape [5]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0  # sample, label

loader = DataLoader(BrokenDataset(), batch_size=4)
batch = next(iter(loader))
# RuntimeError: stack expects each tensor to be equal size,
# but got [5] at entry 0 and [3] at entry 2`}
            />

            <p className="text-muted-foreground">
              The error message says &ldquo;stack expects each tensor to be equal size.&rdquo;
              Now you know why: DataLoader calls{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.stack()</code> to
              combine individual samples into a batch. Stacking requires uniform shapes.
              If your Dataset returns tensors of different sizes, collation fails.
            </p>

            <p className="text-muted-foreground">
              This is a common debugging scenario with custom datasets. The fix is
              always in <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__getitem__</code>&mdash;ensure
              every sample has the same shape (pad, crop, resize, or filter out
              inconsistent samples before creating the Dataset).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Collation Errors">
            If you see &ldquo;stack expects each tensor to be equal size,&rdquo; the
            problem is in your Dataset, not your DataLoader. Check that every
            sample from <code className="text-xs">__getitem__</code> returns the
            same shapes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 13. Practice: Colab Notebook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Try It Yourself"
            subtitle="Build muscle memory with Datasets and DataLoaders"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Reading about data pipelines is not enough. Open the notebook and
                work through these exercises:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>Implement a custom Dataset</strong> for the y=2x+1 data,
                  wrap in DataLoader, iterate and print batch shapes (guided)
                </li>
                <li>
                  <strong>Load MNIST with torchvision.datasets</strong>, apply
                  transforms, inspect batch shapes and value ranges (guided)
                </li>
                <li>
                  <strong>Integrate DataLoader into the training loop</strong> from
                  The Training Loop&mdash;train on y=2x+1 with proper batching (supported)
                </li>
                <li>
                  <strong>Experiment with batch sizes</strong> (1, 32, 256, full-batch)&mdash;measure
                  iterations per epoch and observe loss curve differences (supported)
                </li>
                <li>
                  <strong>Write a custom Dataset for a CSV file</strong> (provided), apply
                  basic transforms, train a model with DataLoader (independent)
                </li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-2-1-datasets-and-dataloaders.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Exercises 1&ndash;2 are guided with starter code and hints. Exercises 3&ndash;4
                are supported (template provided). Exercise 5 is independent.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Inspect First, Train Second">
            Before training with any DataLoader, always grab one batch and check shape,
            dtype, and value range. This catches 90% of data pipeline bugs before they
            become mysterious training failures.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 14. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Dataset = how to access one sample',
                description:
                  '__len__ returns the count, __getitem__ returns one (input, target) pair by index. It does not batch, shuffle, or iterate. It just serves one item at a time.',
              },
              {
                headline: 'DataLoader = how to batch, shuffle, and iterate',
                description:
                  'Wraps a Dataset. Configurable batch_size and shuffle. Yields batched tensors with an added batch dimension. Handles the last incomplete batch automatically.',
              },
              {
                headline: 'Transforms = preprocessing applied per-sample in __getitem__',
                description:
                  'ToTensor() converts images to tensors. Normalize() centers the data. Compose() chains them. Original data on disk is never modified.',
              },
              {
                headline: 'The training loop body does not change',
                description:
                  'Forward, loss, backward, update -- the same heartbeat. DataLoader replaces the data source. The loop body stays identical.',
              },
              {
                headline: 'shuffle=True implements the polling analogy',
                description:
                  'Random batches give unbiased gradient estimates. Without shuffling, sorted data creates biased batches that prevent convergence.',
              },
              {
                headline: 'Inspect one batch first: shape, dtype, device',
                description:
                  'The debugging trinity from Tensors applies to DataLoader output. Print these three for one batch before training -- it catches most pipeline bugs.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 15. Next Step */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Comes Next"
            subtitle="From plumbing to predictions"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have the plumbing. A Dataset that loads MNIST images, transforms
              that convert them to normalized tensors, and a DataLoader that batches
              and shuffles them. Next lesson: you use it. MNIST&mdash;60,000 handwritten
              digits. You will build a model, train it, and see real predictions. The
              training loop you already know, the data pipeline you just learned, and
              a new loss function (cross-entropy) to handle 10 classes instead of one
              continuous target.
            </p>
          </div>
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
