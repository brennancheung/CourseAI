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
  GradientCard,
  ComparisonRow,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * MNIST Project — Lesson 2 of Module 2.2 (Real Data)
 *
 * STRETCH / PROJECT lesson: the student's first end-to-end deep learning project.
 * Builds, trains, and evaluates a fully-connected classifier on MNIST, introducing
 * the classification-specific tools (softmax, cross-entropy, accuracy) along the way.
 *
 * New concepts (3):
 *   1. Cross-entropy loss (INTRODUCED — intuition + formula + code, no info-theory derivation)
 *   2. Softmax function (INTRODUCED — what it does + code)
 *   3. Accuracy metric (INTRODUCED — argmax + comparison)
 *
 * Central insight: Everything the student has built (training loop, nn.Module, DataLoader,
 * regularization concepts) comes together to train a model that actually reads handwriting.
 * The training loop heartbeat does not change — only the instruments.
 */

export function MnistProjectLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="MNIST Project"
            description="Build, train, and evaluate your first real classifier — a network that reads handwritten digits."
            category="Real Data"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Train a fully-connected neural network on 60,000 handwritten digits and
            evaluate it on held-out test data. Along the way, pick up the tools that
            classification requires: softmax, cross-entropy loss, and accuracy. By the end,
            you will have a model that reads handwriting at 97%+ accuracy&mdash;your first
            real deep learning result.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Project Lesson">
            This is a PROJECT lesson. The Colab notebook is the primary deliverable.
            Read through each section here, then build the model yourself in the notebook.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'Fully-connected network only — no convolutional layers (those come in Series 3)',
              'No hyperparameter tuning — we pick reasonable defaults, not search for optimal',
              'No learning rate schedulers or gradient clipping',
              'No saving/loading models — next module',
              'No GPU training — CPU is fast enough for MNIST',
              'Celebrate 97–98% accuracy — that is appropriate for this architecture',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook — Before/After Demo */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What You Are About to Build"
            subtitle="From loading data to reading handwriting"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Datasets and DataLoaders, you loaded MNIST images, inspected their shapes,
              and visualized a sample grid. You had the data but no model. Here is what changes
              today:
            </p>

            <div className="rounded-lg border bg-muted/30 p-5 font-mono text-sm space-y-1">
              <p className="text-muted-foreground">
                <span className="text-emerald-500 font-bold">Prediction: 7</span>{' '}
                (confidence: 99.2%) <span className="text-emerald-500">&#10003;</span>
              </p>
              <p className="text-muted-foreground">
                <span className="text-emerald-500 font-bold">Prediction: 2</span>{' '}
                (confidence: 97.8%) <span className="text-emerald-500">&#10003;</span>
              </p>
              <p className="text-muted-foreground">
                <span className="text-emerald-500 font-bold">Prediction: 1</span>{' '}
                (confidence: 98.5%) <span className="text-emerald-500">&#10003;</span>
              </p>
              <p className="text-muted-foreground">
                <span className="text-rose-500 font-bold">Prediction: 8</span>{' '}
                (confidence: 52.1%) <span className="text-rose-500">&#10007;</span>{' '}
                <span className="text-muted-foreground/60">(actual: 3)</span>
              </p>
            </div>

            <p className="text-muted-foreground">
              By the end of this lesson, you will have trained the model that produced
              predictions like these. Everything you have built so far&mdash;the training
              loop, nn.Module, Dataset and DataLoader&mdash;comes together here. This
              is the moment where &ldquo;I understand deep learning&rdquo; becomes
              &ldquo;I have done deep learning.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="First Real Result">
            Every previous lesson used toy data&mdash;synthetic lines, parabolas, tensors
            with a handful of points. Today you train on 60,000 real images and get a
            result that actually does something.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. The Classification Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Classification Problem"
            subtitle="From continuous output to discrete categories"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every model you have trained so far produced a <strong>continuous
              output</strong>&mdash;a single number. In What Is Learning?, the model predicted
              house prices. In The Training Loop, it learned{' '}
              <InlineMath math="y = 2x + 1" />. The loss function was MSE: how far is
              the predicted number from the true number?
            </p>

            <p className="text-muted-foreground">
              MNIST is different. Each image is a handwritten digit. The model must output
              one of <strong>10 discrete categories</strong> (0 through 9). This changes two
              things:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>How the model makes a prediction:</strong> instead of one number,
                it outputs 10 numbers&mdash;one score per digit class
              </li>
              <li>
                <strong>How we measure wrongness:</strong> MSE does not work well
                for classification. We need a loss function designed for categorical
                outputs.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Let&rsquo;s see why MSE fails before introducing the replacement.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Regression vs Classification">
            <strong>Regression:</strong> predict a continuous number (price, temperature).{' '}
            <strong>Classification:</strong> predict a category (digit, animal, sentiment).
            Same training loop, different loss function.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 4. Softmax: From Scores to Probabilities */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Softmax: From Scores to Probabilities"
            subtitle="Raw model outputs are not probabilities — softmax makes them so"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A neural network with 10 output neurons produces 10 raw
              numbers&mdash;called <strong>logits</strong>. These are arbitrary: they can
              be negative, large, and they do not sum to 1. They are{' '}
              <strong>not</strong> probabilities.
            </p>

            <CodeBlock
              language="python"
              filename="raw_logits.py"
              code={`# Raw model output (logits) for one image
# These are NOT probabilities — negative, don't sum to 1
logits = [-1.2, 0.5, 0.1, -0.3, 0.2, -0.8, 0.4, 3.8, -0.1, 0.7]
# The model "thinks" class 7 is most likely (highest score: 3.8)
# But what does "3.8" mean in terms of confidence?`}
            />

            <p className="text-muted-foreground">
              Remember sigmoid from Activation Functions? It squashes a single value
              to the range [0, 1]. <strong>Softmax is sigmoid generalized to multiple
              classes.</strong> It takes a vector of logits and converts them to
              probabilities that are all positive and sum to 1:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}" />
            </div>

            <p className="text-muted-foreground">
              Let&rsquo;s work through one element to see the math in action. Take
              class 7, which has the highest logit (3.8):
            </p>

            <div className="py-3 px-6 bg-muted/50 rounded-lg space-y-1 text-sm">
              <p className="text-muted-foreground">
                <strong>Step 1:</strong> Compute <InlineMath math="e^{3.8} = 44.70" />
              </p>
              <p className="text-muted-foreground">
                <strong>Step 2:</strong> Sum all exponentials: <InlineMath math="e^{-1.2} + e^{0.5} + \cdots + e^{3.8} + \cdots + e^{0.7} = 54.58" />
              </p>
              <p className="text-muted-foreground">
                <strong>Step 3:</strong> Divide: <InlineMath math="44.70 \;/\; 54.58 = 0.82" />&mdash;class 7 gets 82% probability.
              </p>
            </div>

            <p className="text-muted-foreground">
              Here is the full vector computed with PyTorch:
            </p>

            <CodeBlock
              language="python"
              filename="softmax_example.py"
              code={`import torch
import torch.nn.functional as F

logits = torch.tensor([-1.2, 0.5, 0.1, -0.3, 0.2, -0.8, 0.4, 3.8, -0.1, 0.7])
probs = F.softmax(logits, dim=0)

# Logits:       [-1.2,  0.5,  0.1, -0.3,  0.2, -0.8,  0.4,  3.8, -0.1,  0.7]
# Probabilities: [0.01, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.82, 0.02, 0.04]
#                                                              ^^^^
# Class 7: 82% confidence. All values positive, sum to 1.0
print(f"Sum of probabilities: {probs.sum():.4f}")  # 1.0000`}
            />

            <p className="text-muted-foreground">
              The exponential <InlineMath math="e^{z_i}" /> makes everything positive.
              Dividing by the sum makes everything add up to 1. The highest logit gets
              the highest probability&mdash;and the gap is amplified. The model&rsquo;s
              raw score of 3.8 for class 7 becomes an 82% confidence after softmax.
            </p>

            <p className="text-muted-foreground">
              Important: <strong>you will not apply softmax yourself in the model.</strong> PyTorch&rsquo;s
              loss function handles it internally, as you will see in the next section.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Sigmoid to Softmax">
            Sigmoid squashes one number to [0, 1]. Softmax squashes N numbers to
            probabilities that sum to 1. Same idea, multiple classes.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5. Cross-Entropy Loss: Confidence Penalty */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Cross-Entropy Loss: Confidence Penalty"
            subtitle="A loss function that cares how confident you are when you are wrong"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              MSE measures distance between numbers. For regression, that is exactly right.
              For classification, it has a critical flaw:{' '}
              <strong>it treats all errors equally</strong>, regardless of confidence.
            </p>

            <p className="text-muted-foreground">
              Imagine a model classifying an image of digit 0. It outputs probabilities
              across all 10 classes. The true label is class 0, but the model is
              confidently wrong&mdash;it puts 80% on class 3, only 2% on class 0,
              and spreads the remaining 18% across the other 8 classes. Compare how
              MSE and cross-entropy react:
            </p>

            <ComparisonRow
              left={{
                title: 'MSE: Moderate Penalty',
                color: 'amber',
                items: [
                  'Model says: 80% class 3, 2% class 0, ~2% each elsewhere',
                  'Confidently wrong — should be heavily penalized',
                  'MSE penalty: moderate (just the squared difference)',
                  'MSE does not distinguish "confident wrong" from "slightly wrong"',
                ],
              }}
              right={{
                title: 'Cross-Entropy: Huge Penalty',
                color: 'blue',
                items: [
                  'Model says: 80% class 3, 2% class 0, ~2% each elsewhere',
                  'Confidently wrong — IS heavily penalized',
                  'Cross-entropy penalty: enormous (-log(0.02) = 3.91)',
                  'Being confidently wrong is the worst thing a classifier can do',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Cross-entropy is defined as the negative log of the probability assigned to the
              correct class:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="L = -\log(p_{\text{correct}})" />
            </div>

            <p className="text-muted-foreground">
              The intuition: &ldquo;how surprised should you be?&rdquo; If the model gives
              the correct class high probability, the loss is small. If it gives low
              probability, the loss is huge. Look at the concrete values:
            </p>

            <div className="grid gap-3 md:grid-cols-3">
              <GradientCard title="Confident + Correct" color="emerald">
                <div className="space-y-1 text-sm">
                  <p><InlineMath math="p_{\text{correct}} = 0.95" /></p>
                  <p><InlineMath math="-\log(0.95) = 0.05" /></p>
                  <p className="text-muted-foreground">Tiny loss. Model is right and knows it.</p>
                </div>
              </GradientCard>
              <GradientCard title="Unsure" color="amber">
                <div className="space-y-1 text-sm">
                  <p><InlineMath math="p_{\text{correct}} = 0.50" /></p>
                  <p><InlineMath math="-\log(0.50) = 0.69" /></p>
                  <p className="text-muted-foreground">Moderate loss. Model is uncertain.</p>
                </div>
              </GradientCard>
              <GradientCard title="Confident + Wrong" color="rose">
                <div className="space-y-1 text-sm">
                  <p><InlineMath math="p_{\text{correct}} = 0.01" /></p>
                  <p><InlineMath math="-\log(0.01) = 4.61" /></p>
                  <p className="text-muted-foreground">Huge loss. Model is wrong and confident.</p>
                </div>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              From 0.05 to 4.61&mdash;a 90x difference. Cross-entropy{' '}
              <strong>punishes confident wrong answers severely</strong>. This is exactly
              what classification needs. It is a wrongness score&mdash;like MSE&mdash;but
              one that cares about confidence.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Wrongness Score v2">
            MSE is the wrongness score for regression: how far off is the number?
            Cross-entropy is the wrongness score for classification: how confident
            were you on the wrong answer?
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* nn.CrossEntropyLoss in PyTorch */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In PyTorch, swap{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.MSELoss()</code> for{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.CrossEntropyLoss()</code>.
              Same heartbeat, new instrument:
            </p>

            <CodeBlock
              language="python"
              filename="cross_entropy_usage.py"
              code={`import torch.nn as nn

# Regression (what you used before)
# criterion = nn.MSELoss()

# Classification (what you need now)
criterion = nn.CrossEntropyLoss()

# Usage: takes raw LOGITS and integer labels
logits = torch.randn(4, 10)   # batch of 4, 10 classes
labels = torch.tensor([3, 7, 1, 9])  # true class for each sample

loss = criterion(logits, labels)
print(f"Loss: {loss.item():.4f}")`}
            />

            <p className="text-muted-foreground">
              Two things to notice:{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.CrossEntropyLoss</code> takes{' '}
              <strong>raw logits</strong>, not softmax output. It applies log-softmax
              internally for numerical stability. And labels are <strong>integer class
              indices</strong> (not one-hot vectors).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Do NOT Add Softmax">
            PyTorch&rsquo;s{' '}
            <code className="text-xs">nn.CrossEntropyLoss</code> applies log-softmax
            internally. If you add{' '}
            <code className="text-xs">F.softmax()</code> before the loss, you
            double-apply it. The loss values will be wrong and training will not
            converge properly.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 6. Check 1 — Predict-and-Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                A model outputs logits <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[2.0, -1.0, 0.5]</code> for
                a 3-class problem. The true label is class 0.
              </p>

              <p className="text-muted-foreground text-sm">
                <strong>1.</strong> Which class does the model predict?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    Class 0&mdash;it has the highest logit (2.0). The predicted class
                    is always the one with the highest score.
                  </p>
                </div>
              </details>

              <p className="text-muted-foreground text-sm">
                <strong>2.</strong> Is this prediction correct?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    Yes&mdash;the predicted class (0) matches the true label (0).
                  </p>
                </div>
              </details>

              <p className="text-muted-foreground text-sm">
                <strong>3.</strong> Is the cross-entropy loss high or low?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    Low. Softmax assigns class 0 the highest probability (about 0.79 for
                    these logits), so <InlineMath math="-\log(0.79) \approx 0.24" />&mdash;a
                    small loss. The model is correct and reasonably confident.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 7. Building the Model */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Building the Model"
            subtitle="Flatten the image, stack some layers, classify"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              MNIST images are <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[1, 28, 28]</code> tensors&mdash;1
              channel, 28 pixels tall, 28 wide. A fully-connected layer
              (<code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Linear</code>) expects a flat vector, not
              a 2D grid. So the first step is to <strong>flatten</strong> each image
              from <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[1, 28, 28]</code> into a{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">[784]</code> vector (<InlineMath math="28 \times 28 = 784" />).
              You could call <code className="text-sm bg-muted px-1.5 py-0.5 rounded">x.view(-1, 784)</code> in{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code>, but{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Flatten()</code> does the same
              thing as a module&mdash;it flattens every dimension after the batch dimension
              into a single vector, and it slots cleanly into the layer pipeline.
            </p>

            <p className="text-muted-foreground">
              After flattening, stack three linear layers with ReLU activations.
              The architecture: 784 inputs &rarr; 256 hidden &rarr; 128 hidden &rarr; 10 outputs.
              Simple, readable, effective.
            </p>

            <CodeBlock
              language="python"
              filename="mnist_model_v1.py"
              code={`import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()           # [B, 1, 28, 28] -> [B, 784]
        self.layer1 = nn.Linear(784, 256)     # 784 inputs -> 256 hidden
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)     # 256 -> 128 hidden
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 10)      # 128 -> 10 classes

    def forward(self, x):
        x = self.flatten(x)    # flatten the image
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)    # raw logits -- no softmax!
        return x

model = MNISTClassifier()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~235,146`}
            />

            <p className="text-muted-foreground">
              About 235,000 parameters. Most come from the first layer (784 &times; 256 = 200,704
              weights alone). Notice: the output layer has <strong>no activation
              function</strong>. It produces raw logits, and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.CrossEntropyLoss</code> handles
              the softmax internally.
            </p>

            <p className="text-muted-foreground">
              This is the LEGO bricks pattern from nn.Module&mdash;snap layers together
              in <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__init__</code>,
              wire them in <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code>.
              You have done this before with smaller networks. This one is just deeper.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="No Softmax in forward()">
            The output layer produces raw logits. Do not add softmax or
            log-softmax in <code className="text-xs">forward()</code>.{' '}
            <code className="text-xs">nn.CrossEntropyLoss</code> handles it.
            When you want probabilities at inference time, apply softmax then.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 8. The Training Loop — Putting It All Together */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Training Loop"
            subtitle="Same heartbeat, new instruments"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The training loop body is <strong>identical</strong> to The Training Loop.
              Forward, loss, backward, update. The only differences: the model is deeper,
              the loss is cross-entropy, and the data comes from a DataLoader. The heartbeat
              does not change.
            </p>

            <CodeBlock
              language="python"
              filename="train_mnist.py"
              code={`from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# --- Data (from Datasets and DataLoaders) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std, precomputed from training set
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- Model, loss, optimizer ---
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training ---
num_epochs = 5

for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward + update (the heartbeat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track accuracy
        epoch_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)  # highest logit = predicted class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")`}
            />

            <p className="text-muted-foreground">
              Two new pieces worth noticing:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.argmax(outputs, dim=1)</code>&mdash;the
                predicted class is the index with the highest logit. This is how a
                classification model &ldquo;picks&rdquo; its answer.
              </li>
              <li>
                <strong>Accuracy tracking</strong>&mdash;count correct predictions,
                divide by total. Unlike loss (which the model optimizes), accuracy is the
                metric <em>you</em> care about as a human.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Expected output after 5 epochs: training accuracy around 97&ndash;98%.
              Five epochs takes about 30 seconds on CPU.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Loss vs Accuracy">
            The model optimizes <strong>loss</strong> (cross-entropy). You evaluate
            with <strong>accuracy</strong> (% correct). They correlate but are not the
            same&mdash;a model can have low loss but mediocre accuracy on tricky edge cases.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 9. Evaluation — Test Set Performance */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Evaluation — Test Set Performance"
            subtitle="How does the model do on digits it has never seen?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Training accuracy tells you how well the model memorized the training data.
              What matters is how it performs on <strong>new, unseen data</strong>&mdash;the
              test set. In Datasets and DataLoaders, you saw
              the <code className="text-sm bg-muted px-1.5 py-0.5 rounded">train=True/False</code> argument.
              Now you use it:
            </p>

            <CodeBlock
              language="python"
              filename="evaluate_mnist.py"
              code={`# Load test data (10,000 images the model has never seen)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate
model.eval()  # set to evaluation mode (matters for dropout/batch norm)

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # no gradients needed for evaluation
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100.0 * correct / total
avg_test_loss = test_loss / len(test_loader)
print(f"Test accuracy: {test_accuracy:.1f}%")  # ~97%
print(f"Test loss:     {avg_test_loss:.4f}")`}
            />

            <p className="text-muted-foreground">
              Three things are different from the training loop:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.eval()</code>&mdash;switches
                the model to evaluation mode. For now this has no effect (our model has no
                dropout or batch norm). It will matter when we add regularization.
              </li>
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.no_grad()</code>&mdash;disables
                gradient tracking. You learned about this in Autograd. No gradients needed
                because we are not updating weights&mdash;just measuring performance. Saves
                memory and computation.
              </li>
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">shuffle=False</code>&mdash;no
                need to shuffle the test set. Order does not matter for evaluation.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Expected result: around 97% test accuracy. That means the model correctly
              identifies about 9,700 out of 10,000 handwritten digits it has never seen
              before. Not bad for 235,000 parameters and 30 seconds of training.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="torch.no_grad()">
            You learned about gradient tracking in Autograd. Wrapping evaluation
            in <code className="text-xs">torch.no_grad()</code> disables gradient
            computation&mdash;no backward pass needed when you are just measuring
            performance. Saves memory and speeds things up.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Seeing What the Model Learned */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Seeing What the Model Learned"
            subtitle="What does 97% accuracy actually look like?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              97% is a number. Let&rsquo;s make it concrete. Visualize the model&rsquo;s
              predictions on test images&mdash;correct ones with green borders,
              incorrect ones with red:
            </p>

            <CodeBlock
              language="python"
              filename="visualize_predictions.py"
              code={`import matplotlib.pyplot as plt

# Get one batch of test images
images, labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    predicted = torch.argmax(outputs, dim=1)
    confidences = probs.max(dim=1).values

# Plot correct and incorrect predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Model Predictions (green=correct, red=incorrect)")

shown = 0
for i in range(len(images)):
    if shown >= 10:
        break
    ax = axes[shown // 5, shown % 5]
    ax.imshow(images[i].squeeze(), cmap='gray')

    is_correct = predicted[i] == labels[i]
    color = 'green' if is_correct else 'red'
    title = f"Pred: {predicted[i].item()} ({confidences[i]:.0%})"
    if not is_correct:
        title += f"\\nTrue: {labels[i].item()}"
    ax.set_title(title, color=color, fontsize=9)
    ax.axis('off')
    shown += 1

plt.tight_layout()
plt.show()`}
            />

            <p className="text-muted-foreground">
              Look at the incorrect predictions carefully. They are almost always
              genuinely ambiguous&mdash;a sloppy 3 that looks like an 8, a tilted 7
              that resembles a 1. The model is not failing on easy cases. It is
              struggling where humans would also hesitate.
            </p>

            <p className="text-muted-foreground">
              This is what 3% error looks like: about 300 images out of 10,000 where the
              handwriting is messy enough to confuse a simple fully-connected network.
              Convolutional networks (Series 3) will handle these better by understanding
              spatial patterns&mdash;but that is a future lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The 3% That Fail">
            The model&rsquo;s errors are not random. They cluster around ambiguous
            digits&mdash;images where the handwriting is genuinely hard to read. This
            is a property of the data, not a bug in the model.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 11. Check 2 — Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                Your model gets <strong>99.2% training accuracy</strong> but{' '}
                <strong>96.8% test accuracy</strong>. What is happening, and what would
                you try first?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <p className="text-muted-foreground">
                    This is the <strong>scissors pattern</strong> from Overfitting and
                    Regularization&mdash;the gap between training and test performance
                    signals overfitting. The model has memorized training data details
                    that do not generalize.
                  </p>
                  <p className="text-muted-foreground mt-2">
                    First things to try: add <strong>dropout</strong> (randomly silence
                    neurons during training) or use <strong>weight decay</strong> in the
                    optimizer (penalize large weights). Both reduce the model&rsquo;s ability
                    to memorize, forcing it to learn more general patterns. We do exactly
                    this in the next section.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 12a. The Improved Model — Architecture */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Improved Model"
            subtitle="Adding regularization layers to the architecture"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Overfitting and Regularization, you learned the concepts: dropout silences
              random neurons during training, batch normalization stabilizes activations,
              and weight decay penalizes large weights. You saw these in the RegularizationExplorer
              widget. Now you implement them in actual PyTorch code.
            </p>

            <CodeBlock
              language="python"
              filename="mnist_model_v2.py"
              code={`class ImprovedMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Layer 1: Linear -> BatchNorm -> ReLU -> Dropout
        self.layer1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)      # normalize activations
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)      # silence 30% of neurons

        # Layer 2: Linear -> BatchNorm -> ReLU -> Dropout
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        # Output layer (no activation, no dropout, no batch norm)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.drop1(self.relu1(self.bn1(self.layer1(x))))
        x = self.drop2(self.relu2(self.bn2(self.layer2(x))))
        x = self.layer3(x)
        return x`}
            />

            <p className="text-muted-foreground">
              The pattern for each hidden layer:{' '}
              <strong>Linear &rarr; BatchNorm &rarr; ReLU &rarr; Dropout</strong>. Each piece
              has a job:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.BatchNorm1d(features)</code>&mdash;normalizes
                activations between layers. Same concept from Training Dynamics, now one
                line of code.
              </li>
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Dropout(p=0.3)</code>&mdash;randomly
                zeroes 30% of activations during training. Same &ldquo;randomly silence
                neurons&rdquo; concept from Overfitting and Regularization.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Now <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.train()</code> and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.eval()</code>{' '}
              <strong>actually matter</strong>. In training mode, dropout silences random
              neurons and batch norm uses batch statistics. In eval mode, dropout is
              disabled and batch norm uses the running averages it accumulated during training.
              Forgetting to call{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.eval()</code> before
              testing will give you inconsistent, noisy test results.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="model.eval() vs model.train()">
            <code className="text-xs">model.train()</code>: enables dropout, uses batch
            statistics for batch norm.{' '}
            <code className="text-xs">model.eval()</code>: disables dropout, uses running
            statistics for batch norm. Always set the right mode.
          </ConceptBlock>
          <TipBlock title="The Complete Recipe">
            This is the &ldquo;complete training recipe&rdquo; from Overfitting
            and Regularization, now implemented for real: batch norm + ReLU + dropout
            in each hidden layer, weight decay in the optimizer,{' '}
            <code className="text-xs">model.train()</code>/<code className="text-xs">model.eval()</code> toggling.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 12b. Training the Improved Model + Comparison */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Training the Improved Model"
            subtitle="The scissors pattern, observed in your own training"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Add weight decay to the optimizer&mdash;the L2 regularization penalty
              from Overfitting and Regularization&mdash;and train both models to compare:
            </p>

            <CodeBlock
              language="python"
              filename="train_improved.py"
              code={`model = ImprovedMNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(5):
    model.train()  # enable dropout + batch norm training mode
    epoch_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Evaluate on test set each epoch
    model.eval()  # disable dropout + use running batch norm stats
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    train_acc = 100.0 * correct / total
    test_acc = 100.0 * test_correct / test_total
    print(f"Epoch {epoch+1}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%")`}
            />

            <p className="text-muted-foreground">
              Here is what you should see when you compare the simple model (no
              regularization) against the improved model (batch norm + dropout + weight
              decay) over 5 epochs:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg border bg-muted/20 p-4 space-y-2">
                <p className="text-sm font-semibold">Simple Model (no regularization)</p>
                <div className="font-mono text-xs text-muted-foreground space-y-0.5">
                  <p>Epoch 1: train=93.8%, test=95.5%</p>
                  <p>Epoch 2: train=96.9%, test=96.7%</p>
                  <p>Epoch 3: train=97.8%, test=97.0%</p>
                  <p>Epoch 4: train=98.5%, test=97.1%</p>
                  <p>Epoch 5: train=99.0%, test=97.2%</p>
                </div>
                <p className="text-xs text-rose-500 font-medium">
                  Gap growing: 99.0% train vs 97.2% test (1.8% gap)
                </p>
              </div>
              <div className="rounded-lg border bg-muted/20 p-4 space-y-2">
                <p className="text-sm font-semibold">Improved Model (regularized)</p>
                <div className="font-mono text-xs text-muted-foreground space-y-0.5">
                  <p>Epoch 1: train=92.1%, test=96.2%</p>
                  <p>Epoch 2: train=96.0%, test=97.3%</p>
                  <p>Epoch 3: train=96.8%, test=97.7%</p>
                  <p>Epoch 4: train=97.2%, test=97.9%</p>
                  <p>Epoch 5: train=97.5%, test=98.1%</p>
                </div>
                <p className="text-xs text-emerald-500 font-medium">
                  Gap closing: 97.5% train vs 98.1% test (0.6% gap)
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              This is the scissors pattern from Overfitting and Regularization, now in your
              own training output. The simple model&rsquo;s training accuracy keeps climbing
              while its test accuracy plateaus&mdash;the scissors opening. The improved
              model&rsquo;s training and test accuracy stay close together&mdash;the scissors
              closing. (Test slightly exceeding train is normal with dropout&mdash;dropout
              suppresses neurons during training but uses the full network at test time,
              so the model is slightly stronger during evaluation.)
            </p>

            <p className="text-muted-foreground">
              To see the full picture, plot the training curves side by side:
            </p>

            <CodeBlock
              language="python"
              filename="plot_training_curves.py"
              code={`import matplotlib.pyplot as plt

# After training both models, collect per-epoch train/test accuracy
# simple_train_acc, simple_test_acc = [...], [...]
# improved_train_acc, improved_test_acc = [...], [...]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Simple model
ax1.plot(simple_train_acc, 'b-', label='Train')
ax1.plot(simple_test_acc, 'r--', label='Test')
ax1.set_title('Simple Model (no regularization)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.set_ylim(90, 100)

# Improved model
ax2.plot(improved_train_acc, 'b-', label='Train')
ax2.plot(improved_test_acc, 'r--', label='Test')
ax2.set_title('Improved Model (regularized)')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.set_ylim(90, 100)

plt.suptitle('The Scissors Pattern: Regularization Closes the Gap')
plt.tight_layout()
plt.show()`}
            />

            <p className="text-muted-foreground">
              The simple model&rsquo;s plot shows the train curve pulling away from the test
              curve&mdash;the visual scissors you saw in the RegularizationExplorer widget.
              The improved model&rsquo;s curves stay together. That is the evidence:
              regularization works, and now you have seen it in your own training.
            </p>

            <p className="text-muted-foreground">
              A note on expectations: do not chase 99%+. A fully-connected network on
              MNIST typically tops out around 98%. Getting that last 1&ndash;2% requires
              convolutional layers that understand spatial patterns&mdash;the focus
              of Series 3. Your 98% is an excellent result for this architecture.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="train() vs eval()">
            With dropout and batch norm in the model,{' '}
            <code className="text-xs">model.train()</code> and{' '}
            <code className="text-xs">model.eval()</code> are no longer
            optional. Forgetting <code className="text-xs">model.eval()</code> before
            testing gives noisy, unreliable accuracy numbers.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 13. Practice: Colab Notebook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="The Colab notebook is the real deliverable"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                You have read the walkthrough. Now build the model yourself. The notebook
                guides you through each step:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>Load MNIST</strong> with torchvision&mdash;apply transforms,
                  create DataLoaders for train and test sets
                </li>
                <li>
                  <strong>Build the simple model</strong>&mdash;MNISTClassifier with
                  nn.Flatten, three linear layers, ReLU activations
                </li>
                <li>
                  <strong>Write the training loop</strong>&mdash;cross-entropy loss, Adam
                  optimizer, accuracy tracking per epoch
                </li>
                <li>
                  <strong>Evaluate on the test set</strong>&mdash;torch.no_grad(), model.eval(),
                  compute test accuracy
                </li>
                <li>
                  <strong>Visualize predictions</strong>&mdash;plot correct and incorrect
                  predictions with confidence scores
                </li>
                <li>
                  <strong>Build the improved model</strong>&mdash;add BatchNorm, Dropout,
                  and weight decay; compare training curves
                </li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-2-2-mnist-project.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Steps 1&ndash;4 are guided with starter code. Steps 5&ndash;6 are supported
                with templates. The complete code from this lesson is available in the
                notebook&rsquo;s solutions section.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Start Simple">
            Get the simple model working first (steps 1&ndash;4). See your first 97%.
            Then improve it (steps 5&ndash;6). Building incrementally is how real ML
            projects work.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 14. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Classification uses cross-entropy, not MSE',
                description:
                  'MSE treats all errors equally. Cross-entropy punishes confident wrong answers severely — exactly what classification needs. It is the wrongness score for categorical outputs.',
              },
              {
                headline: 'Softmax converts logits to probabilities',
                description:
                  'Raw model outputs (logits) are arbitrary numbers. Softmax maps them to positive values that sum to 1. nn.CrossEntropyLoss applies it internally — do not add it yourself.',
              },
              {
                headline: 'Accuracy = human metric, loss = model metric',
                description:
                  'The model optimizes cross-entropy loss. You evaluate with accuracy (% correct). They correlate but are not identical. Always track both.',
              },
              {
                headline: 'model.train() / model.eval() matters with regularization',
                description:
                  'Dropout and batch norm behave differently in training vs evaluation. Forgetting model.eval() before testing gives unreliable results.',
              },
              {
                headline: 'The training loop heartbeat did not change',
                description:
                  'Forward, loss, backward, update — the same four lines. Cross-entropy replaces MSE, DataLoader replaces raw tensors, but the pattern is identical.',
              },
              {
                headline: '97–98% is a great result for this architecture',
                description:
                  'A fully-connected network on MNIST tops out around 98%. Getting 99%+ requires convolutional layers — the focus of Series 3.',
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
            subtitle="From building to debugging"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have a working model that reads handwriting. You built it, trained it,
              evaluated it, and improved it with regularization. But what happens when
              things go wrong? Shapes do not match. Loss explodes. The model refuses to
              learn. Next lesson: debugging tools and strategies for when PyTorch training
              goes sideways.
            </p>
          </div>
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
