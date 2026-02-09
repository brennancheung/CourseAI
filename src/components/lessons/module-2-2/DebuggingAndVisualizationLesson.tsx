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
  PhaseCard,
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

/**
 * Debugging and Visualization -- Lesson 3 of Module 2.2 (Real Data)
 *
 * CONSOLIDATE lesson: the student's first systematic debugging toolkit.
 * Teaches three diagnostic instruments (torchinfo, gradient magnitude
 * checking, TensorBoard) applied to familiar models from previous lessons.
 *
 * New tools (3, no new ML theory):
 *   1. torchinfo -- model architecture inspection (DEVELOPED)
 *   2. Gradient magnitude checking -- per-layer gradient health (DEVELOPED)
 *   3. TensorBoard -- live training monitoring (INTRODUCED)
 *
 * Central insight: Debugging is not random guessing -- it is a systematic
 * workflow using specific instruments for specific failure modes.
 */

// Data for the TensorBoard mock-up: 3 learning rate comparison
const lrComparisonData = [
  { epoch: 1, slow: 2.28, good: 2.10, fast: 2.35 },
  { epoch: 2, slow: 2.18, good: 1.42, fast: 2.51 },
  { epoch: 3, slow: 2.05, good: 0.78, fast: 1.89 },
  { epoch: 4, slow: 1.90, good: 0.45, fast: 2.72 },
  { epoch: 5, slow: 1.74, good: 0.31, fast: 1.95 },
  { epoch: 6, slow: 1.58, good: 0.22, fast: 2.88 },
  { epoch: 7, slow: 1.43, good: 0.17, fast: 2.14 },
  { epoch: 8, slow: 1.30, good: 0.14, fast: 2.96 },
  { epoch: 9, slow: 1.18, good: 0.12, fast: 1.80 },
  { epoch: 10, slow: 1.08, good: 0.10, fast: 2.67 },
]

function TensorBoardMockup() {
  return (
    <div className="rounded-lg border bg-[#1a1a2e] p-4 space-y-3">
      {/* Mock TensorBoard header */}
      <div className="flex items-center gap-3 border-b border-white/10 pb-2">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-orange-500/80" />
          <span className="text-xs font-mono text-white/70">TensorBoard</span>
        </div>
        <div className="flex gap-3 ml-auto">
          <span className="text-[10px] font-mono text-white/40 px-2 py-0.5 rounded bg-white/5 border border-white/10">
            SCALARS
          </span>
          <span className="text-[10px] font-mono text-white/20 px-2 py-0.5">
            IMAGES
          </span>
          <span className="text-[10px] font-mono text-white/20 px-2 py-0.5">
            GRAPHS
          </span>
        </div>
      </div>

      {/* Chart area */}
      <div className="space-y-1">
        <p className="text-[10px] font-mono text-white/50 ml-1">Loss/train</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={lrComparisonData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
            <XAxis
              dataKey="epoch"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
              label={{ value: 'epoch', position: 'insideBottom', offset: -2, fontSize: 10, fill: 'rgba(255,255,255,0.3)' }}
            />
            <YAxis
              stroke="rgba(255,255,255,0.3)"
              tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
              domain={[0, 3.2]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e1e3a',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '6px',
                fontSize: 11,
              }}
              labelStyle={{ color: 'rgba(255,255,255,0.6)' }}
            />
            <Legend
              wrapperStyle={{ fontSize: 10, color: 'rgba(255,255,255,0.6)' }}
            />
            <Line
              type="monotone"
              dataKey="slow"
              name="lr=0.001"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="good"
              name="lr=0.01"
              stroke="#34d399"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="fast"
              name="lr=0.1"
              stroke="#fb7185"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Mock run selector */}
      <div className="flex items-center gap-3 border-t border-white/10 pt-2">
        <span className="text-[10px] font-mono text-white/40">Runs:</span>
        <div className="flex gap-2">
          <span className="text-[10px] font-mono text-cyan-400 flex items-center gap-1">
            <span className="inline-block w-2 h-0.5 bg-cyan-400 rounded" />
            mnist_lr_0.001
          </span>
          <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
            <span className="inline-block w-2 h-0.5 bg-emerald-400 rounded" />
            mnist_lr_0.01
          </span>
          <span className="text-[10px] font-mono text-rose-400 flex items-center gap-1">
            <span className="inline-block w-2 h-0.5 bg-rose-400 rounded" />
            mnist_lr_0.1
          </span>
        </div>
      </div>
    </div>
  )
}

export function DebuggingAndVisualizationLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Debugging and Visualization"
            description="Three diagnostic instruments for when PyTorch training goes wrong -- and a systematic workflow to use them."
            category="Real Data"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Learn to systematically diagnose training failures using torchinfo
            (X-ray your model&rsquo;s shapes before training), gradient magnitude
            checking (take the model&rsquo;s pulse during training), and TensorBoard
            (monitor vital signs across training). By the end, you will have a
            debugging checklist that replaces &ldquo;stare at code and
            restart&rdquo; with a repeatable diagnostic workflow.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Tools, Not Theory">
            This is a CONSOLIDATE lesson. No new ML concepts&mdash;only new
            tools applied to concepts you already understand. The cognitive
            demand is &ldquo;learn a new API,&rdquo; not &ldquo;understand a
            new idea.&rdquo;
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'Three tools only -- torchinfo, gradient checking, TensorBoard basics',
              'No advanced TensorBoard features (histograms, embeddings, graph visualization, profiler)',
              'No Weights & Biases, MLflow, or other experiment tracking platforms',
              'No PyTorch Profiler or performance optimization',
              'No hyperparameter tuning or search strategies',
              'No debugging CUDA/GPU-specific errors',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook -- "The Silent Failure" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Silent Failure"
            subtitle="When loss goes down but nothing is actually learning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You just trained a model on MNIST. Here is a training run. Look
              at the loss curve:
            </p>

            <div className="rounded-lg border bg-muted/30 p-5 font-mono text-sm space-y-1">
              <p className="text-muted-foreground">Epoch 1/10: loss=2.302</p>
              <p className="text-muted-foreground">Epoch 2/10: loss=2.118</p>
              <p className="text-muted-foreground">Epoch 3/10: loss=1.847</p>
              <p className="text-muted-foreground">Epoch 4/10: loss=1.523</p>
              <p className="text-muted-foreground">Epoch 5/10: loss=1.201</p>
              <p className="text-muted-foreground">Epoch 6/10: loss=0.943</p>
              <p className="text-muted-foreground">Epoch 7/10: loss=0.782</p>
              <p className="text-muted-foreground">Epoch 8/10: loss=0.651</p>
              <p className="text-muted-foreground">Epoch 9/10: loss=0.574</p>
              <p className="text-muted-foreground">Epoch 10/10: loss=0.512</p>
            </div>

            <p className="text-muted-foreground">
              Loss dropped from 2.3 to 0.5 over 10 epochs. Smooth decrease,
              no NaN, no divergence. Is this model training correctly?
            </p>

            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Reveal the truth
              </summary>
              <div className="mt-3 border-t border-primary/20 pt-3 space-y-3">
                <div className="rounded-lg border border-rose-500/30 bg-rose-500/5 p-4">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-rose-400">Accuracy: 10%</strong>&mdash;random
                    chance on 10 classes. The model learned to always predict the
                    most common class. Loss went down because predicting the class
                    prior is a valid (terrible) solution to cross-entropy. The model
                    outputs nearly identical logits for every input.
                  </p>
                </div>
                <p className="text-muted-foreground text-sm">
                  This is the core problem: <strong>your current diagnostic
                  tool&mdash;watching loss decrease&mdash;is insufficient.</strong>{' '}
                  Loss going down does not mean the model is learning useful
                  features. You need better instruments.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* Hook aside -- placed AFTER the reveal so it does not spoil the tension */}
      <Row>
        <Row.Content>
          <WarningBlock title="Loss Is Not Enough">
            A decreasing loss curve feels reassuring. But loss can decrease
            while the model learns nothing useful. Always monitor accuracy
            (or another task-specific metric) alongside loss.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* 3. Tool 1: torchinfo -- X-Ray Your Model */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tool 1: torchinfo"
            subtitle="X-ray your model's architecture before training"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the nn.Module lesson, you manually counted parameters&mdash;multiplying
              weight dimensions, adding biases. That works for a 3-layer network.
              What about a 50-layer model? Or one where you are not sure if
              nn.Flatten is in the right place?
            </p>

            <p className="text-muted-foreground">
              <strong>torchinfo</strong> is an X-ray machine for your model.
              Pass in your model and an example input size, and it shows you
              every layer&rsquo;s output shape, parameter count, and the total
              number of trainable parameters&mdash;all without running real data
              through the model.
            </p>

            <CodeBlock
              language="python"
              filename="torchinfo_basic.py"
              code={`# Install: pip install torchinfo
from torchinfo import summary

# Your MNIST model from last lesson
model = MNISTClassifier()

# Pass the model and input shape (batch_size, channels, height, width)
summary(model, input_size=(1, 1, 28, 28))`}
            />

            <p className="text-muted-foreground">
              The output looks something like this:
            </p>

            <div className="rounded-lg border bg-muted/30 p-4 font-mono text-xs space-y-0.5 overflow-x-auto">
              <p className="text-muted-foreground">==========================================================================================</p>
              <p className="text-muted-foreground">Layer (type)                    Output Shape         Param #</p>
              <p className="text-muted-foreground">==========================================================================================</p>
              <p className="text-muted-foreground">Flatten                         [1, 784]             0</p>
              <p className="text-muted-foreground">Linear-1                        [1, 256]             200,960</p>
              <p className="text-muted-foreground">ReLU-1                          [1, 256]             0</p>
              <p className="text-muted-foreground">Linear-2                        [1, 128]             32,896</p>
              <p className="text-muted-foreground">ReLU-2                          [1, 128]             0</p>
              <p className="text-muted-foreground">Linear-3                        [1, 10]              1,290</p>
              <p className="text-muted-foreground">==========================================================================================</p>
              <p className="text-muted-foreground">Total params: 235,146</p>
              <p className="text-muted-foreground">Trainable params: 235,146</p>
            </div>

            <p className="text-muted-foreground">
              Every layer, every output shape, every parameter count&mdash;at a
              glance. Now here is why this matters beyond convenience.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The X-Ray Analogy">
            torchinfo shows the inside of your model without running data
            through it. Like a medical X-ray&mdash;you see the structure
            before you operate.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* torchinfo: Catching Shape Errors */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Example: Missing Flatten</strong>&mdash;here is a broken
              version of the MNIST model. Can you spot the bug?
            </p>

            <CodeBlock
              language="python"
              filename="broken_model.py"
              code={`class BrokenMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # Oops -- forgot nn.Flatten()!
        self.layer1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        # x is [B, 1, 28, 28] but layer1 expects [B, 784]
        x = self.relu1(self.layer1(x))  # RuntimeError!
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x`}
            />

            <p className="text-muted-foreground">
              Without torchinfo, you would get a cryptic RuntimeError about
              shape mismatches when you run the first batch. With torchinfo:
            </p>

            <CodeBlock
              language="python"
              filename="torchinfo_catch.py"
              code={`summary(BrokenMNIST(), input_size=(1, 1, 28, 28))
# ERROR: mat1 and mat2 shapes cannot be multiplied (1x28 and 784x256)
# torchinfo traces shapes through each layer and reveals
# exactly where the mismatch occurs -- BEFORE you load any data.`}
            />

            <p className="text-muted-foreground">
              The shape trace shows [1, 1, 28, 28] going into Linear(784, 256).
              The input has 28 features (last dimension of the 4D tensor), but
              the layer expects 784. Fix: add{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Flatten()</code>{' '}
              before the first linear layer. One line, not a fundamental
              redesign.
            </p>

            <p className="text-muted-foreground">
              <strong>Example: Wrong Linear Dimensions</strong>&mdash;a different
              kind of shape bug:
            </p>

            <CodeBlock
              language="python"
              filename="dimension_mismatch.py"
              code={`class MismatchedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 10)    # Bug: expects 64 inputs, gets 128
        # Should be nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.layer2(x)
        return x`}
            />

            <p className="text-muted-foreground">
              torchinfo catches this too&mdash;the output shape column shows
              [1, 128] after layer1, but layer2 expects 64 inputs. The mismatch
              is visible in the summary before you ever call{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model(x)</code>.
              Shape errors are <strong>plumbing problems</strong>, not
              fundamental design flaws. torchinfo helps you find the leaky pipe.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Plumbing, Not Design">
            Shape errors feel catastrophic&mdash;big red tracebacks with
            incomprehensible dimensions. But they are almost always a
            single mismatched number between adjacent layers. torchinfo
            makes the mismatch visible.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 4. Check 1 -- Predict-and-Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check 1: Predict-and-Verify</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                Here is a model definition. Before running torchinfo, predict:
              </p>

              <CodeBlock
                language="python"
                filename="predict_shapes.py"
                code={`class Mystery(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x`}
              />

              <p className="text-muted-foreground text-sm">
                <strong>1.</strong> What is the output shape after each layer
                (given input [1, 1, 28, 28])?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2 text-muted-foreground space-y-1">
                  <p>Flatten: [1, 784]</p>
                  <p>Linear-1 + ReLU: [1, 512]</p>
                  <p>Linear-2 + ReLU: [1, 256]</p>
                  <p>Linear-3: [1, 10]</p>
                </div>
              </details>

              <p className="text-muted-foreground text-sm">
                <strong>2.</strong> How many total parameters?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2 text-muted-foreground space-y-1">
                  <p>Layer 1: 784 x 512 + 512 = 401,920</p>
                  <p>Layer 2: 512 x 256 + 256 = 131,328</p>
                  <p>Layer 3: 256 x 10 + 10 = 2,570</p>
                  <p><strong>Total: 535,818</strong></p>
                </div>
              </details>

              <p className="text-muted-foreground text-sm">
                <strong>3.</strong> If you changed{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">nn.Linear(512, 256)</code> to{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">nn.Linear(128, 256)</code>,
                where would the shape error occur?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                  <p className="text-muted-foreground">
                    At layer2. The output of layer1 is [1, 512], but
                    nn.Linear(128, 256) expects 128 inputs. The mismatch
                    between 512 and 128 causes a RuntimeError.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 5. Tool 2: Gradient Magnitude Checking */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tool 2: Gradient Magnitude Checking"
            subtitle="Take the model's pulse during training"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Backpropagation, you learned about vanishing gradients&mdash;the
              telephone game where signals decay as they pass through layers.
              You learned the symptom: &ldquo;flatline = vanishing, NaN =
              exploding.&rdquo; But how do you actually <strong>check</strong> for
              this in your own model?
            </p>

            <p className="text-muted-foreground">
              The answer: iterate over{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.named_parameters()</code>&mdash;the
              same API you used to construct optimizers in nn.Module&mdash;and
              read the{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.grad</code>{' '}
              attribute you learned about in Autograd. Compute the norm
              (magnitude) of each layer&rsquo;s gradients after one backward pass:
            </p>

            <CodeBlock
              language="python"
              filename="gradient_check.py"
              code={`def log_gradient_norms(model):
    """Print gradient magnitude for each layer's parameters."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name:30s}  grad_norm: {grad_norm:.6f}")

# Use after a backward pass:
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()

print("Gradient magnitudes:")
log_gradient_norms(model)`}
            />

            <p className="text-muted-foreground">
              Here is what healthy gradients look like versus unhealthy ones:
            </p>

            <ComparisonRow
              left={{
                title: 'Healthy: Balanced Gradients',
                color: 'emerald',
                items: [
                  'layer1.weight  grad_norm: 0.0342',
                  'layer1.bias    grad_norm: 0.0198',
                  'layer2.weight  grad_norm: 0.0287',
                  'layer2.bias    grad_norm: 0.0153',
                  'layer3.weight  grad_norm: 0.0411',
                  'layer3.bias    grad_norm: 0.0226',
                ],
              }}
              right={{
                title: 'Unhealthy: Vanishing Early Layers',
                color: 'rose',
                items: [
                  'layer1.weight  grad_norm: 0.0000003',
                  'layer1.bias    grad_norm: 0.0000001',
                  'layer2.weight  grad_norm: 0.0002',
                  'layer2.bias    grad_norm: 0.0001',
                  'layer3.weight  grad_norm: 0.0891',
                  'layer3.bias    grad_norm: 0.0413',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Both models have loss decreasing. Neither shows NaN. But look at
              the left column of the unhealthy model: layer1&rsquo;s gradient
              norms are <strong>100,000 times smaller</strong> than layer3&rsquo;s.
              The early layers are barely learning&mdash;their gradients are
              so tiny that{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">optimizer.step()</code>{' '}
              makes negligible updates.
            </p>

            <p className="text-muted-foreground">
              Remember the telephone game from Backpropagation? This is what
              it looks like in practice. The signal from the loss gets weaker
              at each layer as you go backward. Gradient checking catches
              this <strong>before</strong> you see a flatline in your training
              curve&mdash;it is early detection, not just symptom recognition.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Early Detection">
            You already know &ldquo;flatline = vanishing.&rdquo; Gradient
            checking catches the problem before the flatline appears.
            It is a diagnostic that tells you the model is unhealthy
            while it still looks OK from the outside.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Gradient checking: the practical helper */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In practice, you do not want to print gradient norms every
              iteration&mdash;that floods your terminal. Call the helper every
              N iterations:
            </p>

            <CodeBlock
              language="python"
              filename="gradient_check_in_loop.py"
              code={`for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients every 100 batches
        if batch_idx % 100 == 0:
            print(f"\\nEpoch {epoch}, Batch {batch_idx}:")
            log_gradient_norms(model)`}
            />

            <p className="text-muted-foreground">
              <strong>What to look for:</strong>
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>All norms near zero (&lt;1e-7):</strong> vanishing
                gradients. Early layers are not learning.
              </li>
              <li>
                <strong>Any norms very large (&gt;100) or NaN:</strong> exploding
                gradients. Training is about to diverge.
              </li>
              <li>
                <strong>Early layers much smaller than later layers:</strong> gradient
                decay. The model learns unevenly&mdash;later layers adapt
                while early layers stagnate.
              </li>
              <li>
                <strong>All norms roughly similar magnitude (within 10-100x):</strong> healthy.
                All layers are receiving useful gradient signal.
              </li>
            </ul>

            <p className="text-muted-foreground">
              <strong>What to do about it:</strong>
            </p>

            <div className="rounded-lg border bg-muted/20 p-4 text-sm">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-1">
                  <p className="font-semibold text-rose-400">Vanishing gradients?</p>
                  <ul className="list-disc list-inside text-muted-foreground space-y-0.5 ml-2">
                    <li>Use ReLU instead of sigmoid/tanh</li>
                    <li>Try a different weight initialization</li>
                    <li>Add batch normalization between layers</li>
                    <li>Add skip connections (a future topic)</li>
                  </ul>
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-amber-400">Exploding gradients?</p>
                  <ul className="list-disc list-inside text-muted-foreground space-y-0.5 ml-2">
                    <li>Lower the learning rate</li>
                    <li>Add gradient clipping (a future topic)</li>
                    <li>Check for NaN in your input data</li>
                    <li>Add batch normalization between layers</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Pulse Analogy">
            Gradient magnitude checking is taking the model&rsquo;s
            pulse. A healthy pulse has consistent rhythm (balanced
            magnitudes across layers). A weak pulse (tiny early-layer
            gradients) means the model is struggling even if it looks
            alive.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 7. Tool 3: TensorBoard -- Flight Recorder */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tool 3: TensorBoard"
            subtitle="Monitor your model's vital signs in real time"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In MNIST Project, you plotted training curves with matplotlib.
              That works for one experiment. But what if you want to compare
              three different learning rates? You would need to: run training
              three times, save the metrics from each run, write plotting
              code, and run it. For every new comparison, repeat the cycle.
            </p>

            <p className="text-muted-foreground">
              <strong>TensorBoard</strong> is a flight recorder for your
              training runs. You add two lines of logging code inside the
              training loop, and TensorBoard gives you a live dashboard with
              curves, zoom, and overlay comparison&mdash;across as many runs
              as you want. No custom plotting code needed.
            </p>

            <CodeBlock
              language="python"
              filename="tensorboard_setup.py"
              code={`from torch.utils.tensorboard import SummaryWriter

# Create a writer -- logs go to runs/<experiment_name>
writer = SummaryWriter('runs/mnist_lr_0.001')

for epoch in range(num_epochs):
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

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    # --- Two logging lines. That's it. ---
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")

writer.close()`}
            />

            <p className="text-muted-foreground">
              Notice what happened to the training loop: <strong>the heartbeat
              did not change.</strong> Forward, loss, backward, update&mdash;still
              there, untouched. The logging lines sit outside the inner loop,
              recording epoch-level metrics. This is the &ldquo;same heartbeat,
              new instruments&rdquo; pattern from The Training Loop.
            </p>

            <CodeBlock
              language="bash"
              filename="terminal"
              code={`# Launch TensorBoard (in a separate terminal)
tensorboard --logdir=runs

# Open http://localhost:6006 in your browser`}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Flight Recorder">
            A flight recorder captures data continuously&mdash;you review
            it when something goes wrong. TensorBoard works the same way:
            log metrics during training, inspect them whenever you need to.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* TensorBoard: The Killer Feature -- Run Comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              TensorBoard&rsquo;s killer feature is <strong>run comparison</strong>.
              Train the same model with three different learning rates, each
              logging to a different directory:
            </p>

            <CodeBlock
              language="python"
              filename="tensorboard_comparison.py"
              code={`learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    # Each run gets its own log directory
    writer = SummaryWriter(f'runs/mnist_lr_{lr}')
    model = MNISTClassifier()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        # ... training loop ...
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)

    writer.close()`}
            />

            <p className="text-muted-foreground">
              Open TensorBoard and all three runs appear on the same plot.
              Here is what the dashboard looks like&mdash;three learning
              rates, overlaid on a single chart:
            </p>

            <TensorBoardMockup />

            <p className="text-muted-foreground">
              You can see the behavior you learned in Learning Rate&mdash;now
              visible in your own training dashboard:
            </p>

            <div className="grid gap-3 md:grid-cols-3">
              <GradientCard title="lr = 0.001" color="cyan">
                <p className="text-sm text-muted-foreground">
                  Slow, steady descent. Loss decreases smoothly but takes
                  many epochs to converge. <strong>Too cautious.</strong>
                </p>
              </GradientCard>
              <GradientCard title="lr = 0.01" color="emerald">
                <p className="text-sm text-muted-foreground">
                  Smooth, fast convergence. Loss drops quickly and
                  stabilizes. <strong>Just right.</strong>
                </p>
              </GradientCard>
              <GradientCard title="lr = 0.1" color="rose">
                <p className="text-sm text-muted-foreground">
                  Oscillating, unstable. Loss bounces around or diverges.{' '}
                  <strong>Too aggressive.</strong>
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              This comparison would take 30+ lines of matplotlib code&mdash;collect
              data from three runs, store in lists, write plotting code with
              labels and legends. TensorBoard does it with 2 lines of logging
              per run and zero plotting code.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="matplotlib vs TensorBoard">
            matplotlib: plot after training, one experiment at a time,
            custom code for every visualization. TensorBoard: live during
            training, automatic run comparison, zero plotting code.
            Use matplotlib for publication figures. Use TensorBoard for
            debugging.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* TensorBoard: Negative Example -- Suspiciously Smooth */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>A subtle bug TensorBoard catches:</strong> training on the
              same batch every epoch. This happens when you accidentally
              iterate a single batch instead of the full DataLoader:
            </p>

            <CodeBlock
              language="python"
              filename="same_batch_bug.py"
              code={`# BUG: this only trains on one batch per epoch!
images, labels = next(iter(train_loader))

for epoch in range(num_epochs):
    outputs = model(images)       # Same batch every time
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), epoch)
# The loss curve looks suspiciously smooth...`}
            />

            <p className="text-muted-foreground">
              In TensorBoard, this bug has a telltale signature: the loss curve
              is <strong>suspiciously smooth</strong>. Real training with
              mini-batches has noise in the loss&mdash;each batch is slightly
              different, so the loss jiggles. If your loss curve is a perfect
              smooth line, that is the bug signal&mdash;you are probably
              training on the same data every iteration.
            </p>

            <ComparisonRow
              left={{
                title: 'Real Training (Healthy)',
                color: 'emerald',
                items: [
                  'Loss has natural noise from different mini-batches',
                  'General downward trend with small fluctuations',
                  'Each epoch iterates the full dataset',
                  'Noise is the signal that batching is working',
                ],
              }}
              right={{
                title: 'Same-Batch Bug',
                color: 'rose',
                items: [
                  'Loss is suspiciously smooth',
                  'Perfect monotonic decrease, no noise at all',
                  'Only one batch is being used per epoch',
                  'Model memorizes one batch, fails on everything else',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Too Smooth = Suspicious">
            Perfect training curves are a red flag. Real training on
            mini-batches always has some noise. If your loss looks like
            a perfectly smooth line, check that you are actually iterating
            the DataLoader, not reusing one batch.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Check 2 -- Transfer Question (moved after TensorBoard so all 3 tools are available) */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check 2: Transfer Question</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                Your colleague&rsquo;s model has loss decreasing but accuracy
                stuck at 52% on a <strong>binary classification task</strong>.
                They say: &ldquo;The model is learning, just slowly.&rdquo;
              </p>

              <p className="text-muted-foreground text-sm">
                What would you check first, and what tool would you use?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      52% on binary classification is barely above random
                      chance (50%). Three things to check:
                    </p>
                    <ol className="list-decimal list-inside space-y-1 ml-2">
                      <li>
                        <strong>Per-class accuracy:</strong> Is the model
                        predicting all one class? If class A is 52% of the
                        data, the model could be outputting &ldquo;A&rdquo;
                        for everything and getting 52% accuracy &ldquo;for
                        free.&rdquo;
                      </li>
                      <li>
                        <strong>Gradient magnitudes:</strong> Use{' '}
                        <code className="text-xs bg-muted px-1 py-0.5 rounded">log_gradient_norms()</code>{' '}
                        to check if early layers have vanishing gradients.
                        If they do, the model is only training its last
                        layer.
                      </li>
                      <li>
                        <strong>Monitor both metrics over time:</strong> Use
                        TensorBoard to watch loss AND accuracy together. Loss
                        going down while accuracy stays flat is the
                        &ldquo;silent failure&rdquo; from the hook.
                      </li>
                    </ol>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 8. The Debugging Checklist */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Debugging Checklist"
            subtitle="A systematic workflow, not random guessing"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now have three instruments. Here is how they fit together
              into a systematic debugging workflow:
            </p>

            <PhaseCard number={1} title="Before Training" subtitle="X-ray the model" color="cyan">
              <div className="space-y-1 text-sm text-muted-foreground">
                <p>
                  Run{' '}
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">torchinfo.summary(model, input_size=...)</code>.
                  Verify:
                </p>
                <ul className="list-disc list-inside ml-2 space-y-0.5">
                  <li>Output shapes match between adjacent layers</li>
                  <li>Total parameter count is reasonable</li>
                  <li>Input shape propagates correctly through Flatten</li>
                </ul>
              </div>
            </PhaseCard>

            <PhaseCard number={2} title="First Iteration" subtitle="Take the pulse" color="blue">
              <div className="space-y-1 text-sm text-muted-foreground">
                <p>
                  After the first backward pass, run{' '}
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">log_gradient_norms(model)</code>.
                  Check:
                </p>
                <ul className="list-disc list-inside ml-2 space-y-0.5">
                  <li>Are all layers receiving gradients? (no None values)</li>
                  <li>Are gradient magnitudes balanced across layers?</li>
                  <li>Are early-layer gradients in a reasonable range? (not &lt;1e-7)</li>
                </ul>
              </div>
            </PhaseCard>

            <PhaseCard number={3} title="During Training" subtitle="Monitor the flight recorder" color="emerald">
              <div className="space-y-1 text-sm text-muted-foreground">
                <p>
                  Watch TensorBoard. Log <strong>both</strong> loss and accuracy,{' '}
                  <strong>both</strong> train and test:
                </p>
                <ul className="list-disc list-inside ml-2 space-y-0.5">
                  <li>Loss going down + accuracy going up = healthy</li>
                  <li>Loss going down + accuracy flat = silent failure (hook example)</li>
                  <li>Train accuracy rising + test accuracy flat = scissors pattern (overfitting)</li>
                </ul>
              </div>
            </PhaseCard>

            <PhaseCard number={4} title="If Something Is Wrong" subtitle="Diagnose by symptom" color="orange">
              <div className="space-y-1 text-sm text-muted-foreground">
                <ul className="list-disc list-inside space-y-1">
                  <li>
                    <strong>Loss stuck:</strong> check gradients (vanishing?),
                    check learning rate (too small?)
                  </li>
                  <li>
                    <strong>Loss NaN:</strong> check gradients (exploding?),
                    check data (NaN in inputs?), check learning rate (too large?)
                  </li>
                  <li>
                    <strong>Train good, test bad:</strong> scissors pattern&mdash;add
                    regularization (dropout, weight decay)
                  </li>
                  <li>
                    <strong>Loss down, accuracy flat:</strong> check predictions&mdash;is
                    the model outputting the same class for everything?
                  </li>
                </ul>
              </div>
            </PhaseCard>

            <p className="text-muted-foreground">
              This is a <strong>workflow</strong>, not a collection of tricks.
              Each step uses a specific instrument for a specific purpose.
              When something goes wrong, you do not stare at code hoping for
              insight&mdash;you follow the checklist.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Heartbeat, New Instruments">
            The training loop does not change. You are adding diagnostic
            instruments around it&mdash;torchinfo before, gradient checking at
            the start, TensorBoard throughout. The heartbeat stays the same.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Solving the Opening Puzzle -- revisit the hook with all three tools */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="font-semibold text-sm">
              Solving the Opening Puzzle
            </p>
            <p className="text-muted-foreground">
              Remember the silent failure from the opening? Loss dropped from
              2.3 to 0.5 but accuracy was stuck at 10%. Here is exactly how
              the debugging checklist would have caught it:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
              <li>
                <strong>torchinfo</strong> (before training): passes&mdash;shapes
                are fine. The architecture is correct; the bug is not structural.
              </li>
              <li>
                <strong>Gradient checking</strong> (first iteration): might
                reveal that gradients are flowing only to the last layer, or
                that all gradients point toward the same output class. Either
                would be a warning sign.
              </li>
              <li>
                <strong>TensorBoard</strong> (during training): with both loss
                AND accuracy logged, the dashboard immediately shows loss
                decreasing while accuracy stays at 10%. The bug is visible in
                the first epoch, not discovered after 10.
              </li>
            </ol>
            <p className="text-muted-foreground">
              The checklist does not just find bugs&mdash;it finds them{' '}
              <strong>early</strong>, before you waste time training a broken
              model to completion.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 9. Practice: Colab Notebook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="Practice the debugging workflow hands-on"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The Colab notebook walks you through each tool on your MNIST
                model from last lesson:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>(Guided)</strong> Run torchinfo on your MNIST model.
                  Verify the parameter count matches your manual calculation
                  from MNIST Project.
                </li>
                <li>
                  <strong>(Guided)</strong> Introduce a shape bug (remove
                  Flatten), run torchinfo, identify the problem, fix it.
                </li>
                <li>
                  <strong>(Supported)</strong> Write a{' '}
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">log_gradient_norms()</code>{' '}
                  function. Run it on a healthy model and a poorly-initialized
                  model. Compare magnitudes per layer.
                </li>
                <li>
                  <strong>(Supported)</strong> Add TensorBoard logging to your
                  MNIST training loop. Train for 10 epochs. Open TensorBoard
                  and examine the curves.
                </li>
                <li>
                  <strong>(Supported)</strong> Train 3 runs with different
                  learning rates (0.001, 0.01, 0.1). Compare in TensorBoard.
                  Identify which is too high, too low, and just right.
                </li>
                <li>
                  <strong>(Independent)</strong> Given a &ldquo;broken&rdquo;
                  training script with 3 intentional bugs (a shape error,
                  missing model.eval(), and a subtle data loading bug), use the
                  debugging checklist to find and fix all three.
                </li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-2-3-debugging-and-visualization.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Exercises 1&ndash;2 are guided with starter code. Exercises
                3&ndash;5 are supported with templates. Exercise 6 is
                independent&mdash;you use the debugging checklist to find all
                three bugs yourself.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Scaffolding">
            The exercises go from guided (torchinfo on a known model) to
            independent (find 3 bugs in a broken script). If you get stuck
            on the independent exercise, go back and re-read the debugging
            checklist section.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'torchinfo: X-ray your model before training',
                description:
                  'Run torchinfo.summary() to see every layer\'s output shape, parameter count, and total parameters. Catches shape mismatches before you run any data.',
              },
              {
                headline: 'Gradient checking: take the model\'s pulse during training',
                description:
                  'Iterate model.named_parameters() and check .grad.norm() after backward(). Catches vanishing and exploding gradients before they cause visible symptoms.',
              },
              {
                headline: 'TensorBoard: monitor vital signs across training',
                description:
                  'Add two lines of logging to your training loop. Get live curves, run comparison, and immediate visibility into silent failures -- no custom plotting code.',
              },
              {
                headline: 'The debugging checklist replaces guessing',
                description:
                  'Before training: torchinfo. First iteration: gradient check. During training: TensorBoard. If something breaks: diagnose by symptom. A systematic workflow, not random guessing.',
              },
              {
                headline: 'Loss going down does not mean training is working',
                description:
                  'Always monitor accuracy alongside loss. A model can minimize loss by learning the class prior without extracting useful features. This is the most important lesson today.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 11. Module 2.2 Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="2.2"
            title="Real Data"
            achievements={[
              'Dataset and DataLoader for loading, batching, and shuffling real data',
              'End-to-end MNIST classifier with cross-entropy, softmax, and accuracy',
              'Regularization in practice: dropout, batch norm, weight decay',
              'torchinfo for model architecture inspection',
              'Gradient magnitude checking for training health',
              'TensorBoard for monitoring and comparing training runs',
              'A systematic debugging checklist for when training goes wrong',
            ]}
            nextModule="2.3"
            nextTitle="Practical Patterns"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
