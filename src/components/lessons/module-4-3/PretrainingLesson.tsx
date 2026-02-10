'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  ConstraintBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  SummaryBlock,
  GradientCard,
  ComparisonRow,
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Pretraining — The Training Loop
 *
 * Second lesson in Module 4.3 (Building & Training GPT).
 * Eleventh lesson in Series 4 (LLMs & Transformers).
 *
 * Takes the GPT model built in Lesson 1 and trains it on real text.
 * This is a STRETCH lesson: three new concepts (text dataset
 * preparation, LR scheduling, gradient clipping) layered onto
 * the deeply familiar training loop pattern.
 *
 * Core concepts at APPLIED:
 * - Complete GPT training loop (forward, loss, backward, step)
 *
 * Core concepts at DEVELOPED:
 * - Text dataset preparation (input/target offset, context windows)
 * - Learning rate scheduling (warmup + cosine decay)
 * - Cross-entropy for next-token prediction over 50K vocab
 * - Loss curve interpretation for language models
 *
 * Core concepts at INTRODUCED:
 * - Gradient clipping in practice
 * - AdamW optimizer
 *
 * EXPLICITLY NOT COVERED:
 * - GPU optimization, mixed precision, flash attention (Lesson 3)
 * - Multi-GPU or distributed training
 * - Different tokenization strategies (Module 4.1 covered this)
 * - Hyperparameter search
 * - Fine-tuning or transfer learning (Module 4.4)
 *
 * Previous: Building nanoGPT (Module 4.3, Lesson 1)
 * Next: Scaling & Efficiency (Module 4.3, Lesson 3)
 */

export function PretrainingLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Pretraining on Real Text"
            description="Train your GPT model from scratch&mdash;watch gibberish transform into recognizable English using the same training loop you already know."
            category="Training"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Train the GPT model from Building nanoGPT on real text. Prepare
            a text dataset for language modeling, build the training loop with
            learning rate scheduling and gradient clipping, and watch generated
            text quality improve as loss decreases.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Loop, New Scale">
            The training loop is the same one from Series 2. You have written
            it many times. Three things are new: how you prepare text data,
            learning rate scheduling, and gradient clipping.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'One model (GPT-2 small), one dataset (TinyShakespeare), one training run',
              'Prepare a text dataset for language model training',
              'Build the complete training loop with LR scheduling and gradient clipping',
              'Cross-entropy loss for next-token prediction across all positions',
              'Interpret loss curves and generated text quality',
              'NOT: GPU optimization, mixed precision, flash attention\u2014that\u2019s Lesson 3',
              'NOT: hyperparameter search\u2014use known-good values from nanoGPT',
              'NOT: evaluation beyond loss and qualitative text inspection',
              'NOT: fine-tuning or transfer learning\u2014Module 4.4',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook: Side-by-Side Training Loop Comparison
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Same Heartbeat"
            subtitle="MNIST vs GPT&mdash;spot the difference"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Training a 124-million-parameter language model on Shakespeare
              must require some exotic new algorithm, right? A fundamentally
              different approach from training a tiny classifier on handwritten
              digits?
            </p>

            <p className="text-muted-foreground">
              Here is the MNIST training loop from Series 2 next to the GPT
              training loop you will write today:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <CodeBlock
                  code={`# MNIST Training Loop (Series 2)
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`}
                  language="python"
                  filename="mnist_loop.py"
                />
              </div>
              <div>
                <CodeBlock
                  code={`# GPT Training Loop (this lesson)
for step in range(max_steps):
    x, y = next(train_iter)  # one batch per step

    # Forward
    logits, loss = model(x, y)

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    update_lr(optimizer, step)  # LR schedule`}
                  language="python"
                  filename="gpt_loop.py"
                />
              </div>
            </div>

            <p className="text-muted-foreground">
              The structure is identical. Forward pass, compute loss, backward
              pass, update weights. The GPT version adds gradient clipping
              (one line) and a learning rate schedule update (one line).
              The outer loop iterates over steps instead of epochs, but the
              core is the same. Same heartbeat, two new instruments.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Loop Is the Loop">
            From linear regression to GPT, the training loop never changes.
            Forward, loss, backward, step. The data format changes, the model
            changes, the loss scale changes. The algorithm does not.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Dataset Preparation
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Preparing the Text Dataset"
            subtitle="How raw text becomes training data"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have a model that takes token IDs and predicts the next
              token. You have raw text (Shakespeare). How do you turn one into
              the other?
            </p>

            <p className="text-muted-foreground">
              <strong>Step 1: Tokenize the entire corpus.</strong> Use the
              GPT-2 BPE tokenizer (from Tokenization) to convert the full
              text into a single long sequence of token IDs.
            </p>

            <CodeBlock
              code={`import tiktoken

enc = tiktoken.get_encoding("gpt2")
text = open("shakespeare.txt").read()

tokens = enc.encode(text)
print(f"Total tokens: {len(tokens):,}")
# Total tokens: ~300,000

data = torch.tensor(tokens, dtype=torch.long)`}
              language="python"
              filename="tokenize.py"
            />

            <p className="text-muted-foreground">
              <strong>Step 2: Slice into context windows.</strong> The model
              has a maximum context length (block_size = 256 for training).
              Slice the long sequence into chunks. Each chunk becomes one
              training example.
            </p>

            <p className="text-muted-foreground">
              <strong>Step 3: The input/target offset.</strong> This is the
              key insight. For a chunk of tokens, the input is
              tokens[i:i+T] and the target is tokens[i+1:i+T+1]&mdash;shifted
              by one position. Every position in the input predicts the token
              that follows it.
            </p>

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                Concrete example: &ldquo;The cat sat on the mat&rdquo;
              </p>
              <div className="text-sm text-muted-foreground font-mono space-y-1">
                <p>Tokens: [&ldquo;The&rdquo;, &ldquo;cat&rdquo;, &ldquo;sat&rdquo;, &ldquo;on&rdquo;, &ldquo;the&rdquo;, &ldquo;mat&rdquo;]</p>
                <p className="mt-2 font-sans font-medium text-foreground">Training examples from ONE sequence:</p>
                <p>Position 0: input &ldquo;The&rdquo; {'\u2192'} predict &ldquo;cat&rdquo;</p>
                <p>Position 1: input &ldquo;The cat&rdquo; {'\u2192'} predict &ldquo;sat&rdquo;</p>
                <p>Position 2: input &ldquo;The cat sat&rdquo; {'\u2192'} predict &ldquo;on&rdquo;</p>
                <p>Position 3: input &ldquo;The cat sat on&rdquo; {'\u2192'} predict &ldquo;the&rdquo;</p>
                <p>Position 4: input &ldquo;The cat sat on the&rdquo; {'\u2192'} predict &ldquo;mat&rdquo;</p>
              </div>
              <p className="text-xs text-muted-foreground/80 font-sans">
                One sequence of length 6 produces 5 training examples.
                The causal mask ensures each position only sees tokens
                before it&mdash;all 5 predictions happen in a single
                forward pass.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Just the Last Token">
            During generation, you use only the last position&rsquo;s
            prediction. During <strong>training</strong>, every position
            predicts its next token simultaneously. A sequence of length T
            produces T training examples in one forward pass. This is why
            training is efficient&mdash;not T separate forward passes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <CodeBlock
              code={`from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx   : idx + self.block_size]      # input
        y = self.data[idx+1 : idx + self.block_size + 1]  # target (shifted by 1)
        return x, y

# Split into train/val
n = int(0.9 * len(data))
train_dataset = TextDataset(data[:n], block_size=256)
val_dataset   = TextDataset(data[n:], block_size=256)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)`}
              language="python"
              filename="dataset.py"
            />

            <p className="text-muted-foreground">
              This is the same Dataset/DataLoader pattern from your CNN work.
              The domain changed (text instead of images) but the abstraction
              is identical. The only new idea is the one-position offset
              between input and target.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Shuffling Text Data">
            You shuffle <strong>chunks</strong>, not individual tokens.
            Each chunk preserves its local context (sequential tokens
            within the window). Shuffling randomizes which chunks appear
            in each batch&mdash;the same principle as shuffling images.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Cross-Entropy for Next-Token Prediction
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Cross-Entropy at Scale"
            subtitle="Same formula, 50,257 classes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know cross-entropy from your CNN work&mdash;it measures how
              well the model&rsquo;s predicted distribution matches the true
              label. For MNIST, that was 10 classes (digits 0&ndash;9). For
              GPT, it is 50,257 classes (the BPE vocabulary). The formula
              does not change.
            </p>

            <p className="text-muted-foreground">
              The one wrinkle: the model outputs logits with
              shape <InlineMath math="(B, T, V)" /> where <InlineMath math="V = 50{,}257" />,
              but <code className="text-xs">nn.CrossEntropyLoss</code> expects{' '}
              <InlineMath math="(N, C)" />. The reshape trick:
            </p>

            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <BlockMath math="\text{logits: } (B, T, V) \rightarrow (B \times T, V)" />
              <BlockMath math="\text{targets: } (B, T) \rightarrow (B \times T)" />
            </div>

            <p className="text-muted-foreground">
              Flatten the batch and sequence dimensions together. Now every
              position across every sequence in the batch is treated as an
              independent classification problem&mdash;which it is.
            </p>

            <CodeBlock
              code={`# Already in the GPT forward method from Lesson 1:
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
    targets.view(-1)                    # (B*T,)
)`}
              language="python"
              filename="loss.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Formula, More Classes">
            Cross-entropy does not care whether it is classifying a shirt
            vs a sneaker (10 classes) or predicting the next word (50,257
            classes). Same formula, same PyTorch call. The vocabulary size
            is just another number.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check: Predict the Initial Loss
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Predict the Initial Loss" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Before running the first forward pass, predict: if the
                untrained model assigns equal probability to all 50,257
                tokens, what should the cross-entropy loss be?
              </p>
              <div className="py-2 px-4 bg-black/20 rounded font-mono text-xs">
                -ln(1/50257) = ln(50257) {'\u2248'} 10.82
              </div>
              <p>
                Run it and check. If the initial loss is close to 10.82, the
                model is correctly initialized (uniform predictions). If it
                is much higher, something is wrong.
              </p>
              <p className="text-muted-foreground/80">
                This is the &ldquo;parameter count as architecture
                verification&rdquo; pattern from Building nanoGPT, applied to
                the training setup. One number confirms the whole pipeline.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Sanity Check Pattern">
            Always verify the initial loss before training. It catches
            bugs in the data pipeline, loss computation, and model setup.
            If the loss starts near <InlineMath math="\ln(V)" />, everything
            is wired correctly.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. AdamW Optimizer
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="AdamW: The Transformer Default"
            subtitle="One line change from Adam"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know Adam thoroughly from Series 2&mdash;momentum plus
              RMSProp plus bias correction. AdamW is the same algorithm with
              one difference: weight decay is applied directly to the
              parameters instead of being added to the loss gradient. In
              practice, the change is a single line:
            </p>

            <CodeBlock
              code={`# What you've used before
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# The transformer default — same API
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=6e-4,           # peak learning rate (we'll schedule this)
    betas=(0.9, 0.95),  # slightly less momentum than default
    weight_decay=0.1,   # decoupled weight decay
)`}
              language="python"
              filename="optimizer.py"
            />

            <p className="text-muted-foreground">
              The practical difference: with standard Adam, weight decay
              interacts with the adaptive learning rates in unexpected ways.
              AdamW decouples them, which produces more consistent
              regularization. For transformers, AdamW is the universal
              default. The hyperparameters above are from Karpathy&rsquo;s
              nanoGPT&mdash;known-good values for GPT-2 scale training.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Decoupled Weight Decay">
            Adam adds weight decay to the gradient before the adaptive
            step. AdamW applies it directly to the weights after. The
            result: weight decay regularization strength does not depend
            on the learning rate. One line swap, better behavior.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Learning Rate Scheduling
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Learning Rate Scheduling"
            subtitle="Why constant LR breaks at transformer scale"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Series 2, a constant learning rate worked fine for MNIST.
              Set it to 0.001, train for a few epochs, done. Why
              would that not work here?
            </p>

            <p className="text-muted-foreground">
              Because random transformer weights are <strong>fragile</strong>.
              With 124 million randomly initialized parameters, the early
              gradient signals are noisy and the loss landscape is rough. A
              high learning rate at the start pushes the model into regions
              it cannot recover from. A low learning rate throughout wastes
              time in the early phase when there is easy progress to make.
            </p>

            <ComparisonRow
              left={{
                title: 'Constant LR = 6e-4',
                color: 'rose',
                items: [
                  'Too aggressive for random weights',
                  'Training destabilizes in first 100 steps',
                  'Loss spikes, gradients explode',
                  'May never recover',
                ],
              }}
              right={{
                title: 'Constant LR = 1e-5',
                color: 'amber',
                items: [
                  'Safe for random weights',
                  'Misses the easy early progress',
                  'Painfully slow convergence',
                  'Wastes compute',
                ],
              }}
            />

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                Concrete comparison: same model, same data, only the LR strategy differs
              </p>
              <div className="grid gap-3 md:grid-cols-3 text-xs font-mono">
                <div className="space-y-1">
                  <p className="font-sans font-medium text-rose-400 text-sm">Constant LR = 6e-4</p>
                  <p>Step 100: loss 8.2</p>
                  <p>Step 300: loss NaN {'\u2620\uFE0F'}</p>
                  <p className="text-muted-foreground/60 font-sans">Exploded. Training over.</p>
                </div>
                <div className="space-y-1">
                  <p className="font-sans font-medium text-amber-400 text-sm">Constant LR = 1e-5</p>
                  <p>Step 100: loss 10.1</p>
                  <p>Step 1000: loss 7.8</p>
                  <p>Step 5000: loss 5.1</p>
                  <p>Step 10000: loss 4.3</p>
                  <p className="text-muted-foreground/60 font-sans">Crawling. Wastes compute.</p>
                </div>
                <div className="space-y-1">
                  <p className="font-sans font-medium text-emerald-400 text-sm">Warmup + Cosine Decay</p>
                  <p>Step 100: loss 9.5</p>
                  <p>Step 1000: loss 4.2</p>
                  <p>Step 5000: loss 2.5</p>
                  <p>Step 10000: loss 2.0</p>
                  <p className="text-muted-foreground/60 font-sans">Fast and stable.</p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground/80">
                The high constant LR crashes because random weights cannot
                tolerate aggressive updates. The low constant LR avoids
                crashing but makes almost no progress. The schedule navigates
                both failure modes.
              </p>
            </div>

            <p className="text-muted-foreground">
              The solution: <strong>change the learning rate during
              training</strong>. Start low, ramp up, then gradually decrease.
              Two phases:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Warmup</strong> (first ~5&ndash;10% of training):
                linearly ramp LR from near-zero to the peak value.
                Random weights are fragile&mdash;start gentle, let the
                gradients stabilize before committing to large updates.
              </li>
              <li>
                <strong>Cosine decay</strong> (remaining 90&ndash;95%):
                gradually decrease LR following a cosine curve from peak
                down to ~10% of peak. As the model approaches a good
                solution, take smaller steps to fine-tune without
                overshooting.
              </li>
            </ul>

            {/* LR Schedule Visualization */}
            <div className="flex justify-center py-4">
              <svg
                viewBox="0 0 500 200"
                className="w-full max-w-[500px] overflow-visible"
              >
                {/* Grid lines */}
                <line x1="60" y1="30" x2="60" y2="170" stroke="#333" strokeWidth={1} />
                <line x1="60" y1="170" x2="480" y2="170" stroke="#333" strokeWidth={1} />

                {/* Y-axis labels */}
                <text x="55" y="40" textAnchor="end" fill="#6b7280" fontSize="9" fontFamily="monospace">6e-4</text>
                <text x="55" y="170" textAnchor="end" fill="#6b7280" fontSize="9" fontFamily="monospace">0</text>
                <text x="15" y="105" fill="#6b7280" fontSize="9" textAnchor="middle" transform="rotate(-90, 15, 105)">Learning Rate</text>

                {/* X-axis labels */}
                <text x="60" y="185" textAnchor="middle" fill="#6b7280" fontSize="9">0</text>
                <text x="480" y="185" textAnchor="middle" fill="#6b7280" fontSize="9">10,000</text>
                <text x="270" y="198" textAnchor="middle" fill="#6b7280" fontSize="9">Training Steps</text>

                {/* Warmup region shading */}
                <rect x="60" y="30" width="42" height="140" fill="#38bdf8" opacity={0.06} />

                {/* Warmup phase line (linear ramp) */}
                <line x1="60" y1="165" x2="102" y2="40" stroke="#38bdf8" strokeWidth={2.5} />

                {/* Cosine decay curve */}
                <path
                  d={(() => {
                    const points: string[] = []
                    for (let i = 0; i <= 100; i++) {
                      const t = i / 100
                      const x = 102 + t * (480 - 102)
                      // Cosine decay from peak (40) to min_lr (155)
                      const cosVal = 0.5 * (1 + Math.cos(Math.PI * t))
                      const y = 155 - cosVal * (155 - 40)
                      points.push(`${x},${y}`)
                    }
                    return `M ${points.join(' L ')}`
                  })()}
                  fill="none"
                  stroke="#a78bfa"
                  strokeWidth={2.5}
                />

                {/* Phase annotations */}
                <text x="81" y="22" textAnchor="middle" fill="#38bdf8" fontSize="9" fontWeight="600">Warmup</text>
                <text x="291" y="22" textAnchor="middle" fill="#a78bfa" fontSize="9" fontWeight="600">Cosine Decay</text>

                {/* Annotation arrows */}
                <line x1="102" y1="30" x2="102" y2="170" stroke="#6b7280" strokeWidth={0.5} strokeDasharray="3,3" />
                <text x="106" y="185" fill="#6b7280" fontSize="8">~500</text>

                {/* Peak LR annotation */}
                <line x1="95" y1="40" x2="115" y2="40" stroke="#6b7280" strokeWidth={0.5} strokeDasharray="2,2" />
                <text x="120" y="44" fill="#6b7280" fontSize="8">peak LR</text>

                {/* Min LR annotation */}
                <line x1="420" y1="155" x2="460" y2="155" stroke="#6b7280" strokeWidth={0.5} strokeDasharray="2,2" />
                <text x="460" y="150" fill="#6b7280" fontSize="8">min LR</text>
              </svg>
            </div>

            <CodeBlock
              code={`import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    # Warmup phase: linear ramp from min_lr to max_lr
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / warmup_steps)
    # Cosine decay phase: from max_lr down to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# Example values (from nanoGPT)
max_lr = 6e-4
min_lr = 6e-5     # 10% of peak
warmup_steps = 500
max_steps = 10000`}
              language="python"
              filename="lr_schedule.py"
            />

            <p className="text-muted-foreground">
              Remember the Goldilocks zone from your early learning rate
              experiments? LR scheduling is <strong>dynamic Goldilocks</strong>&mdash;the
              best learning rate changes as training progresses. Early on,
              the model needs gentle updates (warmup). In the middle, it can
              handle aggressive learning. Late in training, it needs precision
              (decay). The schedule navigates all three regimes automatically.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Dynamic Goldilocks">
            With a constant LR, you pick one value and hope it works for
            the entire run. With scheduling, you get the best LR for each
            phase: gentle warmup for fragile random weights, aggressive
            learning in the middle, careful convergence at the end.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Gradient Clipping
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Gradient Clipping"
            subtitle="The safety net you heard about&mdash;now deployed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Training Dynamics, gradient clipping was mentioned as a
              &ldquo;safety net for exploding gradients.&rdquo; Now you
              use it. The idea is simple: compute the global norm of all
              gradients. If it exceeds a threshold, scale all gradients down
              proportionally so the norm equals the threshold.
            </p>

            <CodeBlock
              code={`# One line. That's it.
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`}
              language="python"
              filename="gradient_clipping.py"
            />

            <p className="text-muted-foreground">
              Why do transformers need this? Attention can produce occasional
              large gradients, especially early in training when weights are
              random or when a particular batch contains unusual sequences.
              Without clipping, one bad batch can produce a gradient so
              large that the parameter update undoes many steps of training.
              With clipping, the worst that happens is a slightly smaller-than-expected
              update.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium">How clipping works:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>1. Compute the global gradient norm: <InlineMath math="\|g\| = \sqrt{\sum_i g_i^2}" /> across all parameters</li>
                <li>2. If <InlineMath math="\|g\| > \text{max\_norm}" />: scale all gradients by <InlineMath math="\text{max\_norm} / \|g\|" /></li>
                <li>3. If <InlineMath math="\|g\| \leq \text{max\_norm}" />: do nothing</li>
              </ul>
              <p className="text-xs text-muted-foreground/80">
                The direction of the gradient is preserved. Only the magnitude
                is bounded. Most steps are unaffected&mdash;clipping only
                activates on the occasional large spike.
              </p>
            </div>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium">What this looks like in practice:</p>
              <div className="text-sm text-muted-foreground font-mono space-y-1">
                <p>Step 1200: grad_norm = 0.83 {'\u2192'} no clip</p>
                <p>Step 1201: grad_norm = 0.61 {'\u2192'} no clip</p>
                <p>Step 1202: grad_norm = 0.92 {'\u2192'} no clip</p>
                <p className="text-rose-400 font-semibold">Step 1203: grad_norm = 12.3 {'\u2192'} clipped to 1.0</p>
                <p>Step 1204: grad_norm = 0.74 {'\u2192'} no clip</p>
              </div>
              <p className="text-xs text-muted-foreground/80">
                That 12.3 is a batch with an unusual sequence that produces
                an outsized gradient. Without clipping, that single step would
                move the parameters 12x more than a normal step&mdash;potentially
                undoing hundreds of steps of progress. With clipping, the update
                stays bounded. Training continues as if nothing happened.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Where in the Loop">
            Gradient clipping goes <strong>after</strong>{' '}
            <code className="text-xs">loss.backward()</code> and{' '}
            <strong>before</strong>{' '}
            <code className="text-xs">optimizer.step()</code>. The gradients
            must exist before you can clip them, and the update must use the
            clipped values.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. The Complete Training Loop
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Training Loop"
            subtitle="Everything assembled"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now assemble everything. The training loop below combines the
              familiar forward/backward/update pattern with the three new
              additions: dataset preparation (already done), gradient
              clipping, and LR scheduling. Each line is annotated.
            </p>

            <CodeBlock
              code={`from torch.nn.utils import clip_grad_norm_
from itertools import cycle

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=6e-4,
    betas=(0.9, 0.95), weight_decay=0.1
)

max_steps = 10000
warmup_steps = 500

# Create an infinite iterator that cycles through the dataset
train_iter = iter(cycle(train_loader))

for step in range(max_steps):
    model.train()

    # Update learning rate for this step
    lr = get_lr(step, warmup_steps, max_steps,
                max_lr=6e-4, min_lr=6e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get batch — cycle restarts when the dataset is exhausted
    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)

    # Forward — same as always
    logits, loss = model(x, y)

    # Backward — same as always
    optimizer.zero_grad()
    loss.backward()

    # NEW: clip gradients before update
    clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update — same as always
    optimizer.step()

    # Log periodically
    if step % 100 == 0:
        print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

    # Evaluate on validation set periodically
    if step % 500 == 0:
        model.eval()
        val_loss = 0.0
        val_steps = 20
        val_iter = iter(val_loader)  # fresh iterator over the val set
        with torch.no_grad():
            for _ in range(val_steps):
                xv, yv = next(val_iter)
                xv, yv = xv.to(device), yv.to(device)
                _, vloss = model(xv, yv)
                val_loss += vloss.item()
        print(f"step {step:5d} | val loss {val_loss / val_steps:.4f}")
        model.train()

    # Save checkpoint periodically (same pattern from Series 2)
    if step > 0 and step % 2000 == 0:
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"checkpoint_{step}.pt")`}
              language="python"
              filename="training_loop.py"
            />

            <p className="text-muted-foreground">
              Count the lines that are genuinely new compared to your MNIST
              loop: the <code className="text-xs">get_lr()</code> call, the{' '}
              <code className="text-xs">param_group</code> update, and the{' '}
              <code className="text-xs">clip_grad_norm_</code> call. Three
              lines in the core loop. The validation evaluation and
              checkpointing are the same patterns from Series 2, applied
              here without modification. Everything else&mdash;the forward
              pass, loss computation, <code className="text-xs">zero_grad()</code>,{' '}
              <code className="text-xs">backward()</code>,{' '}
              <code className="text-xs">step()</code>&mdash;is the same
              heartbeat you have been writing since Series 1.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Manual LR Update">
            The code manually sets the learning rate each step via{' '}
            <code className="text-xs">{"param_group['lr']"}</code>. This is
            simpler than using{' '}
            <code className="text-xs">torch.optim.lr_scheduler</code> and
            makes the schedule logic fully transparent. Production code
            often uses the built-in schedulers, but the effect is
            identical.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Periodic Text Generation
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Watching the Model Learn"
            subtitle="Generated text at different loss values"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Loss numbers tell you training is working. Generated text
              shows you <em>what</em> the model has learned. Add periodic
              text generation to the training loop:
            </p>

            <CodeBlock
              code={`# Add inside the training loop, every 1000 steps:
if step % 1000 == 0:
    model.eval()
    prompt = enc.encode("ROMEO:")
    idx = torch.tensor([prompt], device=device)
    tokens = model.generate(idx, max_new_tokens=100, temperature=0.8)
    print(f"\\n--- Step {step} (loss={loss.item():.2f}) ---")
    print(enc.decode(tokens[0].tolist()))
    print()
    model.train()`}
              language="python"
              filename="periodic_generation.py"
            />

            <p className="text-muted-foreground">
              Here is what you will see at different stages of training:
            </p>

            <div className="space-y-3">
              <div className="px-4 py-3 bg-rose-500/5 border border-rose-500/20 rounded-lg">
                <p className="text-xs font-medium text-rose-400 mb-1">Step 0&mdash;Loss ~10.8</p>
                <p className="text-xs font-mono text-muted-foreground">
                  ROMEO:Frag broadly meanwhile precip Winn472 fireplace oxy deserted bure...
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">Random tokens. No structure at all.</p>
              </div>

              <div className="px-4 py-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
                <p className="text-xs font-medium text-amber-400 mb-1">Step 500&mdash;Loss ~4.0</p>
                <p className="text-xs font-mono text-muted-foreground">
                  ROMEO: the the and is to a I of the in the...
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">Common words emerge. No grammar yet.</p>
              </div>

              <div className="px-4 py-3 bg-sky-500/5 border border-sky-500/20 rounded-lg">
                <p className="text-xs font-medium text-sky-400 mb-1">Step 2000&mdash;Loss ~3.0</p>
                <p className="text-xs font-mono text-muted-foreground">
                  ROMEO: What is the matter with the prince of the that the...
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">Recognizable English phrases. Crude grammar emerging.</p>
              </div>

              <div className="px-4 py-3 bg-violet-500/5 border border-violet-500/20 rounded-lg">
                <p className="text-xs font-medium text-violet-400 mb-1">Step 5000&mdash;Loss ~2.5</p>
                <p className="text-xs font-mono text-muted-foreground">
                  ROMEO: What shall I say to thee? I am not worthy of thy love, but I will...
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">Grammatical sentences. Shakespeare-like patterns.</p>
              </div>

              <div className="px-4 py-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
                <p className="text-xs font-medium text-emerald-400 mb-1">Step 10000&mdash;Loss ~2.0</p>
                <p className="text-xs font-mono text-muted-foreground">
                  ROMEO: What shall I say? I do beseech your grace to hear me speak, for I have done no harm...
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">Coherent Shakespeare-like passages. The model has learned language.</p>
              </div>
            </div>

            <p className="text-muted-foreground">
              This is the payoff. The same architecture that produced random
              gibberish in Building nanoGPT now generates recognizable
              English&mdash;because every weight has been updated by the same
              training loop you have used since Series 1.
            </p>

            <p className="text-muted-foreground">
              Notice the pattern in the progression above: the leap from loss
              10.8 to 4.0 (random tokens to real words) is <strong>enormous</strong>.
              The leap from 2.5 to 2.0 (grammatical sentences to slightly
              better sentences) is subtle. The relationship between loss and
              text quality is logarithmic, not linear&mdash;the biggest
              qualitative improvements happen early, and later gains yield
              diminishing returns in perceived quality.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Biggest Leaps Are Early">
            Going from loss 10.8 to 4.0 is a dramatic qualitative
            leap&mdash;gibberish to real words. Going from 2.0 to 1.8 is
            subtle. The loss-to-quality mapping is logarithmic, not
            linear. The most exciting progress happens in the first few
            thousand steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Interpreting the Loss Curve
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Reading the Loss Curve"
            subtitle="What's normal, what's not"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know how to read training curves from Series 1&mdash;the
              same diagnostic skills apply here. But language model loss
              curves look different from MNIST. Here is what to expect.
            </p>

            <div className="space-y-3">
              <GradientCard title="Your loss curve looks jagged with lots of up-and-down variation. Is something wrong?" color="blue">
                <p className="text-sm">
                  No. Language model loss is inherently noisier than MNIST. Some
                  batches contain predictable sequences (&ldquo;to be or not to
                  be&rdquo;); others contain rare words and unusual syntax. The
                  loss varies batch to batch. Use a running average (smoothed
                  loss) to see the trend. The raw noise is normal, not a sign of
                  trouble.
                </p>
              </GradientCard>

              <GradientCard title="Normal: Loss drops fast, then slows" color="blue">
                <p className="text-sm">
                  The easy wins come first: learning common words, basic
                  grammar, frequent patterns. Later improvements are harder:
                  rare vocabulary, complex syntax, long-range coherence. The
                  curve bends&mdash;steep early, flattening later. This is
                  expected.
                </p>
              </GradientCard>

              <GradientCard title="Your loss suddenly spiked at step 3000 and hasn't come back down. What happened?" color="rose">
                <p className="text-sm">
                  A sudden spike that recovers within a few hundred steps is
                  normal (an unusually hard batch). A spike that <em>stays</em>{' '}
                  elevated means training destabilized&mdash;likely a learning
                  rate too high for the current weights. Gradient clipping should
                  prevent the worst cases, but extreme spikes can still happen.
                  If the loss does not recover, reduce the peak learning rate
                  and restart from the last checkpoint.
                </p>
              </GradientCard>

              <GradientCard title="Warning: Loss plateaus very early" color="amber">
                <p className="text-sm">
                  If loss stops decreasing after only a few hundred steps, the
                  learning rate is probably too low. The model is making tiny
                  updates and barely exploring the loss landscape. Check the LR
                  schedule&mdash;is the peak LR too conservative?
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Monitor both training and validation loss&mdash;the same
              &ldquo;scissors pattern&rdquo; from Series 1 applies. When the
              training loss keeps dropping but the validation loss starts
              rising, the model is memorizing rather than learning. With
              TinyShakespeare (~300K tokens) and a 124M-parameter model,
              this will happen eventually&mdash;the model has far more
              capacity than the data can fill. That is fine for learning.
              The next lesson addresses scale, data quality, and the gap
              between training on 300K tokens and 300 billion.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Loss Is Not Accuracy">
            A loss of 2.0 does not mean 20% accuracy. Cross-entropy
            loss is in nats (natural log units). The mapping to text
            quality is nonlinear and depends on the dataset. Compare
            loss values within the same run, not across different
            datasets or model sizes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Notebook Link
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Train It Yourself"
            subtitle="Open the notebook and run the training loop"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The lesson showed you the components. Now assemble and run
              them yourself. The notebook guides you through: loading and
              tokenizing text, building the Dataset, assembling the training
              loop with LR scheduling and gradient clipping, and watching
              your model learn Shakespeare.
            </p>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Train your GPT from scratch on real text.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-3-2-pretraining.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes guided dataset preparation,
                  a training loop skeleton to complete, LR scheduling
                  implementation, and stretch exercises for experimentation.
                  A free Colab GPU is sufficient for TinyShakespeare.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'The training loop is the same loop. Always.',
                description:
                  'Forward, loss, backward, step. From linear regression to GPT, the algorithm never changes. The data format changes, the model changes, the scale changes. The loop does not.',
              },
              {
                headline:
                  'Three new tools for transformer-scale training.',
                description:
                  'Text dataset preparation (input/target offset for next-token prediction), LR scheduling (warmup + cosine decay), and gradient clipping (one line, essential safety net). Everything else is the same heartbeat from Series 2.',
              },
              {
                headline:
                  'Cross-entropy scales to any vocabulary size.',
                description:
                  'Same formula, same PyTorch call. Reshape logits from (B, T, V) to (B*T, V), targets from (B, T) to (B*T). The vocabulary size is just another number.',
              },
              {
                headline:
                  'Loss-to-text-quality is nonlinear.',
                description:
                  'The biggest qualitative leaps happen early (gibberish to words, words to phrases). Later improvements are subtle. Do not expect the same dramatic progress all the way down.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          15. Forward Reference + Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">
              What comes next
            </p>
            <p className="text-sm text-muted-foreground">
              Your training loop works. But it is slow&mdash;training 124M
              parameters on a single GPU with float32 precision on 300K
              tokens of Shakespeare. What changes when you want to train
              GPT-2 for real? What happens when the model is 100x bigger and
              the dataset is 1000x larger? The next lesson covers GPU
              utilization, mixed precision, KV caching, flash attention,
              and the scaling laws that govern how models improve with
              compute.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Complete the notebook, watch your model learn Shakespeare, and experiment with different learning rates and training durations. Then review your session."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
