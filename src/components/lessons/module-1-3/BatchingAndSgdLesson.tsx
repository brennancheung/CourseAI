'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  TryThisBlock,
  WarningBlock,
  ConceptBlock,
  SummaryBlock,
  NextStepBlock,
  ConstraintBlock,
  ComparisonRow,
  GradientCard,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { SGDExplorer } from '@/components/widgets/SGDExplorer'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Batching and SGD — Lesson 4 of Module 1.3 (Training Neural Networks)
 *
 * Teaches the student to understand:
 * - Why batching is necessary (computational argument)
 * - How mini-batches work (shuffle, split, iterate)
 * - SGD vocabulary (batch, mini-batch, stochastic, epoch, iteration)
 * - Gradient noise as a concept (INTRODUCED)
 * - Batch size as a hyperparameter with tradeoffs
 * - Why shuffling matters
 *
 * Target depths:
 * - Mini-batches: DEVELOPED
 * - Stochastic gradient descent: DEVELOPED
 * - Epochs: DEVELOPED
 * - Gradient noise as helpful: INTRODUCED
 */

export function BatchingAndSgdLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Batching and SGD"
            description="From one data point at a time to training on real datasets."
            category="Training Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand why and how training data is divided into mini-batches,
            what stochastic gradient descent is, and how epochs structure the
            training process.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Loop, New Scale">
            You already know the training loop: forward, loss, backward, update.
            This lesson doesn&apos;t change the loop—it changes <strong>how much data</strong> flows
            through it at each step.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'We are NOT covering optimizers (momentum, Adam)—that is the next lesson',
              'No GPU parallelism or data loading details',
              'Focus on the mechanics: batching, epochs, and the tradeoffs',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook — The Scale Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Scale Problem"
            subtitle="Everything so far has been a toy problem"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In every lesson so far, you&apos;ve trained on a handful of data points. One input,
              one output, one gradient, one update. That was deliberate—it kept the focus on the
              algorithm itself.
            </p>
            <p className="text-muted-foreground">
              But real datasets aren&apos;t tiny. ImageNet has <strong>1.2 million</strong> images.
              Common text datasets have billions of tokens. If you computed the gradient over the
              entire dataset before making a single update, one step of gradient descent would
              require 1.2 million forward passes and 1.2 million backward passes. At even 10ms per
              pass, that&apos;s <strong>over 6 hours for a single weight update</strong>.
            </p>
            <p className="text-muted-foreground">
              And typical training needs thousands of updates. You&apos;d be waiting years.
            </p>

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5">
              <p className="text-amber-400 font-semibold text-center">
                Computing the exact gradient over the full dataset is correct—but hopelessly slow.
              </p>
            </div>

            <p className="text-muted-foreground">
              There&apos;s a better way. And it doesn&apos;t require changing anything about the
              training loop you already know.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Same Update Rule">
            The gradient descent update rule stays the same:{' '}
            <InlineMath math="\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla L" />.
            The only thing that changes is <em>which data</em> computes{' '}
            <InlineMath math="\nabla L" />.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Explain — The Polling Analogy and Mini-Batches */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Polling Insight"
            subtitle="You don't need all the data to get a useful answer"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Suppose you want to know the average height of adults in a city of 1 million people.
              Do you measure every single person? Of course not. You take a <strong>random
              sample</strong> of, say, 50 people, and compute the average. It won&apos;t be exactly
              right, but it will be close enough to be useful.
            </p>
            <p className="text-muted-foreground">
              Gradients work the same way. The gradient computed over the full dataset tells you the
              &ldquo;true&rdquo; direction to update your parameters. But the gradient computed
              over a random sample of 32 data points also points in <em>roughly</em> the right
              direction. It&apos;s a noisy estimate—but a useful one.
            </p>
            <p className="text-muted-foreground">
              That random sample is called a <strong>mini-batch</strong>.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground">
                <strong>Full-batch gradient</strong> (average over all N data points):
              </p>
              <BlockMath math="\nabla L = \frac{1}{N}\sum_{i=1}^{N} \nabla L_i" />

              <p className="text-sm text-muted-foreground">
                <strong>Mini-batch gradient</strong> (average over B random data points):
              </p>
              <BlockMath math="\nabla \tilde{L} = \frac{1}{B}\sum_{i=1}^{B} \nabla L_i" />

              <p className="text-xs text-muted-foreground">
                Same structure, same formula. The only difference is <strong>how many</strong> data
                points you sum over. If you pick B data points randomly, the mini-batch gradient
                points in roughly the right direction—and on average, it equals the full gradient.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Jargon Preview">
            Statisticians call this an &ldquo;unbiased estimator&rdquo;—meaning the
            mini-batch gradient isn&apos;t systematically wrong in any direction. You don&apos;t
            need the formal term yet. The key idea: on average, mini-batch gradients
            equal the full gradient.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Concrete example: 1000 houses, batch_size=50 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Walking Through the Mechanics"
            subtitle="1000 houses, batch size 50"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine you&apos;re training a model to predict house prices. You have 1,000 training
              examples, and you choose a batch size of 50. Here&apos;s what happens:
            </p>

            <div className="space-y-3">
              <GradientCard title="Step 1: Shuffle" color="cyan">
                <p className="text-sm">Randomly reorder all 1,000 examples.</p>
              </GradientCard>
              <GradientCard title="Step 2: Split into batches" color="blue">
                <p className="text-sm">
                  1,000 / 50 = <strong>20 batches</strong> of 50 examples each.
                </p>
              </GradientCard>
              <GradientCard title="Step 3: Process each batch" color="purple">
                <p className="text-sm">
                  For each of the 20 batches, run the training loop you already know:
                  forward pass on 50 examples, compute average loss, backward pass to get
                  gradients, update all weights.
                </p>
              </GradientCard>
              <GradientCard title="Result" color="emerald">
                <p className="text-sm">
                  <strong>20 weight updates</strong> from one pass through the data.
                  Compare that to full-batch gradient descent, which would give you just{' '}
                  <strong>1 update</strong> from the same data.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Each individual update is noisier (computed from 50 examples instead of 1,000). But
              you get <strong>20 times more updates</strong>. In practice, this wins—by a lot.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Quick Mental Math">
            <p className="text-sm">
              Updates per epoch = dataset size / batch size.
            </p>
            <p className="text-sm mt-2">
              1000 / 50 = 20 updates.
            </p>
            <p className="text-sm mt-2">
              5 epochs = 100 total updates.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 4. Visual — Gradient Arrows */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Noisy But Useful"
            subtitle="Mini-batch gradients cluster around the true direction"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s the key visual intuition. Imagine you&apos;re standing on the loss
              landscape. The full-batch gradient gives you one precise arrow pointing toward the
              minimum. Each mini-batch gradient gives you a slightly different arrow—some point
              a little left, some a little right—but they all <em>roughly</em> agree on the
              direction.
            </p>

            {/* Gradient arrow visualization */}
            <div className="rounded-lg border bg-muted/30 p-6">
              <div className="flex flex-col items-center gap-4">
                <div className="flex items-center gap-8 flex-wrap justify-center">
                  {/* Full-batch arrow */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="text-xs text-muted-foreground font-medium">Full-batch</div>
                    <svg width="80" height="80" viewBox="0 0 80 80">
                      <circle cx="40" cy="40" r="6" fill="#f97316" />
                      <line
                        x1="40" y1="40" x2="40" y2="10"
                        stroke="#6366f1" strokeWidth="3"
                        markerEnd="url(#arrowFull)"
                      />
                      <defs>
                        <marker id="arrowFull" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                          <path d="M0,0 L8,3 L0,6" fill="#6366f1" />
                        </marker>
                      </defs>
                    </svg>
                    <div className="text-xs text-indigo-400">1 precise arrow</div>
                  </div>

                  <div className="text-muted-foreground text-2xl">&rarr;</div>

                  {/* Mini-batch arrows */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="text-xs text-muted-foreground font-medium">Mini-batch (3 samples)</div>
                    <svg width="80" height="80" viewBox="0 0 80 80">
                      <circle cx="40" cy="40" r="6" fill="#f97316" />
                      <line x1="40" y1="40" x2="30" y2="12" stroke="#6366f1" strokeWidth="2" opacity="0.6"
                        markerEnd="url(#arrowMini1)" />
                      <line x1="40" y1="40" x2="42" y2="8" stroke="#6366f1" strokeWidth="2" opacity="0.6"
                        markerEnd="url(#arrowMini2)" />
                      <line x1="40" y1="40" x2="52" y2="14" stroke="#6366f1" strokeWidth="2" opacity="0.6"
                        markerEnd="url(#arrowMini3)" />
                      <defs>
                        <marker id="arrowMini1" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
                          <path d="M0,0 L6,2.5 L0,5" fill="#6366f1" />
                        </marker>
                        <marker id="arrowMini2" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
                          <path d="M0,0 L6,2.5 L0,5" fill="#6366f1" />
                        </marker>
                        <marker id="arrowMini3" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
                          <path d="M0,0 L6,2.5 L0,5" fill="#6366f1" />
                        </marker>
                      </defs>
                    </svg>
                    <div className="text-xs text-indigo-400">noisy, but roughly right</div>
                  </div>
                </div>

                <p className="text-xs text-muted-foreground text-center max-w-sm">
                  Each mini-batch arrow is slightly off, but they cluster around the true direction.
                  Average them and you&apos;d recover the full-batch arrow.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not &ldquo;Inaccurate&rdquo;">
            The mini-batch gradient is not wrong—it&apos;s a <em>noisy estimate</em>.
            Like a poll that samples 50 people instead of 1 million: not exact, but
            consistently in the right ballpark.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 5. Check 1 — Predict and Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              You have 600 training examples and a batch size of 32. How many batches per epoch?
              How many weight updates per epoch?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Pause and compute before revealing the answer.
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show solution
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  600 / 32 = 18.75, so you get <strong>18 full batches of 32</strong> with 24
                  examples left over. Two common conventions: use 18 batches and drop the remainder,
                  or use 19 batches where the last one has 24 examples.
                </p>
                <p className="text-muted-foreground">
                  Either way, you get <strong>18-19 weight updates per epoch</strong>. Compare that
                  to full-batch gradient descent: <strong>1 update per epoch</strong>.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 6. Explain — Epochs and the Full Algorithm */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Epochs: Cycling Through the Data"
            subtitle="One epoch = one complete pass through the dataset"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              After you&apos;ve processed all 20 batches (in the 1000-house example), you&apos;ve
              seen every training example once. That&apos;s one <strong>epoch</strong>.
            </p>
            <p className="text-muted-foreground">
              One epoch usually isn&apos;t enough. The model needs to see the data multiple times—each
              pass refines the parameters further. Typical training runs do 5, 10, 50, or even
              hundreds of epochs.
            </p>

            <div className="space-y-2">
              <p className="text-sm text-muted-foreground font-semibold">Vocabulary check:</p>
              <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
                <li><strong>Epoch</strong>—one complete pass through all training data</li>
                <li><strong>Iteration</strong> (or <strong>step</strong>)—one gradient update from one batch</li>
                <li><strong>Batch size</strong> (B)—how many examples per update</li>
              </ul>
              <div className="py-3 px-5 bg-muted/50 rounded-lg text-center mt-3">
                <p className="text-sm font-mono text-muted-foreground">
                  Iterations per epoch = N / B &nbsp;&nbsp;|&nbsp;&nbsp;
                  Total iterations = (N / B) &times; num_epochs
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Here&apos;s the full algorithm. Notice that the inner four lines are exactly
              the training loop from Implementing Linear Regression—the only new part is the two
              outer loops.
            </p>

            <CodeBlock
              code={`for epoch in range(num_epochs):
    shuffle(data)                         # randomize order each epoch
    for batch in split(data, batch_size): # split into mini-batches
        predictions = forward(batch)      #   same training loop
        loss = compute_loss(predictions, batch.targets)
        gradients = backward(loss)
        update_weights(gradients, lr)`}
              language="python"
              filename="mini_batch_sgd.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Heartbeat">
            The inner loop is the exact training loop you already know.
            Mini-batch SGD wraps it in two new loops: <strong>epochs</strong> (how many
            times through the data) and <strong>batches</strong> (which slice of data to use).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 7. Negative Example — The Sorted Dataset */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Shuffle?"
            subtitle="What happens when you don't"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might wonder: does the order of the data really matter? If you see all of it in
              an epoch anyway, why bother shuffling?
            </p>
            <p className="text-muted-foreground">
              Imagine your dataset is sorted by label: all cat images first, then all dog images.
              Without shuffling, the first several batches are <em>all cats</em>. The gradient
              says &ldquo;optimize for cats!&rdquo; and the weights shift toward recognizing cats.
              Then the dog batches arrive and yank the parameters back the other way.
            </p>

            <ComparisonRow
              left={{
                title: 'Without shuffling',
                color: 'rose',
                items: [
                  'Batches 1-10: all cats',
                  'Model overfits to cats',
                  'Batches 11-20: all dogs',
                  'Parameters oscillate back',
                  'Slow, unstable convergence',
                ],
              }}
              right={{
                title: 'With shuffling',
                color: 'emerald',
                items: [
                  'Each batch: mix of cats and dogs',
                  'Gradients are representative',
                  'Parameters move consistently',
                  'Stable convergence',
                  'Shuffle again each epoch',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The polling analogy makes this clear: if your &ldquo;random sample&rdquo; is all
              people from one neighborhood, it&apos;s not random at all. Shuffling ensures each
              batch is a representative sample of the full dataset.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Shuffle Every Epoch">
            Shuffling once isn&apos;t enough. If you use the same batch order every epoch,
            the model can memorize the pattern of batches. Re-shuffle before each epoch
            for genuinely random samples.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 8. Explain — The SGD Spectrum */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The SGD Spectrum"
            subtitle="batch_size=1 vs batch_size=ALL vs the sweet spot"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now let&apos;s name what we&apos;ve been building. The algorithm you just saw—using
              mini-batches of data to compute approximate gradients—is called{' '}
              <strong>stochastic gradient descent</strong> (SGD). &ldquo;Stochastic&rdquo; means
              random: the gradient is computed from a random subset of data.
            </p>

            <p className="text-muted-foreground">
              Batch size creates a spectrum:
            </p>

            <div className="grid gap-3 md:grid-cols-3">
              <GradientCard title="Pure SGD" color="rose">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Batch size = 1</li>
                  <li>&bull; Extremely noisy gradient</li>
                  <li>&bull; Like estimating a city&apos;s average height by measuring one random person—you might get a 6&apos;5&rdquo; basketball player</li>
                  <li>&bull; Rarely used in practice</li>
                </ul>
              </GradientCard>
              <GradientCard title="Mini-batch SGD" color="blue">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Batch size = 32-256</li>
                  <li>&bull; Moderate noise</li>
                  <li>&bull; Many updates per epoch</li>
                  <li>&bull; <strong>This is the default</strong></li>
                </ul>
              </GradientCard>
              <GradientCard title="Full-batch GD" color="amber">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Batch size = N (all data)</li>
                  <li>&bull; No noise—exact gradient</li>
                  <li>&bull; One update per epoch</li>
                  <li>&bull; Too slow for large datasets</li>
                </ul>
              </GradientCard>
            </div>

            <div className="rounded-md border border-violet-500/30 bg-violet-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-violet-400">Terminology note:</strong> In modern practice,
              when people say &ldquo;SGD&rdquo; they almost always mean <em>mini-batch</em> SGD.
              Pure single-sample SGD is a textbook concept, not something you&apos;d use. If
              someone says &ldquo;I&apos;m training with SGD, batch size 64,&rdquo; that&apos;s
              mini-batch SGD.
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Name">
            <p className="text-sm">
              <strong>Stochastic</strong> = random.
            </p>
            <p className="text-sm mt-1">
              The gradient is computed from a random sample, making each step stochastic.
              Full-batch gradient descent is <em>deterministic</em> (same data, same gradient).
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Gradient noise as feature */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Noise as a Feature"
            subtitle="The surprising benefit of approximate gradients"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s the part that surprises people. You might expect the noise from
              mini-batches to be purely a downside—a cost you pay for speed. But it turns out
              gradient noise is <em>actually helpful</em>.
            </p>
            <p className="text-muted-foreground">
              Remember the loss landscape from Gradient Descent? For linear regression, it was a
              smooth bowl with one minimum. Neural network loss landscapes are more complex—they
              have multiple valleys, some deep and narrow (&ldquo;sharp&rdquo; minima), others
              wide and flat.
            </p>
            <p className="text-muted-foreground">
              Full-batch gradient descent follows a smooth path into the nearest minimum—whether
              or not that minimum is a good one. Mini-batch SGD bounces around. That
              bouncing can <strong>knock the parameters out of a sharp minimum</strong> and
              into a wide one. Wide minima tend to generalize better (they&apos;re less sensitive
              to small changes in the input).
            </p>

            {/* 1D cross-section: smooth path into sharp min vs noisy path into wide min */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <svg viewBox="0 0 400 160" className="w-full" style={{ maxHeight: 180 }}>
                {/* Loss landscape curve (1D cross-section) */}
                <path
                  d="M 10,40 Q 60,38 90,100 Q 105,140 120,100 Q 135,60 150,75 Q 175,100 200,95 Q 240,85 270,90 Q 290,95 320,80 Q 340,60 360,55 Q 380,52 390,55"
                  fill="none" stroke="#6366f1" strokeWidth="2" opacity="0.5"
                />
                {/* Shade under curve */}
                <path
                  d="M 10,40 Q 60,38 90,100 Q 105,140 120,100 Q 135,60 150,75 Q 175,100 200,95 Q 240,85 270,90 Q 290,95 320,80 Q 340,60 360,55 Q 380,52 390,55 L 390,155 L 10,155 Z"
                  fill="#6366f1" opacity="0.06"
                />

                {/* Sharp minimum region label */}
                <text x="107" y="155" textAnchor="middle" fill="#ef4444" fontSize="9" fontWeight="500">sharp</text>
                {/* Wide minimum region label */}
                <text x="280" y="108" textAnchor="middle" fill="#22c55e" fontSize="9" fontWeight="500">wide</text>

                {/* Full-batch: smooth path rolling into the sharp minimum */}
                <circle cx="30" cy="38" r="3" fill="#f97316" opacity="0.4" />
                <circle cx="50" cy="36" r="3" fill="#f97316" opacity="0.5" />
                <circle cx="70" cy="50" r="3" fill="#f97316" opacity="0.6" />
                <circle cx="85" cy="80" r="3" fill="#f97316" opacity="0.7" />
                <circle cx="95" cy="110" r="3" fill="#f97316" opacity="0.8" />
                <circle cx="105" cy="130" r="3.5" fill="#f97316" opacity="0.9" />
                <circle cx="110" cy="137" r="4" fill="#f97316" />
                <polyline
                  points="30,38 50,36 70,50 85,80 95,110 105,130 110,137"
                  fill="none" stroke="#f97316" strokeWidth="1.5" opacity="0.6"
                  strokeLinejoin="round"
                />
                <text x="110" y="148" textAnchor="middle" fill="#f97316" fontSize="8">stuck</text>

                {/* Mini-batch SGD: noisy path that bounces PAST the sharp min into the wide min */}
                <circle cx="30" cy="35" r="3" fill="#22c55e" opacity="0.3" />
                <circle cx="55" cy="30" r="3" fill="#22c55e" opacity="0.35" />
                <circle cx="72" cy="55" r="3" fill="#22c55e" opacity="0.4" />
                <circle cx="90" cy="95" r="3" fill="#22c55e" opacity="0.45" />
                <circle cx="100" cy="115" r="3" fill="#22c55e" opacity="0.5" />
                <circle cx="118" cy="95" r="3" fill="#22c55e" opacity="0.55" />
                <circle cx="140" cy="68" r="3" fill="#22c55e" opacity="0.6" />
                <circle cx="165" cy="82" r="3" fill="#22c55e" opacity="0.65" />
                <circle cx="195" cy="90" r="3" fill="#22c55e" opacity="0.7" />
                <circle cx="225" cy="82" r="3" fill="#22c55e" opacity="0.75" />
                <circle cx="250" cy="88" r="3" fill="#22c55e" opacity="0.8" />
                <circle cx="270" cy="86" r="3.5" fill="#22c55e" opacity="0.9" />
                <circle cx="278" cy="88" r="4" fill="#22c55e" />
                <polyline
                  points="30,35 55,30 72,55 90,95 100,115 118,95 140,68 165,82 195,90 225,82 250,88 270,86 278,88"
                  fill="none" stroke="#22c55e" strokeWidth="1.5" opacity="0.5"
                  strokeLinejoin="round" strokeDasharray="4 2"
                />

                {/* Axis labels */}
                <text x="5" y="95" fill="currentColor" fontSize="9" opacity="0.5" transform="rotate(-90, 5, 95)">Loss</text>
                <text x="200" y="155" textAnchor="middle" fill="currentColor" fontSize="9" opacity="0.5">Parameters</text>

                {/* Legend */}
                <line x1="15" y1="15" x2="30" y2="15" stroke="#f97316" strokeWidth="2" />
                <text x="34" y="18" fill="#f97316" fontSize="9">Full-batch (smooth, gets trapped)</text>
                <line x1="210" y1="15" x2="225" y2="15" stroke="#22c55e" strokeWidth="2" strokeDasharray="4 2" />
                <text x="229" y="18" fill="#22c55e" fontSize="9">Mini-batch (noisy, escapes)</text>
              </svg>
              <p className="text-xs text-muted-foreground text-center mt-2">
                1D cross-section of the loss landscape. The smooth path rolls into the nearest
                minimum (sharp). The noisy path bounces past it and settles in a wider, better minimum.
              </p>
            </div>

            <div className="rounded-lg border-2 border-violet-500/30 bg-violet-500/5 p-5 text-center">
              <p className="text-violet-400 font-semibold">
                The ball is still rolling downhill. But now the hill is shaking—and that&apos;s
                a good thing.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Sharp vs Wide Minima">
            A <strong>sharp</strong> minimum has steep walls—small changes in parameters cause
            big changes in loss. A <strong>wide</strong> minimum is flat—the model is robust
            to small perturbations. Wide is better for generalization.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 9. Check 2 — Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              Your model is training but the loss has plateaued. A colleague suggests
              increasing the batch size from 32 to 512. What are the tradeoffs?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Think about: accuracy of gradients, number of updates per epoch, noise level,
              and what you just learned about sharp vs wide minima.
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show solution
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  <strong>Pros:</strong> More accurate gradients per step (less noise), smoother
                  convergence, each gradient estimate is closer to the true gradient.
                </p>
                <p className="text-muted-foreground">
                  <strong>Cons:</strong> Fewer updates per epoch (512 vs 32 means ~16x fewer steps),
                  less gradient noise (might get stuck in a sharp minimum), slower to
                  compute each step (more examples to process).
                </p>
                <p className="text-muted-foreground">
                  The plateau might actually be because the model is trapped in a bad minimum. In
                  that case, <em>decreasing</em> the batch size (more noise) might help escape it.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="No Single Right Answer">
            Batch size is a tradeoff, not a formula. The right size depends on your dataset,
            model, hardware, and what problem you&apos;re encountering. Common starting point:
            32 or 64.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Interactive Widget — SGD Explorer */}
      <Row>
        <Row.Content>
          <ExercisePanel
            title="SGD Explorer"
            subtitle="Watch how batch size affects the optimization path"
          >
            <SGDExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ol className="space-y-2 text-sm list-decimal list-inside">
              <li>Start with batch=ALL and click Run—watch the smooth path to the nearest minimum</li>
              <li>Reset, switch to batch=1, and run again—see how noisy the path gets</li>
              <li>Now try batch=32—the sweet spot. Does it escape the sharp minimum?</li>
              <li>Run batch=ALL several times—it always takes the same path (deterministic)</li>
              <li>Then run batch=8 several times—each run follows a different path (stochastic)</li>
            </ol>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* 11. Elaborate — Practical Details */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practical Details"
            subtitle="Batch size as a hyperparameter"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Batch size is a <strong>hyperparameter</strong>—a value you choose, not one the model
              learns. Just like learning rate from the earlier lesson.
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Common values:</strong> 32, 64, 128, 256. Powers of 2 are standard
                because they map efficiently to GPU hardware.
              </li>
              <li>
                <strong>Last batch:</strong> If the dataset size isn&apos;t divisible by the
                batch size, the last batch is smaller. Most frameworks handle this automatically.
              </li>
              <li>
                <strong>Batch size and learning rate interact:</strong> Larger batches produce
                more accurate gradients, so you can often use a larger learning rate. This
                relationship matters more as you tune performance—for now, just know it exists.
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Start Simple">
            When in doubt, use batch size 32 and adjust from there. It&apos;s a reasonable
            default that works across most problems.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 12. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Mini-batches: random subsets of data per gradient update',
                description:
                  'Same update rule, less data per step, but many more steps per epoch. A random sample gives a useful gradient estimate.',
              },
              {
                headline: 'SGD: gradient descent with mini-batch gradients',
                description:
                  'Mini-batch SGD is the default. Pure SGD (batch=1) is too noisy, full-batch GD is too slow. Mini-batch is the sweet spot.',
              },
              {
                headline: 'Epochs: one full pass through the dataset',
                description:
                  'Total updates = (N / B) × num_epochs. More epochs = more refinement.',
              },
              {
                headline: 'Gradient noise helps escape bad minima',
                description:
                  'The randomness from mini-batches prevents the model from getting trapped in sharp, narrow valleys.',
              },
              {
                headline: 'Shuffle before every epoch',
                description:
                  "Without shuffling, batches aren’t representative. Sorted data causes oscillation instead of convergence.",
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 13. Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Next: Optimizers"
            description="SGD with a fixed learning rate struggles in ravines and can't adapt to the loss landscape. Momentum, RMSProp, and Adam fix that."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
