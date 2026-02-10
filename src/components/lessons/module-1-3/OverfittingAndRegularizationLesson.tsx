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
  ConstraintBlock,
  ComparisonRow,
  GradientCard,
  ModuleCompleteBlock,
  ReferencesBlock,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { RegularizationExplorer } from '@/components/widgets/RegularizationExplorer'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Overfitting and Regularization — Lesson 7 of Module 1.3 (Training Neural Networks)
 * Final lesson of Series 1: Foundations (capstone)
 *
 * Teaches the student to:
 * - Diagnose overfitting from training curves (train vs val loss divergence)
 * - Understand model capacity and why neural networks overfit
 * - Apply dropout, weight decay (L2), and early stopping
 * - Know when to use which technique
 *
 * Target depths:
 * - Overfitting/generalization: DEVELOPED (from INTRODUCED in 1.1)
 * - Dropout: DEVELOPED
 * - Weight decay / L2 regularization: DEVELOPED
 * - Early stopping: DEVELOPED
 * - Regularization (general concept): INTRODUCED
 * - Training curves as diagnostic tool: DEVELOPED
 */

export function OverfittingAndRegularizationLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Overfitting and Regularization"
            description="Diagnose overfitting from training curves and apply three techniques to prevent it."
            category="Training Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Learn to read training curves to diagnose overfitting, and apply three
            regularization techniques&mdash;dropout, weight decay, and early stopping&mdash;to
            ensure your neural networks generalize instead of memorize.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Series Capstone">
            This is the final lesson in the Foundations series. After this, you
            have the complete picture of how neural networks learn.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'We cover three regularization techniques: dropout, weight decay, and early stopping. Other approaches (data augmentation, L1 regularization) come later.',
              'No PyTorch implementation—that comes in the next series',
              'No hyperparameter tuning strategies (grid search, etc.)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook — The Full Circle */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Full Circle"
            subtitle="Where it all comes together"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the very first lesson of this course&mdash;What Is Learning?&mdash;you met
              a wiggly curve that hit every data point but missed the real pattern. That was
              overfitting. Now, 16 lessons later, you have built the machinery to create
              overfitting at industrial scale.
            </p>
            <p className="text-muted-foreground">
              You know how networks compute predictions (forward pass), measure error (loss
              functions), propagate gradients (backpropagation), and update weights (optimizers).
              You even know how to keep gradients healthy in deep networks (initialization,
              batch normalization). There is one problem left, and it is the oldest one in the
              course: <strong>your network is too good at learning</strong>.
            </p>

            {/* Before/after training curves illustration */}
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="No Regularization" color="rose">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Training loss: 0.001</li>
                  <li>&bull; Validation loss: 0.45</li>
                  <li>&bull; The model memorized everything</li>
                </ul>
              </GradientCard>
              <GradientCard title="With Regularization" color="emerald">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Training loss: 0.15</li>
                  <li>&bull; Validation loss: 0.18</li>
                  <li>&bull; The model generalized</li>
                </ul>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Same architecture, same data, same optimizer. The only difference: the right
              model uses regularization. This lesson teaches you what regularization means,
              why it works, and how to apply it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Full Circle">
            Lesson 1 introduced the problem (overfitting). Lesson 17 completes
            the solution. Everything in between built the tools you need to
            understand <em>why</em> neural networks overfit and <em>how</em> to
            prevent it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Recap — Overfitting Revisited */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Overfitting Revisited"
            subtitle="Reconnecting to where we started"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In What Is Learning?, overfitting was a wiggly curve that memorized data points.
              The key insight: <strong>lower training error does not mean better
              generalization</strong>. That same principle applies to neural networks, but at
              a much larger scale.
            </p>
            <p className="text-muted-foreground">
              A neural network with enough parameters can fit <em>any</em> function&mdash;including
              one that perfectly passes through every noisy data point. This
              is <strong>capacity</strong>: the model&apos;s ability to learn arbitrarily complex
              patterns. The question is not whether the network <em>can</em> memorize the
              data, but how to prevent it from doing so.
            </p>
            <p className="text-muted-foreground">
              Remember the exam analogy? Training data = textbook (you study from it). Validation
              data = practice test (you check your understanding). Test data = real exam (evaluated
              once, at the end). Overfitting means your textbook score is 100% but your practice
              test score is 50%.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The Capacity Trap">
            You might think overfitting is only a small-dataset problem. But modern
            networks have millions or billions of parameters&mdash;they can memorize
            even large datasets. The ratio of parameters to data points matters,
            not the absolute dataset size.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 4. Explain — Reading Training Curves */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Reading Training Curves"
            subtitle="The diagnostic tool every ML practitioner uses"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the diagnostic tool: plot <strong>training loss</strong> AND{' '}
              <strong>validation loss</strong> on the same chart, over epochs.
            </p>

            {/* Annotated training curve phases */}
            <div className="rounded-lg border bg-card p-4">
              <svg viewBox="0 0 400 160" className="w-full" style={{ maxHeight: 180 }}>
                {/* Background */}
                <rect x="0" y="0" width="400" height="160" fill="#1a1a2e" rx="4" />

                {/* Axis */}
                <line x1="40" y1="135" x2="380" y2="135" stroke="#4a4a6a" strokeWidth="1" />
                <line x1="40" y1="15" x2="40" y2="135" stroke="#4a4a6a" strokeWidth="1" />
                <text x="210" y="155" textAnchor="middle" fill="#888" fontSize="9">Epoch</text>
                <text x="12" y="80" fill="#888" fontSize="9" transform="rotate(-90, 12, 80)">Loss</text>

                {/* Phase backgrounds */}
                <rect x="40" y="15" width="100" height="120" fill="#22c55e" opacity="0.04" />
                <rect x="140" y="15" width="50" height="120" fill="#f97316" opacity="0.06" />
                <rect x="190" y="15" width="190" height="120" fill="#ef4444" opacity="0.04" />

                {/* Phase labels */}
                <text x="90" y="28" textAnchor="middle" fill="#22c55e" fontSize="7" fontWeight="500">Learning</text>
                <text x="165" y="28" textAnchor="middle" fill="#f97316" fontSize="7" fontWeight="500">Sweet spot</text>
                <text x="290" y="28" textAnchor="middle" fill="#ef4444" fontSize="7" fontWeight="500">Overfitting</text>

                {/* Training loss — steadily decreasing */}
                <polyline
                  points="40,120 80,85 120,60 160,42 200,30 240,24 280,21 320,19 360,18"
                  fill="none" stroke="#3b82f6" strokeWidth="2"
                />

                {/* Validation loss — decreases then rises */}
                <polyline
                  points="40,122 80,90 120,68 155,52 170,48 190,52 230,62 270,75 310,90 350,100"
                  fill="none" stroke="#ef4444" strokeWidth="2"
                />

                {/* Sweet spot star */}
                <circle cx="170" cy="48" r="4" fill="#f97316" />
                <text x="170" y="42" textAnchor="middle" fill="#f97316" fontSize="7">best model</text>

                {/* Scissors annotation */}
                <line x1="280" y1="21" x2="280" y2="75" stroke="#eab308" strokeWidth="0.5" strokeDasharray="3 2" opacity="0.6" />
                <text x="295" y="50" fill="#eab308" fontSize="7" opacity="0.8">gap = overfitting</text>

                {/* Legend */}
                <line x1="260" y1="130" x2="280" y2="130" stroke="#3b82f6" strokeWidth="2" />
                <text x="284" y="133" fill="#3b82f6" fontSize="8">Train</text>
                <line x1="310" y1="130" x2="330" y2="130" stroke="#ef4444" strokeWidth="2" />
                <text x="334" y="133" fill="#ef4444" fontSize="8">Validation</text>
              </svg>
            </div>

            <p className="text-muted-foreground">
              Three phases tell the full story:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><strong>Learning phase:</strong> Both curves decrease together. The model is learning real patterns.</li>
              <li><strong>Sweet spot:</strong> Validation loss reaches its minimum. This is where you want to stop.</li>
              <li><strong>Overfitting phase:</strong> Training loss keeps decreasing, validation loss starts increasing. The &ldquo;scissors&rdquo; opens. The model is memorizing.</li>
            </ul>

            <p className="text-muted-foreground">
              The gap between the curves <em>is</em> overfitting, measured directly.
            </p>

            <div className="rounded-md border border-violet-500/30 bg-violet-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-violet-400">Connecting the dots:</strong>{' '}
              In Optimizers, we said &ldquo;validation loss matters more than training
              loss.&rdquo; Now you can see why: training loss always improves with more
              training. Validation loss shows you when improvement stops being real.
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Scissors Pattern">
            When training and validation loss diverge, they look like an opening
            pair of scissors. If you see the scissors opening, your model is
            overfitting. Every technique in this lesson aims to keep the scissors
            closed.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5. Check 1 — Predict and Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              You are training a network and plotting both training and validation loss.
              After epoch 50, training loss is 0.01 and validation loss is 0.03. After
              epoch 150, training loss is 0.001 and validation loss is 0.08. Is the model
              at epoch 150 better or worse than at epoch 50? Why?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  <strong>Worse.</strong> Training loss improved 10x (0.01 to 0.001) but
                  validation loss got 2.7x worse (0.03 to 0.08). The model memorized training
                  data but lost the ability to generalize. Epoch 50 was the better model
                  despite having higher training loss.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 6. Explain — Regularization (the general principle) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Is Regularization?"
            subtitle="The general principle behind all three techniques"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Regularization is any technique that <strong>constrains the model to prevent
              memorization</strong>. The idea: make it harder for the network to perfectly
              fit every training point, so it is forced to learn the underlying pattern instead.
            </p>
            <p className="text-muted-foreground">
              Remember the &ldquo;perfect fit trap&rdquo; from What Is Learning? Regularization
              is the systematic way to avoid that trap.
            </p>

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5">
              <p className="text-amber-400 font-semibold mb-2">
                Regularization increases training loss. That is the point.
              </p>
              <p className="text-sm text-muted-foreground">
                A regularized model&apos;s training loss might be 0.15 instead of 0.001.
                But its validation loss might be 0.18 instead of 0.45.
                Higher training loss, lower validation loss. Better model.
              </p>
            </div>

            <p className="text-muted-foreground">
              For the last 16 lessons, we have been minimizing training loss. That was the
              right focus for understanding the mechanics. But the real goal was always
              generalization. We will cover three techniques&mdash;each works differently, but
              all serve the same goal: better validation performance.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Reframe">
            Training loss alone is not the goal. It never was. The goal is to minimize
            the loss on data the model has <em>not</em> seen. Regularization makes this
            explicit.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 7. Explain — Dropout */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Dropout: Randomly Silence Neurons"
            subtitle="Force the network to learn robust features"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If a network memorizes, it means specific neurons have learned to respond
              to specific training examples. What if we make it impossible for the network
              to rely on any single neuron?
            </p>
            <p className="text-muted-foreground">
              During each training step, randomly set a fraction of neurons&apos; outputs
              to zero. The fraction (called the <strong>dropout rate</strong>, typically{' '}
              <InlineMath math="p = 0.5" />) means roughly half the neurons are
              &ldquo;asleep&rdquo; on any given step.
            </p>

            <div className="rounded-md border border-sky-500/30 bg-sky-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-sky-400">Training vs inference:</strong>{' '}
              Dropout ONLY happens during training. At inference time (making predictions),
              all neurons are active. The full network makes the prediction, not a crippled
              version. During inference, neuron outputs are scaled by{' '}
              <InlineMath math="(1-p)" /> to account for more neurons being active than
              during training.
            </div>

            <p className="text-muted-foreground">
              Think of it as studying for an exam by randomly covering sections of your
              notes each study session. You cannot rely on any one section being available,
              so you learn the connections between sections. On exam day (inference), you
              have access to everything.
            </p>
            <p className="text-muted-foreground">
              Each training step uses a different subset of neurons&mdash;effectively a different
              sub-network. After many steps, the network has been trained as an ensemble of
              millions of overlapping sub-networks. These sub-networks disagree on the noise
              (which is random) but agree on the real pattern (which is consistent). This is
              similar to the polling analogy from Batching and SGD: random subsets approximate the truth.
            </p>
            <p className="text-muted-foreground">
              <strong>Practical defaults:</strong>{' '}
              <InlineMath math="p = 0.5" /> for hidden layers is the classic default.{' '}
              <InlineMath math="p = 0.1" />&ndash;<InlineMath math="0.3" /> is common for
              input layers.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Permanent Damage">
            Dropout does <em>not</em> delete neurons permanently. Different neurons are
            dropped at each training step. At test time, all neurons are active. The
            network is always full-size when it matters.
          </WarningBlock>
          <TipBlock title="Noise as a Feature">
            Remember &ldquo;noise as a feature&rdquo; from Batching and SGD? Dropout adds
            noise too. Mini-batch noise helps escape sharp minima. Dropout noise prevents
            the network from becoming dependent on any single neuron.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 8. Explain — Weight Decay / L2 Regularization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Weight Decay: A Budget for Your Parameters"
            subtitle="Penalize large weights to keep functions smooth"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Another way to prevent memorization: <strong>penalize the model for using
              large weights</strong>. Large weights create sharp, sensitive functions&mdash;small
              changes in input cause big changes in output. That sensitivity is what allows
              the model to react to noise.
            </p>
            <p className="text-muted-foreground">
              Add a penalty term to the loss function:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">L2 Regularization:</p>
              <BlockMath math="L_{\text{total}} = L_{\text{data}} + \lambda \sum_i w_i^2" />
              <p className="text-xs text-muted-foreground">
                <InlineMath math="\lambda" /> (lambda) controls how strongly large weights
                are penalized. Higher <InlineMath math="\lambda" /> = stricter budget.
              </p>
            </div>

            <p className="text-muted-foreground">
              What does this do to the gradient? The extra term adds a push toward zero:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">Modified gradient:</p>
              <BlockMath math="\frac{\partial L}{\partial w} = \frac{\partial L_{\text{data}}}{\partial w} + 2\lambda w" />
              <p className="text-sm text-muted-foreground font-semibold mt-2">Modified update rule:</p>
              <BlockMath math="w_{\text{new}} = w_{\text{old}} \cdot \underbrace{(1 - \alpha \cdot 2\lambda)}_{\text{decay factor}} - \alpha \cdot \frac{\partial L_{\text{data}}}{\partial w}" />
            </div>

            <p className="text-muted-foreground">
              Look at the first term: <InlineMath math="w_{\text{old}}" /> is multiplied
              by <InlineMath math="(1 - \alpha \cdot 2\lambda)" />, a number slightly less
              than 1. Every weight shrinks a little at every step. That is why it is
              called <strong>weight decay</strong>&mdash;the weights literally decay toward zero.
            </p>

            <p className="text-muted-foreground">
              Think of <InlineMath math="\lambda" /> as a budget constraint. The model
              can use any weights it wants, but big weights cost more. A tight budget
              (high <InlineMath math="\lambda" />) forces the model to find solutions with
              smaller weights&mdash;simpler, smoother functions that focus on the dominant
              pattern rather than the noise.
            </p>

            <div className="rounded-md border border-violet-500/30 bg-violet-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-violet-400">Connection to the loss landscape:</strong>{' '}
              The L2 penalty adds a &ldquo;bowl&rdquo; centered at{' '}
              <InlineMath math="w = 0" /> to the loss landscape. This makes sharp minima
              (which require large weights) less attractive, biasing optimization toward
              the wide, smooth minima that generalize better.
            </div>

            <p className="text-muted-foreground">
              In practice, weight decay is applied through the optimizer.{' '}
              <strong>AdamW</strong> (Adam with weight decay) is the modern default&mdash;it
              is what most practitioners use. Remember from Optimizers that Adam&apos;s
              defaults are &ldquo;starting points, not universal&rdquo;? The regularization
              strength <InlineMath math="\lambda" /> is the most common thing you would tune.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Smaller Weights Help">
            Large weights create functions that are sensitive to tiny input changes.
            Overfitting IS sensitivity: the model reacts to noise because its weights
            amplify tiny variations. Keeping weights small keeps functions smooth.
          </InsightBlock>
          <ConceptBlock title="The Update Rule">
            <p className="text-sm">
              This is the same update rule from Gradient Descent, with one extra
              multiplication:
            </p>
            <p className="text-sm mt-1">
              Before: <InlineMath math="w = w - \alpha \cdot g" />
            </p>
            <p className="text-sm mt-1">
              After: <InlineMath math="w = w \cdot (1 - \alpha \cdot 2\lambda) - \alpha \cdot g" />
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 9. Explain — Early Stopping */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Early Stopping: Know When to Quit"
            subtitle="The simplest regularization technique"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The simplest regularization technique: <strong>stop training before
              overfitting starts</strong>.
            </p>
            <p className="text-muted-foreground">
              Monitor validation loss at the end of each epoch. When it stops improving for
              some number of epochs (called <strong>patience</strong>), stop training. Save
              the model weights from the epoch with the best validation loss.
            </p>

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5">
              <p className="text-amber-400 font-semibold mb-2">
                This is not giving up.
              </p>
              <p className="text-sm text-muted-foreground">
                Think of the exam analogy: you are checking your practice test score while
                studying. If your practice test score stops improving&mdash;or starts getting
                worse&mdash;continuing to study is not helping. You would be just memorizing
                the textbook answers. Stop at the moment of peak understanding.
              </p>
            </div>

            <p className="text-muted-foreground">
              <strong>Patience:</strong> Validation loss can fluctuate. One bad epoch does
              not mean overfitting has started. The patience hyperparameter (typically 5&ndash;20
              epochs) says &ldquo;wait this many epochs without improvement before stopping.&rdquo;
              If validation loss improves again within the patience window, the counter resets.
            </p>
            <p className="text-muted-foreground">
              This might feel contradictory: we have spent 16 lessons saying the ball should
              roll to the bottom of the loss landscape. The key realization is that{' '}
              <strong>training loss and validation loss are different landscapes</strong>. The
              ball can reach the bottom of the training loss landscape while climbing{' '}
              <em>up</em> the validation loss landscape. Early stopping says: watch both
              landscapes, and stop when the one that matters starts getting worse.
            </p>
            <p className="text-muted-foreground">
              <strong>Practical simplicity:</strong> Early stopping requires almost no
              additional code: save model weights when validation loss improves, stop when
              patience runs out, reload the best weights. No hyperparameters to tune on the
              model itself&mdash;just the patience value.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Fixed Epoch Count">
            Early stopping is NOT &ldquo;train for 20 epochs.&rdquo; Without monitoring
            validation loss, you are guessing&mdash;you might stop too early (underfitting)
            or too late (overfitting). The epoch at which overfitting begins depends on
            the model, data, and learning rate. You NEED the validation curve to know
            when to stop.
          </WarningBlock>
          <TipBlock title="Two Different Hills">
            The training loss landscape and the validation loss landscape are different
            hills. The ball reaches the bottom of the training hill, but the validation
            hill has a valley partway down. Continuing past it climbs back up.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Explore — RegularizationExplorer Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel
            title="Regularization Explorer"
            subtitle="See how regularization techniques affect training curves and the learned function"
          >
            <RegularizationExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Guided Experiments">
            <ol className="space-y-2 text-sm list-decimal list-inside">
              <li><strong>High capacity, no regularization, 200 epochs.</strong> Watch the scissors open&mdash;validation loss diverges while training loss approaches zero.</li>
              <li><strong>Same setup, toggle Dropout ON (p=0.5).</strong> Watch the scissors close. Notice training loss is noisier but validation loss is lower.</li>
              <li><strong>Replace dropout with Weight Decay ({'λ'}=0.01).</strong> The scissors close differently&mdash;training loss floors higher, but validation loss stays low.</li>
              <li><strong>No dropout or weight decay, enable Early Stopping (patience=10).</strong> When does it stop? Check the green dashed line.</li>
              <li><strong>Medium capacity with all three.</strong> Compare to the high-capacity unregularized model.</li>
              <li><strong>Try Low capacity.</strong> Notice that regularization makes underfitting <em>worse</em>&mdash;it is for overfitting, not underfitting.</li>
            </ol>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* 11. Elaborate — When to Use Which */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When to Use Which"
            subtitle="A menu of options, not a recipe with required ingredients"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You do not need all three. Each technique works independently. Many successful
              models use only one or two. The choice depends on the problem, dataset size,
              and network architecture.
            </p>

            <ComparisonRow
              left={{
                title: 'Dropout',
                color: 'sky',
                items: [
                  'Randomly silences neurons during training',
                  'Creates implicit ensemble of sub-networks',
                  'Best for: large networks with fully connected layers',
                  'Typical p = 0.5 (hidden), 0.1–0.3 (input)',
                ],
              }}
              right={{
                title: 'Weight Decay (L2)',
                color: 'violet',
                items: [
                  'Penalizes large weights, shrinks toward zero',
                  'Makes functions smoother, less sensitive',
                  'Best for: any model (built into AdamW)',
                  'Typical λ = 0.0001–0.01',
                ],
              }}
            />

            <GradientCard title="Early Stopping" color="emerald">
              <ul className="space-y-1 text-sm">
                <li>&bull; Monitors validation loss, stops when it plateaus</li>
                <li>&bull; Saves best model weights automatically</li>
                <li>&bull; Best for: everything (it is free)</li>
                <li>&bull; Typical patience = 5&ndash;20 epochs</li>
              </ul>
            </GradientCard>

            <div className="rounded-md border border-violet-500/30 bg-violet-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-violet-400">The priority order:</strong>{' '}
              (1) Always use early stopping&mdash;it costs nothing.{' '}
              (2) Use AdamW instead of Adam&mdash;weight decay is built in.{' '}
              (3) Add dropout if the network is large and overfitting persists.
            </div>

            <p className="text-muted-foreground">
              Over-regularizing is possible. Imagine a small network that is already
              underfitting&mdash;it cannot capture the pattern even without constraints. Adding
              heavy dropout (say p=0.5) makes it worse: you are silencing half the neurons in
              a model that already does not have enough capacity. Regularization is medicine
              for overfitting, not a supplement you take regardless.
            </p>

            <p className="text-muted-foreground">
              These three techniques, combined with everything from this module (good
              initialization, batch normalization, appropriate optimizer), form the{' '}
              <strong>modern baseline for training neural networks</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Complete Recipe">
            <ul className="space-y-1 text-sm">
              <li>&bull; Xavier/He initialization</li>
              <li>&bull; Batch normalization</li>
              <li>&bull; AdamW optimizer</li>
              <li>&bull; Dropout (if needed)</li>
              <li>&bull; Early stopping</li>
            </ul>
            <p className="text-sm mt-2">
              This is the standard starting point for almost any neural network
              training run.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 12. Check 2 — Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              A colleague shows you their training results: a large network trained for
              300 epochs with Adam (not AdamW), no dropout, no early stopping. Training
              loss is 0.0001, validation loss is 0.52. They say the model needs more layers
              because it is not performing well enough. What do you diagnose, and what
              would you recommend?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Think about the train/val gap. Is the problem capacity or regularization?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  The model is <strong>severely overfitting</strong> (train 0.0001 vs val 0.52
                  = massive gap). More layers would make it <em>worse</em> (more capacity = more
                  memorization). Recommendations:
                </p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-2">
                  <li>Switch to AdamW for weight decay</li>
                  <li>Add dropout</li>
                  <li>Add early stopping with validation monitoring</li>
                  <li>Possibly use a <em>smaller</em> network</li>
                </ul>
                <p className="text-muted-foreground">
                  The problem is not insufficient capacity&mdash;it is insufficient regularization.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 13. Summary — Series Capstone */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Overfitting = memorization, diagnosed by the train/val gap',
                description:
                  'The model memorizes training data instead of learning generalizable patterns. Plot both curves. When the gap grows, you are overfitting.',
              },
              {
                headline: 'Dropout: randomly silence neurons during training',
                description:
                  'Forces the network to learn robust features that do not depend on any single neuron. Only during training—all neurons active at inference. p=0.5 default.',
              },
              {
                headline: 'Weight decay (L2): penalize large weights',
                description:
                  'Modified update rule decays weights toward zero at every step. Keeps functions smooth. AdamW is the practical default.',
              },
              {
                headline: 'Early stopping: monitor validation loss, stop when it plateaus',
                description:
                  'The simplest and most universal technique. Use patience to handle fluctuations. Save the best model weights.',
              },
              {
                headline: 'The complete training recipe',
                description:
                  'Good initialization (Xavier/He) + batch normalization + AdamW + regularization (dropout + weight decay) + early stopping. This is the modern baseline.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting',
                authors: 'Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, 2014',
                url: 'http://jmlr.org/papers/v15/srivastava14a.html',
                note: 'The definitive paper on dropout, one of the most effective regularization techniques in deep learning.',
              },
              {
                title: 'Decoupled Weight Decay Regularization',
                authors: 'Loshchilov & Hutter, 2017',
                url: 'https://arxiv.org/abs/1711.05101',
                note: 'Introduced AdamW, fixing weight decay in Adam — now the default optimizer for most modern deep learning.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Full circle echo */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-violet-500/30 bg-violet-500/5 p-5 text-center">
            <p className="text-violet-400 font-semibold">
              In What Is Learning?, you learned that the goal is generalization, not memorization.
              In this lesson, you have learned the tools to ensure your neural networks generalize.
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              You now have the complete foundations: what learning is, how networks learn,
              and how to make sure they learn the right things.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 14. Module Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="1.3"
            title="Training Neural Networks"
            achievements={[
              'Backpropagation and the chain rule',
              'Computational graphs and automatic differentiation',
              'Mini-batch SGD and gradient noise',
              'Momentum, RMSProp, and Adam optimizers',
              'Vanishing/exploding gradients, initialization, batch norm',
              'Overfitting diagnosis and regularization (dropout, weight decay, early stopping)',
            ]}
            nextModule="2.1"
            nextTitle="PyTorch Fundamentals"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
