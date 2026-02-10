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
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

/**
 * Fashion-MNIST Project -- Lesson 3 of Module 2.3 (Practical Patterns)
 *
 * CONSOLIDATE / PROJECT lesson: the FINAL lesson in Module 2.3 AND in Series 2.
 * The student proves they can independently design, train, debug, and improve
 * a classification model by combining all Series 2 skills without step-by-step
 * scaffolding.
 *
 * New concepts: 0 (zero genuinely new concepts)
 * Minor introductions within lesson:
 *   - Fashion-MNIST as a dataset (same API as MNIST, different classes)
 *   - Per-class accuracy / class-level analysis
 *
 * Target depths:
 *   - Fashion-MNIST dataset: INTRODUCED
 *   - Independent model design: APPLIED
 *   - All Series 2 skills in combination: APPLIED
 *   - Per-class accuracy analysis: INTRODUCED
 */

export function FashionMnistProjectLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Fashion-MNIST Project"
            description="Your second end-to-end project&mdash;same tools, harder problem, less hand-holding. Prove you can design, train, debug, and improve a classifier independently."
            category="Practical Patterns"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Build a fully-connected classifier for Fashion-MNIST that reaches
            ~89&ndash;90% test accuracy. You will load the dataset, design your
            own architecture, train with GPU and checkpointing, diagnose issues
            with the debugging checklist, improve with regularization, and
            analyze per-class performance. Every tool you need has been taught.
            The challenge is putting them together independently.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="CONSOLIDATE Lesson">
            No new concepts. Every skill you need is already in your hands
            from the past ten lessons. This project tests integration and
            independent decision-making.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 1. Context + Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'Fully-connected networks only -- no convolutional layers (CNNs are Series 3)',
              'No data augmentation beyond ToTensor + Normalize',
              'No learning rate schedulers or gradient clipping',
              'No hyperparameter search strategies (grid search, Bayesian optimization)',
              'Realistic target: ~88-90% accuracy with FC layers (not state-of-the-art)',
              'The Colab notebook is the primary deliverable',
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is your second end-to-end project, and it is structured
              differently from the MNIST project. That lesson gave you complete
              code with small gaps to fill. This one gives you a baseline and a
              set of experiments&mdash;you write the code and make the decisions.
            </p>
            <p className="text-muted-foreground">
              The Colab notebook is where the real work happens. This lesson
              page is your guide: it frames the challenge, walks through the
              experiments, and gives you reference code. But the notebook is
              where you type, run, observe, and iterate.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Independence Is the Goal">
            The MNIST project proved the tools work. This project proves{' '}
            <strong>you</strong> can wield them. Same heartbeat, your
            decisions.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 2. Hook -- "Same Shape, Different Challenge" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Same Shape, Different Challenge"
            subtitle="28x28 pixels, 10 classes, same API. But look closer."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Fashion-MNIST is a drop-in replacement for MNIST. Same image size
              (28x28 grayscale), same number of classes (10), same
              torchvision API. One word changes in your loading code. It was
              designed this way on purpose&mdash;to be a harder benchmark that
              is just as easy to load.
            </p>

            <p className="text-muted-foreground">
              But look at what the 10 classes actually are:
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <GradientCard title="MNIST Classes" color="amber">
                <p className="text-sm">
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                </p>
                <p className="text-sm mt-2 text-muted-foreground">
                  Visually distinct. A 3 looks nothing like a 7 to a neural
                  network (different pixel patterns entirely).
                </p>
              </GradientCard>
              <GradientCard title="Fashion-MNIST Classes" color="blue">
                <p className="text-sm">
                  T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal,
                  Shirt, Sneaker, Bag, Ankle boot
                </p>
                <p className="text-sm mt-2 text-muted-foreground">
                  Shirt vs. Coat vs. Pullover? These look almost identical as
                  28x28 silhouettes.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Your MNIST model got ~97% accuracy. <strong>On Fashion-MNIST, the
              same architecture gets ~87%.</strong> Same tools, same training
              loop, same everything&mdash;ten points lower. That gap is the
              challenge for this project. Your job: close it as far as you can
              using the skills you already have.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Expectations">
            Do not expect 97%. State-of-the-art FC models on Fashion-MNIST
            reach ~89&ndash;90%. CNNs reach ~93&ndash;95%. The remaining gap
            is what motivates Series 3. Be proud of 89%.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 3. Fashion-MNIST: The Dataset */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Fashion-MNIST: The Dataset"
            subtitle="One word changes from MNIST"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Loading Fashion-MNIST is identical to loading MNIST. The only
              change is the class name:
            </p>

            <CodeBlock
              language="python"
              filename="load_fashion_mnist.py"
              code={`import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
])

train_data = datasets.FashionMNIST(
    './data', train=True, download=True, transform=transform
)
test_data = datasets.FashionMNIST(
    './data', train=False, download=True, transform=transform
)

# Class names for display
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print(f"Training samples: {len(train_data)}")   # 60,000
print(f"Test samples:     {len(test_data)}")     # 10,000
print(f"Image shape:      {train_data[0][0].shape}")  # [1, 28, 28]`}
            />

            <p className="text-muted-foreground">
              Before looking at any numbers, <strong>look at the data.</strong>{' '}
              Run this in your notebook to see what your model is actually
              classifying:
            </p>

            <CodeBlock
              language="python"
              filename="visualize_fashion_mnist.py"
              code={`import matplotlib.pyplot as plt

# Show a grid of samples -- one row per class
fig, axes = plt.subplots(10, 8, figsize=(10, 14))
fig.suptitle("Fashion-MNIST: 8 samples per class", fontsize=14)

for class_idx in range(10):
    # Find samples of this class
    indices = [i for i, (_, label) in enumerate(train_data) if label == class_idx]
    for col in range(8):
        ax = axes[class_idx, col]
        image, _ = train_data[indices[col]]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.axis('off')
        if col == 0:
            ax.set_title(class_names[class_idx], fontsize=8, loc='left')

plt.tight_layout()
plt.show()`}
            />

            {/* Visual preview of class confusability */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <p className="text-xs font-medium text-muted-foreground mb-3">
                The 10 Fashion-MNIST classes&mdash;spot the problem:
              </p>
              <div className="grid grid-cols-5 gap-2 text-center text-xs">
                {[
                  { name: 'T-shirt/Top', icon: '👕', group: 'hard' },
                  { name: 'Trouser', icon: '👖', group: 'easy' },
                  { name: 'Pullover', icon: '🧥', group: 'hard' },
                  { name: 'Dress', icon: '👗', group: 'easy' },
                  { name: 'Coat', icon: '🧥', group: 'hard' },
                  { name: 'Sandal', icon: '👡', group: 'easy' },
                  { name: 'Shirt', icon: '👔', group: 'hard' },
                  { name: 'Sneaker', icon: '👟', group: 'easy' },
                  { name: 'Bag', icon: '👜', group: 'easy' },
                  { name: 'Ankle Boot', icon: '🥾', group: 'easy' },
                ].map((item) => (
                  <div
                    key={item.name}
                    className={`rounded p-2 ${
                      item.group === 'hard'
                        ? 'bg-rose-500/10 border border-rose-500/30'
                        : 'bg-emerald-500/10 border border-emerald-500/30'
                    }`}
                  >
                    <div className="text-lg">{item.icon}</div>
                    <div className="text-muted-foreground mt-1">{item.name}</div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                <span className="text-rose-500">Red</span>: the confusable group.
                As 28x28 grayscale silhouettes, T-shirt, Pullover, Coat, and Shirt
                are nearly the same shape. <span className="text-emerald-500">Green</span>: visually
                distinct classes your model handles easily.
              </p>
            </div>

            <p className="text-muted-foreground">
              Compare that to MNIST, where a 3 and a 7 have completely
              different pixel layouts. This is why Fashion-MNIST is harder,
              and why your MNIST architecture will not perform as well.
              Run the visualization above in your notebook to see it
              firsthand.
            </p>

            <p className="text-muted-foreground">
              Two more things to notice. First, the normalization values are
              different from MNIST. Fashion-MNIST has a mean of ~0.286 and
              standard deviation of ~0.353 (MNIST was 0.1307 and 0.3081).
              Using the wrong normalization will not crash anything, but it
              shifts the input distribution away from what batch normalization
              expects.
            </p>

            <p className="text-muted-foreground">
              Second, some classes are visually distinct&mdash;trousers look
              nothing like bags. But shirts, coats, and pullovers are all
              upper-body garments with similar silhouettes. Sneakers and ankle
              boots are both footwear. The model must learn subtle differences
              between similar-looking categories.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same API, Different Data">
            <code className="text-xs bg-muted px-1 py-0.5 rounded">datasets.MNIST</code>{' '}
            becomes{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">datasets.FashionMNIST</code>.
            Everything else&mdash;DataLoader, transforms, batch
            iteration&mdash;is identical.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 4. The Baseline: Your MNIST Architecture */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Baseline: Your MNIST Architecture"
            subtitle="Start with what you know. See where it falls short."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Take the architecture from the MNIST project&mdash;three linear
              layers with ReLU, no regularization&mdash;and run it on
              Fashion-MNIST. This is your baseline, the thing to beat.
            </p>

            <CodeBlock
              language="python"
              filename="baseline_model.py"
              code={`import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)`}
            />

            <p className="text-muted-foreground">
              Train this for 5 epochs&mdash;the same as your MNIST
              project&mdash;and observe the results. You should see:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Test accuracy around ~87&ndash;88% (not the ~97% you got on MNIST)</li>
              <li>Training loss still decreasing at epoch 5&mdash;the model has not converged</li>
              <li>A gap between training accuracy (~90&ndash;92%) and test accuracy (~87&ndash;88%)</li>
            </ul>

            <p className="text-muted-foreground">
              That ten-point drop from MNIST is not a bug&mdash;it is the
              dataset being harder. Same model, same code, different data. This
              observation is the starting point for everything that follows.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Baseline Matters">
            Never skip the baseline. It gives you a number to beat, a training
            curve to compare against, and evidence that your pipeline works
            before you start experimenting.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 5. Check 1 -- Diagnose the Baseline */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">
              Check 1: Diagnose the Baseline
            </h3>
            <div className="space-y-4">
              <p className="text-muted-foreground text-sm">
                Before experimenting, diagnose what is happening. Use your
                debugging checklist from <em>Debugging and Visualization</em>{' '}
                and your overfitting diagnosis skills from{' '}
                <em>Overfitting and Regularization</em>.
              </p>

              <div className="space-y-3">
                <p className="text-muted-foreground text-sm">
                  <strong>(a)</strong> The loss is still decreasing at epoch 5.
                  What does this tell you?
                </p>
                <details className="group">
                  <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                    Show answer
                  </summary>
                  <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                    <p className="text-muted-foreground">
                      <strong>The model has not converged.</strong> It is still
                      learning. Train longer&mdash;5 epochs is not enough for
                      this dataset.
                    </p>
                  </div>
                </details>

                <p className="text-muted-foreground text-sm">
                  <strong>(b)</strong> Training accuracy is 92% but test
                  accuracy is 87%. What pattern is this?
                </p>
                <details className="group">
                  <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                    Show answer
                  </summary>
                  <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                    <p className="text-muted-foreground">
                      <strong>The scissors pattern&mdash;overfitting.</strong>{' '}
                      The gap between training and test accuracy means the model
                      is memorizing training data rather than generalizing. Apply
                      regularization to close the gap.
                    </p>
                  </div>
                </details>

                <p className="text-muted-foreground text-sm">
                  <strong>(c)</strong> Which debugging tool would you use to
                  check if the model&rsquo;s gradients are healthy?
                </p>
                <details className="group">
                  <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                    Show answer
                  </summary>
                  <div className="mt-2 text-sm border-t border-primary/20 pt-2">
                    <p className="text-muted-foreground">
                      <strong>The gradient magnitude check</strong>{' '}
                      (<code className="text-xs bg-muted px-1 py-0.5 rounded">log_gradient_norms()</code>)
                      from <em>Debugging and Visualization</em>. Check per-layer gradient
                      norms after a few batches. Healthy: balanced magnitudes
                      across layers. Unhealthy: orders-of-magnitude mismatch.
                    </p>
                  </div>
                </details>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Diagnose Before You Experiment">
            The debugging checklist is a habit, not a last resort. Before
            changing anything, understand what is happening. The baseline
            tells you <strong>where</strong> to focus your experiments.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 6. Experiment 1: Train Longer */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Experiment 1: Train Longer"
            subtitle="The simplest thing you can try"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The baseline&rsquo;s loss was still decreasing at epoch 5. The
              simplest experiment: give it more time. Train for 20 epochs
              instead of 5.
            </p>

            <p className="text-muted-foreground">
              You should see accuracy improve by 1&ndash;2 points
              (88&ndash;89%), then plateau. The loss curve flattens&mdash;the
              model has reached the limit of what this architecture can learn
              without regularization. More epochs alone are not enough.
            </p>

            <p className="text-muted-foreground">
              <strong>If accuracy plateaus for several epochs, that does not
              mean your code is broken.</strong> Use the debugging
              checklist: run the gradient check, verify loss is still
              decreasing (even slowly), confirm no NaNs. If the checklist
              says the code is healthy, the bottleneck is the model
              architecture&mdash;not a bug. The fix is a different
              architecture or more regularization, not a code change.
            </p>

            <p className="text-muted-foreground">
              But notice: the train/test gap grows wider with more epochs. By
              epoch 20, training accuracy may be 94&ndash;95% while test
              accuracy sits at 88&ndash;89%. The scissors are opening. This is
              your signal that the next experiment should be regularization,
              not more training time.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Low Activation Energy">
            Start with the easiest experiment. Changing{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">num_epochs = 5</code>{' '}
            to{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">num_epochs = 20</code>{' '}
            is a one-line change. Observe the result before trying anything
            more complex.
          </TipBlock>
          <WarningBlock title="Stuck ≠ Broken">
            A plateau means the model has reached its capacity, not that the
            code has a bug. The debugging checklist confirms which one it is.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 7. Experiment 2: Add Regularization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Experiment 2: Add Regularization"
            subtitle="Close the scissors with the tools you already have"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Apply the regularization toolkit from the{' '}
              <em>MNIST Project</em> and{' '}
              <em>Overfitting and Regularization</em>: batch normalization,
              dropout, and weight decay. Use the ordering pattern you learned:
              Linear &rarr; BatchNorm &rarr; ReLU &rarr; Dropout.
            </p>

            <CodeBlock
              language="python"
              filename="improved_model.py"
              code={`class ImprovedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 10),  # output layer: no activation, no dropout, no BN
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

# Use AdamW for built-in weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)`}
            />

            <p className="text-muted-foreground">
              Train for 20 epochs with early stopping (patience of 5). Use{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.train()</code>{' '}
              and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.eval()</code>{' '}
              correctly&mdash;dropout and batch norm behave differently in each
              mode. You practiced this in the MNIST project.
            </p>

            <p className="text-muted-foreground">
              You should see test accuracy climb to ~89&ndash;90%. The scissors
              pattern closes: training accuracy may actually go <em>down</em>{' '}
              compared to the unregularized model, but test accuracy goes{' '}
              <em>up</em>. Remember from <em>Overfitting and Regularization</em>:{' '}
              <strong>regularization increases training loss&mdash;that is the
              point.</strong> It prevents memorization, which improves
              generalization.
            </p>

            <ComparisonRow
              left={{
                title: 'Baseline (20 epochs, no regularization)',
                color: 'amber',
                items: [
                  'Train accuracy: ~94-95%',
                  'Test accuracy: ~88-89%',
                  'Gap: ~6 points (scissors open)',
                  'Model memorizes training data',
                ],
              }}
              right={{
                title: 'Improved (20 epochs, with regularization)',
                color: 'emerald',
                items: [
                  'Train accuracy: ~91-92%',
                  'Test accuracy: ~89-90%',
                  'Gap: ~2 points (scissors closed)',
                  'Model generalizes better',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="train() and eval() Are Critical">
            Forgetting{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">model.eval()</code>{' '}
            before evaluation means dropout is still active and batch norm
            uses batch statistics instead of running statistics. Your test
            accuracy will be wrong. This is not optional.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 8. Experiment 3: Architecture Decisions */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Experiment 3: Architecture Decisions"
            subtitle="Now you make the choices"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The regularized model is your new baseline. Can you do better
              with a different architecture? This is where the project becomes
              truly independent. Here are experiments to try&mdash;the lesson
              does not tell you which one works best:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>More capacity:</strong>{' '}
                784 &rarr; 512 &rarr; 256 &rarr; 128 &rarr; 10
                (wider first layer)
              </li>
              <li>
                <strong>Deeper:</strong>{' '}
                784 &rarr; 256 &rarr; 256 &rarr; 128 &rarr; 64 &rarr; 10
                (more layers)
              </li>
              <li>
                <strong>Different dropout:</strong> Try p=0.5 instead of p=0.3.
                Does more regularization help or hurt?
              </li>
              <li>
                <strong>Different weight decay:</strong> Try 0.001 instead of
                0.01. Lighter regularization.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Use TensorBoard (or printed metrics) to compare runs. The
              training curves are your diagnostic tool&mdash;the same skills
              from <em>Debugging and Visualization</em>. Watch the scissors pattern.
              Watch where loss plateaus.
            </p>

            <div className="rounded-lg border border-rose-500/20 bg-rose-500/5 p-4">
              <p className="text-sm font-semibold text-rose-400 mb-2">
                Negative example to try
              </p>
              <p className="text-sm text-muted-foreground">
                Build a very large model (784 &rarr; 1024 &rarr; 512 &rarr; 256
                &rarr; 10) with <strong>no regularization</strong>. Watch what
                happens: training accuracy climbs to 96%+ while test accuracy
                plateaus at ~88%. The scissors open wide. Adding more neurons
                without regularization makes overfitting worse, not better. You
                should be able to diagnose this instantly.
              </p>
            </div>

            <p className="text-muted-foreground">
              <strong>There is no single right answer.</strong> ML
              experimentation is about trying things, observing results, and
              making decisions based on evidence. If your accuracy improves by
              0.5%, that is real progress. If it does not, you learned something
              about the limits of your current approach.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Experimentation Is the Skill">
            The specific architecture matters less than the process: try
            something, observe, diagnose, adapt. This workflow is what you
            carry forward to every future project.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 9. Understanding Your Model: Per-Class Accuracy */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Understanding Your Model: Per-Class Accuracy"
            subtitle="A single accuracy number hides important details"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your model is 89% accurate overall. But is it 89% accurate on
              every class? Almost certainly not. Some classes are easy and some
              are hard&mdash;a single number hides this.
            </p>

            <CodeBlock
              language="python"
              filename="per_class_accuracy.py"
              code={`def per_class_accuracy(model, test_loader, device, class_names):
    """Compute accuracy for each class separately."""
    model.eval()
    correct = torch.zeros(10)
    total = torch.zeros(10)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(10):
                mask = labels == i
                total[i] += mask.sum().item()
                correct[i] += (preds[mask] == labels[mask]).sum().item()

    print("\\nPer-class accuracy:")
    print("-" * 35)
    for i in range(10):
        acc = 100 * correct[i] / total[i]
        bar = "█" * int(acc / 2.5)
        print(f"{class_names[i]:>12s}: {acc:5.1f}%  {bar}")

per_class_accuracy(model, test_loader, device, class_names)`}
            />

            <p className="text-muted-foreground">
              You will see a pattern like this:
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <GradientCard title="Easy Classes (95%+)" color="emerald">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Trouser&mdash;unique silhouette</li>
                  <li>&bull; Sneaker&mdash;distinct shape</li>
                  <li>&bull; Bag&mdash;nothing else looks like it</li>
                  <li>&bull; Ankle boot&mdash;different from sneaker</li>
                </ul>
              </GradientCard>
              <GradientCard title="Hard Classes (75-85%)" color="amber">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Shirt&mdash;confused with coat and pullover</li>
                  <li>&bull; Coat&mdash;confused with pullover and shirt</li>
                  <li>&bull; Pullover&mdash;confused with coat and shirt</li>
                  <li>&bull; T-shirt/top&mdash;confused with shirt and dress</li>
                </ul>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Your model is not uniformly 89% accurate. It is excellent at some
              classes and mediocre at others. The hard classes are the ones that
              look similar to each other as 28x28 silhouettes. This makes
              visual sense&mdash;if you squint at a tiny grayscale image of a
              shirt and a coat, they look nearly identical.
            </p>

            <p className="text-muted-foreground">
              This analysis tells you something important: the remaining errors
              are not random. They are concentrated in visually similar
              categories. An FC network, which flattens the image to a
              784-element vector, loses all spatial structure. It cannot learn
              that sleeves are a distinguishing feature between shirts and
              vests. That is the kind of insight that motivates convolutional
              networks in Series 3.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Beyond a Single Number">
            A single accuracy metric hides important details about model
            behavior. Per-class analysis tells you <strong>where</strong>{' '}
            your model struggles and <strong>why</strong>&mdash;a richer
            diagnostic than overall accuracy alone.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 10. The Complete Pipeline */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Pipeline"
            subtitle="Everything you have built, in one place"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your notebook should contain a complete, production-ready
              pipeline. Every piece comes from a previous lesson:
            </p>

            <CodeBlock
              language="python"
              filename="complete_pipeline.py"
              code={`# 1. Device detection (GPU Training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data loading with transforms (Datasets and DataLoaders)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
train_data = datasets.FashionMNIST('./data', train=True,
                                    download=True, transform=transform)
test_data = datasets.FashionMNIST('./data', train=False,
                                   download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

# 3. Model with regularization (nn.Module + MNIST Project)
model = ImprovedModel().to(device)

# 4. Optimizer with weight decay (Training Loop + Overfitting)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                               weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 5. Training loop with GPU + checkpointing + early stopping
#    (Training Loop + Saving & Loading + GPU Training)
best_acc = 0.0
patience_counter = 0
patience = 5

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 6. Evaluation (MNIST Project)
    test_acc = evaluate(model, test_loader, device)

    # 7. Checkpointing + early stopping (Saving & Loading)
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 8. Restore best model and analyze (Saving & Loading + this lesson)
model.load_state_dict(torch.load('best_model.pth',
                                  map_location=device))
per_class_accuracy(model, test_loader, device, class_names)`}
            />

            <p className="text-muted-foreground">
              Every line in this pipeline connects to a lesson you have
              already completed. Device detection from <em>GPU Training</em>.
              Data loading from <em>Datasets and DataLoaders</em>. The model
              from <em>nn.Module</em> and the <em>MNIST Project</em>. The
              training loop from <em>The Training Loop</em>. Checkpointing
              from <em>Saving, Loading, and Checkpoints</em>. Early stopping
              from <em>Overfitting and Regularization</em>. Per-class analysis
              from this lesson.
            </p>

            <p className="text-muted-foreground">
              <strong>Optional: add mixed precision.</strong> If you are
              training on GPU, you can wrap the forward pass in{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">autocast</code>{' '}
              and use a{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">GradScaler</code>{' '}
              for faster training&mdash;the same pattern from{' '}
              <em>GPU Training</em>:
            </p>

            <CodeBlock
              language="python"
              filename="optional_mixed_precision.py"
              code={`# Optional: mixed precision for faster GPU training
scaler = torch.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()`}
            />

            <p className="text-muted-foreground">
              Fashion-MNIST is small enough that the speedup may be modest,
              but it is a good habit to practice. Try it as a stretch
              goal&mdash;compare training time with and without.
            </p>

            <p className="text-muted-foreground">
              <strong>This pipeline carries forward to every future
              project.</strong> The specifics change&mdash;different
              architectures, different datasets, different
              hyperparameters&mdash;but the structure is the same. You will
              write this pattern hundreds of times.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Heartbeat">
            Count the lessons this pipeline touches: Tensors, nn.Module,
            Training Loop, Datasets and DataLoaders, MNIST Project,
            Debugging, Saving and Loading, GPU Training (including mixed
            precision), Overfitting and Regularization. Nine lessons, one
            pipeline.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 11. Check 2 -- Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">
              Check 2: Transfer Question
            </h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                You are starting a new project on a dataset with 50 classes
                and 100x100 color images. You have not learned CNNs yet. How
                would you approach this using only the tools you have? What
                would you try first, and how would you know if your model is
                working?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      <strong>Flatten the images to 30,000-element vectors</strong>{' '}
                      (100 &times; 100 &times; 3 channels). Start with a simple
                      FC model: 30000 &rarr; 512 &rarr; 256 &rarr; 50. Use
                      cross-entropy loss with 50 output units. Train on GPU
                      (the input is large enough that GPU will help). Use the
                      debugging checklist: torchinfo to verify shapes, gradient
                      check to verify health, training curves to monitor
                      progress.
                    </p>
                    <p>
                      Monitor <strong>per-class accuracy</strong> to identify
                      which of the 50 classes are easy and which are hard. Use
                      early stopping and regularization (dropout, batch norm,
                      weight decay). The accuracy will likely be lower than
                      Fashion-MNIST because the input is much larger and the
                      flattening loses even more spatial structure&mdash;but it
                      is a valid starting point.
                    </p>
                    <p>
                      The workflow is identical: load data, build model, train,
                      diagnose, improve. Same heartbeat, different instruments.
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* Colab Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="The notebook is the deliverable"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The Colab notebook is structured with decreasing scaffolding:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>(Provided)</strong> Data loading&mdash;Fashion-MNIST
                  API is identical to MNIST, no need to reinvent this.
                </li>
                <li>
                  <strong>(Provided)</strong> Baseline model&mdash;run it,
                  observe results, diagnose the gap.
                </li>
                <li>
                  <strong>(Lightly scaffolded)</strong> Experimentation&mdash;
                  structure is given, you write the code. Hints in collapsible
                  cells.
                </li>
                <li>
                  <strong>(Partially guided)</strong> Per-class
                  analysis&mdash;computation provided, you interpret the
                  results.
                </li>
                <li>
                  <strong>(Independent)</strong> Full pipeline&mdash;put it all
                  together with GPU, checkpointing, early stopping, and your
                  best architecture.
                </li>
              </ol>
              <p className="text-sm text-muted-foreground">
                <strong>Stretch goals:</strong> confusion matrix visualization,
                TensorBoard comparison of 3+ architectures, finding the best
                model you can build with FC layers.
              </p>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-3-3-fashion-mnist-project.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Sections 1&ndash;2 are provided. Sections 3&ndash;4 are lightly
                scaffolded with hints. Section 5 is fully independent.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="GPU Recommended">
            Fashion-MNIST trains fast on CPU, but GPU will help when you
            are running many experiments. In Colab: Runtime &rarr; Change
            runtime type &rarr; T4 GPU.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 12. Summary -- What You Can Do Now */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'You can load, train, debug, and improve a classifier independently',
                description:
                  'From data loading through per-class analysis, every step of the pipeline is a skill you own. No scaffolding needed.',
              },
              {
                headline: 'Same heartbeat, harder problem',
                description:
                  'Fashion-MNIST uses the exact same tools and loop structure as MNIST. The difference is the dataset, not the code. Your pipeline transfers.',
              },
              {
                headline: 'Regularization closes the scissors',
                description:
                  'Dropout, batch normalization, and weight decay improve generalization at the cost of training accuracy. That tradeoff is the point.',
              },
              {
                headline: 'Per-class accuracy reveals where your model struggles',
                description:
                  'A single accuracy number hides important structure. Easy classes (trousers, bags) vs hard classes (shirts, coats) tells you what the model cannot distinguish.',
              },
              {
                headline: 'The complete training recipe carries forward',
                description:
                  'Device detection, data loading, regularized model, training with checkpointing and early stopping, per-class evaluation. This is the pattern for every future project.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Emotional payoff -- full circle echo */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-violet-500/30 bg-violet-500/5 p-5 text-center">
            <p className="text-violet-400 font-semibold">
              You did not follow a tutorial. You made decisions, observed
              results, and adapted. That is machine learning.
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              Every lesson from Tensors through this project built one piece of
              your workflow. Now the workflow is yours.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 13. What Comes Next -- Series 3 Preview */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Comes Next"
            subtitle="Your FC model is good. CNNs are the tool for the next level."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your best FC model on Fashion-MNIST tops out around
              89&ndash;90%. That is a strong result&mdash;you should be proud
              of it. But there is a ceiling. An FC network flattens each
              28x28 image into a 784-element vector, destroying all spatial
              structure. It cannot learn that sleeves, collars, and necklines
              are the features that distinguish shirts from coats.
            </p>
            <p className="text-muted-foreground">
              Convolutional neural networks (CNNs) reach 93&ndash;95% on
              Fashion-MNIST. They do this by preserving spatial structure
              and learning local patterns (edges, textures, shapes) that FC
              layers cannot see. The ~5% gap between your best FC model and
              a CNN is what Series 3 exists to close.
            </p>
            <p className="text-muted-foreground">
              That is expansion, not correction. Your FC pipeline is correct
              and complete. CNNs add a new kind of layer that extracts spatial
              features&mdash;everything else (training loop, regularization,
              debugging, checkpointing) stays the same. Same heartbeat, new
              instruments.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The 5% Gap">
            FC: ~89&ndash;90%. CNN: ~93&ndash;95%. That gap comes from
            spatial features that FC layers cannot learn. Series 3 gives you
            the tool to close it.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Module 2.3 Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="2.3"
            title="Practical Patterns"
            achievements={[
              'Saving and loading model state with state_dict and torch.save/load',
              'Checkpoint pattern: model + optimizer + epoch + loss in one file',
              'Resume interrupted training and early stopping with checkpoints',
              'Device-aware training loops that run on CPU or GPU',
              'Mixed precision with autocast and GradScaler',
              'Independent end-to-end project: Fashion-MNIST classifier',
            ]}
            nextModule="3.1"
            nextTitle="Convolutional Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Series 2 Complete -- custom celebration block */}
      <Row>
        <Row.Content>
          <div className="rounded-lg bg-gradient-to-br from-violet-500/15 via-primary/10 to-emerald-500/10 border border-violet-500/30 p-6">
            <h3 className="font-semibold text-lg text-violet-400 mb-2">
              Series 2 Complete: PyTorch
            </h3>
            <p className="text-muted-foreground mb-4">
              You started Series 2 with a single tensor. Ten lessons later,
              you have a complete, production-ready training pipeline:
            </p>
            <div className="grid gap-3 md:grid-cols-2 mb-4">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Module 2.1 &mdash; PyTorch Core
                </p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-2 text-sm">
                  <li>Tensors, autograd, nn.Module, training loop</li>
                </ul>
              </div>
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Module 2.2 &mdash; Real Data
                </p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-2 text-sm">
                  <li>Datasets, DataLoaders, MNIST project, debugging</li>
                </ul>
              </div>
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Module 2.3 &mdash; Practical Patterns
                </p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-2 text-sm">
                  <li>Saving/loading, GPU training, Fashion-MNIST project</li>
                </ul>
              </div>
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  What you proved
                </p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-2 text-sm">
                  <li>You can build, train, debug, save, and deploy a model independently</li>
                </ul>
              </div>
            </div>
            <p className="text-muted-foreground">
              Next up: <strong>Series 3 &mdash; Convolutional Neural Networks</strong>.
              Your training pipeline stays the same. The architecture evolves.
            </p>
          </div>
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
