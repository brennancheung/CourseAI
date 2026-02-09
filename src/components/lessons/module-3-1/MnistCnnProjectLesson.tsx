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
  ConceptBlock,
  SummaryBlock,
  NextStepBlock,
  GradientCard,
  ComparisonRow,
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { cn } from '@/lib/utils'
import { ExternalLink } from 'lucide-react'

/**
 * MNIST CNN Project
 *
 * Third and final lesson in Module 3.1 (Convolutions).
 * CONSOLIDATE lesson: no new concepts. Student assembles everything
 * from Lessons 1 and 2 into a working CNN, trains it on MNIST, and
 * compares it to the dense network they built in Series 2.
 *
 * Depth upgrades:
 * - nn.Conv2d / nn.MaxPool2d: INTRODUCED -> APPLIED
 * - Conv-pool-fc architecture: DEVELOPED -> APPLIED
 * - CNN vs dense comparison: NEW at DEVELOPED
 *
 * Previous: Building a CNN (pooling, stride, padding, conv-pool-fc pattern)
 * Next: CNN architecture evolution (future module)
 */

// ---------------------------------------------------------------------------
// Dimension tracking stage row (reused from BuildingACnnLesson pattern)
// ---------------------------------------------------------------------------

function StageRow({
  label,
  shape,
  color,
  annotation,
}: {
  label: string
  shape: string
  color: string
  annotation?: string
}) {
  const dotColor = stageRowDotColor(color)

  return (
    <div className="flex items-center gap-3 text-sm">
      <div className={cn('w-2 h-2 rounded-full flex-shrink-0', dotColor)} />
      <span className="text-muted-foreground w-44 truncate">{label}</span>
      <span className="font-semibold text-foreground w-24">{shape}</span>
      {annotation && (
        <span className="text-xs text-muted-foreground/70">{annotation}</span>
      )}
    </div>
  )
}

function stageRowDotColor(color: string): string {
  if (color === 'violet') return 'bg-violet-400'
  if (color === 'sky') return 'bg-sky-400'
  if (color === 'amber') return 'bg-amber-400'
  if (color === 'emerald') return 'bg-emerald-400'
  return 'bg-zinc-400'
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function MnistCnnProjectLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="MNIST CNN Project"
            description="Build a CNN for MNIST in PyTorch, train it, and see it beat the dense network you built earlier."
            category="Convolutions"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 1: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Build and train a CNN for MNIST using the architecture from Building
            a CNN, then compare its accuracy and parameter count to the dense
            network from your PyTorch project.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You can compute convolutions, trace data through a conv-pool-fc
            architecture, and read nn.Conv2d code. You have also built and
            trained a dense MNIST network in PyTorch. This lesson connects those
            two experiences.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Implement the CNN architecture you traced in Building a CNN',
              'Train it on MNIST using the same training loop from your PyTorch project',
              'Compare accuracy and parameter count to the dense network',
              'NOT: introducing new concepts, layers, or techniques',
              'NOT: hyperparameter tuning, batch norm, data augmentation, or advanced architectures',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Module Arc">
            In What Convolutions Compute you learned the building block. In
            Building a CNN you assembled the architecture. Now you build one,
            train it, and prove that architecture matters.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Recap -- The Architecture You Will Build
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Architecture You Will Build"
            subtitle="A 30-second refresher from Building a CNN"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the exact architecture you traced step by step in Building
              a CNN. Nothing new here&mdash;just priming your memory so the code
              you are about to write maps directly to this pipeline:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="space-y-2 font-mono text-sm">
                <StageRow label="1. Input" shape="28x28x1" color="zinc" annotation="Grayscale MNIST digit" />
                <StageRow label="2. Conv(32, 3x3, pad=1)" shape="28x28x32" color="violet" annotation="32 filters, same spatial" />
                <StageRow label="3. ReLU" shape="28x28x32" color="violet" annotation="Same ReLU you know" />
                <StageRow label="4. MaxPool(2x2)" shape="14x14x32" color="sky" annotation="Spatial halved" />
                <StageRow label="5. Conv(64, 3x3, pad=1)" shape="14x14x64" color="violet" annotation="64 filters" />
                <StageRow label="6. ReLU" shape="14x14x64" color="violet" />
                <StageRow label="7. MaxPool(2x2)" shape="7x7x64" color="sky" annotation="Spatial halved again" />
                <StageRow label="8. Flatten" shape="3136" color="amber" annotation="7 x 7 x 64" />
                <StageRow label="9. Linear(128) + ReLU" shape="128" color="emerald" />
                <StageRow label="10. Linear(10)" shape="10" color="emerald" annotation="One per digit" />
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              You already know every layer here. The conv layers detect spatial
              features, pooling shrinks dimensions, flatten transitions from
              spatial to flat, and the linear layers classify. The training
              loop&mdash;forward, loss, backward, update&mdash;is identical to
              what you used for the dense network.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Nothing New">
            This is a CONSOLIDATE lesson. Every concept here was taught in a
            prior lesson. The only new skill is assembly&mdash;connecting pieces
            you already understand.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Hook -- The Challenge
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Challenge"
            subtitle="Can architecture alone make a difference?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your dense network from your PyTorch project got around{' '}
              <strong>97% accuracy</strong> on MNIST. That felt impressive&mdash;and
              it is. But you now know what that network does with an image: it
              flattens 28x28 pixels into a 784-element vector and throws away all
              spatial structure.
            </p>
            <p className="text-muted-foreground">
              <strong>Can this CNN do better?</strong> It uses the same data, the
              same optimizer, the same number of training epochs. The{' '}
              <em>only</em> thing that changes is the architecture. Same
              experiment, one variable. Time to find out.
            </p>
            <p className="text-muted-foreground">
              And here is the twist: the simplest possible CNN will beat your
              dense network. <strong>No tricks required.</strong> No batch
              normalization, no dropout, no learning rate scheduling. Two conv
              layers, two pool layers, two linear layers. That is it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="A Controlled Experiment">
            Same data. Same optimizer. Same epochs. Same loss function.
            Architecture is the <strong>only</strong> variable. If the CNN wins,
            the architecture is why.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: API Recap (bridge from INTRODUCED to APPLIED)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="API Recap"
            subtitle="From reading to writing"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Building a CNN, you read nn.Conv2d and nn.MaxPool2d code and
              identified what each argument controls. Now you will type these
              lines yourself. The arguments map directly to the dimension formula
              you already know:
            </p>
            <CodeBlock
              code={`# Conv2d(in_channels, out_channels, kernel_size, padding)
nn.Conv2d(1, 32, kernel_size=3, padding=1)
# 1 input channel (grayscale)
# 32 filters (output channels)
# 3x3 filter size
# padding=1 preserves spatial dimensions

nn.MaxPool2d(kernel_size=2)
# 2x2 window, stride defaults to kernel_size (2)
# Halves spatial dimensions: 28->14, 14->7

nn.Flatten()
# Collapses spatial dimensions into a flat vector
# 7x7x64 becomes 3136`}
              language="python"
              filename="cnn_layers.py"
            />
            <p className="text-sm text-muted-foreground">
              Nothing new. Same arguments you identified in Building a CNN, same
              formula for output sizes. ReLU is the same activation function you
              have used before&mdash;it applies independently to each value in
              the feature map, adding nonlinearity after the linear convolution.
              The only difference: now you write these layers into a class
              definition.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="ReLU in the CNN">
            Same ReLU you have used since Foundations. It applies to every value
            in the feature map independently&mdash;no change to how activation
            functions work. Conv2d computes a linear operation (weighted sum);
            ReLU adds nonlinearity.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Build the CNN (Colab Notebook)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build the CNN"
            subtitle="Hands-on in a Colab notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The Colab notebook is scaffolded so you focus on the CNN-specific
              code without boilerplate friction. Here is what you will find:
            </p>
            <div className="space-y-3">
              <GradientCard title="Notebook Structure" color="blue">
                <ol className="space-y-2 text-sm list-decimal list-inside">
                  <li>
                    <strong>Setup + data loading</strong>&mdash;provided complete.
                    Same MNIST loading from your PyTorch project.
                  </li>
                  <li>
                    <strong>Dense network baseline</strong>&mdash;provided as
                    reference. Your dense network, pre-trained or quickly
                    retrained (1-2 minutes).
                  </li>
                  <li>
                    <strong>CNN class skeleton</strong>&mdash;you fill this in.
                    The class has <code className="text-xs bg-black/20 px-1 rounded">TODO</code>{' '}
                    comments where you write the layers in{' '}
                    <code className="text-xs bg-black/20 px-1 rounded">__init__</code> and the
                    forward pass in{' '}
                    <code className="text-xs bg-black/20 px-1 rounded">forward()</code>.
                  </li>
                  <li>
                    <strong>Dimension verification</strong>&mdash;a test forward
                    pass with random input to check your shapes match the
                    architecture diagram.
                  </li>
                  <li>
                    <strong>Training loop</strong>&mdash;provided complete.
                    Identical to your dense network training loop.
                  </li>
                  <li>
                    <strong>Evaluation + comparison</strong>&mdash;test accuracy,
                    parameter counts, training curves for both models.
                  </li>
                </ol>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Your task: fill in the CNN class. The architecture is the one you
              traced above&mdash;conv-relu-pool-conv-relu-pool-flatten-fc-relu-fc.
              Everything else is handled.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why Scaffolded?">
            This is a CONSOLIDATE lesson. You should spend cognitive effort on
            the CNN architecture&mdash;the part that is new&mdash;not on data
            loading and training loops you have already written. The scaffolding
            lets you focus.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* CNN class skeleton */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the skeleton you will complete. Each{' '}
              <code className="text-xs bg-black/20 px-1 rounded">TODO</code>{' '}
              corresponds to one layer from the architecture diagram:
            </p>
            <CodeBlock
              code={`class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: First conv block
        # Conv2d: 1 -> 32 channels, 3x3 filter, padding=1
        # MaxPool2d: 2x2

        # TODO: Second conv block
        # Conv2d: 32 -> 64 channels, 3x3 filter, padding=1
        # MaxPool2d: 2x2

        # TODO: Classifier
        # Flatten: 7*7*64 = 3136
        # Linear: 3136 -> 128
        # Linear: 128 -> 10

    def forward(self, x):
        # TODO: Pass x through the layers
        # Conv1 -> ReLU -> Pool1
        # Conv2 -> ReLU -> Pool2
        # Flatten -> FC1 -> ReLU -> FC2
        return x`}
              language="python"
              filename="mnist_cnn.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same nn.Module Pattern">
            This is the same nn.Module subclass pattern you used for your dense
            network. Define layers in __init__, connect them in forward(). The
            only difference is <em>which</em> layers you use.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Colab link */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Open the notebook and build your CNN. The skeleton is waiting for
                you.
              </p>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/3-1-3-mnist-cnn-project.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                The notebook includes the data loading, dense network baseline,
                CNN skeleton, training loop, and comparison code. You fill in the
                CNN class.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: Check 1 -- Dimension Verification
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Dimension Verification"
            subtitle="Predict and verify before training"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before running the training loop, verify your architecture is
              correct. Trace through your model and predict the shape at each
              stage:
            </p>
            <GradientCard title="Predict the Shapes" color="emerald">
              <div className="space-y-3 text-sm">
                <p>
                  For a single MNIST image (shape: 1x1x28x28, batch of 1):
                </p>
                <ol className="space-y-1 list-decimal list-inside">
                  <li>What is the shape after the first Conv2d + ReLU?</li>
                  <li>After the first MaxPool2d?</li>
                  <li>After the second Conv2d + ReLU?</li>
                  <li>After the second MaxPool2d?</li>
                  <li>After Flatten?</li>
                  <li>After the final Linear?</li>
                </ol>
                <details className="mt-3">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal answer
                  </summary>
                  <div className="mt-2 space-y-2 font-mono text-xs">
                    <p>1. Conv2d(1,32,3,pad=1) + ReLU: <strong>1x32x28x28</strong></p>
                    <p>2. MaxPool2d(2): <strong>1x32x14x14</strong></p>
                    <p>3. Conv2d(32,64,3,pad=1) + ReLU: <strong>1x64x14x14</strong></p>
                    <p>4. MaxPool2d(2): <strong>1x64x7x7</strong></p>
                    <p>5. Flatten: <strong>1x3136</strong></p>
                    <p>6. Linear(10): <strong>1x10</strong></p>
                  </div>
                </details>
              </div>
            </GradientCard>
            <p className="text-sm text-muted-foreground">
              In the notebook, verify by running{' '}
              <code className="text-xs bg-muted px-1 rounded">
                model(torch.randn(1, 1, 28, 28))
              </code>{' '}
              and checking the output shape. If it is not{' '}
              <code className="text-xs bg-muted px-1 rounded">[1, 10]</code>,
              something in your architecture is off.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Debugging Shapes">
            If the shapes do not match, add print statements inside forward()
            after each layer. The dimension formula from Building a CNN tells you
            exactly what to expect. Most errors come from forgetting padding or
            getting in_channels wrong.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Train and Compare
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Train and Compare"
            subtitle="The moment of truth"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The training loop in the notebook is identical to the one you wrote
              for the dense network. Same optimizer (Adam), same loss function
              (cross-entropy), same number of epochs (5). Run the training cell
              and watch the loss decrease.
            </p>
            <p className="text-muted-foreground">
              Then compare. The notebook computes three things for both models:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><strong>Test accuracy</strong>&mdash;how many digits each model classifies correctly</li>
              <li><strong>Parameter count</strong>&mdash;how many learnable numbers each model has</li>
              <li><strong>Training curves</strong>&mdash;loss vs epoch for both models on the same plot</li>
            </ul>
            <p className="text-sm text-muted-foreground mt-2">
              Look at the training curves you plotted in the notebook&mdash;the
              CNN&apos;s loss drops faster and its accuracy plateaus higher.
              That gap is the architecture advantage made visible.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Comparison section */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is what you should see (your exact numbers may vary slightly):
            </p>
            <ComparisonRow
              left={{
                title: 'Dense Network',
                color: 'amber',
                items: [
                  'Accuracy: ~97%',
                  'Flattens 28x28 to 784 raw pixel values',
                  'Feature extraction (first layer): ~100K params',
                  'Total: ~110K parameters',
                ],
              }}
              right={{
                title: 'CNN',
                color: 'emerald',
                items: [
                  'Accuracy: ~99%+',
                  'Preserves spatial structure through conv layers',
                  'Feature extraction (conv stack): ~19K params',
                  'Total: ~421K parameters',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Surprise">
            The CNN&apos;s conv layers do more with <strong>far less</strong>.
            18K parameters in the conv stack outperform 100K parameters in the
            dense first layer at spatial feature extraction. Weight sharing
            is the reason.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Parameter count code */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You can verify the parameter counts yourself in the notebook:
            </p>
            <CodeBlock
              code={`# Count parameters for each model
dense_params = sum(p.numel() for p in dense_model.parameters())
cnn_params = sum(p.numel() for p in cnn_model.parameters())

print(f"Dense network: {dense_params:,} parameters")
print(f"CNN:           {cnn_params:,} parameters")

# Dense network: ~110,000 parameters
# CNN:           ~421,000 parameters
# The CNN has MORE total parameters (the FC layers add bulk),
# but its conv feature extraction uses far fewer than the
# dense network's first layer alone.`}
              language="python"
              filename="compare.py"
            />
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Check 2 -- Explain the Difference
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Explain the Difference"
            subtitle="Why does the CNN win?"
          />
          <GradientCard title="Explain It Back" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                Before reading on, answer this question in your own words:
              </p>
              <p className="font-medium">
                &ldquo;The CNN&apos;s conv layers use far fewer parameters than
                the dense network&apos;s first layer, yet extract spatial
                features more effectively. Why?&rdquo;
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Key ideas your answer should touch on
                </summary>
                <div className="mt-2 space-y-2">
                  <ul className="list-disc list-inside space-y-1">
                    <li>
                      <strong>Spatial locality:</strong> Conv filters see local
                      neighborhoods (3x3), not all 784 pixels. They learn local
                      patterns efficiently.
                    </li>
                    <li>
                      <strong>Weight sharing:</strong> The same filter is applied
                      at every position. One set of 9 weights detects a feature
                      everywhere in the image.
                    </li>
                    <li>
                      <strong>Feature hierarchy:</strong> Conv-pool stages build
                      from edges to shapes to digits through increasingly
                      abstract representations.
                    </li>
                    <li>
                      <strong>Dense weakness:</strong> A digit shifted a few
                      pixels is an entirely different 784-element vector to the
                      dense network. It must learn the same pattern at every
                      position independently.
                    </li>
                  </ul>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not About Training Time">
            It is tempting to think the dense network just needs more epochs. It
            does not. Train both for 20 epochs&mdash;the dense network plateaus
            around 97-98%. The gap is structural, not about patience.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Why Architecture Matters
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Architecture Matters"
            subtitle="The shifting experiment, the parameter surprise, and the core insight"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>The shifting experiment.</strong> Take a test digit&mdash;say
              a handwritten &ldquo;3&rdquo;&mdash;and shift it 2 pixels to the
              right. What happens?
            </p>
            <p className="text-muted-foreground">
              The 784-element flat vector changes at <em>every position</em>.
              Every pixel value shifts to the next index. To the dense network,
              this is a drastically different input. The weights that learned
              &ldquo;pixel 200 is important for detecting a 3&rdquo; are now
              looking at the wrong pixel. The dense network has no mechanism to
              handle this&mdash;it treats each pixel position as an independent
              feature.
            </p>
            <CodeBlock
              code={`# Shifting changes EVERY element of the flat vector
original = image.view(-1)          # [784]
shifted = shift(image, 2).view(-1) # [784]
(original != shifted).sum()        # ~750 of 784 values differ!

# To the dense network: almost entirely different input
# To the CNN: same filters detect same features,
#             pooling absorbs the shift`}
              language="python"
              filename="shifting.py"
            />
            <p className="text-muted-foreground">
              The CNN handles this effortlessly. The same filters detect the same
              features at slightly different positions (weight sharing), and
              pooling absorbs the small shift (spatial tolerance). A
              &ldquo;3&rdquo; shifted two pixels right activates the same filters
              in slightly different locations, and the max pooling regions still
              capture the features. Weight sharing and pooling together give the
              CNN <strong>spatial invariance</strong> that the dense network
              cannot have.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Flat Vector Problem">
            Remember the hook from What Convolutions Compute: flattening an
            image destroys spatial relationships. This is that same problem made
            concrete&mdash;your own dense network has this weakness. The CNN you
            just built does not.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Parameter count surprise */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>The parameter count surprise.</strong> Walk through the
              arithmetic:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-medium">
                Dense Network (first layer alone):
              </p>
              <p className="font-mono text-sm text-muted-foreground">
                784 inputs x 128 neurons = <strong className="text-foreground">100,352 parameters</strong>
              </p>
              <p className="text-xs text-muted-foreground">
                Every input pixel connects to every neuron. Separate weights for
                every spatial position.
              </p>
            </div>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-medium">
                CNN (entire conv stack):
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-1">
                <p>Conv1: 1 x 32 x 3 x 3 + 32 bias = <strong className="text-foreground">320</strong></p>
                <p>Conv2: 32 x 64 x 3 x 3 + 64 bias = <strong className="text-foreground">18,496</strong></p>
                <p>Total conv parameters: <strong className="text-foreground">18,816</strong></p>
              </div>
              <p className="text-xs text-muted-foreground">
                32 filters of 3x3 = 288 learned values detect features{' '}
                <em>everywhere</em> in the image. Weight sharing is the reason.
              </p>
            </div>
            <p className="text-muted-foreground">
              The dense network&apos;s first layer alone has{' '}
              <strong>more parameters than the CNN&apos;s entire convolutional
              stack</strong>. The CNN&apos;s total parameter count is actually
              higher (~421K vs ~110K) because the FC classifier layers add
              bulk&mdash;especially the 3136&rarr;128 layer. But the spatial
              processing where the architectural advantage lives? 18K
              parameters outperform 100K. The CNN uses the same weights
              everywhere.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Weight Sharing = Efficiency">
            The dense network learns &ldquo;pixel 200 matters&rdquo; and
            &ldquo;pixel 201 matters&rdquo; as separate facts. The CNN learns
            &ldquo;this 3x3 pattern matters&rdquo; once and applies it
            everywhere. Fewer parameters, better generalization.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* The core insight */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="py-6 px-6 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <p className="text-lg font-semibold text-foreground mb-3">
                The Core Insight
              </p>
              <p className="text-muted-foreground">
                <strong>Architecture encodes assumptions about data.</strong> The
                CNN assumes spatial structure exists&mdash;that nearby pixels are
                related and that the same patterns appear at different positions.
                MNIST proves this assumption correct. The dense network makes no
                such assumption and must learn spatial relationships from scratch,
                wasting parameters on what the CNN gets for free.
              </p>
              <p className="text-muted-foreground mt-3">
                Matching architecture to data structure is not an optimization
                trick. It is the key insight of this entire module.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="A Memorable Framing">
            The dense network flattens <strong>pixels</strong>. The CNN flattens{' '}
            <strong>features</strong>. Both flatten&mdash;but the CNN earned the
            right to flatten by extracting meaningful spatial features first.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Flatten misconception addressed */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might notice something that seems contradictory: the CNN also
              uses a flatten step. In What Convolutions Compute, you learned that
              flattening destroys spatial relationships. So why is flatten okay
              in the CNN?
            </p>
            <p className="text-muted-foreground">
              By the time we flatten, the spatial dimensions are 7x7&mdash;a
              49-position summary of the entire 28x28 image, where each position
              represents an <em>abstract learned feature</em>, not a raw pixel.
              This is fundamentally different from flattening the raw 28x28 image
              into 784 pixel values. The CNN earned the right to flatten by
              extracting meaningful spatial features first through two rounds of
              convolution and pooling.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title='Misconception: "Flatten Loses Information"'>
            Flattening raw pixels loses spatial relationships. Flattening{' '}
            <em>features</em> (after conv-pool stages) does not&mdash;the
            spatial information has already been captured in the feature maps.
            The 3136 values represent learned abstractions, not raw data.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'A simple CNN beats a dense network on MNIST.',
                description:
                  'Two conv layers, two pool layers, two FC layers. No tricks, no tuning. The architecture advantage is large enough that simplicity wins.',
              },
              {
                headline:
                  'The conv layers do more with far less.',
                description:
                  'Weight sharing means the CNN uses the same small filter everywhere instead of learning separate weights for every position. 18K conv parameters outperform 100K dense parameters at spatial feature extraction.',
              },
              {
                headline:
                  'The training loop is identical.',
                description:
                  'Forward, loss, backward, update. The only difference is what happens inside model(x). Architecture is the independent variable.',
              },
              {
                headline:
                  'Architecture encodes assumptions about data.',
                description:
                  'The CNN assumes spatial structure exists. MNIST proves the assumption correct. Matching architecture to data structure is the key insight.',
              },
              {
                headline:
                  'The dense network flattens pixels; the CNN flattens features.',
                description:
                  'Both flatten. But the CNN extracts meaningful spatial features first, through conv-pool stages, before collapsing to a flat vector.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Module arc echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Module Complete">
            You started this module by learning what a single convolution
            computes. Then you assembled convolutions into an architecture with
            pooling, stride, and padding. Now you have built one, trained it, and
            proved that architecture matters. That is the foundation for
            everything that follows in CNNs.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: Module Completion + Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="3.1"
            title="Convolutions"
            achievements={[
              'Convolution as sliding filter (multiply-and-sum over local region)',
              'Feature maps as spatial pattern-detection output',
              'Weight sharing and spatial locality',
              'Pooling, stride, padding, and the output size formula',
              'Conv-pool-fc architecture pattern',
              'Built and trained a CNN that beats a dense network on MNIST',
              'Architecture encodes assumptions about data',
            ]}
            nextModule="3.2"
            nextTitle="CNN Architectures"
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="What Comes Next"
            description="You have built your first CNN and seen it outperform a dense network. The natural question: can we go deeper? Better architectures have pushed CNN accuracy much further -- from LeNet's 99% on digits to human-level performance on real photographs. Next, we explore how CNN architectures evolved and what innovations made deeper, more powerful networks possible."
            buttonText="Back to Dashboard"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
