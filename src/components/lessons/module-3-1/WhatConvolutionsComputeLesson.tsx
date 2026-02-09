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
  TryThisBlock,
  ConceptBlock,
  WarningBlock,
  SummaryBlock,
  NextStepBlock,
  ComparisonRow,
  GradientCard,
} from '@/components/lessons'
import { ConvolutionExplorer } from '@/components/widgets/ConvolutionExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { cn } from '@/lib/utils'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * What Convolutions Compute
 *
 * First lesson in the CNN series (Module 3.1).
 * Teaches the convolution operation mechanically, then builds intuition
 * for why spatial filtering beats dense layers for image data.
 *
 * Core concepts at DEVELOPED:
 * - Convolution as sliding filter (multiply-and-sum over local region)
 * - Feature maps (output of convolution)
 * - Edge detection filters
 *
 * Concepts at INTRODUCED:
 * - Spatial structure / locality
 * - Weight sharing
 * - Multiple filters
 *
 * Next: Building CNN Architecture (pooling, stride, padding, full CNN)
 */

export function WhatConvolutionsComputeLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="What Convolutions Compute"
            description="The core operation behind every image-understanding neural network: slide a small filter across an image and compute weighted sums at each position."
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
            Compute what a convolutional filter produces when applied to a 2D
            grid, and explain why this spatial operation detects features that
            dense layers cannot.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You have built dense networks where each neuron computes a weighted
            sum over every input, with learnable parameters updated by the
            training loop. Convolutions use the same weighted sum&mdash;but only
            over a small local neighborhood. That single change is what this
            lesson is about.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'What the convolution operation computes (mechanically)',
              'What filters detect (edge detection as primary example)',
              'Why spatial locality and weight sharing matter',
              'NOT: pooling, stride, padding, or full CNN architecture (that is next lesson)',
              'NOT: training a CNN or nn.Conv2d in depth',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Series Arc">
            This module gives you the building blocks of convolutional networks.
            Today: the core operation. Next: assembling it into architecture.
            Then: building one that works.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Hook - The Flat Vector Problem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Flat Vector Problem"
            subtitle="What your dense network actually saw"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every network you have built so far treats its input as a{' '}
              <strong>flat list of numbers</strong>. For MNIST, you flattened a
              28x28 image into a 784-length vector and fed it to a dense layer.
              This works&mdash;you got decent accuracy.
            </p>
            <p className="text-muted-foreground">
              But think about what that flattening <strong>destroys</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Problem">
            Flattening an image throws away spatial relationships. Pixels that
            are neighbors in the image become distant strangers in the vector.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the 2D image, pixel (0,0) is right next to pixel (0,1)&mdash;they
              are part of the same local region, maybe part of the same stroke of
              a handwritten digit. But in the flattened vector, pixel (0,0) is at
              index 0 and pixel (1,0) is at index 28. The dense layer has{' '}
              <strong>no idea which pixels are neighbors</strong>.
            </p>
            <ComparisonRow
              left={{
                title: 'Dense Layer',
                color: 'amber',
                items: [
                  'Connects to ALL 784 pixels',
                  'No concept of "nearby"',
                  'Must learn edge at (10,10) separately from edge at (50,50)',
                  '784 weights per neuron',
                ],
              }}
              right={{
                title: 'Convolutional Layer',
                color: 'blue',
                items: [
                  'Connects to a small 3x3 neighborhood',
                  'Exploits spatial locality',
                  'Same filter detects edges everywhere',
                  '9 weights per filter',
                ],
              }}
            />
            <p className="text-muted-foreground">
              Convolutions solve both problems with one elegant idea: instead of
              connecting to all pixels, use a small filter that{' '}
              <strong>slides across the image</strong>, computing the same
              weighted sum at every position.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: The Convolution Operation
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Convolution Operation"
            subtitle="Multiply-and-sum over a sliding window"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A convolution uses two ingredients: an <strong>input grid</strong>{' '}
              (a small &quot;image&quot;) and a <strong>filter</strong> (also
              called a <em>kernel</em>)&mdash;just a small grid of numbers,
              typically 3x3.
            </p>
            <p className="text-muted-foreground">
              Here is the computation, step by step:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                Overlay the 3x3 filter on the top-left 3x3 region of the input
              </li>
              <li>Multiply each input value by the corresponding filter weight</li>
              <li>Sum all 9 products&mdash;this gives you one output value</li>
              <li>Slide the filter one position to the right and repeat</li>
              <li>
                When you reach the right edge, move down one row and start from
                the left
              </li>
              <li>
                Continue until the filter has visited every valid position
              </li>
            </ol>
            <p className="text-muted-foreground">
              The result is a smaller output grid called a{' '}
              <strong>feature map</strong>. A 7x7 input with a 3x3 filter
              produces a 5x5 feature map. In general, a filter of size{' '}
              <InlineMath math="F" /> on an input of size{' '}
              <InlineMath math="N" /> produces an output of size{' '}
              <InlineMath math="N - F + 1" />.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Operation, Different Scope">
            This is a <strong>weighted sum</strong>&mdash;the same operation as a
            neuron. The only difference: a neuron connects to all inputs, a
            convolution connects to a small local neighborhood.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Worked example with concrete numbers */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let&apos;s compute one position by hand. Here is a 3x3 region
              from an input image with a vertical edge (left side dark, right
              side bright), and the vertical edge filter:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground">
                <strong>Input patch at position (0, 0):</strong>
              </p>
              <div className="grid grid-cols-3 gap-1 w-fit font-mono text-xs">
                {[0, 0, 1, 0, 0, 1, 0, 0, 1].map((v, i) => (
                  <div key={i} className="w-8 h-8 flex items-center justify-center rounded bg-zinc-700 text-zinc-300">
                    {v}
                  </div>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                <strong>Filter (vertical edge detector):</strong>
              </p>
              <div className="grid grid-cols-3 gap-1 w-fit font-mono text-xs">
                {[-1, 0, 1, -1, 0, 1, -1, 0, 1].map((v, i) => (
                  <div
                    key={i}
                    className={cn(
                      'w-8 h-8 flex items-center justify-center rounded',
                      v < 0 ? 'bg-rose-400/70 text-zinc-100' : v > 0 ? 'bg-violet-400/70 text-zinc-100' : 'bg-zinc-700 text-zinc-400',
                    )}
                  >
                    {v}
                  </div>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                <strong>Multiply element-wise, then sum:</strong>
              </p>
              <p className="text-sm font-mono text-muted-foreground">
                (0&times;-1) + (0&times;0) + (1&times;1) + (0&times;-1) + (0&times;0) + (1&times;1) + (0&times;-1) + (0&times;0) + (1&times;1) = <strong className="text-white">3</strong>
              </p>
              <p className="text-sm text-muted-foreground">
                A strong positive response&mdash;the filter found a vertical edge
                at this position.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Receptive Field">
            The output at position (0,0) was computed from <strong>only</strong>{' '}
            the 9 input pixels in the top-left 3x3 region. It knows nothing
            about the bottom-right corner. This locality is the entire point.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Widget placement: immediately after worked example, before formula */}
      <Row>
        <Row.Content>
          <ExercisePanel
            title="Convolution Explorer"
            subtitle="Select different inputs and filters, then step through or animate"
          >
            <ConvolutionExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ol className="space-y-2 text-sm list-decimal list-inside">
              <li>
                Select &quot;Vertical Edge&quot; input + &quot;Vertical Edge&quot;
                filter. Step through&mdash;watch the feature map light up at the edge.
              </li>
              <li>
                Keep the same input but switch to &quot;Horizontal Edge&quot;
                filter. The feature map goes dark&mdash;no horizontal edges to detect.
              </li>
              <li>
                Try &quot;Uniform&quot; input with any edge filter. Output is
                all zeros&mdash;nothing to detect in a flat image.
              </li>
              <li>
                Try &quot;Corner&quot; input with both edge filters. Each
                one highlights different parts of the corner.
              </li>
            </ol>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In mathematical notation, the output at position{' '}
              <InlineMath math="(i, j)" /> is:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{output}(i, j) = \sum_{m=0}^{2}\sum_{n=0}^{2} \text{filter}(m, n) \cdot \text{input}(i + m, j + n)" />
            </div>
            <p className="text-muted-foreground">
              This formula just says what you already did by hand: for each
              output position, overlay the filter, multiply element-wise, and
              sum. Notice that output position <InlineMath math="(i, j)" />{' '}
              tells you whether the filter&apos;s pattern was found in the
              input region starting at <InlineMath math="(i, j)" />. The
              feature map preserves spatial layout&mdash;the top-left of the
              output corresponds to the top-left of the input.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Output Size">
            For a 3x3 filter on an{' '}
            <InlineMath math="H \times W" /> input, the output is{' '}
            <InlineMath math="(H-2) \times (W-2)" />. The filter needs room
            to fit, so the output shrinks by 2 in each dimension.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Check 1 - Predict and Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict before you peek"
          />
          <div className="space-y-4">
            <GradientCard title="Quick Check" color="emerald">
              <div className="space-y-3">
                <p className="text-sm">
                  Given a <strong>4x4 input</strong> and a <strong>2x2 filter</strong>:
                </p>
                <ol className="space-y-1 text-sm list-decimal list-inside">
                  <li>What size is the output feature map?</li>
                  <li>How many multiply-and-sum operations does the filter perform total?</li>
                </ol>
                <details className="mt-3">
                  <summary className="text-sm font-medium cursor-pointer text-primary">
                    Reveal answer
                  </summary>
                  <div className="mt-2 space-y-2 text-sm">
                    <p>
                      1. Output size: <strong>3x3</strong> (4 - 2 + 1 = 3 in each
                      dimension)
                    </p>
                    <p>
                      2. The filter visits <strong>9 positions</strong> (3x3 grid),
                      performing 4 multiplications at each, for{' '}
                      <strong>36 total multiplications</strong> and 9 sums.
                    </p>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Pattern">
            For filter size <InlineMath math="F" /> on input size{' '}
            <InlineMath math="N" />, output size ={' '}
            <InlineMath math="N - F + 1" />.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: What Filters Detect - Edge Detection
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Filters Detect"
            subtitle="The filter is a pattern detector"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              So far, convolution is just arithmetic. The magic comes from{' '}
              <strong>what the filter values mean</strong>. Consider a vertical
              edge detector:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <div className="grid grid-cols-3 gap-1 w-fit mx-auto">
                {[
                  [-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1],
                ].flat().map((v, i) => (
                  <div
                    key={i}
                    className={cn(
                      'w-10 h-10 flex items-center justify-center rounded font-mono text-sm font-medium',
                      v < 0 ? 'bg-rose-400/70 text-zinc-100' : v > 0 ? 'bg-violet-400/70 text-zinc-100' : 'bg-zinc-700 text-zinc-400',
                    )}
                  >
                    {v}
                  </div>
                ))}
              </div>
            </div>
            <p className="text-muted-foreground">
              The left column has -1 (penalizes bright pixels on the left), the
              center is 0 (ignores), and the right column has +1 (rewards bright
              pixels on the right). When this filter sits on a region where the
              left side is dark and the right side is bright&mdash;a vertical
              edge&mdash;the output is a <strong>large positive number</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Mental Model">
            A filter is a <strong>pattern detector</strong>. It asks &quot;does
            this local region look like my pattern?&quot; at every position. The
            feature map is the answer at every position.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now try the same input with a <strong>horizontal edge filter</strong>.
              If the image has no horizontal edges, the output is near zero
              everywhere. The filter is <strong>selective</strong>: it responds
              strongly to its target pattern and weakly to everything else.
            </p>
            <p className="text-muted-foreground">
              And if you apply the vertical edge filter to a{' '}
              <strong>uniform image</strong> (all pixels the same value), the
              output is <em>exactly</em> zero everywhere. The detector says
              &quot;nothing here.&quot;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Hand-Designed">
            These hand-crafted filters build intuition, but in a real CNN the
            network <strong>learns</strong> filter values via backprop. Nobody
            tells it to look for edges.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Real-photograph bridge: what this looks like at scale */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              These tiny grids might feel academic, but this is exactly what
              happens at scale. Apply the vertical edge filter to a photograph
              of a building, and the feature map lights up along every vertical
              edge&mdash;window frames, door jambs, columns. Apply the horizontal
              edge filter, and you get the rooflines, floor boundaries, and
              horizontal trim.
            </p>
            <div className="py-4 px-6 bg-violet-500/5 border border-violet-500/20 rounded-lg">
              <p className="text-sm text-muted-foreground">
                <strong>What real CNN first-layer filters look like:</strong>{' '}
                Researchers have visualized the filters learned by AlexNet,
                VGGNet, and other trained CNNs. The first layer consistently
                learns edge detectors at various orientations, color contrast
                detectors (red vs. green, blue vs. yellow), and simple texture
                patterns. They look remarkably similar to the hand-crafted
                filters you just used&mdash;but the network discovered them
                entirely on its own through gradient descent.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="From Toy to Real">
            The 7x7 grid and the 1024x1024 photograph use the{' '}
            <strong>identical operation</strong>. The only difference is scale.
            A real CNN applies dozens of 3x3 filters across millions of pixels.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check 2 - Spot the Pattern
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Spot the Pattern"
            subtitle="Which filter produced which output?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Not all filters detect edges. A filter where every value is 1
              (or 1/9) computes the <strong>average</strong> of the
              neighborhood&mdash;it <em>blurs</em> the image instead of
              highlighting sharp transitions. You may have tried this in the
              explorer above.
            </p>
            <GradientCard title="Challenge" color="cyan">
              <div className="space-y-3 text-sm">
                <p>
                  Imagine applying three different filters to the same image of a
                  square (vertical and horizontal edges):
                </p>
                <ul className="space-y-1.5">
                  <li>
                    <strong>Feature map A:</strong> Bright values along the left
                    and right sides of the square
                  </li>
                  <li>
                    <strong>Feature map B:</strong> Bright values along the top
                    and bottom of the square
                  </li>
                  <li>
                    <strong>Feature map C:</strong> All values are roughly the
                    same (no strong pattern)
                  </li>
                </ul>
                <details>
                  <summary className="text-sm font-medium cursor-pointer text-primary">
                    Which filter produced each?
                  </summary>
                  <div className="mt-2 space-y-1.5">
                    <p>
                      <strong>A</strong> = vertical edge filter (responds to
                      left-right transitions)
                    </p>
                    <p>
                      <strong>B</strong> = horizontal edge filter (responds to
                      top-bottom transitions)
                    </p>
                    <p>
                      <strong>C</strong> = blur filter (averages all neighbors,
                      no edge selectivity)
                    </p>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Why This Beats Dense - Weight Sharing and Locality
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why This Beats Dense Layers"
            subtitle="Weight sharing and locality"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Convolutions have two structural advantages over dense layers that
              make them dramatically better for images:
            </p>
            <GradientCard title="Locality" color="blue">
              <p className="text-sm">
                A dense neuron connecting to a 28x28 image has{' '}
                <strong>784 weights</strong>. A 3x3 conv filter has{' '}
                <strong>9 weights</strong>. The filter only looks at
                neighbors&mdash;which is where the useful information is for
                images. Nearby pixels are far more related to each other than
                distant ones.
              </p>
            </GradientCard>
            <GradientCard title="Weight Sharing" color="violet">
              <p className="text-sm">
                The same 9 weights are used at <strong>every position</strong>.
                If the filter learns to detect a vertical edge, it detects
                vertical edges <em>everywhere</em>. A dense network would need to
                learn &quot;vertical edge at (10,10)&quot; and &quot;vertical edge
                at (50,50)&quot; as completely separate patterns.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Parameter Count">
            <p className="text-sm">
              Dense layer for 28x28 with 32 neurons:
              <br />
              784 x 32 = <strong>25,088</strong> parameters
            </p>
            <p className="text-sm mt-2">
              Conv layer with 32 filters (3x3):
              <br />
              32 x 9 = <strong>288</strong> parameters
            </p>
            <p className="text-sm mt-2">
              Nearly <strong>100x fewer</strong> parameters!
            </p>
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A convolution is <strong>not</strong> just a smaller dense layer.
              It is the same filter applied everywhere&mdash;this is weight
              sharing. A single set of weights is reused at every spatial
              position.
            </p>
            <p className="text-muted-foreground">
              You might wonder: why not use a bigger filter to capture more
              context? Consider the extreme case&mdash;a 28x28 filter on a
              28x28 image connects to every pixel. That <em>is</em> a dense
              layer. The power of small filters comes from locality. We will
              see in the next lesson how stacking small filters achieves large
              receptive fields with far fewer parameters.
            </p>
            <p className="text-muted-foreground">
              And the training process is identical to what you already know.
              Forward pass (now includes convolutions), compute loss, backward
              pass, update parameters. Nothing about training changes&mdash;only
              the architecture.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Training Recipe">
            The training loop is the same: forward, loss, backward, update. You
            do not need to learn a new optimization algorithm for CNNs.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Multiple Filters = Multiple Features
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Multiple Filters = Multiple Features"
            subtitle="Each filter asks a different question"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One filter detects vertical edges. Another detects horizontal edges.
              A third might detect diagonals. Each filter produces its own
              feature map. Stack them together: the output of a conv layer is a{' '}
              <strong>stack of feature maps</strong>, one per filter.
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg text-sm text-muted-foreground">
              <p>
                <strong>Analogy:</strong> Think of each filter as asking a
                question about every neighborhood in the image. &quot;Is there
                a vertical edge here? A horizontal edge? A diagonal?&quot; The
                number of filters is how many questions the layer asks at each
                location.
              </p>
            </div>
            <p className="text-muted-foreground">
              This is analogous to having multiple neurons in a dense
              layer&mdash;more filters means a richer representation of the
              input.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Layer Output Shape">
            If you have 32 filters of size 3x3 applied to a 28x28 input, the
            output is a stack of 32 feature maps, each 26x26. The &quot;depth&quot;
            of the output equals the number of filters.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: What the Network Actually Learns
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What the Network Actually Learns"
            subtitle="From hand-crafted to discovered"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You saw earlier that trained CNNs rediscover edge and texture
              detectors on their own. But the real power comes from what
              happens in <strong>deeper layers</strong>.
            </p>
            <p className="text-muted-foreground">
              Later layers combine first-layer features into more complex
              patterns: edges become corners, corners become shapes, shapes
              become objects. This <strong>hierarchical composition</strong> is
              what makes CNNs so powerful&mdash;and it is a topic for a future
              lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Hierarchy of Features">
            Layer 1 detects edges. Layer 2 combines edges into corners and
            textures. Layer 3 combines those into parts of objects. Each layer
            builds on the previous one.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'A convolution slides a small filter across the input.',
                description:
                  'At each position, it computes a weighted sum of the local neighborhood. This is the same multiply-and-sum operation as a neuron, but applied locally.',
              },
              {
                headline:
                  'The output (feature map) shows where the pattern was detected.',
                description:
                  'Large positive values mean the filter\'s pattern is present. Values near zero mean it is absent. The feature map is a spatial "answer key."',
              },
              {
                headline:
                  'Weight sharing makes convolutions efficient and position-invariant.',
                description:
                  'The same filter works at every position. If it learns to detect a vertical edge, it detects vertical edges everywhere, with only 9 parameters.',
              },
              {
                headline:
                  'Multiple filters detect multiple features.',
                description:
                  'A stack of feature maps (one per filter) is the output of a convolutional layer. More filters = richer representation.',
              },
              {
                headline: 'Filter values are learned, not hand-designed.',
                description:
                  'The network learns what patterns to detect via the same backprop and gradient descent you already know. No new training algorithm needed.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Closing mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            A convolutional layer asks a fixed set of questions&mdash;&quot;is
            there a vertical edge here? a horizontal edge? a diagonal?&quot;&mdash;at
            every location in the image. The answers, arranged spatially, are
            the feature maps.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/building-a-cnn"
            title="Building a CNN"
            description="You now know what a single convolution computes. Next: how to assemble convolutions into a full CNN with pooling to shrink dimensions, stride and padding to control sizes, and the classic conv-pool-fc pattern."
            buttonText="Continue to Building a CNN"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
