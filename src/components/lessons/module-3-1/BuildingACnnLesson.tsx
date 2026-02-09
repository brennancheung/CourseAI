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
  GradientCard,
  ComparisonRow,
} from '@/components/lessons'
import { CnnDimensionCalculator } from '@/components/widgets/CnnDimensionCalculator'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { CodeBlock } from '@/components/common/CodeBlock'
import { cn } from '@/lib/utils'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Building a CNN
 *
 * Second lesson in Module 3.1 (Convolutions).
 * Teaches how to assemble convolutions into a full CNN architecture
 * with pooling, stride, padding, and the conv-pool-fc pattern.
 *
 * Core concepts at DEVELOPED:
 * - Pooling (max pooling)
 * - Stride and padding
 * - Conv-pool-fc architecture pattern
 *
 * Concepts at INTRODUCED:
 * - Average pooling (contrast with max)
 * - Receptive field growth through stacking
 * - nn.Conv2d / nn.MaxPool2d API
 *
 * Previous: What Convolutions Compute (convolution operation, feature maps, edge detection)
 * Next: Training Your First CNN
 */

// ---------------------------------------------------------------------------
// Inline grid helpers for the pooling examples
// ---------------------------------------------------------------------------

function GridDisplay({
  data,
  highlights,
  size = 'md',
  label,
}: {
  data: number[][]
  highlights?: Set<string>
  size?: 'sm' | 'md'
  label?: string
}) {
  const cellSize = size === 'sm' ? 'w-8 h-8 text-[10px]' : 'w-10 h-10 text-xs'

  return (
    <div>
      {label && (
        <p className="text-xs text-muted-foreground font-medium mb-1">{label}</p>
      )}
      <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${data[0].length}, minmax(0, 1fr))` }}>
        {data.flatMap((row, r) =>
          row.map((val, c) => {
            const key = `${r}-${c}`
            const isHighlighted = highlights?.has(key)
            return (
              <div
                key={key}
                className={cn(
                  cellSize,
                  'flex items-center justify-center rounded font-mono font-medium',
                  isHighlighted
                    ? 'bg-violet-500/40 text-violet-200'
                    : 'bg-zinc-700/70 text-zinc-300',
                )}
              >
                {val}
              </div>
            )
          }),
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Data for pooling examples
// ---------------------------------------------------------------------------

const POOLING_INPUT: number[][] = [
  [1, 3, 0, 2],
  [4, 6, 1, 0],
  [0, 2, 5, 3],
  [1, 0, 4, 7],
]

const POOLING_MAX_OUTPUT: number[][] = [
  [6, 2],
  [2, 7],
]

const POOLING_AVG_OUTPUT: number[][] = [
  [3.5, 0.75],
  [0.75, 4.75],
]

// Edge detection feature map example (before/after pooling)
const EDGE_FEATURE_MAP: number[][] = [
  [0, 0, 8, 7, 0, 0],
  [0, 0, 9, 8, 0, 0],
  [0, 0, 7, 9, 0, 0],
  [0, 0, 8, 7, 0, 0],
  [0, 0, 9, 8, 0, 0],
  [0, 0, 7, 9, 0, 0],
]

const EDGE_POOLED: number[][] = [
  [0, 9, 0],
  [0, 9, 0],
  [0, 9, 0],
]

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function BuildingACnnLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Building a CNN"
            description="Assemble convolutions into a full architecture with pooling, stride, padding, and the classic conv-pool-fc pattern."
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
            Explain how pooling, stride, and padding control spatial
            dimensions in a CNN, and trace data through the conv-pool-fc
            architecture pattern that powers real convolutional networks.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In What Convolutions Compute, you learned the core operation: a
            small filter slides across an image computing weighted sums,
            producing a feature map. You computed it by hand. This lesson
            assembles that building block into a complete architecture.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'What pooling does (max and average) and why it is useful',
              'How stride and padding control output dimensions',
              'The conv-pool-fc architecture pattern and dimension tracking',
              'NOT: training a CNN or comparing accuracy (that is the next lesson)',
              'NOT: architecture design choices in depth (how many layers, etc.)',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Module Arc">
            In What Convolutions Compute you learned the building block. Today
            you assemble the architecture. Next lesson: you build and train one
            from scratch.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Hook - The "So What?" Problem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title='The "So What?" Problem'
            subtitle="You have feature maps. Now what?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You can compute what a convolution produces. But consider this: your
              edge-detection filter on a 28x28 MNIST image produces a 26x26
              feature map. Use 32 filters and you have 32 feature maps of
              26x26 = <strong>21,632 values</strong>. That is{' '}
              <em>more</em> than the 784 pixels you started with.
            </p>
            <p className="text-muted-foreground">
              And you want to classify this as a digit&mdash;one of 10 classes.
              How do you get from 21,632 spatial feature values to a 10-class
              prediction? Stack another conv layer and the computation gets even
              more expensive.
            </p>
            <p className="text-muted-foreground">
              <strong>Something needs to shrink.</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Problem">
            Feature maps preserve spatial dimensions. But classification is a
            single decision, not a spatial grid. You need a way to compress
            spatial information down to a decision vector.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Explain - Pooling
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Pooling: Shrinking with Purpose"
            subtitle="Compress spatial dimensions while preserving features"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Feature maps are spatially redundant. If there is a strong vertical
              edge at position (5,5), there is probably also a strong response at
              (5,6). We do not need pixel-perfect spatial precision&mdash;approximate
              location is enough for recognition.
            </p>
            <p className="text-muted-foreground">
              <strong>Max pooling</strong> exploits this redundancy. It slides a
              small window (typically 2x2) across the feature map, keeping only
              the <strong>maximum value</strong> in each window. Here is a
              concrete example:
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The &ldquo;Zoom Out&rdquo; Analogy">
            Each pooling step is like stepping back from an image. Close up you
            see individual pixels and edges. Step back and you see shapes and
            regions. Step back further and you see the whole object.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Max pooling worked example */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-4">
              <p className="text-sm text-muted-foreground font-medium">
                Max Pooling: 4x4 input with 2x2 window, stride 2
              </p>
              <div className="flex flex-wrap gap-8 items-start">
                <GridDisplay data={POOLING_INPUT} label="Input (4x4)" />
                <div className="flex items-center self-center text-muted-foreground">
                  {'\u2192'}
                </div>
                <GridDisplay data={POOLING_MAX_OUTPUT} label="Output (2x2)" />
              </div>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  <strong>Top-left window</strong> [1, 3, 4, 6]: max ={' '}
                  <strong className="text-violet-300">6</strong>
                </p>
                <p>
                  <strong>Top-right window</strong> [0, 2, 1, 0]: max ={' '}
                  <strong className="text-violet-300">2</strong>
                </p>
                <p>
                  <strong>Bottom-left window</strong> [0, 2, 1, 0]: max ={' '}
                  <strong className="text-violet-300">2</strong>
                </p>
                <p>
                  <strong>Bottom-right window</strong> [5, 3, 4, 7]: max ={' '}
                  <strong className="text-violet-300">7</strong>
                </p>
              </div>
              <p className="text-sm text-muted-foreground">
                Four numbers summarize what sixteen numbers said. The spatial
                dimensions are <strong>halved</strong> in each direction.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why 2x2, Stride 2?">
            A 2x2 window with stride 2 means every input value belongs to
            exactly one pooling window&mdash;no overlap, no gaps. This cleanly
            halves the spatial dimensions: 28x28 becomes 14x14, 14x14 becomes
            7x7.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Edge detection before/after pooling */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Does max pooling destroy important information? Look at what happens
              to a feature map from a vertical edge detector:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-4">
              <p className="text-sm text-muted-foreground font-medium">
                Edge Feature Map Before and After 2x2 Max Pooling
              </p>
              <div className="flex flex-wrap gap-8 items-start">
                <GridDisplay data={EDGE_FEATURE_MAP} label="Before (6x6)" size="sm" />
                <div className="flex items-center self-center text-muted-foreground">
                  {'\u2192'}
                </div>
                <GridDisplay data={EDGE_POOLED} label="After (3x3)" size="sm" />
              </div>
              <p className="text-sm text-muted-foreground">
                The edge is still clearly visible&mdash;a column of strong values
                in the center. The exact position is fuzzier (which pixel in the
                2x2 region had the edge) but the <strong>presence of the edge is
                preserved</strong>. This spatial tolerance is a feature, not a
                bug: a &ldquo;3&rdquo; shifted one pixel right is still a
                &ldquo;3.&rdquo;
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title='Misconception: "Pooling Destroys Information"'>
            Pooling discards exact positions but preserves feature
            presence. A 2x2 pooled region saying &ldquo;there is a strong edge
            somewhere in this area&rdquo; is almost as useful as knowing the
            exact pixel, while making the next layer 4x faster.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Average pooling comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Max pooling is not the only option. <strong>Average pooling</strong>{' '}
              computes the mean of each window instead of the max:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="flex flex-wrap gap-8 items-start">
                <GridDisplay data={POOLING_INPUT} label="Same input" size="sm" />
                <div className="flex flex-col gap-3">
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-muted-foreground w-12">Max:</span>
                    <GridDisplay data={POOLING_MAX_OUTPUT} size="sm" />
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-muted-foreground w-12">Avg:</span>
                    <GridDisplay data={POOLING_AVG_OUTPUT} size="sm" />
                  </div>
                </div>
              </div>
            </div>
            <ComparisonRow
              left={{
                title: 'Max Pooling',
                color: 'blue',
                items: [
                  'Keeps the strongest response',
                  '"Is the feature present in this region?"',
                  'Preserves sharp edges, peak values',
                  'Most common for feature detection',
                ],
              }}
              right={{
                title: 'Average Pooling',
                color: 'violet',
                items: [
                  'Smooths all responses together',
                  '"How strong is the feature on average?"',
                  'Blurs edges, reduces noise',
                  'Used in some architectures (often at the end)',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="When to Use Which">
            Max pooling is the default for feature detection layers. Average
            pooling appears most often as <strong>global average pooling</strong>{' '}
            at the end of modern architectures (like ResNet), where it replaces
            the flatten + fully-connected pattern entirely. When in doubt, use
            max pooling.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Check 1 - Compute the Pool
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Compute the Pool"
            subtitle="Predict before you peek"
          />
          <GradientCard title="Quick Check" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Given a <strong>4x4 feature map</strong> (same size as the
                worked example), you apply 2x2 max pooling with stride 2:
              </p>
              <div className="py-2 px-4 bg-black/20 rounded font-mono text-xs space-y-0.5">
                <p>[2, 5, 1, 3]</p>
                <p>[0, 8, 4, 1]</p>
                <p>[3, 1, 6, 2]</p>
                <p>[7, 0, 3, 9]</p>
              </div>
              <ol className="space-y-1 list-decimal list-inside">
                <li>What is the output size?</li>
                <li>What is the output value at position (0,1)&mdash;the top-right 2x2 window?</li>
                <li>What is the complete 2x2 output?</li>
              </ol>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal answer
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    1. Output size: <strong>2x2</strong> (4/2 = 2 in each
                    dimension).
                  </p>
                  <p>
                    2. Top-right window [1, 3, 4, 1]: max ={' '}
                    <strong>4</strong>.
                  </p>
                  <p>
                    3. Complete output: [<strong>8</strong>, <strong>4</strong>] /
                    [<strong>7</strong>, <strong>9</strong>].
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Pooling Dimensions">
            With a 2x2 window and stride 2, the output is always{' '}
            <strong>half</strong> the input size in each spatial dimension.
            Channels stay the same&mdash;pooling acts on each channel
            independently.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Explain - Stride and Padding
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Stride and Padding"
            subtitle="Controlling how dimensions change"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In What Convolutions Compute, the filter moved one pixel at a time.
              That was <strong>stride 1</strong>. But what if it jumped two pixels?
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Stride */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="Stride" color="blue">
              <div className="space-y-3 text-sm">
                <p>
                  <strong>Stride</strong> controls how far the filter jumps
                  between positions. Stride 1 means it visits every position.
                  Stride 2 means it skips every other position.
                </p>
                <div className="bg-black/20 rounded p-3 font-mono text-xs space-y-1">
                  <p>6x6 input, 3x3 filter, stride 1 {'\u2192'} 4x4 output</p>
                  <p>6x6 input, 3x3 filter, stride 2 {'\u2192'} 2x2 output</p>
                </div>
                <p>
                  Fewer positions visited = smaller output. Stride 2 roughly
                  halves the spatial dimensions&mdash;the same effect as pooling,
                  but built into the convolution itself.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Stride vs Pooling">
            Stride=2 in a convolution halves dimensions just like 2x2 max
            pooling. Modern architectures often use stride=2 convolutions
            instead of separate pooling layers.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Padding - problem first */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now consider what happens without <strong>padding</strong>. In What
              Convolutions Compute, a 3x3 filter on a 7x7 input gave a 5x5
              output&mdash;it shrank by 2. What happens if you stack conv layers?
            </p>
            <div className="py-4 px-6 bg-rose-500/5 border border-rose-500/20 rounded-lg">
              <p className="text-sm text-muted-foreground font-medium mb-2">
                The Shrinking Problem (no padding, 3x3 filter, stride 1):
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-0.5">
                <p>Layer 1: 32x32 {'\u2192'} 30x30</p>
                <p>Layer 2: 30x30 {'\u2192'} 28x28</p>
                <p>Layer 3: 28x28 {'\u2192'} 26x26</p>
                <p>Layer 4: 26x26 {'\u2192'} 24x24</p>
                <p>Layer 5: 24x24 {'\u2192'} 22x22</p>
              </div>
              <p className="text-sm text-rose-300 mt-2">
                Five layers and 10 pixels are gone from each edge. Border
                pixels participate in fewer convolution windows, so their
                information is systematically underrepresented.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Border Information Loss">
            Without padding, corner pixels contribute to exactly one output
            position. Center pixels contribute to{' '}
            <InlineMath math="F \times F" /> positions. The edges of the image
            are literally less represented in the output.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Padding - solution */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="Padding" color="blue">
              <div className="space-y-3 text-sm">
                <p>
                  <strong>Padding</strong> adds a border of zeros around the
                  input before convolving. With padding=1 on each side:
                </p>
                <div className="bg-black/20 rounded p-3 font-mono text-xs space-y-1">
                  <p>7x7 input + padding=1 {'\u2192'} 9x9 (padded)</p>
                  <p>9x9 with 3x3 filter {'\u2192'} 7x7 output</p>
                </div>
                <p>
                  The output is the <strong>same size</strong> as the input.
                  This is called &ldquo;same&rdquo; padding. The filter can now
                  be centered on every input position, including the borders.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="&ldquo;Same&rdquo; Padding Rule">
            For a 3x3 filter, padding=1 preserves spatial dimensions. For a
            5x5 filter, padding=2. The pattern: padding ={' '}
            <InlineMath math="\lfloor F/2 \rfloor" /> gives &ldquo;same&rdquo;
            output size.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* The general formula */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now you have seen stride and padding separately. The general
              formula for output size combines all three:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{output} = \left\lfloor \frac{N - F + 2P}{S} \right\rfloor + 1" />
            </div>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
              <li>
                <InlineMath math="N" /> = input size
              </li>
              <li>
                <InlineMath math="F" /> = filter (kernel) size
              </li>
              <li>
                <InlineMath math="P" /> = padding
              </li>
              <li>
                <InlineMath math="S" /> = stride
              </li>
            </ul>
            <p className="text-muted-foreground">
              Let&apos;s verify with the examples you already saw:
            </p>
            <div className="py-3 px-5 bg-muted/30 rounded-lg text-sm text-muted-foreground space-y-1 font-mono">
              <p>
                stride=1, no padding: (7 - 3 + 0) / 1 + 1 = 5 {'\u2713'}
              </p>
              <p>
                stride=2, no padding: (6 - 3 + 0) / 2 + 1 = 2 {'\u2713'}
              </p>
              <p>
                stride=1, padding=1: (7 - 3 + 2) / 1 + 1 = 7 {'\u2713'}
              </p>
              <p>
                stride=2, padding=1: (28 - 3 + 2) / 2 + 1 = 14 {'\u2713'}
              </p>
            </div>
            <p className="text-sm text-muted-foreground">
              That last example&mdash;stride=2 with padding=1 on a 28x28
              input&mdash;shows both parameters at work in one formula. Next,
              you will see how this enables a stride=2 conv to replace pooling
              entirely.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Extending the Formula">
            In What Convolutions Compute, you learned{' '}
            <InlineMath math="N - F + 1" />. That was the special case where
            stride=1 and padding=0. The general formula reduces to that when{' '}
            <InlineMath math="S = 1, P = 0" />.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Stride=2 conv vs separate pooling comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In fact, a stride=2 convolution can <strong>replace</strong>{' '}
              a separate pooling layer entirely. Compare these two approaches
              on a 28x28 input:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-1">
                  <p className="text-xs font-medium text-muted-foreground">
                    Option A: Stride=2 conv
                  </p>
                  <div className="font-mono text-xs text-muted-foreground space-y-0.5">
                    <p>Conv(32, 3x3, stride=2, pad=1)</p>
                    <p>
                      (28 - 3 + 2) / 2 + 1 = <strong className="text-foreground">14x14x32</strong>
                    </p>
                  </div>
                </div>
                <div className="space-y-1">
                  <p className="text-xs font-medium text-muted-foreground">
                    Option B: Stride=1 conv + pool
                  </p>
                  <div className="font-mono text-xs text-muted-foreground space-y-0.5">
                    <p>Conv(32, 3x3, stride=1, pad=1) {'\u2192'} 28x28x32</p>
                    <p>
                      MaxPool(2x2) {'\u2192'} <strong className="text-foreground">14x14x32</strong>
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <p className="text-muted-foreground text-sm">
              Same output dimensions, different mechanism. Option A does feature
              extraction and downsampling in one step. Option B separates them
              into two layers. Both are valid&mdash;this is a design choice
              you will see in real architectures.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Brief PyTorch connection */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In PyTorch, these parameters map directly to the constructor
              arguments:
            </p>
            <CodeBlock
              code={`# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
nn.Conv2d(1, 32, 3, stride=1, padding=1)
#         ^  ^^  ^         ^          ^
#         |  ||  |         |          padding P=1
#         |  ||  |         stride S=1
#         |  ||  kernel_size F=3
#         |  |out_channels (number of filters)
#         |  in_channels (channels from previous layer)

nn.MaxPool2d(2)  # kernel_size=2, stride defaults to kernel_size`}
              language="python"
              filename="pytorch_layers.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Reading the API">
            You do not need to memorize the API. The key insight: every
            argument maps to one variable in the output size formula. If you
            understand the formula, you can read any Conv2d or MaxPool2d call.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check 2 - Predict the Shape
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Predict the Shape"
            subtitle="Trace the dimensions"
          />
          <GradientCard title="Dimension Challenge" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A <strong>28x28</strong> input (1 channel) goes through:
              </p>
              <ol className="space-y-1 list-decimal list-inside">
                <li>
                  <code className="text-xs bg-black/20 px-1 rounded">
                    nn.Conv2d(1, 32, 3, padding=1)
                  </code>{' '}
                  &mdash; What is the output shape?
                </li>
                <li>
                  <code className="text-xs bg-black/20 px-1 rounded">
                    nn.MaxPool2d(2)
                  </code>{' '}
                  &mdash; What is the shape now?
                </li>
              </ol>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal answer
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    1. Conv2d with padding=1 and 3x3 filter: (28 - 3 + 2) / 1 + 1
                    = <strong>28x28x32</strong>. Same spatial dimensions, 32
                    channels.
                  </p>
                  <p>
                    2. MaxPool2d(2): 28/2 = <strong>14x14x32</strong>. Spatial
                    dimensions halved, channels unchanged.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Channels Through Pooling">
            Pooling never changes the number of channels. It operates on each
            channel independently. Only convolutions change the channel count.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Interactive Widget - CNN Dimension Calculator
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: CNN Dimension Calculator"
            subtitle="Build a CNN layer by layer and watch the shapes transform"
          />
          <p className="text-muted-foreground mb-4">
            Add layers, tweak parameters, and see how data shapes flow through
            the architecture in real time. Start with the LeNet preset to see a
            working CNN, then experiment.
          </p>
          <ExercisePanel
            title="CNN Dimension Calculator"
            subtitle="Build an architecture and trace the tensor dimensions"
          >
            <CnnDimensionCalculator />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>
                Start with the <strong>LeNet Preset</strong> and trace the
                dimensions through each layer.
              </li>
              <li>
                Remove all padding from the conv layers. How small do the
                feature maps get?
              </li>
              <li>
                Replace a MaxPool2d with a Conv2d using stride=2. Same
                dimension change, different mechanism.
              </li>
              <li>
                Try removing the Flatten layer. What happens to the Linear
                layer?
              </li>
              <li>
                Stack 3 conv layers before the first pool. Watch the
                dimensions stay constant (with padding=1).
              </li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Explain - The Conv-Pool-FC Pattern
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Conv-Pool-FC Pattern"
            subtitle="Assembling the full architecture"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Most CNNs follow a simple pattern: <strong>convolution layers</strong>{' '}
              to extract features, <strong>pooling layers</strong> to shrink
              spatial dimensions, then <strong>fully connected (dense) layers</strong>{' '}
              at the end to classify. You already know both halves&mdash;conv from
              the previous lesson, dense from your earlier work in the
              Foundations series. This lesson connects them.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Dimension tracking example */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a complete CNN tracing an MNIST image from input to
              classification:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="space-y-2 font-mono text-sm">
                <StageRow label="Input" shape="28x28x1" color="zinc" />
                <StageRow label="Conv(32, 3x3, pad=1)" shape="28x28x32" color="violet" annotation="Same spatial, 32 channels" />
                <StageRow label="MaxPool(2x2)" shape="14x14x32" color="sky" annotation="Spatial halved" />
                <StageRow label="Conv(64, 3x3, pad=1)" shape="14x14x64" color="violet" annotation="Same spatial, 64 channels" />
                <StageRow label="MaxPool(2x2)" shape="7x7x64" color="sky" annotation="Spatial halved again" />
                <StageRow label="Flatten" shape="3136" color="amber" annotation="7 x 7 x 64 = 3,136" />
                <StageRow label="Linear(128)" shape="128" color="emerald" />
                <StageRow label="Linear(10)" shape="10" color="emerald" annotation="One per digit class" />
              </div>
            </div>
            <p className="text-muted-foreground">
              Notice the pattern: <strong>spatial dimensions shrink</strong>{' '}
              (28 {'\u2192'} 14 {'\u2192'} 7) while{' '}
              <strong>channel count grows</strong> (1 {'\u2192'} 32 {'\u2192'}{' '}
              64). Then the flatten step collapses everything into a vector, and
              the linear layers make the final decision.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Shape Signature">
            A CNN&apos;s data transforms from wide and shallow (large spatial,
            few channels) to narrow and deep (small spatial, many channels) and
            then to a flat vector. This progression is the visual signature of
            every convolutional network.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Why this pattern works */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Why does this pattern work?</strong> Early conv layers
              detect local features (edges, textures) with small 3x3 filters.
              Pooling shrinks spatial dimensions so the next conv layer&apos;s
              3x3 filter covers more of the original image. Later conv layers
              detect combinations of earlier features (corners, shapes).
            </p>
            <p className="text-muted-foreground">
              Concretely: in What Convolutions Compute, a single 3x3 filter saw
              a 3x3 region of the input. After a 2x2 pool, each pooled position
              already summarizes a 2x2 area. A 3x3 filter on the{' '}
              <em>pooled</em> output therefore responds to a roughly{' '}
              <strong>6x6 region</strong> of the original input&mdash;each of
              the 9 pooled positions it reads covers a 2x2 patch. This is how
              the network goes from detecting edges to detecting shapes to
              detecting whole objects, all with small 3x3 filters.
            </p>
            <p className="text-muted-foreground">
              By the time we flatten, the spatial dimensions are small (7x7) but
              the channel count is rich (64 features). The FC layers take this
              compressed representation and make a decision.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Receptive Field Growth">
            Each conv-pool stage multiplies the effective receptive field. This
            is how the network goes from local features to global
            understanding&mdash;without using the huge filters we showed are
            impractical in What Convolutions Compute.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* The flatten transition */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The <strong>flatten</strong> step deserves attention. It is where
              the network transitions from &ldquo;where are the features?&rdquo;
              (spatial grid) to &ldquo;what do the features mean?&rdquo; (flat
              vector). The 7x7x64 tensor becomes a 3,136-element vector.
              Spatial structure is gone&mdash;by design. The dense layers that
              follow can now combine features from anywhere in the image to make
              a classification.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title='Misconception: "Output Is a Feature Map"'>
            A CNN classifying digits outputs a 10-element vector, not a feature
            map. The transition from spatial to flat is exactly what the
            fully-connected layers do. Feature maps are intermediate, not final.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Check 3 - Explain the Architecture
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Read an Architecture"
            subtitle="Trace and explain"
          />
          <GradientCard title="Architecture Challenge" color="cyan">
            <div className="space-y-3 text-sm">
              <CodeBlock
                code={`model = nn.Sequential(
    nn.Conv2d(1, 16, 5, padding=2),  # Layer A
    nn.ReLU(),
    nn.MaxPool2d(2),                  # Layer B
    nn.Conv2d(16, 32, 5, padding=2),  # Layer C
    nn.ReLU(),
    nn.MaxPool2d(2),                  # Layer D
    nn.Flatten(),                     # Layer E
    nn.Linear(???, 10),               # Layer F
)`}
                language="python"
                filename="cnn_model.py"
              />
              <p>Given a 28x28 input (1 channel):</p>
              <ol className="space-y-1 list-decimal list-inside">
                <li>What is the shape after Layer B (first MaxPool)?</li>
                <li>What is the shape after Layer D (second MaxPool)?</li>
                <li>What number replaces ??? in Layer F?</li>
                <li>What would happen if you removed all padding?</li>
              </ol>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal answer
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    1. After Layer A (Conv, pad=2, 5x5): 28x28x16. After Layer B
                    (Pool): <strong>14x14x16</strong>.
                  </p>
                  <p>
                    2. After Layer C (Conv, pad=2, 5x5): 14x14x32. After Layer D
                    (Pool): <strong>7x7x32</strong>.
                  </p>
                  <p>
                    3. 7 x 7 x 32 = <strong>1568</strong>.
                  </p>
                  <p>
                    4. Without padding: 28{'\u2192'}24{'\u2192'}12{'\u2192'}8
                    {'\u2192'}4. Then 4x4x32 = 512 instead of 1568. Spatial
                    dimensions shrink much faster and border information is lost.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Elaborate - Why This Architecture Works
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why This Architecture Works"
            subtitle="The hierarchy of features"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In What Convolutions Compute, we mentioned that later layers detect
              more complex patterns. Now you can see <strong>why</strong>: each
              conv-pool stage expands the receptive field and compresses spatial
              dimensions, forcing the network to represent information in
              increasingly abstract terms.
            </p>
            <p className="text-muted-foreground">
              Layer 1 detects edges. Layer 2 combines edges into corners and
              textures. Layer 3 combines those into parts of objects. Each layer
              builds on the previous one&mdash;the same hierarchical composition
              that makes convolutions so powerful for vision.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Historical Note">
            LeNet (1998) used this exact conv-pool-fc pattern on handwritten
            digits. It was one of the first successful CNNs. The pattern remains
            the backbone even of complex modern architectures.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The dimension-tracking example above alternates conv-pool-conv-pool.
              Does it have to be that way? <strong>No.</strong> Here is a
              different architecture that stacks three conv layers before the
              first pool:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-medium">
                Conv-Conv-Conv-Pool (no alternating)
              </p>
              <div className="space-y-2 font-mono text-sm">
                <StageRow label="Input" shape="28x28x1" color="zinc" />
                <StageRow label="Conv(32, 3x3, pad=1)" shape="28x28x32" color="violet" annotation="Spatial unchanged" />
                <StageRow label="Conv(32, 3x3, pad=1)" shape="28x28x32" color="violet" annotation="Still 28x28" />
                <StageRow label="Conv(64, 3x3, pad=1)" shape="28x28x64" color="violet" annotation="Three convs, no pooling yet" />
                <StageRow label="MaxPool(2x2)" shape="14x14x64" color="sky" annotation="Now we pool" />
                <StageRow label="Flatten" shape="12544" color="amber" annotation="14 x 14 x 64" />
                <StageRow label="Linear(10)" shape="10" color="emerald" />
              </div>
            </div>
            <p className="text-muted-foreground">
              This works. The three conv layers each detect increasingly
              complex features at full resolution before pooling compresses the
              spatial dimensions. The principle is: spatial dimensions need to
              shrink <em>at some point</em> so the network can see the big
              picture, but there is no rule that says pooling must follow every
              conv layer. The conv-pool-fc pattern is the classic starting
              point, not the only option.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Flexibility">
            Think of conv-pool-fc as a template, not a rigid rule. The key
            invariant is: spatial dimensions decrease, channel depth increases,
            then flatten for classification. How you get there is a design
            choice.
          </TipBlock>
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
                  'Max pooling keeps the strongest response in each region.',
                description:
                  'Shrinks spatial dimensions while preserving feature presence. A 2x2 max pool with stride 2 halves each spatial dimension.',
              },
              {
                headline:
                  'Stride controls the jump; padding controls the borders.',
                description:
                  'Stride > 1 reduces output size by skipping positions. Padding adds zeros around the border to prevent shrinking and preserve edge information.',
              },
              {
                headline:
                  'The general formula ties it all together.',
                description:
                  'output = floor((N - F + 2P) / S) + 1. Every dimension change in a CNN comes from this formula.',
              },
              {
                headline:
                  'The conv-pool-fc pattern builds from pixels to decisions.',
                description:
                  'Convolutions extract spatial features, pooling shrinks dimensions (growing the receptive field), flatten collapses spatial structure, and fully-connected layers classify.',
              },
              {
                headline: 'Spatial shrinks, channels grow, then flatten.',
                description:
                  'The data transforms from wide-and-shallow (28x28x1) to narrow-and-deep (7x7x64) to flat (3136) to output (10). This progression is the signature of every CNN.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Closing mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            A CNN is a series of zoom-outs. Each conv layer detects patterns at
            the current scale. Each pooling step zooms out, so the next conv
            layer detects patterns at a larger scale. By the end, the network
            has gone from pixels to edges to shapes to a classification decision.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/training-your-first-cnn"
            title="Training Your First CNN"
            description="You now know all the pieces of a CNN: convolution, pooling, stride, padding, and the conv-pool-fc architecture. Next: build one from scratch in PyTorch, train it on MNIST, and see it beat the dense network."
            buttonText="Continue to Training Your First CNN"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

// ---------------------------------------------------------------------------
// Dimension tracking stage row
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
