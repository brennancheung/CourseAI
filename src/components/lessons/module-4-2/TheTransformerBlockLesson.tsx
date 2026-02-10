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
  NextStepBlock,
  GradientCard,
  ComparisonRow,
} from '@/components/lessons'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * The Transformer Block
 *
 * Fifth lesson in Module 4.2 (Attention & the Transformer).
 * Eighth lesson in Series 4 (LLMs & Transformers).
 *
 * Assembles multi-head attention, feed-forward networks, residual
 * connections, and layer normalization into the repeating unit
 * that stacks to form a transformer.
 *
 * Core concepts at DEVELOPED:
 * - FFN structure and role (4x expansion, GELU, "writes" role)
 * - Residual stream as cross-layer backbone (upgraded from INTRODUCED)
 * - Transformer block as repeating unit
 * - "Attention reads, FFN writes" mental model
 *
 * Core concepts at INTRODUCED:
 * - Layer normalization (contrast with batch norm)
 * - Pre-norm vs post-norm ordering
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing the transformer block in PyTorch (Module 4.3)
 * - Causal masking (Lesson 6)
 * - The full decoder-only architecture (Lesson 6)
 * - Training transformers (Module 4.3)
 * - Cross-attention, encoder-decoder architecture
 * - RMSNorm or other layer norm variants
 *
 * Previous: Multi-Head Attention (module 4.2, lesson 4)
 * Next: Decoder-Only Transformers (module 4.2, lesson 6)
 */

// ---------------------------------------------------------------------------
// Transformer Block Diagram (inline SVG)
// ---------------------------------------------------------------------------

function TransformerBlockDiagram() {
  return (
    <div className="flex justify-center py-4">
      <svg
        width="380"
        height="520"
        viewBox="0 0 380 520"
        className="overflow-visible"
      >
        {/* ---- Residual Stream (vertical backbone) ---- */}
        {/* Main stream line - bottom to top */}
        <line
          x1="190"
          y1="500"
          x2="190"
          y2="20"
          stroke="#a78bfa"
          strokeWidth={3}
          strokeDasharray="6,4"
          opacity={0.4}
        />

        {/* ---- Input ---- */}
        <rect
          x="130"
          y="480"
          width="120"
          height="30"
          rx="6"
          fill="#a78bfa"
          opacity={0.15}
          stroke="#a78bfa"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="499"
          textAnchor="middle"
          fill="#a78bfa"
          fontSize="12"
          fontWeight="600"
        >
          Input (n, 768)
        </text>

        {/* ---- Arrow up from input ---- */}
        <line
          x1="190"
          y1="480"
          x2="190"
          y2="455"
          stroke="#a78bfa"
          strokeWidth={2}
        />
        <polygon points="185,458 190,450 195,458" fill="#a78bfa" />

        {/* ---- First LayerNorm ---- */}
        <rect
          x="130"
          y="420"
          width="120"
          height="28"
          rx="5"
          fill="#34d399"
          opacity={0.12}
          stroke="#34d399"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="438"
          textAnchor="middle"
          fill="#34d399"
          fontSize="11"
          fontWeight="500"
        >
          LayerNorm
        </text>

        {/* Arrow from input to LN1 */}
        <line
          x1="190"
          y1="450"
          x2="190"
          y2="448"
          stroke="#a78bfa"
          strokeWidth={2}
        />

        {/* ---- First residual branch point ---- */}
        {/* Branch line: goes right from stream, then down, then right to rejoin */}
        <line
          x1="190"
          y1="460"
          x2="310"
          y2="460"
          stroke="#a78bfa"
          strokeWidth={1.5}
          opacity={0.5}
        />
        <line
          x1="310"
          y1="460"
          x2="310"
          y2="360"
          stroke="#a78bfa"
          strokeWidth={1.5}
          opacity={0.5}
        />
        <line
          x1="310"
          y1="360"
          x2="215"
          y2="360"
          stroke="#a78bfa"
          strokeWidth={1.5}
          opacity={0.5}
        />
        {/* Skip connection label */}
        <text
          x="320"
          y="415"
          fill="#a78bfa"
          fontSize="9"
          opacity={0.6}
          fontStyle="italic"
        >
          skip
        </text>

        {/* Arrow from LN1 to MHA */}
        <line
          x1="190"
          y1="420"
          x2="190"
          y2="400"
          stroke="#34d399"
          strokeWidth={1.5}
        />
        <polygon points="186,403 190,396 194,403" fill="#34d399" />

        {/* ---- MHA block ---- */}
        <rect
          x="115"
          y="366"
          width="150"
          height="30"
          rx="6"
          fill="#38bdf8"
          opacity={0.12}
          stroke="#38bdf8"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="385"
          textAnchor="middle"
          fill="#38bdf8"
          fontSize="11"
          fontWeight="600"
        >
          Multi-Head Attention
        </text>

        {/* ---- First Add (residual merge) ---- */}
        <circle
          cx="190"
          cy="350"
          r="12"
          fill="none"
          stroke="#a78bfa"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="354"
          textAnchor="middle"
          fill="#a78bfa"
          fontSize="14"
          fontWeight="bold"
        >
          +
        </text>

        {/* Arrow from MHA to Add1 */}
        <line
          x1="190"
          y1="366"
          x2="190"
          y2="362"
          stroke="#38bdf8"
          strokeWidth={1.5}
        />

        {/* Arrow from Add1 upward */}
        <line
          x1="190"
          y1="338"
          x2="190"
          y2="318"
          stroke="#a78bfa"
          strokeWidth={2}
        />
        <polygon points="186,321 190,314 194,321" fill="#a78bfa" />

        {/* ---- Second LayerNorm ---- */}
        <rect
          x="130"
          y="284"
          width="120"
          height="28"
          rx="5"
          fill="#34d399"
          opacity={0.12}
          stroke="#34d399"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="302"
          textAnchor="middle"
          fill="#34d399"
          fontSize="11"
          fontWeight="500"
        >
          LayerNorm
        </text>

        {/* ---- Second residual branch point ---- */}
        <line
          x1="190"
          y1="325"
          x2="310"
          y2="325"
          stroke="#a78bfa"
          strokeWidth={1.5}
          opacity={0.5}
        />
        <line
          x1="310"
          y1="325"
          x2="310"
          y2="225"
          stroke="#a78bfa"
          strokeWidth={1.5}
          opacity={0.5}
        />
        <line
          x1="310"
          y1="225"
          x2="215"
          y2="225"
          stroke="#a78bfa"
          strokeWidth={1.5}
          opacity={0.5}
        />
        <text
          x="320"
          y="280"
          fill="#a78bfa"
          fontSize="9"
          opacity={0.6}
          fontStyle="italic"
        >
          skip
        </text>

        {/* Arrow from LN2 to FFN */}
        <line
          x1="190"
          y1="284"
          x2="190"
          y2="268"
          stroke="#34d399"
          strokeWidth={1.5}
        />
        <polygon points="186,271 190,264 194,271" fill="#34d399" />

        {/* ---- FFN block ---- */}
        <rect
          x="115"
          y="230"
          width="150"
          height="34"
          rx="6"
          fill="#f59e0b"
          opacity={0.12}
          stroke="#f59e0b"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="244"
          textAnchor="middle"
          fill="#f59e0b"
          fontSize="11"
          fontWeight="600"
        >
          Feed-Forward Network
        </text>
        <text
          x="190"
          y="258"
          textAnchor="middle"
          fill="#f59e0b"
          fontSize="9"
          opacity={0.7}
        >
          768 {'\u2192'} 3072 {'\u2192'} 768
        </text>

        {/* ---- Second Add (residual merge) ---- */}
        <circle
          cx="190"
          cy="215"
          r="12"
          fill="none"
          stroke="#a78bfa"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="219"
          textAnchor="middle"
          fill="#a78bfa"
          fontSize="14"
          fontWeight="bold"
        >
          +
        </text>

        {/* Arrow from FFN to Add2 */}
        <line
          x1="190"
          y1="230"
          x2="190"
          y2="227"
          stroke="#f59e0b"
          strokeWidth={1.5}
        />

        {/* Arrow from Add2 upward to output */}
        <line
          x1="190"
          y1="203"
          x2="190"
          y2="180"
          stroke="#a78bfa"
          strokeWidth={2}
        />
        <polygon points="186,183 190,176 194,183" fill="#a78bfa" />

        {/* ---- Output ---- */}
        <rect
          x="130"
          y="146"
          width="120"
          height="30"
          rx="6"
          fill="#a78bfa"
          opacity={0.15}
          stroke="#a78bfa"
          strokeWidth={1.5}
        />
        <text
          x="190"
          y="165"
          textAnchor="middle"
          fill="#a78bfa"
          fontSize="12"
          fontWeight="600"
        >
          Output (n, 768)
        </text>

        {/* ---- Dimension annotations ---- */}
        <text
          x="68"
          y="438"
          textAnchor="end"
          fill="#9ca3af"
          fontSize="9"
          fontFamily="monospace"
        >
          (n, 768)
        </text>
        <text
          x="68"
          y="385"
          textAnchor="end"
          fill="#9ca3af"
          fontSize="9"
          fontFamily="monospace"
        >
          (n, 768)
        </text>
        <text
          x="68"
          y="302"
          textAnchor="end"
          fill="#9ca3af"
          fontSize="9"
          fontFamily="monospace"
        >
          (n, 768)
        </text>
        <text
          x="68"
          y="248"
          textAnchor="end"
          fill="#9ca3af"
          fontSize="9"
          fontFamily="monospace"
        >
          (n, 768)
        </text>

        {/* ---- Legend ---- */}
        <rect x="20" y="30" width="10" height="10" rx="2" fill="#a78bfa" opacity={0.4} />
        <text x="35" y="39" fill="#9ca3af" fontSize="9">Residual stream</text>

        <rect x="20" y="48" width="10" height="10" rx="2" fill="#38bdf8" opacity={0.4} />
        <text x="35" y="57" fill="#9ca3af" fontSize="9">Multi-Head Attention</text>

        <rect x="20" y="66" width="10" height="10" rx="2" fill="#f59e0b" opacity={0.4} />
        <text x="35" y="75" fill="#9ca3af" fontSize="9">Feed-Forward Network</text>

        <rect x="20" y="84" width="10" height="10" rx="2" fill="#34d399" opacity={0.4} />
        <text x="35" y="93" fill="#9ca3af" fontSize="9">Layer Normalization</text>

        {/* ---- "Pre-norm" annotation ---- */}
        <text
          x="75"
          y="355"
          fill="#9ca3af"
          fontSize="8"
          fontStyle="italic"
          opacity={0.6}
        >
          LN before
        </text>
        <text
          x="75"
          y="365"
          fill="#9ca3af"
          fontSize="8"
          fontStyle="italic"
          opacity={0.6}
        >
          sub-layer
        </text>
        <text
          x="75"
          y="375"
          fill="#9ca3af"
          fontSize="8"
          fontStyle="italic"
          opacity={0.6}
        >
          (pre-norm)
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Stacked Blocks Diagram (zoom-out showing repeating pattern)
// ---------------------------------------------------------------------------

function StackedBlocksDiagram() {
  return (
    <div className="flex justify-center py-4">
      <svg
        width="280"
        height="320"
        viewBox="0 0 280 320"
        className="overflow-visible"
      >
        {/* Residual stream backbone */}
        <line
          x1="140"
          y1="310"
          x2="140"
          y2="10"
          stroke="#a78bfa"
          strokeWidth={2.5}
          strokeDasharray="5,3"
          opacity={0.3}
        />

        {/* Block 1 */}
        <rect
          x="60"
          y="230"
          width="160"
          height="60"
          rx="8"
          fill="#1e1e2e"
          stroke="#6b7280"
          strokeWidth={1}
          opacity={0.8}
        />
        <rect x="70" y="240" width="60" height="16" rx="3" fill="#38bdf8" opacity={0.2} stroke="#38bdf8" strokeWidth={0.8} />
        <text x="100" y="252" textAnchor="middle" fill="#38bdf8" fontSize="8">MHA</text>
        <rect x="140" y="240" width="70" height="16" rx="3" fill="#f59e0b" opacity={0.2} stroke="#f59e0b" strokeWidth={0.8} />
        <text x="175" y="252" textAnchor="middle" fill="#f59e0b" fontSize="8">FFN</text>
        <rect x="70" y="262" width="60" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="100" y="271" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <rect x="140" y="262" width="70" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="175" y="271" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <text x="240" y="265" fill="#9ca3af" fontSize="10" fontWeight="500">Block 1</text>

        {/* Arrow between blocks */}
        <line x1="140" y1="230" x2="140" y2="210" stroke="#a78bfa" strokeWidth={1.5} />
        <polygon points="137,213 140,208 143,213" fill="#a78bfa" />

        {/* Block 2 */}
        <rect
          x="60"
          y="150"
          width="160"
          height="60"
          rx="8"
          fill="#1e1e2e"
          stroke="#6b7280"
          strokeWidth={1}
          opacity={0.8}
        />
        <rect x="70" y="160" width="60" height="16" rx="3" fill="#38bdf8" opacity={0.2} stroke="#38bdf8" strokeWidth={0.8} />
        <text x="100" y="172" textAnchor="middle" fill="#38bdf8" fontSize="8">MHA</text>
        <rect x="140" y="160" width="70" height="16" rx="3" fill="#f59e0b" opacity={0.2} stroke="#f59e0b" strokeWidth={0.8} />
        <text x="175" y="172" textAnchor="middle" fill="#f59e0b" fontSize="8">FFN</text>
        <rect x="70" y="182" width="60" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="100" y="191" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <rect x="140" y="182" width="70" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="175" y="191" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <text x="240" y="185" fill="#9ca3af" fontSize="10" fontWeight="500">Block 2</text>

        {/* Dots for remaining blocks */}
        <text x="140" y="140" textAnchor="middle" fill="#6b7280" fontSize="16" fontWeight="bold">...</text>

        {/* Block N */}
        <rect
          x="60"
          y="70"
          width="160"
          height="60"
          rx="8"
          fill="#1e1e2e"
          stroke="#6b7280"
          strokeWidth={1}
          opacity={0.8}
        />
        <rect x="70" y="80" width="60" height="16" rx="3" fill="#38bdf8" opacity={0.2} stroke="#38bdf8" strokeWidth={0.8} />
        <text x="100" y="92" textAnchor="middle" fill="#38bdf8" fontSize="8">MHA</text>
        <rect x="140" y="80" width="70" height="16" rx="3" fill="#f59e0b" opacity={0.2} stroke="#f59e0b" strokeWidth={0.8} />
        <text x="175" y="92" textAnchor="middle" fill="#f59e0b" fontSize="8">FFN</text>
        <rect x="70" y="102" width="60" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="100" y="111" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <rect x="140" y="102" width="70" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="175" y="111" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <text x="240" y="105" fill="#9ca3af" fontSize="10" fontWeight="500">Block N</text>

        {/* Input label */}
        <text x="140" y="305" textAnchor="middle" fill="#a78bfa" fontSize="10" fontWeight="500">
          Embeddings
        </text>

        {/* Output label */}
        <text x="140" y="60" textAnchor="middle" fill="#a78bfa" fontSize="10" fontWeight="500">
          Final Output
        </text>

        {/* Residual stream label */}
        <text x="28" y="190" fill="#a78bfa" fontSize="9" opacity={0.5} fontStyle="italic"
          transform="rotate(-90, 28, 190)"
        >
          residual stream
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function TheTransformerBlockLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Transformer Block"
            description="Multi-head attention is only one-third of the story. Assemble the repeating unit that stacks to form GPT."
            category="Attention"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how multi-head attention, a feed-forward network,
            residual connections, and layer normalization compose into the{' '}
            <strong>transformer block</strong>&mdash;the single repeating unit
            that stacks to form GPT-2, GPT-3, and every modern LLM.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            Multi-head attention (built from scratch), residual connections
            (from ResNets), batch normalization, and two-layer neural networks.
            Every piece of the transformer block is familiar&mdash;the novelty
            is how they compose.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The transformer block: MHA + FFN + residual connections + layer norm',
              '"Attention reads, FFN writes" as the organizing mental model',
              'Layer normalization introduced by contrast with batch norm',
              'The FFN\u2019s 4x expansion factor and parameter dominance',
              'Pre-norm vs post-norm (brief\u2014know the distinction and standard choice)',
              'Why this block can stack (shape preservation)',
              'NOT: implementing in PyTorch\u2014that\u2019s Module 4.3',
              'NOT: causal masking\u2014that\u2019s the next lesson',
              'NOT: the full decoder-only architecture\u2014next lesson',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook: The Missing 2/3
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Missing Two-Thirds"
            subtitle="Attention is the headline act, but it's not the whole show"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have spent four lessons building multi-head attention. It is
              natural to think the transformer <em>is</em> attention. Here is
              a number that might change that.
            </p>

            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-3">
              <p className="text-sm font-medium text-violet-400">
                GPT-2 (124M parameters)&mdash;where do they live?
              </p>
              <div className="grid gap-3 md:grid-cols-3 text-sm text-muted-foreground">
                <div className="px-3 py-2 bg-sky-500/5 border border-sky-500/20 rounded-lg text-center">
                  <p className="text-sky-400 font-medium text-lg">~28M</p>
                  <p className="text-xs">Attention layers</p>
                  <p className="text-xs text-muted-foreground/50">~23%</p>
                </div>
                <div className="px-3 py-2 bg-amber-500/5 border border-amber-500/20 rounded-lg text-center">
                  <p className="text-amber-400 font-medium text-lg">~57M</p>
                  <p className="text-xs">FFN layers</p>
                  <p className="text-xs text-muted-foreground/50">~46%</p>
                </div>
                <div className="px-3 py-2 bg-muted/50 rounded-lg text-center">
                  <p className="text-muted-foreground font-medium text-lg">~38M</p>
                  <p className="text-xs">Embeddings</p>
                  <p className="text-xs text-muted-foreground/50">~31%</p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground/70">
                The feed-forward network (FFN) layers contain{' '}
                <strong>twice as many parameters as attention</strong>. What are
                they doing?
              </p>
            </div>

            <p className="text-muted-foreground">
              We have been building the attention mechanism in detail because it
              is the conceptually new piece&mdash;the mechanism that lets tokens
              communicate with each other. But multi-head attention alone is not
              a transformer. It is one component inside a larger structure: the{' '}
              <strong>transformer block</strong>.
            </p>

            <p className="text-muted-foreground">
              The last lesson ended with a seed:{' '}
              <em>
                &ldquo;Attention reads from the residual stream; the FFN writes
                new information into it.&rdquo;
              </em>{' '}
              This lesson delivers on that promise.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Puzzle">
            If attention is so important, why does the FFN have more
            parameters? This question drives the entire lesson. The answer
            reshapes how you think about transformers.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain: Starting the Block Diagram
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Building the Block"
            subtitle="Start with what you know, add what&rsquo;s missing"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You already know multi-head attention as a complete black box.
              Input: a sequence of vectors, each{' '}
              <InlineMath math="d_{\text{model}}" /> dimensions. Output: a
              sequence of vectors, same shape, enriched with context from other
              tokens. Let&rsquo;s wrap it properly.
            </p>

            <p className="text-muted-foreground">
              <strong>First addition: residual connection around MHA.</strong>{' '}
              You know this from ResNets&mdash;same{' '}
              <InlineMath math="F(x) + x" /> pattern. The attention output is
              the &ldquo;edit,&rdquo; the input is the &ldquo;document.&rdquo;
              You already saw this in Values and the Attention Output, where
              the attention output was added to the original embedding. Same
              idea, now wrapped around the full multi-head mechanism.
            </p>

            <p className="text-muted-foreground">
              <strong>Second addition: normalization.</strong> Before the input
              goes into MHA, we normalize it. You know that normalizing
              activations helps training&mdash;you learned batch normalization
              in the training dynamics and ResNets lessons. But batch norm has
              a problem for sequences. Let&rsquo;s see why.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Building from Known Pieces">
            Residual connections: from ResNets. Normalization: from batch
            norm. Two-layer networks: from Series 1&ndash;2. Every component
            of the transformer block is a close cousin of something you
            already know.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Layer Norm Section
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Layer Normalization"
            subtitle="Same idea as batch norm, different axis"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              As you learned in the training dynamics and ResNets lessons,
              normalizing activations between layers helps deep networks train
              stably. The transformer block normalizes before each sub-layer.
              But the normalization method you know&mdash;batch
              norm&mdash;will not work here.
            </p>

            <p className="text-muted-foreground">
              Batch normalization normalizes each feature across all examples
              in a batch. For a batch of images, this works well&mdash;each
              feature (pixel channel) gets normalized using statistics from all
              images. But for sequences of tokens, it breaks down.
            </p>

            <p className="text-muted-foreground">
              Consider a batch of sentences with different lengths. Batch norm
              would normalize, say, the 5th token&rsquo;s feature across all
              sentences. But the 5th token of &ldquo;The cat sat&rdquo; and the
              5th token of &ldquo;Despite the overwhelming evidence that
              suggested otherwise...&rdquo; are in completely different
              positions semantically. Averaging their statistics is
              nonsensical.
            </p>

            <ComparisonRow
              left={{
                title: 'Batch Norm',
                color: 'rose',
                items: [
                  'Normalizes across examples (column-wise)',
                  'Statistics depend on what else is in the batch',
                  'Different behavior at train vs eval (running averages)',
                  'Breaks with variable-length sequences',
                ],
              }}
              right={{
                title: 'Layer Norm',
                color: 'emerald',
                items: [
                  'Normalizes across features within one example (row-wise)',
                  'Each token normalized independently',
                  'Same behavior at train and eval (no running averages)',
                  'Works naturally with any sequence length',
                ],
              }}
            />

            <p className="text-muted-foreground">
              <strong>Layer normalization</strong> takes the opposite approach:
              for each token independently, normalize across all{' '}
              <InlineMath math="d_{\text{model}}" /> features to mean 0 and
              variance 1, then apply learned scale (
              <InlineMath math="\gamma" />) and shift (
              <InlineMath math="\beta" />) parameters&mdash;just like batch
              norm&rsquo;s learnable parameters, but applied per-token rather
              than per-feature.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta" />
              <div className="text-sm text-muted-foreground space-y-1">
                <p>
                  <InlineMath math="\mu, \sigma^2" /> = mean and variance
                  computed across features for <strong>this one token</strong>
                </p>
                <p>
                  <InlineMath math="\gamma, \beta" /> = learned scale and shift
                  (same shape as <InlineMath math="d_{\text{model}}" />)
                </p>
                <p>
                  <InlineMath math="\epsilon" /> = small constant for numerical
                  stability (same as batch norm)
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              The formula is identical to batch norm&rsquo;s. The only
              difference is <em>which dimension</em> the statistics are
              computed over. Batch norm: across the batch. Layer norm: across
              features. Same normalization idea, different axis.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Shape, Different Axis">
            Batch norm and layer norm use the same formula. The difference is
            what they average over. Layer norm treats each token as a
            self-contained unit&mdash;its statistics depend only on that
            token&rsquo;s own feature vector, not on anything else in the batch.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1: Layer Norm Understanding
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Check: Layer Norm vs Batch Norm" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>1.</strong> Why can&rsquo;t we use batch norm in a
                transformer?
              </p>
              <p>
                <strong>2.</strong> Does layer norm need different behavior at
                train vs eval time?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>1.</strong> Variable-length sequences make
                    cross-example normalization at each position meaningless.
                    The 5th token of different sentences has no reason to share
                    statistics. Layer norm normalizes each token independently,
                    so sequence length and batch composition do not matter.
                  </p>
                  <p>
                    <strong>2.</strong> No. Layer norm computes per-example
                    statistics at every forward pass&mdash;there are no running
                    averages to maintain. No{' '}
                    <code className="text-xs">model.train()</code> vs{' '}
                    <code className="text-xs">model.eval()</code> distinction
                    for normalization behavior.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain: The FFN
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Feed-Forward Network"
            subtitle="The other half of the transformer block"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              After multi-head attention reads context from other tokens, the
              result passes through a <strong>feed-forward network</strong>{' '}
              (FFN). This is a plain two-layer neural network&mdash;the same
              architecture you have been building since Series 1. Two{' '}
              <code className="text-xs">nn.Linear</code> layers with an
              activation function between them:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2" />
              <div className="text-sm text-muted-foreground space-y-1">
                <p>
                  <InlineMath math="W_1" />: shape{' '}
                  <InlineMath math="(d_{\text{model}},\; 4 \cdot d_{\text{model}})" />{' '}
                  &mdash; <strong>expands</strong> from 768 to 3072
                </p>
                <p>
                  <InlineMath math="W_2" />: shape{' '}
                  <InlineMath math="(4 \cdot d_{\text{model}},\; d_{\text{model}})" />{' '}
                  &mdash; <strong>compresses</strong> from 3072 back to 768
                </p>
                <p>
                  GELU: the &ldquo;smooth ReLU&rdquo; from the activation
                  functions lesson. The decision guide said &ldquo;GELU for
                  transformers&rdquo;&mdash;this is where it lives.
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              The <strong>4x expansion factor</strong> is the key design
              choice. The first layer projects into a space four times wider
              than{' '}
              <InlineMath math="d_{\text{model}}" />. In GPT-2, that means 768
              dimensions expand to 3072. The second layer compresses back to
              768. Why expand and then compress?
            </p>

            <p className="text-muted-foreground">
              The wider hidden layer creates a higher-dimensional space where
              the model can compute complex feature combinations that do not
              fit in the smaller{' '}
              <InlineMath math="d_{\text{model}}" /> space. Think of it as a
              workspace: the model needs more room to think than to
              communicate. The compression step keeps only what is useful,
              discarding the intermediate computations.
            </p>

            {/* Parameter count */}
            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-amber-400">
                Parameter count: FFN vs Attention
              </p>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>
                  <strong>FFN per block:</strong>{' '}
                  <InlineMath math="2 \times d_{\text{model}} \times d_{\text{ff}} = 2 \times 768 \times 3072 = 4{,}718{,}592" />
                </p>
                <p>
                  <strong>Attention per block</strong> (Q+K+V+O):{' '}
                  <InlineMath math="4 \times d_{\text{model}}^2 = 4 \times 768^2 = 2{,}359{,}296" />
                </p>
                <p className="text-amber-400/80 font-medium pt-1">
                  The FFN has <strong>2x the parameters</strong> of attention
                  in every block. This answers the hook&rsquo;s puzzle.
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Like MHA, the FFN output is also added to its input via a
              residual connection. Same{' '}
              <InlineMath math="F(x) + x" /> pattern, same &ldquo;editing the
              document&rdquo; analogy. The block now has{' '}
              <strong>two residual connections</strong>&mdash;one around MHA,
              one around FFN.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Attention Reads, FFN Writes">
            Attention gathers context from other tokens&mdash;it{' '}
            <strong>reads</strong> from the residual stream. The FFN processes
            what attention found and updates the representation&mdash;it{' '}
            <strong>writes</strong> back. Different operations, complementary
            roles.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. The Complete Block
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Transformer Block"
            subtitle="All four components, assembled"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The full pre-norm transformer block in two lines:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-4">
              <div className="space-y-2">
                <BlockMath math="x' = x + \text{MHA}(\text{LayerNorm}(x))" />
                <BlockMath math="\text{output} = x' + \text{FFN}(\text{LayerNorm}(x'))" />
              </div>
              <p className="text-xs text-muted-foreground/70">
                Each sub-layer (MHA, FFN) is wrapped in a residual connection.
                LayerNorm is applied <em>before</em> each sub-layer (pre-norm).
              </p>
            </div>

            <p className="text-muted-foreground">
              That is the entire block. Four components, two residual
              connections. Here is the data flow:
            </p>

            <TransformerBlockDiagram />

            <p className="text-muted-foreground">
              Notice the shape at every stage:{' '}
              <InlineMath math="(n, d_{\text{model}})" /> in,{' '}
              <InlineMath math="(n, d_{\text{model}})" /> out. The block is{' '}
              <strong>shape-preserving</strong>. This is why identical blocks
              can stack&mdash;the output of block 1 is exactly the right shape
              to be the input of block 2.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Pre-Norm Placement">
            Layer norm goes <strong>before</strong> each sub-layer (inside the
            residual branch), not after. This is the modern standard used by
            GPT-2, GPT-3, and virtually all current LLMs.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Pre-norm vs Post-norm (brief)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              A brief note: pre-norm vs post-norm
            </p>

            <ComparisonRow
              left={{
                title: 'Post-Norm (Original, 2017)',
                color: 'amber',
                items: [
                  'x\' = LayerNorm(x + MHA(x))',
                  'Norm on the residual stream itself',
                  'Requires careful learning rate warmup',
                  'Used in the original "Attention Is All You Need" paper',
                ],
              }}
              right={{
                title: 'Pre-Norm (Modern Standard)',
                color: 'emerald',
                items: [
                  'x\' = x + MHA(LayerNorm(x))',
                  'Norm inside the branch, stream stays clean',
                  'More stable training at depth',
                  'GPT-2, GPT-3, LLaMA, and most modern LLMs',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The difference matters: with pre-norm, the residual stream has a{' '}
              <strong>clean identity path</strong> from input to output. No
              layer norm sits on the main highway. With post-norm, every
              gradient flowing through the residual stream must pass through a
              layer norm operation at every block&mdash;that is 24 layer norms
              in a 12-block GPT-2, each one rescaling the gradient. With
              pre-norm, the gradient has a clean additive path that bypasses
              all norms entirely. The difference compounds with depth, which
              is why post-norm becomes unstable in very deep models and
              requires careful learning rate warmup to train at all.
            </p>

            <p className="text-muted-foreground">
              All diagrams and formulas in this lesson use pre-norm, because
              that is what you will encounter in practice.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why This Matters">
            Pre-norm is a seemingly small change (reorder two operations) with
            a big impact on trainability. The original transformer paper used
            post-norm; almost every model since GPT-2 uses pre-norm.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Check 2: Apply the Mental Model
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Tracing Through Blocks"
            subtitle="Apply the &ldquo;attention reads, FFN writes&rdquo; mental model"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>1.</strong> A token just went through block 5 of a
                12-block GPT-2. Its representation now contains information
                from attention (context from other tokens) and FFN processing.
                What happens next?
              </p>
              <p>
                <strong>2.</strong> Why does stacking more blocks help?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>1.</strong> It enters block 6 as input. Block
                    6&rsquo;s MHA reads from the updated residual stream
                    (which now includes contributions from blocks 1&ndash;5).
                    Block 6&rsquo;s FFN processes what block 6&rsquo;s
                    attention found and writes its update. The representation
                    gets progressively richer.
                  </p>
                  <p>
                    <strong>2.</strong> Each block refines the representation.
                    Earlier blocks capture simpler patterns (adjacent tokens,
                    basic syntax). Later blocks capture more complex patterns
                    (long-range dependencies, semantic relationships). This is
                    the same hierarchical feature principle from CNNs&mdash;early
                    layers detect edges, later layers detect objects&mdash;but
                    applied to contextual relationships between tokens.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          11. Elaborate: The Residual Stream as Backbone
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Residual Stream"
            subtitle="From skip connection to central backbone"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know residual connections from ResNets, where they wrap
              2&ndash;3 conv layers and help individual blocks learn useful
              residuals. It is tempting to think the transformer uses them
              the same way&mdash;same{' '}
              <InlineMath math="F(x) + x" />, same story. The mechanism{' '}
              <em>is</em> identical. But the role is fundamentally larger.
            </p>

            <p className="text-muted-foreground">
              In a transformer, there are <strong>two</strong> residual
              connections per block (one around MHA, one around FFN), and the
              residual stream they form is the{' '}
              <strong>backbone of the entire model</strong>. It flows from
              the embedding layer all the way to the final output. Every
              sub-layer in every block&mdash;every MHA, every FFN&mdash;reads
              from and writes to this stream.
            </p>

            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-violet-400">
                The shared document analogy (extended)
              </p>
              <p className="text-sm text-muted-foreground">
                The residual stream is a shared document that starts as the raw
                embedding. Block 1&rsquo;s attention reads it and adds context
                notes. Block 1&rsquo;s FFN reads the annotated version and adds
                processed insights. Block 2 reads the version enriched by
                block 1. By block 12, the document has been annotated by{' '}
                <strong>24 sub-layers</strong>&mdash;12 attention reads and 12
                FFN writes.
              </p>
            </div>

            <StackedBlocksDiagram />

            <p className="text-muted-foreground">
              <strong>Without residual connections</strong>, consider what
              happens concretely. Block 1&rsquo;s MHA computes attention
              over the input tokens. Early in training, attention weights are
              nearly uniform&mdash;each token attends roughly equally to all
              others. The output is close to the average of all token
              embeddings. Without the residual connection, this near-uniform
              average is <em>all</em> that passes to the FFN. The
              token&rsquo;s original identity&mdash;the specific information
              that distinguishes &ldquo;cat&rdquo; from &ldquo;sat&rdquo;&mdash;is
              destroyed, replaced by a generic blend.
            </p>

            <p className="text-muted-foreground">
              With the residual connection, that near-uniform average is{' '}
              <strong>added to</strong> the original embedding. The
              token&rsquo;s identity flows through untouched, and
              attention&rsquo;s contribution is additive&mdash;it can only
              help, never erase. Each sub-layer learns only the{' '}
              <strong>delta</strong>&mdash;the small edit that improves the
              representation&mdash;rather than needing to simultaneously
              preserve all existing information and compute its own
              contribution. This is the same problem you saw in ResNets,
              but more severe: a 12-block transformer has 24 sub-layers, each
              of which would need to learn a near-perfect identity function
              plus its own residual.
            </p>

            <p className="text-muted-foreground">
              <strong>Gradient flow</strong> benefits directly. With residual
              connections, gradients have a direct path from the output to any
              layer&rsquo;s input. In a 12-block GPT-2 with 2 sub-layers per
              block, there are 24 residual additions. The gradient highway from
              ResNets, but deeper and more critical.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Gradient Payoff">
            In a 12-block GPT-2, gradients can flow from the output through{' '}
            <strong>24 residual additions</strong> straight to the input
            embedding&mdash;no vanishing through deep chains of layers. Same
            gradient highway as ResNets, but deeper and more critical.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Elaborate: Why the FFN Matters
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why the FFN Matters"
            subtitle='Not "just plumbing" between attention layers'
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The FFN looks simple compared to multi-head attention. Two linear
              layers and an activation&mdash;you have been building these since
              your first neural network. It is tempting to dismiss it as
              plumbing. That would be a mistake.
            </p>

            <p className="text-muted-foreground">
              <strong>Without the FFN</strong>, consider what each block
              actually computes. Attention produces a weighted sum of V
              vectors&mdash;and V vectors are linear projections of the input.
              A weighted sum of linear projections is still a linear operation.
              So attention&rsquo;s output for any token is always a point
              inside the <strong>convex hull</strong> of its input
              vectors&mdash;it can only blend existing representations, never
              escape them.
            </p>

            <p className="text-muted-foreground">
              Concretely: suppose three tokens have embeddings at positions A,
              B, and C in 768-dimensional space. Attention can produce any
              weighted average of A, B, and C&mdash;any point inside the
              triangle they form. But it <em>cannot</em> produce a point{' '}
              <em>outside</em> that triangle. The FFN&rsquo;s GELU activation
              breaks this constraint. It applies a nonlinear transformation
              that can push the representation to entirely new regions of the
              space. Without it, stacking more blocks changes nothing&mdash;a
              stack of linear operations is still linear, still stuck inside
              the same convex hull.
            </p>

            <p className="text-muted-foreground">
              Research by Geva et al. suggests FFN layers function as{' '}
              <strong>key-value memories</strong>: the first layer&rsquo;s rows
              are &ldquo;keys&rdquo; matching input patterns, and the second
              layer&rsquo;s rows are &ldquo;values&rdquo; storing associated
              information. Specific FFN neurons activate for specific
              concepts&mdash;there are neurons that fire for facts about
              specific entities. The FFN is where the model stores what it has
              learned.
            </p>

            <p className="text-muted-foreground">
              Two-thirds of the model&rsquo;s parameters live in FFN layers.
              Those parameters store the knowledge the model has learned.
              Attention decides <em>which tokens are relevant</em> to each
              other. The FFN decides <em>what to do</em> with that information.
              The transformer needs both.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Reads + Writes = Understanding">
            Attention alone can copy and blend information but cannot transform
            it. The FFN&rsquo;s nonlinearity is what makes each block more
            than a routing layer. <strong>Attention decides what&rsquo;s
            relevant. The FFN decides what to do about it.</strong>
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Check 3: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Transfer Question"
            subtitle="Apply your understanding to a design decision"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A colleague proposes making the transformer &ldquo;more
                efficient&rdquo; by reducing the FFN expansion factor from 4x
                to 1x&mdash;no expansion at all, just{' '}
                <InlineMath math="d_{\text{model}} \rightarrow d_{\text{model}} \rightarrow d_{\text{model}}" />.
                What would you tell them?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The expansion gives the FFN a higher-dimensional space to
                    compute in. In GPT-2, reducing 4x to 1x would cut FFN
                    parameters from ~4.7M to ~1.2M per block&mdash;a 75%
                    reduction in the component that holds the model&rsquo;s
                    learned knowledge.
                  </p>
                  <p>
                    The 4x expansion is not wasteful&mdash;it is where the
                    work happens. The wider hidden layer lets the model compute
                    complex feature combinations that cannot be represented in
                    the smaller{' '}
                    <InlineMath math="d_{\text{model}}" /> space. Reducing it
                    would dramatically cut the model&rsquo;s capacity to store
                    and process knowledge, even though the efficiency gain
                    looks appealing on paper.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
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
                  'The transformer block is MHA + FFN + 2 residual connections + 2 layer norms.',
                description:
                  'Four components, two sub-layers. The same block repeats N times (12 in GPT-2, 96 in GPT-3). Input shape equals output shape, enabling stacking.',
              },
              {
                headline:
                  'Attention reads, FFN writes.',
                description:
                  'Attention gathers context from other tokens (reads the residual stream). The FFN processes what attention found and updates the representation (writes to the stream). Different operations, complementary roles.',
              },
              {
                headline:
                  'The residual stream is the central backbone of the entire model.',
                description:
                  'Every sub-layer in every block reads from and writes to the residual stream. It flows from embedding to output, carrying accumulated information through all layers.',
              },
              {
                headline:
                  'Layer norm stabilizes training; pre-norm is the modern standard.',
                description:
                  'Layer norm normalizes across features within a single token (unlike batch norm, which normalizes across examples). Pre-norm places it inside the residual branch, keeping the main stream clean for gradient flow.',
              },
              {
                headline:
                  '~1/3 of parameters in attention, ~2/3 in FFN. The FFN is not plumbing\u2014it\u2019s where knowledge lives.',
                description:
                  'The FFN\u2019s 4x expansion creates a higher-dimensional workspace for complex computations. Research suggests FFN neurons store factual knowledge as key-value memories.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
            <p className="text-sm font-medium text-violet-400">
              Mental model: &ldquo;Attention reads, FFN writes&rdquo;
            </p>
            <p className="text-sm text-muted-foreground">
              The transformer block is a read-process cycle. Attention reads
              context from other tokens. The FFN processes and transforms.
              The residual stream carries everything forward. Stack{' '}
              <InlineMath math="N" /> blocks, and each one refines the
              representation&mdash;reading from an increasingly rich stream and
              writing back what it discovers.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          15. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/decoder-only-transformers"
            title="Decoder-Only Transformers"
            description="Stack N blocks. Add causal masking so tokens can&rsquo;t look ahead. That&rsquo;s GPT."
            buttonText="Continue to Decoder-Only Transformers"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
