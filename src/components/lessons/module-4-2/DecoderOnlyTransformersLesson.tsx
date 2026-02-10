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
  ModuleCompleteBlock,
  ReferencesBlock,
} from '@/components/lessons'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Decoder-Only Transformers
 *
 * Sixth and final lesson in Module 4.2 (Attention & the Transformer).
 * Ninth lesson in Series 4 (LLMs & Transformers).
 *
 * Assembles the complete GPT architecture: causal masking + stacked
 * transformer blocks + input pipeline + output projection. This is a
 * CONSOLIDATE lesson -- one new mechanism (causal masking), the rest
 * is assembly of known components.
 *
 * Core concepts at DEVELOPED:
 * - Causal masking (why, how, where)
 * - Full GPT architecture end-to-end
 * - GPT parameter counting
 *
 * Core concepts at INTRODUCED:
 * - Output projection (d_model -> vocab logits)
 * - Encoder-decoder vs decoder-only distinction
 * - Why decoder-only won for LLMs
 *
 * Core concepts at MENTIONED:
 * - BERT as encoder-only
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing GPT in PyTorch (Module 4.3)
 * - Training, loss curves, learning rate scheduling (Module 4.3)
 * - KV caching, flash attention, efficient inference (Module 4.3)
 * - Cross-attention mechanics in detail (Series 6)
 * - Finetuning, instruction tuning, RLHF (Module 4.4)
 *
 * Previous: The Transformer Block (module 4.2, lesson 5)
 * Next: Module 4.3 (Building nanoGPT)
 */

// ---------------------------------------------------------------------------
// Causal Mask Comparison SVG -- before/after showing full vs masked attention
// ---------------------------------------------------------------------------

function CausalMaskDiagram() {
  const tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
  const cellSize = 36
  const labelOffset = 50
  const matrixSize = tokens.length * cellSize

  function renderMatrix(
    offsetX: number,
    title: string,
    isMasked: boolean,
  ) {
    const elements: React.ReactElement[] = []

    // Title
    elements.push(
      <text
        key={`title-${offsetX}`}
        x={offsetX + labelOffset + matrixSize / 2}
        y={16}
        textAnchor="middle"
        fill="#e2e8f0"
        fontSize="13"
        fontWeight="600"
      >
        {title}
      </text>,
    )

    // Column labels (top)
    tokens.forEach((token, j) => {
      elements.push(
        <text
          key={`col-${offsetX}-${j}`}
          x={offsetX + labelOffset + j * cellSize + cellSize / 2}
          y={38}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="9"
          fontFamily="monospace"
        >
          {token}
        </text>,
      )
    })

    // Row labels (left) + cells
    tokens.forEach((token, i) => {
      // Row label
      elements.push(
        <text
          key={`row-${offsetX}-${i}`}
          x={offsetX + labelOffset - 6}
          y={48 + i * cellSize + cellSize / 2 + 3}
          textAnchor="end"
          fill="#9ca3af"
          fontSize="9"
          fontFamily="monospace"
        >
          {token}
        </text>,
      )

      tokens.forEach((_, j) => {
        const isAllowed = !isMasked || j <= i
        const cx = offsetX + labelOffset + j * cellSize
        const cy = 48 + i * cellSize

        // Cell background
        elements.push(
          <rect
            key={`cell-${offsetX}-${i}-${j}`}
            x={cx}
            y={cy}
            width={cellSize}
            height={cellSize}
            fill={isAllowed ? '#6366f1' : '#1e1e2e'}
            opacity={isAllowed ? 0.25 : 0.6}
            stroke="#374151"
            strokeWidth={0.5}
          />,
        )

        // Cell content
        if (!isAllowed) {
          // X mark for blocked
          elements.push(
            <text
              key={`x-${offsetX}-${i}-${j}`}
              x={cx + cellSize / 2}
              y={cy + cellSize / 2 + 4}
              textAnchor="middle"
              fill="#ef4444"
              fontSize="12"
              opacity={0.5}
            >
              {'×'}
            </text>,
          )
        }

        // Highlight the "cheating" cell in unmasked version
        if (!isMasked && i === 2 && j === 3) {
          elements.push(
            <rect
              key={`highlight-${offsetX}`}
              x={cx + 1}
              y={cy + 1}
              width={cellSize - 2}
              height={cellSize - 2}
              fill="none"
              stroke="#f59e0b"
              strokeWidth={2}
              rx={3}
            />,
          )
          elements.push(
            <text
              key={`highlight-label-${offsetX}`}
              x={cx + cellSize / 2}
              y={cy + cellSize / 2 + 3}
              textAnchor="middle"
              fill="#f59e0b"
              fontSize="8"
              fontWeight="bold"
            >
              leak!
            </text>,
          )
        }
      })
    })

    return elements
  }

  const totalWidth = 2 * labelOffset + 2 * matrixSize + 40
  const totalHeight = 48 + matrixSize + 20

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={totalWidth}
        height={totalHeight}
        viewBox={`0 0 ${totalWidth} ${totalHeight}`}
        className="overflow-visible"
      >
        {renderMatrix(0, 'Full Attention (bidirectional)', false)}
        {renderMatrix(labelOffset + matrixSize + 40, 'Causal Mask (decoder-only)', true)}

        {/* Arrow between matrices */}
        <line
          x1={labelOffset + matrixSize + 8}
          y1={48 + matrixSize / 2}
          x2={labelOffset + matrixSize + 32}
          y2={48 + matrixSize / 2}
          stroke="#9ca3af"
          strokeWidth={1.5}
        />
        <polygon
          points={`${labelOffset + matrixSize + 28},${48 + matrixSize / 2 - 4} ${labelOffset + matrixSize + 34},${48 + matrixSize / 2} ${labelOffset + matrixSize + 28},${48 + matrixSize / 2 + 4}`}
          fill="#9ca3af"
        />
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Full GPT Architecture Diagram (inline SVG)
// ---------------------------------------------------------------------------

function GptArchitectureDiagram() {
  return (
    <div className="flex justify-center py-4">
      <svg
        width="400"
        height="720"
        viewBox="0 0 400 720"
        className="overflow-visible"
      >
        {/* ---- Legend ---- */}
        <rect x="20" y="10" width="10" height="10" rx="2" fill="#a78bfa" opacity={0.4} />
        <text x="35" y="19" fill="#9ca3af" fontSize="9">Residual stream</text>

        <rect x="20" y="28" width="10" height="10" rx="2" fill="#38bdf8" opacity={0.4} />
        <text x="35" y="37" fill="#9ca3af" fontSize="9">Transformer block</text>

        <rect x="20" y="46" width="10" height="10" rx="2" fill="#f59e0b" opacity={0.4} />
        <text x="35" y="55" fill="#9ca3af" fontSize="9">Output projection</text>

        <rect x="20" y="64" width="10" height="10" rx="2" fill="#34d399" opacity={0.4} />
        <text x="35" y="73" fill="#9ca3af" fontSize="9">Layer norm</text>

        <rect x="20" y="82" width="10" height="10" rx="2" fill="#c084fc" opacity={0.4} />
        <text x="35" y="91" fill="#9ca3af" fontSize="9">Embedding + PE</text>

        {/* ---- Residual stream backbone (dashed) ---- */}
        <line
          x1="200"
          y1="680"
          x2="200"
          y2="115"
          stroke="#a78bfa"
          strokeWidth={2.5}
          strokeDasharray="6,4"
          opacity={0.25}
        />

        {/* ---- Token IDs (bottom) ---- */}
        <rect
          x="120"
          y="680"
          width="160"
          height="30"
          rx="6"
          fill="#6b7280"
          opacity={0.15}
          stroke="#6b7280"
          strokeWidth={1}
        />
        <text
          x="200"
          y="699"
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="11"
          fontWeight="500"
        >
          Token IDs
        </text>
        <text
          x="290"
          y="699"
          fill="#6b7280"
          fontSize="9"
          fontFamily="monospace"
        >
          (n,)
        </text>

        {/* Arrow up */}
        <line x1="200" y1="680" x2="200" y2="660" stroke="#9ca3af" strokeWidth={1.5} />
        <polygon points="196,663 200,656 204,663" fill="#9ca3af" />

        {/* ---- Token Embedding ---- */}
        <rect
          x="120"
          y="628"
          width="160"
          height="28"
          rx="5"
          fill="#c084fc"
          opacity={0.12}
          stroke="#c084fc"
          strokeWidth={1.5}
        />
        <text
          x="200"
          y="646"
          textAnchor="middle"
          fill="#c084fc"
          fontSize="11"
          fontWeight="500"
        >
          Token Embedding
        </text>
        <text
          x="290"
          y="646"
          fill="#6b7280"
          fontSize="9"
          fontFamily="monospace"
        >
          (n, 768)
        </text>

        {/* Arrow up */}
        <line x1="200" y1="628" x2="200" y2="612" stroke="#c084fc" strokeWidth={1.5} />
        <polygon points="196,615 200,608 204,615" fill="#c084fc" />

        {/* ---- + Positional Encoding ---- */}
        <circle
          cx="200"
          cy="596"
          r="12"
          fill="none"
          stroke="#c084fc"
          strokeWidth={1.5}
        />
        <text
          x="200"
          y="600"
          textAnchor="middle"
          fill="#c084fc"
          fontSize="14"
          fontWeight="bold"
        >
          +
        </text>
        <text
          x="222"
          y="600"
          fill="#c084fc"
          fontSize="9"
          opacity={0.7}
        >
          PE
        </text>

        {/* Arrow up */}
        <line x1="200" y1="584" x2="200" y2="562" stroke="#a78bfa" strokeWidth={2} />
        <polygon points="196,565 200,558 204,565" fill="#a78bfa" />

        {/* ---- Transformer Block 1 ---- */}
        <rect
          x="100"
          y="498"
          width="200"
          height="60"
          rx="8"
          fill="#1e1e2e"
          stroke="#38bdf8"
          strokeWidth={1.5}
          opacity={0.9}
        />
        <rect x="110" y="508" width="70" height="16" rx="3" fill="#38bdf8" opacity={0.15} stroke="#38bdf8" strokeWidth={0.8} />
        <text x="145" y="520" textAnchor="middle" fill="#38bdf8" fontSize="8">MHA + mask</text>
        <rect x="190" y="508" width="100" height="16" rx="3" fill="#f59e0b" opacity={0.15} stroke="#f59e0b" strokeWidth={0.8} />
        <text x="240" y="520" textAnchor="middle" fill="#f59e0b" fontSize="8">FFN (768{'→'}3072{'→'}768)</text>
        <rect x="110" y="530" width="70" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="145" y="539" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <rect x="190" y="530" width="100" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="240" y="539" textAnchor="middle" fill="#34d399" fontSize="7">LN + Residual</text>
        <text x="315" y="533" fill="#9ca3af" fontSize="10" fontWeight="500">Block 1</text>

        {/* Arrow up */}
        <line x1="200" y1="498" x2="200" y2="480" stroke="#a78bfa" strokeWidth={1.5} />
        <polygon points="196,483 200,476 204,483" fill="#a78bfa" />

        {/* ---- Transformer Block 2 ---- */}
        <rect
          x="100"
          y="416"
          width="200"
          height="60"
          rx="8"
          fill="#1e1e2e"
          stroke="#38bdf8"
          strokeWidth={1.5}
          opacity={0.9}
        />
        <rect x="110" y="426" width="70" height="16" rx="3" fill="#38bdf8" opacity={0.15} stroke="#38bdf8" strokeWidth={0.8} />
        <text x="145" y="438" textAnchor="middle" fill="#38bdf8" fontSize="8">MHA + mask</text>
        <rect x="190" y="426" width="100" height="16" rx="3" fill="#f59e0b" opacity={0.15} stroke="#f59e0b" strokeWidth={0.8} />
        <text x="240" y="438" textAnchor="middle" fill="#f59e0b" fontSize="8">FFN</text>
        <rect x="110" y="448" width="70" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="145" y="457" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <rect x="190" y="448" width="100" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="240" y="457" textAnchor="middle" fill="#34d399" fontSize="7">LN + Residual</text>
        <text x="315" y="451" fill="#9ca3af" fontSize="10" fontWeight="500">Block 2</text>

        {/* Dots */}
        <text x="200" y="405" textAnchor="middle" fill="#6b7280" fontSize="18" fontWeight="bold">...</text>

        {/* ---- Transformer Block N ---- */}
        <rect
          x="100"
          y="338"
          width="200"
          height="60"
          rx="8"
          fill="#1e1e2e"
          stroke="#38bdf8"
          strokeWidth={1.5}
          opacity={0.9}
        />
        <rect x="110" y="348" width="70" height="16" rx="3" fill="#38bdf8" opacity={0.15} stroke="#38bdf8" strokeWidth={0.8} />
        <text x="145" y="360" textAnchor="middle" fill="#38bdf8" fontSize="8">MHA + mask</text>
        <rect x="190" y="348" width="100" height="16" rx="3" fill="#f59e0b" opacity={0.15} stroke="#f59e0b" strokeWidth={0.8} />
        <text x="240" y="360" textAnchor="middle" fill="#f59e0b" fontSize="8">FFN</text>
        <rect x="110" y="370" width="70" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="145" y="379" textAnchor="middle" fill="#34d399" fontSize="7">LN</text>
        <rect x="190" y="370" width="100" height="12" rx="2" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.5} />
        <text x="240" y="379" textAnchor="middle" fill="#34d399" fontSize="7">LN + Residual</text>
        <text x="315" y="373" fill="#9ca3af" fontSize="10" fontWeight="500">Block 12</text>

        {/* Arrow up */}
        <line x1="200" y1="338" x2="200" y2="318" stroke="#a78bfa" strokeWidth={1.5} />
        <polygon points="196,321 200,314 204,321" fill="#a78bfa" />

        {/* ---- Final Layer Norm ---- */}
        <rect
          x="130"
          y="286"
          width="140"
          height="28"
          rx="5"
          fill="#34d399"
          opacity={0.12}
          stroke="#34d399"
          strokeWidth={1.5}
        />
        <text
          x="200"
          y="304"
          textAnchor="middle"
          fill="#34d399"
          fontSize="11"
          fontWeight="500"
        >
          Final LayerNorm
        </text>
        <text
          x="280"
          y="304"
          fill="#6b7280"
          fontSize="9"
          fontFamily="monospace"
        >
          (n, 768)
        </text>

        {/* Arrow up */}
        <line x1="200" y1="286" x2="200" y2="266" stroke="#34d399" strokeWidth={1.5} />
        <polygon points="196,269 200,262 204,269" fill="#34d399" />

        {/* ---- Output Projection ---- */}
        <rect
          x="110"
          y="232"
          width="180"
          height="30"
          rx="6"
          fill="#f59e0b"
          opacity={0.12}
          stroke="#f59e0b"
          strokeWidth={1.5}
        />
        <text
          x="200"
          y="251"
          textAnchor="middle"
          fill="#f59e0b"
          fontSize="11"
          fontWeight="600"
        >
          Output Projection (Unembedding)
        </text>
        <text
          x="300"
          y="251"
          fill="#6b7280"
          fontSize="9"
          fontFamily="monospace"
        >
          (n, 50257)
        </text>

        {/* Arrow up */}
        <line x1="200" y1="232" x2="200" y2="212" stroke="#f59e0b" strokeWidth={1.5} />
        <polygon points="196,215 200,208 204,215" fill="#f59e0b" />

        {/* ---- Softmax ---- */}
        <rect
          x="130"
          y="180"
          width="140"
          height="28"
          rx="5"
          fill="#6366f1"
          opacity={0.12}
          stroke="#6366f1"
          strokeWidth={1.5}
        />
        <text
          x="200"
          y="198"
          textAnchor="middle"
          fill="#6366f1"
          fontSize="11"
          fontWeight="500"
        >
          Softmax
        </text>

        {/* Arrow up */}
        <line x1="200" y1="180" x2="200" y2="160" stroke="#6366f1" strokeWidth={1.5} />
        <polygon points="196,163 200,156 204,163" fill="#6366f1" />

        {/* ---- Probability Distribution (top) ---- */}
        <rect
          x="110"
          y="125"
          width="180"
          height="30"
          rx="6"
          fill="#6366f1"
          opacity={0.15}
          stroke="#6366f1"
          strokeWidth={1.5}
        />
        <text
          x="200"
          y="144"
          textAnchor="middle"
          fill="#6366f1"
          fontSize="11"
          fontWeight="600"
        >
          P(next token | context)
        </text>

        {/* ---- Weight tying annotation ---- */}
        <line
          x1="115"
          y1="644"
          x2="80"
          y2="644"
          stroke="#c084fc"
          strokeWidth={1}
          strokeDasharray="3,2"
          opacity={0.5}
        />
        <line
          x1="80"
          y1="644"
          x2="80"
          y2="247"
          stroke="#c084fc"
          strokeWidth={1}
          strokeDasharray="3,2"
          opacity={0.5}
        />
        <line
          x1="80"
          y1="247"
          x2="108"
          y2="247"
          stroke="#c084fc"
          strokeWidth={1}
          strokeDasharray="3,2"
          opacity={0.5}
        />
        <text
          x="62"
          y="445"
          fill="#c084fc"
          fontSize="8"
          opacity={0.6}
          fontStyle="italic"
          transform="rotate(-90, 62, 445)"
        >
          weight tying (shared W)
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function DecoderOnlyTransformersLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Decoder-Only Transformers"
            description="One new mechanism, then assemble every piece you've built into the complete GPT architecture."
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
            Understand how <strong>causal masking</strong> prevents tokens from
            attending to future positions, then assemble the complete GPT
            architecture end-to-end: token embedding + positional encoding{' '}
            {'→'} N transformer blocks with causal masking {'→'}{' '}
            output projection {'→'} next-token probabilities.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            The complete transformer block (MHA + FFN + residual + layer norm),
            the input pipeline (tokenization, embeddings, positional encoding),
            and autoregressive generation. Every piece is familiar&mdash;this
            lesson adds one new mechanism and connects everything.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Causal masking: why it exists, how it works, where it operates',
              'The complete GPT architecture assembled end-to-end',
              'Total parameter counting for GPT-2 (~124M verified)',
              'Why decoder-only won for LLMs (brief)',
              'Encoder-decoder contrast for naming context',
              'NOT: implementing in PyTorch—that&apos;s Module 4.3',
              'NOT: training, loss curves, or learning rate scheduling—Module 4.3',
              'NOT: KV caching, flash attention, or efficient inference—Module 4.3',
              'NOT: finetuning, instruction tuning, or RLHF—Module 4.4',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook: The Cheating Problem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Cheating Problem"
            subtitle="Something is wrong with this attention matrix"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The previous lesson ended with a promise:{' '}
              <em>
                &ldquo;Stack N blocks. Add causal masking. That&rsquo;s GPT.&rdquo;
              </em>{' '}
              Before we can deliver, we need to see why causal masking exists.
              Consider this scenario.
            </p>

            <p className="text-muted-foreground">
              You are training a language model on the sequence{' '}
              <strong>&ldquo;The cat sat on the mat.&rdquo;</strong> The task
              is next-token prediction: at each position, predict what comes
              next. At position 3 (&ldquo;sat&rdquo;), the model should predict
              &ldquo;on.&rdquo;
            </p>

            <p className="text-muted-foreground">
              Now look at the attention matrix below. Every token can attend to
              every other token&mdash;the full bidirectional attention you have
              been computing for five lessons. Look at row 3 (&ldquo;sat&rdquo;).
            </p>

            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-amber-400">
                The problem: position 3 attends to position 4
              </p>
              <p className="text-sm text-muted-foreground">
                &ldquo;sat&rdquo; has a high attention weight on
                &ldquo;on&rdquo;&mdash;the <strong>exact token it is trying to
                predict</strong>. The model is not learning to predict the next
                token. It is looking at the answer and copying it.
              </p>
            </div>

            <p className="text-muted-foreground">
              This is the same problem as data leakage in supervised learning:
              the model sees the labels during training and achieves perfect
              training loss, but it has learned nothing useful. At inference
              time, when the model must generate token 4, token 4 does not exist
              yet. A model trained with full attention has learned to copy, not
              to predict.
            </p>

            <p className="text-muted-foreground">
              <strong>The question:</strong> how do we train on full sequences
              efficiently&mdash;getting{' '}
              <InlineMath math="n" /> training examples from one forward
              pass&mdash;without letting each position see its own answer?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Data Leakage">
            This is the same problem you saw in Series 1: if the model
            can see the labels during training, validation metrics are
            meaningless. Here, the &ldquo;label&rdquo; for position 3 is the
            token at position 4. Looking at position 4 IS leaking the label.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain: Causal Masking (core new concept)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Causal Masking"
            subtitle="The constraint that makes parallel training safe"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The solution is a constraint: <strong>position{' '}
              <InlineMath math="i" /> can only attend to positions 1
              through <InlineMath math="i" /></strong>. Never to positions{' '}
              <InlineMath math="i+1" /> and beyond. This is called{' '}
              <strong>causal masking</strong> (or &ldquo;autoregressive
              masking&rdquo;).
            </p>

            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-violet-400">
                Analogy: the exam with an answer key
              </p>
              <p className="text-sm text-muted-foreground">
                Imagine taking a test where the answer key is printed next to
                each question. You would get a perfect score, but you would
                learn nothing. Causal masking is the cardboard sleeve that
                covers the answers below the question you are currently working
                on. During the test (training), the sleeve ensures you can only
                see what came before. After the test (inference), there IS no
                answer key&mdash;you are generating the answers yourself.
              </p>
            </div>

            <p className="text-muted-foreground">
              <strong>The mechanism:</strong> before applying softmax to the
              attention scores, set all entries where{' '}
              <InlineMath math="j > i" /> to{' '}
              <InlineMath math="-\infty" />. Since{' '}
              <InlineMath math="e^{-\infty} = 0" />, those positions contribute
              exactly zero after softmax. The remaining positions automatically
              renormalize&mdash;their weights still sum to 1.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium text-muted-foreground">
                Same scaled dot-product attention, one additional step:
              </p>
              <BlockMath math="\text{scores} = \frac{QK^T}{\sqrt{d_k}}" />
              <BlockMath math="\text{masked\_scores}_{ij} = \begin{cases} \text{scores}_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}" />
              <BlockMath math="\text{weights} = \text{softmax}(\text{masked\_scores})" />
            </div>

            <CausalMaskDiagram />

            <p className="text-muted-foreground">
              The triangular shape is the defining visual. Row 1 (&ldquo;The&rdquo;)
              sees only itself. Row 3 (&ldquo;sat&rdquo;) sees &ldquo;The,&rdquo;
              &ldquo;cat,&rdquo; and &ldquo;sat&rdquo;&mdash;but not
              &ldquo;on,&rdquo; &ldquo;the,&rdquo; or &ldquo;mat.&rdquo; Row 6
              (&ldquo;mat&rdquo;) sees the entire sequence. Each row represents
              an increasingly wide context window.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Triangular Pattern">
            The causal mask is a lower-triangular matrix. Row{' '}
            <InlineMath math="i" /> has exactly <InlineMath math="i" />{' '}
            non-zero entries. This is the visual signature of
            autoregressive attention&mdash;you will see it everywhere.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Worked Example: Masking in action
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Worked Example"
            subtitle='Row 3 ("sat"): before and after masking'
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider the attention computation for position 3
              (&ldquo;sat&rdquo;). The raw scores from{' '}
              <InlineMath math="QK^T / \sqrt{d_k}" /> might look like this:
            </p>

            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-3">
              <div className="text-sm text-muted-foreground space-y-2">
                <p>
                  <strong>Raw scores (row 3):</strong>{' '}
                  <InlineMath math="[2.1,\; 3.5,\; 1.8,\; 4.2,\; 0.9,\; 1.1]" />
                </p>
                <p>
                  <strong>After masking:</strong>{' '}
                  <InlineMath math="[2.1,\; 3.5,\; 1.8,\; -\infty,\; -\infty,\; -\infty]" />
                </p>
                <p>
                  <strong>After softmax:</strong>{' '}
                  <InlineMath math="[0.27,\; 0.55,\; 0.18,\; 0,\; 0,\; 0]" />
                </p>
              </div>
              <p className="text-xs text-muted-foreground/70">
                The weights for positions 1&ndash;3 sum to 1.0. Positions 4&ndash;6
                contribute exactly nothing. The softmax redistributes all probability
                mass to the unmasked positions automatically.
              </p>
            </div>

            <p className="text-muted-foreground">
              Notice that position 4 (&ldquo;on&rdquo;) had the{' '}
              <em>highest</em> raw score (4.2)&mdash;the model genuinely finds
              it relevant. But the mask eliminates it completely. This is the
              point: the model must learn to predict &ldquo;on&rdquo; using only
              the context &ldquo;The cat sat,&rdquo; not by peeking at the
              answer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Renormalization Is Free">
            Softmax naturally sums to 1 over all finite inputs. Setting entries
            to <InlineMath math="-\infty" /> before softmax means those entries
            become 0 and the remaining entries absorb all the probability
            mass. No special renormalization step is needed.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Misconception #1: Masking is not a training trick
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>A critical distinction:</strong> causal masking is not a
              training trick that gets removed at test time. It is not like
              dropout (which is turned off during evaluation) or batch norm
              (which uses running averages). At inference time, future tokens
              do not exist. When the model is generating token 6, tokens 7, 8,
              9, and beyond have not been produced yet. There is nothing to
              unmask.
            </p>

            <p className="text-muted-foreground">
              Causal masking during training{' '}
              <strong>simulates the inference constraint</strong>. The model
              practices predicting the next token using only past context
              because that is all it will ever have when generating text. The
              mask is not a restriction added to make training harder&mdash;it
              is a reflection of reality.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Like Dropout">
            Dropout: on during training, off during inference. Causal masking:
            on during training because it mirrors the inference reality. At
            inference, the mask is not &ldquo;removed&rdquo;&mdash;future tokens
            simply do not exist yet.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Misconception #2: Parallel training
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Training is parallel, not sequential.</strong> A common
              confusion: since the model predicts &ldquo;one token at a
              time&rdquo; during generation, it must process tokens one at a
              time during training too. This is wrong.
            </p>

            <p className="text-muted-foreground">
              During training, the model processes the{' '}
              <strong>entire sequence simultaneously</strong> in one forward
              pass. Every position predicts its next token in parallel.
              Position 1 predicts token 2, position 2 predicts token 3,
              position 5 predicts token 6&mdash;all at the same time. This
              gives us <InlineMath math="n" /> training examples from a single
              sequence of <InlineMath math="n" /> tokens.
            </p>

            <ComparisonRow
              left={{
                title: 'Training',
                color: 'blue',
                items: [
                  'Entire sequence processed in parallel',
                  'All positions predict simultaneously',
                  'Causal mask prevents future leakage',
                  'N training examples per sequence',
                ],
              }}
              right={{
                title: 'Inference',
                color: 'cyan',
                items: [
                  'Tokens generated one at a time',
                  'Each new token appended to context',
                  'No mask needed—future tokens don’t exist',
                  'Autoregressive loop from Module 4.1',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Causal masking is what makes parallel training safe. Position 3
              only attends to positions 1&ndash;3 even though positions 4, 5, 6 are
              right there in the same tensor. Without the mask, parallel
              training would leak future information.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Efficiency Payoff">
            Sequential token-by-token training would be catastrophically
            slow. A 1024-token sequence gives 1024 training examples in a
            single forward pass, but only if causal masking prevents leakage.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Check 1: The mask in action
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Check: The Mask in Action" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>1.</strong> Draw the causal mask for a 4-token sequence.
                Which entries are <InlineMath math="-\infty" />? After softmax,
                what is the sum of weights in row 1? In row 4?
              </p>
              <p>
                <strong>2.</strong> Position 5 in a 10-token sequence. How many
                tokens can it attend to? What happens to positions 6&ndash;10 in the
                attention computation?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>1.</strong> 4{'×'}4 lower-triangular matrix.
                    The 6 entries above the diagonal are{' '}
                    <InlineMath math="-\infty" />. Row 1 has one entry (which
                    must be 1.0 after softmax). Row 4 has four entries (which
                    sum to 1.0). Both rows sum to 1&mdash;softmax always sums
                    to 1 over the unmasked entries.
                  </p>
                  <p>
                    <strong>2.</strong> Position 5 can attend to 5 tokens
                    (positions 1&ndash;5). Positions 6&ndash;10 are set to{' '}
                    <InlineMath math="-\infty" /> before softmax, producing
                    weights of exactly 0. They contribute nothing to
                    position 5&rsquo;s output.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          9. Explain: The Output Projection (gap resolution)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Output Projection"
            subtitle="From hidden states to next-token probabilities"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Embeddings &amp; Positional Encoding, you saw a brief aside:
              embedding maps vocabulary {'→'}{' '}
              <InlineMath math="d_{\text{model}}" />, and the output layer maps{' '}
              <InlineMath math="d_{\text{model}}" /> {'→'} vocabulary. Now
              we make that concrete.
            </p>

            <p className="text-muted-foreground">
              The final transformer block produces an output of shape{' '}
              <InlineMath math="(n, d_{\text{model}})" />&mdash;one 768-dimensional
              vector per token position. To predict the next token, we need a
              probability distribution over the entire vocabulary of 50,257
              tokens.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\text{logits} = x \cdot W_{\text{out}} \quad \text{where } W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times V}" />
              <div className="text-sm text-muted-foreground space-y-1">
                <p>
                  <InlineMath math="x" />: output from final block, shape{' '}
                  <InlineMath math="(n, 768)" />
                </p>
                <p>
                  <InlineMath math="W_{\text{out}}" />: shape{' '}
                  <InlineMath math="(768, 50257)" />&mdash;an{' '}
                  <code className="text-xs">nn.Linear</code>
                </p>
                <p>
                  <InlineMath math="\text{logits}" />: shape{' '}
                  <InlineMath math="(n, 50257)" />&mdash;raw scores for every
                  token in the vocabulary
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Apply softmax to the logits, and you get{' '}
              <InlineMath math="P(\text{next token} \mid \text{context})" />{' '}
              for each position. Same idea as the MNIST output layer from
              Series 1&mdash;instead of 10 digit classes, you have 50,257 token
              classes.
            </p>

            <p className="text-muted-foreground">
              <strong>Weight tying:</strong> many models, including GPT-2, share
              the embedding weight matrix with the output projection
              (transposed). The embedding matrix has shape{' '}
              <InlineMath math="(V, d_{\text{model}})" /> and the output
              projection needs shape{' '}
              <InlineMath math="(d_{\text{model}}, V)" />&mdash;they are
              transposes of each other. Sharing this matrix means the model
              learns a single mapping between token space and embedding space,
              and it saves ~38M parameters.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Embedding in Reverse">
            Embedding maps a token ID to a 768-dimensional vector. The output
            projection maps a 768-dimensional vector back to a score for every
            token ID. Same mapping, opposite direction.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Explain: Assembling the Full Architecture
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete GPT Architecture"
            subtitle="Every piece, assembled end-to-end"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every component you have learned across Modules 4.1 and 4.2
              connects into a single forward pass. Here is the complete data
              flow:
            </p>

            <div className="space-y-2">
              <div className="px-4 py-2 bg-muted/30 rounded-lg text-sm text-muted-foreground">
                <strong>1.</strong> Tokenize input text (BPE from Tokenization)
              </div>
              <div className="px-4 py-2 bg-muted/30 rounded-lg text-sm text-muted-foreground">
                <strong>2.</strong> Look up token embeddings:{' '}
                <InlineMath math="\text{nn.Embedding}(50257, 768)" />
              </div>
              <div className="px-4 py-2 bg-muted/30 rounded-lg text-sm text-muted-foreground">
                <strong>3.</strong> Add positional encoding:{' '}
                <InlineMath math="\text{input}_i = \text{embed}(\text{token}_i) + \text{PE}(i)" />
              </div>
              <div className="px-4 py-2 bg-sky-500/10 border border-sky-500/20 rounded-lg text-sm text-muted-foreground">
                <strong>4.</strong> Pass through{' '}
                <InlineMath math="N" /> transformer blocks, each with{' '}
                <strong>causal masking</strong> in the attention
              </div>
              <div className="px-4 py-2 bg-muted/30 rounded-lg text-sm text-muted-foreground">
                <strong>5.</strong> Apply final layer norm
              </div>
              <div className="px-4 py-2 bg-muted/30 rounded-lg text-sm text-muted-foreground">
                <strong>6.</strong> Output projection:{' '}
                <InlineMath math="\text{nn.Linear}(768, 50257)" />
              </div>
              <div className="px-4 py-2 bg-muted/30 rounded-lg text-sm text-muted-foreground">
                <strong>7.</strong> Softmax {'→'} probability distribution
                over next token
              </div>
            </div>

            <GptArchitectureDiagram />

            <div className="px-4 py-3 bg-sky-500/10 border border-sky-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-sky-400">
                GPT-2 Configuration
              </p>
              <div className="grid gap-2 grid-cols-2 md:grid-cols-3 text-sm text-muted-foreground">
                <div>
                  <span className="text-sky-400/70 text-xs">Vocabulary</span>
                  <p className="font-mono text-xs">50,257</p>
                </div>
                <div>
                  <span className="text-sky-400/70 text-xs">d_model</span>
                  <p className="font-mono text-xs">768</p>
                </div>
                <div>
                  <span className="text-sky-400/70 text-xs">Layers</span>
                  <p className="font-mono text-xs">12</p>
                </div>
                <div>
                  <span className="text-sky-400/70 text-xs">Heads</span>
                  <p className="font-mono text-xs">12</p>
                </div>
                <div>
                  <span className="text-sky-400/70 text-xs">d_ff</span>
                  <p className="font-mono text-xs">3,072</p>
                </div>
                <div>
                  <span className="text-sky-400/70 text-xs">Context</span>
                  <p className="font-mono text-xs">1,024</p>
                </div>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Every Piece Is Familiar">
            Look at the architecture diagram. Token embedding: Embeddings &amp;
            Positional Encoding. Transformer blocks: The Transformer Block.
            Causal masking: this lesson. Output projection: this lesson. There
            is nothing new&mdash;only the assembly.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Parameter Counting (closure)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Counting Every Parameter"
            subtitle="Verifying GPT-2 at ~124M"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In The Transformer Block, you computed per-block parameter counts.
              Now add up the entire model. Every number below uses GPT-2&rsquo;s
              configuration.
            </p>

            <div className="space-y-3">
              <div className="px-4 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-violet-400">Token Embeddings</p>
                <p>
                  <InlineMath math="50{,}257 \times 768 = 38{,}597{,}376" />
                </p>
              </div>

              <div className="px-4 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-violet-400">Position Embeddings</p>
                <p>
                  <InlineMath math="1{,}024 \times 768 = 786{,}432" />
                </p>
              </div>

              <div className="px-4 py-2 bg-sky-500/10 border border-sky-500/20 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-sky-400">Per Block: Attention (Q + K + V + O)</p>
                <p>
                  <InlineMath math="4 \times 768^2 = 2{,}359{,}296" />
                </p>
              </div>

              <div className="px-4 py-2 bg-amber-500/10 border border-amber-500/20 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-amber-400">Per Block: FFN</p>
                <p>
                  <InlineMath math="2 \times 768 \times 3{,}072 = 4{,}718{,}592" />
                </p>
              </div>

              <div className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-emerald-400">Per Block: Layer Norms (2 per block)</p>
                <p>
                  <InlineMath math="2 \times 2 \times 768 = 3{,}072" />
                </p>
              </div>

              <div className="px-4 py-2 bg-muted/50 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-muted-foreground">12 Blocks Total</p>
                <p>
                  <InlineMath math="12 \times (2{,}359{,}296 + 4{,}718{,}592 + 3{,}072) = 84{,}971{,}520" />
                </p>
              </div>

              <div className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-emerald-400">Final Layer Norm</p>
                <p>
                  <InlineMath math="2 \times 768 = 1{,}536" />
                </p>
              </div>

              <div className="px-4 py-2 bg-muted/50 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-muted-foreground">Output Projection</p>
                <p>
                  Weight-tied with token embeddings&mdash;0 additional parameters
                </p>
              </div>
            </div>

            <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg text-center space-y-1">
              <p className="text-sm text-muted-foreground">
                <strong>Total:</strong>{' '}
                <InlineMath math="38{,}597{,}376 + 786{,}432 + 84{,}971{,}520 + 1{,}536" />
              </p>
              <p className="text-lg font-bold text-primary">
                = 124,356,864 (~124.4M)
              </p>
              <p className="text-xs text-muted-foreground/70">
                This matches the known GPT-2 &ldquo;small&rdquo; model at 124M parameters.
                (These counts omit bias terms, which add ~83K&mdash;negligible at this scale.)
              </p>
            </div>

            <p className="text-muted-foreground">
              The distribution holds from The Transformer Block: embeddings
              account for ~31%, attention layers ~23%, and FFN layers ~46%. The
              FFN stores the model&rsquo;s learned knowledge in nearly half of
              all parameters.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Satisfying Closure">
            In The Transformer Block, you saw ~7.1M per block and knew GPT-2
            had ~124M total. Now you can account for every parameter: 12 blocks
            + embeddings + position embeddings + final layer norm. The numbers
            add up.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Check 2: Architecture Comprehension
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Check: Architecture Comprehension" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A colleague says they want to make GPT-2 better by doubling{' '}
                <InlineMath math="d_{\text{model}}" /> from 768 to 1536 while
                keeping everything else the same. How does this affect:
              </p>
              <p>
                <strong>(a)</strong> Per-block parameters?
              </p>
              <p>
                <strong>(b)</strong> Total model parameters?
              </p>
              <p>
                <strong>(c)</strong> Context length?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>(a)</strong> Attention parameters quadruple
                    (<InlineMath math="4 \times d_{\text{model}}^2" />, and{' '}
                    <InlineMath math="1536^2 = 4 \times 768^2" />). FFN
                    parameters <strong>double</strong>, not quadruple: the
                    question says &ldquo;everything else the same,&rdquo; so{' '}
                    <InlineMath math="d_{\text{ff}}" /> stays at 3072. Each FFN
                    matrix goes from{' '}
                    <InlineMath math="768 \times 3072" /> to{' '}
                    <InlineMath math="1536 \times 3072" />&mdash;2x. (In
                    practice, <InlineMath math="d_{\text{ff}}" /> is usually set
                    to <InlineMath math="4 \times d_{\text{model}}" />, which
                    would make FFN also quadruple&mdash;but the question
                    specifies keeping <InlineMath math="d_{\text{ff}}" /> fixed.)
                  </p>
                  <p>
                    <strong>(b)</strong> Attention parameters quadruple,
                    FFN parameters double, embedding parameters double. Overall
                    somewhere between 2x and 4x&mdash;roughly 3x total, since
                    both FFN and embeddings (the larger components) only double.
                  </p>
                  <p>
                    <strong>(c)</strong> Context length is{' '}
                    <strong>unchanged</strong>. It is a separate hyperparameter
                    (1024 in GPT-2) controlled by the positional encoding, not
                    by <InlineMath math="d_{\text{model}}" />.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Elaborate: Why Decoder-Only Won
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why &ldquo;Decoder-Only&rdquo;?"
            subtitle="The original Transformer, and why GPT kept only half"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The original 2017 Transformer was designed for machine
              translation. It had two separate stacks:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Encoder:</strong> processes the input sentence with{' '}
                <strong>bidirectional</strong> attention (every token sees every
                other token). No causal mask.
              </li>
              <li>
                <strong>Decoder:</strong> generates the output sentence with{' '}
                <strong>causal</strong> attention (each token sees only past
                outputs), plus <strong>cross-attention</strong> to read from the
                encoder&rsquo;s output.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Three architectural variants emerged:
            </p>

            <div className="grid gap-3 md:grid-cols-3">
              <GradientCard title="Encoder-Only" color="amber">
                <div className="text-sm space-y-1">
                  <p><strong>Example:</strong> BERT</p>
                  <p>Bidirectional attention</p>
                  <p>Great for understanding (classification, NER)</p>
                  <p>Cannot generate text autoregressively</p>
                </div>
              </GradientCard>
              <GradientCard title="Encoder-Decoder" color="violet">
                <div className="text-sm space-y-1">
                  <p><strong>Example:</strong> T5, original Transformer</p>
                  <p>Two stacks + cross-attention</p>
                  <p>Designed for sequence-to-sequence</p>
                  <p>More complex to scale</p>
                </div>
              </GradientCard>
              <GradientCard title="Decoder-Only" color="emerald">
                <div className="text-sm space-y-1">
                  <p><strong>Example:</strong> GPT-2, GPT-3, GPT-4</p>
                  <p>Causal attention only</p>
                  <p>One stack, one objective</p>
                  <p>Generates AND understands</p>
                </div>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              GPT kept only the decoder stack&mdash;causal attention, no
              encoder, no cross-attention. This is where the name
              &ldquo;decoder-only&rdquo; comes from. But the name is
              misleading.
            </p>

            <p className="text-muted-foreground">
              <strong>&ldquo;Decoder&rdquo; does not mean &ldquo;can only
              decode.&rdquo;</strong> It means &ldquo;uses causal masking.&rdquo;
              GPT models demonstrably understand their input&mdash;they answer
              questions, summarize text, follow instructions. The input IS the
              context, processed through all <InlineMath math="N" /> blocks with
              the same attention + FFN machinery. The name is historical,
              inherited from the original encoder-decoder Transformer, not a
              description of capability.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The Name Is Historical">
            &ldquo;Decoder-only&rdquo; does not mean the model can only output
            text and cannot understand it. It means the model uses the decoder
            architecture (causal masking) without a separate encoder. GPT
            models understand through the same mechanism they generate with.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          14. Why decoder-only won + GPT-2 vs GPT-3
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Why did decoder-only win for LLMs?
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Simplicity:</strong> one stack, one attention type, one
                training objective (next-token prediction). No encoder, no
                cross-attention, no encoder-decoder alignment to debug.
              </li>
              <li>
                <strong>Scaling:</strong> next-token prediction on vast text
                corpora is a simple, universal training signal that scales with
                data and compute. The scaling laws showed that decoder-only
                models improve predictably with scale.
              </li>
              <li>
                <strong>Generality:</strong> one model handles generation AND
                understanding. No need for separate architectures per task.
                Prompt the model differently, get different behaviors.
              </li>
            </ul>

            <p className="text-muted-foreground">
              The simplest architecture that works is the one that scales. And
              scale is what produced GPT-3 and beyond.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium text-muted-foreground">
                Same architecture, different scale
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-muted-foreground">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4"></th>
                      <th className="text-right py-2 px-3 text-sky-400">GPT-2</th>
                      <th className="text-right py-2 px-3 text-violet-400">GPT-3</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4">Layers</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">12</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">96</td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4">d_model</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">768</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">12,288</td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4">Heads</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">12</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">96</td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4">d_k</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">64</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">128</td>
                    </tr>
                    <tr>
                      <td className="py-1.5 pr-4 font-medium">Parameters</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs text-sky-400">124M</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs text-violet-400">175B</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground/70">
                The architecture you just learned IS the architecture behind
                GPT-3. What changed between versions is the scale, not the
                blueprint. GPT-3 has 1,400x more parameters with the same
                decoder-only transformer design.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Scale, Not Architecture">
            GPT-2 and GPT-3 are the same architecture with different
            hyperparameters. The decoder-only transformer you just learned IS
            the architecture behind the most capable LLMs. The blueprint
            does not change&mdash;only the numbers do.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          15. Brief aside: efficiency of masking (misconception #4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>A brief note on efficiency:</strong> you might wonder
              whether computing the full{' '}
              <InlineMath math="QK^T" /> matrix and then zeroing out the upper
              triangle wastes compute. Valid concern. In practice,
              implementations can avoid computing the masked entries
              entirely&mdash;techniques like flash attention fuse the masking
              with the attention computation. For understanding the
              architecture, the conceptual picture&mdash;compute scores, mask,
              softmax&mdash;is correct and complete. The compute optimization
              is a Module 4.3 topic.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          16. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Causal masking: set future positions to -∞ before softmax.',
                description:
                  'Each position can only attend to past and present. This prevents data leakage during parallel training and mirrors the inference reality where future tokens do not exist.',
              },
              {
                headline:
                  'The full GPT architecture is assembly, not invention.',
                description:
                  'Token embedding + positional encoding → N transformer blocks (with causal masking) → final layer norm → output projection → softmax. Every piece is something you already know.',
              },
              {
                headline:
                  'GPT-2: ~124M parameters. GPT-3: ~175B. Same architecture.',
                description:
                  'The decoder-only transformer you just learned IS the architecture behind modern LLMs. What changes between versions is the scale—layers, d_model, heads—not the blueprint.',
              },
              {
                headline:
                  '"Decoder-only" means causal masking, not "can only decode."',
                description:
                  'The name is historical, from the original encoder-decoder Transformer. The simplicity of one stack, one objective, one attention type is what enabled scaling to hundreds of billions of parameters.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Module completion echo */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
            <p className="text-sm font-medium text-violet-400">
              The journey through Module 4.2
            </p>
            <p className="text-sm text-muted-foreground">
              Over six lessons, you built the entire architecture behind modern
              LLMs. Lesson 1: raw attention (feel the limitation). Lesson 2: Q
              and K (fix the matching problem). Lesson 3: V and the residual
              stream (fix the contribution problem). Lesson 4: multi-head
              attention (capture diverse relationships). Lesson 5: the
              transformer block (assemble the repeating unit). This lesson:
              causal masking and the complete architecture. Every piece exists
              because the previous version was insufficient.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          17. Module Complete
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="4.2"
            title="Attention & the Transformer"
            achievements={[
              'Raw dot-product attention and context-dependent representations',
              'Q/K projections and scaled dot-product attention',
              'V projections, attention output, and the residual stream',
              'Multi-head attention with dimension splitting and W_O',
              'The transformer block: MHA + FFN + residual + layer norm',
              'Causal masking and the complete decoder-only GPT architecture',
            ]}
            nextModule="4.3"
            nextTitle="Building & Training GPT"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          18. Next Step (Module 4.3 seed)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">
              What comes next
            </p>
            <p className="text-sm text-muted-foreground">
              You understand the complete architecture. In Module 4.3, you will{' '}
              <strong>build it</strong>. nanoGPT: assemble every component in
              PyTorch, train it on real text, watch the loss drop and the
              generated text improve from random noise to coherent English. Then
              you will load GPT-2&rsquo;s actual pretrained weights into your
              architecture and generate text with a real model. The architecture
              is the blueprint. Next, you build the house.
            </p>
          </div>
        </Row.Content>
      </Row>
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Language Models are Unsupervised Multitask Learners',
                authors: 'Radford et al., 2019',
                url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
                note: 'The GPT-2 paper describing the decoder-only transformer architecture with the exact configuration analyzed in this lesson.',
              },
            ]}
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
