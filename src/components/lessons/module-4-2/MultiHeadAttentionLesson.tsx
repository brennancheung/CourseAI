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
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Multi-Head Attention
 *
 * Fourth lesson in Module 4.2 (Attention & the Transformer).
 * Seventh lesson in Series 4 (LLMs & Transformers).
 *
 * Extends single-head attention to multi-head: multiple smaller heads
 * running in parallel, each capturing a different notion of relevance,
 * with concatenation + W_O output projection to combine their findings.
 *
 * Core concepts at DEVELOPED:
 * - Multiple heads operating in parallel on lower-dimensional subspaces
 * - Dimension splitting: d_k = d_model / h (budget allocation, not compute multiplication)
 * - Output projection W_O as learned cross-head mixing
 *
 * Core concepts at INTRODUCED:
 * - Head specialization patterns in trained models (messy, emergent)
 *
 * Concepts REINFORCED:
 * - Single-head attention formula (each head runs the same formula)
 * - Q, K, V as learned projections (now per-head)
 * - "Three lenses, one embedding" (now three lenses per head)
 * - nn.Linear (W_O is another instance)
 * - Concatenation (torch.cat on head outputs)
 * - Residual stream (multi-head output goes into it)
 *
 * EXPLICITLY NOT COVERED:
 * - The transformer block (attention + FFN + residual + layer norm) -- Lesson 5
 * - Causal masking -- Lesson 6
 * - Cross-attention -- out of scope for this module
 * - nn.MultiheadAttention as a PyTorch module -- student builds from scratch
 * - How specialization emerges during training -- Module 4.3
 *
 * Previous: Values and the Attention Output (module 4.2, lesson 3)
 * Next: The Transformer Block (module 4.2, lesson 5)
 */

// ---------------------------------------------------------------------------
// 4-token embeddings, now d_model=6 (extended from d=3 in Lessons 1-3)
// Keeps the same tokens for continuity. First 3 dims are the original
// embeddings; dims 4-6 are new to support splitting into 2 heads of d_k=3.
// ---------------------------------------------------------------------------

const TINY_EMBEDDINGS_6D = [
  { token: 'The', vec: [0.9, 0.1, -0.2, 0.3, -0.1, 0.5] },
  { token: 'cat', vec: [-0.3, 0.8, 0.5, 0.7, 0.2, -0.4] },
  { token: 'sat', vec: [0.1, 0.6, 0.4, -0.2, 0.9, 0.1] },
  { token: 'here', vec: [0.4, -0.2, 0.7, 0.1, 0.5, 0.8] },
]

// ---------------------------------------------------------------------------
// Head 1 projection matrices (6x3 each)
// ---------------------------------------------------------------------------

const W_Q1 = [
  [1, 0, -1],
  [0, 1, 1],
  [1, -1, 0],
  [0, 0, 1],
  [-1, 1, 0],
  [0, -1, 1],
]

const W_K1 = [
  [0, 1, 0],
  [-1, 0, 1],
  [1, 1, -1],
  [0, -1, 0],
  [1, 0, 1],
  [-1, 1, 0],
]

const W_V1 = [
  [1, -1, 0],
  [0, 0, 1],
  [-1, 1, 1],
  [0, 1, 0],
  [1, 0, -1],
  [0, -1, 1],
]

// ---------------------------------------------------------------------------
// Head 2 projection matrices (6x3 each) -- different from Head 1
// ---------------------------------------------------------------------------

const W_Q2 = [
  [0, 1, 0],
  [1, 0, -1],
  [-1, 1, 1],
  [1, 0, 0],
  [0, -1, 1],
  [1, 1, 0],
]

const W_K2 = [
  [1, 0, 1],
  [0, -1, 0],
  [-1, 1, 0],
  [1, 1, -1],
  [0, 0, 1],
  [-1, 0, 1],
]

const W_V2 = [
  [0, 1, 1],
  [1, -1, 0],
  [0, 0, -1],
  [-1, 1, 0],
  [1, 0, 1],
  [0, 1, -1],
]

// ---------------------------------------------------------------------------
// W_O: output projection (6x6) -- mixes across both heads
// ---------------------------------------------------------------------------

const W_O: number[][] = [
  [1, 0, 0, 0, 1, 0],
  [0, 1, 0, 1, 0, 0],
  [0, 0, 1, 0, 0, 1],
  [1, 0, -1, 0, 1, 0],
  [0, 1, 0, -1, 0, 1],
  [0, 0, 1, 1, 0, 0],
]

// ---------------------------------------------------------------------------
// Computation helpers
// ---------------------------------------------------------------------------

function matVecMul(mat: number[][], vec: number[]): number[] {
  return mat[0].map((_, col) => {
    let sum = 0
    for (let row = 0; row < vec.length; row++) {
      sum += vec[row] * mat[row][col]
    }
    return Math.round(sum * 100) / 100
  })
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i]
  }
  return Math.round(sum * 100) / 100
}

function softmaxRow(values: number[]): number[] {
  const max = Math.max(...values)
  const exps = values.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => Math.round((e / sum) * 1000) / 1000)
}

function scaleRow(values: number[], dk: number): number[] {
  const scale = Math.sqrt(dk)
  return values.map((v) => Math.round((v / scale) * 100) / 100)
}

function weightedSum(weights: number[], vectors: number[][]): number[] {
  const dim = vectors[0].length
  const result = new Array(dim).fill(0) as number[]
  for (let i = 0; i < weights.length; i++) {
    for (let d = 0; d < dim; d++) {
      result[d] += weights[i] * vectors[i][d]
    }
  }
  return result.map((v) => Math.round(v * 1000) / 1000)
}

function matVecMulSquare(mat: number[][], vec: number[]): number[] {
  return mat.map((row) => {
    let sum = 0
    for (let i = 0; i < row.length; i++) {
      sum += row[i] * vec[i]
    }
    return Math.round(sum * 1000) / 1000
  })
}

function vecAdd(a: number[], b: number[]): number[] {
  return a.map((val, i) => Math.round((val + b[i]) * 1000) / 1000)
}

// ---------------------------------------------------------------------------
// Compute attention for a single head
// ---------------------------------------------------------------------------

function computeHead(
  embeddings: { token: string; vec: number[] }[],
  wQ: number[][],
  wK: number[][],
  wV: number[][],
) {
  const Q = embeddings.map((item) => ({
    token: item.token,
    vec: matVecMul(wQ, item.vec),
  }))

  const K = embeddings.map((item) => ({
    token: item.token,
    vec: matVecMul(wK, item.vec),
  }))

  const V = embeddings.map((item) => ({
    token: item.token,
    vec: matVecMul(wV, item.vec),
  }))

  const dk = Q[0].vec.length
  const rawScores = Q.map((qi) =>
    K.map((kj) => dotProduct(qi.vec, kj.vec)),
  )
  const scaledScores = rawScores.map((row) => scaleRow(row, dk))
  const weights = scaledScores.map((row) => softmaxRow(row))

  const vVecs = V.map((v) => v.vec)
  const outputs = weights.map((weightRow) => weightedSum(weightRow, vVecs))

  return { Q, K, V, rawScores, scaledScores, weights, outputs }
}

// ---------------------------------------------------------------------------
// Precompute both heads
// ---------------------------------------------------------------------------

const HEAD_1 = computeHead(TINY_EMBEDDINGS_6D, W_Q1, W_K1, W_V1)
const HEAD_2 = computeHead(TINY_EMBEDDINGS_6D, W_Q2, W_K2, W_V2)

// Concatenate head outputs: for each token, [head1_out | head2_out] -> (4, 6)
const CONCAT_OUTPUTS = TINY_EMBEDDINGS_6D.map((_, i) => [
  ...HEAD_1.outputs[i],
  ...HEAD_2.outputs[i],
])

// Apply W_O: (6,6) x (6,) -> (6,) for each token
const MHA_OUTPUTS = CONCAT_OUTPUTS.map((concatVec) =>
  matVecMulSquare(W_O, concatVec),
)

// Residual: MHA output + original embedding
const RESIDUAL_OUTPUTS = MHA_OUTPUTS.map((mhaOut, i) =>
  vecAdd(mhaOut, TINY_EMBEDDINGS_6D[i].vec),
)

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatVec(vec: number[]): string {
  return `[${vec.map((v) => v.toFixed(2)).join(', ')}]`
}

function formatVec3(vec: number[]): string {
  return `[${vec.map((v) => v.toFixed(3)).join(', ')}]`
}

function weightColor(w: number): string {
  if (w >= 0.35) return 'text-emerald-400 font-medium'
  if (w >= 0.28) return 'text-sky-400'
  return 'text-muted-foreground'
}

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function MultiHeadAttentionLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Multi-Head Attention"
            description="One attention head captures one type of relationship. Multiple heads in parallel capture many&mdash;without increasing compute."
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
            Understand why a single attention head can only capture one notion
            of relevance, how multi-head attention runs{' '}
            <InlineMath math="h" /> independent heads in parallel (each in a
            lower-dimensional subspace with its own{' '}
            <InlineMath math="W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}" />
            ), and how concatenation plus an output projection{' '}
            <InlineMath math="W_O" /> combines their findings into the final
            representation.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built single-head attention across three lessons:{' '}
            <InlineMath math="\text{output} = \text{softmax}(QK^T / \sqrt{d_k})\,V" />.
            This lesson runs that same formula multiple times in parallel&mdash;with
            smaller dimensions and independent projections.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why one attention head isn\u2019t enough (different relationship types)',
              'Multiple heads with independent W_Q, W_K, W_V\u2014each in a d_k = d_model/h dimensional subspace',
              'Dimension splitting as budget allocation (same total compute, not multiplied)',
              'Concatenation of head outputs + output projection W_O as learned cross-head mixing',
              'Hand-traced worked example: 4 tokens, d_model=6, h=2, d_k=3',
              'What heads actually learn in trained models (messy, emergent)',
              'Has notebook: implement multi-head attention from scratch',
              'NOT: the transformer block, layer norm, or feed-forward network\u2014that\u2019s next lesson',
              'NOT: causal masking\u2014Lesson 6',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Recap (brief)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where We Left Off"
            subtitle="Single-head attention is complete"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Across three lessons you built the full single-head attention
              formula piece by piece. Each step solved a limitation you felt in
              the version without it:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  <strong>The Problem Attention Solves:</strong> raw{' '}
                  <InlineMath math="XX^T" /> attention&mdash;felt the dual-role
                  limitation.
                </p>
                <p>
                  <strong>Queries &amp; Keys:</strong> Q and K for asymmetric
                  matching, <InlineMath math="\sqrt{d_k}" /> scaling.
                </p>
                <p>
                  <strong>Values &amp; Attention Output:</strong> V for
                  contribution, residual stream for preservation.
                </p>
              </div>
              <div className="pt-2 border-t border-border/50">
                <BlockMath math="\text{output} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V" />
              </div>
            </div>
            <p className="text-muted-foreground">
              One set of <InlineMath math="W_Q, W_K, W_V" /> produces{' '}
              <strong>one</strong> attention pattern&mdash;one notion of
              &ldquo;what&rsquo;s relevant to what.&rdquo; At the end of the
              last lesson, we planted a question: what happens when a token
              needs to attend for <em>multiple reasons</em>?
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          4. Hook: The "it" problem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One Head, Two Relationships"
            subtitle="A single attention pattern can&rsquo;t be in two places at once"
          />
          <div className="space-y-4">
            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <p className="text-lg text-muted-foreground text-center py-2">
                &ldquo;The cat sat on the mat because{' '}
                <strong className="text-violet-400">it</strong> was soft.&rdquo;
              </p>
            </div>

            <p className="text-muted-foreground">
              The pronoun &ldquo;it&rdquo; needs to attend to two different
              tokens for two different reasons:
            </p>

            <ComparisonRow
              left={{
                title: 'Coreference',
                color: 'orange',
                items: [
                  '"it" \u2192 "mat"',
                  'What does "it" refer to?',
                  'Requires a Q/K projection sensitive to noun-pronoun links',
                ],
              }}
              right={{
                title: 'Property Attribution',
                color: 'emerald',
                items: [
                  '"it" \u2192 "soft"',
                  'What property is being described?',
                  'Requires a Q/K projection sensitive to adjective-subject links',
                ],
              }}
            />

            <GradientCard title="Prediction Exercise" color="emerald">
              <div className="space-y-3 text-sm">
                <p>
                  Can one Q vector for &ldquo;it&rdquo; produce high attention
                  on both &ldquo;mat&rdquo; and &ldquo;soft&rdquo;
                  simultaneously? Think about what the Q and K vectors would
                  need to look like.
                </p>
                <details className="mt-3">
                  <summary className="font-medium cursor-pointer text-primary">
                    Think about it, then reveal
                  </summary>
                  <div className="mt-2 space-y-2">
                    <p>
                      For &ldquo;it&rdquo; to attend strongly to both
                      &ldquo;mat&rdquo; and &ldquo;soft,&rdquo;{' '}
                      <InlineMath math="q_{\text{it}} \cdot k_{\text{mat}}" />{' '}
                      and{' '}
                      <InlineMath math="q_{\text{it}} \cdot k_{\text{soft}}" />{' '}
                      both need to be large. But &ldquo;mat&rdquo; (a noun) and
                      &ldquo;soft&rdquo; (an adjective) have very different
                      semantics, so their K vectors point in different
                      directions. One Q vector can&rsquo;t be simultaneously
                      aligned with both.
                    </p>
                    <p className="text-muted-foreground/70">
                      This isn&rsquo;t a scaling problem. Making{' '}
                      <InlineMath math="d_k" /> bigger doesn&rsquo;t help&mdash;a
                      single dot product computes a single scalar score per
                      pair. The issue is that one Q/K projection extracts{' '}
                      <strong>one</strong> notion of relevance.
                    </p>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Limitation">
            A single attention head computes one set of weights&mdash;one
            &ldquo;lens&rdquo; through which every token views every other
            token. But language has many simultaneous relationships:
            coreference, syntax, semantics, position. One lens can&rsquo;t
            capture them all.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain 4a: The Solution -- Multiple Heads
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Solution: Multiple Heads"
            subtitle="A team of specialists instead of one generalist"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Instead of one researcher reading the entire paper and tracking
              every thread, assign a team of specialists. Each reads through
              their own lens. At the end, they pool their findings. No single
              specialist sees everything, but the team covers more ground than
              any individual could.
            </p>

            <p className="text-muted-foreground">
              Multi-head attention does exactly this. Instead of one set of{' '}
              <InlineMath math="W_Q, W_K, W_V" />, you have{' '}
              <InlineMath math="h" /> independent sets&mdash;each with its{' '}
              <strong>own</strong> learned projection matrices. These are
              completely independent sets of parameters&mdash;Head 1&rsquo;s{' '}
              <InlineMath math="W_Q" /> has no connection to Head 2&rsquo;s{' '}
              <InlineMath math="W_Q" />. They start as different random
              matrices and learn to specialize in different ways:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <div className="space-y-2 font-mono text-sm text-muted-foreground">
                <p>
                  <span className="text-sky-400">Head 1:</span>{' '}
                  <InlineMath math="W_Q^{(1)}, W_K^{(1)}, W_V^{(1)}" />
                </p>
                <p>
                  <span className="text-amber-400">Head 2:</span>{' '}
                  <InlineMath math="W_Q^{(2)}, W_K^{(2)}, W_V^{(2)}" />
                </p>
                <p className="text-muted-foreground/50">
                  &nbsp;&nbsp;&nbsp;&nbsp;...
                </p>
                <p>
                  <span className="text-emerald-400">Head h:</span>{' '}
                  <InlineMath math="W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}" />
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Each head runs the <strong>same attention formula</strong> you
              already know&mdash;
              <InlineMath math="\text{softmax}(Q^{(i)} {K^{(i)}}^T / \sqrt{d_k})\,V^{(i)}" />.
              The formula inside each head is identical to single-head
              attention. The only difference: each head has its own projection
              matrices, so it learns its own notion of relevance.
            </p>

            {/* Side-by-side attention patterns illustration */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Hypothetical weights from a trained model on &ldquo;The cat sat on the mat because it was soft&rdquo;:
              </p>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="px-3 py-2 bg-sky-500/5 border border-sky-500/20 rounded-lg">
                  <p className="text-xs text-sky-400 font-medium mb-2">
                    Head 1: Coreference
                  </p>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>&ldquo;it&rdquo; &rarr; <strong className="text-sky-400">mat</strong> (0.61)</p>
                    <p>&ldquo;it&rdquo; &rarr; cat (0.15)</p>
                    <p>&ldquo;it&rdquo; &rarr; soft (0.08)</p>
                    <p className="text-xs text-muted-foreground/50 mt-2 italic">
                      This head learned to resolve pronouns
                    </p>
                  </div>
                </div>
                <div className="px-3 py-2 bg-amber-500/5 border border-amber-500/20 rounded-lg">
                  <p className="text-xs text-amber-400 font-medium mb-2">
                    Head 2: Property Attribution
                  </p>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>&ldquo;it&rdquo; &rarr; <strong className="text-amber-400">soft</strong> (0.52)</p>
                    <p>&ldquo;it&rdquo; &rarr; was (0.22)</p>
                    <p>&ldquo;it&rdquo; &rarr; mat (0.11)</p>
                    <p className="text-xs text-muted-foreground/50 mt-2 italic">
                      This head learned to track adjective-subject links
                    </p>
                  </div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground/70">
                Different projection matrices produce different attention
                patterns. The same input, two lenses, two views.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Heads Share Input, Nothing Else">
            Each head receives the <strong>same</strong> input{' '}
            <InlineMath math="X" />, but has <strong>completely
            independent</strong> learned weights{' '}
            <InlineMath math="W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}" />.
            Two heads processing the same input produce entirely different
            Q, K, V vectors and entirely different attention patterns.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain 4b: Dimension Splitting
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Dimension Splitting: The Key Insight"
            subtitle="Split the budget, don&rsquo;t multiply it"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If we run <InlineMath math="h" /> separate attention heads, do we
              need <InlineMath math="h" /> times the parameters and compute?
            </p>

            <p className="text-muted-foreground">
              <strong>No.</strong> We split the existing{' '}
              <InlineMath math="d_{\text{model}}" /> budget:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="d_k = \frac{d_{\text{model}}}{h}" />
              <p className="text-sm text-muted-foreground">
                Each head&rsquo;s <InlineMath math="W_Q^{(i)}" /> is{' '}
                <InlineMath math="(d_{\text{model}}, d_k)" /> instead of{' '}
                <InlineMath math="(d_{\text{model}}, d_{\text{model}})" />.
                Smaller projections per head, but more heads.
              </p>
            </div>

            {/* Geometric: dimension strip visualization */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                GPT-2: <InlineMath math="d_{\text{model}} = 768" />,{' '}
                <InlineMath math="h = 12" /> heads,{' '}
                <InlineMath math="d_k = 64" /> each
              </p>
              <div className="flex justify-center py-2">
                <svg
                  width="360"
                  height="80"
                  viewBox="0 0 360 80"
                  className="overflow-visible"
                >
                  {/* Single-head bar */}
                  <rect
                    x="10" y="5" width="340" height="24"
                    rx="4" fill="none"
                    stroke="#a78bfa" strokeWidth={1.5}
                  />
                  <text
                    x="180" y="21" fill="#a78bfa" fontSize="10"
                    fontFamily="monospace" textAnchor="middle"
                  >
                    Single head: d_k = 768 (one big subspace)
                  </text>

                  {/* Multi-head bar: 12 colored strips */}
                  {Array.from({ length: 12 }).map((_, i) => {
                    const colors = [
                      '#38bdf8', '#f59e0b', '#34d399', '#a78bfa',
                      '#fb7185', '#22d3ee', '#f97316', '#818cf8',
                      '#4ade80', '#fbbf24', '#38bdf8', '#f59e0b',
                    ]
                    const x = 10 + i * (340 / 12)
                    const w = 340 / 12 - 1
                    return (
                      <rect
                        key={i}
                        x={x}
                        y="45"
                        width={w}
                        height="24"
                        rx="2"
                        fill={colors[i]}
                        opacity={0.3}
                        stroke={colors[i]}
                        strokeWidth={1}
                      />
                    )
                  })}
                  <text
                    x="180" y="62" fill="#9ca3af" fontSize="8"
                    fontFamily="monospace" textAnchor="middle"
                  >
                    12 heads: d_k = 64 each (same total width)
                  </text>
                </svg>
              </div>
              <p className="text-xs text-muted-foreground/70">
                Same total dimensionality, partitioned into 12 independent
                subspaces. The budget is split, not multiplied.
              </p>
            </div>

            {/* Compute equivalence */}
            <div className="space-y-3">
              <p className="text-muted-foreground">
                <strong>Compute equivalence:</strong> each head&rsquo;s{' '}
                <InlineMath math="QK^T" /> is{' '}
                <InlineMath math="(n \times d_k) \cdot (d_k \times n)" />{' '}
                instead of{' '}
                <InlineMath math="(n \times d_{\text{model}}) \cdot (d_{\text{model}} \times n)" />.
                Total FLOPs across all heads:
              </p>
              <div className="py-3 px-6 bg-muted/50 rounded-lg">
                <BlockMath math="h \cdot n^2 \cdot d_k = n^2 \cdot (h \cdot d_k) = n^2 \cdot d_{\text{model}}" />
              </div>
              <p className="text-muted-foreground">
                Identical to a single head with{' '}
                <InlineMath math="d_k = d_{\text{model}}" />. The computation
                is split, not multiplied. Multi-head attention is{' '}
                <strong>free</strong> in terms of attention compute&mdash;you
                just partition it differently.
              </p>
            </div>

            {/* Misconception 1: More heads = better */}
            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-amber-400">
                Misconception: &ldquo;More heads is always better&rdquo;
              </p>
              <p className="text-sm text-muted-foreground">
                With <InlineMath math="d_{\text{model}} = 768" /> and 768
                heads, each head gets <InlineMath math="d_k = 1" />. A
                1-dimensional dot product is a <strong>single scalar
                multiplication</strong>&mdash;the head can only capture one
                feature of similarity. Absurdly limited.
              </p>
              <p className="text-sm text-muted-foreground">
                At the other extreme, 1 head gives{' '}
                <InlineMath math="d_k = 768" />&mdash;rich relationships but
                only one type. The tradeoff is real: more heads means more
                diverse perspectives but less expressive power per head.
              </p>
              <p className="text-sm text-muted-foreground">
                GPT-2 uses 12 heads (<InlineMath math="d_k = 64" />). GPT-3
                uses 96 heads (<InlineMath math="d_k = 128" /> with{' '}
                <InlineMath math="d_{\text{model}} = 12288" />). These are
                design choices with tradeoffs, not &ldquo;more is better.&rdquo;
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Split, Not Multiplied">
            Multi-head attention has the <strong>same</strong> total compute
            as single-head. Each head operates in a smaller subspace (
            <InlineMath math="d_k = d_{\text{model}}/h" />), but the total
            across all heads equals one large attention operation. You gain
            diversity for free.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Explain 4c: Concatenation + W_O
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Combining Heads: Concatenation + W_O"
            subtitle="Pooling the team&rsquo;s findings"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each head produces an output of shape{' '}
              <InlineMath math="(n, d_k)" />. The natural next step is to
              concatenate all <InlineMath math="h" /> of them along the last
              dimension (<code className="text-xs">torch.cat</code>), recovering
              the original width: <InlineMath math="(n, h \cdot d_k) = (n, d_{\text{model}})" />.
            </p>

            <p className="text-muted-foreground">
              But concatenation alone has a problem: each head&rsquo;s
              contribution stays <strong>isolated</strong> in its own{' '}
              <InlineMath math="d_k" />-dimensional slice. Head 1&rsquo;s
              findings are locked in dimensions 1&ndash;64, Head 2&rsquo;s in
              65&ndash;128, and so on. There&rsquo;s no cross-head
              communication&mdash;the heads filed their reports but never met
              to discuss them.
            </p>

            <p className="text-muted-foreground">
              The output projection <InlineMath math="W_O" /> fixes this. It&rsquo;s
              a learned{' '}
              <InlineMath math="(d_{\text{model}} \times d_{\text{model}})" />{' '}
              matrix&mdash;another <code className="text-xs">nn.Linear</code>&mdash;applied
              after concatenation. The full formula:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_O" />
            </div>

            {/* W_O is not just reshaping */}
            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-amber-400">
                W<sub>O</sub> is NOT just reshaping
              </p>
              <p className="text-sm text-muted-foreground">
                <InlineMath math="W_O" /> is a learned{' '}
                <InlineMath math="d_{\text{model}} \times d_{\text{model}}" />{' '}
                matrix with <InlineMath math="d_{\text{model}}^2" /> parameters.
                It <strong>mixes information across heads</strong>&mdash;head
                3&rsquo;s output can influence the final representation in
                dimensions that head 1 wrote to.
              </p>
              <p className="text-sm text-muted-foreground">
                Without <InlineMath math="W_O" />, each head&rsquo;s
                contribution is locked in its <InlineMath math="d_k" />-dimensional
                slice. With <InlineMath math="W_O" />, the model learns how to
                blend and synthesize signals from all heads.
              </p>
            </div>

            {/* Full shape walkthrough */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Shape check (full pipeline):
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-1">
                <p>
                  Input X: <InlineMath math="(n, d_{\text{model}})" />
                </p>
                <p>
                  Per head: <InlineMath math="Q^{(i)}, K^{(i)}, V^{(i)}" /> each{' '}
                  <InlineMath math="(n, d_k)" />
                </p>
                <p>
                  Per head output: <InlineMath math="(n, d_k)" />
                </p>
                <p>
                  Concat: <InlineMath math="(n, h \cdot d_k) = (n, d_{\text{model}})" />
                </p>
                <p>
                  After <InlineMath math="W_O" />:{' '}
                  <InlineMath math="(n, d_{\text{model}})" />&mdash;same shape
                  as input
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="W_O: The Team Meeting">
            Concatenation is each specialist filing their report.{' '}
            <InlineMath math="W_O" /> is the team meeting where they synthesize
            findings&mdash;letting insights from one head inform dimensions
            that another head owns. It&rsquo;s a learned mixing layer, not a
            formatting step.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Check: Prediction Exercise
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Dimension Reasoning"
            subtitle="Test your understanding of the budget tradeoff"
          />
          <GradientCard title="Prediction Exercise" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You have <InlineMath math="d_{\text{model}} = 512" /> and{' '}
                <InlineMath math="h = 8" /> heads. Answer:
              </p>
              <ol className="list-decimal list-inside space-y-1">
                <li>
                  What is <InlineMath math="d_k" />?
                </li>
                <li>
                  How many parameters does <InlineMath math="W_O" /> have?
                </li>
                <li>
                  If you add a 9th head, what changes?
                </li>
              </ol>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>1.</strong>{' '}
                    <InlineMath math="d_k = 512 / 8 = 64" />.
                  </p>
                  <p>
                    <strong>2.</strong>{' '}
                    <InlineMath math="W_O" /> is{' '}
                    <InlineMath math="512 \times 512 = 262{,}144" /> parameters.
                  </p>
                  <p>
                    <strong>3.</strong> 512 doesn&rsquo;t divide evenly by 9.
                    Either <InlineMath math="d_k" /> drops to{' '}
                    <InlineMath math="\lfloor 512/9 \rfloor \approx 56" />{' '}
                    (messy, wasted dimensions), or you increase{' '}
                    <InlineMath math="d_{\text{model}}" />. This is why{' '}
                    <InlineMath math="h" /> typically divides{' '}
                    <InlineMath math="d_{\text{model}}" /> evenly. The
                    constraint is{' '}
                    <InlineMath math="d_{\text{model}} = h \cdot d_k" />.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          9. Explore: Worked Example (d_model=6, h=2, d_k=3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Worked Example: Two Heads in Action"
            subtitle="Same 4 tokens, d_model=6, h=2, d_k=3"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Same four tokens from the last three lessons, now with 6-dimensional
              embeddings to support splitting into 2 heads of{' '}
              <InlineMath math="d_k = 3" /> each. Each head has its own{' '}
              <InlineMath math="W_Q, W_K, W_V" /> (6&times;3 matrices).
            </p>

            {/* Show 6D embeddings */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                4-token embeddings (d_model = 6):
              </p>
              <div className="space-y-1.5 font-mono text-sm">
                {TINY_EMBEDDINGS_6D.map((item) => (
                  <div
                    key={item.token}
                    className="flex items-center gap-3 text-muted-foreground"
                  >
                    <span className="text-violet-400 w-12">
                      &ldquo;{item.token}&rdquo;
                    </span>
                    <span className="text-sky-400">{formatVec(item.vec)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Continuity">
            Same 4 tokens from every lesson in this module. The first 3
            dimensions are the original embeddings; dimensions 4&ndash;6 are
            new to support the 2-head split.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Head 1 computation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Head 1: Compute Q, K, V, attention weights, and output
            </p>
            <p className="text-muted-foreground text-sm">
              Head 1 uses its own{' '}
              <InlineMath math="W_Q^{(1)}, W_K^{(1)}, W_V^{(1)}" /> (each
              6&times;3). Same attention formula as the last three lessons,
              just in a 3-dimensional subspace.
            </p>

            {/* Head 1 attention weights */}
            <div className="px-4 py-3 bg-sky-500/5 border border-sky-500/20 rounded-lg space-y-3">
              <p className="text-xs text-sky-400 font-medium">
                Head 1 attention weights:
              </p>
              <div className="overflow-x-auto">
                <table className="text-sm font-mono w-full">
                  <thead>
                    <tr className="text-muted-foreground/60 text-xs">
                      <th className="text-left pr-3 pb-2">weights</th>
                      {TINY_EMBEDDINGS_6D.map((item) => (
                        <th
                          key={item.token}
                          className="text-center px-2 pb-2 text-violet-400/70"
                        >
                          {item.token}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {TINY_EMBEDDINGS_6D.map((rowItem, i) => (
                      <tr key={rowItem.token}>
                        <td className="text-violet-400/70 pr-3 py-1">
                          {rowItem.token}
                        </td>
                        {HEAD_1.weights[i].map((weight, j) => (
                          <td
                            key={j}
                            className={`text-center px-2 py-1 ${weightColor(weight)}`}
                          >
                            {weight.toFixed(3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                Look at <strong>&ldquo;here&rdquo;</strong>: it attends most
                strongly to &ldquo;cat&rdquo; (0.586)&mdash;a clear
                cross-token relationship. &ldquo;cat&rdquo; in turn attends
                most to &ldquo;sat&rdquo; (0.441). Head 1&rsquo;s projections
                produce attention that reaches <em>across</em> the sequence to
                other tokens.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* Head 2 computation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Head 2: Same input, different projections, different pattern
            </p>

            {/* Head 2 attention weights */}
            <div className="px-4 py-3 bg-amber-500/5 border border-amber-500/20 rounded-lg space-y-3">
              <p className="text-xs text-amber-400 font-medium">
                Head 2 attention weights:
              </p>
              <div className="overflow-x-auto">
                <table className="text-sm font-mono w-full">
                  <thead>
                    <tr className="text-muted-foreground/60 text-xs">
                      <th className="text-left pr-3 pb-2">weights</th>
                      {TINY_EMBEDDINGS_6D.map((item) => (
                        <th
                          key={item.token}
                          className="text-center px-2 pb-2 text-violet-400/70"
                        >
                          {item.token}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {TINY_EMBEDDINGS_6D.map((rowItem, i) => (
                      <tr key={rowItem.token}>
                        <td className="text-violet-400/70 pr-3 py-1">
                          {rowItem.token}
                        </td>
                        {HEAD_2.weights[i].map((weight, j) => (
                          <td
                            key={j}
                            className={`text-center px-2 py-1 ${weightColor(weight)}`}
                          >
                            {weight.toFixed(3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                Now look at <strong>&ldquo;here&rdquo;</strong> in Head 2: it
                attends most strongly to <em>itself</em> (0.628)&mdash;a
                self-reinforcing pattern. &ldquo;sat&rdquo; also attends most
                to itself (0.322). Head 2&rsquo;s projections produce
                attention that favors local and self-attention, a completely
                different pattern from Head 1.
              </p>
            </div>

            <p className="text-muted-foreground">
              Now compare the two heads on the <strong>same
              token</strong>&mdash;&ldquo;here.&rdquo; In Head 1,
              &ldquo;here&rdquo; attends most strongly to &ldquo;cat&rdquo;
              (0.586)&mdash;a cross-token relationship. In Head 2,
              &ldquo;here&rdquo; attends most strongly to itself
              (0.628)&mdash;a self-reinforcing pattern. Same token, same
              input, but the two heads extract completely different views of
              what matters. This is multi-head attention in action: different
              projection matrices produce genuinely different attention
              patterns.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Different Weights, Same Input">
            Both heads see the same 4 tokens, but their attention patterns
            differ because their learned projection matrices differ. Head
            1&rsquo;s notion of &ldquo;what&rsquo;s relevant&rdquo; is
            independent from Head 2&rsquo;s.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Concatenation + W_O */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Concatenate + apply W<sub>O</sub>
            </p>

            <p className="text-muted-foreground text-sm">
              Each head produced an output of shape (4, 3). Concatenate along
              the last dimension to get (4, 6), then apply{' '}
              <InlineMath math="W_O" /> (6&times;6):
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Head outputs &rarr; concat &rarr; W<sub>O</sub> &rarr; MHA
                output:
              </p>
              {TINY_EMBEDDINGS_6D.map((item, i) => (
                <div key={item.token} className="space-y-1">
                  <div className="flex items-start gap-2 font-mono text-xs text-muted-foreground">
                    <span className="text-violet-400 w-12 shrink-0">
                      &ldquo;{item.token}&rdquo;
                    </span>
                    <div className="space-y-0.5">
                      <p>
                        <span className="text-sky-400">H1</span>:{' '}
                        {formatVec3(HEAD_1.outputs[i])}{' '}
                        <span className="text-muted-foreground/50">|</span>{' '}
                        <span className="text-amber-400">H2</span>:{' '}
                        {formatVec3(HEAD_2.outputs[i])}
                      </p>
                      <p>
                        <span className="text-muted-foreground/50">
                          &rarr; W<sub>O</sub> &rarr;
                        </span>{' '}
                        <span className="text-emerald-400 font-medium">
                          {formatVec3(MHA_OUTPUTS[i])}
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Residual */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Add residual (MHA output + original embedding):
              </p>
              {TINY_EMBEDDINGS_6D.map((item, i) => (
                <div
                  key={item.token}
                  className="flex items-center gap-2 font-mono text-xs text-muted-foreground"
                >
                  <span className="text-violet-400 w-12">
                    &ldquo;{item.token}&rdquo;
                  </span>
                  <span className="text-emerald-400">
                    {formatVec3(MHA_OUTPUTS[i])}
                  </span>
                  <span className="text-muted-foreground/50">+</span>
                  <span className="text-sky-400">
                    {formatVec(TINY_EMBEDDINGS_6D[i].vec)}
                  </span>
                  <span className="text-muted-foreground/50">=</span>
                  <span className="text-violet-400 font-medium">
                    {formatVec3(RESIDUAL_OUTPUTS[i])}
                  </span>
                </div>
              ))}
              <p className="text-xs text-muted-foreground/70">
                Same shape as input&mdash;(4, 6). Ready for the residual
                stream, just like single-head attention.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Elaborate: What Heads Actually Learn
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Heads Actually Learn"
            subtitle="Messy and emergent, not designed and clean"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              It&rsquo;s tempting to think each head has a specific linguistic
              job&mdash;head 1 handles syntax, head 2 handles coreference,
              head 3 handles semantics. A clean org chart for the model&rsquo;s
              attention.
            </p>

            <p className="text-muted-foreground">
              Research on trained models tells a messier story. Studies of
              BERT and GPT-2 have found:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Some heads attend to the <strong>previous token</strong>&mdash;a
                positional pattern, not a semantic one
              </li>
              <li>
                Some track <strong>syntactic relationships</strong> like
                subject-verb agreement
              </li>
              <li>
                Some are <strong>nearly uniform</strong>&mdash;attending to all
                tokens roughly equally (apparently redundant)
              </li>
              <li>
                Most heads don&rsquo;t map cleanly to any single linguistic
                function
              </li>
              <li>
                Pruning studies show you can remove{' '}
                <strong>20&ndash;40% of heads</strong> with minimal performance
                loss
              </li>
            </ul>

            <p className="text-muted-foreground">
              This is the same principle you saw with convolution filters: the
              architecture provides <strong>capacity</strong> (filters, heads),
              and training determines what each one learns. The difference
              between a learned pattern and a designed pattern applies in both
              cases&mdash;heads, like filters, discover what&rsquo;s useful
              rather than being assigned a role.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Capacity, Not Assignment">
            Multi-head attention gives the model the <strong>capacity</strong>{' '}
            to capture multiple relationship types. What it actually learns to
            capture is determined by the training data and objective, not by
            the architecture. The heads are free slots; training fills them.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Check: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Transfer Question"
            subtitle="Compare single-head to multi-head"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A colleague builds a model with{' '}
                <InlineMath math="d_{\text{model}} = 256" /> and{' '}
                <InlineMath math="h = 1" /> (single head,{' '}
                <InlineMath math="d_k = 256" />). Another builds{' '}
                <InlineMath math="d_{\text{model}} = 256" /> and{' '}
                <InlineMath math="h = 4" /> (
                <InlineMath math="d_k = 64" />).
              </p>
              <p>
                Which model has more parameters in the attention projections?
                Which captures more diverse relationships?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Parameters:</strong> Each head&rsquo;s Q/K/V
                    projections total{' '}
                    <InlineMath math="3 \times d_{\text{model}} \times d_k" />.
                    For <InlineMath math="h" /> heads:{' '}
                    <InlineMath math="3 \times d_{\text{model}} \times d_k \times h = 3 \times 256 \times 256 = 196{,}608" />{' '}
                    parameters in both cases (same total). The 4-head model
                    adds <InlineMath math="W_O" />: 256&times;256 = 65,536
                    extra parameters. So the 4-head model has{' '}
                    <strong>slightly more</strong> parameters.
                  </p>
                  <p>
                    <strong>Diversity:</strong> The 4-head model captures{' '}
                    <strong>more diverse relationships</strong>&mdash;4
                    independent attention patterns instead of 1. The tradeoff:
                    each head operates in a 64-dimensional subspace instead of
                    256, so each individual head is less expressive. But the
                    ensemble of 4 covers more ground.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          12. Practice: Notebook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Multi-Head Attention from Scratch"
            subtitle="Build it yourself in Python"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook walks you through implementing multi-head attention
              step by step:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Exercise 1 (guided):</strong> Implement single-head
                attention as a function&mdash;recap from the last lesson, but
                now in code you can reuse.
              </li>
              <li>
                <strong>Exercise 2 (supported):</strong> Implement multi-head
                attention from scratch&mdash;split into h heads, run attention
                on each, concatenate, apply W<sub>O</sub>. Use{' '}
                <InlineMath math="d_{\text{model}}=6, h=2" /> on the 4-token
                example. Verify the output shape.
              </li>
              <li>
                <strong>Exercise 3 (supported):</strong> Verify compute
                equivalence&mdash;time single-head (
                <InlineMath math="d_k = d_{\text{model}}" />) vs. multi-head
                on a larger example. Compare FLOPs and wall-clock time.
              </li>
              <li>
                <strong>Exercise 4 (independent):</strong> Implement multi-head
                attention as an <code className="text-xs">nn.Module</code> with
                proper <code className="text-xs">nn.Linear</code> layers.
                Forward pass handles batched input.
              </li>
              <li>
                <strong>Exercise 5 (stretch):</strong> Load GPT-2 pretrained
                weights. Extract attention weights for all 12 heads on a sample
                sentence. Visualize the 12 attention heatmaps side by side.
                Which heads show positional patterns? Semantic patterns?
              </li>
            </ul>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Implement multi-head attention from scratch and explore what
                  real heads learn.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-2-4-multi-head-attention.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  Exercises 1&ndash;3 build to multi-head attention. Exercise 4
                  makes it an nn.Module. Exercise 5 explores real GPT-2 heads.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Key Experiments">
            <ul className="space-y-2 text-sm">
              <li>&bull; Compare single-head vs. multi-head attention patterns on the same input</li>
              <li>&bull; Verify the compute equivalence numerically</li>
              <li>&bull; Visualize GPT-2&rsquo;s 12 heads&mdash;which are interpretable?</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Multi-head attention runs h independent attention heads in parallel, each in a d_k = d_model/h dimensional subspace.',
                description:
                  'Each head has its own W_Q, W_K, W_V\u2014completely independent learned projections. Different heads capture different types of relationships from the same input.',
              },
              {
                headline:
                  'Dimension splitting is budget allocation, not compute multiplication.',
                description:
                  'The total compute across all heads equals one large attention operation. You gain diversity of attention patterns for free\u2014the same d_model budget, partitioned differently.',
              },
              {
                headline:
                  'W_O is a learned cross-head mixing layer, not just reshaping.',
                description:
                  'After concatenating head outputs, W_O (a d_model \u00D7 d_model matrix) lets information from any head influence any dimension of the final output. It\u2019s how the model synthesizes what different heads learned.',
              },
              {
                headline:
                  'Head specialization is emergent and messy, not designed and clean.',
                description:
                  'The architecture provides capacity for diverse attention patterns. What each head actually learns is determined by training\u2014some heads are interpretable, many are not, and 20\u201340% can often be pruned with minimal loss.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo + seed for transformer block */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-violet-400">
                Mental model: &ldquo;Multiple lenses, pooled findings&rdquo;
              </p>
              <p className="text-sm text-muted-foreground">
                Each head looks through its own set of three lenses (
                <InlineMath math="W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}" />
                ), and <InlineMath math="W_O" /> synthesizes what they all
                saw. The &ldquo;three lenses, one embedding&rdquo; mental
                model from before is now &ldquo;three lenses per head,{' '}
                <InlineMath math="h" /> heads, one synthesis step.&rdquo;
              </p>
            </div>

            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-violet-400">
                We have the MHA layer&mdash;what wraps around it?
              </p>
              <p className="text-sm text-muted-foreground">
                Multi-head attention is the core computational unit. But it
                doesn&rsquo;t stand alone. In the full transformer, multi-head
                attention feeds into a feed-forward network, with residual
                connections and layer normalization around each. The next lesson
                assembles these pieces into the <strong>transformer
                block</strong>&mdash;the repeating unit that stacks to form GPT.
              </p>
              <p className="text-xs text-muted-foreground/70 italic mt-1">
                Attention reads from the residual stream; the FFN writes new
                information into it. Different roles, same stream.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/the-transformer-block"
            title="The Transformer Block"
            description="Multi-head attention + feed-forward network + residual connections + layer normalization&mdash;the repeating unit that stacks to form GPT."
            buttonText="Continue to The Transformer Block"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
