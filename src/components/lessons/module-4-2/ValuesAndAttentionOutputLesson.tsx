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
 * Values and the Attention Output
 *
 * Third lesson in Module 4.2 (Attention & the Transformer).
 * Sixth lesson in Series 4 (LLMs & Transformers).
 *
 * Completes single-head attention by introducing V (the third learned
 * projection) and the residual stream. Resolves the "matching vs contributing"
 * dual-role limitation — K encodes what makes a token relevant for matching,
 * V encodes what information it contributes when attended to.
 *
 * Core concepts at DEVELOPED:
 * - V as a third learned projection (separating matching from contributing)
 * - Full single-head attention formula: output = softmax(QK^T / sqrt(d_k)) V
 *
 * Core concepts at INTRODUCED:
 * - Residual stream (attention output added to input, not replacing it)
 *
 * Concepts REINFORCED:
 * - Q and K projections (extended with V as the third lens)
 * - Weighted average (now of V vectors, not raw embeddings)
 * - Residual connections / F(x) + x from ResNets (contextual bridge)
 * - nn.Linear (V is yet another nn.Linear layer)
 * - Job fair analogy (extended with the resume card)
 *
 * EXPLICITLY NOT COVERED:
 * - Multi-head attention — Lesson 4
 * - The output projection W_O — Lesson 4
 * - The full transformer block — Lesson 5
 * - Layer normalization — Lesson 5
 * - Causal masking — Lesson 6
 *
 * Previous: Queries, Keys, and the Relevance Function (module 4.2, lesson 2)
 * Next: Multi-Head Attention (module 4.2, lesson 4)
 */

// ---------------------------------------------------------------------------
// Same 4-token embeddings from Lessons 1 & 2 (three-lesson continuity)
// ---------------------------------------------------------------------------

const TINY_EMBEDDINGS = [
  { token: 'The', vec: [0.9, 0.1, -0.2] },
  { token: 'cat', vec: [-0.3, 0.8, 0.5] },
  { token: 'sat', vec: [0.1, 0.6, 0.4] },
  { token: 'here', vec: [0.4, -0.2, 0.7] },
]

// Same W_Q and W_K from Lesson 2 (continuity)
const W_Q = [
  [1, 0, -1],
  [0, 1, 1],
  [1, -1, 0],
]

const W_K = [
  [0, 1, 0],
  [-1, 0, 1],
  [1, 1, -1],
]

// NEW: W_V — the value projection matrix
// Chosen to produce clearly different vectors from W_K,
// with small values for hand-traceability
const W_V = [
  [1, -1, 0],
  [0, 0, 1],
  [-1, 1, 1],
]

// ---------------------------------------------------------------------------
// Computation helpers (same as Lesson 2)
// ---------------------------------------------------------------------------

function matVecMul(mat: number[][], vec: number[]): number[] {
  return mat.map((row) => {
    let sum = 0
    for (let i = 0; i < row.length; i++) {
      sum += row[i] * vec[i]
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
  const result = new Array(dim).fill(0)
  for (let i = 0; i < weights.length; i++) {
    for (let d = 0; d < dim; d++) {
      result[d] += weights[i] * vectors[i][d]
    }
  }
  return result.map((v) => Math.round(v * 1000) / 1000)
}

function vecAdd(a: number[], b: number[]): number[] {
  return a.map((val, i) => Math.round((val + b[i]) * 1000) / 1000)
}

// ---------------------------------------------------------------------------
// Compute Q, K, V vectors
// ---------------------------------------------------------------------------

const Q_VECTORS = TINY_EMBEDDINGS.map((item) => ({
  token: item.token,
  vec: matVecMul(W_Q, item.vec),
}))

const K_VECTORS = TINY_EMBEDDINGS.map((item) => ({
  token: item.token,
  vec: matVecMul(W_K, item.vec),
}))

const V_VECTORS = TINY_EMBEDDINGS.map((item) => ({
  token: item.token,
  vec: matVecMul(W_V, item.vec),
}))

// Compute QK^T scores, scale, and softmax
const QK_SCORES: number[][] = Q_VECTORS.map((qi) =>
  K_VECTORS.map((kj) => dotProduct(qi.vec, kj.vec))
)

const SCALED_SCORES: number[][] = QK_SCORES.map((row) => scaleRow(row, 3))

const ATTENTION_WEIGHTS = SCALED_SCORES.map((row) => softmaxRow(row))

// Compute attention output for each token: output_i = sum_j(weight_ij * v_j)
const V_VECS_RAW = V_VECTORS.map((v) => v.vec)
const ATTENTION_OUTPUTS = ATTENTION_WEIGHTS.map((weightRow) =>
  weightedSum(weightRow, V_VECS_RAW)
)

// Compute residual outputs: residual_i = attention_output_i + embedding_i
const RESIDUAL_OUTPUTS = ATTENTION_OUTPUTS.map((attnOut, i) =>
  vecAdd(attnOut, TINY_EMBEDDINGS[i].vec)
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

export function ValuesAndAttentionOutputLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Values and the Attention Output"
            description="The third projection separates &ldquo;what makes me relevant&rdquo; from &ldquo;what I contribute&rdquo;&mdash;completing single-head attention."
            category="Attention"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 1: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how a third learned projection (V&nbsp;=&nbsp;W<sub>V</sub>&nbsp;&times;&nbsp;embedding)
            separates &ldquo;what makes a token relevant&rdquo; (K) from &ldquo;what it
            contributes when attended to&rdquo; (V), completing single-head attention
            as <InlineMath math="\text{output} = \text{softmax}(QK^T / \sqrt{d_k})\,V" />&mdash;and
            why the attention output is <strong>added</strong> to the input via the residual
            stream rather than replacing it.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In Queries, Keys, and the Relevance Function, you built the full attention
            weight matrix: <InlineMath math="\text{softmax}(QK^T / \sqrt{d_k})" />. Those
            weights tell each token how much to attend to every other token. This lesson
            answers: what information does each token actually contribute?
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why V exists (separating matching from contributing)',
              'What V computes (V = W_V \u00D7 embedding, a third learned projection)',
              'The full single-head attention formula: output = softmax(QK\u1D40/\u221Ad_k) V',
              'The residual stream: attention output is ADDED to the input, not substituted',
              'Hand-tracing the full computation with the same 4 tokens from the last two lessons',
              'Has notebook: implement single-head attention end-to-end',
              'NOT: multi-head attention (multiple Q/K/V sets in parallel)\u2014that\u2019s next lesson',
              'NOT: the transformer block, layer normalization, or causal masking',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Hook -- The Pattern You've Seen Before
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Pattern You&rsquo;ve Seen Before"
            subtitle="One vector, two roles. You know the fix."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              At the end of Queries, Keys, and the Relevance Function, we planted a question:
            </p>

            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <p className="text-sm text-violet-400 italic">
                &ldquo;The key that makes a token relevant and the information it should
                provide when attended to are different things. Should we use the same
                embedding for matching AND contributing? We already know the answer to
                that kind of question.&rdquo;
              </p>
            </div>

            <p className="text-muted-foreground">
              You&rsquo;ve resolved this pattern before. In The Problem Attention Solves,
              one embedding served as both &ldquo;what am I looking for?&rdquo; and &ldquo;what
              do I offer?&rdquo;&mdash;you felt the limitation, and Q/K fixed it. The
              same pattern appears again: K (what makes me relevant for matching) is a
              different role than &ldquo;what information I should provide when selected.&rdquo;
            </p>

            <p className="text-muted-foreground">
              One vector, two roles. You know the fix.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground/70 font-medium mb-2">
                The formula from the last lesson&mdash;the weights are complete:
              </p>
              <BlockMath math="\text{weights} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)" />
              <p className="text-sm text-muted-foreground mt-2">
                But weights need something to weight. What vectors do we take the
                weighted average of?
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Pattern Recognition">
            This is the <strong>same design pattern</strong> for the third time: one
            representation serving two distinct roles. Lesson 1 resolved seeking/offering.
            Lesson 2 delivered Q/K. Now: matching vs. contributing. Same fix&mdash;a new
            learned projection.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: The Third Projection -- V
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Third Projection&mdash;V"
            subtitle="Separating &ldquo;what makes me relevant&rdquo; from &ldquo;what I have to say&rdquo;"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the current formula, after computing the attention weights, what do we
              weight? If we multiply the weights by the raw embeddings, the contribution
              of each token to the output is its full embedding&mdash;the same vector used
              to compute everything else.
            </p>

            <p className="text-muted-foreground">
              But consider &ldquo;The cat chased the mouse.&rdquo; What
              makes &ldquo;chased&rdquo; relevant to &ldquo;cat&rdquo; (captured by K) is
              its action-verb semantics&mdash;&ldquo;I&rsquo;m a verb, relevant to a noun
              seeking its action.&rdquo; What &ldquo;chased&rdquo; should <em>contribute</em> to
              &ldquo;cat&rdquo;&rsquo;s representation is different information
              entirely&mdash;past tense, transitivity, directed motion. The matching signal
              and the contribution signal serve different purposes.
            </p>

            <p className="text-muted-foreground">
              The fix: <strong>V = W<sub>V</sub> &times; embedding</strong>. A third learned
              lens that extracts the contribution signal&mdash;what each token has to say
              when it&rsquo;s attended to.
            </p>

            {/* Job fair extension */}
            <ComparisonRow
              left={{
                title: 'What I Bring (K)',
                color: 'orange',
                items: [
                  '"I\'m an action-verb, relevant to nouns"',
                  'Gets you the match',
                  'The offering card at the job fair',
                ],
              }}
              right={{
                title: 'My Resume (V)',
                color: 'emerald',
                items: [
                  '"Past tense, transitive, implies motion"',
                  'What you actually deliver when matched',
                  'The detailed resume you hand over',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Remember the job fair? You had two cards: a &ldquo;what I&rsquo;m looking
              for&rdquo; card (Q) and a &ldquo;what I bring to the table&rdquo; card (K).
              But when you&rsquo;re actually matched with a team, you don&rsquo;t hand them
              the offering card. You hand them your <strong>resume</strong>&mdash;a detailed
              description of your actual skills and experience. The offering card got you
              the match; the resume is what you actually deliver.
            </p>

            {/* Three lenses summary */}
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-medium">
                Three lenses, one embedding:
              </p>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  <strong className="text-sky-400">Query (W<sub>Q</sub>):</strong>{' '}
                  &ldquo;What am I seeking?&rdquo;
                </p>
                <p>
                  <strong className="text-amber-400">Key (W<sub>K</sub>):</strong>{' '}
                  &ldquo;What do I advertise for matching?&rdquo;
                </p>
                <p>
                  <strong className="text-emerald-400">Value (W<sub>V</sub>):</strong>{' '}
                  &ldquo;What do I actually have to say?&rdquo;
                </p>
              </div>
              <p className="text-xs text-muted-foreground/70">
                Same embedding, three different matrices, three different views. The model
                learns all three from data.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Third Lens">
            W<sub>Q</sub> reveals what the token seeks. W<sub>K</sub> reveals what the
            token advertises for matching. W<sub>V</sub> reveals what the token actually
            has to say. Same embedding, three lenses, three views.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Geometric SVG: three projections from one embedding */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In PyTorch, V is just another <code className="text-xs">nn.Linear</code> layer&mdash;the
              same building block as Q and K:
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg font-mono text-sm text-muted-foreground">
              <p><span className="text-sky-400">self.W_Q</span> = nn.Linear(d_model, d_k, bias=False)</p>
              <p><span className="text-amber-400">self.W_K</span> = nn.Linear(d_model, d_k, bias=False)</p>
              <p><span className="text-emerald-400">self.W_V</span> = nn.Linear(d_model, d_v, bias=False)</p>
            </div>
            <p className="text-muted-foreground">
              You now have three <code className="text-xs">nn.Linear</code> layers. That&rsquo;s
              the entire Q/K/V mechanism. No new operation&mdash;just a third application of
              the same building block you&rsquo;ve used since Module 2.1.
            </p>

            {/* Three-projection SVG */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Geometric picture: same point, three projections
              </p>
              <div className="flex justify-center py-2">
                <svg width="320" height="180" viewBox="0 0 320 180" className="overflow-visible">
                  {/* Origin point (embedding) */}
                  <circle cx="160" cy="140" r="5" fill="#a78bfa" />
                  <text x="168" y="155" fill="#a78bfa" fontSize="11" fontFamily="monospace">embedding</text>

                  {/* W_Q arrow (up-left) */}
                  <line x1="160" y1="140" x2="50" y2="40" stroke="#38bdf8" strokeWidth={2} strokeDasharray="4 2" />
                  <polygon points="50,40 58,47 62,39" fill="#38bdf8" />
                  <text x="20" y="33" fill="#38bdf8" fontSize="11" fontFamily="monospace">Q</text>
                  <text x="70" y="78" fill="#38bdf8" fontSize="9" fontFamily="monospace" opacity={0.7}>W_Q</text>

                  {/* W_K arrow (up-center-right) */}
                  <line x1="160" y1="140" x2="200" y2="25" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 2" />
                  <polygon points="200,25 192,32 202,34" fill="#f59e0b" />
                  <text x="208" y="22" fill="#f59e0b" fontSize="11" fontFamily="monospace">K</text>
                  <text x="187" y="73" fill="#f59e0b" fontSize="9" fontFamily="monospace" opacity={0.7}>W_K</text>

                  {/* W_V arrow (up-right) -- NEW */}
                  <line x1="160" y1="140" x2="290" y2="55" stroke="#34d399" strokeWidth={2} strokeDasharray="4 2" />
                  <polygon points="290,55 282,52 286,62" fill="#34d399" />
                  <text x="298" y="52" fill="#34d399" fontSize="11" fontFamily="monospace">V</text>
                  <text x="240" y="90" fill="#34d399" fontSize="9" fontFamily="monospace" opacity={0.7}>W_V</text>

                  {/* Caption */}
                  <text x="48" y="175" fill="currentColor" fontSize="10" fontFamily="monospace" opacity={0.4}>same input, three matrices, three destinations</text>
                </svg>
              </div>
              <p className="text-xs text-muted-foreground/70">
                The same embedding lands at three different locations. Q and K are used
                for scoring (dot product). V is used for the output (weighted average).
                All three are learned during training.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="V Is Not a Token Property">
            V, like Q and K, is a property of the <strong>learned projection
            matrix</strong>, not the token itself. Different layers learn
            different W<sub>V</sub> matrices, extracting different contribution
            signals from the same embedding.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Check -- Predict K vs V Distinction
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Predict the K vs V Distinction"
            subtitle="Different roles, different information"
          />
          <GradientCard title="Prediction Exercise" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Consider: &ldquo;The cat chased the mouse.&rdquo;
              </p>
              <p>
                When &ldquo;chased&rdquo; is attended to by &ldquo;cat,&rdquo; what
                should K encode (for matching) vs what should V encode (for contributing)?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong className="text-amber-400">K</strong> should encode
                    &ldquo;I&rsquo;m an action verb, relevant to a noun seeking its
                    action.&rdquo; This is the matching signal&mdash;what got &ldquo;chased&rdquo;
                    selected by &ldquo;cat&rdquo;&rsquo;s query.
                  </p>
                  <p>
                    <strong className="text-emerald-400">V</strong> should encode different
                    information&mdash;tense (past), transitivity (takes a direct object),
                    direction of motion (implies something moved toward something else).
                    This is what &ldquo;chased&rdquo; <em>contributes</em> to &ldquo;cat&rdquo;&rsquo;s
                    representation.
                  </p>
                  <p className="text-muted-foreground/70">
                    The matching signal and the contribution signal serve different purposes.
                    Without V, both roles are locked into one vector.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 5: The Full Formula
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Full Formula"
            subtitle="Three lessons, three targeted replacements"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The formula has evolved across three lessons. Each step is a targeted
              replacement&mdash;the structure hasn&rsquo;t changed, only <em>what</em> we
              compute scores with and <em>what</em> we take the weighted average of:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-4">
              <div className="space-y-3">
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground/70 font-medium">
                    The Problem Attention Solves&mdash;raw embeddings for everything:
                  </p>
                  <BlockMath math="\text{output} = \text{softmax}(XX^T)\, X" />
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground/70 font-medium">
                    Queries &amp; Keys&mdash;replaced X with Q and K for scoring:
                  </p>
                  <BlockMath math="\text{weights} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)" />
                </div>

                <div className="space-y-1 border-t border-border/50 pt-3">
                  <p className="text-xs text-emerald-400/80 font-medium">
                    This lesson&mdash;replaced X with V for contributing:
                  </p>
                  <BlockMath math="\text{output} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V" />
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              Each step exists because the version without it was insufficient. The
              formula&rsquo;s structure hasn&rsquo;t changed&mdash;it&rsquo;s still
              &ldquo;compute weights, then take a weighted average.&rdquo; What changed is
              what we score with (Q/K instead of raw embeddings) and what we average (V
              instead of raw embeddings).
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground font-medium">
                The output for token <em>i</em>:
              </p>
              <BlockMath math="\text{output}_i = \sum_j \text{weight}_{ij} \cdot v_j" />
              <p className="text-sm text-muted-foreground">
                Each token computes its <strong>own</strong> weighted average of the V
                vectors, using its own row of attention weights. The output has the same
                shape as the input&mdash;one vector per token.
              </p>
            </div>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-xs text-muted-foreground/70 font-medium">Shape check:</p>
              <div className="font-mono text-sm text-muted-foreground space-y-1">
                <p>Input: n tokens &times; d<sub>model</sub> dimensions</p>
                <p>Attention weights: n &times; n</p>
                <p>V: n &times; d<sub>v</sub> (output dimension of W<sub>V</sub>; often d<sub>v</sub> = d<sub>k</sub>)</p>
                <p>Output: n &times; d<sub>v</sub>&mdash;one vector per token</p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Structure, Better Parts">
            The attention formula is still &ldquo;compute weights, then average.&rdquo;
            Q/K improved the weights. V improved what gets averaged. Each replacement
            solved a specific limitation you felt in a previous lesson.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Worked Example with Concrete Numbers
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Worked Example: The Full Computation"
            subtitle="Same 4 tokens, now with V&mdash;from embedding to output"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Same 4 tokens from the last two lessons. Same embeddings, same W<sub>Q</sub> and
              W<sub>K</sub>. Now with a new 3&times;3 W<sub>V</sub> matrix.
            </p>

            {/* Show embeddings (callback) */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Same embeddings from the last two lessons:
              </p>
              <div className="space-y-1.5 font-mono text-sm">
                {TINY_EMBEDDINGS.map((item) => (
                  <div key={item.token} className="flex items-center gap-3 text-muted-foreground">
                    <span className="text-violet-400 w-12">&ldquo;{item.token}&rdquo;</span>
                    <span className="text-sky-400">{formatVec(item.vec)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Show W_V matrix */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-emerald-400 font-medium mb-2">
                W<sub>V</sub> (value projection)&mdash;new for this lesson:
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-1">
                {W_V.map((row, i) => (
                  <p key={i}>[{row.join(',  ')}]</p>
                ))}
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Three-Lesson Arc">
            Using the same 4 tokens across all three lessons lets you see the
            computation grow. Same input, progressively richer processing.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Step 1: Compute V vectors */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 1: Compute value vectors (V = W<sub>V</sub> &times; embedding)
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              {V_VECTORS.map((item) => (
                <div key={item.token} className="flex items-center gap-3 font-mono text-sm text-muted-foreground">
                  <span className="text-violet-400 w-12">&ldquo;{item.token}&rdquo;</span>
                  <span className="text-muted-foreground/50">&rarr;</span>
                  <span className="text-emerald-400">v = {formatVec(item.vec)}</span>
                </div>
              ))}
            </div>

            <p className="text-muted-foreground text-sm">
              Compare these to the K vectors from the last lesson:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="px-4 py-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-amber-400/80 font-medium mb-2">
                  K vectors (matching signal):
                </p>
                <div className="font-mono text-xs text-muted-foreground space-y-1">
                  {K_VECTORS.map((item) => (
                    <p key={item.token}>
                      <span className="text-violet-400/70">{item.token}</span>{' '}
                      <span className="text-amber-400">{formatVec(item.vec)}</span>
                    </p>
                  ))}
                </div>
              </div>
              <div className="px-4 py-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-emerald-400/80 font-medium mb-2">
                  V vectors (contribution signal):
                </p>
                <div className="font-mono text-xs text-muted-foreground space-y-1">
                  {V_VECTORS.map((item) => (
                    <p key={item.token}>
                      <span className="text-violet-400/70">{item.token}</span>{' '}
                      <span className="text-emerald-400">{formatVec(item.vec)}</span>
                    </p>
                  ))}
                </div>
              </div>
            </div>

            <p className="text-muted-foreground text-sm">
              Different matrix, different vectors&mdash;even though they come from the same
              embeddings. This is the &ldquo;three lenses&rdquo; in action.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Different Matrices, Different Vectors">
            K and V for &ldquo;cat&rdquo; are{' '}
            <span className="font-mono text-xs text-amber-400">{formatVec(K_VECTORS[1].vec)}</span> and{' '}
            <span className="font-mono text-xs text-emerald-400">{formatVec(V_VECTORS[1].vec)}</span>.
            Same embedding, different projections, different information extracted.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Step 2: Reuse attention weights */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 2: Attention weights (already computed in the last lesson)
            </p>
            <p className="text-muted-foreground text-sm">
              We already have <InlineMath math="\text{softmax}(QK^T / \sqrt{3})" /> from
              Queries and Keys. The weights don&rsquo;t change&mdash;V only affects what
              gets weighted, not the weights themselves.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg overflow-x-auto">
              <table className="text-sm font-mono w-full">
                <thead>
                  <tr className="text-muted-foreground/60 text-xs">
                    <th className="text-left pr-3 pb-2">weights</th>
                    {TINY_EMBEDDINGS.map((item) => (
                      <th key={item.token} className="text-center px-2 pb-2 text-violet-400/70">{item.token}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {TINY_EMBEDDINGS.map((rowItem, i) => (
                    <tr key={rowItem.token}>
                      <td className="text-violet-400/70 pr-3 py-1">{rowItem.token}</td>
                      {ATTENTION_WEIGHTS[i].map((weight, j) => (
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
          </div>
        </Row.Content>
      </Row>

      {/* Step 3: Compute the output for "cat" */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 3: Compute the attention output for &ldquo;cat&rdquo;
            </p>
            <p className="text-muted-foreground text-sm">
              &ldquo;Cat&rdquo;&rsquo;s output is the weighted sum of all V vectors, using
              &ldquo;cat&rdquo;&rsquo;s row of attention weights:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="font-mono text-sm text-muted-foreground space-y-2">
                <p className="text-xs text-muted-foreground/70">output<sub>cat</sub> =</p>
                {TINY_EMBEDDINGS.map((item, j) => (
                  <div key={item.token} className="flex items-center gap-2 pl-4">
                    <span className="text-sky-400 w-14">{ATTENTION_WEIGHTS[1][j].toFixed(3)}</span>
                    <span className="text-muted-foreground/50">&times;</span>
                    <span className="text-emerald-400">v<sub>{item.token}</sub> {formatVec(V_VECTORS[j].vec)}</span>
                    {j < TINY_EMBEDDINGS.length - 1 && (
                      <span className="text-muted-foreground/50 ml-1">+</span>
                    )}
                  </div>
                ))}
                <div className="pt-2 border-t border-border/50">
                  <p>
                    <span className="text-muted-foreground/50">=</span>{' '}
                    <span className="text-emerald-400 font-medium">{formatVec3(ATTENTION_OUTPUTS[1])}</span>
                  </p>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              You might picture attention as producing a single summary vector for the whole
              sequence&mdash;combining all tokens into one representation. It does not. Each
              token gets its <strong>own</strong> output vector, because each token has its own
              row of attention weights. &ldquo;Cat&rdquo; and &ldquo;sat&rdquo; have different
              attention weights (different rows of the weight matrix), so they compute different
              weighted averages of the <em>same</em> V vectors. Four tokens in, four vectors
              out.
            </p>

            {/* Show all 4 output vectors */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-xs text-muted-foreground/70 font-medium">
                All attention outputs:
              </p>
              {TINY_EMBEDDINGS.map((item, i) => (
                <div key={item.token} className="flex items-center gap-3 font-mono text-sm text-muted-foreground">
                  <span className="text-violet-400 w-12">&ldquo;{item.token}&rdquo;</span>
                  <span className="text-muted-foreground/50">&rarr;</span>
                  <span className="text-emerald-400">{formatVec3(ATTENTION_OUTPUTS[i])}</span>
                </div>
              ))}
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="One Vector Per Token">
            The output has the <strong>same shape</strong> as the input: one vector per
            token. Attention doesn&rsquo;t produce a single summary&mdash;it produces a
            context-enriched representation for <em>each</em> position.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Brief aside: What if W_V = W_K? */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-amber-400">
                What if W<sub>V</sub> = W<sub>K</sub>?
              </p>
              <p className="text-sm text-muted-foreground">
                If V and K used the same projection matrix, the contribution of each
                token would be its key vector&mdash;the matching signal. The weighted
                average would blend what made tokens <em>relevant</em> rather than
                what they have to <em>say</em>. The model loses a degree of freedom:
                the matching signal and the contribution signal are locked together.
              </p>
              <p className="text-sm text-muted-foreground">
                With separate W<sub>V</sub>, the model can learn that what makes
                &ldquo;chased&rdquo; relevant to &ldquo;cat&rdquo; (action verb semantics)
                is different from what &ldquo;chased&rdquo; contributes (tense, transitivity).
                Two separate roles, two separate projections.
              </p>
            </div>

            {/* What if W_V = Identity? (positive example from planning doc) */}
            <div className="px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-emerald-400">
                What if W<sub>V</sub> = identity matrix?
              </p>
              <p className="text-sm text-muted-foreground">
                Then V = I &times; embedding = the raw embedding itself. The output
                becomes <InlineMath math="\text{softmax}(QK^T / \sqrt{d_k}) \cdot X" />&mdash;exactly
                the formula from The Problem Attention Solves, but with better scoring
                (Q/K instead of raw dot products). V <strong>generalizes</strong> the
                raw-embedding output; identity is a special case. You&rsquo;ll verify
                this hands-on in the notebook.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: The Residual Stream
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Residual Stream"
            subtitle="Attention edits the embedding&mdash;it doesn&rsquo;t replace it"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We have the attention output&mdash;a context-dependent representation for each
              token. But does this <em>replace</em> the original embedding?
            </p>

            <p className="text-muted-foreground">
              Consider a token surrounded by uninformative context&mdash;stop words or
              padding. Its attention output would be a bland average of uninformative V
              vectors. If this replaces the original embedding, the token <strong>loses its
              identity</strong>. A meaningful word buried in noise would be erased.
            </p>

            <p className="text-muted-foreground">
              The fix: the attention output is <strong>added</strong> to the input, not
              substituted.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\text{final\_output}_i = \text{attention\_output}_i + \text{embedding}_i" />
              <p className="text-sm text-muted-foreground">
                This is the same pattern you implemented in the ResNets lesson. The
                residual block: <InlineMath math="\text{output} = F(x) + x" />. &ldquo;Editing,
                not writing.&rdquo; The attention output is the &ldquo;edit&rdquo;&mdash;a
                context-dependent correction that enriches the original embedding.
              </p>
            </div>

            {/* Concrete: show the residual output for "cat" */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Residual output for &ldquo;cat&rdquo;:
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-1">
                <p>
                  <span className="text-emerald-400">attention_output</span>{' '}
                  {formatVec3(ATTENTION_OUTPUTS[1])}
                </p>
                <p>
                  <span className="text-violet-400">+ embedding</span>{' '}
                  {formatVec(TINY_EMBEDDINGS[1].vec)}
                </p>
                <div className="pt-1 border-t border-border/50">
                  <p>
                    <span className="text-sky-400 font-medium">= final_output</span>{' '}
                    {formatVec3(RESIDUAL_OUTPUTS[1])}
                  </p>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              If attention learns nothing useful for a token (attention output is near zero),
              the final output is just the original embedding. The token keeps its meaning.
              The residual stream makes &ldquo;do nothing&rdquo; safe.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="From ResNets to Transformers">
            In ResNets, skip connections help gradients flow and make identity easy to
            learn. In the transformer, the residual stream does the same&mdash;but it
            also serves a deeper role: it preserves the original embedding while attention
            enriches it with context. You&rsquo;ll see in the next lessons how this
            becomes the backbone of the entire architecture when layers stack.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Check -- Transfer Question (Residual)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Transfer Question"
            subtitle="What breaks without the residual stream?"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A colleague removes the residual connection from their attention layer:
                they use <InlineMath math="\text{output} = \text{attention}(x)" /> instead
                of <InlineMath math="\text{output} = \text{attention}(x) + x" />. What breaks?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Two things break:</strong>
                  </p>
                  <p>
                    <strong>1. Information loss.</strong> A token surrounded by uninformative
                    context loses its identity. The attention output is a bland average of
                    neighbors&rsquo; V vectors&mdash;the token&rsquo;s own embedding information
                    is gone. With the residual connection, the original is always preserved.
                  </p>
                  <p>
                    <strong>2. Gradient flow.</strong> In a deep model, gradients must flow
                    entirely through the attention computation&mdash;no direct path from output
                    back to input. Remember the gradient highway from the ResNets lesson?
                    The residual connection provides that direct path. Without it, training a
                    deep transformer becomes much harder.
                  </p>
                  <p className="text-muted-foreground/70">
                    The residual connection provides both information preservation and
                    gradient flow&mdash;the same two benefits as in ResNets.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Notebook Exercise
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Complete Single-Head Attention"
            subtitle="Implement the full computation from embedding to output"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook covers the complete single-head attention pipeline:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Exercise 1:</strong> Implement V projection. Given embeddings X
                (4 tokens, d=64) and weight matrix W<sub>V</sub> (64&times;64), compute
                V = W<sub>V</sub> &times; X. Verify that V vectors differ from K vectors
                computed in the Lesson 2 notebook.
              </li>
              <li>
                <strong>Exercise 2:</strong> Implement complete single-head attention. Given
                X, W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, d<sub>k</sub>: compute Q,
                K, V, scores = QK<sup>T</sup>/&radic;d<sub>k</sub>,
                weights = softmax(scores), output = weights &times; V. Wrap it in a
                function <code className="text-xs">single_head_attention(X, W_Q, W_K, W_V)</code>.
              </li>
              <li>
                <strong>Exercise 3:</strong> Add the residual connection. Compute
                final_output = attention_output + X. Verify the shape matches the input.
              </li>
              <li>
                <strong>Exercise 4:</strong> Experiment&mdash;set W<sub>V</sub> = identity
                matrix. Compare the output to softmax(QK<sup>T</sup>/&radic;d<sub>k</sub>)
                &times; X (no V projection). They should be identical, proving V generalizes
                the raw-embedding output from The Problem Attention Solves.
              </li>
              <li>
                <strong>Exercise 5:</strong> Experiment&mdash;set W<sub>V</sub> = W<sub>K</sub>.
                Compare the output to when W<sub>V</sub> is independent. Observe that the
                contribution signal is locked to the matching signal.
              </li>
              <li>
                <strong>Stretch:</strong> Load pretrained GPT-2 weights. Extract one attention
                head&rsquo;s W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>. Run a sentence
                through and compare the V vectors to the K vectors for the same tokens&mdash;are
                they similar or different?
              </li>
            </ul>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook and implement single-head attention end-to-end.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-2-3-values-and-attention-output.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  Less scaffolding than previous notebooks&mdash;you&rsquo;ve implemented
                  Q, K, and raw attention already. This builds directly on that work.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Key Experiments">
            <ul className="space-y-2 text-sm">
              <li>&bull; Set W<sub>V</sub> = identity and recover Lesson 1&rsquo;s formula</li>
              <li>&bull; Set W<sub>V</sub> = W<sub>K</sub> and see the contribution lock</li>
              <li>&bull; Compare real GPT-2 K and V vectors for the same tokens</li>
            </ul>
          </TipBlock>
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
                  'V is a third learned projection that separates matching from contributing.',
                description:
                  'K encodes "what makes me relevant for matching." V encodes "what information I contribute when attended to." Three lenses, one embedding, three different views. The model learns all three from data.',
              },
              {
                headline:
                  'The full single-head attention formula: output = softmax(QK\u1D40/\u221Ad_k) V.',
                description:
                  'Each step in this formula exists because the version without it was insufficient. Raw X for scoring \u2192 Q/K. Raw X for contributing \u2192 V. Unscaled scores \u2192 \u221Ad_k scaling. Three targeted improvements over three lessons.',
              },
              {
                headline:
                  'The attention output is ADDED to the original embedding via the residual stream.',
                description:
                  '"Editing, not writing." The original embedding is always preserved. The attention output is a context-dependent correction, not a replacement. If attention learns nothing useful, the token keeps its original meaning.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Three-lesson arc echo + seed for multi-head */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-3">
              <p className="text-sm font-medium text-violet-400">
                The three-lesson arc
              </p>
              <p className="text-sm text-muted-foreground">
                You built this formula piece by piece. The Problem Attention Solves: raw
                dot-product attention (felt the dual-role limitation). Queries &amp; Keys:
                Q and K for asymmetric matching (felt the scaling problem, fixed it). This
                lesson: V for contribution, residual stream for preservation. Single-head
                attention is complete.
              </p>
            </div>

            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-violet-400">
                But one head isn&rsquo;t enough
              </p>
              <p className="text-sm text-muted-foreground">
                Single-head attention computes <strong>one</strong> notion of relevance.
                Consider: &ldquo;The cat sat on the mat because it was soft.&rdquo;
                &ldquo;It&rdquo; needs to attend to &ldquo;mat&rdquo; for coreference AND
                to &ldquo;soft&rdquo; for meaning. One set of Q/K/V weights can only capture
                one type of relationship at a time.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/multi-head-attention"
            title="Multi-Head Attention"
            description="Single-head attention captures one type of relationship. Multi-head attention runs multiple attention operations in parallel&mdash;each with its own Q/K/V, each capturing a different pattern."
            buttonText="Continue to Multi-Head Attention"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
