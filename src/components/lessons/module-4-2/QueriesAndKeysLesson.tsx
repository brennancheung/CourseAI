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
 * Queries, Keys, and the Relevance Function
 *
 * Second lesson in Module 4.2 (Attention & the Transformer).
 * Fifth lesson in Series 4 (LLMs & Transformers).
 *
 * Resolves the dual-role limitation from The Problem Attention Solves:
 * two learned projection matrices (W_Q, W_K) transform each embedding
 * into a query ("what am I looking for?") and a key ("what do I offer?"),
 * breaking the symmetry of raw dot-product attention. Also teaches
 * why scaling by sqrt(d_k) is essential to prevent softmax saturation.
 *
 * Core concepts at DEVELOPED:
 * - Q and K as learned linear projections (separate seeking/offering roles)
 * - Scaling by sqrt(d_k) to prevent softmax saturation
 *
 * Concepts REINFORCED:
 * - Dot-product attention formula (extended from XX^T to QK^T/sqrt(d_k))
 * - Dual-role limitation (resolved)
 * - Attention matrix symmetry (broken by Q/K)
 * - Vanishing gradients (callback for scaling motivation)
 * - Temperature/softmax saturation (callback for scaling)
 *
 * EXPLICITLY NOT COVERED:
 * - V projection (what a token contributes when attended to) -- Lesson 3
 * - The full attention output (the weighted sum using V) -- Lesson 3
 * - Multi-head attention -- Lesson 4
 * - The transformer block -- Lesson 5
 * - Causal masking -- Lesson 6
 *
 * Previous: The Problem Attention Solves (module 4.2, lesson 1)
 * Next: Values and the Attention Output (module 4.2, lesson 3)
 */

// ---------------------------------------------------------------------------
// Same 4-token embeddings from Lesson 1 (continuity for before/after)
// ---------------------------------------------------------------------------

const TINY_EMBEDDINGS = [
  { token: 'The', vec: [0.9, 0.1, -0.2] },
  { token: 'cat', vec: [-0.3, 0.8, 0.5] },
  { token: 'sat', vec: [0.1, 0.6, 0.4] },
  { token: 'here', vec: [0.4, -0.2, 0.7] },
]

// W_Q and W_K: small integer-ish values for hand-traceability
// Chosen so QK^T is clearly asymmetric
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

// ---------------------------------------------------------------------------
// Computation helpers
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

// Compute Q and K vectors for each token
const Q_VECTORS = TINY_EMBEDDINGS.map((item) => ({
  token: item.token,
  vec: matVecMul(W_Q, item.vec),
}))

const K_VECTORS = TINY_EMBEDDINGS.map((item) => ({
  token: item.token,
  vec: matVecMul(W_K, item.vec),
}))

// Compute QK^T (4x4 matrix)
const QK_SCORES: number[][] = Q_VECTORS.map((qi) =>
  K_VECTORS.map((kj) => dotProduct(qi.vec, kj.vec))
)

// Compute raw XX^T for comparison (from Lesson 1)
const RAW_SCORES: number[][] = TINY_EMBEDDINGS.map((xi) =>
  TINY_EMBEDDINGS.map((xj) => dotProduct(xi.vec, xj.vec))
)

// Softmax of QK^T rows
const QK_WEIGHTS = QK_SCORES.map((row) => softmaxRow(row))

// Softmax of raw XX^T rows
const RAW_WEIGHTS = RAW_SCORES.map((row) => softmaxRow(row))

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatVec(vec: number[]): string {
  return `[${vec.map((v) => v.toFixed(2)).join(', ')}]`
}

// ---------------------------------------------------------------------------
// Heatmap cell color helper
// ---------------------------------------------------------------------------

function weightColor(w: number): string {
  if (w >= 0.35) return 'text-emerald-400 font-medium'
  if (w >= 0.28) return 'text-sky-400'
  return 'text-muted-foreground'
}

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function QueriesAndKeysLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Queries, Keys, and the Relevance Function"
            description="Two learned projection matrices break the symmetry limitation&mdash;giving each token separate vectors for seeking and offering."
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
            Understand how learned linear projections (Q&nbsp;=&nbsp;W<sub>Q</sub>&nbsp;&times;&nbsp;embedding,
            K&nbsp;=&nbsp;W<sub>K</sub>&nbsp;&times;&nbsp;embedding) let each token create separate
            &ldquo;seeking&rdquo; and &ldquo;offering&rdquo; vectors, breaking the symmetry
            limitation of raw attention&mdash;and why scaling by{' '}
            <InlineMath math="\sqrt{d_k}" /> is essential to keep the model trainable.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In The Problem Attention Solves, you built dot-product attention from scratch and felt
            its limitation: one embedding per token for both seeking and offering. This lesson
            fixes that with two learned projection matrices.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why Q and K exist (resolving the dual-role limitation)',
              'What Q and K compute (learned linear projections of the same embedding)',
              'How QK\u1D40 produces an asymmetric relevance matrix',
              'Why scaling by \u221Ad_k is essential (softmax saturation, vanishing gradients)',
              'Hand-tracing the full computation with the same 4 tokens from last lesson',
              'Has notebook: implement Q, K projections and verify asymmetry',
              'NOT: V projection (what a token contributes when attended to)\u2014that\u2019s next lesson',
              'NOT: multi-head attention, transformer block, or causal masking',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Hook -- The Cliffhanger Resolution
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Cliffhanger Resolution"
            subtitle="You already answered the question. Now let&rsquo;s build the mechanism."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              At the end of The Problem Attention Solves, you answered a transfer question:
            </p>

            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <p className="text-sm text-violet-400 italic">
                &ldquo;If every token had two vectors&mdash;one for &lsquo;what I&rsquo;m looking
                for&rsquo; and one for &lsquo;what I&rsquo;m advertising&rsquo;&mdash;would
                the attention matrix still be symmetric?&rdquo;
              </p>
            </div>

            <p className="text-muted-foreground">
              You reasoned: <strong>no</strong>. If token A&rsquo;s seeking vector dotted with
              token B&rsquo;s advertising vector gives a different result than B&rsquo;s seeking
              dotted with A&rsquo;s advertising, the symmetry breaks. Each direction can express a
              different reason for relevance.
            </p>
            <p className="text-muted-foreground">
              This lesson delivers exactly that mechanism. Two learned matrices that transform
              each embedding into two different vectors: a <strong>query</strong> (what am I
              looking for?) and a <strong>key</strong> (what do I have to offer?).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="From Feeling to Mechanism">
            You felt the dual-role limitation. You predicted the fix. Now you&rsquo;ll see the
            exact mathematical construction that makes it work.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Two Projections from One Embedding
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Two Projections from One Embedding"
            subtitle="The job fair analogy"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine a job fair. Every attendee writes <strong>two cards</strong>:
            </p>

            <ComparisonRow
              left={{
                title: 'What I\'m Looking For',
                color: 'cyan',
                items: [
                  'A software engineer seeks a team with ML challenges',
                  'This card represents what they want from others',
                  'Different people seek different things',
                ],
              }}
              right={{
                title: 'What I Bring to the Table',
                color: 'orange',
                items: [
                  'The same engineer offers 5 years of Python experience',
                  'This card represents what they provide to others',
                  'What you seek and what you offer are different',
                ],
              }}
            />

            <p className="text-muted-foreground">
              A recruiter scores each seeker-card against each provider-card. One person&rsquo;s
              &ldquo;looking for&rdquo; card is fundamentally different from their &ldquo;bring to the
              table&rdquo; card. The score for person A looking at person B is not the same as
              B looking at A&mdash;because A&rsquo;s seeking card matches against B&rsquo;s
              offering card, and vice versa.
            </p>
            <p className="text-muted-foreground">
              This is exactly what Q and K do for attention. Two different <strong>learned
              matrices</strong> transform the same embedding into two different vectors:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  <strong className="text-sky-400">Query:</strong>{' '}
                  <InlineMath math="Q = W_Q \cdot X" />&ensp;&mdash;&ensp;&ldquo;what is this token looking for?&rdquo;
                </p>
                <p>
                  <strong className="text-amber-400">Key:</strong>{' '}
                  <InlineMath math="K = W_K \cdot X" />&ensp;&mdash;&ensp;&ldquo;what does this token have to offer?&rdquo;
                </p>
              </div>
              <p className="text-xs text-muted-foreground/70">
                <InlineMath math="W_Q" /> and <InlineMath math="W_K" /> are separate learned
                weight matrices. Same input embedding, different transformation, different output
                vector.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Learned Lens">
            Think of <InlineMath math="W_Q" /> as a lens that reveals what the token
            is <em>seeking</em>. <InlineMath math="W_K" /> is a different lens that reveals
            what the token <em>offers</em>. Same embedding, different lens, different view.
            The lens is what the model learns during training.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* nn.Linear connection + Warning about misconception */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In PyTorch, each projection is just an <code className="text-xs">nn.Linear</code> layer&mdash;the
              same building block you&rsquo;ve been using since your first neural network in Module 2.1:
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg font-mono text-sm text-muted-foreground">
              <p><span className="text-sky-400">self.W_Q</span> = nn.Linear(d_model, d_k, bias=False)</p>
              <p><span className="text-amber-400">self.W_K</span> = nn.Linear(d_model, d_k, bias=False)</p>
            </div>
            <p className="text-muted-foreground">
              Nothing new here. A learned weight matrix multiplied by the input. The only
              difference is that now we apply <em>two</em> different matrices to the same input
              to get two different outputs.
            </p>

            {/* Geometric/spatial brief: same vector, two projections */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Geometric picture: same point, two projections
              </p>
              <div className="flex justify-center py-2">
                <svg width="280" height="160" viewBox="0 0 280 160" className="overflow-visible">
                  {/* Origin point (embedding) */}
                  <circle cx="140" cy="120" r="5" fill="#a78bfa" />
                  <text x="148" y="135" fill="#a78bfa" fontSize="11" fontFamily="monospace">embedding</text>

                  {/* W_Q arrow (up-left) */}
                  <line x1="140" y1="120" x2="60" y2="35" stroke="#38bdf8" strokeWidth={2} strokeDasharray="4 2" />
                  <polygon points="60,35 68,42 72,34" fill="#38bdf8" />
                  <text x="30" y="28" fill="#38bdf8" fontSize="11" fontFamily="monospace">Q</text>
                  <text x="75" y="68" fill="#38bdf8" fontSize="9" fontFamily="monospace" opacity={0.7}>W_Q</text>

                  {/* W_K arrow (up-right) */}
                  <line x1="140" y1="120" x2="230" y2="45" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 2" />
                  <polygon points="230,45 222,42 226,52" fill="#f59e0b" />
                  <text x="238" y="42" fill="#f59e0b" fontSize="11" fontFamily="monospace">K</text>
                  <text x="195" y="73" fill="#f59e0b" fontSize="9" fontFamily="monospace" opacity={0.7}>W_K</text>

                  {/* Labels */}
                  <text x="70" y="155" fill="currentColor" fontSize="10" fontFamily="monospace" opacity={0.4}>same input, different matrices, different destinations</text>
                </svg>
              </div>
              <p className="text-xs text-muted-foreground/70">
                The same embedding vector lands at different locations depending on which
                matrix projects it. Q and K live in the same <InlineMath math="d_k" />-dimensional
                space&mdash;that&rsquo;s why the dot product between them is meaningful.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Q and K Are Not Token Properties">
            Q and K are properties of the <strong>learned projection matrices</strong>, not
            the token itself. Change the matrices (by training longer), and the same
            token&rsquo;s Q and K change. Different layers of the same model produce different
            Q and K vectors from the same embedding, because each layer has its own{' '}
            <InlineMath math="W_Q" /> and <InlineMath math="W_K" />.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: The Relevance Matrix QK^T
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Relevance Matrix QK&#x1D40;"
            subtitle="From similarity to learned relevance"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <InlineMath math="QK^T" /> replaces <InlineMath math="XX^T" />. Entry{' '}
              <InlineMath math="(i, j)" /> is <InlineMath math="q_i \cdot k_j" />&mdash;&ldquo;how
              much does token <em>i</em>&rsquo;s query match token <em>j</em>&rsquo;s key?&rdquo;
            </p>

            <p className="text-muted-foreground">
              This is <strong>not</strong> similarity between raw embeddings. It&rsquo;s a learned
              relevance function. The projections transform embeddings into a space where dot
              products measure &ldquo;is this token&rsquo;s offering what that token is
              seeking?&rdquo;
            </p>

            <p className="text-muted-foreground">
              Remember the &ldquo;bank&rdquo; and &ldquo;steep&rdquo; example from last
              lesson&mdash;low embedding similarity but high relevance for disambiguation.
              In a trained model, <InlineMath math="W_Q" /> and <InlineMath math="W_K" /> would
              learn projections that make &ldquo;steep&rdquo; and &ldquo;bank&rdquo; produce a
              high <InlineMath math="q \cdot k" /> score&mdash;even though their raw embeddings
              aren&rsquo;t similar. That&rsquo;s the power of learned projections: the model can
              train the matrices to map vectors into a space where dot products reflect relevance
              rather than raw similarity.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground">
                <strong>Why QK&#x1D40; is asymmetric:</strong>
              </p>
              <div className="space-y-1 text-sm text-muted-foreground font-mono">
                <p>entry(i, j) = q<sub>i</sub> &middot; k<sub>j</sub></p>
                <p>entry(j, i) = q<sub>j</sub> &middot; k<sub>i</sub></p>
              </div>
              <p className="text-sm text-muted-foreground">
                Since <InlineMath math="q_i = W_Q \cdot x_i" /> and{' '}
                <InlineMath math="k_i = W_K \cdot x_i" />, and{' '}
                <InlineMath math="W_Q \neq W_K" />, the query of token A dotted with the key
                of token B is <strong>not the same</strong> as B&rsquo;s query dotted with
                A&rsquo;s key.
              </p>
              <p className="text-sm text-muted-foreground">
                This dot product is meaningful because Q and K live in the
                same <InlineMath math="d_k" />-dimensional space&mdash;they arrived there via
                different matrices, but they share the same coordinate system. If they were in
                different spaces, the dot product between them would be meaningless.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Seeking &ne; Offering">
            &ldquo;The seeking vector of token A dotted with the offering vector of token B
            is not the same as B&rsquo;s seeking dotted with A&rsquo;s offering.&rdquo;
            Different questions, different scores.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Check -- Predict Asymmetry
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Predict the Asymmetry"
            subtitle="Reason through a concrete case"
          />
          <GradientCard title="Prediction Exercise" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Consider: &ldquo;The cat chased the mouse.&rdquo;
              </p>
              <p>
                With Q/K projections, should &ldquo;cat&rdquo;&rsquo;s attention to
                &ldquo;chased&rdquo; equal &ldquo;chased&rdquo;&rsquo;s attention to
                &ldquo;cat&rdquo;? Why or why not?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>No.</strong> &ldquo;Cat&rdquo; is seeking
                    what-action-did-I-do&mdash;its query vector reflects that.
                    &ldquo;Chased&rdquo; is seeking who-did-the-chasing&mdash;a completely
                    different query.
                  </p>
                  <p>
                    Same pair of tokens, different queries, same keys. Different dot products,
                    different scores. The symmetry from raw attention is gone.
                  </p>
                  <p className="text-muted-foreground/70">
                    This is the resolution of the &ldquo;cat chased mouse&rdquo; asymmetry
                    problem from the previous lesson.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: Worked Example with Concrete Numbers
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Worked Example: Same Tokens, Now with Projections"
            subtitle="Hand-trace Q, K, and QK&#x1D40; with real numbers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Same 4 tokens from The Problem Attention Solves. Same 3-dimensional embeddings.
              Now with two 3&times;3 projection matrices.
            </p>

            {/* Show the embeddings (callback) */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Same embeddings from last lesson:
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

            {/* Show W_Q and W_K */}
            <div className="grid gap-4 md:grid-cols-2">
              <div className="px-4 py-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-sky-400 font-medium mb-2">
                  W<sub>Q</sub> (query projection):
                </p>
                <div className="font-mono text-sm text-muted-foreground space-y-1">
                  {W_Q.map((row, i) => (
                    <p key={i}>[{row.join(',  ')}]</p>
                  ))}
                </div>
              </div>
              <div className="px-4 py-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-amber-400 font-medium mb-2">
                  W<sub>K</sub> (key projection):
                </p>
                <div className="font-mono text-sm text-muted-foreground space-y-1">
                  {W_K.map((row, i) => (
                    <p key={i}>[{row.join(',  ')}]</p>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Tokens, Different Mechanism">
            Using the same 4 tokens from last lesson lets you see the before/after directly.
            Same input, different processing, different attention pattern.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Step 1: Compute Q vectors */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 1: Compute query vectors (Q = W<sub>Q</sub> &times; embedding)
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              {Q_VECTORS.map((item) => (
                <div key={item.token} className="flex items-center gap-3 font-mono text-sm text-muted-foreground">
                  <span className="text-violet-400 w-12">&ldquo;{item.token}&rdquo;</span>
                  <span className="text-muted-foreground/50">&rarr;</span>
                  <span className="text-sky-400">q = {formatVec(item.vec)}</span>
                </div>
              ))}
            </div>

            <p className="text-muted-foreground font-medium">
              Step 2: Compute key vectors (K = W<sub>K</sub> &times; embedding)
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              {K_VECTORS.map((item) => (
                <div key={item.token} className="flex items-center gap-3 font-mono text-sm text-muted-foreground">
                  <span className="text-violet-400 w-12">&ldquo;{item.token}&rdquo;</span>
                  <span className="text-muted-foreground/50">&rarr;</span>
                  <span className="text-amber-400">k = {formatVec(item.vec)}</span>
                </div>
              ))}
            </div>

            <p className="text-muted-foreground text-sm">
              Same embeddings, different matrices, different vectors. This is the &ldquo;learned
              lens&rdquo; in action&mdash;the same input looks different through different
              projections. If we used entirely different <InlineMath math="W_Q" /> and{' '}
              <InlineMath math="W_K" /> matrices, the attention pattern would be completely
              different&mdash;same tokens, same embeddings, different behavior. The matrices
              are what the model learns, and they determine the attention pattern.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Views of One Token">
            &ldquo;Cat&rdquo;&rsquo;s query vector is{' '}
            <span className="font-mono text-xs text-sky-400">{formatVec(Q_VECTORS[1].vec)}</span>.
            Its key vector is{' '}
            <span className="font-mono text-xs text-amber-400">{formatVec(K_VECTORS[1].vec)}</span>.
            Different vectors from the same embedding.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Step 3: Compute QK^T */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 3: Compute QK&#x1D40; (each entry is q<sub>i</sub> &middot; k<sub>j</sub>)
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg overflow-x-auto">
              <table className="text-sm font-mono w-full">
                <thead>
                  <tr className="text-muted-foreground/60 text-xs">
                    <th className="text-left pr-3 pb-2">QK&#x1D40;</th>
                    {TINY_EMBEDDINGS.map((item) => (
                      <th key={item.token} className="text-center px-2 pb-2 text-violet-400/70">{item.token}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {TINY_EMBEDDINGS.map((rowItem, i) => (
                    <tr key={rowItem.token}>
                      <td className="text-violet-400/70 pr-3 py-1">{rowItem.token}</td>
                      {QK_SCORES[i].map((score, j) => (
                        <td
                          key={j}
                          className={`text-center px-2 py-1 ${
                            i === j ? 'text-amber-400 font-medium' : 'text-muted-foreground'
                          }`}
                        >
                          {score.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground text-sm">
              Look at this matrix carefully. Compare entry (cat, sat) with entry (sat, cat).
              Are they the same?{' '}
              <strong>
                {QK_SCORES[1][2] === QK_SCORES[2][1]
                  ? 'They happen to be equal in this case.'
                  : 'No\u2014they\u2019re different.'}
              </strong>{' '}
              This matrix is <strong>not symmetric</strong>. The cliffhanger from last lesson
              is resolved.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Asymmetry Achieved">
            Remember <InlineMath math="XX^T" /> from last lesson was symmetric&mdash;
            <InlineMath math="a \cdot b = b \cdot a" /> always. Now{' '}
            <InlineMath math="q_i \cdot k_j \neq q_j \cdot k_i" /> because the query and
            key of each token come from different projection matrices.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Step 4: Apply softmax */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 4: Apply softmax to each row
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg overflow-x-auto">
              <table className="text-sm font-mono w-full">
                <thead>
                  <tr className="text-muted-foreground/60 text-xs">
                    <th className="text-left pr-3 pb-2">softmax(QK&#x1D40;)</th>
                    {TINY_EMBEDDINGS.map((item) => (
                      <th key={item.token} className="text-center px-2 pb-2 text-violet-400/70">{item.token}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {TINY_EMBEDDINGS.map((rowItem, i) => (
                    <tr key={rowItem.token}>
                      <td className="text-violet-400/70 pr-3 py-1">{rowItem.token}</td>
                      {QK_WEIGHTS[i].map((weight, j) => (
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

            <p className="text-muted-foreground text-sm">
              Each row sums to 1. These are the attention weights with Q/K
              projections&mdash;a very different pattern from the raw{' '}
              <InlineMath math="XX^T" /> weights you computed last lesson.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Side-by-side comparison: raw vs QK^T */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Side-by-side: raw attention vs. projected attention
            </p>
            <p className="text-muted-foreground text-sm">
              Same 4 tokens, same embeddings. Left: the symmetric weights from raw{' '}
              <InlineMath math="XX^T" /> (last lesson). Right: the asymmetric weights from{' '}
              <InlineMath math="QK^T" /> (this lesson).
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              {/* Raw attention heatmap */}
              <div className="px-4 py-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-rose-400/80 font-medium mb-2">
                  Raw: softmax(XX&#x1D40;)&mdash;symmetric
                </p>
                <table className="text-xs font-mono w-full">
                  <thead>
                    <tr className="text-muted-foreground/50">
                      <th className="text-left pr-2 pb-1"></th>
                      {TINY_EMBEDDINGS.map((item) => (
                        <th key={item.token} className="text-center px-1 pb-1">{item.token}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {TINY_EMBEDDINGS.map((rowItem, i) => (
                      <tr key={rowItem.token}>
                        <td className="text-muted-foreground/50 pr-2 py-0.5">{rowItem.token}</td>
                        {RAW_WEIGHTS[i].map((weight, j) => (
                          <td
                            key={j}
                            className={`text-center px-1 py-0.5 ${weightColor(weight)}`}
                          >
                            {weight.toFixed(3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* QK^T attention heatmap */}
              <div className="px-4 py-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-emerald-400/80 font-medium mb-2">
                  Projected: softmax(QK&#x1D40;)&mdash;asymmetric
                </p>
                <table className="text-xs font-mono w-full">
                  <thead>
                    <tr className="text-muted-foreground/50">
                      <th className="text-left pr-2 pb-1"></th>
                      {TINY_EMBEDDINGS.map((item) => (
                        <th key={item.token} className="text-center px-1 pb-1">{item.token}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {TINY_EMBEDDINGS.map((rowItem, i) => (
                      <tr key={rowItem.token}>
                        <td className="text-muted-foreground/50 pr-2 py-0.5">{rowItem.token}</td>
                        {QK_WEIGHTS[i].map((weight, j) => (
                          <td
                            key={j}
                            className={`text-center px-1 py-0.5 ${weightColor(weight)}`}
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

            <p className="text-muted-foreground text-sm">
              The raw attention weights (left) are nearly symmetric&mdash;the raw scores{' '}
              <InlineMath math="x_i \cdot x_j = x_j \cdot x_i" /> are identical, and
              only softmax&rsquo;s row-wise normalization introduces slight differences.
              The projected weights (right) can be very different in each direction: how
              much &ldquo;cat&rdquo; attends to &ldquo;sat&rdquo; is not the same as how much
              &ldquo;sat&rdquo; attends to &ldquo;cat.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Before and After">
            Same tokens, same embeddings. The only difference is two 3&times;3 matrices. Yet
            the attention pattern is fundamentally changed: asymmetric, with learned relevance
            replacing raw similarity.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Notebook teaser (full exercises after scaling section)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/5 border border-primary/20 rounded-lg">
            <p className="text-sm text-muted-foreground">
              You&rsquo;ll implement Q and K projections from scratch in the notebook at the
              end of this lesson&mdash;including an experiment where you set{' '}
              <InlineMath math="W_Q = W_K" /> and watch symmetry return. First, one more
              critical piece: why scaling matters.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Why Scaling by sqrt(d_k) Matters
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Scaling by &radic;d&#8342; Matters"
            subtitle="From a numerical nuisance to a training necessity"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We&rsquo;ve been using <InlineMath math="d_k = 3" />. Real models use{' '}
              <InlineMath math="d_k = 64" /> or <InlineMath math="d_k = 128" />. What
              changes when the dimension grows?
            </p>

            <p className="text-muted-foreground">
              Here&rsquo;s the key fact: if each element of <InlineMath math="q" /> and{' '}
              <InlineMath math="k" /> has mean 0 and variance 1, then{' '}
              <InlineMath math="q \cdot k" /> has mean 0 and <strong>variance{' '}
              <InlineMath math="d_k" /></strong>. Each dimension contributes one term to the
              sum, and the variances add up.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Typical dot product magnitudes by dimension:
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-1">
                <div className="flex items-center gap-3">
                  <span className="w-20 text-emerald-400">d<sub>k</sub> = 3</span>
                  <span>typical dot product &asymp; &plusmn;1.7</span>
                  <span className="text-emerald-400/70 ml-auto text-xs">fine</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-20 text-amber-400">d<sub>k</sub> = 64</span>
                  <span>typical dot product &asymp; &plusmn;8.0</span>
                  <span className="text-amber-400/70 ml-auto text-xs">getting large</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-20 text-rose-400">d<sub>k</sub> = 512</span>
                  <span>typical dot product &asymp; &plusmn;22.6</span>
                  <span className="text-rose-400/70 ml-auto text-xs">catastrophic</span>
                </div>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Variance Adds Up">
            Each dimension of the dot product adds one independent random variable.
            With <InlineMath math="d_k" /> dimensions, the variance of the sum
            is <InlineMath math="d_k" /> times the variance of each term. The standard
            deviation grows as <InlineMath math="\sqrt{d_k}" />.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Temperature callback */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the temperature slider from What is a Language Model? When you set{' '}
              <InlineMath math="T = 0.1" />, softmax concentrated nearly all mass on the top
              token. Large dot products have exactly the same effect&mdash;they act like
              dividing by a tiny temperature.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                What softmax does with large inputs:
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-2">
                <div>
                  <span className="text-emerald-400">softmax([1.7, -0.3, 0.5, 0.8])</span>
                  <span className="text-muted-foreground/50 mx-2">&rarr;</span>
                  <span>[0.543, 0.073, 0.163, 0.221]</span>
                  <span className="text-emerald-400/70 ml-2 text-xs">useful distribution</span>
                </div>
                <div>
                  <span className="text-rose-400">softmax([22, -15, 3, 8])</span>
                  <span className="text-muted-foreground/50 mx-2">&rarr;</span>
                  <span>[1.000, 0.000, 0.000, 0.000]</span>
                  <span className="text-rose-400/70 ml-2 text-xs">one-hot&mdash;no learning</span>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              When softmax outputs are near 0 and 1, its gradients are near zero. You already
              know what happens when gradients vanish&mdash;the telephone game from training
              dynamics. The signal dies, the model stops learning. At{' '}
              <InlineMath math="d_k = 512" /> without scaling, the model is effectively frozen.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Cosmetic">
            Scaling by <InlineMath math="\sqrt{d_k}" /> is not an optional cleanup step. Without
            it, high-dimensional models collapse: softmax saturates, gradients vanish, training
            fails. The scaling factor is the difference between a model that learns and one
            that doesn&rsquo;t.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* The fix */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              The fix: divide by <InlineMath math="\sqrt{d_k}" /> before softmax
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\text{scores} = \frac{QK^T}{\sqrt{d_k}}" />
              <p className="text-sm text-muted-foreground">
                The variance of the dot product is <InlineMath math="d_k" />. Dividing by{' '}
                <InlineMath math="\sqrt{d_k}" /> brings the variance back to 1, regardless of
                dimension. Softmax receives inputs of similar magnitude whether{' '}
                <InlineMath math="d_k = 3" /> or <InlineMath math="d_k = 512" />.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Without Scaling (d_k = 512)" color="rose">
                <div className="space-y-1 text-sm">
                  <p>Dot products &asymp; &plusmn;22.6</p>
                  <p>Softmax &rarr; nearly one-hot</p>
                  <p>Gradients &rarr; near zero</p>
                  <p>Training &rarr; <strong>collapses</strong></p>
                </div>
              </GradientCard>
              <GradientCard title="With Scaling (d_k = 512)" color="emerald">
                <div className="space-y-1 text-sm">
                  <p>Scaled scores &asymp; &plusmn;1.0</p>
                  <p>Softmax &rarr; useful distribution</p>
                  <p>Gradients &rarr; healthy</p>
                  <p>Training &rarr; <strong>learns normally</strong></p>
                </div>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Why <InlineMath math="\sqrt{d_k}" /> specifically? Because the variance of the
              dot product is <InlineMath math="d_k" />, so dividing by{' '}
              <InlineMath math="\sqrt{d_k}" /> normalizes it to variance 1. It&rsquo;s not
              arbitrary&mdash;it&rsquo;s the exact correction that keeps score magnitudes
              constant regardless of dimension.
            </p>

            <p className="text-muted-foreground">
              You might wonder: if scaling fixes the problem, why not just make{' '}
              <InlineMath math="d_k" /> as large as possible? More dimensions capture
              finer-grained relevance, but each projection matrix has{' '}
              <InlineMath math="d_{\text{model}} \times d_k" /> parameters. In practice,
              models keep <InlineMath math="d_k" /> intentionally small (typically 64) and
              use <strong>multiple attention heads</strong> that each get their own projections.
              Why that tradeoff works is a question for the multi-head attention lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="d_k = 3 Works Fine Without It">
            At <InlineMath math="d_k = 3" />, typical dot products are already around &plusmn;1.7.
            Softmax handles that just fine. This is why our toy example didn&rsquo;t need
            scaling&mdash;but real models with <InlineMath math="d_k = 64" /> or higher
            absolutely do.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* The complete formula so far */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              The formula so far
            </p>
            <p className="text-muted-foreground">
              Putting Q, K, and the scaling factor together, the attention weights are:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="space-y-1 text-sm text-muted-foreground">
                <p>1. Project: <InlineMath math="Q = W_Q \cdot X" />, <InlineMath math="K = W_K \cdot X" /></p>
                <p>2. Score: <InlineMath math="S = QK^T / \sqrt{d_k}" /></p>
                <p>3. Normalize: <InlineMath math="W = \text{softmax}(S)" /> (row-wise)</p>
              </div>
              <div className="pt-2 border-t border-border/50">
                <BlockMath math="\text{weights} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)" />
              </div>
            </div>
            <p className="text-muted-foreground">
              Compare to last lesson&rsquo;s formula:{' '}
              <InlineMath math="\text{softmax}(XX^T)" />. Two additions: projections{' '}
              (<InlineMath math="W_Q" />, <InlineMath math="W_K" />) and scaling{' '}
              (<InlineMath math="\sqrt{d_k}" />). Same structure, but now the model can learn
              what to attend to rather than relying on raw embedding similarity.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pattern, Two Additions">
            The attention computation didn&rsquo;t fundamentally change. It&rsquo;s still
            &ldquo;compute scores, normalize with softmax.&rdquo; Q and K replace raw
            embeddings. Scaling prevents a numerical issue. Everything else is identical.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Check -- Transfer Question (Scaling)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Transfer Question"
            subtitle="Test your understanding of scaling"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A colleague says: &ldquo;I removed the <InlineMath math="\sqrt{d_k}" /> scaling
                and my small <InlineMath math="d_k = 8" /> model still trains fine, so scaling
                doesn&rsquo;t matter.&rdquo;
              </p>
              <p>
                What would you tell them?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    At <InlineMath math="d_k = 8" />, typical dot products are around &plusmn;2.8.
                    That&rsquo;s large enough to somewhat sharpen the softmax, but not large
                    enough to fully saturate it. The model can still learn.
                  </p>
                  <p>
                    At <InlineMath math="d_k = 64" /> or <InlineMath math="d_k = 128" />, those
                    same dot products would be &plusmn;8 or &plusmn;11.3&mdash;large enough to
                    push softmax toward one-hot, killing gradients. The colleague&rsquo;s test
                    doesn&rsquo;t generalize to the dimensions used by real transformer models.
                  </p>
                  <p className="text-muted-foreground/70">
                    This is exactly why it&rsquo;s called &ldquo;scaled&rdquo; dot-product
                    attention. The scaling is part of the standard formula for a reason.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Consolidated Notebook Exercise
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Q, K, and Scaled Attention"
            subtitle="Implement everything from this lesson in code"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook covers the full Q/K pipeline&mdash;projections, asymmetry, and scaling:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Exercise 1:</strong> Implement Q and K projections from scratch (matrix
                multiplication, not nn.Linear yet). Compute <InlineMath math="QK^T" /> and
                verify it&rsquo;s not symmetric. Visualize as a heatmap and compare to
                raw <InlineMath math="XX^T" /> from last lesson&rsquo;s notebook.
              </li>
              <li>
                <strong>Exercise 2:</strong> Set <InlineMath math="W_Q = W_K" /> and watch
                symmetry return. Then make them different and watch it break. This proves the
                asymmetry comes from the matrices being different.
              </li>
              <li>
                <strong>Exercise 3:</strong> Implement scaled attention weights. Divide scores
                by <InlineMath math="\sqrt{d_k}" />, apply softmax. Compare the weight
                distribution with and without scaling.
              </li>
              <li>
                <strong>Exercise 4:</strong> Experiment with dimension. Set{' '}
                <InlineMath math="d_k" /> to 8, 64, 512. For each, compute{' '}
                <InlineMath math="QK^T" /> without scaling and look at the softmax output.
                At what dimension does the distribution become effectively one-hot?
              </li>
              <li>
                <strong>Stretch:</strong> Use pretrained GPT-2 embeddings. Extract one attention
                head&rsquo;s <InlineMath math="W_Q" /> and <InlineMath math="W_K" /> matrices.
                Compute attention weights for &ldquo;The cat chased the mouse.&rdquo; Is the
                pattern asymmetric?
              </li>
            </ul>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook and implement scaled dot-product attention with Q and K projections.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-2-2-queries-and-keys.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  Less scaffolding than last lesson&rsquo;s notebook&mdash;you already know the
                  attention computation pattern.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Key Experiments">
            <ul className="space-y-2 text-sm">
              <li>&bull; Set <InlineMath math="W_Q = W_K" /> and watch symmetry return</li>
              <li>&bull; Compare softmax outputs at d<sub>k</sub>=8, 64, and 512 without scaling</li>
              <li>&bull; Add the <InlineMath math="\sqrt{d_k}" /> divisor and see it recover</li>
            </ul>
          </TryThisBlock>
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
                  'Q and K are learned linear projections that separate seeking and offering.',
                description:
                  'Two different weight matrices (W_Q, W_K) transform the same embedding into a query ("what am I looking for?") and a key ("what do I have to offer?"). This breaks the symmetry of raw dot-product attention.',
              },
              {
                headline:
                  'QK\u1D40 computes a learned relevance function, not raw embedding similarity.',
                description:
                  'Entry (i, j) is q_i \u00B7 k_j\u2014how much does token i\u2019s query match token j\u2019s key? The projections can learn that "steep" is relevant to "bank" even when their raw embeddings aren\u2019t similar.',
              },
              {
                headline:
                  'Scaling by \u221Ad_k prevents softmax saturation as dimensions grow.',
                description:
                  'Dot product variance grows with d_k. Without scaling, high-dimensional scores push softmax toward one-hot, killing gradients. Dividing by \u221Ad_k normalizes variance to 1, keeping the model trainable.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Forward reference: seeding V and the output */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-violet-400">
                What&rsquo;s next: the missing piece
              </p>
              <p className="text-sm text-muted-foreground">
                We have attention <strong>weights</strong>&mdash;how much each token should
                attend to each other token. But weights need something to weight. When we
                compute the final weighted average, what vectors do we average?
              </p>
              <p className="text-sm text-muted-foreground">
                Right now, we&rsquo;d use the raw embeddings. But think about it: the key that
                makes a token relevant (K) and the information it should <em>provide</em> when
                attended to are different things. &ldquo;Chased&rdquo; is relevant to
                &ldquo;cat&rdquo; because of its action semantics (captured by K), but what
                &ldquo;chased&rdquo; <em>contributes</em> to &ldquo;cat&rdquo;&rsquo;s
                representation should be different&mdash;maybe verb tense, transitivity, or
                past-action features.
              </p>
              <p className="text-sm text-muted-foreground">
                Should we use the same embedding for matching AND contributing? We already
                know the answer to that kind of question.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/values-and-attention-output"
            title="Values and the Attention Output"
            description="Q and K fixed matching. There&rsquo;s one more role separation needed: the third projection that separates &ldquo;what makes me relevant&rdquo; from &ldquo;what information I provide.&rdquo;"
            buttonText="Continue to Values &amp; Attention Output"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
