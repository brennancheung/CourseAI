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
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { AttentionMatrixWidget } from '@/components/widgets/AttentionMatrixWidget'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * The Problem Attention Solves
 *
 * First lesson in Module 4.2 (Attention & the Transformer).
 * Fourth lesson in Series 4 (LLMs & Transformers).
 *
 * Teaches how dot-product attention allows tokens to create context-dependent
 * representations by computing weighted averages where the weights are
 * determined by the input itself. Deliberately uses RAW embeddings (no Q/K/V)
 * to expose the limitation: one vector per token for both "seeking" and "offering."
 *
 * Core concepts at DEVELOPED:
 * - Context-dependent representations as a goal
 * - Weighted average as a mechanism
 * - Dot-product attention (data-dependent weight computation)
 *
 * Concepts at INTRODUCED:
 * - Dual-role limitation (one embedding for both seeking and offering)
 * - Attention matrix symmetry as a design flaw
 *
 * EXPLICITLY NOT COVERED:
 * - Q, K, V projections (not even named)
 * - Multi-head attention
 * - Scaled dot-product attention
 * - Causal masking
 *
 * Previous: Embeddings & Positional Encoding (module 4.1, lesson 3)
 * Next: Queries, Keys, and the Relevance Function (module 4.2, lesson 2)
 */

// ---------------------------------------------------------------------------
// Static data for worked example
// ---------------------------------------------------------------------------

const TINY_EMBEDDINGS = [
  { token: 'The', vec: [0.9, 0.1, -0.2] },
  { token: 'cat', vec: [-0.3, 0.8, 0.5] },
  { token: 'sat', vec: [0.1, 0.6, 0.4] },
  { token: 'here', vec: [0.4, -0.2, 0.7] },
]

// Precomputed dot products for the worked example
function computeTinyDotProducts() {
  const n = TINY_EMBEDDINGS.length
  const scores: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) {
      let dot = 0
      for (let d = 0; d < 3; d++) {
        dot += TINY_EMBEDDINGS[i].vec[d] * TINY_EMBEDDINGS[j].vec[d]
      }
      row.push(Math.round(dot * 100) / 100)
    }
    scores.push(row)
  }
  return scores
}

function softmaxRow(values: number[]): number[] {
  const max = Math.max(...values)
  const exps = values.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => Math.round((e / sum) * 1000) / 1000)
}

const TINY_SCORES = computeTinyDotProducts()
const TINY_WEIGHTS = TINY_SCORES.map((row) => softmaxRow(row))

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function TheProblemAttentionSolvesLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Problem Attention Solves"
            description="How tokens can create context-dependent representations using only dot products and softmax&mdash;tools you already have."
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
            Understand how dot-product attention lets tokens create context-dependent
            representations by computing weighted averages where the input itself
            determines the weights&mdash;and feel the specific limitation that motivates
            the next lesson.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In Embeddings &amp; Positional Encoding, you built the complete input pipeline:
            text &rarr; tokens &rarr; IDs &rarr; embedding vectors + position. Now you&rsquo;ll
            see what happens <strong>next</strong>&mdash;how the model processes those vectors.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How tokens communicate via dot-product attention using raw embeddings',
              'Why static (context-free) embeddings are insufficient for language',
              'The mechanics: similarity scores, softmax normalization, weighted averaging',
              'The limitation: one vector per token for both "seeking" and "offering"',
              'Has notebook: compute raw attention from scratch in PyTorch',
              'NOT: Learned projection matrices that fix the limitation you’ll discover—that’s the next lesson’s answer',
              'NOT: multi-head attention, transformer architecture, or causal masking',
              'NOT: scaled dot-product attention (the scaling factor is tied to the next lesson)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Recap — Dot Product as Similarity
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap: Dot Product as Similarity"
            subtitle="The tool we&rsquo;ll use to measure relevance"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the embedding notebook, you used cosine similarity to compare token vectors.
              The dot product is closely related&mdash;it measures how much two vectors
              point in the same direction.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="a \cdot b = \sum_{i=1}^{d} a_i \cdot b_i" />
            </div>

            <p className="text-muted-foreground">
              When two vectors point the same way, their dot product is <strong>large and positive</strong>.
              When they&rsquo;re perpendicular, it&rsquo;s <strong>near zero</strong>.
              When they point in opposite directions, it&rsquo;s <strong>negative</strong>.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-xs text-muted-foreground/70 mb-2">Two 3-dimensional vectors:</p>
              <div className="space-y-1.5 font-mono text-sm">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <span className="text-sky-400">a = [2, 1, 0]</span>
                  <span className="text-muted-foreground/50">,</span>
                  <span className="text-violet-400">b = [1, 3, 0]</span>
                </div>
                <div className="text-muted-foreground">
                  a &middot; b = (2&times;1) + (1&times;3) + (0&times;0) = <span className="text-emerald-400 font-medium">5</span>
                  <span className="text-muted-foreground/50 ml-2">&mdash; positive, they point in similar directions</span>
                </div>
              </div>
              <div className="space-y-1.5 font-mono text-sm mt-3">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <span className="text-sky-400">a = [2, 1, 0]</span>
                  <span className="text-muted-foreground/50">,</span>
                  <span className="text-rose-400">c = [-1, 2, 0]</span>
                </div>
                <div className="text-muted-foreground">
                  a &middot; c = (2&times;-1) + (1&times;2) + (0&times;0) = <span className="text-amber-400 font-medium">0</span>
                  <span className="text-muted-foreground/50 ml-2">&mdash; zero, they&rsquo;re perpendicular</span>
                </div>
              </div>
            </div>

            {/* Geometric visual: two 2D vectors with angle */}
            <div className="flex justify-center gap-6 flex-wrap py-2">
              {/* Similar direction */}
              <div className="flex flex-col items-center gap-1">
                <svg width="120" height="100" viewBox="0 0 120 100" className="overflow-visible">
                  {/* Grid hint */}
                  <line x1="10" y1="80" x2="110" y2="80" stroke="currentColor" opacity={0.1} />
                  <line x1="20" y1="10" x2="20" y2="90" stroke="currentColor" opacity={0.1} />
                  {/* Vector a */}
                  <line x1="20" y1="80" x2="100" y2="25" stroke="#38bdf8" strokeWidth={2} />
                  <polygon points="100,25 90,30 93,38" fill="#38bdf8" />
                  <text x="102" y="22" fill="#38bdf8" fontSize="11" fontFamily="monospace">a</text>
                  {/* Vector b */}
                  <line x1="20" y1="80" x2="105" y2="40" stroke="#a78bfa" strokeWidth={2} />
                  <polygon points="105,40 94,40 96,48" fill="#a78bfa" />
                  <text x="107" y="38" fill="#a78bfa" fontSize="11" fontFamily="monospace">b</text>
                  {/* Angle arc */}
                  <path d="M 45,62 A 30,30 0 0,1 50,56" fill="none" stroke="#6ee7b7" strokeWidth={1.5} />
                  <text x="52" y="62" fill="#6ee7b7" fontSize="9" fontFamily="monospace">&theta;</text>
                </svg>
                <span className="text-xs text-emerald-400 font-medium">similar direction</span>
                <span className="text-xs text-muted-foreground/60">large dot product</span>
              </div>
              {/* Perpendicular */}
              <div className="flex flex-col items-center gap-1">
                <svg width="120" height="100" viewBox="0 0 120 100" className="overflow-visible">
                  <line x1="10" y1="80" x2="110" y2="80" stroke="currentColor" opacity={0.1} />
                  <line x1="20" y1="10" x2="20" y2="90" stroke="currentColor" opacity={0.1} />
                  {/* Vector a (horizontal) */}
                  <line x1="20" y1="80" x2="100" y2="80" stroke="#38bdf8" strokeWidth={2} />
                  <polygon points="100,80 92,75 92,85" fill="#38bdf8" />
                  <text x="102" y="78" fill="#38bdf8" fontSize="11" fontFamily="monospace">a</text>
                  {/* Vector b (vertical) */}
                  <line x1="20" y1="80" x2="20" y2="20" stroke="#a78bfa" strokeWidth={2} />
                  <polygon points="20,20 15,28 25,28" fill="#a78bfa" />
                  <text x="25" y="18" fill="#a78bfa" fontSize="11" fontFamily="monospace">b</text>
                  {/* Right angle indicator */}
                  <polyline points="28,80 28,72 20,72" fill="none" stroke="#6ee7b7" strokeWidth={1.5} />
                </svg>
                <span className="text-xs text-amber-400 font-medium">perpendicular</span>
                <span className="text-xs text-muted-foreground/60">zero dot product</span>
              </div>
              {/* Opposite */}
              <div className="flex flex-col items-center gap-1">
                <svg width="120" height="100" viewBox="0 0 120 100" className="overflow-visible">
                  <line x1="10" y1="50" x2="110" y2="50" stroke="currentColor" opacity={0.1} />
                  <line x1="60" y1="10" x2="60" y2="90" stroke="currentColor" opacity={0.1} />
                  {/* Vector a (right) */}
                  <line x1="60" y1="50" x2="105" y2="50" stroke="#38bdf8" strokeWidth={2} />
                  <polygon points="105,50 97,45 97,55" fill="#38bdf8" />
                  <text x="107" y="48" fill="#38bdf8" fontSize="11" fontFamily="monospace">a</text>
                  {/* Vector b (left) */}
                  <line x1="60" y1="50" x2="15" y2="50" stroke="#a78bfa" strokeWidth={2} />
                  <polygon points="15,50 23,45 23,55" fill="#a78bfa" />
                  <text x="5" y="48" fill="#a78bfa" fontSize="11" fontFamily="monospace">b</text>
                </svg>
                <span className="text-xs text-rose-400 font-medium">opposite direction</span>
                <span className="text-xs text-muted-foreground/60">negative dot product</span>
              </div>
            </div>

            <p className="text-muted-foreground">
              This is the connection to cosine similarity you used before:{' '}
              <InlineMath math="a \cdot b = \|a\| \|b\| \cos\theta" />. The dot product
              is cosine similarity scaled by the magnitudes of both vectors. For our purposes,
              the key fact is: <strong>larger dot product = more similar vectors</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Idea, Different Packaging">
            Cosine similarity normalizes away magnitude. Raw dot product keeps it. Both
            measure &ldquo;how much do these vectors point the same way?&rdquo; Attention
            uses raw dot products because magnitude carries useful information.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Hook — The Polysemy Promise
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Polysemy Promise"
            subtitle="Delivering on a forward reference"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Embeddings &amp; Positional Encoding, you saw an amber warning box:
            </p>

            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
              <p className="text-sm text-amber-400 font-medium">
                &ldquo;The embedding gives each token one vector. &lsquo;Bank&rsquo; (river) and
                &lsquo;bank&rsquo; (money) get the same embedding. Context-dependent meaning
                comes later, from attention.&rdquo;
              </p>
            </div>

            <p className="text-muted-foreground">
              Time to deliver on that promise. Consider these two sentences:
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: '"The bank was steep and muddy"',
              color: 'cyan',
              items: [
                '"bank" = riverbank, terrain',
                'Context: steep, muddy',
                'Meaning: geographical feature',
              ],
            }}
            right={{
              title: '"The bank raised interest rates"',
              color: 'orange',
              items: [
                '"bank" = financial institution',
                'Context: raised, interest, rates',
                'Meaning: company, finance',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="One Vector, Two Meanings">
            The embedding table has <strong>one</strong> row for &ldquo;bank.&rdquo;
            Both sentences get the exact same vector. The model can&rsquo;t distinguish
            river-bank from money-bank based on the embedding alone.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <p className="text-muted-foreground">
              You&mdash;a human reader&mdash;effortlessly resolve the ambiguity. You read
              &ldquo;steep and muddy&rdquo; and know it&rsquo;s a riverbank. You read
              &ldquo;interest rates&rdquo; and know it&rsquo;s a financial institution.
              You use <strong>context</strong>.
            </p>
            <p className="text-muted-foreground">
              The embedding table can&rsquo;t do this. It maps each token to a fixed vector
              regardless of what&rsquo;s around it. We need a mechanism that creates
              <strong> context-dependent representations</strong>&mdash;where the effective
              meaning of &ldquo;bank&rdquo; changes based on the surrounding words.
            </p>
            <p className="text-muted-foreground">
              By the end of this lesson, you&rsquo;ll see how.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 4: From Bag of Words to Weighted Average
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="From Bag of Words to Weighted Average"
            subtitle="Three attempts at context"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Attempt 1: Average all embeddings
            </p>
            <p className="text-muted-foreground">
              The simplest way to create a &ldquo;context-aware&rdquo; representation:
              average all the embeddings in the sentence. Each token gets the same
              summary vector.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="The Bag-of-Words Problem (Again)" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                You already know this doesn&rsquo;t work. From Embeddings &amp; Positional Encoding:
              </p>
              <div className="px-3 py-2 bg-background/30 rounded font-mono text-xs space-y-1">
                <p>&ldquo;Dog bites man&rdquo; &rarr; avg(embed(&ldquo;dog&rdquo;), embed(&ldquo;bites&rdquo;), embed(&ldquo;man&rdquo;)) = <span className="text-rose-400">[0.31, 0.42, ...]</span></p>
                <p>&ldquo;Man bites dog&rdquo; &rarr; avg(embed(&ldquo;man&rdquo;), embed(&ldquo;bites&rdquo;), embed(&ldquo;dog&rdquo;)) = <span className="text-rose-400">[0.31, 0.42, ...]</span></p>
              </div>
              <p>
                Same tokens, same average. Every token gets the same summary.
                And &ldquo;dog bites man&rdquo; is identical to &ldquo;man bites dog.&rdquo;
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Problems">
            Uniform averaging has two flaws: (1) every token gets the <em>same</em> context
            summary, and (2) word order disappears. We need each token to get a
            <em> different</em> mix of the others.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Attempt 2: Weighted average
            </p>
            <p className="text-muted-foreground">
              Instead of giving every token equal weight, use a <strong>weighted average</strong>:
              each token in the context contributes a different amount.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70">
                A weighted average blends values using weights that sum to 1:
              </p>
              <div className="py-2 px-4">
                <BlockMath math="\text{output}_i = w_1 \cdot \text{embed}_1 + w_2 \cdot \text{embed}_2 + \ldots + w_n \cdot \text{embed}_n" />
              </div>
              <p className="text-xs text-muted-foreground/70">
                where <InlineMath math="w_1 + w_2 + \ldots + w_n = 1" /> and each <InlineMath math="w_j \geq 0" />.
              </p>
              <p className="text-xs text-muted-foreground/70">
                If <InlineMath math="w_3" /> is large and the others are small, the output is
                mostly <InlineMath math="\text{embed}_3" />&mdash;the blend is biased toward
                that token.
              </p>
            </div>

            <p className="text-muted-foreground">
              This is better. If each token gets <em>different</em> weights, each
              token gets a different context summary. But there&rsquo;s a critical question:
            </p>

            <p className="text-muted-foreground font-medium text-lg text-center py-2">
              Who decides the weights?
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Not a person&mdash;the model runs on millions of sentences automatically.
              Not fixed parameters learned once during training&mdash;the right weights
              for &ldquo;The bank was steep&rdquo; are completely different from the right
              weights for &ldquo;The bank raised rates.&rdquo;
            </p>
            <p className="text-muted-foreground">
              The weights need to depend on <strong>the actual input</strong>. A new
              sentence should produce new weights, computed on the fly.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Paradigm Shift">
            In CNNs, the filter weights are <strong>fixed</strong>&mdash;the same filter
            for every input. Here, the weights must be <strong>data-dependent</strong>&mdash;freshly
            computed from each new sentence. This is the conceptual revolution of attention.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <p className="text-muted-foreground font-medium">
              Attempt 3: Let the input determine the weights
            </p>
            <p className="text-muted-foreground">
              Here&rsquo;s the breakthrough. You already have a tool that measures
              similarity between two vectors: the dot product. And you have a tool that
              converts arbitrary numbers into weights that sum to 1: softmax.
            </p>
            <p className="text-muted-foreground">
              Put them together: for each token, compute its dot product with every other
              token (similarity scores), apply softmax to get weights, then take a
              weighted average of all embeddings. The <strong>input itself determines
              what matters</strong>&mdash;no learned weights involved in choosing which tokens
              are relevant, just dot products and softmax operating mechanically on whatever vectors come in.
            </p>
            <p className="text-muted-foreground">
              This is dot-product attention.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 5: Dot-Product Attention — Worked Example
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Dot-Product Attention"
            subtitle="Step by step with tiny numbers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For now, we&rsquo;re working with raw token embeddings&mdash;no positional
              encoding. This isolates the attention mechanism itself. In the full transformer,
              positional encoding is included, but that doesn&rsquo;t change how the attention
              computation works.
            </p>
            <p className="text-muted-foreground">
              Let&rsquo;s trace through the entire computation with a tiny example:
              4 tokens, each with a 3-dimensional embedding. Small enough to verify every number by hand.
            </p>

            {/* Show the embeddings */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">Our 4 tokens and their embeddings:</p>
              <div className="space-y-1.5 font-mono text-sm">
                {TINY_EMBEDDINGS.map((item) => (
                  <div key={item.token} className="flex items-center gap-3 text-muted-foreground">
                    <span className="text-violet-400 w-12">&ldquo;{item.token}&rdquo;</span>
                    <span className="text-sky-400">[{item.vec.join(', ')}]</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Real Dimensions">
            Real transformers use 768 or more dimensions. We use 3 so you can see
            every multiplication. The mechanism is identical at any dimension.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Step 1: Compute dot products */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 1: Compute all pairwise dot products
            </p>
            <p className="text-muted-foreground">
              For each pair of tokens, compute the dot product of their embeddings.
              This gives us a 4&times;4 matrix of similarity scores:
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg overflow-x-auto">
              <table className="text-sm font-mono w-full">
                <thead>
                  <tr className="text-muted-foreground/60 text-xs">
                    <th className="text-left pr-3 pb-2">XX&#x1D40;</th>
                    {TINY_EMBEDDINGS.map((item) => (
                      <th key={item.token} className="text-center px-2 pb-2 text-violet-400/70">{item.token}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {TINY_EMBEDDINGS.map((rowItem, i) => (
                    <tr key={rowItem.token}>
                      <td className="text-violet-400/70 pr-3 py-1">{rowItem.token}</td>
                      {TINY_SCORES[i].map((score, j) => (
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
              Notice: the diagonal (each token&rsquo;s dot product with itself) tends to be
              high. And the matrix is <strong>symmetric</strong>&mdash;score(cat, sat) = score(sat, cat).
              That&rsquo;s because <InlineMath math="a \cdot b = b \cdot a" /> always.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="XX&#x1D40;">
            This 4&times;4 matrix is the result of multiplying the embedding matrix <InlineMath math="X" /> (shape
            4&times;3) by its transpose <InlineMath math="X^T" /> (shape 3&times;4). One matrix
            multiplication gives you all pairwise dot products at once.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Step 2: Softmax */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 2: Apply softmax to each row
            </p>
            <p className="text-muted-foreground">
              Raw dot products are arbitrary numbers. We need weights that are non-negative
              and sum to 1&mdash;a probability distribution. Softmax does exactly this,
              applied to each row independently:
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg overflow-x-auto">
              <table className="text-sm font-mono w-full">
                <thead>
                  <tr className="text-muted-foreground/60 text-xs">
                    <th className="text-left pr-3 pb-2">softmax(XX&#x1D40;)</th>
                    {TINY_EMBEDDINGS.map((item) => (
                      <th key={item.token} className="text-center px-2 pb-2 text-violet-400/70">{item.token}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {TINY_EMBEDDINGS.map((rowItem, i) => (
                    <tr key={rowItem.token}>
                      <td className="text-violet-400/70 pr-3 py-1">{rowItem.token}</td>
                      {TINY_WEIGHTS[i].map((weight, j) => (
                        <td
                          key={j}
                          className={`text-center px-2 py-1 ${
                            weight >= 0.3 ? 'text-emerald-400 font-medium' : 'text-muted-foreground'
                          }`}
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
              Each row now sums to 1. These are the <strong>attention weights</strong>. The
              &ldquo;cat&rdquo; row tells you how much &ldquo;cat&rdquo; attends to each other
              token: it gives the most weight to tokens whose embeddings are most similar to its own.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Softmax">
            This is the same softmax from temperature sampling and classification.
            Higher input &rarr; higher weight. It just normalizes a row of scores into
            a probability distribution.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Step 3: Weighted average */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Step 3: Compute weighted average for each token
            </p>
            <p className="text-muted-foreground">
              For each token, multiply every embedding by that token&rsquo;s attention weight
              for it, then sum. The result is a new vector&mdash;a <strong>context-dependent
              representation</strong>.
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70">
                For &ldquo;cat&rdquo; (row 2), using its attention weights:
              </p>
              <div className="font-mono text-sm text-muted-foreground space-y-1">
                <p>
                  output<sub>cat</sub> = {TINY_WEIGHTS[1][0].toFixed(3)} &times; embed(&ldquo;The&rdquo;) + {TINY_WEIGHTS[1][1].toFixed(3)} &times; embed(&ldquo;cat&rdquo;) + {TINY_WEIGHTS[1][2].toFixed(3)} &times; embed(&ldquo;sat&rdquo;) + {TINY_WEIGHTS[1][3].toFixed(3)} &times; embed(&ldquo;here&rdquo;)
                </p>
              </div>
              <p className="text-xs text-muted-foreground/70">
                The result is a blend of all embeddings, biased toward the tokens that &ldquo;cat&rdquo;
                finds most relevant (highest dot-product similarity).
              </p>
            </div>

            <p className="text-muted-foreground">
              Crucially, <strong>each token gets a different weighted average</strong>.
              &ldquo;The&rdquo; emphasizes different tokens than &ldquo;cat&rdquo; does, because
              their embeddings produce different dot products and therefore different weights.
              Every token now has a context-dependent representation.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Formula build-up */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              The full formula
            </p>
            <p className="text-muted-foreground">
              All three steps collapse into one expression. If <InlineMath math="X" /> is the
              matrix of embeddings (each row is one token&rsquo;s embedding):
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="space-y-1 text-sm text-muted-foreground">
                <p>1. Similarity scores: <InlineMath math="S = XX^T" /></p>
                <p>2. Attention weights: <InlineMath math="W = \text{softmax}(S)" /> (row-wise)</p>
                <p>3. Context-dependent output: <InlineMath math="\text{output} = W \cdot X" /></p>
              </div>
              <div className="pt-2 border-t border-border/50">
                <BlockMath math="\text{Attention}(X) = \text{softmax}(XX^T)\, X" />
              </div>
            </div>

            <p className="text-muted-foreground">
              Three matrix operations. That&rsquo;s it. The entire mechanism is dot products,
              softmax, and another matrix multiplication. No new math&mdash;just a new way of
              combining tools you already have.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Data-Dependent Weights">
            The weight matrix <InlineMath math="W" /> is <strong>not</strong> a learned parameter.
            It&rsquo;s freshly computed from every new input <InlineMath math="X" />.
            Different sentences produce different weight matrices. This is what makes
            attention fundamentally different from a fixed linear layer or convolution filter.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Locality misconception: attention is NOT like a convolution filter */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Notice something about that 4&times;4 score matrix: every token computes a score
              with <strong>every other token</strong>, regardless of distance. &ldquo;The&rdquo;
              (position 1) has a score with &ldquo;here&rdquo; (position 4) just as readily as
              with &ldquo;cat&rdquo; (position 2). There is no distance penalty, no locality
              window, no concept of &ldquo;nearby.&rdquo;
            </p>
            <p className="text-muted-foreground">
              Consider the sentence: &ldquo;The cat sat on the mat because <strong>it</strong> was
              soft.&rdquo; To understand what &ldquo;it&rdquo; refers to, the model needs to connect
              &ldquo;it&rdquo; to &ldquo;mat&rdquo;&mdash;6 positions away, skipping
              &ldquo;on,&rdquo; &ldquo;the,&rdquo; and &ldquo;because&rdquo; entirely.
              Attention handles this naturally: it computes dot products between &ldquo;it&rdquo;
              and <em>all</em> tokens in the sequence, so &ldquo;mat&rdquo; gets a score just like
              any adjacent token would.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Like Convolutions">
            If you&rsquo;re coming from CNNs, resist the intuition that &ldquo;nearby
            matters most.&rdquo; A convolution filter sees a fixed local window (3&times;3,
            5&times;5). Attention computes <strong>all-pairs</strong> scores in a single
            step&mdash;a token can attend just as strongly to a word 20 positions away as to
            its immediate neighbor. This global reach is a fundamental difference.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check — Predict the Attention Pattern
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Predict the Pattern"
            subtitle="Test your intuition before computing"
          />
          <GradientCard title="Prediction Exercise" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Consider the sentence: &ldquo;The cat sat on the mat.&rdquo;
              </p>
              <p>
                Before looking at the widget below, predict: which token will &ldquo;cat&rdquo;
                attend to most strongly? Will &ldquo;the&rdquo; (position 1) and &ldquo;the&rdquo;
                (position 5) have the same attention pattern?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    &ldquo;Cat&rdquo; will attend most strongly to tokens with similar embeddings.
                    &ldquo;Sat&rdquo; and &ldquo;mat&rdquo; are likely closer to &ldquo;cat&rdquo;
                    in embedding space than &ldquo;the&rdquo; or &ldquo;on.&rdquo;
                  </p>
                  <p>
                    Both instances of &ldquo;the&rdquo; have the <strong>exact same embedding</strong>,
                    so they produce the <strong>exact same attention weights</strong>. Raw
                    dot-product attention can&rsquo;t distinguish them&mdash;only positional
                    encoding (which we&rsquo;re ignoring for clarity) could.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Interactive Attention Heatmap Widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: Attention Heatmap"
            subtitle="See the attention weights for any sentence"
          />
          <p className="text-muted-foreground mb-4">
            Type a sentence and watch the attention matrix form. Each row shows how much
            one token &ldquo;attends to&rdquo; every other token. Brighter cells mean
            higher attention weight. Hover to see exact values and compare a cell to its
            mirror position.
          </p>
          <ExercisePanel
            title="Raw Dot-Product Attention Matrix"
            subtitle="Type a sentence to see its attention pattern"
          >
            <AttentionMatrixWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>&bull; Try &ldquo;The bank was steep&rdquo; vs &ldquo;The bank was closed.&rdquo; Does &ldquo;bank&rdquo; attend differently?</li>
              <li>&bull; Try a sentence with repeated words. Do both instances attend the same way?</li>
              <li>&bull; Try a very short sentence (2-3 words). What happens?</li>
              <li>&bull; <strong>Key question:</strong> Is the matrix symmetric? Hover a cell and look at the dashed outline showing the mirror cell. Are the raw scores the same?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: The Dual-Role Limitation
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Crack in the Foundation"
            subtitle="One vector, two incompatible jobs"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If you explored the widget, you may have noticed something: the raw dot-product
              score matrix is <strong>symmetric</strong>. Token A&rsquo;s score for Token B
              equals Token B&rsquo;s score for Token A. Always.
            </p>
            <p className="text-muted-foreground">
              This isn&rsquo;t a coincidence. It&rsquo;s a mathematical certainty:{' '}
              <InlineMath math="a \cdot b = b \cdot a" />. The dot product is commutative.
              Softmax preserves the relative ordering within a row, but across rows the scores
              themselves are perfectly mirrored.
            </p>
            <p className="text-muted-foreground">
              Why is this a problem?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Forced Symmetry">
            Softmax is applied <em>per row</em>, so the final <em>weights</em> aren&rsquo;t
            perfectly symmetric (different rows can have different denominators). But the
            raw <em>scores</em>&mdash;which determine the ranking&mdash;are always symmetric.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="The Asymmetry Problem" color="amber">
            <div className="space-y-3 text-sm">
              <p className="font-medium">
                &ldquo;The cat chased the mouse.&rdquo;
              </p>
              <ul className="space-y-2 ml-2">
                <li>
                  &bull; For &ldquo;cat,&rdquo; the word &ldquo;chased&rdquo; is relevant because
                  it tells you <strong>what the cat did</strong>.
                </li>
                <li>
                  &bull; For &ldquo;chased,&rdquo; the word &ldquo;cat&rdquo; is relevant because
                  it tells you <strong>who did the chasing</strong>.
                </li>
              </ul>
              <p>
                Same pair of words, but relevant for <strong>different reasons</strong>.
                &ldquo;Cat&rdquo; is looking for an action. &ldquo;Chased&rdquo; is looking
                for an agent. Yet the raw dot product gives them the <em>same score</em> in both
                directions.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Different Reasons, Same Score">
            The relationship between &ldquo;cat&rdquo; and &ldquo;chased&rdquo; is
            inherently asymmetric: subject and verb have different roles. But a symmetric
            score can&rsquo;t express this difference.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The root cause: each token has <strong>one embedding vector</strong>. That single
              vector must simultaneously serve two incompatible roles:
            </p>

            <ComparisonRow
              left={{
                title: 'When This Token Is Looking',
                color: 'cyan',
                items: [
                  'The embedding determines what it searches for',
                  '"cat" looks for action words, context clues',
                  'The vector acts as a "what do I need?" signal',
                ],
              }}
              right={{
                title: 'When Other Tokens Look At It',
                color: 'orange',
                items: [
                  'The same embedding determines what it offers',
                  '"cat" advertises "I\'m an animal, a noun, a subject"',
                  'The vector acts as a "what do I have?" signal',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Think of it like a cocktail party. When you&rsquo;re listening, you want to find
              people talking about topics you care about. When someone scans the room, they
              see you as &ldquo;software engineer who likes cooking.&rdquo; Your &ldquo;what
              I&rsquo;m searching for&rdquo; and your &ldquo;what I advertise&rdquo; should be
              different&mdash;but with one embedding vector, they&rsquo;re forced to be the same.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Negative example */}
      <Row>
        <Row.Content>
          <GradientCard title="Where Raw Dot-Product Attention Breaks" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                Consider: &ldquo;The <strong>bank</strong> was steep.&rdquo;
              </p>
              <p>
                For understanding &ldquo;bank,&rdquo; the word &ldquo;steep&rdquo; is
                highly relevant&mdash;it tells you this is a <em>river</em>bank.
                But &ldquo;bank&rdquo; and &ldquo;steep&rdquo; are in different semantic
                clusters. Their embedding vectors aren&rsquo;t similar at all.
              </p>
              <p>
                Raw dot-product attention uses <strong>embedding similarity</strong> as a proxy
                for <strong>relevance</strong>. But similarity and relevance are not the same
                thing. &ldquo;Bank&rdquo; and &ldquo;river&rdquo; are similar; &ldquo;bank&rdquo;
                and &ldquo;steep&rdquo; are relevant. A financial context word like
                &ldquo;interest&rdquo; is more <em>similar</em> to &ldquo;bank&rdquo; than
                &ldquo;steep&rdquo; is&mdash;even when &ldquo;steep&rdquo; is the word that
                disambiguates meaning.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Similarity &ne; Relevance">
            Similarity asks: &ldquo;Are these words from the same cluster?&rdquo;
            Relevance asks: &ldquo;Does this word help me understand the other?&rdquo;
            These are related but different questions. Attention needs relevance, but
            raw dot products only measure similarity.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Check — Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Transfer Question"
            subtitle="Reason about a fix"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                If every token had <strong>two</strong> different vectors instead of one&mdash;one
                for &ldquo;what I&rsquo;m looking for&rdquo; and one for &ldquo;what I&rsquo;m
                advertising&rdquo;&mdash;would the attention matrix still be symmetric? Why or why not?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>No.</strong> If token A has a &ldquo;seeking&rdquo; vector{' '}
                    <InlineMath math="s_A" /> and token B has an &ldquo;advertising&rdquo;
                    vector <InlineMath math="a_B" />, the score for A looking at B is{' '}
                    <InlineMath math="s_A \cdot a_B" />. The reverse is{' '}
                    <InlineMath math="s_B \cdot a_A" />.
                  </p>
                  <p>
                    Since <InlineMath math="s_A \neq a_A" /> and <InlineMath math="s_B \neq a_B" /> in
                    general, these dot products are different. The symmetry breaks. Each
                    direction can now express a different reason for relevance.
                  </p>
                  <p className="text-muted-foreground/70">
                    This is exactly what the next lesson builds: two separate projection
                    matrices that create a &ldquo;seeking&rdquo; vector and an &ldquo;advertising&rdquo;
                    vector from each embedding.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Notebook Exercise
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It: Raw Dot-Product Attention"
            subtitle="The notebook exercise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now implement everything you&rsquo;ve seen. The notebook walks you through:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Compute <InlineMath math="XX^T" /> for a 4-token, 8-dim example</li>
              <li>Apply softmax row-wise to get attention weights</li>
              <li>Multiply weights by <InlineMath math="X" /> to get context-dependent representations</li>
              <li>Visualize the attention weight matrix as a heatmap and confirm it&rsquo;s symmetric</li>
              <li>Use pretrained GPT-2 embeddings to compute attention for real sentences (stretch)</li>
            </ul>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook and implement raw dot-product attention from scratch.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-2-1-the-problem-attention-solves.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes scaffolded exercises with solutions. You&rsquo;ll
                  implement the full attention computation and verify the symmetry property.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Three Matrix Operations">
            The core code is short: <code className="text-xs">scores = X @ X.T</code>,{' '}
            <code className="text-xs">weights = softmax(scores, dim=-1)</code>,{' '}
            <code className="text-xs">output = weights @ X</code>. Three lines. The point
            is understanding <em>why</em> each line is there.
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
                headline: 'Attention is a weighted average where the input determines the weights.',
                description:
                  'Dot products between embeddings measure similarity. Softmax converts those scores into weights that sum to 1. The weighted average produces a context-dependent representation for each token.',
              },
              {
                headline: 'The weights are data-dependent, not fixed parameters.',
                description:
                  'Unlike convolution filters (which are the same for every input), attention weights are freshly computed from each new sentence. Different inputs produce different weight matrices.',
              },
              {
                headline: 'The formula: Attention(X) = softmax(XXᵀ) X',
                description:
                  'Three matrix operations: compute pairwise similarity (XXᵀ), normalize rows (softmax), and take weighted averages (multiply by X).',
              },
              {
                headline: 'Raw dot-product attention has a fundamental limitation.',
                description:
                  'Each token has one embedding for both "what I’m looking for" and "what I’m advertising." The score matrix is symmetric, but real linguistic relationships are asymmetric.',
              },
              {
                headline: 'Similarity is not the same as relevance.',
                description:
                  'Dot products measure how similar two embeddings are. But relevance—"does this word help me understand the other?"—is a different question that requires more than raw embedding similarity.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Forward reference */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
            <p className="text-sm font-medium text-violet-400">
              What&rsquo;s next
            </p>
            <p className="text-sm text-muted-foreground">
              The next lesson introduces two separate projection matrices that let each token
              create a &ldquo;what I&rsquo;m looking for&rdquo; vector AND a &ldquo;what
              I&rsquo;m advertising&rdquo; vector. The symmetry problem disappears, and
              relevance replaces similarity as the basis for attention weights.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Residual stream seed */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
            <p className="text-xs font-medium text-muted-foreground/70">
              A seed for later
            </p>
            <p className="text-xs text-muted-foreground/60">
              In the full transformer, the attention output is <strong>added</strong> to the
              original embedding, not substituted for it. A token in an uninformative context
              keeps its original meaning. This &ldquo;residual stream&rdquo; idea&mdash;which
              you know from skip connections in ResNets&mdash;appears again in a few lessons.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/queries-and-keys"
            title="Queries, Keys, and the Relevance Function"
            description="Each token gets one vector that must serve as both &ldquo;what I&rsquo;m looking for&rdquo; and &ldquo;what I have to offer.&rdquo; Next: giving each token two separate vectors so the asymmetry problem disappears."
            buttonText="Continue to Queries &amp; Keys"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
