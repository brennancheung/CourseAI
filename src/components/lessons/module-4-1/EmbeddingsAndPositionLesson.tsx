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
  ModuleCompleteBlock,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { EmbeddingSpaceExplorer } from '@/components/widgets/EmbeddingSpaceExplorer'
import { PositionalEncodingHeatmap } from '@/components/widgets/PositionalEncodingHeatmap'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Embeddings & Positional Encoding
 *
 * Third and final lesson in Module 4.1 (Language Modeling Fundamentals).
 * Teaches how integer token IDs become rich vector representations via
 * learned embedding lookup, and how positional encoding injects sequence
 * order information that the model would otherwise lack.
 *
 * Core concepts at DEVELOPED:
 * - Token embeddings (learned lookup table, nn.Embedding)
 * - Sinusoidal positional encoding
 * - Embedding + position addition
 *
 * Concepts at INTRODUCED:
 * - One-hot encoding (bridge concept)
 * - Learned positional encoding
 *
 * Concepts at MENTIONED:
 * - RoPE (Rotary Position Embeddings)
 * - Unembedding / output layer symmetry
 *
 * Previous: Tokenization (module 4.1, lesson 2)
 * Next: Attention (module 4.2, lesson 1)
 */

// ---------------------------------------------------------------------------
// Static data for one-hot matrix example
// ---------------------------------------------------------------------------

const ONE_HOT_MATRIX_ROWS = [
  { token: '"cat"', id: 0, oneHot: [1, 0, 0], embedding: [0.2, -0.5, 0.8, 0.1] },
  { token: '"dog"', id: 1, oneHot: [0, 1, 0], embedding: [-0.3, 0.7, 0.4, -0.6] },
  { token: '"the"', id: 2, oneHot: [0, 0, 1], embedding: [0.9, 0.1, -0.2, 0.5] },
]

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function EmbeddingsAndPositionLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Embeddings & Positional Encoding"
            description="How integer token IDs become rich vectors the model can compute with&mdash;and why the model needs to be told where each token is."
            category="Language Modeling"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 1: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how token IDs are transformed into dense vector representations
            via learned embedding lookup, and how positional encoding injects sequence
            order into a model that would otherwise treat tokens as a bag of words.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In What Is a Language Model?, you learned that the model predicts the next
            token. In Tokenization, you built the algorithm that converts text to
            integer IDs. This lesson answers: how do those integers become vectors?
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How integer token IDs become dense vectors (nn.Embedding)',
              'Why one-hot encoding fails and how embeddings fix it',
              'Why position matters: the bag-of-words problem',
              'Sinusoidal positional encoding (the original transformer approach)',
              'Learned positional encoding (the simpler, now more common approach)',
              'NOT: how the model uses these vectors internally (attention, Q/K/V)\u2014that\u2019s Module 4.2',
              'NOT: training embeddings from scratch on a dataset\u2014Module 4.3',
              'NOT: Word2Vec, GloVe, or standalone embedding methods\u2014different paradigm',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Brief Recap — The Pipeline So Far
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Pipeline So Far"
            subtitle="You have integers. Now what?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You can now produce a sequence of integers from text. Given &ldquo;The cat
              sat on the mat,&rdquo; BPE yields something like{' '}
              <code className="text-sm bg-muted/50 px-1.5 py-0.5 rounded">
                [464, 3797, 3332, 319, 262, 2603]
              </code>
              . Each integer is an index into a vocabulary of ~50,000 tokens.
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg">
              <div className="flex flex-wrap items-center gap-2 text-sm font-mono">
                <span className="text-muted-foreground/70">&ldquo;The cat sat&rdquo;</span>
                <span className="text-muted-foreground/50">&rarr;</span>
                <span className="text-violet-400">BPE</span>
                <span className="text-muted-foreground/50">&rarr;</span>
                <span className="text-sky-400">[464, 3797, 3332]</span>
                <span className="text-muted-foreground/50">&rarr;</span>
                <span className="text-orange-400">???</span>
                <span className="text-muted-foreground/50">&rarr;</span>
                <span className="text-muted-foreground/40">model</span>
              </div>
            </div>
            <p className="text-muted-foreground">
              But a neural network multiplies inputs by weight matrices and passes them
              through activation functions. You can&rsquo;t multiply an integer by a
              weight matrix in any meaningful way. So what goes in that <span className="text-orange-400 font-mono">???</span> box?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Steps">
            The full input pipeline: text &rarr; tokens &rarr; integer IDs &rarr; embedding
            vectors + positional encoding &rarr; the tensor the model processes.
            This lesson builds the last step.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Hook — Integers Aren't Meaningful
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Problem: Integers Aren&rsquo;t Meaningful"
            subtitle="Token IDs are arbitrary"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Token IDs for three words: <strong>&ldquo;cat&rdquo;</strong> is token 2364.{' '}
              <strong>&ldquo;dog&rdquo;</strong> is token 8976.{' '}
              <strong>&ldquo;the&rdquo;</strong> is token 464.
            </p>
            <p className="text-muted-foreground">
              Is &ldquo;cat&rdquo; closer to &ldquo;dog&rdquo; than to &ldquo;the&rdquo;?
              Not according to the integers. <InlineMath math="|2364 - 464| = 1900" />,
              but <InlineMath math="|2364 - 8976| = 6612" />. The integer says &ldquo;cat&rdquo;
              is <em>closer</em> to &ldquo;the&rdquo; than to &ldquo;dog.&rdquo;
            </p>
            <p className="text-muted-foreground">
              That&rsquo;s absurd. Token IDs are assigned by BPE merge order&mdash;they&rsquo;re
              arbitrary indices, not meaningful coordinates. The model needs a
              representation where <strong>similarity is meaningful</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="IDs Don&rsquo;t Carry Meaning">
            BPE assigns IDs based on merge frequency, not semantics.
            Token 464 (&ldquo;the&rdquo;) and token 465 (&ldquo;.&rdquo;) are adjacent
            numbers with nothing in common. The embedding layer&rsquo;s job is to assign
            meaning to arbitrary IDs.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: One-Hot Encoding — The Naive Approach
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One-Hot Encoding"
            subtitle="The naive approach"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The simplest way to turn an integer into a vector: create a
              <InlineMath math="V" />-dimensional vector (where <InlineMath math="V" /> is
              the vocabulary size), set all entries to 0, and put a 1 at the token&rsquo;s
              index.
            </p>

            {/* Concrete one-hot example */}
            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70">
                A tiny 5-word vocabulary: [&ldquo;apple&rdquo;, &ldquo;banana&rdquo;, &ldquo;cat&rdquo;, &ldquo;dog&rdquo;, &ldquo;the&rdquo;]
              </p>
              <div className="space-y-1.5 font-mono text-sm">
                <div className="flex items-center gap-3">
                  <span className="text-muted-foreground w-20">&ldquo;cat&rdquo; (ID 2):</span>
                  <span className="text-sky-400">[0, 0, <strong>1</strong>, 0, 0]</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-muted-foreground w-20">&ldquo;dog&rdquo; (ID 3):</span>
                  <span className="text-sky-400">[0, 0, 0, <strong>1</strong>, 0]</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-muted-foreground w-20">&ldquo;the&rdquo; (ID 4):</span>
                  <span className="text-sky-400">[0, 0, 0, 0, <strong>1</strong>]</span>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              Two problems with this approach:
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Problem 1: Waste */}
      <Row>
        <Row.Content>
          <GradientCard title="Problem 1: Enormous, Sparse Vectors" color="rose">
            <div className="space-y-2 text-sm">
              <p>
                For a real 50,000-token vocabulary, each token becomes a
                50,000-dimensional vector with exactly one nonzero entry. That&rsquo;s
                49,999 zeros per token. For a 100-token sequence, you&rsquo;d have a
                matrix of shape 100 &times; 50,000 with 99.998% zeros.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Problem 2: No similarity */}
      <Row>
        <Row.Content>
          <GradientCard title="Problem 2: Every Token Is Equally Distant" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                Every pair of one-hot vectors is at the same Euclidean distance:{' '}
                <InlineMath math="\sqrt{2}" />. &ldquo;Cat&rdquo; is exactly as far
                from &ldquo;dog&rdquo; as from &ldquo;the.&rdquo; One-hot encoding
                <strong> cannot represent similarity</strong>.
              </p>
              <div className="px-3 py-2 bg-background/30 rounded text-xs font-mono space-y-1">
                <p>distance(&ldquo;cat&rdquo;, &ldquo;dog&rdquo;) = <InlineMath math="\sqrt{0^2 + 0^2 + 1^2 + (-1)^2 + 0^2} = \sqrt{2}" /></p>
                <p>distance(&ldquo;cat&rdquo;, &ldquo;the&rdquo;) = <InlineMath math="\sqrt{0^2 + 0^2 + 1^2 + 0^2 + (-1)^2} = \sqrt{2}" /></p>
              </div>
              {/* 3D one-hot equidistance visual */}
              <div className="flex items-center justify-center py-2">
                <svg width="220" height="180" viewBox="0 0 220 180" className="overflow-visible text-foreground">
                  {/* Axes */}
                  <line x1="110" y1="140" x2="190" y2="110" stroke="currentColor" strokeWidth="1" opacity="0.2" />
                  <line x1="110" y1="140" x2="30" y2="110" stroke="currentColor" strokeWidth="1" opacity="0.2" />
                  <line x1="110" y1="140" x2="110" y2="30" stroke="currentColor" strokeWidth="1" opacity="0.2" />

                  {/* Distance lines between points (equilateral triangle) */}
                  <line x1="190" y1="110" x2="30" y2="110" stroke="currentColor" strokeWidth="1" strokeDasharray="4 3" opacity="0.3" />
                  <line x1="190" y1="110" x2="110" y2="30" stroke="currentColor" strokeWidth="1" strokeDasharray="4 3" opacity="0.3" />
                  <line x1="30" y1="110" x2="110" y2="30" stroke="currentColor" strokeWidth="1" strokeDasharray="4 3" opacity="0.3" />

                  {/* Distance labels */}
                  <text x="110" y="122" textAnchor="middle" fill="currentColor" fontSize="10" opacity="0.7">&radic;2</text>
                  <text x="158" y="64" textAnchor="middle" fill="currentColor" fontSize="10" opacity="0.7">&radic;2</text>
                  <text x="62" y="64" textAnchor="middle" fill="currentColor" fontSize="10" opacity="0.7">&radic;2</text>

                  {/* Points */}
                  <circle cx="190" cy="110" r="6" fill="#38bdf8" />
                  <circle cx="30" cy="110" r="6" fill="#38bdf8" />
                  <circle cx="110" cy="30" r="6" fill="#38bdf8" />

                  {/* Labels */}
                  <text x="196" y="104" fill="#38bdf8" fontSize="11" fontWeight="600">&ldquo;cat&rdquo;</text>
                  <text x="2" y="104" fill="#38bdf8" fontSize="11" fontWeight="600">&ldquo;dog&rdquo;</text>
                  <text x="118" y="26" fill="#38bdf8" fontSize="11" fontWeight="600">&ldquo;the&rdquo;</text>

                  {/* Coordinate labels */}
                  <text x="196" y="120" fill="currentColor" fontSize="9" opacity="0.5">(1,0,0)</text>
                  <text x="2" y="120" fill="currentColor" fontSize="9" opacity="0.5">(0,1,0)</text>
                  <text x="118" y="40" fill="currentColor" fontSize="9" opacity="0.5">(0,0,1)</text>

                  {/* Origin */}
                  <circle cx="110" cy="140" r="2" fill="currentColor" opacity="0.3" />
                  <text x="116" y="154" fill="currentColor" fontSize="9" opacity="0.4">origin</text>
                </svg>
              </div>
              <p className="text-center text-xs text-muted-foreground">
                Three tokens in 3D one-hot space: every pair is exactly <InlineMath math="\sqrt{2}" /> apart.
                No &ldquo;close&rdquo; or &ldquo;far&rdquo;&mdash;every pair is identical distance.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Deeper Issue">
            Dimensionality is one problem, but the real issue is that one-hot
            <strong> cannot encode similarity</strong>. Even a 10-word vocabulary would
            benefit from embeddings. The model needs &ldquo;cat&rdquo; and &ldquo;dog&rdquo;
            to be closer to each other than to &ldquo;the.&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* The saving grace: one-hot @ matrix = row selection */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Saving Grace"
            subtitle="One-hot times a matrix = row selection"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One-hot encoding does have one useful property. If you multiply a one-hot
              vector by a matrix, you get back exactly one row of that matrix. Watch:
            </p>

            {/* Worked example: 3 tokens, 4-dim embeddings */}
            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-4">
              <p className="text-xs text-muted-foreground/70">
                Vocabulary of 3 tokens, embedding dimension of 4. The embedding matrix <InlineMath math="W" /> has shape 3&times;4:
              </p>
              <div className="overflow-x-auto">
                <table className="text-sm font-mono">
                  <thead>
                    <tr className="text-muted-foreground/60 text-xs">
                      <th className="pr-4 text-left">Token</th>
                      <th className="pr-3 text-left">One-hot</th>
                      <th className="pr-2 text-left">&times;</th>
                      <th className="text-left">W</th>
                      <th className="pl-3 text-left">=</th>
                      <th className="pl-3 text-left">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ONE_HOT_MATRIX_ROWS.map((row) => (
                      <tr key={row.id} className="text-muted-foreground">
                        <td className="pr-4 text-violet-400">{row.token}</td>
                        <td className="pr-3 text-sky-400">[{row.oneHot.join(', ')}]</td>
                        <td className="pr-2">&times;</td>
                        <td>W</td>
                        <td className="pl-3">=</td>
                        <td className="pl-3 text-amber-400">[{row.embedding.join(', ')}]</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground/70">
                Each result is just the corresponding <strong>row</strong> of W.
                The one-hot vector [0, 1, 0] selects row 1. The matrix multiplication
                is a fancy way of saying &ldquo;look up row <em>i</em>.&rdquo;
              </p>
            </div>

            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <BlockMath math="\text{one\_hot}(i) \times W = W[i, :]" />
            </div>

            <p className="text-muted-foreground">
              This is the key insight. You don&rsquo;t need to actually create a 50,000-dimensional
              sparse vector and multiply it by a matrix. You can just <strong>look up the row
              directly</strong>. That&rsquo;s exactly what <code className="text-sm bg-muted/50 px-1 rounded">nn.Embedding</code> does.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Matrix Is Key">
            The one-hot vector is wasteful packaging. The matrix <InlineMath math="W" /> is what
            matters&mdash;each row IS the embedding for one token. <code className="text-xs">nn.Embedding</code> stores
            this matrix and lets you index into it directly.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Embeddings as Learned Lookup
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Embeddings: Learned Lookup"
            subtitle="A weight matrix indexed by token ID"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              PyTorch&rsquo;s <code className="text-sm bg-muted/50 px-1 rounded">nn.Embedding</code> is
              a learnable matrix with a convenient indexing shortcut. Create it with the
              vocabulary size and embedding dimension:
            </p>

            <CodeBlock
              code={`import torch
import torch.nn as nn

# Vocab size = 50,000 tokens, embedding dim = 768
embedding = nn.Embedding(50000, 768)

# Look up the embedding for token ID 2364 ("cat")
token_id = torch.tensor([2364])
vector = embedding(token_id)     # shape: [1, 768]

# This is identical to indexing the weight matrix directly:
same_vector = embedding.weight[2364]  # shape: [768]`}
              language="python"
              filename="embedding_basics.py"
            />

            <p className="text-muted-foreground">
              Think of it like a dictionary where the definitions are learned, not
              written by a human. You start with a dictionary where every word&rsquo;s
              definition is random noise. During training, the definitions get refined
              until words with similar usage have similar definitions.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Mathematically Equivalent">
            <code className="text-xs">nn.Embedding(V, d)</code> and
            <code className="text-xs"> nn.Linear(V, d)</code> with one-hot input
            produce the same output. Embedding is just the efficient version&mdash;it
            skips creating the sparse vector and goes straight to the row.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Learned, not fixed */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Embeddings are learned, not fixed
            </p>
            <p className="text-muted-foreground">
              This is where embeddings differ from tokenization. Tokenization is a
              fixed, deterministic preprocessing step&mdash;the same input always
              produces the same token IDs. Embeddings are <strong>part of the
              model</strong>. They have gradients. They update during training.
            </p>

            <CodeBlock
              code={`# The embedding matrix IS a learnable parameter
print(embedding.weight.requires_grad)  # True

# It shows up in model.parameters()
for name, param in model.named_parameters():
    print(name, param.shape)
# "embedding.weight" torch.Size([50000, 768])`}
              language="python"
              filename="embedding_is_learnable.py"
            />

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground/70 font-medium">
                Before vs. after training:
              </p>
              <div className="space-y-2 text-sm font-mono">
                <div>
                  <p className="text-muted-foreground/60 text-xs mb-1">At initialization (random noise):</p>
                  <p className="text-muted-foreground">embed(&ldquo;the&rdquo;) = [0.42, -0.18, 0.73, ...]</p>
                  <p className="text-muted-foreground">embed(&ldquo;a&rdquo;) &nbsp; = [-0.55, 0.91, 0.02, ...]</p>
                  <p className="text-xs text-rose-400 mt-1">Not similar at all&mdash;nowhere near each other</p>
                </div>
                <div>
                  <p className="text-muted-foreground/60 text-xs mb-1">After training on billions of tokens:</p>
                  <p className="text-muted-foreground">embed(&ldquo;the&rdquo;) = [0.91, 0.12, -0.21, ...]</p>
                  <p className="text-muted-foreground">embed(&ldquo;a&rdquo;) &nbsp; = [0.89, 0.15, -0.19, ...]</p>
                  <p className="text-xs text-emerald-400 mt-1">Nearly identical&mdash;they appear in similar contexts</p>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              If embeddings were fixed preprocessing, this clustering couldn&rsquo;t emerge.
              The structure is <strong>learned</strong>&mdash;backpropagation pushes tokens
              that appear in similar contexts closer together, just like it adjusts any
              other weight in the model.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Preprocessing">
            Embeddings are the <strong>first layer</strong> of the model,
            not a preprocessing step. They have <code className="text-xs">requires_grad=True</code>.
            They learn. Tokenization is preprocessing. Embeddings are model parameters.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Polysemy warning */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
            <p className="text-sm font-medium text-amber-400">
              One embedding per token, regardless of context
            </p>
            <p className="text-sm text-muted-foreground">
              The embedding gives each token <strong>one</strong> vector. &ldquo;Bank&rdquo;
              (river) and &ldquo;bank&rdquo; (money) get the same embedding. Context-dependent
              meaning comes later, from attention (Module 4.2). The embedding is a
              starting-point representation, not the final one.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: Check 1 — Parameter Count
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Parameter Count"
            subtitle="How big is this lookup table?"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You have <code>nn.Embedding(50000, 768)</code>. How many learnable
                parameters does this layer have?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>50,000 &times; 768 = 38,400,000</strong>. That&rsquo;s 38.4 million
                    parameters&mdash;just for the embedding table.
                  </p>
                  <p>
                    Now the vocabulary size discussion from Tokenization makes concrete
                    sense: &ldquo;more entries = more parameters&rdquo; means each extra token
                    adds 768 new learnable values.
                  </p>
                  <p>
                    <strong>Follow-up:</strong> If you double the vocabulary size? Parameter
                    count doubles (76.8M). Halve the embedding dimension? Parameter count
                    halves (19.2M).
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Embedding Space Widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: Embedding Space"
            subtitle="See how similar tokens cluster"
          />
          <p className="text-muted-foreground mb-4">
            This visualization shows how trained embeddings tend to cluster. The positions
            are simplified to make cluster structure visible&mdash;in real models, clusters
            are messier but the pattern holds: semantically similar tokens group together.
            Hover to see nearest neighbors. Click cluster buttons to highlight groups.
          </p>
          <ExercisePanel
            title="Embedding Space Explorer"
            subtitle="Hover tokens, search, or select clusters"
          >
            <EmbeddingSpaceExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Explore">
            <ul className="space-y-2 text-sm">
              <li>&bull; Search for &ldquo;king&rdquo; and &ldquo;queen.&rdquo; How close are they?</li>
              <li>&bull; Click &ldquo;Numbers.&rdquo; Are they in order?</li>
              <li>&bull; Compare &ldquo;happy&rdquo; and &ldquo;sad&rdquo;&mdash;close or far?</li>
              <li>&bull; Search for &ldquo;man&rdquo; and &ldquo;woman.&rdquo; Now &ldquo;king&rdquo; and &ldquo;queen.&rdquo; Notice the direction?</li>
              <li>&bull; Click &ldquo;Function Words&rdquo; (the, a, is, was...). Where do they cluster?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <p className="text-muted-foreground">
              This clustering is what training produces. At initialization, every token is
              random noise&mdash;no clusters, no similarity. After training, the model
              has learned that &ldquo;cat&rdquo; and &ldquo;dog&rdquo; appear in similar
              contexts, so their embeddings end up nearby.
            </p>
            <p className="text-muted-foreground">
              Remember the &ldquo;parameters are learnable knobs&rdquo; mental model from
              gradient descent? Embedding vectors are exactly those knobs. Backpropagation
              pushes &ldquo;cat&rdquo; and &ldquo;dog&rdquo; closer because using
              similar vectors for both reduces the prediction loss.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: The Bag-of-Words Problem (Position Motivation)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Bag-of-Words Problem"
            subtitle="Embeddings alone can&rsquo;t see order"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We have a vector for each token. But we have a problem.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Same Tokens, Different Meaning" color="orange">
            <div className="space-y-3 text-sm">
              <p>
                &ldquo;Dog bites man&rdquo; and &ldquo;Man bites dog&rdquo; contain the
                exact same tokens with the exact same embeddings. Without any notion of
                order, the model sees <strong>identical inputs</strong> for both sentences.
              </p>
              <div className="px-3 py-2 bg-background/30 rounded font-mono text-xs space-y-1">
                <p>&ldquo;Dog bites man&rdquo; &rarr; {'{'}embed(&ldquo;dog&rdquo;), embed(&ldquo;bites&rdquo;), embed(&ldquo;man&rdquo;){'}'}</p>
                <p>&ldquo;Man bites dog&rdquo; &rarr; {'{'}embed(&ldquo;man&rdquo;), embed(&ldquo;bites&rdquo;), embed(&ldquo;dog&rdquo;){'}'}</p>
              </div>
              <p>
                As a set, these are identical. The embeddings are the same. The meaning
                is entirely determined by order. A set of embeddings, without position,
                is literally a <strong>bag of words</strong>.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Order IS Meaning">
            In language, word order often determines meaning entirely.
            &ldquo;Dog bites man&rdquo; is routine; &ldquo;man bites dog&rdquo; is
            headline news. If the model can&rsquo;t see order, it can&rsquo;t do language.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* CNN contrast */}
      <Row>
        <Row.Content>
          <div className="space-y-3 mb-4">
            <p className="text-muted-foreground">
              Recall how convolutions work from What Convolutions Compute: a small
              filter slides over spatial neighborhoods, comparing each pixel to its
              immediate neighbors. Because the filter moves one step at a time, the
              output at position (i,&thinsp;j) comes directly from the input at and
              around position (i,&thinsp;j). Position is baked into the architecture&mdash;nearby
              pixels are naturally compared because the filter physically covers them together.
            </p>
            <p className="text-muted-foreground">
              Transformers don&rsquo;t slide. They process all tokens in parallel with
              no inherent notion of order. There is no &ldquo;nearby&rdquo; by default.
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'How CNNs See Position',
              color: 'sky',
              items: [
                'Filter slides over spatial positions',
                'Output at position (i,j) comes from input at (i,j)',
                'Position is implicit in the architecture',
                'Spatial locality is a built-in assumption',
              ],
            }}
            right={{
              title: 'Embeddings Without Position',
              color: 'amber',
              items: [
                'All positions processed identically',
                'No sliding, no spatial structure',
                '"Dog bites man" = "Man bites dog"',
                'Order information is completely missing',
              ],
            }}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <p className="text-muted-foreground">
              If you care about order&mdash;and language <em>is</em> order&mdash;you
              have to <strong>inject position information explicitly</strong>.
              That&rsquo;s positional encoding.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Sinusoidal Positional Encoding
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Sinusoidal Positional Encoding"
            subtitle="The original transformer approach"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before seeing the formula, think about what a positional encoding needs to do:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
              <li>Each position gets a <strong>unique</strong> encoding</li>
              <li>Nearby positions should have <strong>similar</strong> encodings</li>
              <li>It should work for <strong>any sequence length</strong> (not just lengths seen during training)</li>
              <li>It should be <strong>deterministic</strong> (no learning needed)</li>
            </ol>
            <p className="text-muted-foreground">
              Sinusoidal waves at different frequencies satisfy all four. Low-frequency
              waves change slowly (nearby positions are similar), high-frequency waves
              change fast (distant positions differ), waves extend to any length, and
              the pattern is fixed.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)" />
              <BlockMath math="PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)" />
            </div>

            <p className="text-muted-foreground">
              Where <InlineMath math="pos" /> is the position in the sequence,{' '}
              <InlineMath math="i" /> is the dimension index, and <InlineMath math="d" /> is
              the embedding dimension. Even dimensions use sine, odd dimensions use cosine.
              The <InlineMath math="10000^{2i/d}" /> denominator creates waves at exponentially
              increasing wavelengths: the first dimensions oscillate rapidly (like a second hand),
              later dimensions oscillate slowly (like an hour hand).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Clock Analogy">
            Think of it like reading a clock: the second hand gives you fine-grained
            time, the minute hand gives coarser time, the hour hand gives you the
            broadest time. Combining all three uniquely identifies any moment. Positional
            encoding works the same way across its dimensions.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Heatmap widget */}
      <Row>
        <Row.Content>
          <ExercisePanel
            title="Positional Encoding Heatmap"
            subtitle="20 positions &times; 64 dimensions"
          >
            <PositionalEncodingHeatmap />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="What to Notice">
            <ul className="space-y-2 text-sm">
              <li>&bull; Left columns oscillate rapidly across rows&mdash;these are high-frequency waves (small denominator)</li>
              <li>&bull; Right columns change slowly&mdash;low-frequency waves (huge denominator)</li>
              <li>&bull; Hover over adjacent rows: their patterns are similar (nearby positions)</li>
              <li>&bull; Hover over distant rows: their patterns differ (distant positions)</li>
              <li>&bull; Each row is unique&mdash;no two positions have the same encoding</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Addition, not concatenation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Addition, not concatenation
            </p>
            <p className="text-muted-foreground">
              The positional encoding is <strong>added</strong> to the token embedding,
              not concatenated:
            </p>

            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <BlockMath math="\text{input}_i = \text{embedding}(\text{token}_i) + \text{PE}(i)" />
            </div>

            <p className="text-muted-foreground">
              Both the embedding and the positional encoding have the same dimension
              (<InlineMath math="d" />). The result is also <InlineMath math="d" />-dimensional.
              Position and meaning are blended into a single representation, not kept as
              separate features.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Why Not Concatenate?">
            Concatenation would double the dimension and require different downstream
            architecture. Addition keeps the dimension the same and lets the model
            learn to use both signals from a single representation.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: Learned Positional Encoding
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Learned Positional Encoding"
            subtitle="The simpler, now more common approach"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              There&rsquo;s a simpler alternative: just create another embedding
              table and <strong>learn</strong> the position vectors.
            </p>

            <CodeBlock
              code={`# Learned positional encoding: another embedding table
# max_seq_len = 2048, same dimension as token embeddings
pos_embedding = nn.Embedding(2048, 768)

# Position IDs are just 0, 1, 2, ...
positions = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]

# The input to the model:
input = token_embedding(token_ids) + pos_embedding(positions)`}
              language="python"
              filename="learned_positional_encoding.py"
            />

            <p className="text-muted-foreground">
              This is what GPT-2 and most modern models use. It&rsquo;s simpler&mdash;no
              sin/cos formula, just another lookup table that training optimizes.
            </p>
            <p className="text-muted-foreground">
              The tradeoff: learned positional encoding can&rsquo;t generalize to
              sequence lengths not seen during training. If <code className="text-sm bg-muted/50 px-1 rounded">max_seq_len</code> is 2048,
              there&rsquo;s no learned embedding for position 2049. Sinusoidal encoding
              can extrapolate because the formula works for any position. In practice,
              learned positional encoding works just as well or better for the sequence
              lengths models are trained on.
            </p>
            <p className="text-muted-foreground text-sm text-muted-foreground/70">
              Modern models like LLaMA use Rotary Position Embeddings (RoPE), which
              encode <em>relative</em> position between tokens rather than absolute
              position. We&rsquo;ll encounter RoPE in Module 4.3.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Embedding Tables">
            The model&rsquo;s input layer has <strong>two</strong> lookup
            tables: one for token identity (50,000 rows) and one for position
            (e.g., 2,048 rows). Both are learned. Their outputs are added together.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: Check 2 — Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Transfer Question"
            subtitle="Apply what you&rsquo;ve learned"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You&rsquo;re building a language model for DNA sequences (A, C, G, T&mdash;a
                4-token vocabulary). Would you use sinusoidal or learned positional encoding?
                Does it change if your training data has sequences of length 1,000 but you
                want to process sequences of length 5,000 at inference?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    With a 4-token vocabulary, embedding dimension might be small. For
                    length extrapolation (1,000 &rarr; 5,000), <strong>sinusoidal
                    can generalize</strong>&mdash;the formula works at any position.
                    Learned positional encoding can&rsquo;t&mdash;there is no learned
                    embedding for position 1,001+.
                  </p>
                  <p>
                    If you only process sequences up to 1,000, either works. The moment
                    you need longer sequences at inference than at training, sinusoidal
                    (or relative position methods like RoPE) has an advantage.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Notebook Exercise
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It: Embeddings & Position in PyTorch"
            subtitle="The notebook exercise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now implement everything you&rsquo;ve seen. The notebook walks you through:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Create <code className="text-sm bg-muted/50 px-1 rounded">nn.Embedding</code> and verify the lookup</li>
              <li>Prove the one-hot equivalence: <code className="text-sm bg-muted/50 px-1 rounded">one_hot @ embedding.weight</code> vs <code className="text-sm bg-muted/50 px-1 rounded">embedding(token_id)</code></li>
              <li>Compute cosine similarity between token embeddings</li>
              <li>Implement sinusoidal positional encoding from the formula</li>
              <li>Combine token embeddings + positional encoding into the model input</li>
              <li>Explore pretrained GPT-2 embeddings (stretch)</li>
            </ul>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook and implement embeddings and positional encoding from scratch.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-1-3-embeddings-and-position.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes scaffolded exercises with solutions. You&rsquo;ll
                  verify the one-hot equivalence, implement sinusoidal PE, and explore
                  pretrained embedding space.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Verify, Don&rsquo;t Memorize">
            The code is short&mdash;<code className="text-xs">nn.Embedding</code> is one line,
            sinusoidal PE is ~10 lines. The point isn&rsquo;t memorizing syntax. It&rsquo;s
            seeing that embeddings are just matrix indexing and positional encoding is just
            sin/cos at different frequencies.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 13: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Token IDs are arbitrary integers. Embeddings map them to dense vectors where similarity is meaningful.',
                description:
                  'nn.Embedding is a learnable weight matrix indexed by integer. Mathematically equivalent to one-hot \u00d7 matrix, but without creating the sparse vector.',
              },
              {
                headline: 'Without positional encoding, embeddings create a bag of words.',
                description:
                  'The model can\u2019t tell \u201cdog bites man\u201d from \u201cman bites dog.\u201d Position must be injected explicitly.',
              },
              {
                headline: 'Sinusoidal PE uses multi-frequency waves. Learned PE just learns a vector per position.',
                description:
                  'Sinusoidal can extrapolate to unseen lengths. Learned is simpler and often works as well for training-length sequences.',
              },
              {
                headline: 'Token embedding + positional encoding = the model\u2019s input.',
                description:
                  'The complete pipeline: text \u2192 BPE tokens \u2192 integer IDs \u2192 embedding vectors + position \u2192 the tensor the transformer processes.',
              },
              {
                headline: 'Embeddings are learned parameters, not preprocessing.',
                description:
                  'They start as random noise and training shapes them so similar tokens cluster. They\u2019re the first layer of the model, with gradients and everything.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            The input pipeline is now complete: text &rarr; tokens &rarr; IDs &rarr; embedding
            vectors + position &rarr; the tensor the model processes. Everything from
            here&mdash;attention, transformer blocks, the whole model&mdash;operates on
            these vectors.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Symmetry aside */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
            <p className="text-sm font-medium text-violet-400">
              Notice the symmetry
            </p>
            <p className="text-sm text-muted-foreground">
              The embedding table maps 50K token IDs to <InlineMath math="d" />-dimensional
              vectors. The output layer maps <InlineMath math="d" />-dimensional hidden
              states back to 50K logits. Everything interesting happens in
              that <InlineMath math="d" />-dimensional space.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 14: Module Complete + Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="4.1"
            title="Language Modeling Fundamentals"
            achievements={[
              'Language modeling as next-token prediction',
              'Subword tokenization and BPE (built from scratch)',
              'Token embeddings as learned lookup tables',
              'Positional encoding (sinusoidal and learned)',
              'The complete input pipeline: text \u2192 tokens \u2192 IDs \u2192 embeddings + position',
            ]}
            nextModule="4.2"
            nextTitle="Attention"
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/attention"
            title="Attention"
            description="These embedding vectors are what the model processes. But each token&rsquo;s vector is independent&mdash;the embedding for &ldquo;cat&rdquo; is the same whether the context is &ldquo;the cat sat&rdquo; or &ldquo;the cat died.&rdquo; Next: the mechanism that makes tokens context-aware."
            buttonText="Continue to Attention"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
