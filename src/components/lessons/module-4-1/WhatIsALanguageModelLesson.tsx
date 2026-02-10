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
import { TemperatureExplorer } from '@/components/widgets/TemperatureExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * What is a Language Model?
 *
 * First lesson in Module 4.1 (Language Modeling Fundamentals).
 * Teaches the student that a language model is a probability
 * distribution over tokens conditioned on context, and that
 * autoregressive generation produces text by repeatedly
 * sampling and appending.
 *
 * Core concepts at DEVELOPED:
 * - Language modeling as next-token prediction
 * - Autoregressive generation (sample, append, repeat)
 *
 * Concepts at INTRODUCED:
 * - Probability distribution over vocabulary
 * - Conditional probability P(next | context)
 * - Temperature and sampling
 *
 * Concepts at MENTIONED:
 * - Base model vs chat model distinction
 *
 * Previous: Transfer Learning Project (module 3.3, Series 3 capstone)
 * Next: Tokenization (module 4.1, lesson 2)
 */

// ---------------------------------------------------------------------------
// Inline helper: Static probability bar chart (Recharts)
// ---------------------------------------------------------------------------

const NEXT_TOKEN_PROBS = [
  { token: 'mat', probability: 0.35 },
  { token: 'floor', probability: 0.20 },
  { token: 'couch', probability: 0.15 },
  { token: 'bed', probability: 0.10 },
  { token: 'roof', probability: 0.05 },
  { token: 'other\u2026', probability: 0.15 },
]

type ProbTooltipPayload = {
  payload: { token: string; probability: number }
}

function ProbTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: ProbTooltipPayload[]
}) {
  if (!active || !payload || payload.length === 0) return null
  const data = payload[0].payload
  return (
    <div className="bg-popover border border-border rounded-md px-3 py-2 text-sm shadow-md">
      <p className="font-medium text-foreground">&ldquo;{data.token}&rdquo;</p>
      <p className="text-muted-foreground">
        Probability: {(data.probability * 100).toFixed(0)}%
      </p>
    </div>
  )
}

function probBarFill(probability: number): string {
  if (probability >= 0.30) return '#8b5cf6' // violet-500
  if (probability >= 0.15) return '#a78bfa' // violet-400
  if (probability >= 0.10) return '#c4b5fd' // violet-300
  return '#ddd6fe' // violet-200
}

// ---------------------------------------------------------------------------
// Inline helper: Autoregressive step visualization
// ---------------------------------------------------------------------------

type GenerationStep = {
  promptEnd: number // index of last word in the original prompt
  context: string
  sampled: string
  topProbs: { token: string; prob: string }[]
}

const GENERATION_WALKTHROUGH: GenerationStep[] = [
  {
    promptEnd: 5,
    context: 'The cat sat on the',
    sampled: 'mat',
    topProbs: [
      { token: 'mat', prob: '0.35' },
      { token: 'floor', prob: '0.20' },
      { token: 'couch', prob: '0.15' },
      { token: 'bed', prob: '0.10' },
    ],
  },
  {
    promptEnd: 5,
    context: 'The cat sat on the mat',
    sampled: 'and',
    topProbs: [
      { token: 'and', prob: '0.30' },
      { token: '.', prob: '0.25' },
      { token: ',', prob: '0.18' },
      { token: 'while', prob: '0.08' },
    ],
  },
  {
    promptEnd: 5,
    context: 'The cat sat on the mat and',
    sampled: 'purred',
    topProbs: [
      { token: 'purred', prob: '0.15' },
      { token: 'looked', prob: '0.14' },
      { token: 'fell', prob: '0.10' },
      { token: 'waited', prob: '0.09' },
    ],
  },
  {
    promptEnd: 5,
    context: 'The cat sat on the mat and purred',
    sampled: 'softly',
    topProbs: [
      { token: 'softly', prob: '0.22' },
      { token: '.', prob: '0.20' },
      { token: 'loudly', prob: '0.12' },
      { token: 'contentedly', prob: '0.08' },
    ],
  },
  {
    promptEnd: 5,
    context: 'The cat sat on the mat and purred softly',
    sampled: '.',
    topProbs: [
      { token: '.', prob: '0.45' },
      { token: ',', prob: '0.15' },
      { token: 'in', prob: '0.08' },
      { token: 'as', prob: '0.06' },
    ],
  },
]

function ContextWithHighlights({ context, promptEnd }: { context: string; promptEnd: number }) {
  const words = context.split(' ')
  const promptPart = words.slice(0, promptEnd).join(' ')
  const generatedPart = words.slice(promptEnd).join(' ')

  if (!generatedPart) {
    return <>{promptPart}</>
  }

  return (
    <>
      {promptPart}{' '}
      <span className="text-violet-400 font-semibold">{generatedPart}</span>
    </>
  )
}

function GenerationStepCard({ step, index }: { step: GenerationStep; index: number }) {
  return (
    <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
      <div className="flex items-center gap-2">
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-violet-500/20 text-violet-400 text-xs font-bold flex items-center justify-center">
          {index + 1}
        </span>
        <p className="text-sm text-muted-foreground">
          <span className="font-medium text-foreground">Context:</span>{' '}
          &ldquo;<ContextWithHighlights context={step.context} promptEnd={step.promptEnd} />{' '}
          <span className="text-violet-400 font-medium">___</span>&rdquo;
        </p>
      </div>
      <div className="ml-8 flex flex-wrap gap-2">
        {step.topProbs.map((tp) => (
          <span
            key={tp.token}
            className={`text-xs px-2 py-0.5 rounded-full ${
              tp.token === step.sampled
                ? 'bg-violet-500/20 text-violet-300 font-medium'
                : 'bg-muted text-muted-foreground'
            }`}
          >
            &ldquo;{tp.token}&rdquo; {tp.prob}
          </span>
        ))}
      </div>
      <p className="ml-8 text-xs text-muted-foreground">
        Sampled: <strong className="text-violet-400">&ldquo;{step.sampled}&rdquo;</strong>{' '}
        {'\u2192'} append to context
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function WhatIsALanguageModelLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="What is a Language Model?"
            description="A language model predicts the next token given context&mdash;the same supervised learning you already know, applied to text."
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
            Understand language modeling as next-token prediction&mdash;a probability
            distribution over a vocabulary, conditioned on context, that generates text
            through repeated sampling.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You have trained models to predict house prices from features and digit classes
            from pixels. You understand softmax, cross-entropy loss, and the training loop.
            This lesson reframes all of that for text.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'What a language model does: next-token prediction',
              'How autoregressive generation works: the sample-append-repeat loop',
              'What temperature does to the output distribution',
              'Why "just predict the next token" produces capable models',
              'NOT: how the model works internally (attention, transformers)\u2014Module 4.2',
              'NOT: tokenization or embeddings\u2014Lessons 2 and 3',
              'NOT: how models are trained at scale\u2014Module 4.3',
              'NOT: code or implementation\u2014this is conceptual only',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Hook — You Already Use a Language Model
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="You Already Use a Language Model"
            subtitle="Every day, in your pocket"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Open your phone and start typing a message. After &ldquo;I&apos;ll be there
              in&rdquo;, your keyboard suggests &ldquo;5&rdquo;, &ldquo;a&rdquo;,
              &ldquo;10&rdquo;. Those suggestions are a language model&mdash;a tiny one,
              running on your phone, predicting the next token based on what you&apos;ve
              typed so far.
            </p>

            {/* Phone autocomplete mockup */}
            <div className="mx-auto max-w-sm rounded-xl border border-border bg-muted/30 overflow-hidden">
              {/* Message bubble area */}
              <div className="px-4 pt-4 pb-3">
                <div className="inline-block px-4 py-2 rounded-2xl bg-violet-500/20 text-sm text-foreground">
                  I&apos;ll be there in
                  <span className="inline-block w-0.5 h-4 bg-violet-400 ml-0.5 align-middle animate-pulse" />
                </div>
              </div>
              {/* Suggestion bar */}
              <div className="flex items-center justify-center gap-2 px-4 py-2.5 bg-muted/60 border-t border-border">
                {['5', 'a', '10'].map((suggestion, i) => (
                  <span
                    key={suggestion}
                    className={`flex-1 text-center py-1.5 rounded-md text-sm font-medium ${
                      i === 0
                        ? 'bg-violet-500/20 text-violet-300'
                        : 'bg-muted text-muted-foreground'
                    } ${i < 2 ? 'border-r border-border' : ''}`}
                  >
                    {suggestion}
                  </span>
                ))}
              </div>
              <div className="h-6 bg-muted/40 border-t border-border" />
            </div>
            <p className="text-xs text-muted-foreground text-center">
              Your keyboard&apos;s suggestion bar is a language model&mdash;predicting the next token from context.
            </p>

            <p className="text-muted-foreground">
              ChatGPT and Claude do the same thing. Same fundamental task, much bigger model.
              The difference is scale and training data, not the core idea.
            </p>
            <p className="text-muted-foreground">
              But how can &ldquo;just predict the next word&rdquo; produce coherent
              paragraphs? How can it answer questions? How can it write code? Let&apos;s
              strip away the mystique and build a precise mental model.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Demystifying LLMs">
            The conceptual machinery you need to understand language models
            is the same machinery you&apos;ve been building for three series:
            supervised learning, softmax, cross-entropy, the training loop. The
            novelty is in the <em>composition</em>, not in new math.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Explain — Next-Token Prediction
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Next-Token Prediction"
            subtitle="The core task"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Quick grounding: probability distributions
            </p>
            <p className="text-muted-foreground">
              A probability distribution is a set of outcomes, each with a probability,
              where all probabilities sum to 1. You&apos;ve already worked with these&mdash;the
              softmax output of your MNIST classifier was a probability distribution over 10
              digit classes. &ldquo;Digit 7&rdquo; gets 0.92, &ldquo;Digit 1&rdquo; gets
              0.05, and so on.
            </p>
            <p className="text-muted-foreground">
              Now imagine scaling that from 10 classes to <strong>50,000 tokens</strong>. Each
              token in the vocabulary gets a probability. Same idea, much bigger distribution.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Token, Not Word">
            We say &ldquo;token&rdquo; instead of &ldquo;word&rdquo; deliberately. A token
            is the unit the model works with&mdash;it might be a word, a piece of a word,
            or a punctuation mark. The next lesson covers tokenization in depth.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              The task
            </p>
            <p className="text-muted-foreground">
              Given a sequence of tokens, predict the probability distribution over what
              comes next. Consider the context &ldquo;The cat sat on the&rdquo;:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-medium">
                Probability distribution for the next token:
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart
                  data={NEXT_TOKEN_PROBS}
                  margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
                  barSize={36}
                >
                  <XAxis
                    dataKey="token"
                    tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                    interval={0}
                    axisLine={{ stroke: 'var(--border)' }}
                    tickLine={{ stroke: 'var(--border)' }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    domain={[0, 0.4]}
                    axisLine={{ stroke: 'var(--border)' }}
                    tickLine={{ stroke: 'var(--border)' }}
                    width={45}
                  />
                  <Tooltip
                    content={<ProbTooltip />}
                    cursor={{ fill: 'var(--muted)', opacity: 0.3 }}
                  />
                  <Bar dataKey="probability" radius={[3, 3, 0, 0]}>
                    {NEXT_TOKEN_PROBS.map((entry) => (
                      <Cell
                        key={entry.token}
                        fill={probBarFill(entry.probability)}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-muted-foreground/70 mt-2">
                All probabilities sum to 1.0 across the entire vocabulary.
              </p>
            </div>
            <p className="text-muted-foreground">
              Notice that the model doesn&apos;t output a single answer. It outputs a
              spread of probabilities. &ldquo;mat&rdquo; is the most likely, but &ldquo;floor&rdquo;
              and &ldquo;couch&rdquo; are plausible too. The right prediction is this entire
              distribution, not just the top token.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="50,000 Classes">
            In MNIST, softmax produced 10 probabilities (one per digit). In language
            modeling, softmax produces one probability per token in the vocabulary&mdash;typically
            30,000 to 100,000 probabilities. Same operation, vastly bigger output.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Formal notation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              The formal notation
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg text-center">
              <BlockMath math="P(x_t \mid x_1, x_2, \ldots, x_{t-1})" />
            </div>
            <p className="text-muted-foreground">
              This reads: &ldquo;the probability of the next token{' '}
              <InlineMath math="x_t" />, given everything that came before it.&rdquo;
              You&apos;ve been doing this all along&mdash;predict the output given
              the input. This is just the notation for it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Conditional Probability">
            The vertical bar &ldquo;|&rdquo; means &ldquo;given.&rdquo;{' '}
            <InlineMath math="P(A \mid B)" /> = the probability of A, given that B
            is true. In every model you&apos;ve built, you&apos;ve been computing
            conditional probabilities&mdash;this is just the formal way to write it.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Connection to supervised learning */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              This is supervised learning
            </p>
            <ComparisonRow
              left={{
                title: 'What You Know',
                color: 'blue',
                items: [
                  'House prices: input = features, target = price',
                  'MNIST: input = pixels, target = digit',
                  'Loss = cross-entropy ("confidence penalty")',
                  'Training loop: forward \u2192 loss \u2192 backward \u2192 update',
                ],
              }}
              right={{
                title: 'Language Modeling',
                color: 'violet',
                items: [
                  'LM: input = preceding tokens, target = next token',
                  'Same: input \u2192 model \u2192 prediction \u2192 compare to target',
                  'Loss = cross-entropy (same "confidence penalty")',
                  'Same training loop: forward \u2192 loss \u2192 backward \u2192 update',
                ],
              }}
            />
            <p className="text-muted-foreground">
              The key insight: the training data generates its own labels. Every position in a
              text is simultaneously a training example. Given &ldquo;The cat sat on the mat&rdquo;,
              position 3 has input = &ldquo;The cat sat&rdquo; and target = &ldquo;on&rdquo;.
              Position 4 has input = &ldquo;The cat sat on&rdquo; and target = &ldquo;the&rdquo;.
              No human labeling required.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Self-Supervised Labels">
            In MNIST, someone had to label every image with its digit. In language modeling,
            the text itself provides the labels. Every token is both a feature (when it&apos;s
            in the context) and a label (when it&apos;s the next token). This is why language
            models can train on trillions of tokens of raw text.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Check 1 — Predict and Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Predict the Distribution"
            subtitle="Test your mental model"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Given the context &ldquo;I went to the&rdquo;&mdash;sketch what the
                probability distribution over next tokens might look like.{' '}
                <strong>Which tokens should have high probability? Which should have
                near-zero?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>High probability:</strong> &ldquo;store&rdquo;, &ldquo;park&rdquo;,
                    &ldquo;doctor&rdquo;, &ldquo;gym&rdquo;, &ldquo;beach&rdquo;&mdash;common
                    places you go to.
                  </p>
                  <p>
                    <strong>Near-zero:</strong> &ldquo;purple&rdquo;, &ldquo;seventeen&rdquo;,
                    &ldquo;the&rdquo;&mdash;grammatically impossible or nonsensical in context.
                    Notice &ldquo;the&rdquo; has near-zero probability because &ldquo;I went to
                    the the&rdquo; is ungrammatical. The model has to learn grammar implicitly
                    just to assign reasonable probabilities.
                  </p>
                  <p>
                    <strong>The realization:</strong> to predict well, the model must know which
                    words fit grammatically AND which are semantically plausible. Grammar and
                    meaning are compressed into the probability distribution.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 5: Explain — Autoregressive Generation
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Autoregressive Generation"
            subtitle="From predicting one token to generating a paragraph"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The model predicts one token. But how do we get a paragraph? The answer is a
              loop:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
              <li>Feed the context into the model</li>
              <li>Get a probability distribution over the vocabulary</li>
              <li>Sample a token from that distribution</li>
              <li>Append the sampled token to the context</li>
              <li>Go back to step 1 with the longer context</li>
            </ol>
            <p className="text-muted-foreground">
              This is called <strong>autoregressive generation</strong>&mdash;the model&apos;s
              outputs feed back as its inputs. Each generated token becomes part of the context
              for the next prediction.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Feedback Loop">
            &ldquo;Autoregressive&rdquo; means the output depends on previous outputs. The
            model generates one token at a time, and every choice it makes shapes what comes
            next. This is why the same model can produce different text each time&mdash;a
            different sample at step 3 sends the entire generation down a different path.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Autoregressive loop diagram */}
      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
            graph LR
              A["Context tokens"] --> B["Language<br/>Model"]
              B --> C["Probability<br/>distribution"]
              C --> D["Sample<br/>a token"]
              D --> E["Append to<br/>context"]
              E --> A
              style A fill:#1e1b4b,stroke:#8b5cf6,color:#e9d5ff
              style B fill:#1e1b4b,stroke:#8b5cf6,color:#e9d5ff
              style C fill:#1e1b4b,stroke:#a78bfa,color:#e9d5ff
              style D fill:#1e1b4b,stroke:#a78bfa,color:#e9d5ff
              style E fill:#1e1b4b,stroke:#a78bfa,color:#e9d5ff
          `} />
          <p className="text-xs text-muted-foreground text-center mt-2">
            The autoregressive generation loop: predict, sample, append, repeat.
          </p>
        </Row.Content>
      </Row>

      {/* Walkthrough */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Walkthrough: generating 5 tokens from &ldquo;The cat sat on the&rdquo;
            </p>
            <div className="space-y-3">
              {GENERATION_WALKTHROUGH.map((step, i) => (
                <GenerationStepCard key={i} step={step} index={i} />
              ))}
            </div>
            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <p className="text-sm text-muted-foreground">
                <strong className="text-foreground">Result:</strong> &ldquo;The cat sat on the
                mat and purred softly.&rdquo;&mdash;generated one token at a time, with each
                token sampled from a probability distribution.
              </p>
            </div>
            <p className="text-muted-foreground text-sm">
              A note on training vs generation: during <strong>training</strong>, the model
              sees the entire text at once (this is more efficient). During <strong>generation</strong>,
              it goes one token at a time. The training process is parallel; the generation
              process is sequential. This distinction matters for understanding speed, but
              not for understanding the core concept.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Sampling, Not Choosing">
            The model <em>samples</em> from the distribution&mdash;it doesn&apos;t always pick
            the highest-probability token. That&apos;s why the same prompt can produce
            different text each time. Picking only the top token is called &ldquo;greedy
            decoding&rdquo; and produces repetitive, boring text.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check 2 — Spot the Difference
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: CNNs vs Language Models"
            subtitle="Compare what you know"
          />
          <GradientCard title="Spot the Difference" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                How is autoregressive generation different from how a CNN processes
                an image?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    A CNN sees the <strong>entire image at once</strong> and produces one
                    output (a class label). An autoregressive language model generates{' '}
                    <strong>one token at a time</strong>, feeding each output back as input.
                  </p>
                  <p>
                    CNNs assume spatial locality&mdash;nearby pixels relate. Language models
                    assume sequential dependence&mdash;meaning builds left to right. Different
                    data structure, different generation pattern. Same idea from the MNIST
                    CNN Project: <em>architecture encodes assumptions about data.</em>
                  </p>
                  <p>
                    <strong>Key distinction:</strong> autoregressive is a <em>generation strategy</em>,
                    not something unique to text. You could generate an image pixel-by-pixel
                    autoregressively&mdash;and some models do exactly that (e.g., PixelRNN).
                    You could generate audio sample-by-sample, or music note-by-note. CNNs
                    don&apos;t use this strategy because images don&apos;t have a natural left-to-right
                    order. But anywhere data is sequential, autoregressive generation applies.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Explore — Temperature Widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Temperature: Controlling Randomness"
            subtitle="What the temperature slider actually does"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You&apos;ve probably seen a &ldquo;temperature&rdquo; slider in ChatGPT or
              an API. Some people call it &ldquo;creativity.&rdquo; That&apos;s misleading.
              Temperature does one thing: it reshapes the probability distribution before
              sampling. It doesn&apos;t make the model smarter or more creative.
            </p>
            <p className="text-muted-foreground">
              Remember softmax? It converts raw numbers (logits) into probabilities.
              Temperature divides the logits by <InlineMath math="T" /> before applying
              softmax:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg text-center">
              <BlockMath math="\text{softmax}\left(\frac{\text{logits}}{T}\right)" />
            </div>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Low T (e.g., 0.1):</strong> dividing by a small number makes
                the logits huge {'\u2192'} softmax becomes winner-take-all {'\u2192'} nearly
                deterministic
              </li>
              <li>
                <strong>T = 1.0:</strong> standard softmax, no modification
              </li>
              <li>
                <strong>High T (e.g., 3.0):</strong> dividing by a large number compresses
                the logits toward zero {'\u2192'} softmax becomes nearly uniform {'\u2192'} nearly
                random
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Temperature Does Not Change Knowledge">
            A common misconception: &ldquo;higher temperature makes the model more
            creative.&rdquo; The model&apos;s knowledge (its parameters) is fixed.
            Temperature only changes how the model <em>samples</em> from what it already
            knows. At T=2.0, you&apos;re just more likely to pick low-probability tokens.
            Whether that looks &ldquo;creative&rdquo; is a matter of luck.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ExercisePanel
            title="Temperature Explorer"
            subtitle="Adjust temperature and watch the probability distribution change"
          >
            <TemperatureExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>{'\u2022'} Set T=0.1. Which token dominates? What happens to the others?</li>
              <li>{'\u2022'} Set T=1.0. How spread out is the distribution?</li>
              <li>{'\u2022'} Set T=3.0. Could any token be sampled? Is this useful?</li>
              <li>{'\u2022'} Watch the entropy stat as you drag. Higher entropy = more randomness.</li>
              <li>{'\u2022'} Notice: at every temperature, the <em>ranking</em> of tokens stays the same. Temperature changes probabilities, not preferences.</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground text-sm">
              Beyond temperature, there are other sampling strategies like <strong>top-k</strong>{' '}
              (only consider the top K most likely tokens) and <strong>top-p / nucleus
              sampling</strong> (only consider tokens whose cumulative probability exceeds a
              threshold). These are additional knobs for controlling the tradeoff between
              coherence and variety. We won&apos;t develop them here&mdash;just know they exist.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Elaborate — Why Next-Token Prediction is Powerful
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why &ldquo;Just Predict the Next Token&rdquo; is Powerful"
            subtitle="The simplest task that requires learning everything"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              &ldquo;Predict the next token&rdquo; sounds too simple to produce intelligent
              behavior. But think about what good prediction requires.
            </p>
            <p className="text-muted-foreground">
              To predict the next token in &ldquo;The patient was diagnosed with a rare form
              of&rdquo;, the model needs knowledge of medicine, grammar, plausible diseases,
              and contextual coherence. To predict well in code, it needs to understand
              programming languages. To predict well in legal text, it needs legal reasoning
              patterns.
            </p>
            <p className="text-muted-foreground">
              The task is simple to <strong>state</strong> but requires compressing enormous
              amounts of knowledge into the model&apos;s parameters to <strong>do well</strong>.
              It&apos;s like &ldquo;predict the next move in chess&rdquo;&mdash;simple to
              describe, but doing it well requires understanding all of chess.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Universal Training Signal">
            Next-token prediction is a <em>universal</em> training signal. Any pattern in
            language&mdash;grammar, facts, reasoning, style, humor&mdash;is captured by it.
            You don&apos;t need separate training for &ldquo;learn grammar&rdquo; and
            &ldquo;learn facts.&rdquo; Learning to predict the next token forces the model
            to learn all of it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Base model vs chat model */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Base models vs chat models
            </p>
            <p className="text-muted-foreground">
              There&apos;s an important distinction to be aware of. Base language models are
              trained on raw text&mdash;books, websites, code. They learn to <em>continue</em>{' '}
              text, not answer questions. Give a base model &ldquo;What is the capital of
              France?&rdquo; and it might continue with &ldquo;What is the capital of Germany?
              What is the capital of...&rdquo; because it learned that questions come in lists.
            </p>
            <p className="text-muted-foreground">
              Chat models (ChatGPT, Claude) have additional finetuning&mdash;supervised
              finetuning on instruction-following data, plus reinforcement learning from human
              feedback (RLHF)&mdash;to make them conversational. The base model is what
              this lesson teaches. The finetuning is Module 4.4.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Pretrained + Finetuned">
            This is the same pattern from Transfer Learning: take a pretrained model
            (trained on a huge dataset) and finetune it for a specific task. The base
            language model is the pretrained backbone; chat behavior is the finetuned head.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* The model doesn't understand */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Does the model &ldquo;understand&rdquo;?
            </p>
            <p className="text-muted-foreground">
              A language model that&apos;s seen lots of geography text will put high
              probability on &ldquo;Paris&rdquo; after &ldquo;The capital of France
              is.&rdquo; But it doesn&apos;t &ldquo;know&rdquo; Paris is the capital of
              France the way you do. It learned text patterns. A small language model
              asked to continue &ldquo;2+3=&rdquo; might say &ldquo;5&rdquo; or
              &ldquo;23&rdquo;&mdash;it learned pattern-matching, not arithmetic.
            </p>
            <p className="text-muted-foreground">
              Whether this constitutes understanding is a philosophical question. For
              engineering purposes: <strong>it predicts tokens</strong>. That&apos;s the
              mental model that will serve you well.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Don&apos;t Anthropomorphize">
            It&apos;s tempting to say the model &ldquo;thinks&rdquo; or &ldquo;reasons.&rdquo;
            For this course, stick to the mechanistic description: the model assigns
            probabilities to tokens based on learned patterns. This framing keeps you
            honest about what the model actually does.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Check 3 — Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Wikipedia as Training Data"
            subtitle="Apply the language modeling frame"
          />
          <GradientCard title="Transfer Question" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You trained a model to classify cats vs dogs. Now imagine training a
                language model on all of Wikipedia.
              </p>
              <ol className="space-y-1 list-decimal list-inside">
                <li>What are the input-target pairs?</li>
                <li>What is the loss function?</li>
                <li>What does the model learn?</li>
              </ol>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    1. Every position in every article is a training example.
                    Input = preceding tokens. Target = actual next token. No human
                    labeling needed.
                  </p>
                  <p>
                    2. Cross-entropy loss&mdash;the same &ldquo;confidence penalty&rdquo;
                    you used when training classifiers. The model is penalized for putting low
                    probability on the actual next token.
                  </p>
                  <p>
                    3. Patterns of Wikipedia text: facts, sentence structure, topic
                    transitions, formatting conventions, grammar, and the relationships
                    between concepts. All compressed into the parameters to minimize
                    prediction error.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
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
                headline: 'A language model is a probability distribution over tokens, conditioned on context.',
                description:
                  'Given preceding tokens, the model outputs a probability for every token in the vocabulary. Same softmax you know from MNIST, scaled to 50,000+ classes.',
              },
              {
                headline: 'Autoregressive generation: sample, append, repeat.',
                description:
                  'The model predicts one token at a time. Each generated token becomes part of the context for the next prediction. This feedback loop produces text.',
              },
              {
                headline: 'Temperature reshapes the distribution, not the model\u2019s knowledge.',
                description:
                  'Low temperature = nearly deterministic (greedy). High temperature = nearly random. The model\u2019s parameters are unchanged\u2014only the sampling strategy changes.',
              },
              {
                headline: 'Next-token prediction is simple to state, but doing it well requires learning everything about language.',
                description:
                  'Grammar, facts, reasoning patterns, style\u2014all compressed into one objective. The simplicity of the task is a feature: it\u2019s a universal training signal.',
              },
              {
                headline: 'This is supervised learning with self-generated labels.',
                description:
                  'Input = context, target = next token, loss = cross-entropy. The same training loop you\u2019ve been running since the Foundations series.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            &ldquo;ML is function approximation&rdquo; becomes: <strong>a language model
            approximates P(next token | context)</strong>. Every concept you&apos;ve built&mdash;loss
            functions, gradients, parameters, training loops&mdash;applies directly. The domain
            changed from images to text. The machinery is the same.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/tokenization"
            title="Tokenization"
            description="We&rsquo;ve been saying &ldquo;token&rdquo; without really defining what a token is. Is it a word? A character? Something else? It turns out the answer matters enormously&mdash;and in the next lesson, you&rsquo;ll build a tokenizer from scratch and see exactly how text becomes the integer sequences your model works with."
            buttonText="Continue to Tokenization"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
