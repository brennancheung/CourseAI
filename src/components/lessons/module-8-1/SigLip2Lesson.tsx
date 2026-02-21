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
  ConceptBlock,
  GradientCard,
  ComparisonRow,
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * SigLIP 2 — Sigmoid Loss for Vision-Language Pretraining
 *
 * Lesson 1 in Module 8.1 (Vision & Vision-Language Models). Series 8 Special Topics.
 * Cognitive load: BUILD (targeted replacement within a known framework).
 *
 * Teaches how SigLIP replaces CLIP's softmax cross-entropy loss with sigmoid-based
 * per-pair binary classification, removing the batch-size dependency:
 * - Why CLIP's softmax loss creates batch-size dependency (the problem)
 * - How sigmoid loss scores each pair independently (the solution)
 * - SigLIP 2 training improvements: multi-stage, self-distillation, multi-resolution
 * - SigLIP as a building block for VLMs (PaliGemma)
 *
 * Core concepts:
 * - Sigmoid loss for contrastive learning: DEVELOPED
 * - Batch-size dependency in softmax contrastive loss: DEVELOPED
 * - SigLIP 2 training improvements: INTRODUCED
 * - Self-distillation: INTRODUCED
 * - SigLIP as downstream building block: INTRODUCED
 *
 * Previous: Z-Image (module 7.4, lesson 4) — end of Series 7
 * Next: SAM 3 (module 8.1, lesson 2)
 */

export function SigLip2Lesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="SigLIP 2"
            description="How replacing one line of code&mdash;softmax with sigmoid&mdash;removes CLIP's dependence on enormous batch sizes, and what SigLIP 2 adds on top."
            category="Vision & Vision-Language Models"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Context + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand why CLIP&rsquo;s softmax cross-entropy loss creates a structural
            dependence on large batch sizes, how SigLIP replaces it with sigmoid-based
            per-pair binary classification to make batch size a choice rather than a
            constraint, and what training methodology improvements SigLIP 2 adds on top.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            From <LessonLink slug="clip">CLIP</LessonLink>, you know contrastive learning, the dual-encoder
            architecture, the similarity matrix, and symmetric cross-entropy loss.
            From <LessonLink slug="activation-functions">Activation Functions</LessonLink>, you know the sigmoid
            function. From <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>, you know ViT and
            patchify. All of these carry over directly.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why CLIP\'s softmax loss creates batch-size dependency (the problem)',
              'How sigmoid loss removes this dependency by scoring each pair independently (the solution)',
              'SigLIP 2 training improvements: multi-stage training, self-distillation, multi-resolution',
              'SigLIP as a vision encoder for VLMs like PaliGemma',
              'NOT: implementing SigLIP from scratch',
              'NOT: the full PaliGemma or Gemini architecture',
              'NOT: other contrastive learning variants (MoCo, SimCLR, BYOL)',
              'NOT: mathematical proof of batch-size independence\u2014intuitive argument with concrete examples',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap -- CLIP Fundamentals */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Recap: CLIP in 60 Seconds"
            subtitle="Two encoders, one shared space, contrastive loss"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before we dive in: SigLIP uses the <strong>same dual-encoder
              architecture</strong>, the <strong>same shared embedding
              space</strong>, and the <strong>same cosine similarity
              computation</strong> as CLIP. The only change is the loss function.
              Everything you know about CLIP&rsquo;s architecture and
              applications carries over directly.
            </p>
            <p className="text-muted-foreground">
              In <LessonLink slug="clip">CLIP</LessonLink>, you learned the core setup: two separate
              encoders (one for images, one for text) trained together to produce
              embeddings in a shared space. Matching text-image pairs end up nearby;
              non-matching pairs end up far apart. The loss function creates the
              alignment, not the architecture. The{' '}
              <strong>conference analogy</strong>: N people enter a room with name
              tags (text) and photos (images). The training game: match each photo
              to the right name tag.
            </p>
            <p className="text-muted-foreground">
              The loss function is symmetric cross-entropy on the similarity
              matrix. Each row asks &ldquo;which text matches this image?&rdquo;
              Each column asks &ldquo;which image matches this text?&rdquo; The
              labels are always the diagonal.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="CLIP Core Mental Model">
            <ul className="space-y-1 text-sm">
              <li>&bull; Two encoders, one shared space</li>
              <li>&bull; The loss function creates the alignment</li>
              <li>&bull; Negatives come from the batch</li>
              <li>&bull; The shared space enables zero-shot transfer</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Gap resolution: softmax normalization creates coupling */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The detail worth examining more closely: the softmax denominator
            </p>
            <p className="text-muted-foreground">
              When you compute the probability that image <InlineMath math="i" /> matches
              text <InlineMath math="j" />, the softmax formula is:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="P(\text{text}_j \mid \text{image}_i) = \frac{e^{s_{ij} / \tau}}{\sum_{k=1}^{N} e^{s_{ik} / \tau}}" />
            </div>
            <p className="text-muted-foreground">
              That denominator sums over <strong>every item in the batch</strong>.
              With batch size 4, the denominator has 4 terms. With batch size
              32,768, it has 32,768 terms. Every similarity in the row affects the
              probability of every match. The gradient for any single pair depends
              on what else is in the batch. This is the normalization that creates{' '}
              <strong>coupling</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Softmax Creates Competition">
            You have deep experience with softmax creating competition between
            entries&mdash;attention weights, classification logits,
            temperature-controlled generation. This same competition property
            is exactly what creates CLIP&rsquo;s batch-size dependency.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook -- The Batch-Size Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Batch-Size Problem"
            subtitle="CLIP's batch size of 32,768 was not a choice"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="clip">CLIP</LessonLink>, you saw that CLIP trained with a batch size
              of 32,768 and learned that more negatives means a harder task and
              better features. But here is the question we did not examine: what
              happens if you train CLIP with batch size 256?
            </p>
            <p className="text-muted-foreground">
              The answer: it breaks. Not &ldquo;slightly worse&rdquo;&mdash;
              <strong>qualitatively different representations</strong>. The
              zero-shot classification that works beautifully at batch 32,768
              fails at batch 256. This is not a &ldquo;more data is better&rdquo;
              situation. It is a structural property of the loss function.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Tuning Issue">
            The batch-size dependency is a <strong>mathematical</strong> property
            of the softmax normalization, not a data quantity issue. Even with
            the same total number of negative examples, softmax behaves
            differently at different batch sizes because the denominator sums
            over all items in the batch.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Walk through the softmax denominator at two scales:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Batch = 4" color="emerald">
                <ul className="space-y-1">
                  <li>&bull; Softmax over 4 items</li>
                  <li>&bull; Each negative gets ~25% of probability mass</li>
                  <li>&bull; Gradient for any one negative: ~1/4 of the normalization</li>
                  <li>&bull; The softmax outputs are all close to 0.25</li>
                  <li>&bull; Weak, noisy gradient signal</li>
                </ul>
              </GradientCard>
              <GradientCard title="Batch = 32,768" color="violet">
                <ul className="space-y-1">
                  <li>&bull; Softmax over 32,768 items</li>
                  <li>&bull; Each negative gets ~0.003% of probability mass</li>
                  <li>&bull; Gradient for any one negative: diluted by ~8000&times;</li>
                  <li>&bull; Fine-grained probability distribution</li>
                  <li>&bull; Strong, informative gradient signal</li>
                </ul>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Think of it as a <strong>multiple-choice exam</strong>. CLIP&rsquo;s
              loss asks: &ldquo;which of these N items is the correct match?&rdquo;
              With 4 options, it is easy. With 32,768 options, the model must make
              fine-grained distinctions. But here is the problem: you{' '}
              <strong>need</strong> 32,768 options for the softmax to produce a
              meaningful probability distribution. With only 4 options, the
              gradient signal is too weak to learn discriminative features.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="The Question" color="orange">
            <p>
              CLIP&rsquo;s batch size is not a preference&mdash;it is a structural
              requirement of the softmax loss. <strong>Can we change the loss
              function so that batch size is a choice, not a constraint?</strong>
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 5: Explain Part 1 -- Sigmoid Loss */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Sigmoid Loss: The Core Idea"
            subtitle="From multiple choice to true/false"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The solution is beautifully simple: replace softmax (global
              normalization) with sigmoid (per-element scoring).
            </p>
            <p className="text-muted-foreground">
              Instead of asking <strong>&ldquo;which of these N items is the
              correct match?&rdquo;</strong> (multiple choice), ask{' '}
              <strong>&ldquo;is this specific pair a match?&rdquo;</strong>{' '}
              (true/false). Each cell in the N&times;N similarity matrix becomes
              an independent binary classification problem.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Multiple Choice vs True/False">
            CLIP&rsquo;s loss is a <strong>multiple-choice exam</strong>&mdash;pick
            the correct match from N options. Difficulty depends on how many
            options there are. SigLIP&rsquo;s loss is a{' '}
            <strong>true/false exam</strong>&mdash;for each pair, answer: match or
            not? Each question stands alone.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For each cell in the similarity matrix:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Matching pair</strong> (diagonal): label = +1. We want{' '}
                <InlineMath math="\sigma(s_{ij} \cdot t)" /> close to 1.
              </li>
              <li>
                <strong>Non-matching pair</strong> (off-diagonal): label = &minus;1. We want{' '}
                <InlineMath math="\sigma(-s_{ij} \cdot t)" /> close to 1
                (equivalently, <InlineMath math="\sigma(s_{ij} \cdot t)" /> close to 0).
              </li>
            </ul>
            <p className="text-muted-foreground">
              The loss for each cell is binary cross-entropy. In compact form:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\mathcal{L}_{ij} = -\log \sigma(y_{ij} \cdot s_{ij} \cdot t)" />
              <p className="text-sm text-muted-foreground text-center">
                where <InlineMath math="y_{ij} = +1" /> for matches, <InlineMath math="-1" /> for non-matches,
                and <InlineMath math="t" /> is the learned temperature
              </p>
            </div>
            <p className="text-muted-foreground">
              The total loss is the mean over all N&sup2; cells.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Sigmoid: Per-Element, No Normalization">
            You know the sigmoid function from{' '}
            <LessonLink slug="activation-functions">Activation Functions</LessonLink>: it maps a single real number
            to [0, 1] without referencing anything else. No denominator summing
            over a set. No competition between entries. This is exactly the
            property SigLIP exploits.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Why this removes batch-size dependency */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Why this removes batch-size dependency
            </p>
            <p className="text-muted-foreground">
              The loss for cell (i, j) depends <strong>only</strong> on{' '}
              <InlineMath math="s_{ij}" /> and <InlineMath math="y_{ij}" />. It does
              not reference <InlineMath math="s_{i1}, s_{i2}, \ldots, s_{iN}" />.
              No denominator summing over the batch. No normalization. Each
              pair&rsquo;s gradient is self-contained.
            </p>
            <p className="text-muted-foreground">
              Picture it spatially: softmax is a <strong>bucket containing
              exactly 1 unit of probability</strong>. With 4 items, each gets a
              substantial share. With 32,768 items, each gets a tiny drop. The
              total is fixed&mdash;adding more items dilutes every other item&rsquo;s
              share. Sigmoid is <strong>N independent gauges</strong>, each
              reading between 0 and 1 on its own. Adding more gauges does not
              change any existing gauge&rsquo;s reading. That is batch-size
              independence.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The &ldquo;Of Course&rdquo; Moment">
            CLIP&rsquo;s batch size of 32,768 was not a hyperparameter
            choice&mdash;it was a structural requirement of the softmax
            denominator. <em>Of course</em> a loss function where each pair is
            scored independently would remove this dependency. <em>Of course</em>{' '}
            that function is the sigmoid, which you already know maps a single
            value to [0, 1] without referencing anything else.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Visual: side-by-side 4x4 matrix comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Side-by-side: softmax vs sigmoid on the same similarity matrix
            </p>
            <div className="grid gap-6 md:grid-cols-2">
              {/* CLIP: softmax with arrows showing coupling */}
              <div className="rounded-lg bg-muted/30 p-4 space-y-3">
                <p className="text-sm font-medium text-foreground text-center">
                  CLIP: Softmax (row-wise normalization)
                </p>
                <svg viewBox="0 0 200 200" className="w-full max-w-[200px] mx-auto" aria-label="CLIP softmax: arrows connect all cells in each row, showing global normalization">
                  {/* Grid cells */}
                  {/* Row 0 */}
                  <rect x="20" y="20" width="35" height="35" rx="3" fill="#8b5cf6" opacity="0.8" />
                  <rect x="60" y="20" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="100" y="20" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="140" y="20" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  {/* Row 1 */}
                  <rect x="20" y="60" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="60" y="60" width="35" height="35" rx="3" fill="#8b5cf6" opacity="0.8" />
                  <rect x="100" y="60" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="140" y="60" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  {/* Row 2 */}
                  <rect x="20" y="100" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="60" y="100" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="100" y="100" width="35" height="35" rx="3" fill="#8b5cf6" opacity="0.8" />
                  <rect x="140" y="100" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  {/* Row 3 */}
                  <rect x="20" y="140" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="60" y="140" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="100" y="140" width="35" height="35" rx="3" fill="#334155" opacity="0.6" />
                  <rect x="140" y="140" width="35" height="35" rx="3" fill="#8b5cf6" opacity="0.8" />
                  {/* Arrows showing coupling in row 0 */}
                  <line x1="55" y1="37" x2="60" y2="37" stroke="#f59e0b" strokeWidth="1.5" />
                  <line x1="95" y1="37" x2="100" y2="37" stroke="#f59e0b" strokeWidth="1.5" />
                  <line x1="135" y1="37" x2="140" y2="37" stroke="#f59e0b" strokeWidth="1.5" />
                  {/* Arrows showing coupling in row 1 */}
                  <line x1="55" y1="77" x2="60" y2="77" stroke="#f59e0b" strokeWidth="1.5" />
                  <line x1="95" y1="77" x2="100" y2="77" stroke="#f59e0b" strokeWidth="1.5" />
                  <line x1="135" y1="77" x2="140" y2="77" stroke="#f59e0b" strokeWidth="1.5" />
                  {/* Label */}
                  <text x="100" y="193" fontSize="9" fill="#94a3b8" textAnchor="middle">
                    All cells in a row are coupled
                  </text>
                </svg>
                <p className="text-xs text-muted-foreground text-center">
                  Arrows show coupling: every cell&rsquo;s probability depends
                  on all other cells in the same row.
                </p>
              </div>
              {/* SigLIP: sigmoid with independent cells */}
              <div className="rounded-lg bg-muted/30 p-4 space-y-3">
                <p className="text-sm font-medium text-foreground text-center">
                  SigLIP: Sigmoid (per-cell independent)
                </p>
                <svg viewBox="0 0 200 200" className="w-full max-w-[200px] mx-auto" aria-label="SigLIP sigmoid: each cell is independently colored, no connections between cells">
                  {/* Grid cells — matches are green, non-matches are red, no arrows */}
                  {/* Row 0 */}
                  <rect x="20" y="20" width="35" height="35" rx="3" fill="#22c55e" opacity="0.8" />
                  <rect x="60" y="20" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="100" y="20" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="140" y="20" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  {/* Row 1 */}
                  <rect x="20" y="60" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="60" y="60" width="35" height="35" rx="3" fill="#22c55e" opacity="0.8" />
                  <rect x="100" y="60" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="140" y="60" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  {/* Row 2 */}
                  <rect x="20" y="100" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="60" y="100" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="100" y="100" width="35" height="35" rx="3" fill="#22c55e" opacity="0.8" />
                  <rect x="140" y="100" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  {/* Row 3 */}
                  <rect x="20" y="140" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="60" y="140" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="100" y="140" width="35" height="35" rx="3" fill="#ef4444" opacity="0.5" />
                  <rect x="140" y="140" width="35" height="35" rx="3" fill="#22c55e" opacity="0.8" />
                  {/* Labels inside cells */}
                  <text x="37" y="42" fontSize="7" fill="white" textAnchor="middle">+1</text>
                  <text x="77" y="42" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="117" y="42" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="157" y="42" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="37" y="82" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="77" y="82" fontSize="7" fill="white" textAnchor="middle">+1</text>
                  <text x="117" y="82" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="157" y="82" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="37" y="122" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="77" y="122" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="117" y="122" fontSize="7" fill="white" textAnchor="middle">+1</text>
                  <text x="157" y="122" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="37" y="162" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="77" y="162" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="117" y="162" fontSize="7" fill="white" textAnchor="middle">&minus;1</text>
                  <text x="157" y="162" fontSize="7" fill="white" textAnchor="middle">+1</text>
                  {/* Label */}
                  <text x="100" y="193" fontSize="9" fill="#94a3b8" textAnchor="middle">
                    Each cell scored independently
                  </text>
                </svg>
                <p className="text-xs text-muted-foreground text-center">
                  No arrows. Each cell is an independent binary classification:
                  green = match (+1), red = non-match (&minus;1).
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* Side-by-side code */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Side-by-side code: the only difference is the loss computation
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <CodeBlock
                  code={`# CLIP loss
logits = sim * temperature
loss = (
  F.cross_entropy(logits, targets)
  + F.cross_entropy(logits.T, targets)
) / 2`}
                  language="python"
                  filename="clip_loss.py"
                />
              </div>
              <div>
                <CodeBlock
                  code={`# SigLIP loss
logits = sim * temperature
labels = 2 * torch.eye(N) - 1
loss = -F.logsigmoid(
  labels * logits
).mean()`}
                  language="python"
                  filename="siglip_loss.py"
                />
              </div>
            </div>
            <p className="text-muted-foreground">
              Same encoders, same similarity matrix, same temperature. The CLIP
              loss uses <code className="text-sm">cross_entropy</code> (row and column
              normalization). The SigLIP loss uses{' '}
              <code className="text-sm">logsigmoid</code> (per-element). The{' '}
              <code className="text-sm">labels * logits</code> trick is elegant:
              multiplying by +1 keeps the logit positive for matches, multiplying
              by &minus;1 flips the sign for non-matches, so{' '}
              <code className="text-sm">logsigmoid</code> pushes both in the right
              direction.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="One Line of Code">
            The entire architectural difference between CLIP and SigLIP is the
            loss function. Same dual-encoder architecture. Same shared embedding
            space. Same cosine similarity computation. Same zero-shot
            classification procedure. One line changes.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Concrete example: trace sigmoid loss on specific cells */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Concrete example: tracing the loss for individual cells
            </p>
            <p className="text-muted-foreground">
              Take one matching pair from the 4&times;4 matrix: image 0 and text 0, with
              similarity <InlineMath math="s_{00} = 0.9" /> and temperature <InlineMath math="t = 10" />.
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground">
                <strong>Matching pair:</strong> label = +1
              </p>
              <BlockMath math="\mathcal{L}_{00} = -\log \sigma(+1 \times 0.9 \times 10) = -\log \sigma(9.0) \approx 0.00012" />
              <p className="text-sm text-muted-foreground">
                High similarity, low loss. This computation depends{' '}
                <strong>only</strong> on <InlineMath math="s_{00}" />.
              </p>
            </div>
            <p className="text-muted-foreground">
              Now a non-matching pair: image 0 and text 1, with similarity{' '}
              <InlineMath math="s_{01} = 0.2" />.
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground">
                <strong>Non-matching pair:</strong> label = &minus;1
              </p>
              <BlockMath math="\mathcal{L}_{01} = -\log \sigma(-1 \times 0.2 \times 10) = -\log \sigma(-2.0) \approx 2.13" />
              <p className="text-sm text-muted-foreground">
                The model gave a non-matching pair a positive similarity&mdash;the
                loss is high, pushing the model to reduce this similarity. This
                computation depends <strong>only</strong> on <InlineMath math="s_{01}" />.
              </p>
            </div>
            <p className="text-muted-foreground">
              Neither computation references any other cell. Check for yourself:
              does the loss for cell (0, 1) depend on cell (0, 3)? No. Does it
              depend on what else is in the batch? No. Each pair is self-contained.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Independence Check">
            For any cell (i, j), ask: &ldquo;does this loss value change if I
            add 1,000 more pairs to the batch?&rdquo; For sigmoid: no. For
            softmax: yes (the denominator gains 1,000 more terms). That is the
            entire difference.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Address misconception: "sigmoid doesn't use negatives" */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Addressing a likely misconception: &ldquo;does SigLIP waste the
              negative examples?&rdquo;
            </p>
            <p className="text-muted-foreground">
              In <LessonLink slug="clip">CLIP</LessonLink>, you learned that more negatives means a
              harder task and better features. Sigmoid loss scores each pair
              independently, which might sound like it &ldquo;wastes&rdquo; the
              negative information. It does not.
            </p>
            <p className="text-muted-foreground">
              For a batch of N image-text pairs, there are N matching pairs
              (diagonal) and N&sup2; &minus; N non-matching pairs (off-diagonal).
              Each non-matching pair is scored as a negative with label &minus;1.
              The difference is <strong>how</strong>: softmax normalizes across
              the row (competition), sigmoid scores each cell independently
              (binary classification). The negative signal is still there&mdash;it
              comes from the off-diagonal cells being pushed toward 0.
            </p>
            <ComparisonRow
              left={{
                title: 'CLIP: 2N Loss Terms',
                color: 'amber',
                items: [
                  'N row-wise cross-entropy terms',
                  'N column-wise cross-entropy terms',
                  'Each term normalizes across the batch',
                  'Total: 2N terms',
                ],
              }}
              right={{
                title: 'SigLIP: N\u00B2 Loss Terms',
                color: 'blue',
                items: [
                  'One binary classification per cell',
                  'N\u00B2 independent loss terms total',
                  'Each term is self-contained',
                  'More training signal per batch',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="More Negatives Still Help">
            SigLIP does NOT mean &ldquo;small batch training.&rdquo; Larger
            batches still provide more negative pairs and more training signal.
            The difference: with SigLIP, the improvement from larger batches is
            gradual and graceful, not a hard threshold. SigLIP at batch 256
            works fine. SigLIP at batch 32,768 is even better.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check 1 — Predict and Verify */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>A batch has 4 image-text pairs. The similarity matrix is:</p>
              <div className="overflow-x-auto">
                <table className="mx-auto text-xs border-collapse font-mono">
                  <thead>
                    <tr>
                      <th className="p-1" />
                      <th className="p-1 text-center">text₀</th>
                      <th className="p-1 text-center">text₁</th>
                      <th className="p-1 text-center">text₂</th>
                      <th className="p-1 text-center">text₃</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="p-1 text-right pr-2">img₀:</td>
                      <td className="p-1 text-center font-bold">0.9</td>
                      <td className="p-1 text-center">0.1</td>
                      <td className="p-1 text-center">0.2</td>
                      <td className="p-1 text-center">0.3</td>
                    </tr>
                    <tr>
                      <td className="p-1 text-right pr-2">img₁:</td>
                      <td className="p-1 text-center">0.2</td>
                      <td className="p-1 text-center font-bold">0.8</td>
                      <td className="p-1 text-center">0.1</td>
                      <td className="p-1 text-center">0.4</td>
                    </tr>
                    <tr>
                      <td className="p-1 text-right pr-2">img₂:</td>
                      <td className="p-1 text-center">0.1</td>
                      <td className="p-1 text-center">0.3</td>
                      <td className="p-1 text-center font-bold">0.7</td>
                      <td className="p-1 text-center">0.2</td>
                    </tr>
                    <tr>
                      <td className="p-1 text-right pr-2">img₃:</td>
                      <td className="p-1 text-center">0.3</td>
                      <td className="p-1 text-center">0.2</td>
                      <td className="p-1 text-center">0.1</td>
                      <td className="p-1 text-center font-bold">0.6</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p className="mt-4">
                <strong>Question 1:</strong> In CLIP&rsquo;s loss, does the
                gradient for cell (0, 1) depend on cell (0, 3)?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>Yes.</strong> Both cells are in the same softmax
                    row. The denominator sums{' '}
                    e<sup>s₀₀</sup> + e<sup>s₀₁</sup> + e<sup>s₀₂</sup> + e<sup>s₀₃</sup>.
                    Changing s₀₃ changes the denominator, which changes the
                    probability for cell (0, 1).
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> In SigLIP&rsquo;s loss, does the
                gradient for cell (0, 1) depend on cell (0, 3)?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>No.</strong> Each cell is an independent binary
                    classification. <InlineMath math="\sigma(s_{01} \cdot t)" /> does
                    not reference <InlineMath math="s_{03}" /> or any other cell.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 3:</strong> If we add 1,000 more pairs (batch
                becomes 1,004), which loss function&rsquo;s computation for cell
                (0, 1) changes?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>Only CLIP&rsquo;s.</strong> The softmax denominator
                    now has 1,004 terms instead of 4. SigLIP&rsquo;s loss for
                    cell (0, 1) is unchanged&mdash;it depends only on
                    s₀₁ and its label.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 4:</strong> How many loss terms does CLIP
                compute? How many does SigLIP compute?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>CLIP: 2N = 8</strong> (4 row-wise cross-entropy + 4
                    column-wise). <strong>SigLIP: N&sup2; = 16</strong> (one
                    binary classification per cell). SigLIP extracts more
                    training signal from the same batch.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 7: Explain Part 2 -- SigLIP in Practice */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SigLIP in Practice"
            subtitle="Same architecture, same applications, freed from the batch-size constraint"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The original SigLIP (2023) uses the same architecture as CLIP:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Same ViT image encoder&mdash;recall &ldquo;tokenize the image&rdquo;
                from <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>
              </li>
              <li>Same transformer text encoder</li>
              <li>
                Learned temperature parameter&mdash;same as CLIP, your temperature
                intuition carries over directly
              </li>
              <li>Same training setup: contrastive learning on image-text pairs</li>
            </ul>
            <p className="text-muted-foreground">
              The key result: SigLIP matches or exceeds CLIP performance at much
              smaller batch sizes. It works well from batch 256 to batch 32,768.
              CLIP only works well at the high end.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="A Practical Note">
            With sigmoid loss, the ratio of positive to negative examples in each
            batch is 1:N (one match per image, N&minus;1 non-matches). SigLIP
            learns a bias term to handle this class imbalance&mdash;a standard
            technique from binary classification.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Negative/boundary example: SigLIP != small-batch training */}
      <Row>
        <Row.Content>
          <GradientCard title="SigLIP Does NOT Mean Small Batches" color="amber">
            <div className="space-y-2">
              <p>
                SigLIP at batch 256 works fine. But SigLIP at batch 32,768 is{' '}
                <strong>even better</strong>&mdash;more negative pairs means more
                training signal. The advantage is not avoidance of large batches.
                It is <strong>flexibility</strong>:
              </p>
              <ul className="space-y-1 ml-4">
                <li>&bull; CLIP at batch 256: fails (softmax normalization unreliable)</li>
                <li>&bull; CLIP at batch 32,768: excellent</li>
                <li>&bull; SigLIP at batch 256: good (each pair scored independently)</li>
                <li>&bull; SigLIP at batch 32,768: excellent (more signal, graceful scaling)</li>
              </ul>
              <p>
                The improvement from larger batches is gradual and graceful, not a
                hard threshold.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 8: Check 2 -- Transfer Question */}
      <Row>
        <Row.Content>
          <GradientCard title="Check Your Understanding" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                A colleague is training a vision-language model for specialized
                medical imaging. They have 50,000 image-report pairs and are
                using CLIP&rsquo;s softmax loss with batch_size=64 (limited by GPU
                memory with large medical images).
              </p>

              <p className="mt-4">
                <strong>Question 1:</strong> Why might CLIP&rsquo;s softmax loss
                perform poorly at batch_size=64?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Softmax at batch 64 means each row&rsquo;s probability
                    distribution is over only 64 items. The denominator is too
                    small for reliable normalization&mdash;the gradient signal is
                    weak and noisy.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> Would switching to SigLIP&rsquo;s
                sigmoid loss help? Why or why not?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>Yes.</strong> Sigmoid loss scores each pair
                    independently, no batch-size minimum. At batch 64, SigLIP
                    still gets 64&sup2; = 4,096 binary classification signals per
                    batch, each with a self-contained gradient.
                  </p>
                </div>
              </details>

            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 9: Explain Part 3 -- What SigLIP 2 Adds */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What SigLIP 2 Adds"
            subtitle="Same loss, better training recipe"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The sigmoid loss was the original SigLIP contribution (2023). SigLIP
              2 (2025) keeps the same loss but adds training methodology
              improvements that significantly boost performance. The &ldquo;2&rdquo;
              represents a different kind of advance: not a new loss function, but
              a better recipe for training the same architecture.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The '2' Is Not a Minor Update">
            SigLIP 2&rsquo;s improvements beyond the loss function are
            substantial and independently important: multi-stage training,
            self-distillation, multi-resolution, multilingual data. Different
            kind of innovation: loss function design vs training methodology.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Multi-stage training */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Multi-stage training
            </p>
            <div className="space-y-3">
              <GradientCard title="Stage 1: Contrastive Pretraining" color="cyan">
                <p>
                  Standard contrastive pretraining on large-scale image-text
                  data&mdash;the same as original SigLIP. Build the initial
                  shared embedding space with sigmoid loss.
                </p>
              </GradientCard>
              <GradientCard title="Stage 2: Self-Distillation" color="blue">
                <p>
                  Continue training with the Stage 1 model as a teacher. The
                  model refines its own representations using soft targets from
                  its earlier self.
                </p>
              </GradientCard>
              <GradientCard title="Stage 3: Multi-Resolution Fine-Tuning" color="violet">
                <p>
                  Process images at multiple resolutions during training. The
                  model develops robust features that work across different image
                  sizes and aspect ratios.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* Self-distillation explanation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Self-distillation: the model teaches itself
            </p>
            <p className="text-muted-foreground">
              <strong>Self-distillation</strong> uses the model from a previous
              training stage as a teacher for the next stage. Unlike standard
              knowledge distillation (where a large teacher trains a small
              student), self-distillation uses the{' '}
              <strong>same architecture</strong>. The model&rsquo;s earlier version
              provides soft targets&mdash;probability distributions rather than
              hard labels&mdash;that capture learned relationships the binary
              labels miss.
            </p>
            <p className="text-muted-foreground">
              Think of it as &ldquo;the model refining its own understanding by
              revisiting the same material with better judgment from its first
              pass.&rdquo; The Stage 1 model learned a good embedding space. The
              Stage 2 model uses those learned relationships as a richer training
              signal than the original binary match/no-match labels.
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium text-foreground">
                Concrete example: hard vs soft targets
              </p>
              <p className="text-sm text-muted-foreground">
                Consider a row of the similarity matrix for an image of
                &ldquo;a golden retriever on a beach.&rdquo; The hard binary
                label says only the exact paired caption is a match:
              </p>
              <p className="text-sm text-muted-foreground font-mono">
                Hard target: [0, 0, 1, 0]
              </p>
              <p className="text-sm text-muted-foreground">
                But the Stage 1 model&rsquo;s soft predictions capture that
                &ldquo;a dog playing in sand&rdquo; is quite similar, even
                though it is not the exact pair:
              </p>
              <p className="text-sm text-muted-foreground font-mono">
                Soft target: [0.05, 0.10, 0.70, 0.15]
              </p>
              <p className="text-sm text-muted-foreground">
                In Stage 2, the model trains with these soft similarities as
                targets. It learns that near-matches are not as wrong as
                complete mismatches&mdash;a richer signal than binary 0 or 1.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Self-Distillation vs Knowledge Distillation">
            <ul className="space-y-1 text-sm">
              <li>&bull; <strong>Knowledge distillation:</strong> large teacher &rarr; small student (different models)</li>
              <li>&bull; <strong>Self-distillation:</strong> earlier checkpoint &rarr; later checkpoint (same model)</li>
            </ul>
            No separate larger model needed. The model teaches itself from its
            own previous state.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Multi-resolution and multilingual */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Multi-resolution and multilingual support
            </p>
            <p className="text-muted-foreground">
              <strong>Multi-resolution:</strong> images processed at multiple
              resolutions during later training stages. This helps the model
              develop robust features that work across different image sizes and
              aspect ratios&mdash;important because real-world images are not all
              the same size.
            </p>
            <p className="text-muted-foreground">
              <strong>Multilingual:</strong> training data includes multilingual
              image-text pairs, producing a vision encoder that works with text in
              many languages. The shared embedding space is not English-only.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="SigLIP 2 = Two Kinds of Innovation" color="orange">
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="font-medium mb-1">Loss Function (2023)</p>
                <p>
                  Sigmoid replaces softmax. Per-pair binary classification
                  removes batch-size dependency. One targeted change.
                </p>
              </div>
              <div>
                <p className="font-medium mb-1">Training Recipe (2025)</p>
                <p>
                  Multi-stage training, self-distillation, multi-resolution,
                  multilingual data. Same loss, substantially better results.
                </p>
              </div>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Check 3: Transfer question about SigLIP 2's training recipe */}
      <Row>
        <Row.Content>
          <GradientCard title="Check Your Understanding: SigLIP 2's Recipe" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                Returning to the medical imaging scenario: your colleague has
                switched to sigmoid loss (which helped at batch 64). Now they
                want to improve further.
              </p>

              <p className="mt-4">
                <strong>Question:</strong> What from SigLIP 2&rsquo;s training
                recipe might be useful for their specialized medical imaging
                domain?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Multi-resolution training is especially useful for medical
                    images (which come in many sizes and aspect ratios).
                    Self-distillation could help the model refine its features on
                    the smaller dataset. Starting from a pretrained SigLIP 2
                    model and fine-tuning&mdash;the same &ldquo;hire
                    experienced, train specific&rdquo; pattern from{' '}
                    <LessonLink slug="transfer-learning">Transfer Learning</LessonLink>&mdash;would be the
                    practical approach.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 10: Elaborate -- SigLIP as a Building Block */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SigLIP as a Building Block"
            subtitle="The vision encoder inside modern VLMs"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SigLIP&rsquo;s real-world impact: it is the vision encoder in
              PaliGemma (Google&rsquo;s open VLM) and other models. This is where
              the practical payoff lives.
            </p>
            <p className="text-muted-foreground">
              <strong>Why contrastive pretraining produces good vision
              encoders:</strong> the shared embedding space learned during
              contrastive training means the vision encoder has learned to extract
              features that are <strong>meaningful in terms of language</strong>.
              When you use this encoder in a VLM (connecting it to a language
              model), the vision features are already partially aligned with the
              language model&rsquo;s representation space. This is why contrastive
              pretraining (CLIP/SigLIP) is the default recipe for vision encoders
              in VLMs.
            </p>
            <p className="text-muted-foreground">
              The pattern from <LessonLink slug="transfer-learning">Transfer Learning</LessonLink>: &ldquo;hire
              experienced, train specific.&rdquo; The SigLIP encoder is the
              &ldquo;experienced hire&rdquo;&mdash;it understands images in terms
              that relate to language. Plug it into a VLM and train the connection.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Contrastive = Good Backbone">
            A vision encoder trained with contrastive learning does not just
            recognize objects&mdash;it represents them in a space where language
            comparisons are natural. This is fundamentally more useful for
            vision-language models than a classification-trained encoder.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Downstream uses of SigLIP encoders
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Zero-Shot Classification" color="emerald">
                <p>
                  Same procedure as CLIP: encode class descriptions, encode the
                  image, find the best match. The shared embedding space
                  generalizes to any text description.
                </p>
              </GradientCard>
              <GradientCard title="Image Retrieval" color="blue">
                <p>
                  Find images matching a text query, or texts matching an image.
                  Same embedding space, different application direction.
                </p>
              </GradientCard>
              <GradientCard title="VLM Vision Backbone" color="violet">
                <p>
                  Plug the SigLIP encoder into a language model (PaliGemma,
                  etc.). Vision features already aligned with language&mdash;the
                  connection is easier to train.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'CLIP\u2019s softmax normalizes across the entire batch, making batch size a structural requirement.',
                description:
                  'The softmax denominator sums over every item in the row. The quality of the probability distribution\u2014and therefore the gradient signal\u2014depends on batch size. CLIP needs batch size 32,768 to produce meaningful gradients.',
              },
              {
                headline:
                  'SigLIP replaces softmax with sigmoid: each pair scored independently as match or not.',
                description:
                  'Each cell in the similarity matrix becomes an independent binary classification. The loss for any pair depends only on that pair\u2019s similarity and label\u2014no denominator, no coupling, no batch-size dependency.',
              },
              {
                headline:
                  'Batch-size independence\u2014SigLIP works well at batch 256 or 32,768.',
                description:
                  'Larger batches still help (more negatives = more training signal), but the improvement is gradual and graceful. No hard threshold. Batch size becomes a choice, not a constraint.',
              },
              {
                headline:
                  'SigLIP 2 adds training methodology: multi-stage, self-distillation, multi-resolution.',
                description:
                  'The sigmoid loss was the 2023 contribution. SigLIP 2 (2025) keeps the same loss but adds a better training recipe\u2014multi-stage training with self-distillation and multi-resolution processing.',
              },
              {
                headline:
                  'SigLIP encoders power modern VLMs like PaliGemma.',
                description:
                  'Contrastive pretraining produces vision features naturally aligned with language. Plug the encoder into a language model and the connection is easier to train\u2014the \u201chire experienced, train specific\u201d pattern from Transfer Learning.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>
              CLIP asks &ldquo;which of these N items is my match?&rdquo;
              (multiple choice&mdash;harder with more options). SigLIP asks
              &ldquo;is this specific pair a match?&rdquo; (true/false&mdash;same
              difficulty regardless of batch size). One line of code changes.
              Everything else stays the same.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Sigmoid Loss for Language Image Pre-Training',
                authors: 'Zhai et al., 2023 (Google)',
                url: 'https://arxiv.org/abs/2303.15343',
                note: 'The original SigLIP paper. Section 2 describes the sigmoid loss and why it removes batch-size dependency.',
              },
              {
                title: 'SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding',
                authors: 'Tschannen et al., 2025 (Google DeepMind)',
                url: 'https://arxiv.org/abs/2502.14786',
                note: 'The SigLIP 2 paper. Section 3 covers multi-stage training and self-distillation.',
              },
              {
                title: 'Learning Transferable Visual Models From Natural Language Supervision',
                authors: 'Radford et al., 2021 (OpenAI)',
                url: 'https://arxiv.org/abs/2103.00020',
                note: 'The original CLIP paper. Understanding CLIP is the prerequisite for understanding why SigLIP matters.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SigLIP solved a specific problem in vision-language pretraining:
              making contrastive learning work without requiring enormous batch
              sizes. Next, we will look at a different kind of vision model: SAM
              (Segment Anything Model), which takes the &ldquo;foundation
              model&rdquo; approach to image segmentation&mdash;training one model
              that can segment any object in any image, guided by a prompt.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/sam-3"
            title="Up Next: SAM 3"
            description="The Segment Anything Model&mdash;promptable image segmentation as a foundation model approach to vision."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
