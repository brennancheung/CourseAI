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
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

/**
 * Finetuning for Classification
 *
 * First lesson in Module 4.4 (Beyond Pretraining).
 * Hands-on lesson with notebook (4-4-1-finetuning-for-classification.ipynb).
 *
 * Cognitive load: BUILD — two new concepts that connect directly to
 * established patterns (CNN transfer learning + causal masking).
 *
 * Core concepts at DEVELOPED:
 * - Adding a classification head to a pretrained language model
 * - Extracting a sequence representation from a causal transformer (last token)
 *
 * Core concepts at INTRODUCED:
 * - Frozen vs unfrozen backbone comparison
 *
 * EXPLICITLY NOT COVERED:
 * - Instruction tuning or SFT (Lesson 2)
 * - RLHF or alignment (Lesson 3)
 * - LoRA or parameter-efficient finetuning (Lesson 4)
 * - Token classification (NER, POS tagging)
 * - Prompt-based classification (zero-shot, few-shot)
 * - HuggingFace Trainer API
 *
 * Previous: Loading Real Weights (Module 4.3, Lesson 4)
 * Next: Instruction Tuning (Module 4.4, Lesson 2)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Causal Attention Pattern
// Shows which tokens each position can attend to, highlighting that
// the last token has the most context.
// ---------------------------------------------------------------------------

function CausalAttentionDiagram() {
  const tokens = ['This', 'movie', 'was', 'terrible', '<end>']
  const n = tokens.length
  const cellSize = 44
  const labelW = 70
  const topLabelH = 60
  const padding = 16
  const gridW = n * cellSize
  const gridH = n * cellSize
  const svgW = labelW + gridW + padding * 2
  const svgH = topLabelH + gridH + padding * 2 + 30

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Column headers (keys / "attends to") */}
        {tokens.map((tok, j) => (
          <text
            key={`col-${j}`}
            x={labelW + padding + j * cellSize + cellSize / 2}
            y={topLabelH - 8}
            textAnchor="middle"
            fill="#e2e8f0"
            fontSize="10"
            fontFamily="monospace"
          >
            {tok}
          </text>
        ))}

        {/* Row labels (queries / "from position") */}
        {tokens.map((tok, i) => (
          <text
            key={`row-${i}`}
            x={labelW + padding - 8}
            y={topLabelH + padding + i * cellSize + cellSize / 2 + 3}
            textAnchor="end"
            fill={i === n - 1 ? '#34d399' : '#e2e8f0'}
            fontSize="10"
            fontFamily="monospace"
            fontWeight={i === n - 1 ? '700' : '400'}
          >
            {tok}
          </text>
        ))}

        {/* Grid cells */}
        {tokens.map((_rowTok, i) =>
          tokens.map((_colTok, j) => {
            const canAttend = j <= i
            const x = labelW + padding + j * cellSize
            const y = topLabelH + padding + i * cellSize
            const isLastRow = i === n - 1

            return (
              <g key={`cell-${i}-${j}`}>
                <rect
                  x={x}
                  y={y}
                  width={cellSize}
                  height={cellSize}
                  fill={
                    canAttend
                      ? isLastRow
                        ? '#34d39930'
                        : '#6366f120'
                      : '#00000020'
                  }
                  stroke="#334155"
                  strokeWidth={0.5}
                />
                {canAttend && (
                  <text
                    x={x + cellSize / 2}
                    y={y + cellSize / 2 + 4}
                    textAnchor="middle"
                    fill={isLastRow ? '#34d399' : '#6366f1'}
                    fontSize="14"
                    fontWeight="600"
                  >
                    {'✓'}
                  </text>
                )}
              </g>
            )
          })
        )}

        {/* Context count annotations on right */}
        {tokens.map((_tok, i) => {
          const y = topLabelH + padding + i * cellSize + cellSize / 2 + 3
          const count = i + 1
          const isLast = i === n - 1
          return (
            <text
              key={`count-${i}`}
              x={labelW + padding + gridW + 8}
              y={y}
              fill={isLast ? '#34d399' : '#9ca3af'}
              fontSize="9"
              fontWeight={isLast ? '700' : '400'}
            >
              {count === 1 ? '1 token' : `${count} tokens`}
            </text>
          )
        })}

        {/* Legend at bottom */}
        <text
          x={svgW / 2}
          y={svgH - 6}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="9"
        >
          Last token attends to ALL tokens = most context
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: GPT-2 with Classification Head
// Shows the architecture modification: frozen transformer backbone
// with lm_head replaced by a classification head.
// ---------------------------------------------------------------------------

function ClassificationHeadDiagram() {
  const boxW = 180
  const boxH = 32
  const gap = 12
  const centerX = 200
  const startY = 20

  const layers: Array<{
    label: string
    color: string
    textColor: string
    note?: string
    dashed?: boolean
  }> = [
    { label: 'Input tokens', color: '#1e293b', textColor: '#94a3b8' },
    { label: 'Token + Position Embedding', color: '#312e81', textColor: '#a5b4fc' },
    {
      label: 'Transformer Blocks (x12)',
      color: '#312e81',
      textColor: '#a5b4fc',
      note: 'FROZEN',
    },
    { label: 'Final Layer Norm', color: '#312e81', textColor: '#a5b4fc' },
    {
      label: 'hidden_states[:, -1, :]',
      color: '#064e3b',
      textColor: '#6ee7b7',
      note: 'LAST TOKEN',
    },
    {
      label: 'nn.Linear(768, num_classes)',
      color: '#7c2d12',
      textColor: '#fdba74',
      note: 'NEW HEAD',
    },
    { label: 'Class logits', color: '#1e293b', textColor: '#94a3b8' },
  ]

  const totalH = startY + layers.length * (boxH + gap) + 40
  const svgW = 400

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={totalH}
        viewBox={`0 0 ${svgW} ${totalH}`}
        className="overflow-visible"
      >
        {/* Frozen bracket */}
        <line
          x1={centerX - boxW / 2 - 20}
          y1={startY + (boxH + gap) * 1}
          x2={centerX - boxW / 2 - 20}
          y2={startY + (boxH + gap) * 3 + boxH}
          stroke="#6366f1"
          strokeWidth={2}
          strokeDasharray="4,4"
          opacity={0.6}
        />
        <text
          x={centerX - boxW / 2 - 26}
          y={startY + (boxH + gap) * 2 + boxH / 2}
          textAnchor="end"
          fill="#6366f1"
          fontSize="9"
          fontWeight="600"
          opacity={0.8}
        >
          Frozen
        </text>

        {layers.map((layer, i) => {
          const y = startY + i * (boxH + gap)
          const x = centerX - boxW / 2

          return (
            <g key={i}>
              {/* Box */}
              <rect
                x={x}
                y={y}
                width={boxW}
                height={boxH}
                rx={4}
                fill={layer.color}
                stroke={
                  layer.note === 'NEW HEAD'
                    ? '#f97316'
                    : layer.note === 'LAST TOKEN'
                      ? '#34d399'
                      : '#334155'
                }
                strokeWidth={layer.note ? 1.5 : 0.5}
                strokeDasharray={layer.dashed ? '4,4' : '0'}
              />

              {/* Label */}
              <text
                x={centerX}
                y={y + boxH / 2 + 4}
                textAnchor="middle"
                fill={layer.textColor}
                fontSize="10"
                fontFamily="monospace"
              >
                {layer.label}
              </text>

              {/* Note badge */}
              {layer.note && (
                <text
                  x={centerX + boxW / 2 + 8}
                  y={y + boxH / 2 + 3}
                  fill={
                    layer.note === 'NEW HEAD'
                      ? '#f97316'
                      : layer.note === 'LAST TOKEN'
                        ? '#34d399'
                        : '#6366f1'
                  }
                  fontSize="8"
                  fontWeight="700"
                >
                  {layer.note}
                </text>
              )}

              {/* Arrow down */}
              {i < layers.length - 1 && (
                <line
                  x1={centerX}
                  y1={y + boxH}
                  x2={centerX}
                  y2={y + boxH + gap}
                  stroke="#475569"
                  strokeWidth={1}
                  markerEnd="url(#arrowDown)"
                />
              )}
            </g>
          )
        })}

        {/* Arrow marker */}
        <defs>
          <marker
            id="arrowDown"
            markerWidth="6"
            markerHeight="4"
            refX="3"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 6 2, 0 4" fill="#475569" />
          </marker>
        </defs>

        {/* Removed lm_head annotation (grayed-out dashed box) */}
        {(() => {
          const headY = startY + 5 * (boxH + gap)
          const removedX = centerX - boxW / 2 - 140
          const removedW = 120
          return (
            <g opacity={0.5}>
              <rect
                x={removedX}
                y={headY}
                width={removedW}
                height={boxH}
                rx={4}
                fill="none"
                stroke="#64748b"
                strokeWidth={1}
                strokeDasharray="4,3"
              />
              <text
                x={removedX + removedW / 2}
                y={headY + boxH / 2 + 3}
                textAnchor="middle"
                fill="#64748b"
                fontSize="9"
                fontFamily="monospace"
                textDecoration="line-through"
              >
                lm_head (removed)
              </text>
              <line
                x1={removedX + removedW}
                y1={headY + boxH / 2}
                x2={centerX - boxW / 2 - 4}
                y2={headY + boxH / 2}
                stroke="#64748b"
                strokeWidth={0.5}
                strokeDasharray="3,3"
              />
            </g>
          )
        })()}

        {/* Legend */}
        <g transform={`translate(${centerX - 100}, ${totalH - 20})`}>
          <rect x={0} y={-4} width={10} height={10} rx={2} fill="#312e81" stroke="#6366f1" strokeWidth={1} />
          <text x={14} y={4} fill="#9ca3af" fontSize="8">Pretrained (frozen)</text>
          <rect x={100} y={-4} width={10} height={10} rx={2} fill="#064e3b" stroke="#34d399" strokeWidth={1} />
          <text x={114} y={4} fill="#9ca3af" fontSize="8">Selection</text>
          <rect x={170} y={-4} width={10} height={10} rx={2} fill="#7c2d12" stroke="#f97316" strokeWidth={1} />
          <text x={184} y={4} fill="#9ca3af" fontSize="8">New (trainable)</text>
        </g>
      </svg>
    </div>
  )
}

export function FinetuningForClassificationLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Finetuning for Classification"
            description="Adapt your pretrained GPT-2 for text classification&mdash;the same transfer learning pattern you used with CNNs, applied to a language model."
            category="Fine-tuning"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints (Section 1 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Take a pretrained GPT-2 model and adapt it for text
            classification by adding a classification head, choosing the
            right hidden state as the sequence representation, and training
            with a frozen backbone. You already know this pattern from
            CNNs&mdash;now apply it to a transformer.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="You Know This Pattern">
            In the CNN transfer learning lesson, you froze a ResNet
            backbone and replaced the classification head. This lesson
            does the exact same thing with GPT-2. The only genuinely new
            question: which hidden state represents the whole sequence?
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Add a classification head to pretrained GPT-2',
              'Choose which hidden state to use as the sequence representation (and why)',
              'Train with a frozen backbone on a sentiment classification task',
              'Compare frozen vs unfrozen finetuning (introduced, not deeply developed)',
              'NOT: instruction tuning or SFT—that’s the next lesson',
              'NOT: LoRA or parameter-efficient finetuning—Lesson 4',
              'NOT: RLHF or alignment—Lesson 3',
              'NOT: prompt-based classification, token classification, or HuggingFace Trainer API',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook — The Bridge (Section 2 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Bridge"
            subtitle="You have done this before"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the transfer learning lesson, you took a ResNet pretrained
              on ImageNet and adapted it for flower classification. The
              recipe: freeze the backbone, replace{' '}
              <code className="text-xs">model.fc</code>, train on your
              labeled data. Three steps.
            </p>

            <p className="text-muted-foreground">
              You are about to do the exact same thing with GPT-2.
            </p>

            <ComparisonRow
              left={{
                title: 'CNN Transfer Learning',
                color: 'blue',
                items: [
                  '1. Load pretrained ResNet',
                  '2. Freeze conv layers (requires_grad=False)',
                  '3. Replace model.fc with nn.Linear(512, num_classes)',
                  '4. Train on labeled images',
                ],
              }}
              right={{
                title: 'GPT-2 Transfer Learning',
                color: 'amber',
                items: [
                  '1. Load pretrained GPT-2',
                  '2. Freeze transformer blocks (requires_grad=False)',
                  '3. Replace lm_head with nn.Linear(768, num_classes)',
                  '4. Train on labeled text',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The pattern is identical. Only the feature extractor changed.
              A CNN extracts visual features from pixels. A transformer
              extracts language features from tokens. Both produce a
              representation you can classify.
            </p>

            <GradientCard title="Prediction Checkpoint" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  Look at the two columns above. The steps are nearly
                  word-for-word identical. But there{' '}
                  <strong>is</strong> one genuine difference between
                  adapting a CNN and adapting a causal transformer.
                </p>
                <p>
                  <strong>What is it?</strong> Think about what the
                  backbone gives you in each case.
                </p>
                <details className="mt-3">
                  <summary className="font-medium cursor-pointer text-primary">
                    Think about it, then reveal
                  </summary>
                  <div className="mt-2 space-y-2">
                    <p>
                      A CNN gives you a <strong>spatial feature map</strong>{' '}
                      that you can pool into a single vector (global average
                      pooling). A causal transformer gives you{' '}
                      <strong>one hidden state per token position</strong>.
                      Which position do you use?
                    </p>
                    <p>
                      That is the one genuinely new concept in this
                      lesson.
                    </p>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Strategy, New Domain">
            &ldquo;Hire experienced, train specific.&rdquo; That mental
            model from the CNN transfer learning lesson applies directly.
            The pretrained GPT-2 is the experienced employee. The
            classification head is the job-specific training.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain — Which Hidden State? (Section 3 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Which Hidden State?"
            subtitle="The one genuinely new concept"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              When you adapted ResNet, global average pooling collapsed
              the spatial feature map into a single vector. Every spatial
              position contributed equally. That worked because a CNN
              processes the entire image at once&mdash;every position has
              seen all the context it needs.
            </p>

            <p className="text-muted-foreground">
              GPT-2 is different. After the forward pass, you have one
              hidden state per token position&mdash;a tensor of shape{' '}
              <code className="text-xs">(batch, seq_len, 768)</code>.
              You need to pick <strong>one</strong> of those positions to
              represent the entire sequence.
            </p>

            <p className="text-muted-foreground">
              The answer comes directly from something you already know:{' '}
              <strong>causal masking</strong>.
            </p>

            <CausalAttentionDiagram />

            <p className="text-muted-foreground">
              The first token (&ldquo;This&rdquo;) has attended to{' '}
              <strong>only itself</strong>. One token of context. It knows
              nothing about the rest of the sentence.
            </p>

            <p className="text-muted-foreground">
              The last token has attended to{' '}
              <strong>every previous token</strong>. It has the most
              context of any position. It is the only position that has
              &ldquo;read&rdquo; the entire input.
            </p>

            <p className="text-muted-foreground">
              <strong>
                Of course you use the last token.
              </strong>{' '}
              The last token&rsquo;s hidden state is the only position
              with full sequence context. It has processed all the
              information in the input. Using any other position throws
              away information.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Causal Masking Decides">
            This is not an arbitrary choice. Causal masking means each
            position can only see tokens before it. The last position has
            seen everything. The first position has seen nothing but
            itself. The architecture dictates the answer.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: first token */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Negative example:</strong> What if you used the{' '}
              <strong>first</strong> token&rsquo;s hidden state instead?
              You would be classifying the entire review based on a
              representation that has seen exactly one token. For
              &ldquo;This movie was terrible,&rdquo; you would classify
              based on the representation of &ldquo;This&rdquo; alone. It
              has no idea that the review is negative.
            </p>

            <p className="text-muted-foreground">
              You might wonder about BERT, which uses a special{' '}
              <code className="text-xs">[CLS]</code> token at position
              0 for classification. That works because BERT is{' '}
              <strong>bidirectional</strong>&mdash;every position
              attends to every other position. In BERT, the first token
              sees everything. In GPT, the first token sees nothing.
              The architecture determines the strategy.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="First Token = Worst Choice">
            Using the first token for classification in a causal model
            means classifying based on zero context. It is the
            worst possible choice, not because of convention, but
            because of the causal mask.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Check 1 — Predict and Verify (Section 4 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Average All Positions?" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                In CNNs, global average pooling averages all spatial
                positions. What if you averaged all token hidden states
                instead of taking the last one? Would it work better or
                worse? Why?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Worse.</strong> Early positions have limited
                    context due to causal masking. Position 0 has seen 1
                    token. Position 1 has seen 2. Averaging them all
                    dilutes the full-context representation of the last
                    token with the limited-context representations of
                    earlier tokens.
                  </p>
                  <p>
                    This is different from CNNs, where every spatial
                    position has equal access to the entire receptive
                    field. In a CNN, averaging is reasonable. In a causal
                    transformer, it mixes rich representations with poor
                    ones.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          6. Explain — The Classification Head (Section 5 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Classification Head"
            subtitle="Remove lm_head, add a linear layer"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The pretrained GPT-2 ends with{' '}
              <code className="text-xs">lm_head</code>: a linear layer
              that projects from 768 dimensions to 50,257 (the vocabulary
              size) for next-token prediction. For classification, you
              replace that with a linear layer that projects to the number
              of classes.
            </p>

            <CodeBlock
              code={`class GPT2ForClassification(nn.Module):
    def __init__(self, gpt_model, num_classes):
        super().__init__()
        # Keep the pretrained transformer backbone
        self.transformer = gpt_model.transformer

        # Replace lm_head with a classification head
        # 768 = GPT-2's hidden dimension (n_embd)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids):
        # 1. Run through the transformer backbone
        hidden_states = self.transformer(input_ids)
        # hidden_states shape: (batch, seq_len, 768)

        # 2. Take the LAST token's hidden state
        last_hidden = hidden_states[:, -1, :]
        # last_hidden shape: (batch, 768)

        # 3. Classify
        logits = self.classifier(last_hidden)
        # logits shape: (batch, num_classes)

        return logits`}
              language="python"
              filename="classification_model.py"
            />

            <p className="text-muted-foreground">
              That is the entire model. The transformer backbone does
              the heavy lifting&mdash;extracting language features from
              the input text. The classification head maps those features
              to class predictions.
            </p>

            <p className="text-muted-foreground">
              Notice that we are not using{' '}
              <code className="text-xs">lm_head</code> at all. In the
              pretrained model, <code className="text-xs">lm_head</code>{' '}
              shared its weights with the token embedding via weight
              tying&mdash;you verified this with{' '}
              <code className="text-xs">data_ptr()</code> in the
              loading-real-weights lesson. By replacing{' '}
              <code className="text-xs">lm_head</code> with a
              classification head, we break that tie. The embedding
              matrix stays (we still need to embed input tokens), but
              the output projection now maps to class labels instead of
              vocabulary logits.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Parameter Count">
            The classification head is tiny: 768 {'×'} num_classes
            parameters. For binary sentiment (2 classes), that is 1,536
            parameters. The backbone has 124 million. You are training
            0.001% of the total model.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Worked example */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium">
              Worked example: tracing a sentence through the model
            </p>

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3 text-sm text-muted-foreground">
              <p>
                Input: <code className="text-xs">&quot;This movie was terrible&quot;</code>
              </p>
              <p>
                <strong>Step 1:</strong> Tokenize{' '}
                {'→'}{' '}
                <code className="text-xs">[1212, 3807, 373, 7818]</code>{' '}
                (4 tokens)
              </p>
              <p>
                <strong>Step 2:</strong> Forward through transformer{' '}
                {'→'}{' '}
                hidden states of shape{' '}
                <code className="text-xs">(1, 4, 768)</code>
              </p>
              <p>
                <strong>Step 3:</strong> Take last token{' '}
                {'→'}{' '}
                <code className="text-xs">hidden_states[:, -1, :]</code>{' '}
                {'→'}{' '}
                shape <code className="text-xs">(1, 768)</code>
              </p>
              <p>
                <strong>Step 4:</strong> Classification head{' '}
                {'→'}{' '}
                <code className="text-xs">nn.Linear(768, 2)</code>{' '}
                {'→'}{' '}
                logits <code className="text-xs">[1.2, -0.8]</code>
              </p>
              <p>
                <strong>Step 5:</strong> Cross-entropy loss with label 0
                (negative) {'→'} backprop through classifier only
              </p>
            </div>

            <p className="text-muted-foreground">
              You might expect classification to require a different
              tokenizer or special input format. It does not&mdash;the
              exact same BPE tokenization and forward pass from
              generation applies here. The model&rsquo;s existing
              representations already capture what it needs to classify.
            </p>

            <p className="text-muted-foreground">
              Notice what did not change: the tokenizer, the forward pass
              through the transformer, the hidden state computation. The{' '}
              <strong>only</strong> difference from generation is what you
              do with the output. Generation takes logits over the
              vocabulary. Classification takes the last hidden state to a
              new linear layer.
            </p>

            <ClassificationHeadDiagram />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Input Pipeline">
            You do not need a special tokenizer or input format for
            classification. Same tiktoken encoding. Same model forward
            pass. Same hidden states. The only change is what you do
            with the output.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Explain — Freezing and Training (Section 6 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Freezing and Training"
            subtitle="Same training loop, different objective"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Freeze the backbone exactly the same way you did with
              ResNet: set{' '}
              <code className="text-xs">requires_grad = False</code> on
              every parameter in the transformer, then train only the
              classification head.
            </p>

            <CodeBlock
              code={`# Freeze the entire transformer backbone
for param in model.transformer.parameters():
    param.requires_grad = False

# Only the classifier head will be updated
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)

# Training loop — same heartbeat as always
for epoch in range(num_epochs):
    for input_ids, labels in train_loader:
        logits = model(input_ids)
        loss = nn.CrossEntropyLoss()(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`}
              language="python"
              filename="training.py"
            />

            <p className="text-muted-foreground">
              Compare this to the pretraining loop you wrote in the
              pretraining lesson. The structure is identical: forward,
              loss, backward, step. The differences are surface-level:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Loss compares against <strong>class labels</strong>{' '}
                (not next-token targets)
              </li>
              <li>
                Optimizer updates only <strong>classifier
                parameters</strong> (not the full model)
              </li>
              <li>
                No LR warmup needed&mdash;the head is small and trains
                fast
              </li>
              <li>
                <code className="text-xs">nn.CrossEntropyLoss</code>{' '}
                with 2 classes instead of 50,257
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Heartbeat">
            Forward, loss, backward, step. The training loop structure has
            not changed since your very first linear regression in Series 1.
            The &ldquo;same heartbeat, new instruments&rdquo; pattern
            continues.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Check 2 — Transfer Question (Section 7 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Swap the Task" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You trained a sentiment head (2 classes: positive,
                negative). Now you want topic classification (4 classes:
                sports, politics, technology, entertainment).
              </p>
              <p>
                <strong>What changes in the code? What stays the same?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Changes:</strong>{' '}
                    <code className="text-xs">nn.Linear(768, 2)</code>{' '}
                    {'→'}{' '}
                    <code className="text-xs">nn.Linear(768, 4)</code>{' '}
                    and the dataset/labels. That is it.
                  </p>
                  <p>
                    <strong>Stays the same:</strong> The frozen backbone,
                    the tokenizer, the forward pass, the training loop,
                    the last-token selection, the loss function (cross-entropy
                    works for any number of classes).
                  </p>
                  <p>
                    This confirms the backbone is a{' '}
                    <strong>general feature extractor</strong>. The same
                    pretrained features support any classification task by
                    swapping the head.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>

          <p className="text-muted-foreground mt-4">
            The only change:{' '}
            <code className="text-xs">nn.Linear(768, 2)</code> becomes{' '}
            <code className="text-xs">nn.Linear(768, 4)</code>. The
            backbone is a general text feature extractor&mdash;swap the
            head dimensions and the dataset, and the same frozen model
            supports any classification task.
          </p>
        </Row.Content>
      </Row>

      {/* ================================================================
          9. Explore — Notebook (Section 8 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="Finetune GPT-2 for sentiment classification"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The lesson showed you the concepts. Now implement it. The
              notebook walks you through the complete pipeline: loading
              pretrained GPT-2, tokenizing SST-2 examples, implementing
              the classification head, training with a frozen backbone,
              evaluating accuracy, and comparing frozen vs unfrozen
              performance.
            </p>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Finetune GPT-2 for sentiment classification on SST-2.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-4-1-finetuning-for-classification.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes: loading pretrained GPT-2 via
                  HuggingFace, tokenizing SST-2 with tiktoken, implementing
                  the classification head, training with frozen backbone,
                  evaluating accuracy, unfreezing last N blocks with
                  differential learning rates, and generating text before
                  and after finetuning to observe catastrophic forgetting
                  (or lack thereof). Use a GPU runtime in
                  Colab&mdash;even frozen-backbone training benefits from
                  GPU acceleration for the transformer forward pass.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Notebook Exercises">
            <ul className="space-y-2 text-sm">
              <li>{'•'} Load GPT-2 and tokenize SST-2 examples</li>
              <li>{'•'} Implement the classification head</li>
              <li>{'•'} Train with frozen backbone and evaluate</li>
              <li>{'•'} Unfreeze last N blocks with differential LR</li>
              <li>{'•'} Generate text before and after finetuning</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Elaborate — Frozen vs Unfrozen (Section 9 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Frozen vs Unfrozen"
            subtitle="The same decision framework as CNNs"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the CNN transfer learning lesson, you learned a
              decision framework: the choice between frozen backbone
              (feature extraction) and full finetuning depends on dataset
              size and domain similarity. The same logic applies here.
            </p>

            <ComparisonRow
              left={{
                title: 'Frozen Backbone',
                color: 'blue',
                items: [
                  'Fast training (only head parameters)',
                  'Low memory (no gradient storage for backbone)',
                  'Safe from catastrophic forgetting',
                  'Works well when pretrained domain is close to target',
                  'Default starting strategy',
                ],
              }}
              right={{
                title: 'Full Finetuning',
                color: 'amber',
                items: [
                  'Slower training (all 124M parameters)',
                  'High memory (gradients for everything)',
                  'Risk of catastrophic forgetting',
                  'Massive overfitting risk: 124M params on a small dataset',
                  'Potentially higher accuracy with enough data',
                  'Use when frozen accuracy plateaus',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The strongest argument for starting frozen:{' '}
              <strong>overfitting</strong>. The classification head has
              1,536 parameters for binary classification&mdash;well
              matched to a dataset of thousands of examples. Full
              finetuning makes all 124M parameters trainable on that
              same small dataset. That is a massive capacity mismatch.
              The model has far more parameters than the task needs, and
              will memorize the training set instead of learning the
              pattern.
            </p>

            <p className="text-muted-foreground">
              For transformers, the &ldquo;domain&rdquo; consideration
              is different from CNNs. A language model pretrained on web
              text has broad domain coverage&mdash;it has likely seen
              text similar to most classification tasks. So frozen
              backbone feature extraction often works surprisingly well,
              even for specialized text.
            </p>

            <p className="text-muted-foreground">
              <strong>Partial unfreezing</strong> is the middle ground.
              Unfreeze the last few transformer blocks and train with
              differential learning rates&mdash;exactly like unfreezing{' '}
              <code className="text-xs">layer4</code> in ResNet while
              keeping earlier layers frozen. In the CNN project, you used{' '}
              <code className="text-xs">1e-4</code> for the backbone and{' '}
              <code className="text-xs">1e-3</code> for the head. The
              same strategy works here.
            </p>

            <p className="text-muted-foreground">
              <strong>Catastrophic forgetting:</strong> With a frozen
              backbone, the model&rsquo;s internal representations do not
              change at all. Generate text before and after
              finetuning&mdash;the output is identical. The model has not
              &ldquo;forgotten&rdquo; anything. With aggressive full
              finetuning on a narrow task, the model can lose its general
              language capabilities. This is something you can observe
              directly in the notebook.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Catastrophic Forgetting">
            If you unfreeze and train aggressively on a narrow task,
            the model can forget how to generate coherent text. With a
            frozen backbone, the model&rsquo;s representations are
            unchanged&mdash;it can still generate text exactly as before.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Summary (Section 10 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'A pretrained transformer is a text feature extractor.',
                description:
                  'Add a classification head, freeze the backbone, train the head. The same pattern as CNN transfer learning—only the feature extractor changed.',
              },
              {
                headline:
                  'Use the last token’s hidden state.',
                description:
                  'Causal masking means the last token has attended to all previous tokens—it is the only position with full sequence context. The architecture dictates this choice.',
              },
              {
                headline:
                  'The classification head is tiny.',
                description:
                  'For binary classification: 768 × 2 = 1,536 trainable parameters out of 124 million total. You are training 0.001% of the model.',
              },
              {
                headline:
                  'Frozen vs unfrozen is the same decision framework.',
                description:
                  'Start frozen (safe, fast). Unfreeze later blocks with differential LR if accuracy plateaus. Same strategy as CNN transfer learning.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          12. Next Step (Section 11 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">
              What comes next
            </p>
            <p className="text-sm text-muted-foreground">
              Classification finetuning adapts a model for one specific
              narrow task&mdash;positive vs negative, or sports vs
              politics. But what if you want a model that can follow{' '}
              <strong>any instruction</strong>? That requires a different
              kind of adaptation&mdash;not a new head, but a new training
              dataset. Next: instruction tuning.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Complete the notebook: implement the classification head, train on SST-2, compare frozen vs unfrozen, and generate text before and after finetuning. Then review your session."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
