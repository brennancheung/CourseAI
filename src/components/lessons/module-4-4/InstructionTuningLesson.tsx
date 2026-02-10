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
 * Instruction Tuning (SFT)
 *
 * Second lesson in Module 4.4 (Beyond Pretraining).
 * Hands-on lesson with notebook (4-4-2-instruction-tuning.ipynb).
 *
 * Cognitive load: STRETCH — three new concepts but familiar mechanics.
 * The central insight: SFT teaches FORMAT, not knowledge.
 *
 * Core concepts at DEVELOPED:
 * - Instruction dataset format (instruction/response pairs)
 * - Chat templates and special tokens
 * - "SFT teaches format, not knowledge" (the central misconception to defeat)
 *
 * Core concepts at APPLIED:
 * - SFT training mechanics (student runs SFT in notebook)
 *
 * EXPLICITLY NOT COVERED:
 * - RLHF, DPO, or preference-based alignment (Lesson 3)
 * - LoRA or parameter-efficient finetuning (Lesson 4)
 * - Building production-quality instruction datasets
 * - Multi-turn conversation handling in depth
 * - Constitutional AI or RLAIF (Series 5)
 * - Evaluation of instruction-tuned models (beyond qualitative)
 * - Prompt engineering or in-context learning
 *
 * Previous: Finetuning for Classification (Module 4.4, Lesson 1)
 * Next: RLHF & Alignment (Module 4.4, Lesson 3)
 */

// ---------------------------------------------------------------------------
// Inline SVG: SFT Pipeline Diagram
// Shows the three-stage pipeline: pretraining -> SFT -> resulting behavior.
// The architecture stays the same; only the data changes.
// ---------------------------------------------------------------------------

function SftPipelineDiagram() {
  const svgW = 460
  const svgH = 220
  const boxW = 120
  const boxH = 70
  const gap = 40
  const startX = 30
  const topY = 20
  const dataY = 130

  const stages = [
    {
      label: 'Pretraining',
      data: 'Web text, books, code',
      behavior: 'Text completer',
      color: '#312e81',
      textColor: '#a5b4fc',
      dataColor: '#6366f140',
    },
    {
      label: 'SFT',
      data: 'Instruction/response pairs',
      behavior: 'Instruction follower',
      color: '#064e3b',
      textColor: '#6ee7b7',
      dataColor: '#34d39940',
    },
    {
      label: 'RLHF',
      data: 'Human preferences',
      behavior: 'Aligned assistant',
      color: '#44403c',
      textColor: '#78716c',
      dataColor: '#78716c30',
    },
  ]

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Architecture label spanning all stages */}
        <text
          x={svgW / 2}
          y={topY + boxH + 18}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="9"
          fontStyle="italic"
        >
          Same GPT architecture throughout—only the data changes
        </text>

        {stages.map((stage, i) => {
          const x = startX + i * (boxW + gap)

          return (
            <g key={stage.label}>
              {/* Stage box */}
              <rect
                x={x}
                y={topY}
                width={boxW}
                height={boxH}
                rx={6}
                fill={stage.color}
                stroke={i === 2 ? '#78716c' : stage.textColor}
                strokeWidth={i === 2 ? 0.5 : 1}
                strokeDasharray={i === 2 ? '4,3' : '0'}
              />
              <text
                x={x + boxW / 2}
                y={topY + 28}
                textAnchor="middle"
                fill={stage.textColor}
                fontSize="11"
                fontWeight="600"
              >
                {stage.label}
              </text>
              <text
                x={x + boxW / 2}
                y={topY + 48}
                textAnchor="middle"
                fill={stage.textColor}
                fontSize="8"
                opacity={0.8}
              >
                {stage.behavior}
              </text>

              {/* Data box below */}
              <rect
                x={x}
                y={dataY}
                width={boxW}
                height={42}
                rx={4}
                fill={stage.dataColor}
                stroke={i === 2 ? '#78716c40' : `${stage.textColor}40`}
                strokeWidth={0.5}
              />
              <text
                x={x + boxW / 2}
                y={dataY + 16}
                textAnchor="middle"
                fill={i === 2 ? '#78716c' : stage.textColor}
                fontSize="8"
                fontWeight="500"
              >
                Training data:
              </text>
              <text
                x={x + boxW / 2}
                y={dataY + 30}
                textAnchor="middle"
                fill={i === 2 ? '#78716c' : stage.textColor}
                fontSize="8"
              >
                {stage.data}
              </text>

              {/* Arrow to data */}
              <line
                x1={x + boxW / 2}
                y1={topY + boxH}
                x2={x + boxW / 2}
                y2={dataY}
                stroke={i === 2 ? '#78716c40' : '#475569'}
                strokeWidth={0.5}
              />

              {/* Arrow to next stage */}
              {i < stages.length - 1 && (
                <g>
                  <line
                    x1={x + boxW + 4}
                    y1={topY + boxH / 2}
                    x2={x + boxW + gap - 4}
                    y2={topY + boxH / 2}
                    stroke={i === 1 ? '#78716c60' : '#475569'}
                    strokeWidth={1}
                    markerEnd="url(#sftArrow)"
                  />
                </g>
              )}
            </g>
          )
        })}

        {/* "This lesson" bracket */}
        <rect
          x={startX + boxW + gap - 5}
          y={topY - 14}
          width={boxW + 10}
          height={12}
          rx={3}
          fill="#34d39920"
        />
        <text
          x={startX + boxW + gap + boxW / 2}
          y={topY - 5}
          textAnchor="middle"
          fill="#34d399"
          fontSize="8"
          fontWeight="600"
        >
          THIS LESSON
        </text>

        {/* "Lesson 3" bracket */}
        <rect
          x={startX + 2 * (boxW + gap) - 5}
          y={topY - 14}
          width={boxW + 10}
          height={12}
          rx={3}
          fill="#78716c15"
        />
        <text
          x={startX + 2 * (boxW + gap) + boxW / 2}
          y={topY - 5}
          textAnchor="middle"
          fill="#78716c"
          fontSize="8"
        >
          NEXT LESSON
        </text>

        <defs>
          <marker
            id="sftArrow"
            markerWidth="6"
            markerHeight="4"
            refX="5"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 6 2, 0 4" fill="#475569" />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Loss Masking Diagram
// Shows which tokens contribute to the loss (response only) vs which are
// ignored (instruction tokens).
// ---------------------------------------------------------------------------

function LossMaskingDiagram() {
  const tokens = [
    { text: '<|im_start|>', role: 'special', masked: true },
    { text: 'user', role: 'role', masked: true },
    { text: '\\n', role: 'special', masked: true },
    { text: 'Write', role: 'instruction', masked: true },
    { text: ' a', role: 'instruction', masked: true },
    { text: ' haiku', role: 'instruction', masked: true },
    { text: '<|im_end|>', role: 'special', masked: true },
    { text: '<|im_start|>', role: 'special', masked: true },
    { text: 'assistant', role: 'role', masked: true },
    { text: '\\n', role: 'special', masked: true },
    { text: 'Silicon', role: 'response', masked: false },
    { text: ' dreams', role: 'response', masked: false },
    { text: ' awake', role: 'response', masked: false },
    { text: '<|im_end|>', role: 'response', masked: false },
  ]

  const cellW = 70
  const cellH = 50
  const startX = 10
  const startY = 30
  const svgW = startX + tokens.length * cellW + 10
  const svgH = startY + cellH + 60

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Header labels */}
        <text
          x={startX + 3.5 * cellW}
          y={18}
          textAnchor="middle"
          fill="#6366f1"
          fontSize="9"
          fontWeight="600"
        >
          PROMPT (loss ignored)
        </text>
        <text
          x={startX + 11.5 * cellW}
          y={18}
          textAnchor="middle"
          fill="#34d399"
          fontSize="9"
          fontWeight="600"
        >
          RESPONSE (loss computed)
        </text>

        {tokens.map((tok, i) => {
          const x = startX + i * cellW
          const isMasked = tok.masked

          return (
            <g key={i}>
              <rect
                x={x}
                y={startY}
                width={cellW - 2}
                height={cellH}
                rx={3}
                fill={isMasked ? '#6366f115' : '#34d39920'}
                stroke={isMasked ? '#6366f140' : '#34d39960'}
                strokeWidth={0.5}
              />
              <text
                x={x + cellW / 2 - 1}
                y={startY + cellH / 2 + 3}
                textAnchor="middle"
                fill={isMasked ? '#6366f1' : '#34d399'}
                fontSize="8"
                fontFamily="monospace"
              >
                {tok.text}
              </text>

              {/* Label = -100 or target */}
              <text
                x={x + cellW / 2 - 1}
                y={startY + cellH + 16}
                textAnchor="middle"
                fill={isMasked ? '#64748b' : '#34d399'}
                fontSize="7"
                fontFamily="monospace"
              >
                {isMasked ? 'label: -100' : 'label: target'}
              </text>
            </g>
          )
        })}

        {/* Divider line between prompt and response */}
        <line
          x1={startX + 10 * cellW - 1}
          y1={startY - 5}
          x2={startX + 10 * cellW - 1}
          y2={startY + cellH + 25}
          stroke="#475569"
          strokeWidth={1}
          strokeDasharray="3,3"
        />
      </svg>
    </div>
  )
}

export function InstructionTuningLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Instruction Tuning (SFT)"
            description="How a dataset of instruction-response pairs transforms a text completer into an instruction follower&mdash;using the same training loop you already know."
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
            In the last lesson, you adapted GPT-2 for one specific task:
            sentiment classification. That required a labeled dataset and a
            new classification head&mdash;one head per task. But ChatGPT does
            not have a separate head for every possible task. How does one
            model follow <strong>any</strong> instruction? This lesson answers
            that question: understand how supervised finetuning (SFT) on
            instruction-response pairs transforms a base model from a text
            completer into an instruction follower, then perform SFT yourself
            in the notebook.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How instruction-tuned models (like ChatGPT) differ from base models (like GPT-2)',
              'Instruction dataset format: instruction/response pairs with system/user/assistant roles',
              'Chat templates and special tokens as structural delimiters',
              'SFT mechanics: same training loop, same loss, different data',
              'Hands-on SFT in the notebook',
              'NOT: RLHF, DPO, or alignment\u2014that\u2019s the next lesson',
              'NOT: LoRA or parameter-efficient finetuning\u2014Lesson 4',
              'NOT: building production-quality instruction datasets',
              'NOT: evaluation beyond qualitative observation',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook — The Before/After (Section 2 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Before/After"
            subtitle="Same architecture, radically different behavior"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Send the same prompt to two models. Both are GPT-style
              transformers. Both predict the next token. Both use the same
              architecture you built from scratch.
            </p>

            <ComparisonRow
              left={{
                title: 'Base Model (GPT-2)',
                color: 'amber',
                items: [
                  'Prompt: "Write a haiku about machine learning"',
                  'Output: "Write a haiku about machine learning. Haiku is a traditional form of Japanese poetry that has been used for centuries to..."',
                  'Continues the text as if completing a document',
                ],
              }}
              right={{
                title: 'Instruction-Tuned Model',
                color: 'emerald',
                items: [
                  'Prompt: "Write a haiku about machine learning"',
                  'Output: "Silicon dreams awake / Patterns learned from endless data / Wisdom without soul"',
                  'Follows the instruction and produces a haiku',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The base model does not understand it is being asked to{' '}
              <strong>do</strong> something. It sees text and predicts what
              comes next in a document. The instruction-tuned model recognizes
              the prompt as an instruction and generates a response.
            </p>

            <GradientCard title="Prediction Checkpoint" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  Based on the Finetuning for Classification lesson, you might
                  guess: a new head was added and trained for instruction
                  following. Some kind of &ldquo;instruction head&rdquo;
                  replacing{' '}
                  <code className="text-xs">lm_head</code>.
                </p>
                <p>
                  <strong>That guess is wrong.</strong> No new head. No new
                  architecture. The answer is simpler than you think.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Weights, Different Behavior">
            The instruction-tuned model started as a base model just like
            GPT-2. Same architecture, same pretraining. Something changed
            it&mdash;but it was not a new head or a new loss function.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain — What Base Models Actually Do (Section 3 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Base Models Actually Do"
            subtitle="The knowledge is already there"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A base model is a text completer. It predicts the next token given
              context. That is all it does. But this does not mean it lacks
              knowledge.
            </p>

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3 text-sm text-muted-foreground">
              <p>
                <strong>Prompt 1:</strong>{' '}
                <code className="text-xs">&quot;The capital of France is&quot;</code>
              </p>
              <p>
                <strong>Base model output:</strong>{' '}
                <code className="text-xs">&quot;Paris. It is located on the banks of the Seine...&quot;</code>
              </p>
              <p className="border-t border-border pt-3">
                <strong>Prompt 2:</strong>{' '}
                <code className="text-xs">&quot;What is the capital of France?&quot;</code>
              </p>
              <p>
                <strong>Base model output:</strong>{' '}
                <code className="text-xs">&quot;What is the capital of France? This question is commonly asked in geography classes. The answer, of course, is...&quot;</code>
              </p>
            </div>

            <p className="text-muted-foreground">
              The base model <strong>knows</strong> that Paris is the capital of
              France. Given the right prompt structure, it completes with the
              correct answer. But given a question, it does not answer&mdash;it
              continues the document. It treats the question as text that would
              appear in a textbook or web page, and predicts what comes next in
              that context.
            </p>

            <p className="text-muted-foreground">
              The base model has the <strong>knowledge</strong> but not the{' '}
              <strong>behavior</strong>. It does not understand that a question
              expects a direct answer. It understands that text continues with
              more text.
            </p>

            <GradientCard title="The Central Insight" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>SFT teaches format, not knowledge.</strong>
                </p>
                <p>
                  Think of the base model as a brilliant expert who only speaks
                  in monologue. They know everything&mdash;every fact, every
                  concept. But they cannot hold a conversation. Ask them a
                  question and they continue lecturing.
                </p>
                <p>
                  SFT does not give the expert new knowledge. It teaches them
                  conversational form&mdash;that a question expects an answer,
                  that an instruction expects a response.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Knowledge vs Behavior">
            The &ldquo;capital of France&rdquo; example is the key evidence.
            The model already has the knowledge from pretraining. It just
            expresses it as text completion rather than as a direct answer.
            SFT changes how the model expresses what it already knows.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain — The Instruction Dataset (Section 4 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Instruction Dataset"
            subtitle="What SFT training data looks like"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SFT data is simple: pairs of instructions and responses. The model
              learns a pattern&mdash;when you see text structured as an
              instruction, produce text structured as a response.
            </p>

            <CodeBlock
              code={`# A single instruction-response pair
{
  "instruction": "Explain why the sky is blue in simple terms.",
  "response": "The sky appears blue because of Rayleigh scattering. ..."
}

# More examples from an instruction dataset:
{
  "instruction": "Write a Python function to check if a number is prime.",
  "response": "def is_prime(n):\\n    if n < 2:\\n        return False\\n    ..."
}

{
  "instruction": "Summarize the plot of Romeo and Juliet in two sentences.",
  "response": "Two young lovers from feuding families fall in love ..."
}

{
  "instruction": "What is the difference between a list and a tuple in Python?",
  "response": "A list is mutable — you can add, remove, and change ..."
}`}
              language="python"
              filename="instruction_dataset.jsonl"
            />

            <p className="text-muted-foreground">
              The diversity matters. The dataset includes factual questions,
              creative tasks, coding tasks, summarization&mdash;hundreds of
              different task types. The model does not learn to perform any
              single task. It learns the{' '}
              <strong>meta-pattern</strong>: instruction in, response out.
            </p>

            <p className="text-muted-foreground">
              <strong>Dataset size is surprisingly small.</strong> Pretraining
              uses billions of tokens. SFT? Stanford&rsquo;s Alpaca used 52K
              instruction examples. The LIMA paper showed that 1,000
              carefully curated examples can produce competitive results. The
              entire behavioral transformation from &ldquo;text completer&rdquo;
              to &ldquo;instruction follower&rdquo; happens with remarkably
              little data.
            </p>

            <p className="text-muted-foreground">
              This asymmetry reinforces the central insight: pretraining must
              teach the model everything about language, facts, reasoning, and
              code. That requires enormous data. SFT only teaches a new{' '}
              <strong>format</strong> for expressing what the model already
              knows. Format is a much simpler pattern than knowledge.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Data Quality > Quantity">
            LIMA showed 1,000 high-quality examples can match datasets 50x
            larger. For SFT, a few thousand carefully curated examples
            outperform millions of low-quality ones. The model already has
            the knowledge&mdash;it only needs clean examples of the format.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1 — Predict and Verify (Section 5 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: What Loss Function?" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You have seen the instruction dataset. Now think about
                training. SFT uses a special &ldquo;instruction-following
                loss function,&rdquo; right?
              </p>
              <p>
                <strong>
                  What loss function do you think SFT uses? Why?
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Cross-entropy on next-token prediction.</strong>{' '}
                    The exact same loss from pretraining. The model is still
                    generating text token by token&mdash;the response tokens
                    are the targets. No new loss function. No new objective.
                    The only change is the data.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain — SFT Training Mechanics (Section 6 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SFT Training Mechanics"
            subtitle="Same heartbeat, different data"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the SFT training loop next to the pretraining loop you
              wrote in the Pretraining on Real Text lesson. Look carefully.
            </p>

            <ComparisonRow
              left={{
                title: 'Pretraining Loop',
                color: 'blue',
                items: [
                  '1. Sample a batch of web text chunks',
                  '2. Forward pass through GPT',
                  '3. loss = F.cross_entropy(logits, targets)',
                  '4. loss.backward()',
                  '5. optimizer.step()',
                ],
              }}
              right={{
                title: 'SFT Loop',
                color: 'emerald',
                items: [
                  '1. Sample a batch of formatted instruction/response pairs',
                  '2. Forward pass through GPT',
                  '3. loss = F.cross_entropy(logits, targets)',
                  '4. loss.backward()',
                  '5. optimizer.step()',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Steps 2 through 5 are identical. The <strong>only</strong>{' '}
              difference is step 1: what data goes in. Instead of random web
              text chunks, each training example is a formatted
              instruction-response pair. Same loop. Same loss. Same
              optimizer.{' '}
              <strong>Same heartbeat.</strong>
            </p>

            <SftPipelineDiagram />

            <p className="text-muted-foreground">
              And unlike classification finetuning from the previous
              lesson, <strong>no new head</strong>. The model keeps its
              original{' '}
              <code className="text-xs">lm_head</code>&mdash;the same linear
              layer that projects hidden states to vocabulary logits. The
              model is still predicting the next token. It is just learning
              to predict <strong>instruction-appropriate</strong> next tokens.
            </p>

            <SectionHeader
              title="Loss Masking"
              subtitle="The one genuinely new mechanical concept"
            />

            <p className="text-muted-foreground">
              There is one important detail that differs from pretraining.
              During SFT, loss is computed only on the{' '}
              <strong>response tokens</strong>, not the instruction tokens.
              The model should learn to <strong>generate</strong> responses,
              not to predict instruction tokens (those are given as input).
            </p>

            <LossMaskingDiagram />

            <p className="text-muted-foreground">
              Instruction tokens get a label of{' '}
              <code className="text-xs">-100</code>, which PyTorch&rsquo;s{' '}
              <code className="text-xs">CrossEntropyLoss</code> ignores by
              default. Response tokens get their actual next-token targets.
              This is called <strong>prompt masking</strong> or{' '}
              <strong>loss masking</strong>.
            </p>

            <CodeBlock
              code={`# Simplified loss masking
# tokens:  [<|im_start|>, user, \\n, Write, a, haiku, <|im_end|>, <|im_start|>, assistant, \\n, Silicon, dreams, awake, <|im_end|>]
# labels:  [-100,         -100, -100, -100, -100, -100,  -100,     -100,         -100,      -100, dreams,  awake,  ...,   <|im_end|>]
#                                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                                                                  Only these contribute to loss

labels = tokens.clone()
labels[:prompt_length] = -100  # Mask the instruction tokens

loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)`}
              language="python"
              filename="loss_masking.py"
            />

            <p className="text-muted-foreground">
              Why mask? Without masking, the model would spend training
              signal learning to predict instruction tokens&mdash;a waste of
              capacity. The instruction is the input. The response is what the
              model needs to learn to produce. Loss masking focuses all
              training signal on the response.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Loop, Third Time">
            You have now written the same training loop three times:
            pretraining, classification finetuning, and SFT. Forward, loss,
            backward, step. The heartbeat never changes. Only the data and
            what you compute loss on changes.
          </InsightBlock>
          <WarningBlock title="Loss Masking Matters">
            Without loss masking, the model wastes training signal predicting
            instruction tokens it already has. This can slow convergence and
            degrade response quality. Always mask the prompt.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain — Chat Templates and Special Tokens (Section 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Chat Templates and Special Tokens"
            subtitle="How the model knows where the instruction ends"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The model needs to know where the instruction ends and the
              response should begin. In raw text, there is no boundary.
              The solution: <strong>special tokens</strong> that act as
              structural delimiters.
            </p>

            <CodeBlock
              code={`# ChatML format — one of many chat template standards
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a haiku about machine learning.<|im_end|>
<|im_start|>assistant
Silicon dreams awake
Patterns learned from endless data
Wisdom without soul<|im_end|>`}
              language="text"
              filename="chat_template.txt"
            />

            <p className="text-muted-foreground">
              These are <strong>real tokens</strong> in the vocabulary. The
              model learns that after seeing{' '}
              <code className="text-xs">{'<|im_start|>assistant\\n'}</code>,
              it should generate a response. The special tokens have no
              pretrained meaning&mdash;they did not exist in the pretraining
              data. They acquire meaning entirely from SFT.
            </p>

            <p className="text-muted-foreground">
              Remember from the Tokenization lesson: the vocabulary is just
              a lookup table mapping tokens to integer IDs. Special tokens
              are new entries added to this table. The embedding matrix gets
              extended with new rows for these tokens, initialized randomly.
              Through SFT, the model learns what they mean.
            </p>

            <p className="text-muted-foreground">
              <strong>Multiple template formats exist.</strong> ChatML (shown
              above), Llama format, Alpaca format&mdash;each model family uses
              its own. The specific format matters because the model was
              trained to recognize specific delimiters.
            </p>

            <CodeBlock
              code={`# Different models use different templates:

# ChatML (OpenAI-style)
"<|im_start|>user\\nHello<|im_end|>\\n<|im_start|>assistant\\n"

# Llama format
"[INST] Hello [/INST]"

# Alpaca format
"### Instruction:\\nHello\\n\\n### Response:\\n"`}
              language="text"
              filename="template_formats.txt"
            />

            <p className="text-muted-foreground">
              Multi-turn conversations are just longer templates with
              alternating user/assistant blocks. Same structure, repeated.
              The template format handles any number of turns.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Tokens All the Way Down">
            Special tokens are not magic. They are integer IDs in the
            vocabulary, just like every other token. The model learns their
            meaning from the SFT training data, where they consistently
            appear as structural boundaries.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 2 — Transfer Question (Section 8 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Wrong Template" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You train a base model with SFT using the ChatML template.
                At inference time, you accidentally use Llama&rsquo;s template
                format instead. What happens and why?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-3">
                  <p>
                    <strong>Degraded or incoherent responses.</strong> The
                    model was trained to recognize ChatML&rsquo;s special
                    tokens ({' '}
                    <code className="text-xs">{'<|im_start|>'}</code>,{' '}
                    <code className="text-xs">{'<|im_end|>'}</code>) as
                    structural boundaries. Llama&rsquo;s tokens (
                    <code className="text-xs">[INST]</code>,{' '}
                    <code className="text-xs">[/INST]</code>) are unknown or
                    have different learned associations.
                  </p>

                  <div className="bg-muted/50 rounded-lg p-3 space-y-2 text-xs font-mono">
                    <p className="font-sans text-xs font-semibold text-emerald-400">
                      Correct template (ChatML):
                    </p>
                    <p className="text-muted-foreground whitespace-pre-wrap">{'<|im_start|>user\nExplain gravity briefly.<|im_end|>\n<|im_start|>assistant'}</p>
                    <p className="text-emerald-400 whitespace-pre-wrap">{'\u2192 "Gravity is the force that attracts objects with mass toward each other..."'}</p>

                    <p className="font-sans text-xs font-semibold text-rose-400 pt-2 border-t border-border">
                      Wrong template (Llama format on ChatML-trained model):
                    </p>
                    <p className="text-muted-foreground whitespace-pre-wrap">{'[INST] Explain gravity briefly. [/INST]'}</p>
                    <p className="text-rose-400 whitespace-pre-wrap">{'\u2192 "[INST] Explain gravity briefly. [/INST] [INST] What is the meaning of life? [/INST] [INST]..."'}</p>
                  </div>

                  <p>
                    The model cannot find the structural boundary that signals
                    &ldquo;start generating a response here.&rdquo; Instead
                    of answering, it continues the pattern it sees&mdash;more
                    tokens that look like template fragments, as if
                    completing a document full of template examples.
                  </p>
                  <p>
                    This is why chat templates are <strong>functional
                    structure</strong>, not cosmetic formatting.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Explore — Notebook (Section 9 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="Perform SFT on a small instruction dataset"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The lesson showed you the concepts. Now implement SFT. The
              notebook walks you through the complete pipeline: exploring an
              instruction dataset, implementing chat template formatting,
              tokenizing with special tokens, implementing loss masking,
              training, and observing the behavioral shift from text completion
              to instruction following.
            </p>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Perform supervised finetuning on a small instruction dataset
                  and observe the before/after behavioral change.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-4-2-instruction-tuning.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes: exploring an instruction dataset
                  (Alpaca format), implementing chat template formatting with
                  special tokens, tokenizing formatted examples, implementing
                  loss masking (labels with -100 for prompt tokens), running
                  SFT training for a small number of steps, and generating
                  from the model before and after SFT to observe the
                  behavioral shift. Use a GPU runtime in
                  Colab&mdash;finetuning requires gradient computation
                  through the full model.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Notebook Exercises">
            <ul className="space-y-2 text-sm">
              <li>{'\u2022'} Explore an instruction dataset (examine entries)</li>
              <li>{'\u2022'} Implement chat template formatting</li>
              <li>{'\u2022'} Tokenize with special tokens and inspect token IDs</li>
              <li>{'\u2022'} Implement loss masking (labels = -100 for prompt)</li>
              <li>{'\u2022'} Run SFT training</li>
              <li>{'\u2022'} Compare base vs SFT model on several prompts</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Elaborate — Why SFT Works With So Little Data (Section 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Does SFT Work With So Little Data?"
            subtitle="You already know the answer"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have trained models three times now. Each time, the model
              learned what its data taught it:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Pretraining data</strong> was web text, so the model
                learned to complete web text.
              </li>
              <li>
                <strong>Classification data</strong> was labeled sentiment, so
                the model learned to output sentiment labels.
              </li>
              <li>
                <strong>Instruction data</strong> is instruction-response
                pairs, so&hellip; of course the model learns to produce
                responses to instructions.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Same mechanism every time. The only question is why SFT needs
              so little data&mdash;and you already know the answer from
              classification finetuning. The classification head had only
              1,536 parameters. It learned to classify with thousands of
              examples because the backbone already extracted good features.
              SFT is the same principle at the behavior level: the model
              already has vast knowledge from pretraining. Teaching it a new{' '}
              <strong>format</strong> for expressing that knowledge is a far
              simpler task than teaching the knowledge itself.
            </p>

            <p className="text-muted-foreground">
              <strong>Catastrophic forgetting is less risky here</strong>{' '}
              than with classification finetuning. Classification finetuning
              trains on a narrow task (sentiment labels) that is structurally
              different from natural language. SFT trains on
              instruction-response pairs that are still natural
              language&mdash;the model does not forget how to generate text;
              it learns a new format for generating text. But excessive SFT
              or low-quality data can still cause problems.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Principle, New Level">
            Classification finetuning: the backbone has good features, so
            the head learns quickly. SFT: the model has vast knowledge, so
            the format shift happens quickly. Both succeed because
            pretraining already did the heavy lifting.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Elaborate — Classification vs SFT: Full Picture (Section 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Classification Finetuning vs SFT"
            subtitle="Two forms of adaptation, one underlying technique"
          />
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'Classification Finetuning',
                color: 'blue',
                items: [
                  'Adds a NEW head (nn.Linear)',
                  'Narrow task: sentiment, topic, etc.',
                  'Data: (text, label) pairs',
                  'Architecture CHANGES (lm_head replaced)',
                  'Trains the head on a frozen backbone',
                  'Changes WHAT the model outputs (class labels)',
                ],
              }}
              right={{
                title: 'Supervised Finetuning (SFT)',
                color: 'emerald',
                items: [
                  'Keeps the ORIGINAL lm_head',
                  'Broad task: follow any instruction',
                  'Data: (instruction, response) pairs',
                  'Architecture UNCHANGED',
                  'Trains some or all of the original model',
                  'Changes HOW the model uses its existing output',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Both are supervised learning on a pretrained backbone. The
              distinction: classification finetuning changes{' '}
              <strong>what</strong> the model outputs (class labels instead
              of tokens). SFT changes <strong>how</strong> the model uses its
              existing output mechanism (instruction-appropriate tokens
              instead of document-continuation tokens).
            </p>

            <p className="text-muted-foreground">
              In Finetuning for Classification, the mental model was
              &ldquo;a pretrained transformer is a text feature
              extractor.&rdquo; That framing emphasizes backbone-as-features
              + separate-head. SFT breaks that pattern: there is no separate
              head. The model keeps its original{' '}
              <code className="text-xs">lm_head</code> because the task{' '}
              <strong>is</strong> still next-token prediction&mdash;just on
              differently formatted data.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a &ldquo;Feature Extractor&rdquo; Here">
            The &ldquo;feature extractor + new head&rdquo; mental model
            from Lesson 1 does not apply to SFT. In SFT, the model keeps
            its original output mechanism. The adaptation is in the data
            and the weights, not the architecture.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Summary (Section 12 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline: 'SFT teaches format, not knowledge.',
                description:
                  'The base model already has vast knowledge from pretraining. SFT on instruction-response pairs teaches it to express that knowledge in a conversational, instruction-following format.',
              },
              {
                headline: 'No new architecture, no new loss function.',
                description:
                  'Same cross-entropy on next-token prediction. Same training loop. The only change is the data\u2014formatted instruction-response pairs instead of web text.',
              },
              {
                headline: 'Loss masking focuses training on responses.',
                description:
                  'Instruction tokens are masked (label = -100) so the model learns to generate responses, not to predict the prompt. This is the one genuinely new mechanical concept.',
              },
              {
                headline: 'Chat templates are functional, not cosmetic.',
                description:
                  'Special tokens are structural delimiters the model learns to recognize. Using the wrong template degrades output because the model cannot find the boundaries.',
              },
              {
                headline: 'SFT is surprisingly data-efficient.',
                description:
                  'Thousands of examples can transform behavior because format is simpler than knowledge. Pretraining does the heavy lifting; SFT reshapes the output.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Next Step (Section 13 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">
              What comes next
            </p>
            <p className="text-sm text-muted-foreground">
              SFT produces a model that follows instructions. But following
              instructions is not enough. The model can be helpful but also
              harmful, verbose, sycophantic, or confidently wrong. It learned
              the <strong>format</strong> of being helpful, but it has no
              training signal for what &ldquo;helpful&rdquo; actually means.
              Next: RLHF and alignment&mdash;training the model to produce
              responses that humans actually prefer.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Complete the notebook: explore the instruction dataset, implement chat templates and loss masking, run SFT, and compare base vs instruction-tuned outputs. Then review your session."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
