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
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * In-Context Learning
 *
 * First lesson in Module 5.2 (Reasoning & In-Context Learning).
 * STRETCH lesson — introduces a paradigm that challenges the student's
 * understanding of what "learning" means.
 * Inline SVG diagram (ICL Attention Pattern).
 *
 * Cognitive load: 2 genuinely new concepts:
 *   1. In-context learning as gradient-free task learning from examples in prompt
 *   2. Attention-based mechanism that explains why ICL works
 *
 * Core concepts at DEVELOPED:
 * - In-context learning (the phenomenon)
 * - Attention as ICL mechanism
 *
 * Concepts at INTRODUCED:
 * - ICL limitations (ordering, fragility)
 *
 * Concepts at MENTIONED:
 * - "Transformers as learning algorithms" theoretical perspective
 *
 * EXPLICITLY NOT COVERED:
 * - Systematic prompt engineering techniques (Lesson 2)
 * - Chain-of-thought or step-by-step reasoning (Lesson 3)
 * - Retrieval-augmented generation (Lesson 2)
 * - Implementing ICL in code beyond notebook exercises
 * - Theoretical framework for why ICL emerges during pretraining
 * - Comparing ICL performance to finetuning quantitatively
 * - Role prompting or system prompts (Lesson 2)
 *
 * Previous: Evaluating LLMs (Module 5.1, Lesson 4)
 * Next: Prompt Engineering (Module 5.2, Lesson 2)
 */

// ---------------------------------------------------------------------------
// Inline SVG: ICL Attention Pattern Diagram
// Shows how a test input's query vectors attend to example inputs and outputs
// in a few-shot prompt. Highlights the retrieval pattern.
// ---------------------------------------------------------------------------

function ICLAttentionDiagram() {
  const svgW = 600
  const svgH = 340

  // Token positions in the prompt
  const tokens = [
    { label: '"amazing"', type: 'example-input' as const, x: 40 },
    { label: '→', type: 'arrow' as const, x: 115 },
    { label: 'Positive', type: 'example-output' as const, x: 145 },
    { label: '"terrible"', type: 'example-input' as const, x: 230 },
    { label: '→', type: 'arrow' as const, x: 310 },
    { label: 'Negative', type: 'example-output' as const, x: 340 },
    { label: '"slow"', type: 'test-input' as const, x: 460 },
    { label: '???', type: 'test-output' as const, x: 540 },
  ]

  const tokenY = 200
  const queryY = 260

  const colorMap = {
    'example-input': '#6366f1',
    'example-output': '#22c55e',
    arrow: '#64748b',
    'test-input': '#f59e0b',
    'test-output': '#ef4444',
  }

  // Attention lines from test input to example positions
  const attentionLines = [
    { fromX: 480, toX: 60, toLabel: '"amazing"', weight: 0.25, color: '#6366f1' },
    { fromX: 480, toX: 165, toLabel: 'Positive', weight: 0.15, color: '#22c55e' },
    { fromX: 480, toX: 255, toLabel: '"terrible"', weight: 0.20, color: '#6366f1' },
    { fromX: 480, toX: 360, toLabel: 'Negative', weight: 0.40, color: '#22c55e' },
  ]

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Title */}
        <text
          x={svgW / 2}
          y={20}
          textAnchor="middle"
          fill="#e2e8f0"
          fontSize="13"
          fontWeight="600"
        >
          Attention Pattern During In-Context Learning
        </text>
        <text
          x={svgW / 2}
          y={38}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="9"
          fontStyle="italic"
        >
          The test input&apos;s queries attend to example inputs (matching) and outputs (retrieving the answer pattern)
        </text>

        {/* Section labels */}
        <text x={140} y={70} textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="500">
          Example 1
        </text>
        <line x1={60} y1={76} x2={220} y2={76} stroke="#334155" strokeWidth={1} />

        <text x={330} y={70} textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="500">
          Example 2
        </text>
        <line x1={250} y1={76} x2={410} y2={76} stroke="#334155" strokeWidth={1} />

        <text x={500} y={70} textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="500">
          Test
        </text>
        <line x1={440} y1={76} x2={560} y2={76} stroke="#f59e0b" strokeWidth={1} opacity={0.4} />

        {/* Attention lines (drawn first so tokens render on top) */}
        {attentionLines.map((line, i) => {
          const opacity = 0.3 + line.weight * 1.2
          const strokeW = 1 + line.weight * 4
          return (
            <g key={i}>
              <line
                x1={line.fromX}
                y1={tokenY + 18}
                x2={line.toX}
                y2={tokenY + 18}
                stroke={line.color}
                strokeWidth={strokeW}
                opacity={opacity}
                strokeDasharray={line.weight > 0.25 ? 'none' : '4 3'}
              />
              {/* Attention weight label */}
              <text
                x={(line.fromX + line.toX) / 2}
                y={tokenY + 14 + (i % 2 === 0 ? -25 : 35)}
                textAnchor="middle"
                fill={line.color}
                fontSize="8"
                opacity={0.8}
              >
                {(line.weight * 100).toFixed(0)}%
              </text>
            </g>
          )
        })}

        {/* Token boxes */}
        {tokens.map((token, i) => {
          const isArrow = token.type === 'arrow'
          const boxW = isArrow ? 20 : token.label.length * 8 + 16
          const boxH = 28
          const boxX = token.x - boxW / 2
          const boxY = tokenY - boxH / 2 + 18

          return (
            <g key={i}>
              {!isArrow && (
                <rect
                  x={boxX}
                  y={boxY}
                  width={boxW}
                  height={boxH}
                  rx={4}
                  fill={colorMap[token.type]}
                  opacity={0.15}
                  stroke={colorMap[token.type]}
                  strokeWidth={1}
                />
              )}
              <text
                x={token.x}
                y={tokenY + 23}
                textAnchor="middle"
                fill={colorMap[token.type]}
                fontSize={isArrow ? '14' : '10'}
                fontWeight={isArrow ? '400' : '600'}
                fontFamily="monospace"
              >
                {token.label}
              </text>
            </g>
          )
        })}

        {/* Query arrow from test input */}
        <text
          x={480}
          y={queryY + 15}
          textAnchor="middle"
          fill="#f59e0b"
          fontSize="9"
          fontWeight="500"
        >
          Query: &quot;What output follows a negative-sentiment input?&quot;
        </text>

        {/* Legend */}
        <g transform="translate(30, 290)">
          <rect x={0} y={0} width={10} height={10} rx={2} fill="#6366f1" opacity={0.6} />
          <text x={16} y={9} fill="#94a3b8" fontSize="9">Input matching (Q&middot;K)</text>

          <rect x={160} y={0} width={10} height={10} rx={2} fill="#22c55e" opacity={0.6} />
          <text x={176} y={9} fill="#94a3b8" fontSize="9">Output retrieval (V blending)</text>

          <rect x={350} y={0} width={10} height={10} rx={2} fill="#f59e0b" opacity={0.6} />
          <text x={366} y={9} fill="#94a3b8" fontSize="9">Test input</text>
        </g>

        {/* Key insight annotation */}
        <text
          x={svgW / 2}
          y={svgH - 8}
          textAnchor="middle"
          fill="#64748b"
          fontSize="8"
          fontStyle="italic"
        >
          Thicker lines = stronger attention. The test input attends most to the structurally similar example output (&quot;Negative&quot;).
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function InContextLearningLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="In-Context Learning"
            description="A model trained only on next-token prediction can learn new tasks from examples in the prompt&mdash;without any weight update. This should not work. It does. The mechanism is attention."
            category="Reasoning & ICL"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints (Section 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            This lesson explains how transformers can learn new tasks from
            examples placed in the prompt, without any weight update, and
            identifies attention as the specific mechanism that makes this
            possible. You will see why this is surprising, how it works
            mechanically, and where it breaks down.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'What in-context learning is (zero-shot, few-shot prompting)',
              'Why it is surprising (no weight updates, not in the training objective)',
              'How attention mechanistically enables ICL',
              'Key limitations: context window, ordering sensitivity, fragility',
              'The GPT-3 discovery as historical context',
              'NOT: systematic prompt engineering techniques (next lesson)',
              'NOT: chain-of-thought or step-by-step reasoning (Lesson 3)',
              'NOT: retrieval-augmented generation',
              'NOT: implementing ICL from scratch in code',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="New Module">
            Module 5.1 was about controlling model behavior&mdash;alignment,
            red teaming, evaluation. This module shifts perspective: what can
            models actually <em>do</em>?
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap (Section 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What You Already Know"
            subtitle="Two concepts that make this lesson possible"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the attention lessons, you built attention from scratch. The
              defining insight was that <strong>attention weights are computed
              from the input</strong>&mdash;not fixed parameters. Every new
              input produces new weights. The input decides what matters.
            </p>
            <p className="text-muted-foreground">
              In Finetuning for Classification, you classified
              sentiment by adding a classification head and training on labeled
              data. That required gradient descent, a labeled dataset, and
              weight updates. The model changed to perform the task.
            </p>
            <p className="text-muted-foreground">
              Hold both of those in mind. This lesson is about what happens
              when you skip the second one entirely.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Ideas, One Collision">
            These two ideas&mdash;attention as data-dependent computation
            and finetuning as weight updates&mdash;are about to collide in
            a way you do not expect.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: Demo + Puzzle (Section 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Puzzle"
            subtitle="Same model, no training, new task"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know how to classify sentiment&mdash;you did it in
              Finetuning for Classification by training a model on labeled
              data. That required a dataset, a training loop, and changing
              the model&rsquo;s weights. Now watch this:
            </p>

            <CodeBlock
              code={`Review: "This movie was amazing" -> Positive
Review: "The food was terrible" -> Negative
Review: "The scenery was breathtaking" -> Positive
Review: "The service was slow" -> ???`}
              language="text"
              filename="few-shot-prompt.txt"
            />

            <p className="text-muted-foreground">
              The model outputs <strong>&ldquo;Negative.&rdquo;</strong> No
              training. No weight updates. No classification head. The same
              frozen model that generates Shakespeare can now classify
              sentiment&mdash;simply because of what is in the prompt.
            </p>

            <GradientCard title="The Question" color="orange">
              <p className="text-sm">
                The model&rsquo;s parameters did not change. No optimizer
                ran. No gradients were computed. And yet it performed a task
                it was never explicitly trained to do. <strong>How?</strong>
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Weight Updates">
            Your instinct from Series 1-3 is that &ldquo;learning&rdquo;
            means gradient descent and weight updates. ICL challenges that
            definition. No parameters change. The &ldquo;learning&rdquo;
            happens somewhere else entirely.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain Part 1: The GPT-3 Discovery (Section 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The GPT-3 Discovery"
            subtitle="A base model that learns tasks from examples"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In 2020, Brown et al. showed that a large enough language model
              can perform tasks it was never explicitly trained on, simply by
              conditioning on examples in the prompt. They called
              it <strong>in-context learning</strong> (ICL), and it was the
              central finding of the GPT-3 paper. Two key terms:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Zero-shot:</strong> Task instruction only, no
                examples. &ldquo;Classify the sentiment of this
                review:&rdquo;
              </li>
              <li>
                <strong>Few-shot:</strong> Task instruction + examples (the
                demo above). The model sees input-output pairs before the
                test input.
              </li>
            </ul>

            <p className="text-muted-foreground">
              The critical detail: <strong>GPT-3 was a base
              model.</strong> No SFT, no RLHF. ICL is a property of
              pretraining, not instruction tuning. Remember &ldquo;SFT
              teaches format, not knowledge&rdquo;? ICL shows that the base
              model already has remarkable capability&mdash;SFT just makes it
              more accessible through conversational formatting.
            </p>

            <ComparisonRow
              left={{
                title: 'Finetuning Approach',
                color: 'amber',
                items: [
                  'Add a classification head',
                  'Collect labeled training data',
                  'Run gradient descent',
                  'Model weights change',
                  'Task-specific model',
                ],
              }}
              right={{
                title: 'In-Context Learning',
                color: 'blue',
                items: [
                  'Same model, no modifications',
                  'Put examples in the prompt',
                  'No gradients computed',
                  'Model weights frozen',
                  'General-purpose model',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Base Model Capability">
            ICL does not require instruction tuning. The base model&mdash;trained
            only on next-token prediction&mdash;can already learn from
            examples. SFT makes it easier to use, but the capability is
            already there.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain Part 2: Why It Works (Section 5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why It Works: Attention as the Mechanism"
            subtitle="The same mechanism you already know, operating on a longer context"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We said no weights change. So where does the task-specific
              behavior come from? It comes from the same mechanism you
              already know: <strong>attention.</strong>
            </p>

            <p className="text-muted-foreground">
              Walk through what happens when the model processes the test
              input in a few-shot prompt:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                The prompt contains examples (input-output pairs) followed by
                a test input.
              </li>
              <li>
                When the model processes the test input, its <strong>query
                vectors</strong> are computed from the test input&rsquo;s
                tokens.
              </li>
              <li>
                These query vectors compute dot products with
                the <strong>key vectors</strong> of <em>all</em> tokens in
                the context&mdash;including the example inputs and outputs.
              </li>
              <li>
                If <InlineMath math="W_Q" /> and <InlineMath math="W_K" /> have
                learned to project inputs and outputs into a shared relevance
                space, the test input&rsquo;s queries will produce high scores
                against the example inputs&rsquo; keys (because they are
                structurally similar).
              </li>
              <li>
                The attention weights then blend the <strong>value
                vectors</strong> of the nearby tokens&mdash;including the
                example <em>output</em> tokens.
              </li>
              <li>
                The blended representation carries information about what
                output should follow this type of input.
              </li>
            </ul>

            <ICLAttentionDiagram />
            <p className="text-muted-foreground text-sm text-center mt-2">
              The test input &ldquo;slow&rdquo; attends to example inputs
              (structural matching via Q&middot;K) and example outputs
              (answer retrieval via V blending). Thicker lines indicate
              stronger attention.
            </p>

            <GradientCard title="The &ldquo;Of Course&rdquo; Moment" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  You already knew this. Attention weights are
                  data-dependent. If examples are in the context, they are
                  part of the data. <strong>Of course</strong> the model
                  attends to them. <strong>Of course</strong> similar inputs
                  match. The mechanism is not new. You just had not thought
                  about what happens when the context contains examples of a
                  task.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground text-sm">
              Some researchers have taken this further, showing that
              transformer attention can implement something resembling
              gradient descent within the forward pass (von Oswald et al.,
              2023)&mdash;but that theoretical framework is beyond our scope
              here. What matters for us is the practical mechanism.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <p className="text-muted-foreground text-sm mb-2">
                Connect to the formula you already know:
              </p>
              <div className="text-center">
                <InlineMath math="\text{output} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V" />
              </div>
              <p className="text-muted-foreground text-sm mt-2">
                The <InlineMath math="Q" /> comes from the test input.
                The <InlineMath math="K" /> and <InlineMath math="V" /> come
                from the <em>entire context</em>, including examples.
                The formula has not changed. The context has.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Prompt Is a Program">
            Each prompt is a different program. The examples are data, the
            format is the instruction set, and attention is the interpreter
            that executes the program at inference time.
          </InsightBlock>
          <TipBlock title="Software Analogy">
            If you are a software engineer, think of the prompt as
            configuration, not code. The model&rsquo;s weights are the
            compiled binary. You do not recompile to change
            behavior&mdash;you change the config file (the prompt).
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Check 1: Predict-and-Verify (Section 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: The Prompt Is the Program" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                Same model. Same weights. Same test input:
                &ldquo;The weather is beautiful.&rdquo;
              </p>

              <div className="space-y-2">
                <p><strong>Prompt A</strong> has 3 sentiment examples:</p>
                <p className="ml-4 text-muted-foreground font-mono text-xs">
                  &ldquo;Great movie&rdquo; -&gt; Positive, &ldquo;Bad food&rdquo; -&gt; Negative, &ldquo;Nice park&rdquo; -&gt; Positive
                </p>

                <p><strong>Prompt B</strong> has 3 translation examples:</p>
                <p className="ml-4 text-muted-foreground font-mono text-xs">
                  &ldquo;Hello&rdquo; -&gt; &ldquo;Bonjour&rdquo;, &ldquo;Goodbye&rdquo; -&gt; &ldquo;Au revoir&rdquo;, &ldquo;Thank you&rdquo; -&gt; &ldquo;Merci&rdquo;
                </p>
              </div>

              <p>
                <strong>What will the model output for &ldquo;The weather is
                beautiful&rdquo; in each prompt?</strong>
              </p>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p>
                    <strong>Prompt A:</strong> &ldquo;Positive&rdquo;
                    (sentiment label).
                  </p>
                  <p>
                    <strong>Prompt B:</strong> &ldquo;Le temps est
                    beau&rdquo; (French translation).
                  </p>
                  <p className="text-muted-foreground">
                    Same model. Same weights. Same test input. Different
                    examples, different behavior. The prompt is the program.
                  </p>
                  <p className="text-muted-foreground">
                    Trace the mechanism: the <InlineMath math="Q" /> vectors
                    from &ldquo;The weather is beautiful&rdquo; are the same
                    in both prompts (same test input, same weights). But
                    the <InlineMath math="K" /> and <InlineMath math="V" /> vectors
                    are different because the examples are different.
                    Different <InlineMath math="K" /> means different
                    attention weights. Different <InlineMath math="V" /> means
                    different blended output. Different task.
                  </p>
                  <p className="text-muted-foreground">
                    If the weights had been updated by the examples, the
                    model&rsquo;s behavior on an empty prompt would also
                    change. It does not&mdash;the weights are frozen. The
                    &ldquo;learning&rdquo; is in the activations (attention
                    patterns), not the parameters.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          8. Elaborate: What ICL Is and Is Not (Section 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What ICL Is and Is Not"
            subtitle="Powerful but fragile&mdash;defining the boundaries"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The sentiment example was familiar&mdash;you might think the
              model just memorized &ldquo;sentiment classification&rdquo;
              from training data. So here is a harder test.
            </p>

            <GradientCard title="Novel Mapping: Proof Beyond Memorization" color="blue">
              <div className="space-y-2 text-sm">
                <p className="font-mono text-xs">
                  &ldquo;sdag&rdquo; -&gt; &ldquo;happy&rdquo;<br />
                  &ldquo;trel&rdquo; -&gt; &ldquo;sad&rdquo;<br />
                  &ldquo;blix&rdquo; -&gt; &ldquo;angry&rdquo;<br />
                  &ldquo;wump&rdquo; -&gt; ???
                </p>
                <p>
                  The model produces an emotion word, following the pattern.
                  This mapping was <strong>not</strong> in training data.
                  &ldquo;sdag&rdquo; is not a word. There is no internet
                  page mapping it to &ldquo;happy.&rdquo; The model learned
                  the function <em>from the examples alone</em>.
                </p>
                <p className="text-muted-foreground">
                  If ICL were just memorized template matching from training
                  data, it would fail on truly novel mappings. It does not
                  always fail. ICL is more than retrieval.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Ordering Sensitivity: ICL Is Not Comprehension" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Take the same 5 sentiment examples. Shuffle them into
                  different orders. Test accuracy on the same 10 inputs with
                  each ordering.
                </p>
                <p>
                  <strong>Result: accuracy swings by 20-30 percentage
                  points</strong> depending on example order. Same examples,
                  same test inputs, wildly different performance.
                </p>
                <p className="text-muted-foreground">
                  If the model truly &ldquo;understood&rdquo; the task, order
                  would not matter. ICL performance depends on surface
                  features that attention is sensitive to: recency (causal
                  masking means later tokens attend to everything, earlier
                  tokens attend to less), format consistency, and prompt
                  structure.
                </p>
              </div>
            </GradientCard>

            <div className="space-y-4">
              <p className="text-muted-foreground">
                So what is ICL, exactly? It is attention-based computation
                that can generalize to novel patterns within the scope of
                what the attention mechanism can compute in a single forward
                pass. Not memorized retrieval (novel mappings work). Not
                human-like comprehension (ordering matters). Something in
                between.
              </p>

              <p className="text-muted-foreground">
                A natural way to think about ICL is that the model
                &ldquo;reads&rdquo; the examples, &ldquo;figures out&rdquo;
                the pattern, and &ldquo;applies&rdquo; it&mdash;as if it
                understands the task. If that were true, what would happen
                if you flipped the labels? Min et al. (2022)
                showed that ICL performance is sometimes robust to
                randomly <strong>flipping labels</strong> in the examples.
                Give the model examples where &ldquo;great&rdquo; maps to
                &ldquo;Negative&rdquo; and &ldquo;terrible&rdquo; maps to
                &ldquo;Positive,&rdquo; and it still often classifies
                correctly. This suggests the <em>format</em> and
                the <em>distribution of inputs</em> matter more than the
                actual input-output mapping for many tasks. The model picks up
                &ldquo;this is a classification task with these
                categories&rdquo; from the structure, not just &ldquo;input X
                maps to output Y.&rdquo;
              </p>
            </div>

            <GradientCard title="Capability = Vulnerability" color="amber">
              <div className="space-y-2 text-sm">
                <p>
                  In Red Teaming &amp; Adversarial Evaluation, you saw that
                  few-shot jailbreaking works by putting adversarial examples
                  in the prompt. Now you understand the mechanism: the
                  model&rsquo;s attention treats those examples the same way
                  it treats any examples. ICL does not distinguish between
                  helpful and harmful patterns.
                </p>
                <p className="text-muted-foreground">
                  Every capability is also a vulnerability. The same
                  attention mechanism that lets the model learn sentiment
                  classification from 3 examples also lets it learn
                  harmful behaviors from 3 adversarial examples.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Between Retrieval and Comprehension">
            ICL is not memorized pattern matching (it works on novel
            mappings). It is not human-like understanding (ordering
            matters). It is attention-based computation&mdash;powerful but
            constrained by what a single forward pass can compute.
          </InsightBlock>
          <WarningBlock title="&ldquo;More Examples = Better&rdquo;">
            Adding more examples does not always help. Example ordering,
            selection, and format can matter more than quantity. Three
            well-chosen examples can outperform twenty poorly ordered ones.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 2: Transfer Question (Section 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: The Limits of ICL" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A developer puts 50 examples of a classification task into a
                prompt to get better accuracy. The model&rsquo;s context
                window is 4096 tokens, and the 50 examples use 3800 tokens,
                leaving 296 tokens for the test input and output.
              </p>
              <p>
                <strong>What problems might arise?</strong>
              </p>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about the attention mechanism, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <ul className="space-y-2">
                    <li>
                      &bull; <strong>Context window nearly full:</strong> Only
                      296 tokens remain for the test input and output. If the
                      test input is long, it may be truncated.
                    </li>
                    <li>
                      &bull; <strong>Causal masking and position
                      effects:</strong> With causal masking, only the last few
                      tokens of the test input attend to all 50 examples.
                      Earlier test tokens attend to fewer. The attention
                      pattern is asymmetric.
                    </li>
                    <li>
                      &bull; <strong>More is not always better:</strong> With
                      50 examples, ordering and selection effects are
                      amplified. The 20-30 point accuracy swing from ordering
                      is even worse with more examples competing for attention.
                    </li>
                    <li>
                      &bull; <strong>Diminishing returns:</strong> ICL
                      performance typically plateaus after 10-20 examples.
                      The additional 30-40 examples consume context window
                      without proportional benefit.
                    </li>
                  </ul>
                  <p className="text-muted-foreground">
                    The limitations of ICL are fundamentally the limitations
                    of the context window and the attention mechanism. ICL
                    is powerful but bounded by the forward pass.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Practice: Notebook Exercises (Section 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Explore ICL behavior empirically"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Four exercises that explore ICL empirically. Each exercise can
              be completed independently, but they build on each other
              thematically&mdash;from basic observation to systematic
              experimentation.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Zero-Shot vs Few-Shot (Guided)" color="blue">
                <p className="text-sm">
                  Use an LLM API to classify 10 sentiment examples zero-shot
                  and few-shot (3 examples). Compare accuracy. See how
                  examples in the prompt genuinely change model behavior.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Novel Task ICL (Supported)" color="blue">
                <p className="text-sm">
                  Create made-up mappings the model has never seen in
                  training. Test with 3-5 examples + a novel input. Try
                  mappings of increasing complexity. Discover where ICL
                  generalizes and where it fails.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Ordering Sensitivity (Supported)" color="blue">
                <p className="text-sm">
                  Take 5 sentiment examples. Test accuracy on 10 test inputs
                  with 5 different orderings. Plot accuracy per ordering.
                  See the recency and position effects consistent with
                  attention-based mechanism.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: ICL vs Finetuning (Independent)" color="blue">
                <p className="text-sm">
                  Compare few-shot ICL accuracy vs a finetuned classifier on
                  the same sentiment task. When does ICL win? When does
                  finetuning win? Discover they are complementary
                  tools&mdash;ICL is fast and flexible, finetuning is accurate
                  and robust.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                  Each exercise builds intuition for how ICL behaves in
                  practice&mdash;strengths, limitations, and surprises.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-2-1-in-context-learning.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  Open in Google Colab
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M6 3H3a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3M9 2h5v5M8 8l5-5"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            Exercise 1 establishes the basic phenomenon. Exercise 2 tests
            boundaries. Exercise 3 reveals fragility. Exercise 4 compares
            with finetuning. Each builds a more nuanced understanding.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Summary (Section 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'In-context learning: task learning from examples in the prompt, without weight updates.',
                description:
                  'A transformer trained on next-token prediction can classify sentiment, translate text, or follow novel patterns simply by placing examples in the prompt. No gradients, no optimizer, no parameter changes.',
              },
              {
                headline:
                  'The mechanism is attention.',
                description:
                  'Examples in the prompt create retrieval patterns in the attention weights. The test input\'s queries match example inputs\' keys, and the values carry the answer pattern. Same formula, longer context.',
              },
              {
                headline:
                  'ICL is a base model capability, not a product of instruction tuning.',
                description:
                  'GPT-3 demonstrated ICL as a base model, before any SFT or RLHF. The capability emerges from pretraining on diverse text. SFT makes it more accessible, but the mechanism is already there.',
              },
              {
                headline:
                  'Powerful but fragile: sensitive to ordering, format, and context window limits.',
                description:
                  'ICL is not human-like comprehension. Performance swings 20-30 points from reordering examples. More examples do not always help. The limitations are the limitations of attention and the forward pass.',
              },
              {
                headline:
                  'The prompt is a program; attention is the interpreter.',
                description:
                  'Different examples, different behavior. Same weights, different programs. Every prompt configures the model for a different task through attention over the context.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          12. References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Language Models are Few-Shot Learners',
                authors: 'Brown et al., 2020',
                url: 'https://arxiv.org/abs/2005.14165',
                note: 'The GPT-3 paper that introduced in-context learning as a phenomenon. Section 3 contains the few-shot evaluation results. Read the abstract and Section 1 for the core argument.',
              },
              {
                title: 'Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?',
                authors: 'Min et al., 2022',
                url: 'https://arxiv.org/abs/2202.12837',
                note: 'Shows that label correctness matters less than expected for ICL. The label-flipping experiments are in Section 3. Key finding: format and input distribution matter more than the actual mapping.',
              },
              {
                title: 'Transformers Learn In-Context by Gradient Descent',
                authors: 'von Oswald et al., 2023',
                url: 'https://arxiv.org/abs/2212.07677',
                note: 'Theoretical work on why ICL emerges. Shows transformers can implement gradient descent in their forward pass. Advanced reading\u2014mentioned in the lesson but not developed.',
              },
              {
                title: 'What Can Transformers Learn In-Context? A Case Study of Simple Function Classes',
                authors: 'Garg et al., 2022',
                url: 'https://arxiv.org/abs/2208.01066',
                note: 'Systematic study of what functions transformers can learn in-context. Shows ICL works for linear functions, sparse functions, and simple neural networks.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Next Step (Section 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="You now understand that examples in the prompt steer the model&rsquo;s behavior through attention. But putting examples in a prompt is ad hoc&mdash;which examples? In what format? What if you need specific output structure? The next lesson systematizes this: prompt engineering as programming, not conversation."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
