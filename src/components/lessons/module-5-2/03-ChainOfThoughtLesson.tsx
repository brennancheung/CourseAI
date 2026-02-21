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

/**
 * Chain-of-Thought Reasoning
 *
 * Third lesson in Module 5.2 (Reasoning & In-Context Learning).
 * STRETCH lesson — reframes the autoregressive loop as a computational amplifier.
 *
 * Cognitive load: 2 new concepts:
 *   1. Intermediate tokens as computation (core mechanism)
 *   2. Process supervision vs outcome supervision (lighter, INTRODUCED)
 *
 * Core concepts at DEVELOPED:
 * - Intermediate tokens as computation (the CoT mechanism)
 * - When CoT helps vs does not (problem-difficulty criterion)
 * - Zero-shot CoT vs few-shot CoT distinction
 *
 * Concepts at INTRODUCED:
 * - Process supervision vs outcome supervision
 * - Limits of CoT (error propagation, quality over quantity)
 *
 * Concepts at MENTIONED:
 * - Self-consistency / majority voting
 *
 * EXPLICITLY NOT COVERED:
 * - Reasoning models trained with RL to use CoT effectively (Lesson 4)
 * - Test-time compute scaling (Lesson 4)
 * - Search during inference, tree-of-thought, beam search (Lesson 4)
 * - Self-consistency implementation details
 * - Automated CoT generation or optimization
 *
 * Previous: Prompt Engineering (Module 5.2, Lesson 2)
 * Next: Reasoning Models (Module 5.2, Lesson 4)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Computation Per Problem Diagram
// Shows direct answer (1 forward pass) vs CoT (multiple forward passes
// with growing context). Each arrow is a full forward pass.
// ---------------------------------------------------------------------------

function ComputationDiagram() {
  const svgW = 640
  const svgH = 380

  // Colors
  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const directColor = '#f87171' // rose for wrong answer
  const cotColor = '#a78bfa' // violet for CoT
  const contextBg = '#1e293b'
  const arrowColor = '#60a5fa' // blue for forward pass arrows

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
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          Computation Per Problem
        </text>
        <text
          x={svgW / 2}
          y={36}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          Each arrow is one full forward pass through N transformer blocks
        </text>

        {/* ---- LEFT SIDE: Direct Answer ---- */}
        <text
          x={150}
          y={65}
          textAnchor="middle"
          fill={directColor}
          fontSize="12"
          fontWeight="600"
        >
          Direct Answer
        </text>

        {/* Input box */}
        <rect
          x={40}
          y={80}
          width={220}
          height={32}
          rx={6}
          fill={contextBg}
          stroke={dimText}
          strokeWidth={1}
          opacity={0.8}
        />
        <text x={50} y={100} fill={dimText} fontSize="10" fontFamily="monospace">
          17 x 24 = ?
        </text>

        {/* Single forward pass arrow */}
        <line
          x1={150}
          y1={112}
          x2={150}
          y2={150}
          stroke={arrowColor}
          strokeWidth={2}
        />
        <polygon points="150,155 145,145 155,145" fill={arrowColor} />
        <text x={165} y={138} fill={arrowColor} fontSize="8" fontWeight="500">
          1 forward pass
        </text>

        {/* Output box (wrong) */}
        <rect
          x={90}
          y={160}
          width={120}
          height={32}
          rx={6}
          fill={directColor}
          opacity={0.12}
          stroke={directColor}
          strokeWidth={1.5}
        />
        <text
          x={150}
          y={180}
          textAnchor="middle"
          fill={directColor}
          fontSize="11"
          fontWeight="600"
        >
          &quot;384&quot; (wrong)
        </text>

        {/* Label */}
        <text
          x={150}
          y={215}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          1 pass = N blocks of computation
        </text>

        {/* ---- RIGHT SIDE: Chain-of-Thought ---- */}
        <text
          x={470}
          y={65}
          textAnchor="middle"
          fill={cotColor}
          fontSize="12"
          fontWeight="600"
        >
          Chain-of-Thought
        </text>

        {/* Step 1: Initial context */}
        <rect
          x={350}
          y={80}
          width={240}
          height={28}
          rx={5}
          fill={contextBg}
          stroke={dimText}
          strokeWidth={1}
          opacity={0.8}
        />
        <text x={360} y={98} fill={dimText} fontSize="9" fontFamily="monospace">
          17 x 24 = ? Let&apos;s think step by step.
        </text>

        {/* Arrow 1 */}
        <line
          x1={470}
          y1={108}
          x2={470}
          y2={124}
          stroke={arrowColor}
          strokeWidth={1.5}
        />
        <polygon points="470,128 466,121 474,121" fill={arrowColor} />

        {/* Step 2: + first reasoning token */}
        <rect
          x={350}
          y={132}
          width={240}
          height={28}
          rx={5}
          fill={contextBg}
          stroke={cotColor}
          strokeWidth={1}
          opacity={0.6}
        />
        <text x={360} y={150} fill={cotColor} fontSize="9" fontFamily="monospace">
          + 17 x 20 = 340
        </text>

        {/* Arrow 2 */}
        <line
          x1={470}
          y1={160}
          x2={470}
          y2={176}
          stroke={arrowColor}
          strokeWidth={1.5}
        />
        <polygon points="470,180 466,173 474,173" fill={arrowColor} />

        {/* Step 3: + second reasoning token */}
        <rect
          x={350}
          y={184}
          width={240}
          height={28}
          rx={5}
          fill={contextBg}
          stroke={cotColor}
          strokeWidth={1}
          opacity={0.6}
        />
        <text x={360} y={202} fill={cotColor} fontSize="9" fontFamily="monospace">
          + 17 x 4 = 68
        </text>

        {/* Arrow 3 */}
        <line
          x1={470}
          y1={212}
          x2={470}
          y2={228}
          stroke={arrowColor}
          strokeWidth={1.5}
        />
        <polygon points="470,232 466,225 474,225" fill={arrowColor} />

        {/* Step 4: Final answer */}
        <rect
          x={350}
          y={236}
          width={240}
          height={28}
          rx={5}
          fill={cotColor}
          opacity={0.12}
          stroke={cotColor}
          strokeWidth={1.5}
        />
        <text
          x={470}
          y={254}
          textAnchor="middle"
          fill={cotColor}
          fontSize="10"
          fontWeight="600"
        >
          + 340 + 68 = 408 (correct)
        </text>

        {/* Forward pass count label */}
        <text x={600} y={106} fill={arrowColor} fontSize="8" fontWeight="500">
          pass 1
        </text>
        <text x={600} y={158} fill={arrowColor} fontSize="8" fontWeight="500">
          pass 2
        </text>
        <text x={600} y={210} fill={arrowColor} fontSize="8" fontWeight="500">
          pass 3
        </text>
        <text x={600} y={254} fill={arrowColor} fontSize="8" fontWeight="500">
          pass 4
        </text>

        {/* Growing context bracket */}
        <line
          x1={340}
          y1={85}
          x2={340}
          y2={260}
          stroke={cotColor}
          strokeWidth={1}
          opacity={0.3}
        />
        <text
          x={335}
          y={175}
          textAnchor="end"
          fill={cotColor}
          fontSize="8"
          opacity={0.6}
          transform="rotate(-90, 335, 175)"
        >
          context grows with each step
        </text>

        {/* Bottom comparison */}
        <text
          x={svgW / 2}
          y={300}
          textAnchor="middle"
          fill={brightText}
          fontSize="11"
          fontWeight="600"
        >
          Same model, same architecture, same weights
        </text>
        <text
          x={svgW / 2}
          y={318}
          textAnchor="middle"
          fill={dimText}
          fontSize="10"
        >
          Direct: 1 forward pass = N blocks of computation
        </text>
        <text
          x={svgW / 2}
          y={336}
          textAnchor="middle"
          fill={cotColor}
          fontSize="10"
        >
          CoT: 4+ steps, each involving multiple tokens and forward passes
        </text>
        <text
          x={svgW / 2}
          y={360}
          textAnchor="middle"
          fill={arrowColor}
          fontSize="10"
          fontWeight="600"
        >
          The only difference: how many tokens are generated
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function ChainOfThoughtLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Chain-of-Thought Reasoning"
            description="A model that gets 17 &times; 24 wrong in one shot gets it right when asked to show its work. The model did not become smarter. It ran more forward passes. That distinction changes how you think about what language models can and cannot do."
            category="Reasoning & ICL"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            This lesson teaches you why chain-of-thought prompting works
            mechanically&mdash;that intermediate reasoning tokens give the model
            additional forward passes worth of computation&mdash;and how to
            identify when CoT helps versus when it does not.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why "let\'s think step by step" works mechanically (intermediate tokens as additional forward passes)',
              'When CoT helps (multi-step problems) and when it does not (simple tasks)',
              'Zero-shot CoT vs few-shot CoT',
              'Process supervision vs outcome supervision (how to evaluate reasoning chains)',
              'NOT: reasoning models trained with RL to use CoT effectively (next lesson)',
              'NOT: test-time compute scaling (next lesson)',
              'NOT: search during inference, tree-of-thought, or beam search (next lesson)',
              'NOT: self-consistency implementation details (mentioned only)',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="A New Perspective">
            This lesson reframes something you already know&mdash;the
            autoregressive feedback loop&mdash;as a computational amplifier.
            The mechanism is not new. The way you see it will be.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where You Left Off"
            subtitle="Two facts you already know, not yet connected"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You built the <code className="text-sm bg-muted px-1 rounded">generate()</code> method
              in Building nanoGPT: sample a token, append it to the context,
              run another forward pass. Autoregressive generation is a feedback
              loop&mdash;outputs become inputs. And you know that each forward
              pass runs through the same fixed architecture: the same N
              transformer blocks, the same dimensions, the same operations,
              whether the question is &ldquo;2 + 2&rdquo; or &ldquo;prove the
              Riemann hypothesis.&rdquo;
            </p>
            <p className="text-muted-foreground">
              This lesson connects these two facts. Prompt Engineering ended
              with a deliberate cliffhanger: &ldquo;there is a class of
              problems where even the best-structured prompt fails: problems
              that require more computation than a single forward pass
              provides.&rdquo; Now you find out what that means.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Pieces, One Insight">
            Autoregressive loop + fixed computation per forward
            pass = the mechanism behind chain-of-thought. You
            understand both pieces well from previous lessons. This
            lesson is the connection.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: Before/After Contrast
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Same Model, Different Answer"
            subtitle="17 × 24 with and without intermediate tokens"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider multi-step arithmetic: 17 &times; 24. A human would
              decompose this: 17 &times; 20 = 340, 17 &times; 4 = 68,
              340 + 68 = 408. Multiple steps, each building on the last.
            </p>

            <ComparisonRow
              left={{
                title: 'Direct Prompt',
                color: 'rose',
                items: [
                  '"What is 17 × 24?"',
                  'Model outputs: "384"',
                  'Wrong.',
                  'One forward pass: input → answer',
                ],
              }}
              right={{
                title: 'CoT Prompt',
                color: 'emerald',
                items: [
                  '"What is 17 × 24? Let\'s work through this step by step."',
                  '17 × 20 = 340',
                  '17 × 4 = 68',
                  '340 + 68 = 408. Correct.',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Same model. Same weights. Same question. The only difference is
              that the second prompt caused the model to generate intermediate
              tokens before producing the final answer.
            </p>

            <GradientCard title="The Puzzle" color="orange">
              <p className="text-sm">
                The model did not become smarter. Its weights did not change.
                Its architecture did not change. It ran the same forward pass
                it always runs. Yet it went from wrong to right.{' '}
                <strong>What changed?</strong>
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Magic">
            The model did not &ldquo;decide to try harder.&rdquo; There is no
            difficulty knob. The answer is mechanical, and you already have
            the pieces to figure it out.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain Part 1: The Fixed Computation Budget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Fixed Computation Budget"
            subtitle="Every token gets the same compute, regardless of difficulty"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every time the model predicts a token, it runs one forward pass:
              the input goes through N transformer blocks, and out comes a
              probability distribution over the vocabulary. This is true
              whether the question is &ldquo;2 + 2&rdquo; or &ldquo;17 &times;
              24&rdquo; or &ldquo;prove the Riemann hypothesis.&rdquo; The
              model gets the <strong>same computational budget per
              token</strong>, no matter what.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-muted-foreground text-sm font-medium">
                One forward pass, every time:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-1 text-sm">
                <li>Token embedding + positional encoding (one vector per token)</li>
                <li>N transformer blocks, each running MHA + FFN on the full context</li>
                <li>Output projection to vocabulary logits</li>
                <li>Sample one token</li>
              </ol>
              <p className="text-muted-foreground text-sm mt-2">
                This runs identically for every token prediction. N = 12 for
                GPT-2, N = 96 for GPT-3&mdash;but whatever N is, it is the
                same for every token. The model cannot &ldquo;think
                harder&rdquo; on a difficult problem. There is no difficulty
                knob.
              </p>
            </div>

            <p className="text-muted-foreground">
              Look at the code you wrote in Building nanoGPT. Each iteration of
              the generation loop
              calls <code className="text-sm bg-muted px-1 rounded">forward()</code> once.
              One forward pass, one token. That is the model&rsquo;s entire
              computational budget for that token.
            </p>

            <p className="text-muted-foreground">
              Here is an analogy that makes this concrete. Asking a model to
              answer &ldquo;17 &times; 24&rdquo; in one forward pass is like
              asking you to multiply two-digit numbers without writing anything
              down. Some people can manage it; most cannot. Writing intermediate
              results is not &ldquo;trying harder&rdquo;&mdash;it is using
              external memory to decompose a problem that exceeds your working
              memory. The model&rsquo;s context window is its scratchpad.
            </p>

            <GradientCard title="The Question" color="violet">
              <p className="text-sm">
                What if a problem requires more computation than one forward
                pass provides? You cannot make the model &ldquo;think
                harder.&rdquo; The architecture is fixed. But there is another
                way to get more computation.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Fixed Budget">
            A forward pass is the model&rsquo;s thinking budget per token.
            Same N blocks for &ldquo;2 + 2&rdquo; and for multi-digit
            multiplication. The model cannot allocate more compute to harder
            problems&mdash;unless it generates more tokens.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain Part 2: Tokens as Computation (Core Mechanism)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tokens as Computation"
            subtitle="The autoregressive loop is a computational amplifier"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Walk through what happens step by step when the model sees
              &ldquo;17 &times; 24 = ? Let&rsquo;s work through this step by
              step.&rdquo;
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 text-sm">
                <li>
                  <strong>Forward pass #1:</strong> The model sees the full prompt.
                  It generates the first reasoning token: &ldquo;17&rdquo;
                </li>
                <li>
                  <strong>Forward pass #2:</strong> Context now
                  includes &ldquo;17&rdquo;. Generates &ldquo;&times;&rdquo;
                </li>
                <li>
                  <strong>Forward pass #3:</strong> Context now
                  includes &ldquo;17 &times;&rdquo;. Generates &ldquo;20&rdquo;
                </li>
                <li>
                  <strong>Forward passes #4-7:</strong> The model generates
                  &ldquo;= 340&rdquo; token by token. Each forward pass takes the
                  entire context&mdash;original question plus all previously
                  generated reasoning tokens&mdash;as input.
                </li>
                <li>
                  <strong>Forward passes #8-11:</strong> &ldquo;17 &times; 4 =
                  68&rdquo;. The intermediate result &ldquo;340&rdquo; is now in
                  the context. Subsequent forward passes can attend to it.
                </li>
                <li>
                  <strong>Forward passes #12-15:</strong> &ldquo;340 + 68 =
                  408&rdquo;. Both intermediate results are available for
                  attention.
                </li>
              </ol>
            </div>

            <p className="text-muted-foreground">
              Direct answer: ~1 forward pass maps the question to an answer.
              CoT: ~15 forward passes, each building on the context of previous
              ones. Same model, same architecture, <strong>15&times; more
              computation</strong>.
            </p>

            {/* The "of course" beat */}
            <div className="py-4 px-6 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <p className="text-muted-foreground text-sm">
                <strong className="text-violet-400">You already knew
                this.</strong>{' '}
                Autoregressive generation is a feedback loop: outputs become
                inputs. Each token gets a full forward pass through N blocks.
                If you generate more tokens, you get more forward passes. The
                intermediate reasoning tokens are not decoration&mdash;they are
                additional computation.{' '}
                <em>
                  Of course they help: more forward passes means more
                  computation, and the intermediate results are available in the
                  context for subsequent forward passes to attend to.
                </em>
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Insight">
            Each intermediate token triggers another forward pass. The
            autoregressive loop is not just a generation mechanism&mdash;it
            is a computational amplifier. More tokens = more forward passes =
            more computation per problem.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Misconceptions 1 & 2
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What CoT Is Not"
            subtitle="Two misconceptions to address immediately"
          />
          <div className="space-y-4">
            <GradientCard title="Misconception: &ldquo;The model thinks during CoT&rdquo;" color="rose">
              <div className="space-y-3 text-sm">
                <p>
                  It is tempting to say the model &ldquo;thinks through&rdquo;
                  the problem. It does not. Each token is generated by the same
                  mechanical process: forward pass, sample from distribution,
                  append. The model is not deliberating. It is generating tokens
                  that happen to contain useful intermediate results, and those
                  results feed back as context for subsequent forward passes.
                </p>
                <p className="text-muted-foreground">
                  If you replace the correct intermediate steps with
                  plausible-sounding but wrong steps, the model will continue
                  from the wrong steps and produce a wrong answer. It is
                  not &ldquo;checking its reasoning&rdquo;&mdash;it is
                  continuing from whatever context exists.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Misconception: &ldquo;CoT is showing internal work&rdquo;" color="rose">
              <div className="space-y-3 text-sm">
                <p>
                  CoT is not the model &ldquo;showing its work&rdquo;&mdash;
                  exposing reasoning it would have done internally anyway. If
                  the model could compute 17 &times; 24 internally, it would
                  get the right answer without CoT. It cannot, because one
                  forward pass through N blocks is not enough computation for
                  multi-digit multiplication.
                </p>
                <p className="text-muted-foreground">
                  The intermediate tokens
                  are <strong>enabling</strong> computation that could not
                  happen in a single pass, not <em>displaying</em> computation
                  that already happened. The model with the direct answer has
                  exactly N blocks of computation. With CoT, it has N blocks
                  times the number of intermediate tokens.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not &ldquo;Thinking&rdquo;">
            The &ldquo;thinking&rdquo; is the token-by-token feedback loop,
            not an internal deliberation. Same mechanism as every other
            generation&mdash;the difference is that intermediate tokens
            happen to carry useful results.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Visual: Computation Diagram
          ================================================================ */}
      <Row>
        <Row.Content>
          <ComputationDiagram />

          <div className="space-y-4 mt-2">
            <p className="text-muted-foreground">
              Connect this to the code you wrote. Each iteration of
              the <code className="text-sm bg-muted px-1 rounded">generate()</code> loop
              is one forward pass. Direct answer: ~1 iteration. CoT: ~15
              iterations. The code is identical&mdash;the only difference is
              how many tokens are generated.
            </p>

            <CodeBlock
              code={`# Your generate() method from Building nanoGPT
# Each iteration = one forward pass = one token
for _ in range(max_new_tokens):
    # Forward pass: context → logits (same N blocks every time)
    logits = model(context)
    # Sample one token
    next_token = sample(logits)
    # Append to context (output becomes input)
    context = concat(context, next_token)

# Direct answer:  max_new_tokens ≈ 1   → 1 forward pass
# CoT:            max_new_tokens ≈ 15  → 15 forward passes
# Same code. Same model. More iterations = more computation.`}
              language="python"
              filename="generate.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Count the Passes">
            The diagram makes the core insight spatial and countable. You can
            literally count the forward passes. Direct answer: 1. CoT: many.
            Each one runs through the same N transformer blocks.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 1: Predict-and-Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Predict the Computation" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A farmer has 3 fields. Each field has 7 rows of corn. Each row
                has 12 plants. How many corn plants does the farmer have?
              </p>

              <div className="space-y-2">
                <p>
                  <strong>Prompt A:</strong> &ldquo;Answer directly: A farmer
                  has 3 fields...&rdquo;
                </p>
                <p>
                  <strong>Prompt B:</strong> &ldquo;Think step by step: A
                  farmer has 3 fields...&rdquo;
                </p>
              </div>

              <p><strong>Predict before revealing:</strong></p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>Which prompt is more likely to get the right answer?</li>
                <li>How many forward passes does each prompt use approximately?</li>
                <li>What intermediate results would appear in Prompt B&rsquo;s output?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p>
                    <strong>Prompt B</strong> is more likely to get the right
                    answer. It produces intermediate results:
                  </p>
                  <ul className="list-disc list-inside space-y-1 ml-2 text-muted-foreground">
                    <li>3 fields &times; 7 rows = 21 rows total</li>
                    <li>21 rows &times; 12 plants = 252 plants</li>
                  </ul>
                  <p className="text-muted-foreground">
                    Each intermediate result (21, 252) is generated by its own
                    forward passes and becomes available in context for the next
                    step. Prompt A must compute 3 &times; 7 &times; 12 in a
                    single forward pass&mdash;mapping the entire problem to the
                    answer in N transformer blocks.
                  </p>
                  <p className="text-muted-foreground">
                    Prompt A: ~1-2 forward passes. Prompt B: ~10-15 forward
                    passes. The extra passes are the computation that makes the
                    answer correct.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Elaborate: When CoT Helps and When It Does Not
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When CoT Helps and When It Does Not"
            subtitle="The criterion is computational complexity, not difficulty"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a second example&mdash;a word problem that requires
              extracting information and performing multiple operations:
            </p>

            <ComparisonRow
              left={{
                title: 'Direct Answer',
                color: 'amber',
                items: [
                  '"If a store has 3 shelves with 8 books each and removes 5 books, how many are left?"',
                  'Model may output: "24" (ignoring the removal)',
                  'Or: "19" (correct, but unreliable)',
                  'One forward pass must do extraction + multiplication + subtraction',
                ],
              }}
              right={{
                title: 'CoT Answer',
                color: 'emerald',
                items: [
                  'Same question + "Let\'s think step by step."',
                  '"3 shelves × 8 books = 24 books"',
                  '"24 − 5 = 19 books"',
                  'Intermediate result (24) is in context for the subtraction step',
                ],
              }}
            />

            <GradientCard title="Misconception: &ldquo;Always use CoT&rdquo;" color="rose">
              <p className="text-sm">
                You might assume that CoT always helps&mdash;after all, more
                computation is always better, right? It is not. CoT helps only
                when the problem exceeds single-forward-pass capacity. For tasks
                that fit in one pass, CoT adds nothing and can waste context
                window or introduce errors through overthinking.
              </p>
            </GradientCard>

            <p className="text-muted-foreground">
              Compare with a task where CoT does <em>not</em> help:
            </p>

            <GradientCard title="Negative Example: CoT on Factual Recall" color="amber">
              <div className="space-y-3 text-sm">
                <ComparisonRow
                  left={{
                    title: 'Without CoT',
                    color: 'blue',
                    items: [
                      '"What is the capital of France?"',
                      'Model: "Paris"',
                      'Correct. One forward pass is enough.',
                    ],
                  }}
                  right={{
                    title: 'With CoT',
                    color: 'amber',
                    items: [
                      '"What is the capital of France? Let\'s think step by step."',
                      '"France is a country in Western Europe... its capital is Paris."',
                      'Same answer. Extra tokens added nothing.',
                    ],
                  }}
                />
                <p className="text-muted-foreground">
                  Factual recall is a single lookup in the model&rsquo;s
                  parametric knowledge. Additional forward passes add
                  nothing useful. The answer was already computable in one
                  pass. The extra tokens waste context window and can
                  even introduce errors through overthinking.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="The Criterion" color="blue">
              <p className="text-sm">
                CoT helps when the problem requires more computation than one
                forward pass provides. Multi-step arithmetic, word problems,
                logical reasoning, constraint satisfaction&mdash;these exceed
                single-pass capacity. Factual recall, sentiment
                classification, simple text completion&mdash;these fit within
                a single pass. The criterion is <strong>computational
                complexity</strong>, not difficulty in the human sense.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Boundary">
            CoT helps when the problem needs more computation than one
            forward pass provides. For tasks that fit in a single
            pass&mdash;factual recall, classification&mdash;CoT adds
            nothing and can hurt.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Error Propagation + Quality Over Quantity
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quality Over Quantity"
            subtitle="Errors propagate, and longer is not always better"
          />
          <div className="space-y-4">
            <GradientCard title="Negative Example: Error Propagation" color="rose">
              <div className="space-y-3 text-sm">
                <p>What happens when an intermediate step is wrong?</p>
                <div className="py-3 px-4 bg-muted/30 rounded-lg font-mono text-xs space-y-1">
                  <p>17 &times; 20 = <span className="text-rose-400 font-bold">350</span> (wrong)</p>
                  <p>17 &times; 4 = 68</p>
                  <p><span className="text-rose-400 font-bold">350</span> + 68 = <span className="text-rose-400 font-bold">418</span> (wrong, built on the error)</p>
                </div>
                <p className="text-muted-foreground">
                  The model does not &ldquo;catch&rdquo; the error. It
                  continues from whatever context exists. If the context
                  contains &ldquo;350,&rdquo; subsequent forward passes build
                  on &ldquo;350.&rdquo; The model is not reasoning&mdash;it is
                  generating tokens that feed back as context.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              This also means <strong>longer chains are not always
              better</strong>. A rambling 15-step chain can be worse than a
              concise 3-step chain. Irrelevant intermediate tokens add noise
              to the context, diluting attention to the relevant intermediate
              results. This is the same principle you saw in In-Context
              Learning: more examples do not always help, and in
              Prompt Engineering: irrelevant retrieved documents dilute
              attention.
            </p>

            <GradientCard title="Quality Matters, Not Just Quantity" color="violet">
              <p className="text-sm">
                Each intermediate token must provide a useful intermediate
                result that subsequent tokens can build on. Random or
                irrelevant tokens do not help even though they technically add
                forward passes&mdash;the additional context they create is
                noise, not signal. The criterion is not &ldquo;more
                tokens&rdquo; but &ldquo;more <em>useful</em> tokens.&rdquo;
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Error Propagation">
            The model does not catch its own mistakes. Each step builds
            on the context from previous steps. A wrong intermediate
            result corrupts everything downstream.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Misconception 5: CoT as Emergent Capability
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Not a New Mechanism"
            subtitle="CoT is the autoregressive loop applied to reasoning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              CoT is sometimes framed as a mysterious &ldquo;emergent
              capability&rdquo; that appears at scale. It is not a new
              mechanism. It is the
              same <code className="text-sm bg-muted px-1 rounded">generate()</code> loop
              you implemented in Building nanoGPT, applied to reasoning
              problems.
            </p>

            <p className="text-muted-foreground">
              Small models fail at CoT not because they lack the
              mechanism&mdash;they have the same autoregressive loop&mdash;but
              because they cannot generate <em>useful</em> intermediate steps.
              The mechanism is universal. The usefulness depends on whether the
              model can produce coherent intermediate reasoning tokens that
              subsequent forward passes can build on. Large models succeed at
              CoT because they have learned reasoning patterns from training
              data, and those patterns produce useful intermediate results.
            </p>

            <p className="text-muted-foreground">
              Remember from In-Context Learning: &ldquo;the formula has not
              changed; the context has.&rdquo; The same principle applies here.
              The forward pass is the same. The context&mdash;enriched by
              intermediate reasoning tokens&mdash;is what changes.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Loop, New Perspective">
            CoT is not magic. It is the autoregressive feedback loop
            generating tokens that happen to be useful intermediate
            reasoning steps. Small models have the same loop but
            cannot generate useful steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Zero-Shot CoT vs Few-Shot CoT
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Zero-Shot CoT vs Few-Shot CoT"
            subtitle="Two ways to trigger reasoning tokens"
          />
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'Zero-Shot CoT',
                color: 'blue',
                items: [
                  'Add "Let\'s think step by step" to the prompt',
                  'No examples needed',
                  'Kojima et al. 2022',
                  'Works because the phrasing triggers reasoning-style tokens from pretraining data',
                  'A prompt engineering technique: a format instruction that triggers a specific output pattern',
                ],
              }}
              right={{
                title: 'Few-Shot CoT',
                color: 'violet',
                items: [
                  'Provide examples that include reasoning chains',
                  'Wei et al. 2022 (the original CoT paper)',
                  'The model learns the step-by-step format from the examples',
                  'An ICL technique: examples with reasoning chains as the "program"',
                  'More reliable when the reasoning format matters',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Both work for the same mechanistic reason: they cause the model
              to generate intermediate tokens that provide additional forward
              passes. Zero-shot CoT is simpler; few-shot CoT gives you more
              control over the reasoning format through examples.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <p className="text-muted-foreground text-sm">
                <strong>Self-consistency</strong> (Wang et al. 2022): generate
                multiple reasoning chains, take the majority-vote answer. If
                one chain can have errors, running several and voting reduces
                the chance of error. We will see a more principled version of
                this idea in the next lesson on reasoning models.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Connection to Prior Lessons">
            Zero-shot CoT is a prompt engineering technique (format
            instruction). Few-shot CoT is an ICL technique (examples as
            program). Both fit the framework you already have.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          14. Process Supervision vs Outcome Supervision
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Process Supervision vs Outcome Supervision"
            subtitle="Two ways to evaluate a reasoning chain"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We said CoT quality matters&mdash;errors in intermediate steps
              propagate. How do you evaluate whether a reasoning chain is good?
            </p>

            <ComparisonRow
              left={{
                title: 'Outcome Supervision',
                color: 'amber',
                items: [
                  'Evaluate only the final answer',
                  'Correct answer → good chain',
                  'Wrong answer → bad chain',
                  'Simple but loses information',
                  'A chain with correct reasoning but a calculation error at the last step is marked the same as completely wrong logic',
                ],
              }}
              right={{
                title: 'Process Supervision',
                color: 'emerald',
                items: [
                  'Evaluate each step individually',
                  'Is step 1 correct?',
                  'Is step 2 a valid follow-up to step 1?',
                  'More informative but much harder',
                  'Requires per-step labels, not just a final answer',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Why does this matter? If you are training a model to produce
              better reasoning chains (which is what the next lesson covers),
              the supervision signal determines what the model learns. Outcome
              supervision rewards any chain that reaches the right
              answer&mdash;even if the reasoning is flawed. Process supervision
              rewards correct reasoning steps, which produces more reliable
              chains.
            </p>

            <GradientCard title="The Distinction That Matters" color="violet">
              <div className="space-y-3 text-sm">
                <p>
                  Consider two chains for a math problem, both producing the
                  correct final answer:
                </p>
                <ul className="space-y-2 ml-2">
                  <li>
                    &bull; <strong>Chain A:</strong> Correct reasoning at every
                    step. Each intermediate result is valid.
                  </li>
                  <li>
                    &bull; <strong>Chain B:</strong> An error in step 2 that
                    happens to cancel out in step 3, producing the correct
                    final answer by luck.
                  </li>
                </ul>
                <p className="text-muted-foreground">
                  Outcome supervision rates both equally. Process supervision
                  rates Chain A higher. Chain B&rsquo;s reasoning pattern will
                  fail on a different problem where the errors do not happen to
                  cancel. Outcome supervision asks &ldquo;did you get the right
                  answer?&rdquo; Process supervision asks &ldquo;did you reason
                  correctly at every step?&rdquo;
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Supervision Matters">
            Process supervision is harder but more informative. It rewards
            correct reasoning, not just correct answers. The distinction
            becomes central in the next lesson on reasoning models.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          15. Check 2: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Apply the Concepts" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A developer is building a system that uses an LLM to grade math
                homework. The LLM reads a student&rsquo;s solution (with
                intermediate steps) and decides if it is correct.
              </p>

              <ol className="list-decimal list-inside space-y-2 ml-2">
                <li>
                  Should the LLM use CoT to reason about whether each step
                  is correct? Why or why not?
                </li>
                <li>
                  Is this more like process supervision or outcome
                  supervision?
                </li>
                <li>
                  What failure mode should the developer worry about?
                </li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p><strong>1. Yes&mdash;use CoT.</strong></p>
                  <p className="text-muted-foreground">
                    Evaluating multi-step mathematical reasoning requires more
                    computation than a single forward pass. The LLM needs to
                    check each step, which benefits from generating intermediate
                    verification tokens.
                  </p>

                  <p><strong>2. Process supervision.</strong></p>
                  <p className="text-muted-foreground">
                    The LLM is evaluating each step individually rather than
                    just checking the final answer. It needs to assess whether
                    each intermediate step is a valid logical follow-up to the
                    previous one.
                  </p>

                  <p><strong>3. Error propagation + continuation.</strong></p>
                  <p className="text-muted-foreground">
                    Two failure modes: (a) if the LLM makes an error in
                    evaluating step 2, it may approve subsequent incorrect
                    steps that build on its own wrong evaluation, and (b) the
                    LLM might &ldquo;continue&rdquo; the student&rsquo;s
                    reasoning rather than critically examining it&mdash;generating
                    tokens that extend the student&rsquo;s work rather than
                    verifying it.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          16. Practice: Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Explore when and why CoT works"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Four exercises that explore the mechanism behind chain-of-thought.
              Each can be completed independently, but they share a
              theme: understanding when and why intermediate tokens change the
              answer.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Direct vs CoT Comparison (Guided)" color="blue">
                <p className="text-sm">
                  Solve 10 arithmetic problems with and without CoT. Both
                  single-step and multi-step. Predict which will benefit from
                  CoT before running. Compare accuracy and see the
                  computational complexity boundary in action.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Token Counting as Computation (Supported)" color="blue">
                <p className="text-sm">
                  For 5 problems solved with CoT, count the intermediate tokens
                  generated. Plot tokens vs problem complexity. See that more
                  complex problems generate more intermediate tokens&mdash;and
                  each token is an additional forward pass.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Error Propagation Experiment (Supported)" color="blue">
                <p className="text-sm">
                  Take a multi-step problem, manually corrupt one intermediate
                  step, and ask the model to continue. Corrupt at different
                  positions (early vs late) and compare impact. See that the
                  model does not &ldquo;catch&rdquo; errors.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: Find the CoT Boundary (Independent)" color="blue">
                <p className="text-sm">
                  Design a set of problems of increasing complexity in a domain
                  of your choice. Test each with and without CoT. Find the
                  approximate threshold where CoT starts helping. No skeleton
                  provided&mdash;you design the experiment.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                  Each exercise explores a different aspect of the CoT
                  mechanism with immediate, empirical feedback.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-2-3-chain-of-thought.ipynb"
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
            Exercise 1 establishes the empirical fact. Exercise 2 quantifies
            the computation. Exercise 3 tests error propagation. Exercise 4
            finds the boundary where CoT starts helping.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          17. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'Chain-of-thought works because each intermediate token triggers an additional forward pass.',
                description:
                  'The model does not "think harder." It runs more forward passes, each building on the context of previous ones. More tokens = more computation per problem.',
              },
              {
                headline:
                  'The model\'s computational budget per token is fixed by architecture.',
                description:
                  'Every token prediction runs through the same N transformer blocks. CoT is a way to exceed that budget by generating more tokens. The budget is fixed; the spending is not.',
              },
              {
                headline:
                  'CoT helps on multi-step problems, not on simple tasks.',
                description:
                  'The criterion is computational complexity: does the problem require more computation than one forward pass provides? Factual recall, classification, and simple completion fit in one pass. Multi-step arithmetic, reasoning, and planning do not.',
              },
              {
                headline:
                  'The model does not "reason" during CoT, and quality matters more than quantity.',
                description:
                  'It generates tokens that feed back as context for subsequent forward passes. Errors in intermediate steps propagate because the model continues from whatever context exists. A concise chain with correct intermediate results outperforms a rambling chain with noise.',
              },
              {
                headline:
                  'Process supervision evaluates each step; outcome supervision evaluates only the final answer.',
                description:
                  'Process supervision is harder but more informative. It rewards correct reasoning, not just correct answers. This distinction becomes central when training models to reason effectively.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          18. Mental Model Echo
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Mental Model" color="violet">
            <p className="text-sm italic">
              A forward pass is the model&rsquo;s thinking budget per token.
              CoT is spending more budget by generating more tokens. The budget
              is fixed; the spending is not.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          19. References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
                authors: 'Wei et al., 2022',
                url: 'https://arxiv.org/abs/2201.11903',
                note: 'The original CoT paper. Demonstrates few-shot CoT on arithmetic, commonsense, and symbolic reasoning. Section 2 shows the prompting format.',
              },
              {
                title: 'Large Language Models are Zero-Shot Reasoners',
                authors: 'Kojima et al., 2022',
                url: 'https://arxiv.org/abs/2205.11916',
                note: 'Shows that simply adding "Let\'s think step by step" improves reasoning without any examples. The zero-shot CoT paper.',
              },
              {
                title: 'Self-Consistency Improves Chain of Thought Reasoning in Language Models',
                authors: 'Wang et al., 2022',
                url: 'https://arxiv.org/abs/2203.11171',
                note: 'Introduces majority voting over multiple CoT chains. A simple technique for more robust reasoning. Read Section 1 for the core idea.',
              },
              {
                title: "Let's Verify Step by Step",
                authors: 'Lightman et al., 2023',
                url: 'https://arxiv.org/abs/2305.20050',
                note: 'The OpenAI paper on process supervision vs outcome supervision. Shows that process supervision produces more reliable mathematical reasoning. Directly relevant to the process/outcome distinction in this lesson.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          20. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="You now understand why chain-of-thought works: intermediate tokens give the model more computation via additional forward passes. But CoT as we&rsquo;ve seen it is ad hoc&mdash;you add &ldquo;let&rsquo;s think step by step&rdquo; and hope the model generates useful reasoning. What if you trained the model to reason effectively? What if you could make the model spend more computation on harder problems and less on easier ones? The next lesson covers reasoning models&mdash;RL-trained CoT, test-time compute scaling, and the paradigm shift from &ldquo;bigger model&rdquo; to &ldquo;more thinking time.&rdquo;"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
