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
  ModuleCompleteBlock,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'

/**
 * Reasoning Models
 *
 * Fourth and final lesson in Module 5.2 (Reasoning & In-Context Learning).
 * BUILD lesson — connects established frameworks (CoT mechanism + RL training).
 *
 * Cognitive load: 2-3 new concepts:
 *   1. RL for reasoning (known RLHF applied to reasoning quality)
 *   2. Test-time compute scaling (paradigm shift)
 *   3. Search during inference (self-consistency DEVELOPED from MENTIONED)
 *
 * Core concepts at DEVELOPED:
 * - RL for reasoning (mechanism, connection to RLHF)
 * - Test-time compute scaling (the paradigm shift, the tradeoff)
 * - Process supervision vs outcome supervision (upgraded from INTRODUCED)
 * - Self-consistency / search during inference (upgraded from MENTIONED)
 *
 * Concepts at INTRODUCED:
 * - Specific reasoning model architectures (o1-style, know they exist)
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing RL for reasoning in code
 * - Specific model architectures or training details of o1, DeepSeek-R1, etc.
 * - Tree-of-thought or MCTS implementation details
 * - Mathematical formalization of the RL objective
 * - Distillation of reasoning models
 * - Agentic patterns, tool use, or multi-step planning
 *
 * Previous: Chain-of-Thought Reasoning (Module 5.2, Lesson 3)
 * Next: Next module (multimodal or similar)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Scaling Paradigm Diagram
// Two axes: model size (x) and inference compute (y).
// Traditional scaling moves right; test-time compute scaling moves up.
// ---------------------------------------------------------------------------

function ScalingParadigmDiagram() {
  const svgW = 580
  const svgH = 400

  // Colors
  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const traditionalColor = '#f59e0b' // amber for traditional scaling
  const testTimeColor = '#a78bfa' // violet for test-time compute
  const axisColor = '#64748b'
  const gridColor = '#1e293b'

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
          y={24}
          textAnchor="middle"
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          Two Axes of Scaling
        </text>
        <text
          x={svgW / 2}
          y={42}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          Both increase total computation, through different mechanisms
        </text>

        {/* Grid lines */}
        {[120, 180, 240, 300].map((y) => (
          <line
            key={`h-${y}`}
            x1={80}
            y1={y}
            x2={520}
            y2={y}
            stroke={gridColor}
            strokeWidth={1}
          />
        ))}
        {[160, 240, 320, 400].map((x) => (
          <line
            key={`v-${x}`}
            x1={x}
            y1={60}
            x2={x}
            y2={340}
            stroke={gridColor}
            strokeWidth={1}
          />
        ))}

        {/* Axes */}
        <line
          x1={80}
          y1={340}
          x2={520}
          y2={340}
          stroke={axisColor}
          strokeWidth={2}
        />
        <polygon points="525,340 515,335 515,345" fill={axisColor} />
        <line
          x1={80}
          y1={340}
          x2={80}
          y2={60}
          stroke={axisColor}
          strokeWidth={2}
        />
        <polygon points="80,55 75,65 85,65" fill={axisColor} />

        {/* Axis labels */}
        <text
          x={300}
          y={375}
          textAnchor="middle"
          fill={dimText}
          fontSize="11"
          fontWeight="500"
        >
          Model Size (parameters)
        </text>
        <text
          x={30}
          y={200}
          textAnchor="middle"
          fill={dimText}
          fontSize="11"
          fontWeight="500"
          transform="rotate(-90, 30, 200)"
        >
          Inference Compute (reasoning tokens)
        </text>

        {/* Starting point */}
        <circle cx={160} cy={300} r={6} fill="#60a5fa" />
        <text
          x={160}
          y={320}
          textAnchor="middle"
          fill="#60a5fa"
          fontSize="9"
          fontWeight="500"
        >
          Base model
        </text>

        {/* Traditional scaling arrow (horizontal) */}
        <line
          x1={170}
          y1={300}
          x2={420}
          y2={300}
          stroke={traditionalColor}
          strokeWidth={2.5}
          strokeDasharray="6,3"
        />
        <polygon
          points="425,300 415,295 415,305"
          fill={traditionalColor}
        />
        <text
          x={295}
          y={290}
          textAnchor="middle"
          fill={traditionalColor}
          fontSize="10"
          fontWeight="600"
        >
          Traditional: bigger model
        </text>
        <text
          x={295}
          y={260}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          GPT-2 (1.5B) → GPT-3 (175B) → GPT-4 (?)
        </text>

        {/* Larger model point */}
        <circle cx={440} cy={300} r={6} fill={traditionalColor} />
        <text
          x={440}
          y={320}
          textAnchor="middle"
          fill={traditionalColor}
          fontSize="9"
          fontWeight="500"
        >
          Larger model
        </text>

        {/* Test-time compute scaling arrow (vertical) */}
        <line
          x1={160}
          y1={290}
          x2={160}
          y2={110}
          stroke={testTimeColor}
          strokeWidth={2.5}
          strokeDasharray="6,3"
        />
        <polygon points="160,105 155,115 165,115" fill={testTimeColor} />
        <text
          x={245}
          y={195}
          textAnchor="start"
          fill={testTimeColor}
          fontSize="10"
          fontWeight="600"
        >
          New: more thinking time
        </text>
        <text
          x={245}
          y={210}
          textAnchor="start"
          fill={dimText}
          fontSize="8"
        >
          Same model, more reasoning tokens
        </text>

        {/* Reasoning model point */}
        <circle cx={160} cy={100} r={6} fill={testTimeColor} />
        <text
          x={160}
          y={88}
          textAnchor="middle"
          fill={testTimeColor}
          fontSize="9"
          fontWeight="500"
        >
          Reasoning model (same size, more inference)
        </text>

        {/* Key insight box */}
        <rect
          x={310}
          y={120}
          width={200}
          height={50}
          rx={6}
          fill={testTimeColor}
          fillOpacity={0.08}
          stroke={testTimeColor}
          strokeWidth={1}
          strokeOpacity={0.3}
        />
        <text
          x={410}
          y={140}
          textAnchor="middle"
          fill={brightText}
          fontSize="9"
          fontWeight="600"
        >
          A 7B reasoning model can
        </text>
        <text
          x={410}
          y={155}
          textAnchor="middle"
          fill={brightText}
          fontSize="9"
          fontWeight="600"
        >
          outperform a 70B base model
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: RL Training Loop Diagram
// Shows the cycle: generate chain -> check answer -> compute reward -> update
// ---------------------------------------------------------------------------

function RLTrainingLoopDiagram() {
  const svgW = 520
  const svgH = 280

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const accentColor = '#a78bfa' // violet
  const rewardColor = '#34d399' // emerald
  const policyColor = '#60a5fa' // blue
  const bgColor = '#1e293b'

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
          fontSize="12"
          fontWeight="600"
        >
          RL Training Loop for Reasoning
        </text>

        {/* Step 1: Math Problem */}
        <rect
          x={20}
          y={50}
          width={110}
          height={50}
          rx={8}
          fill={bgColor}
          stroke={dimText}
          strokeWidth={1}
        />
        <text
          x={75}
          y={72}
          textAnchor="middle"
          fill={brightText}
          fontSize="10"
          fontWeight="500"
        >
          Math Problem
        </text>
        <text
          x={75}
          y={88}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          (verifiable answer)
        </text>

        {/* Arrow 1 */}
        <line
          x1={130}
          y1={75}
          x2={162}
          y2={75}
          stroke={policyColor}
          strokeWidth={1.5}
        />
        <polygon points="166,75 158,70 158,80" fill={policyColor} />

        {/* Step 2: Model Generates Chain */}
        <rect
          x={168}
          y={50}
          width={130}
          height={50}
          rx={8}
          fill={accentColor}
          opacity={0.12}
          stroke={accentColor}
          strokeWidth={1.5}
        />
        <text
          x={233}
          y={72}
          textAnchor="middle"
          fill={accentColor}
          fontSize="10"
          fontWeight="500"
        >
          Generate Reasoning
        </text>
        <text
          x={233}
          y={88}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          Chain (policy output)
        </text>

        {/* Arrow 2 */}
        <line
          x1={298}
          y1={75}
          x2={330}
          y2={75}
          stroke={policyColor}
          strokeWidth={1.5}
        />
        <polygon points="334,75 326,70 326,80" fill={policyColor} />

        {/* Step 3: Check Answer */}
        <rect
          x={336}
          y={50}
          width={110}
          height={50}
          rx={8}
          fill={bgColor}
          stroke={rewardColor}
          strokeWidth={1.5}
        />
        <text
          x={391}
          y={72}
          textAnchor="middle"
          fill={rewardColor}
          fontSize="10"
          fontWeight="500"
        >
          Check Answer
        </text>
        <text
          x={391}
          y={88}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          Correct? Yes/No
        </text>

        {/* Arrow 3 (down) */}
        <line
          x1={391}
          y1={100}
          x2={391}
          y2={142}
          stroke={rewardColor}
          strokeWidth={1.5}
        />
        <polygon points="391,146 386,138 396,138" fill={rewardColor} />

        {/* Step 4: Compute Reward */}
        <rect
          x={316}
          y={148}
          width={150}
          height={50}
          rx={8}
          fill={rewardColor}
          opacity={0.12}
          stroke={rewardColor}
          strokeWidth={1.5}
        />
        <text
          x={391}
          y={170}
          textAnchor="middle"
          fill={rewardColor}
          fontSize="10"
          fontWeight="500"
        >
          Compute Reward
        </text>
        <text
          x={391}
          y={186}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          +1 correct / -1 incorrect
        </text>

        {/* Arrow 4 (left) */}
        <line
          x1={316}
          y1={173}
          x2={214}
          y2={173}
          stroke={policyColor}
          strokeWidth={1.5}
        />
        <polygon points="210,173 218,168 218,178" fill={policyColor} />

        {/* Step 5: Update Policy */}
        <rect
          x={80}
          y={148}
          width={130}
          height={50}
          rx={8}
          fill={policyColor}
          opacity={0.12}
          stroke={policyColor}
          strokeWidth={1.5}
        />
        <text
          x={145}
          y={170}
          textAnchor="middle"
          fill={policyColor}
          fontSize="10"
          fontWeight="500"
        >
          Update Policy
        </text>
        <text
          x={145}
          y={186}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          Generate better chains
        </text>

        {/* Arrow 5 (back up to generate) */}
        <line
          x1={180}
          y1={148}
          x2={216}
          y2={105}
          stroke={policyColor}
          strokeWidth={1.5}
          strokeDasharray="4,3"
        />
        <polygon points="218,101 210,106 215,112" fill={policyColor} />

        {/* Annotation: same as RLHF */}
        <rect
          x={60}
          y={225}
          width={400}
          height={40}
          rx={6}
          fill={accentColor}
          fillOpacity={0.06}
          stroke={accentColor}
          strokeWidth={1}
          strokeOpacity={0.2}
        />
        <text
          x={260}
          y={243}
          textAnchor="middle"
          fill={accentColor}
          fontSize="9"
          fontWeight="500"
        >
          Same RL loop as RLHF. The only difference: the reward signal.
        </text>
        <text
          x={260}
          y={257}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          RLHF: &quot;Is this helpful?&quot; (human preference) → Reasoning: &quot;Is the answer correct?&quot; (verifiable)
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Self-Consistency Worked Example
// ---------------------------------------------------------------------------

function SelfConsistencyExample() {
  const chains = [
    {
      id: 1,
      reasoning: 'Rate = $12/hr. Worked 5 hrs Mon, 3 hrs Tue, 4 hrs Wed = 12 hrs total. 12 × $12 = $144.',
      answer: 144,
      correct: true,
    },
    {
      id: 2,
      reasoning: 'Mon: $60, Tue: $36, Wed: $48. Total = $60 + $36 + $48 = $144.',
      answer: 144,
      correct: true,
    },
    {
      id: 3,
      reasoning: 'Total hours = 5 + 3 + 4 = 12. Earnings = 12 × 12 = $144.',
      answer: 144,
      correct: true,
    },
    {
      id: 4,
      reasoning: 'Mon: 5 × 12 = 60. Tue: 3 × 12 = 36. Wed: 4 × 12 = 46. Total = 60 + 36 + 46 = $142.',
      answer: 142,
      correct: false,
    },
    {
      id: 5,
      reasoning: '5 + 3 + 4 = 11 hours. 11 × $12 = $132.',
      answer: 132,
      correct: false,
    },
  ]

  return (
    <div className="space-y-4">
      <div className="py-3 px-4 bg-muted/30 rounded-lg">
        <p className="text-sm text-muted-foreground font-medium mb-2">
          Problem: &ldquo;Alex earns $12/hour. He worked 5 hours Monday, 3 hours Tuesday,
          and 4 hours Wednesday. How much did he earn?&rdquo;
        </p>
      </div>

      <div className="grid gap-2">
        {chains.map((chain) => (
          <div
            key={chain.id}
            className={`py-2 px-3 rounded-lg border text-sm ${
              chain.correct
                ? 'border-emerald-500/30 bg-emerald-500/5'
                : 'border-rose-500/30 bg-rose-500/5'
            }`}
          >
            <div className="flex items-start gap-2">
              <span className="text-muted-foreground font-mono text-xs mt-0.5">
                Chain {chain.id}:
              </span>
              <div className="flex-1">
                <span className="text-muted-foreground text-xs">
                  {chain.reasoning}
                </span>
              </div>
              <span
                className={`font-mono font-bold text-xs whitespace-nowrap ${
                  chain.correct ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                ${chain.answer}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="py-3 px-4 bg-violet-500/10 border border-violet-500/20 rounded-lg">
        <p className="text-sm text-muted-foreground">
          <strong className="text-violet-400">Majority vote:</strong>{' '}
          3 chains say $144, 1 says $142, 1 says $132.{' '}
          <strong>Answer: $144 (correct).</strong>{' '}
          Chain 4 has a multiplication error (4 &times; 12 = 46). Chain 5 has an
          addition error (5 + 3 + 4 = 11). Different chains make different errors.
          Voting averages out the noise.
        </p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function ReasoningModelsLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Reasoning Models"
            description="A base model with chain-of-thought sometimes reasons well and sometimes does not&mdash;the quality is left to chance. What if you trained the model to reason well? Reasoning models apply the same RL you know from RLHF, but with a different reward: &ldquo;did you get the right answer?&rdquo; And once a model can reason reliably, a new scaling axis opens up: instead of building bigger models, you can let the model think longer."
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
            This lesson teaches you how reinforcement learning trains models to
            generate effective reasoning chains, why spending more computation at
            inference time can substitute for larger models, and how this
            represents a paradigm shift from scaling model size to scaling
            inference computation.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How RL trains models to generate effective reasoning chains (conceptual mechanism)',
              'Outcome reward models (ORMs) vs process reward models (PRMs)',
              'Test-time compute scaling: trading inference compute for model size',
              'Self-consistency and best-of-N as search strategies during inference',
              'The paradigm shift from "scale the model" to "scale the inference compute"',
              'NOT: implementing RL for reasoning in code',
              'NOT: specific model architectures or training details of o1, DeepSeek-R1, etc.',
              'NOT: tree-of-thought or MCTS implementation details',
              'NOT: distillation of reasoning models into smaller models',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Module Capstone">
            This is the final lesson in the Reasoning &amp; ICL module. It
            connects the CoT mechanism (tokens as computation) with RL training
            (from RLHF). Two established frameworks, one new application.
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
            subtitle="Two established facts, not yet connected"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              From Chain-of-Thought Reasoning, you know that intermediate tokens
              give the model additional forward passes worth of computation.
              Each token triggers another full forward pass through N
              transformer blocks. More tokens = more computation per problem.
              But the base model has no training signal for reasoning
              quality&mdash;it generates reasoning tokens based on patterns
              absorbed during pretraining. Sometimes the reasoning is correct.
              Sometimes it is not. The quality is left to chance.
            </p>
            <p className="text-muted-foreground">
              From RLHF &amp; Alignment, you know that reinforcement learning
              can optimize model behavior toward a reward signal. The RLHF
              pipeline takes a pretrained model and trains it to generate
              responses that score well on a reward model trained from human
              preferences.
            </p>
            <p className="text-muted-foreground">
              What if the reward signal were not &ldquo;is this response helpful
              and harmless?&rdquo; but &ldquo;did you solve the math problem
              correctly?&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Pieces, One Application">
            You understand tokens-as-computation from the CoT lesson and RL
            training from RLHF. This lesson connects them: RL can train the
            model to use those extra tokens effectively.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: Before/After Contrast
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Same Architecture, Different Reliability"
            subtitle="Base model CoT vs reasoning model on the same problems"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider 5 attempts at 17 &times; 24 with a base model prompted
              to &ldquo;think step by step.&rdquo; You saw this problem in
              Chain-of-Thought Reasoning&mdash;now watch how the chains differ.
              The base model sometimes gets the arithmetic wrong (17 &times; 4 = 72?)
              or skips steps. The reasoning model produces structured,
              consistent decompositions every time.
            </p>

            <ComparisonRow
              left={{
                title: 'Base Model + CoT Prompt (3 of 5 attempts)',
                color: 'amber',
                items: [
                  '#1: 17×20=340, 17×4=68, 340+68=408 ✓',
                  '#2: 17×20=350, 17×4=68, 350+68=418 ✗',
                  '#3: 17×24... 17×20=340, 340+68=408 ✓',
                  '#4: 17×20=340, 17×4=72, 340+72=412 ✗',
                  '#5: 17×20=340, 17×4=68, 340+68=408 ✓',
                  'Result: 3/5 correct (60%)',
                ],
              }}
              right={{
                title: 'Reasoning Model (5 of 5 attempts)',
                color: 'emerald',
                items: [
                  '#1: Break 24 into 20+4. 17×20=340. 17×4=68. 340+68=408 ✓',
                  '#2: 24=20+4. 17×20=340. 17×4=68. Sum: 408 ✓',
                  '#3: 17×24: 17×20=340, 17×4=68, total=408 ✓',
                  '#4: Decompose: 17×20=340, 17×4=68. 340+68=408 ✓',
                  '#5: 24=20+4. 17(20)+17(4)=340+68=408 ✓',
                  'Result: 5/5 correct (100%)',
                ],
              }}
            />

            <GradientCard title="The Puzzle" color="orange">
              <p className="text-sm">
                Same architecture. Same number of parameters. Same problem. Both
                use chain-of-thought. The reasoning model produces consistently
                better chains&mdash;structured, checkable, and reliable. The
                base model&rsquo;s chains are hit-or-miss.{' '}
                <strong>What changed?</strong>
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Bigger Model">
            The reasoning model does not have more parameters. It was not
            trained on more pretraining data. The architecture is identical.
            The difference is in how it was trained after pretraining.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain Part 1: RL for Reasoning
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="RL for Reasoning"
            subtitle="Same RL loop, different reward signal"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The answer: training with reinforcement learning where the reward
              signal is answer correctness. The model generates a reasoning
              chain for a math problem. If the final answer is correct, it gets
              a positive reward. If incorrect, a negative reward. Over thousands
              of problems, the model&rsquo;s policy shifts toward generating
              chains that lead to correct answers.
            </p>

            <RLTrainingLoopDiagram />

            <CodeBlock
              code={`# RL training loop for reasoning (pseudocode)
# Compare to RLHF: same loop, different reward source

for problem in training_problems:
    # 1. Model generates a reasoning chain
    chain = model.generate(problem, use_cot=True)

    # 2. Extract the final answer from the chain
    predicted_answer = extract_answer(chain)

    # 3. Compute reward (verifiable, no human needed!)
    #    RLHF reward:     reward_model(response)        ← human preferences
    #    Reasoning reward: predicted == ground_truth     ← correctness
    reward = +1.0 if predicted_answer == ground_truth else -1.0

    # 4. Update policy to generate better chains
    loss = rl_loss(chain, reward, reference_model)
    optimizer.step(loss)`}
              language="python"
              filename="reasoning_rl.py"
            />

            <p className="text-muted-foreground">
              Look at this code alongside the RLHF pipeline you learned in RLHF
              &amp; Alignment. The loop structure is identical: generate,
              evaluate, compute reward, update. The only difference is where the
              reward comes from. In RLHF, a reward model trained on human
              preferences scores the response. In reasoning training, the reward
              is whether the answer is correct&mdash;no human needed for math
              and code problems where correctness is verifiable.
            </p>

            {/* "Of course" beat */}
            <GradientCard title="Of Course" color="violet">
              <p className="text-sm italic">
                You already knew that tokens are computation&mdash;each
                intermediate token triggers another forward pass. You already
                knew that RL can optimize behavior toward a reward signal. Of
                course you can use RL to optimize how the model generates
                reasoning tokens. The mechanism is not new. The application is.
              </p>
            </GradientCard>

            <GradientCard title="Training the Scratchpad" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In Chain-of-Thought Reasoning, you learned that intermediate
                  tokens are the model&rsquo;s scratchpad&mdash;external working
                  memory. RL training does not teach the model new knowledge. It
                  teaches the model to <strong>use the scratchpad
                  effectively</strong>.
                </p>
                <p className="text-muted-foreground">
                  Think of it as the difference between a student who scribbles
                  randomly on scratch paper and one who has been trained to show
                  structured, checkable work. Same paper. Same pen. Better use of
                  the scratchpad.
                </p>
              </div>
            </GradientCard>

            {/* Misconception 1 */}
            <GradientCard title="Misconception: &ldquo;Reasoning models are just bigger models&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  It is natural to assume better performance = more parameters.
                  That has been the dominant scaling paradigm for years (GPT-2
                  to GPT-3 to GPT-4). But reasoning models are not bigger. A 7B
                  reasoning model can outperform a 70B base model on math
                  benchmarks. Same architecture, same parameter count. The
                  difference is training&mdash;RL shapes how the model uses the
                  forward passes it already has.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pipeline Pattern">
            &ldquo;Same pipeline, different data source&rdquo;&mdash;you
            saw this in Constitutional AI. The pattern recurs: same RL
            machinery, applied to helpfulness (RLHF), then alignment
            (CAI), now reasoning.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1: Predict-and-Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Predict What Goes Wrong" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A reasoning model is trained with RL where the reward is
                &ldquo;final answer is correct.&rdquo;
              </p>

              <p><strong>Predict before revealing:</strong></p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>What kind of reasoning behavior does this reward incentivize?</li>
                <li>What might go wrong with outcome-only reward?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p>
                    The model learns to generate chains that reach correct
                    answers. But outcome-only reward has the same problem as
                    every proxy you saw in RLHF &amp; Alignment:
                  </p>
                  <ul className="list-disc list-inside space-y-1 ml-2 text-muted-foreground">
                    <li>
                      The model might learn <strong>shortcuts</strong>&mdash;lucky
                      guesses that happen to produce the right answer without
                      genuine reasoning
                    </li>
                    <li>
                      It cannot distinguish correct reasoning from{' '}
                      <strong>cancelling errors</strong>&mdash;a chain with a
                      sign error in step 3 that cancels with another error in
                      step 5 gets the same reward as 10 correct steps
                    </li>
                    <li>
                      This is the <strong>reward hacking</strong> pattern from
                      RLHF &amp; Alignment applied to reasoning
                    </li>
                  </ul>
                  <p className="text-muted-foreground">
                    Outcome reward has the same problem as every proxy: it can
                    be gamed. What if we could reward the reasoning process
                    itself?
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain Part 2: Process Supervision (DEVELOP from INTRODUCED)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Process Supervision"
            subtitle="Developing the distinction from Chain-of-Thought Reasoning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You met process supervision vs outcome supervision in
              Chain-of-Thought Reasoning: outcome evaluates the final answer,
              process evaluates each step. Now develop it&mdash;why does this
              distinction matter for training reasoning models?
            </p>

            <GradientCard title="Concrete Example: Same Answer, Different Reasoning" color="violet">
              <div className="space-y-3 text-sm">
                <p>
                  Two chains for the same problem, both reaching the correct
                  answer of 408:
                </p>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="py-2 px-3 bg-emerald-500/10 rounded border border-emerald-500/20">
                    <p className="font-medium text-emerald-400 text-xs mb-1">Chain A: Correct reasoning</p>
                    <p className="text-muted-foreground text-xs font-mono">
                      17 &times; 20 = 340<br />
                      17 &times; 4 = 68<br />
                      340 + 68 = 408
                    </p>
                  </div>
                  <div className="py-2 px-3 bg-rose-500/10 rounded border border-rose-500/20">
                    <p className="font-medium text-rose-400 text-xs mb-1">Chain B: Cancelling errors</p>
                    <p className="text-muted-foreground text-xs font-mono">
                      17 &times; 20 = 350 <span className="text-rose-400">(wrong)</span><br />
                      17 &times; 4 = 58 <span className="text-rose-400">(wrong)</span><br />
                      350 + 58 = 408 <span className="text-amber-400">(correct by luck)</span>
                    </p>
                  </div>
                </div>
                <p className="text-muted-foreground">
                  <strong>ORM:</strong> Both chains get the same reward (correct
                  answer). <strong>PRM:</strong> Chain A scores high on all
                  steps. Chain B gets penalized at steps 1 and 2. Chain
                  B&rsquo;s reasoning pattern will fail on a different problem
                  where the errors do not happen to cancel.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              That concrete example captures the distinction. Now generalize it:
            </p>

            <ComparisonRow
              left={{
                title: 'Outcome Reward Model (ORM)',
                color: 'amber',
                items: [
                  'Sees the full chain, scores the final answer',
                  'Cheap to train (just need answer labels)',
                  'Sparse reward: one signal for the entire chain',
                  'Gameable: correct answer does not imply correct reasoning',
                  'A chain with cancelling errors gets the same score as perfect reasoning',
                ],
              }}
              right={{
                title: 'Process Reward Model (PRM)',
                color: 'emerald',
                items: [
                  'Evaluates each step individually',
                  'Rich signal: feedback per step, not per chain',
                  'Trains models that reason correctly, not just get lucky',
                  'Requires step-level labels (expensive, but automatable)',
                  'Penalizes wrong steps even if final answer is correct',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Outcome supervision is the proxy. Process supervision is closer to
              the true objective. The same pattern you saw in RLHF &amp;
              Alignment: the more precisely you can specify what &ldquo;good&rdquo;
              means, the better the training signal. And recall the reward
              hacking lesson&mdash;outcome supervision for reasoning is exactly
              the kind of proxy a model can exploit.
            </p>

            {/* Misconception 3 */}
            <GradientCard title="Misconception: &ldquo;Process supervision means a human checks every step&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Process reward models (PRMs) are trained models that evaluate
                  steps automatically. The initial step-level labels may come from
                  humans or from automated verification, but once the PRM is
                  trained, it evaluates steps at scale without human involvement.
                </p>
                <p className="text-muted-foreground">
                  This is the same scaling pattern you saw in Constitutional
                  AI: human criteria get encoded into an AI evaluator that
                  operates at scale. Human signal &rarr; AI evaluator. The RLAIF
                  pattern applied to reasoning supervision.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Proxy vs True Objective">
            Outcome supervision asks &ldquo;did you get the right
            answer?&rdquo; Process supervision asks &ldquo;did you reason
            correctly at every step?&rdquo; The same proxy gap from reward
            hacking, applied to reasoning training.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain Part 3: Test-Time Compute Scaling
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Test-Time Compute Scaling"
            subtitle="A new axis of scaling opens up"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If reasoning models can generate effective chains, a new scaling
              dimension opens up. Until now, the dominant way to improve model
              performance has been to make the model bigger: more parameters
              means more computation per forward pass means better performance.
              GPT-2 had 1.5 billion parameters. GPT-3 had 175 billion.
              Each generation scaled up.
            </p>

            <p className="text-muted-foreground">
              But there is another way to increase total computation: make the
              model <strong>think longer</strong>. More reasoning tokens means
              more forward passes means more computation per problem. Same
              hardware. Same model. More compute.
            </p>

            <ScalingParadigmDiagram />

            <ComparisonRow
              left={{
                title: 'Traditional Scaling',
                color: 'amber',
                items: [
                  'Make the model bigger (more parameters)',
                  'More computation per forward pass',
                  'Fixed at training time',
                  'Same compute for every problem',
                  'Scaling cost: training + inference',
                ],
              }}
              right={{
                title: 'Test-Time Compute Scaling',
                color: 'violet',
                items: [
                  'Let the model think longer (more tokens)',
                  'More forward passes per problem',
                  'Adjustable at inference time',
                  'Variable compute per problem difficulty',
                  'Scaling cost: inference only',
                ],
              }}
            />

            <p className="text-muted-foreground">
              <strong>Bigger brain vs more thinking time.</strong> A student who
              thinks for 30 minutes outperforms a student who glances for 5
              seconds, even if the second student is &ldquo;smarter.&rdquo; You
              have experienced this yourself: sometimes a quick glance at the
              code is enough, sometimes you need to trace through carefully. The
              question becomes: when is it better to get a smarter student vs
              giving the current student more time?
            </p>

            {/* Misconception 4 */}
            <GradientCard title="Misconception: &ldquo;Test-time compute scaling means better hardware&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  &ldquo;Compute&rdquo; in everyday usage often means
                  hardware&mdash;GPUs, TPUs. &ldquo;Scaling compute&rdquo;
                  sounds like buying more GPUs. But test-time compute scaling
                  means generating more tokens through the same model on the same
                  hardware. A model running on identical hardware can use 10&times;
                  more compute on a hard problem by generating 10&times; more
                  reasoning tokens.
                </p>
                <p className="text-muted-foreground">
                  Connect this to the established mental model: &ldquo;tokens
                  are computation.&rdquo; More tokens = more forward passes =
                  more computation. The hardware is the same; the number of
                  forward passes changes.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              The key insight: on many problems, spending 10&times; more
              inference compute on a smaller reasoning model outperforms a
              10&times; larger base model that answers in one shot. A 7B
              reasoning model that generates 100 reasoning tokens can outperform
              a 70B base model on competition math. The smaller model runs more
              forward passes; the larger model runs fewer but each is more
              powerful. On problems that require multi-step reasoning, the
              additional forward passes win.
            </p>

            <p className="text-muted-foreground">
              OpenAI&rsquo;s o1 and DeepSeek-R1 are concrete examples of this
              approach. o1 generates internal reasoning tokens before producing
              a visible answer&mdash;the user sees only the final result, but
              the model ran many forward passes behind the scenes. DeepSeek-R1
              demonstrated that pure RL training (without supervised
              fine-tuning on reasoning examples) can produce effective
              reasoning behavior from scratch.
            </p>

            {/* Misconception 2 */}
            <GradientCard title="Misconception: &ldquo;The model decides to think harder on harder problems&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  It is tempting to anthropomorphize. When you face a hard
                  problem, you decide to think more carefully. But the model
                  does not &ldquo;decide&rdquo; anything. It generates tokens
                  according to a learned policy shaped by RL. On harder
                  problems, the policy produces longer chains because longer
                  chains on harder problems were rewarded during training.
                </p>
                <p className="text-muted-foreground">
                  The &ldquo;decision&rdquo; is a learned policy, not
                  deliberation. The same mechanism as every other
                  generation&mdash;sample from the distribution, append to
                  context, repeat. RL shaped which distributions get sampled.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Paradigm Shift">
            From &ldquo;how big is the model?&rdquo; to &ldquo;how much does
            the model think?&rdquo; Two independent axes of scaling. Echoes
            the &ldquo;axes not ladder&rdquo; framework from The Alignment
            Techniques Landscape.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Explain Part 4: Search During Inference
              (DEVELOP self-consistency from MENTIONED)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Search During Inference"
            subtitle="Developing self-consistency from Chain-of-Thought Reasoning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Chain-of-Thought Reasoning, self-consistency was
              mentioned: &ldquo;generate multiple chains, take the
              majority vote.&rdquo; Now develop the mechanism.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-muted-foreground text-sm font-medium">
                Self-consistency in three steps:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-1 text-sm">
                <li>
                  Sample N reasoning chains for the same problem (different
                  random seeds produce different chains)
                </li>
                <li>Extract the final answer from each chain</li>
                <li>Take the majority vote</li>
              </ol>
            </div>

            <SelfConsistencyExample />

            <p className="text-muted-foreground">
              Self-consistency generalizes beyond simple arithmetic. Consider a
              harder problem: &ldquo;A train leaves City A at 60 mph. Two hours
              later, a second train leaves City A at 90 mph. How far from City A
              do they meet?&rdquo; One chain sets up an algebra equation
              (60t = 90(t &minus; 2)), another uses a rate-of-closing approach
              (the gap is 120 miles, closing at 30 mph, so they meet after 4
              more hours), and a third builds a table of positions at each hour.
              Three genuinely different reasoning strategies&mdash;but all three
              converge on 360 miles. The majority vote works not because the
              chains are similar, but because correct reasoning, regardless of
              approach, converges on the same answer.
            </p>

            <p className="text-muted-foreground">
              Why does this work? If each chain has a probability greater than
              50% of being correct, then N independent chains with majority
              voting have a higher probability of producing the correct answer
              than any single chain. This is the same principle as asking 5
              people and going with the majority&mdash;individual errors average
              out when they are independent.
            </p>

            <p className="text-muted-foreground">
              Connect to error propagation from Chain-of-Thought Reasoning: one
              chain can have an error in step 3. A different chain makes a
              different error, or no error at all. Voting averages out the noise
              because different chains make different mistakes.
            </p>

            <div className="space-y-4">
              <p className="text-muted-foreground">
                <strong>Best-of-N with a verifier:</strong> instead of majority
                vote, use a reward model (ORM or PRM) to score each chain and
                select the best one. This is search: generate candidates,
                evaluate, select.
              </p>

              <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
                <li>
                  <strong>Self-consistency</strong> = search with majority-vote
                  selection
                </li>
                <li>
                  <strong>Best-of-N</strong> = search with verifier selection
                </li>
                <li>
                  Both are forms of <strong>test-time compute
                  scaling</strong>: generating more chains = spending more
                  inference compute
                </li>
              </ul>
            </div>

            {/* Negative example: reasoning model overthinking */}
            <GradientCard title="Negative Example: When Reasoning Backfires" color="rose">
              <div className="space-y-3 text-sm">
                <p className="font-medium">
                  Prompt: &ldquo;What is the capital of France?&rdquo;
                </p>
                <div className="py-3 px-4 bg-muted/30 rounded-lg font-mono text-xs space-y-1">
                  <p className="text-muted-foreground">
                    Well, France is a country in Western Europe. Its capital is Paris.
                  </p>
                  <p className="text-muted-foreground">
                    But wait&mdash;historically, the capital has moved. During WWII,
                    the Vichy government was based in Vichy. And before that, Versailles
                    served as the seat of government at various points...
                  </p>
                  <p className="text-muted-foreground">
                    However, the modern capital is <span className="text-rose-400">definitely Paris.
                    Or is it? Let me reconsider&mdash;yes, Paris is the current capital.</span>
                  </p>
                </div>
                <p className="text-muted-foreground">
                  A factual recall question answered correctly in one forward pass.
                  The reasoning model&rsquo;s learned policy generates a long chain
                  anyway&mdash;introducing doubt, wasting tokens, and risking a
                  confused answer. Even a model <em>trained</em> to reason can
                  over-apply reasoning to problems that do not need it.
                </p>
              </div>
            </GradientCard>

            {/* Misconception 5 */}
            <GradientCard title="Misconception: &ldquo;More reasoning tokens are always better&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Diminishing returns are real. Going from 1 to 5 chains is a
                  large improvement. Going from 50 to 100 chains is marginal.
                  And the compute cost is linear&mdash;100 chains cost 100&times;
                  more than 1 chain.
                </p>
                <p className="text-muted-foreground">
                  The example above shows it concretely: on simple problems, long
                  reasoning chains introduce second-guessing and waste compute.
                  The optimal amount of reasoning depends on problem
                  difficulty&mdash;this is precisely what test-time compute
                  scaling calibrates.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Search as Scaling">
            Self-consistency and best-of-N are forms of search: generate
            multiple candidates, select the best. Each additional chain is
            more inference compute. Search trades compute for reliability.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Check 2: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Apply the Concepts" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A company wants to deploy an LLM for customer support. They can
                either: (A) use a large base model (70B) that answers
                immediately, or (B) use a smaller reasoning model (7B) that
                generates a reasoning chain before answering. The questions are
                a mix of simple FAQ lookups (60%) and complex troubleshooting
                (40%).
              </p>

              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>Which approach is better for each type of question?</li>
                <li>Would you use the same strategy for both?</li>
                <li>What would a hybrid approach look like?</li>
                <li>What does process supervision look like in this context?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p>
                    <strong>Simple FAQs:</strong> Direct answer is fine.
                    Reasoning chains add latency and cost with no accuracy
                    benefit. This is the computational complexity criterion from
                    Chain-of-Thought Reasoning&mdash;FAQ lookups fit in a single
                    forward pass.
                  </p>
                  <p>
                    <strong>Complex troubleshooting:</strong> The reasoning model
                    outperforms. Structured chains are checkable, and the model
                    can decompose multi-step diagnostic processes. This exceeds
                    single-pass capacity.
                  </p>
                  <p>
                    <strong>Hybrid approach:</strong> Route by estimated
                    complexity. Simple questions get direct answers (less
                    compute, lower latency). Complex questions get reasoning
                    chains (more compute, higher accuracy). This is adaptive
                    compute allocation&mdash;the core of test-time compute
                    scaling.
                  </p>
                  <p className="text-muted-foreground">
                    <strong>Process supervision:</strong> Flag steps where the
                    PRM is uncertain and escalate to human review. On
                    troubleshooting queries, a PRM can identify reasoning steps
                    that are unreliable, rather than waiting for the customer to
                    report a wrong answer (outcome supervision).
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          11. Practice: Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Explore reasoning models empirically"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Four exercises that explore the concepts from this lesson with
              real models. Each builds on the previous but can be done
              independently.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Base Model CoT vs Reasoning Model (Guided)" color="blue">
                <p className="text-sm">
                  Give the same 10 math/reasoning problems to a base model with
                  CoT prompting and a reasoning-focused model. Compare accuracy
                  and inspect reasoning chain quality&mdash;not just final
                  answers. Predict which problems the base model gets wrong that
                  the reasoning model gets right.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Self-Consistency Experiment (Supported)" color="blue">
                <p className="text-sm">
                  For 5 reasoning problems, generate N chains (N = 1, 3, 5, 10,
                  20) and compute majority-vote accuracy at each N. Plot accuracy
                  vs N. Identify the point of diminishing returns. Try with both
                  easy and hard problems.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Process vs Outcome Evaluation (Supported)" color="blue">
                <p className="text-sm">
                  For 5 problems with known step-by-step solutions, generate
                  reasoning chains and evaluate two ways: (a) does the final
                  answer match? and (b) is each step correct? Find cases where
                  the outcome is correct but a step is wrong.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: Test-Time Compute Allocation (Independent)" color="blue">
                <p className="text-sm">
                  Design an experiment: given a fixed compute budget, compare
                  equal allocation (same reasoning tokens per problem) vs
                  adaptive allocation (more tokens for harder problems, fewer for
                  easy ones). Measure overall accuracy. No skeleton
                  provided&mdash;you design the experiment.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                  Each exercise explores a different aspect of reasoning models
                  with immediate, empirical feedback.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-2-4-reasoning-models.ipynb"
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
            Exercise 1 establishes the empirical difference. Exercise 2
            quantifies self-consistency. Exercise 3 reveals the process/outcome
            gap. Exercise 4 tests adaptive compute allocation.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'Reasoning models apply RL to train effective use of chain-of-thought.',
                description:
                  'Same RL loop as RLHF, different reward signal. Instead of "is this helpful?" the reward is "is the answer correct?" The model learns to generate structured, reliable reasoning chains.',
              },
              {
                headline:
                  'Process supervision trains better reasoning than outcome supervision.',
                description:
                  'Evaluating steps, not just answers. Outcome supervision is the proxy; process supervision is closer to the true objective. The reward hacking lesson applied to reasoning.',
              },
              {
                headline:
                  'Test-time compute scaling is a new scaling axis: from "how big is the model?" to "how much does the model think?"',
                description:
                  'Instead of bigger models, let models think longer. More reasoning tokens = more forward passes = more computation. A 7B reasoning model can outperform a 70B base model by spending more inference compute. Model size and inference compute are two independent axes of scaling.',
              },
              {
                headline:
                  'Search during inference trades compute for reliability.',
                description:
                  'Self-consistency (majority vote) and best-of-N (verifier selection) generate multiple reasoning chains and select the best. Each additional chain is more inference compute, with diminishing returns.',
              },
              {
                headline:
                  'The optimal amount of reasoning depends on problem difficulty.',
                description:
                  'Simple tasks need less thinking; complex tasks benefit from more. Adaptive compute allocation outperforms uniform allocation. This is what test-time compute scaling calibrates.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <GradientCard title="Mental Model" color="violet">
            <p className="text-sm italic">
              The forward pass budget is fixed. CoT spends more budget by
              generating more tokens. RL trains the model to spend that budget
              wisely. And test-time compute scaling says: if spending more
              budget helps, why not give it as much budget as it needs?
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Module Complete
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="5.2"
            title="Reasoning & In-Context Learning"
            achievements={[
              'In-context learning as attention-based task learning from examples',
              'Prompt engineering as structured programming for LLMs',
              'Chain-of-thought: tokens as computation via the autoregressive loop',
              'Reasoning models: RL-trained CoT with test-time compute scaling',
              'Process supervision vs outcome supervision for reasoning quality',
              'Self-consistency and search during inference',
            ]}
            nextModule="6.1"
            nextTitle="Generative Foundations"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          14. References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Training Verifiers to Solve Math Word Problems',
                authors: 'Cobbe et al., 2021',
                url: 'https://arxiv.org/abs/2110.14168',
                note: 'Introduces the GSM8K benchmark and outcome-based verification for math reasoning. Section 3 covers the verifier approach that motivated later process supervision work.',
              },
              {
                title: "Let's Verify Step by Step",
                authors: 'Lightman et al., 2023',
                url: 'https://arxiv.org/abs/2305.20050',
                note: 'The OpenAI paper showing process supervision outperforms outcome supervision for mathematical reasoning. Directly relevant to the ORM vs PRM comparison in this lesson.',
              },
              {
                title: 'Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters',
                authors: 'Snell et al., 2024',
                url: 'https://arxiv.org/abs/2408.03314',
                note: 'Formalizes test-time compute scaling. Shows when inference compute scaling outperforms model scaling. Section 1 provides the core argument.',
              },
              {
                title: 'Self-Consistency Improves Chain of Thought Reasoning in Language Models',
                authors: 'Wang et al., 2022',
                url: 'https://arxiv.org/abs/2203.11171',
                note: 'Introduces majority voting over multiple CoT chains. The self-consistency method developed from MENTIONED in this lesson. Section 1 for the core idea.',
              },
              {
                title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning',
                authors: 'DeepSeek-AI, 2025',
                url: 'https://arxiv.org/abs/2501.12948',
                note: 'A concrete example of RL-trained reasoning. Shows that pure RL (without SFT) can produce reasoning behavior. Read Section 1 for the approach.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          15. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Series complete"
            description="From in-context learning to trained reasoning&mdash;each technique works because of a specific, mechanistic reason. Next up: generative foundations, starting with the shift from classification to generation."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
