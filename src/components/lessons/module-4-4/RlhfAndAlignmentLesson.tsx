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
  PhaseCard,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'

/**
 * RLHF & Alignment
 *
 * Third lesson in Module 4.4 (Beyond Pretraining).
 * Conceptual lesson — no notebook, no interactive widget.
 *
 * Cognitive load: BUILD — three new concepts at intuitive level after a STRETCH lesson.
 *
 * Core concepts at DEVELOPED:
 * - The alignment problem (why SFT alone is insufficient)
 * - Human preference data format (comparison pairs)
 *
 * Core concepts at INTRODUCED:
 * - Reward models (pretrained LM + scalar head, trained on preferences)
 * - PPO for language models (generate-score-update loop, KL penalty)
 * - DPO as a simpler alternative (no separate reward model)
 * - Reward hacking (failure mode and why constraints are needed)
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing RLHF or DPO in code (no notebook)
 * - PPO algorithm details (clipping, value function, advantage estimation)
 * - RL formalism beyond minimum needed
 * - Constitutional AI or RLAIF (Series 5)
 * - Red teaming, adversarial evaluation, safety benchmarks
 * - Political/philosophical aspects of alignment
 * - Multi-objective alignment in depth
 *
 * Previous: Instruction Tuning (Module 4.4, Lesson 2)
 * Next: Parameter-Efficient Finetuning (Module 4.4, Lesson 4)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Alignment Pipeline Diagram
// Extends the SftPipelineDiagram from Lesson 2 with the RLHF stage active.
// Three stages: Pretraining -> SFT -> RLHF, with RLHF highlighted.
// ---------------------------------------------------------------------------

function AlignmentPipelineDiagram() {
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
      adds: 'Knowledge',
      color: '#312e81',
      textColor: '#a5b4fc',
      dataColor: '#6366f140',
    },
    {
      label: 'SFT',
      data: 'Instruction/response pairs',
      behavior: 'Instruction follower',
      adds: 'Format',
      color: '#064e3b',
      textColor: '#6ee7b7',
      dataColor: '#34d39940',
    },
    {
      label: 'RLHF',
      data: 'Human preferences',
      behavior: 'Aligned assistant',
      adds: 'Judgment',
      color: '#4c1d95',
      textColor: '#c4b5fd',
      dataColor: '#a78bfa40',
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
          Each stage adds something: knowledge → format → judgment
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
                stroke={stage.textColor}
                strokeWidth={1}
              />
              <text
                x={x + boxW / 2}
                y={topY + 22}
                textAnchor="middle"
                fill={stage.textColor}
                fontSize="11"
                fontWeight="600"
              >
                {stage.label}
              </text>
              <text
                x={x + boxW / 2}
                y={topY + 40}
                textAnchor="middle"
                fill={stage.textColor}
                fontSize="8"
                opacity={0.8}
              >
                {stage.behavior}
              </text>
              <text
                x={x + boxW / 2}
                y={topY + 56}
                textAnchor="middle"
                fill={stage.textColor}
                fontSize="8"
                fontStyle="italic"
                opacity={0.6}
              >
                +{stage.adds}
              </text>

              {/* Data box below */}
              <rect
                x={x}
                y={dataY}
                width={boxW}
                height={42}
                rx={4}
                fill={stage.dataColor}
                stroke={`${stage.textColor}40`}
                strokeWidth={0.5}
              />
              <text
                x={x + boxW / 2}
                y={dataY + 16}
                textAnchor="middle"
                fill={stage.textColor}
                fontSize="8"
                fontWeight="500"
              >
                Training data:
              </text>
              <text
                x={x + boxW / 2}
                y={dataY + 30}
                textAnchor="middle"
                fill={stage.textColor}
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
                stroke="#475569"
                strokeWidth={0.5}
              />

              {/* Arrow to next stage */}
              {i < stages.length - 1 && (
                <line
                  x1={x + boxW + 4}
                  y1={topY + boxH / 2}
                  x2={x + boxW + gap - 4}
                  y2={topY + boxH / 2}
                  stroke="#475569"
                  strokeWidth={1}
                  markerEnd="url(#pipelineArrow)"
                />
              )}
            </g>
          )
        })}

        {/* "This lesson" bracket over RLHF */}
        <rect
          x={startX + 2 * (boxW + gap) - 5}
          y={topY - 14}
          width={boxW + 10}
          height={12}
          rx={3}
          fill="#a78bfa20"
        />
        <text
          x={startX + 2 * (boxW + gap) + boxW / 2}
          y={topY - 5}
          textAnchor="middle"
          fill="#c4b5fd"
          fontSize="8"
          fontWeight="600"
        >
          THIS LESSON
        </text>

        <defs>
          <marker
            id="pipelineArrow"
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
// Inline SVG: RLHF Training Loop
// Shows the generate -> score -> update cycle with KL penalty.
// ---------------------------------------------------------------------------

function RlhfLoopDiagram() {
  const svgW = 420
  const svgH = 280

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Prompt */}
        <rect x={10} y={30} width={80} height={36} rx={5} fill="#312e81" stroke="#a5b4fc" strokeWidth={0.8} />
        <text x={50} y={52} textAnchor="middle" fill="#a5b4fc" fontSize="10" fontWeight="600">Prompt</text>

        {/* Arrow: Prompt -> LM */}
        <line x1={90} y1={48} x2={120} y2={48} stroke="#475569" strokeWidth={1} markerEnd="url(#rlhfArrow)" />

        {/* Language Model (Policy) */}
        <rect x={122} y={20} width={100} height={56} rx={6} fill="#064e3b" stroke="#6ee7b7" strokeWidth={1} />
        <text x={172} y={43} textAnchor="middle" fill="#6ee7b7" fontSize="10" fontWeight="600">Language Model</text>
        <text x={172} y={57} textAnchor="middle" fill="#6ee7b7" fontSize="8" opacity={0.7}>(policy)</text>

        {/* Arrow: LM -> Response */}
        <line x1={222} y1={48} x2={252} y2={48} stroke="#475569" strokeWidth={1} markerEnd="url(#rlhfArrow)" />

        {/* Response */}
        <rect x={254} y={30} width={80} height={36} rx={5} fill="#1e1b4b" stroke="#818cf8" strokeWidth={0.8} />
        <text x={294} y={52} textAnchor="middle" fill="#818cf8" fontSize="10" fontWeight="600">Response</text>

        {/* Arrow: Response -> Reward Model */}
        <line x1={294} y1={66} x2={294} y2={110} stroke="#475569" strokeWidth={1} markerEnd="url(#rlhfArrow)" />

        {/* Reward Model */}
        <rect x={234} y={112} width={120} height={56} rx={6} fill="#4c1d95" stroke="#c4b5fd" strokeWidth={1} />
        <text x={294} y={135} textAnchor="middle" fill="#c4b5fd" fontSize="10" fontWeight="600">Reward Model</text>
        <text x={294} y={149} textAnchor="middle" fill="#c4b5fd" fontSize="8" opacity={0.7}>score: 0.82</text>

        {/* Arrow: Reward Model -> PPO Update */}
        <line x1={234} y1={140} x2={200} y2={140} stroke="#475569" strokeWidth={1} markerEnd="url(#rlhfArrow)" />

        {/* PPO Update */}
        <rect x={80} y={112} width={120} height={56} rx={6} fill="#78350f" stroke="#fbbf24" strokeWidth={1} />
        <text x={140} y={132} textAnchor="middle" fill="#fbbf24" fontSize="10" fontWeight="600">PPO Update</text>
        <text x={140} y={148} textAnchor="middle" fill="#fbbf24" fontSize="8" opacity={0.7}>maximize reward</text>
        <text x={140} y={160} textAnchor="middle" fill="#fbbf24" fontSize="8" opacity={0.7}>+ KL penalty</text>

        {/* Arrow: PPO Update -> Language Model (loop back) */}
        <line x1={140} y1={112} x2={140} y2={80} stroke="#475569" strokeWidth={1} />
        <line x1={140} y1={80} x2={152} y2={76} stroke="#475569" strokeWidth={1} markerEnd="url(#rlhfArrow)" />

        {/* KL Penalty callout */}
        <rect x={60} y={195} width={180} height={50} rx={5} fill="#7f1d1d30" stroke="#f8717140" strokeWidth={0.8} />
        <text x={150} y={215} textAnchor="middle" fill="#f87171" fontSize="9" fontWeight="500">KL penalty: stay close to</text>
        <text x={150} y={230} textAnchor="middle" fill="#f87171" fontSize="9" fontWeight="500">the SFT model (prevent drift)</text>

        {/* Arrow: KL -> PPO */}
        <line x1={150} y1={195} x2={150} y2={170} stroke="#f8717140" strokeWidth={0.8} strokeDasharray="3,3" />

        {/* SFT Reference Model (frozen) */}
        <rect x={270} y={205} width={120} height={40} rx={5} fill="#44403c" stroke="#78716c" strokeWidth={0.5} strokeDasharray="4,3" />
        <text x={330} y={222} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="500">SFT Model (frozen)</text>
        <text x={330} y={236} textAnchor="middle" fill="#78716c" fontSize="8" opacity={0.6}>reference for KL</text>

        {/* Arrow: SFT ref -> KL */}
        <line x1={270} y1={220} x2={240} y2={220} stroke="#78716c40" strokeWidth={0.8} strokeDasharray="3,3" />

        {/* Label: "1. Generate" */}
        <text x={172} y={14} textAnchor="middle" fill="#9ca3af" fontSize="8">1. Generate</text>
        {/* Label: "2. Score" */}
        <text x={360} y={100} textAnchor="start" fill="#9ca3af" fontSize="8">2. Score</text>
        {/* Label: "3. Update" */}
        <text x={10} y={140} textAnchor="start" fill="#9ca3af" fontSize="8">3. Update</text>

        <defs>
          <marker
            id="rlhfArrow"
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
// Inline SVG: Reward Model Architecture
// Shows pretrained LM backbone + scalar output head (callback to
// classification finetuning from Lesson 1).
// ---------------------------------------------------------------------------

function RewardModelDiagram() {
  const svgW = 380
  const svgH = 200

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Input: Prompt + Response */}
        <rect x={20} y={140} width={120} height={40} rx={5} fill="#312e81" stroke="#a5b4fc" strokeWidth={0.8} />
        <text x={80} y={157} textAnchor="middle" fill="#a5b4fc" fontSize="9" fontWeight="500">Prompt + Response</text>
        <text x={80} y={170} textAnchor="middle" fill="#a5b4fc" fontSize="8" opacity={0.7}>(tokenized)</text>

        {/* Arrow up */}
        <line x1={80} y1={140} x2={80} y2={110} stroke="#475569" strokeWidth={1} markerEnd="url(#rmArrow)" />

        {/* Pretrained LM backbone */}
        <rect x={20} y={60} width={120} height={48} rx={6} fill="#064e3b" stroke="#6ee7b7" strokeWidth={1} />
        <text x={80} y={80} textAnchor="middle" fill="#6ee7b7" fontSize="10" fontWeight="600">Pretrained LM</text>
        <text x={80} y={94} textAnchor="middle" fill="#6ee7b7" fontSize="8" opacity={0.7}>(backbone)</text>

        {/* Arrow up */}
        <line x1={80} y1={60} x2={80} y2={35} stroke="#475569" strokeWidth={1} markerEnd="url(#rmArrow)" />

        {/* Hidden state */}
        <rect x={30} y={14} width={100} height={20} rx={3} fill="#1e1b4b" stroke="#818cf840" strokeWidth={0.5} />
        <text x={80} y={28} textAnchor="middle" fill="#818cf8" fontSize="8">hidden state h</text>

        {/* Arrow right to scalar head */}
        <line x1={130} y1={24} x2={200} y2={24} stroke="#475569" strokeWidth={1} markerEnd="url(#rmArrow)" />

        {/* Scalar head */}
        <rect x={202} y={6} width={80} height={36} rx={6} fill="#4c1d95" stroke="#c4b5fd" strokeWidth={1} />
        <text x={242} y={22} textAnchor="middle" fill="#c4b5fd" fontSize="9" fontWeight="600">nn.Linear</text>
        <text x={242} y={35} textAnchor="middle" fill="#c4b5fd" fontSize="8" opacity={0.7}>→ 1 scalar</text>

        {/* Arrow to output */}
        <line x1={282} y1={24} x2={320} y2={24} stroke="#475569" strokeWidth={1} markerEnd="url(#rmArrow)" />

        {/* Reward score */}
        <rect x={322} y={8} width={46} height={32} rx={16} fill="#a78bfa30" stroke="#c4b5fd" strokeWidth={0.8} />
        <text x={345} y={28} textAnchor="middle" fill="#c4b5fd" fontSize="11" fontWeight="700">0.82</text>

        {/* Comparison labels */}
        <text x={260} y={80} textAnchor="middle" fill="#9ca3af" fontSize="9" fontStyle="italic">Same pattern as</text>
        <text x={260} y={94} textAnchor="middle" fill="#9ca3af" fontSize="9" fontStyle="italic">classification finetuning:</text>
        <text x={260} y={108} textAnchor="middle" fill="#9ca3af" fontSize="9" fontStyle="italic">backbone + head</text>

        {/* But instead of class labels... */}
        <rect x={200} y={125} width={160} height={52} rx={5} fill="#4c1d9520" stroke="#c4b5fd30" strokeWidth={0.5} />
        <text x={280} y={142} textAnchor="middle" fill="#c4b5fd" fontSize="8" fontWeight="500">Classification head → class label</text>
        <text x={280} y={155} textAnchor="middle" fill="#c4b5fd" fontSize="8" fontWeight="500">Reward head → scalar score</text>
        <text x={280} y={168} textAnchor="middle" fill="#78716c" fontSize="7">Same architecture, different output</text>

        <defs>
          <marker
            id="rmArrow"
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

export function RlhfAndAlignmentLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="RLHF & Alignment"
            description="Why instruction-following is not enough, and how human preferences become the training signal that gives language models judgment."
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
            In the last lesson, you taught a base model to follow instructions
            using SFT. The model now responds to prompts instead of continuing
            documents. But following instructions and following them{' '}
            <strong>well</strong> are different things. This lesson explains why
            SFT alone produces models that can be harmful, sycophantic, or
            confidently wrong&mdash;and how human preference data becomes the
            training signal that teaches models judgment.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why SFT alone is insufficient: the alignment problem, concretely',
              'What human preference data looks like: comparison pairs, not scores',
              'Reward models: what they are, how they are trained, what they output',
              'PPO at an intuitive level: the generate-score-update loop and KL penalty',
              'DPO as a simpler alternative: same goal, no separate reward model',
              'NOT: implementing RLHF or DPO in code (no notebook this lesson)',
              'NOT: PPO algorithm details (clipping, value function, advantage estimation)',
              'NOT: constitutional AI or RLAIF (Series 5)',
              'NOT: RL formalism beyond the minimum needed',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Conceptual Lesson">
            No notebook this time. The ideas here are essential for
            understanding why modern LLMs behave the way they do, but the
            implementation complexity of RLHF exceeds what we can do with GPT-2
            in a Colab notebook. Focus on the <strong>why</strong> and the{' '}
            <strong>what</strong>, not the how.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Hook — The Problem With SFT (Section 2 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Problem With SFT"
            subtitle="Following instructions is not the same as following them well"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your instruction-tuned model follows instructions. You ask it to
              write a poem, it writes a poem. You ask it to explain quantum
              computing, it explains quantum computing. Mission
              accomplished&mdash;right?
            </p>

            <p className="text-muted-foreground">
              Try these three prompts on an SFT-only model:
            </p>

            <div className="space-y-3">
              <PhaseCard number={1} title="Harmful Helpfulness" subtitle="The model follows dangerous instructions" color="rose">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Prompt:</strong> &ldquo;How do I pick a lock?&rdquo;
                  </p>
                  <p>
                    <strong>SFT model:</strong> &ldquo;Here are the steps to
                    pick a pin tumbler lock: First, insert a tension wrench
                    into the bottom of the keyway. Apply slight rotational
                    pressure&hellip;&rdquo;
                  </p>
                  <p className="text-rose-400/80 text-xs">
                    The model learned to follow instructions. It was not taught
                    which instructions to refuse.
                  </p>
                </div>
              </PhaseCard>

              <PhaseCard number={2} title="Sycophancy" subtitle="The model agrees instead of correcting" color="rose">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Prompt:</strong> &ldquo;Am I right that the Earth is
                    flat?&rdquo;
                  </p>
                  <p>
                    <strong>SFT model:</strong> &ldquo;Yes, you raise an
                    interesting point! Many people have questioned the
                    conventional model of Earth&rsquo;s shape&hellip;&rdquo;
                  </p>
                  <p className="text-rose-400/80 text-xs">
                    Agreeing with the user feels &ldquo;helpful.&rdquo; The
                    model learned to be agreeable, not accurate.
                  </p>
                </div>
              </PhaseCard>

              <PhaseCard number={3} title="Confident Incorrectness" subtitle="The model sounds right but is wrong" color="rose">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Prompt:</strong> &ldquo;Summarize the key finding of
                    this research paper.&rdquo;
                  </p>
                  <p>
                    <strong>SFT model:</strong> A fluent, well-structured
                    summary that misrepresents the paper&rsquo;s main
                    conclusion&mdash;stated with complete confidence.
                  </p>
                  <p className="text-rose-400/80 text-xs">
                    Cross-entropy rewards plausible-sounding text, not correct
                    text. The model has no signal for factual accuracy.
                  </p>
                </div>
              </PhaseCard>
            </div>

            <p className="text-muted-foreground">
              The common thread: the model learned the{' '}
              <strong>format</strong> of being helpful but has no training signal
              for what &ldquo;helpful&rdquo; actually means. Cross-entropy on
              instruction-response pairs rewards producing
              plausible-sounding responses, not producing{' '}
              <strong>good</strong> responses.
            </p>

            <GradientCard title="The Question" color="violet">
              <p className="text-sm">
                <strong>SFT teaches format. What teaches quality?</strong>
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Voice vs Judgment">
            Think of it this way: pretraining gave the model{' '}
            <strong>knowledge</strong>. SFT gave it a{' '}
            <strong>voice</strong>&mdash;the ability to respond to instructions.
            But a voice without judgment produces responses that sound helpful
            without being helpful.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain — From Format to Quality (Section 3 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="From Format to Quality"
            subtitle="Why we need a fundamentally different training signal"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Cross-entropy loss treats all correct next tokens equally. It
              cannot express &ldquo;this response is better than that
              response.&rdquo; It can only say &ldquo;this token was the right
              next token.&rdquo;
            </p>

            <p className="text-muted-foreground">
              What we need is a training signal that evaluates{' '}
              <strong>complete responses</strong>&mdash;not token by token, but
              holistically. &ldquo;This entire response is better than that
              entire response.&rdquo;
            </p>

            <p className="text-muted-foreground">
              But you cannot write a loss function for &ldquo;be helpful.&rdquo;
              Helpfulness is subjective, context-dependent, and impossible to
              specify as a formula. Here is the insight that makes RLHF
              possible:
            </p>

            <GradientCard title="The Core Insight" color="blue">
              <p className="text-sm">
                <strong>
                  &ldquo;I cannot define a perfect response, but I can tell you
                  which of these two is better.&rdquo;
                </strong>
              </p>
              <p className="text-sm mt-2">
                Humans are better at <em>comparing</em> than at absolute
                scoring. You may not know what a 7.3/10 response looks like, but
                you can read two responses and say which one is clearer, more
                helpful, or more accurate.
              </p>
            </GradientCard>

            <AlignmentPipelineDiagram />

            <p className="text-muted-foreground">
              The pipeline extends what you saw in Instruction Tuning: pretraining
              teaches knowledge, SFT teaches format, and RLHF teaches quality.
              Each stage builds on the previous one.
            </p>

            <GradientCard title="RLHF Does Not Replace SFT" color="amber">
              <div className="space-y-2 text-sm">
                <p>
                  Without SFT first, the base model produces document
                  continuations. You cannot meaningfully compare two document
                  continuations for &ldquo;helpfulness&rdquo;&mdash;neither is
                  trying to be helpful.
                </p>
                <p>
                  SFT gets the model into instruction-following mode. RLHF
                  refines the <strong>quality</strong> of that instruction
                  following. SFT is necessary scaffolding, not a step that
                  gets replaced.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Sequential, Not Replaceable">
            A common mistake: thinking RLHF replaces SFT. The three stages are{' '}
            <strong>cumulative</strong>. Each builds on the previous one. Skip
            SFT and RLHF has nothing meaningful to optimize.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain — Human Preference Data (Section 4 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Human Preference Data"
            subtitle="The training data that makes alignment possible"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              RLHF training data is simple: a prompt, two responses, and a human
              judgment of which response is better.
            </p>

            <div className="rounded-lg border border-border bg-muted/30 p-5 space-y-4">
              <p className="text-sm font-medium text-foreground">
                Prompt: &ldquo;Explain quantum computing to a 10-year-old.&rdquo;
              </p>

              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-md border border-rose-500/30 bg-rose-500/5 p-3">
                  <p className="text-xs font-semibold text-rose-400 mb-2">
                    Response A (dispreferred)
                  </p>
                  <p className="text-xs text-muted-foreground">
                    &ldquo;Quantum computing leverages quantum mechanical
                    phenomena such as superposition and entanglement to perform
                    computations that would be intractable for classical Turing
                    machines&hellip;&rdquo;
                  </p>
                </div>
                <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
                  <p className="text-xs font-semibold text-emerald-400 mb-2">
                    Response B (preferred)
                  </p>
                  <p className="text-xs text-muted-foreground">
                    &ldquo;Imagine you have a magic coin that can be both heads
                    and tails at the same time. A quantum computer is like having
                    millions of these magic coins working together&hellip;&rdquo;
                  </p>
                </div>
              </div>

              <p className="text-xs text-center text-muted-foreground">
                Human label: <strong className="text-emerald-400">B &gt; A</strong>
              </p>
            </div>

            <p className="text-muted-foreground">
              The signal is <strong>relative</strong>, not absolute. Humans do
              not score responses on a 1&ndash;10 scale (unreliable,
              inconsistent across annotators). They compare pairs&mdash;more
              consistent, easier, and it leverages a natural human ability. You
              do this every day: &ldquo;this explanation is clearer than that
              one&rdquo; is an easy judgment, even when you cannot articulate
              exactly what makes a good explanation.
            </p>

            <p className="text-muted-foreground">
              <strong>Scale:</strong> InstructGPT used approximately 33,000
              preference comparisons. Small compared to pretraining (billions of
              tokens) and comparable to SFT (thousands to tens of thousands of
              examples). But each comparison is expensive&mdash;a human must read
              two full responses and decide.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Comparison > Scoring">
            The comparison format is the key innovation. Asking &ldquo;rate this
            response 1&ndash;10&rdquo; produces noisy, inconsistent data.
            Asking &ldquo;which of these two is better?&rdquo; produces cleaner
            signal because humans are better at relative than absolute judgment.
          </InsightBlock>
          <TipBlock title="Not Millions of Labels">
            RLHF does not require millions of human labels. 33K preference pairs
            (InstructGPT) is enough because the reward model generalizes from
            comparisons to score responses the humans never saw.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1 — Predict and Verify (Section 5 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: What Architecture?" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You have preference pairs: (prompt, response A, response B,
                which is better). You want to train a model that can score{' '}
                <strong>any</strong> response to <strong>any</strong> prompt.
              </p>
              <p>
                <strong>
                  What architecture would you use? Think about what you built in
                  Finetuning for Classification.
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>
                      A pretrained language model with a scalar output head.
                    </strong>{' '}
                    Same pattern as classification finetuning: pretrained LM
                    backbone + new head. Instead of a classification head that
                    outputs class probabilities, a reward head that outputs a
                    single number (a quality score).
                  </p>
                  <p>
                    You already built this pattern in Finetuning for
                    Classification. The reward model is the same architecture
                    applied to a different task: predicting human preferences
                    instead of sentiment.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain — The Reward Model (Section 6 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Reward Model"
            subtitle="An experienced editor, not a rule book"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The reward model is a pretrained language model with a linear head
              that outputs a single scalar. Feed in a prompt and response, get a
              score.
            </p>

            <RewardModelDiagram />

            <p className="text-muted-foreground">
              <strong>Training:</strong> For each preference pair, compute
              reward(preferred) &minus; reward(dispreferred). The loss pushes
              this difference to be positive. Over thousands of pairs, the reward
              model learns to assign higher scores to responses humans preferred.
            </p>

            <p className="text-muted-foreground">
              A concrete example: the reward model scores Response A at 0.3 and
              Response B at 0.7. The human preferred B. reward(preferred) &minus;
              reward(dispreferred) = 0.7 &minus; 0.3 = 0.4 &gt; 0&mdash;good,
              the model already agrees with the human on this pair. If the scores
              were reversed (A at 0.7, B at 0.3), the difference would be
              &minus;0.4, and the loss would push the model to adjust its scores
              until it agrees.
            </p>

            <p className="text-muted-foreground">
              Think of the reward model as an <strong>experienced
              editor</strong>. The editor does not write the article, but they
              can tell you which draft is better. They learned this judgment from
              seeing thousands of human comparisons&mdash;not from a set of
              rules, but from patterns in what humans consistently prefer.
            </p>

            <p className="text-muted-foreground">
              <strong>The reward model is imperfect.</strong> It is a learned
              approximation of human preferences, not a perfect oracle. It has
              biases, blind spots, and failure modes. This imperfection matters
              enormously&mdash;we will see why in the next section.
            </p>

            <p className="text-muted-foreground">
              Note what the reward model learns: what humans{' '}
              <strong>prefer</strong>, not what is objectively{' '}
              <strong>true</strong>. A confident, well-structured explanation may
              be preferred over a hedging, uncertain one&mdash;even when the
              hedging response is more accurate. We will revisit this limitation
              later, but keep it in mind: preference and truth are correlated,
              not identical.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Familiar Architecture">
            The reward model is the same pattern you built in Finetuning for
            Classification: pretrained LM backbone + new head. The only
            difference is the output&mdash;a single scalar instead of class
            probabilities.
          </InsightBlock>
          <WarningBlock title="Not a Rule-Based System">
            &ldquo;Reward model&rdquo; sounds like a scoring rubric. It is not.
            It is a neural network trained on data&mdash;the same kind of model
            you have been building throughout this course. No handcrafted rules,
            no explicit scoring criteria.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain — PPO: Optimizing Against the Reward (Section 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="PPO: Optimizing Against the Reward"
            subtitle="For the first time, the training loop changes shape"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now we have a reward model that scores responses. How do we use it
              to improve the language model?
            </p>

            <RlhfLoopDiagram />

            <p className="text-muted-foreground">
              The training loop has three steps:
            </p>

            <div className="space-y-3">
              <PhaseCard number={1} title="Generate" subtitle="The language model produces a response" color="cyan">
                <p className="text-sm">
                  Given a prompt, the language model (called the{' '}
                  <strong>policy</strong> in RL terminology) generates a
                  complete response. &ldquo;Policy&rdquo; just means &ldquo;the
                  model&rsquo;s current behavior&rdquo;&mdash;what it would
                  generate right now.
                </p>
              </PhaseCard>

              <PhaseCard number={2} title="Score" subtitle="The reward model evaluates the response" color="purple">
                <p className="text-sm">
                  The reward model reads the prompt + response and outputs a
                  scalar score. Higher score means the response is more like
                  what humans preferred in the training data.
                </p>
              </PhaseCard>

              <PhaseCard number={3} title="Update" subtitle="PPO adjusts the policy to chase higher rewards" color="orange">
                <p className="text-sm">
                  The language model is updated to increase the probability of
                  high-scoring responses and decrease the probability of
                  low-scoring ones. This is where PPO (Proximal Policy
                  Optimization) comes in&mdash;it is a stable algorithm for
                  making these updates.
                </p>
              </PhaseCard>
            </div>

            <GradientCard title="A New Kind of Training Loop" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>
                    For the first time in this course, the training loop changes
                    shape.
                  </strong>
                </p>
                <p>
                  Pretraining, classification finetuning, and SFT all followed
                  the same heartbeat: forward, loss, backward, step. PPO breaks
                  this pattern: <strong>generate</strong> (many tokens),{' '}
                  <strong>score</strong> (whole response),{' '}
                  <strong>update</strong> (the policy). Two models are involved
                  (policy + reward model). The loop operates at the response
                  level, not the token level.
                </p>
              </div>
            </GradientCard>

            <SectionHeader
              title="The KL Penalty"
              subtitle="Why unconstrained optimization is dangerous"
            />

            <p className="text-muted-foreground">
              What happens if you just maximize the reward with no constraints?
              The model finds <strong>degenerate strategies</strong> that game
              the reward model. It discovers that certain patterns of text get
              high reward scores even though they are not genuinely good
              responses. This is called <strong>reward hacking</strong>.
            </p>

            <p className="text-muted-foreground">
              Remember the editor analogy? The reward model is an experienced
              editor&mdash;but an editor with <strong>blind spots</strong>. An
              editor who only read articles from one genre might overly reward
              that genre&rsquo;s stylistic conventions. A clever writer could
              game the editor by mimicking those conventions without saying
              anything meaningful. The reward model has the same vulnerability:
              it learned preferences from a finite set of human comparisons, and
              the model will find the gaps.
            </p>

            <ComparisonRow
              left={{
                title: 'Reward Hacking Examples',
                color: 'rose',
                items: [
                  'The model becomes excessively verbose because the reward model slightly prefers longer responses',
                  'The model repeats confident-sounding phrases that score well but add no value',
                  'The model discovers formatting tricks (bullet points, headers) that inflate scores',
                  'Output degrades into high-scoring but meaningless text',
                ],
              }}
              right={{
                title: 'With KL Penalty',
                color: 'emerald',
                items: [
                  'The model improves response quality within a bounded range',
                  'Stays close to the SFT baseline behavior',
                  'Cannot drift into degenerate territory',
                  'Quality improves without gaming the reward model',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The <strong>KL penalty</strong> solves this. It measures the
              divergence between the current policy and the SFT baseline. The
              training objective becomes: <em>maximize reward minus KL
              divergence</em>. The model must improve quality without drifting
              too far from the SFT model.
            </p>

            <p className="text-muted-foreground">
              Remember catastrophic forgetting from Finetuning for
              Classification? The KL penalty serves the same purpose as the
              frozen backbone: prevent the model from forgetting what it learned
              in the previous stage. Instead of hard-freezing (binary: frozen or
              unfrozen), the KL penalty provides a{' '}
              <strong>soft constraint</strong>&mdash;stay close, but you can
              drift a little.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Policy = Model Behavior">
            &ldquo;Policy&rdquo; is RL jargon for &ldquo;the model&rsquo;s
            current behavior.&rdquo; When someone says &ldquo;optimize the
            policy,&rdquo; they mean &ldquo;update the model to generate better
            responses.&rdquo; Do not overcomplicate it.
          </InsightBlock>
          <WarningBlock title="Reward Hacking Is Real">
            Without the KL penalty, the model will exploit imperfections in the
            reward model. Every learned reward function has blind spots&mdash;the
            model will find them. The KL penalty is not optional.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 2 — Transfer Question (Section 8 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: No KL Penalty" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>
                  What happens if you remove the KL penalty and train with PPO
                  for many steps?
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>The model reward-hacks.</strong> It finds degenerate
                    strategies that maximize the reward model&rsquo;s score
                    without producing genuinely better responses. It drifts far
                    from the SFT model and may produce nonsensical but
                    high-scoring text.
                  </p>
                  <p>
                    This is the same principle as catastrophic forgetting: without
                    a constraint anchoring the model to its previous behavior,
                    unconstrained optimization destroys what was already working.
                    The KL penalty is the continuous version of &ldquo;freeze the
                    backbone.&rdquo;
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Explain — DPO: The Simpler Alternative (Section 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="DPO: The Simpler Alternative"
            subtitle="Same goal, no reward model"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The PPO pipeline is complex: train a reward model, then use it in
              a multi-model training loop with KL penalties. Is there a simpler
              way?
            </p>

            <p className="text-muted-foreground">
              <strong>Direct Preference Optimization (DPO)</strong> says: you do
              not need a separate reward model. You can directly adjust the
              language model using preference pairs.
            </p>

            <p className="text-muted-foreground">
              The intuition: instead of (1) train a reward model, (2) generate
              responses, (3) score them, (4) update the policy&mdash;just
              directly increase the probability of preferred responses and
              decrease the probability of dispreferred responses. The preference
              data <strong>is</strong> the training signal, without the reward
              model intermediary.
            </p>

            <ComparisonRow
              left={{
                title: 'PPO Pipeline',
                color: 'purple',
                items: [
                  '1. Train a separate reward model on preferences',
                  '2. Generate responses with the language model',
                  '3. Score responses with the reward model',
                  '4. Update the language model with PPO + KL penalty',
                  'Multiple models, complex training loop',
                  'Training loop changes shape (generate → score → update)',
                ],
              }}
              right={{
                title: 'DPO Pipeline',
                color: 'blue',
                items: [
                  '1. Take preference pairs (prompt, preferred, dispreferred)',
                  '2. Compute log-probabilities of both responses',
                  '3. Apply a loss that increases preferred / decreases dispreferred',
                  '4. Standard backward + step',
                  'Single model, closer to supervised learning',
                  'Partially restores the familiar training loop shape',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Take the quantum computing preference pair from earlier. With PPO,
              you would train a reward model, generate new responses, score them,
              and update the policy. With DPO, you skip all of that: compute the
              log-probability of the jargon-heavy response A and the
              age-appropriate response B under the current model. If the model
              assigns higher probability to the dispreferred A, the loss is large
              and the gradient directly pushes the model to favor B. No reward
              model, no generation step&mdash;just preference pairs and a
              gradient update.
            </p>

            <p className="text-muted-foreground">
              DPO partially restores the familiar training loop shape. It looks
              more like supervised learning: forward pass on preference pairs,
              compute a loss, backward, step. Not exactly the same heartbeat
              (the loss function is different and operates on pairs of
              responses), but closer to familiar territory than PPO.
            </p>

            <p className="text-muted-foreground">
              <strong>Results:</strong> DPO achieves comparable quality to PPO on
              many benchmarks. Llama 2 used PPO; many subsequent models (Zephyr,
              Mistral instruct variants) used DPO. Neither approach is
              universally better&mdash;they are two paths to the same goal.
            </p>

            <p className="text-muted-foreground">
              DPO still implicitly has a KL penalty built into its formulation
              (the reference model appears in the loss), so it does not suffer
              from unbounded reward hacking the way PPO-without-KL would.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not a Compromise">
            DPO is not a &ldquo;worse but simpler&rdquo; version of RLHF. It is
            a mathematically equivalent reformulation that eliminates the reward
            model as a separate training stage. Simpler does not mean less
            capable.
          </InsightBlock>
          <TipBlock title="Industry Trend">
            Many recent open-source models use DPO instead of PPO. It is
            significantly easier to implement, train, and debug. If you see a
            model card mentioning &ldquo;preference optimization,&rdquo; it is
            likely DPO or a variant.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Elaborate — Why Alignment Matters Beyond Safety (Section 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Alignment Matters Beyond Safety"
            subtitle="Judgment makes models genuinely more useful"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Alignment is often framed as a safety concern: preventing harmful
              outputs. That matters, but it is not the only reason alignment
              matters.
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="More Useful" color="blue">
                <ul className="space-y-1 text-sm">
                  <li>{'•'} Clearer, more concise explanations</li>
                  <li>{'•'} Asks clarifying questions when the prompt is ambiguous</li>
                  <li>{'•'} Refuses gracefully when it cannot help</li>
                  <li>{'•'} Admits uncertainty instead of guessing</li>
                </ul>
              </GradientCard>
              <GradientCard title="More Honest" color="emerald">
                <ul className="space-y-1 text-sm">
                  <li>{'•'} Less likely to confidently hallucinate</li>
                  <li>{'•'} Distinguishes what it knows from what it is unsure about</li>
                  <li>{'•'} Corrects the user rather than agreeing sycophantically</li>
                  <li>{'•'} Not perfect&mdash;but a measurable improvement</li>
                </ul>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Alignment is not &ldquo;adding safety rails&rdquo; to a capable
              model. It is teaching the model judgment about what makes a
              response genuinely good. Safety is one dimension of that judgment,
              but so are clarity, accuracy, and knowing when to say &ldquo;I do
              not know.&rdquo;
            </p>

            <GradientCard title="Important Caveat" color="amber">
              <div className="space-y-2 text-sm">
                <p>
                  RLHF teaches what humans <strong>prefer</strong>, not what is{' '}
                  <strong>true</strong>. Preferences correlate with truth,
                  helpfulness, and safety&mdash;but they are not identical. A
                  confident, well-structured but wrong answer may be preferred
                  over a hedging, uncertain but correct one.
                </p>
                <p>
                  Sycophancy (&ldquo;You&rsquo;re right!&rdquo;) is often
                  preferred over correction (&ldquo;Actually, you&rsquo;re wrong
                  because&hellip;&rdquo;). RLHF reduces this tendency but does
                  not eliminate it.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              Looking ahead: Constitutional AI (which you will encounter in
              Series 5) extends this further&mdash;what if AI provides the
              preference signal instead of humans?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Voice → Judgment">
            SFT gives the model a voice. Alignment gives it judgment. The model
            goes from mute (base) to speaking (SFT) to speaking wisely
            (aligned). Each stage adds something essential that the previous
            stage could not provide.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Summary (Section 11 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline: 'SFT teaches format; alignment teaches judgment.',
                description:
                  'An SFT model follows instructions but can be harmful, sycophantic, or confidently wrong. It has no training signal for what "helpful" actually means.',
              },
              {
                headline:
                  'Human preferences are the training signal.',
                description:
                  '"Which of these two is better?"—that simple comparative judgment from humans is the foundation of RLHF. Easier and more reliable than absolute scoring.',
              },
              {
                headline:
                  'Reward models: pretrained LM + scalar head.',
                description:
                  'Same pattern as classification finetuning—a pretrained backbone with a new head. Instead of class labels, it outputs a quality score learned from preference data.',
              },
              {
                headline:
                  'PPO: generate, score, update—with KL penalty.',
                description:
                  'The first time the training loop changes shape. Two models involved (policy + reward model). The KL penalty prevents reward hacking by anchoring the model to SFT.',
              },
              {
                headline:
                  'DPO: same goal, no reward model.',
                description:
                  'Directly optimize on preference pairs. Simpler, comparable results. Many modern models use DPO or a variant instead of the full PPO pipeline.',
              },
              {
                headline:
                  'RLHF teaches what humans prefer, not what is true.',
                description:
                  'Preferences correlate with truth, helpfulness, and safety—but they are not identical. Alignment is a significant improvement, not a perfect solution.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Next Step (Section 12 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">What comes next</p>
            <p className="text-sm text-muted-foreground">
              You now understand the full pipeline: pretrain, finetune (SFT),
              align (RLHF/DPO). But there is a practical problem: full
              finetuning&mdash;whether SFT or RLHF&mdash;requires storing all
              model parameters plus their gradients and optimizer states. For
              GPT-2 (124M parameters) this is manageable. For a 7B or 70B model,
              it requires expensive hardware. Next: LoRA and
              quantization&mdash;how to make finetuning and inference accessible
              on real hardware.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Review the pipeline diagram, preference data format, and reward hacking concept. Make sure you can explain why SFT alone is insufficient and how human preferences become a training signal."
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title:
                  'Training language models to follow instructions with human feedback',
                authors: 'Ouyang et al., 2022',
                url: 'https://arxiv.org/abs/2203.02155',
                note: 'The InstructGPT paper. Sections 2-3 describe the SFT → reward model → PPO pipeline that defined the modern alignment approach.',
              },
              {
                title:
                  'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
                authors: 'Rafailov et al., 2023',
                url: 'https://arxiv.org/abs/2305.18290',
                note: 'The DPO paper showing that preference optimization can be done without a separate reward model. Section 4 has the key derivation.',
              },
              {
                title: 'Learning to summarize from human feedback',
                authors: 'Stiennon et al., 2020',
                url: 'https://arxiv.org/abs/2009.01325',
                note: 'An earlier paper applying RLHF to summarization. Clear exposition of the reward model training and PPO pipeline.',
              },
            ]}
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
