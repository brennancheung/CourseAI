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
  PhaseCard,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'

/**
 * Constitutional AI
 *
 * First lesson in Module 5.1 (Advanced Alignment).
 * First lesson in Series 5 (Recent LLM Advances).
 * Conceptual lesson — no heavy notebook, inline SVG diagrams.
 *
 * Cognitive load: STRETCH — three new concepts, conceptual depth after
 * crossing a series boundary.
 *
 * Core concepts at DEVELOPED:
 * - Constitutional AI principles as explicit alignment criteria
 * - Critique-and-revision as a data generation mechanism
 * - RLAIF (AI-generated preference labels replacing human labels)
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing CAI in code (conceptual lesson)
 * - DPO variations or other alignment techniques (Lesson 2)
 * - Red teaming or adversarial evaluation (Lesson 3)
 * - Alignment benchmarks (Lesson 4)
 * - The political/philosophical debate about what principles should be
 * - The specific principles used by Anthropic or any other company
 * - Training a reward model or running PPO
 *
 * Previous: LoRA, Quantization & Inference (Module 4.4, Lesson 4)
 * Next: Alignment Techniques Landscape (Module 5.1, Lesson 2)
 */

// ---------------------------------------------------------------------------
// Inline SVG: RLHF vs Constitutional AI Pipeline Comparison
// Side-by-side comparison showing exactly what changes and what stays.
// ---------------------------------------------------------------------------

function PipelineComparisonDiagram() {
  const svgW = 520
  const svgH = 400

  // Column layout
  const colW = 220
  const leftX = 20
  const rightX = 280
  const boxW = 180
  const boxH = 48

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Column headers */}
        <rect x={leftX} y={4} width={colW} height={22} rx={4} fill="#4c1d9520" />
        <text
          x={leftX + colW / 2}
          y={19}
          textAnchor="middle"
          fill="#c4b5fd"
          fontSize="10"
          fontWeight="600"
        >
          RLHF (Series 4)
        </text>

        <rect x={rightX} y={4} width={colW} height={22} rx={4} fill="#06493520" />
        <text
          x={rightX + colW / 2}
          y={19}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="10"
          fontWeight="600"
        >
          Constitutional AI
        </text>

        {/* --- Stage 1: SFT Data --- */}
        {/* RLHF: Human-written demonstrations */}
        <rect
          x={leftX + (colW - boxW) / 2}
          y={40}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#312e81"
          stroke="#a5b4fc"
          strokeWidth={0.8}
        />
        <text
          x={leftX + colW / 2}
          y={58}
          textAnchor="middle"
          fill="#a5b4fc"
          fontSize="9"
          fontWeight="600"
        >
          Stage 1: SFT
        </text>
        <text
          x={leftX + colW / 2}
          y={72}
          textAnchor="middle"
          fill="#a5b4fc"
          fontSize="8"
          opacity={0.7}
        >
          Human-written demonstrations
        </text>

        {/* CAI: Critique-and-revision */}
        <rect
          x={rightX + (colW - boxW) / 2}
          y={40}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#064e3b"
          stroke="#6ee7b7"
          strokeWidth={1}
        />
        <text
          x={rightX + colW / 2}
          y={58}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="9"
          fontWeight="600"
        >
          Stage 1: SFT
        </text>
        <text
          x={rightX + colW / 2}
          y={72}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="8"
          opacity={0.7}
        >
          AI critique-and-revision data
        </text>

        {/* "CHANGED" label for Stage 1 */}
        <rect
          x={rightX + colW - 52}
          y={40}
          width={50}
          height={14}
          rx={3}
          fill="#6ee7b730"
        />
        <text
          x={rightX + colW - 27}
          y={50}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="7"
          fontWeight="600"
        >
          CHANGED
        </text>

        {/* Arrows from Stage 1 to Stage 2 */}
        <line
          x1={leftX + colW / 2}
          y1={88}
          x2={leftX + colW / 2}
          y2={110}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />
        <line
          x1={rightX + colW / 2}
          y1={88}
          x2={rightX + colW / 2}
          y2={110}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />

        {/* --- Stage 2: Preference Data Source --- */}
        {/* RLHF: Human annotators */}
        <rect
          x={leftX + (colW - boxW) / 2}
          y={112}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#312e81"
          stroke="#a5b4fc"
          strokeWidth={0.8}
        />
        <text
          x={leftX + colW / 2}
          y={130}
          textAnchor="middle"
          fill="#a5b4fc"
          fontSize="9"
          fontWeight="600"
        >
          Preference Labels
        </text>
        <text
          x={leftX + colW / 2}
          y={144}
          textAnchor="middle"
          fill="#a5b4fc"
          fontSize="8"
          opacity={0.7}
        >
          Human annotators (~33K)
        </text>

        {/* CAI: AI applying principles */}
        <rect
          x={rightX + (colW - boxW) / 2}
          y={112}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#064e3b"
          stroke="#6ee7b7"
          strokeWidth={1}
        />
        <text
          x={rightX + colW / 2}
          y={130}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="9"
          fontWeight="600"
        >
          Preference Labels
        </text>
        <text
          x={rightX + colW / 2}
          y={144}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="8"
          opacity={0.7}
        >
          AI applying principles (millions)
        </text>

        {/* "CHANGED" label for Stage 2 */}
        <rect
          x={rightX + colW - 52}
          y={112}
          width={50}
          height={14}
          rx={3}
          fill="#6ee7b730"
        />
        <text
          x={rightX + colW - 27}
          y={122}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="7"
          fontWeight="600"
        >
          CHANGED
        </text>

        {/* Arrows from Stage 2 to Reward Model */}
        <line
          x1={leftX + colW / 2}
          y1={160}
          x2={leftX + colW / 2}
          y2={182}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />
        <line
          x1={rightX + colW / 2}
          y1={160}
          x2={rightX + colW / 2}
          y2={182}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />

        {/* --- Stage 3: Reward Model Training --- */}
        {/* Both: Same reward model */}
        <rect
          x={leftX + (colW - boxW) / 2}
          y={184}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#44403c"
          stroke="#78716c"
          strokeWidth={0.8}
        />
        <text
          x={leftX + colW / 2}
          y={202}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="9"
          fontWeight="600"
        >
          Reward Model Training
        </text>
        <text
          x={leftX + colW / 2}
          y={216}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="8"
          opacity={0.7}
        >
          LM backbone + scalar head
        </text>

        <rect
          x={rightX + (colW - boxW) / 2}
          y={184}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#44403c"
          stroke="#78716c"
          strokeWidth={0.8}
        />
        <text
          x={rightX + colW / 2}
          y={202}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="9"
          fontWeight="600"
        >
          Reward Model Training
        </text>
        <text
          x={rightX + colW / 2}
          y={216}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="8"
          opacity={0.7}
        >
          Same architecture
        </text>

        {/* "SAME" label for Stage 3 */}
        <rect
          x={rightX + colW - 42}
          y={184}
          width={40}
          height={14}
          rx={3}
          fill="#78716c30"
        />
        <text
          x={rightX + colW - 22}
          y={194}
          textAnchor="middle"
          fill="#78716c"
          fontSize="7"
          fontWeight="600"
        >
          SAME
        </text>

        {/* Arrows from Reward Model to RL Training */}
        <line
          x1={leftX + colW / 2}
          y1={232}
          x2={leftX + colW / 2}
          y2={254}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />
        <line
          x1={rightX + colW / 2}
          y1={232}
          x2={rightX + colW / 2}
          y2={254}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />

        {/* --- Stage 4: RL Training --- */}
        <rect
          x={leftX + (colW - boxW) / 2}
          y={256}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#44403c"
          stroke="#78716c"
          strokeWidth={0.8}
        />
        <text
          x={leftX + colW / 2}
          y={274}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="9"
          fontWeight="600"
        >
          RL Training (PPO/DPO)
        </text>
        <text
          x={leftX + colW / 2}
          y={288}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="8"
          opacity={0.7}
        >
          Generate, score, update + KL
        </text>

        <rect
          x={rightX + (colW - boxW) / 2}
          y={256}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#44403c"
          stroke="#78716c"
          strokeWidth={0.8}
        />
        <text
          x={rightX + colW / 2}
          y={274}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="9"
          fontWeight="600"
        >
          RL Training (PPO/DPO)
        </text>
        <text
          x={rightX + colW / 2}
          y={288}
          textAnchor="middle"
          fill="#a8a29e"
          fontSize="8"
          opacity={0.7}
        >
          Same optimization
        </text>

        {/* "SAME" label for Stage 4 */}
        <rect
          x={rightX + colW - 42}
          y={256}
          width={40}
          height={14}
          rx={3}
          fill="#78716c30"
        />
        <text
          x={rightX + colW - 22}
          y={266}
          textAnchor="middle"
          fill="#78716c"
          fontSize="7"
          fontWeight="600"
        >
          SAME
        </text>

        {/* Arrows to output */}
        <line
          x1={leftX + colW / 2}
          y1={304}
          x2={leftX + colW / 2}
          y2={326}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />
        <line
          x1={rightX + colW / 2}
          y1={304}
          x2={rightX + colW / 2}
          y2={326}
          stroke="#47556960"
          strokeWidth={1}
          markerEnd="url(#caiArrow)"
        />

        {/* --- Output: Aligned Model --- */}
        <rect
          x={leftX + (colW - boxW) / 2}
          y={328}
          width={boxW}
          height={36}
          rx={5}
          fill="#4c1d9520"
          stroke="#c4b5fd40"
          strokeWidth={0.8}
        />
        <text
          x={leftX + colW / 2}
          y={350}
          textAnchor="middle"
          fill="#c4b5fd"
          fontSize="9"
          fontWeight="500"
        >
          Aligned Model
        </text>

        <rect
          x={rightX + (colW - boxW) / 2}
          y={328}
          width={boxW}
          height={36}
          rx={5}
          fill="#06493520"
          stroke="#6ee7b740"
          strokeWidth={0.8}
        />
        <text
          x={rightX + colW / 2}
          y={350}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="9"
          fontWeight="500"
        >
          Aligned Model
        </text>

        {/* Summary annotation */}
        <text
          x={svgW / 2}
          y={388}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="8"
          fontStyle="italic"
        >
          Same destination, different data source. The optimization is identical.
        </text>

        <defs>
          <marker
            id="caiArrow"
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
// Inline SVG: Critique-and-Revision Pipeline
// Shows how a prompt flows through critique, revision, and becomes SFT data.
// ---------------------------------------------------------------------------

function CritiqueRevisionDiagram() {
  const svgW = 460
  const svgH = 320

  const boxW = 160
  const boxH = 44
  const centerX = svgW / 2

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Step 1: Prompt + Initial Response */}
        <rect
          x={centerX - boxW / 2}
          y={10}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#312e81"
          stroke="#a5b4fc"
          strokeWidth={0.8}
        />
        <text
          x={centerX}
          y={28}
          textAnchor="middle"
          fill="#a5b4fc"
          fontSize="9"
          fontWeight="600"
        >
          1. Prompt + Initial Response
        </text>
        <text
          x={centerX}
          y={42}
          textAnchor="middle"
          fill="#a5b4fc"
          fontSize="8"
          opacity={0.7}
        >
          (may be harmful or low-quality)
        </text>

        {/* Arrow down */}
        <line
          x1={centerX}
          y1={54}
          x2={centerX}
          y2={72}
          stroke="#475569"
          strokeWidth={1}
          markerEnd="url(#crArrow)"
        />

        {/* Step 2: Select Principle */}
        <rect
          x={centerX - boxW / 2}
          y={74}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#064e3b"
          stroke="#6ee7b7"
          strokeWidth={1}
        />
        <text
          x={centerX}
          y={92}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="9"
          fontWeight="600"
        >
          2. Select Principle
        </text>
        <text
          x={centerX}
          y={106}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="8"
          opacity={0.7}
        >
          from the constitution
        </text>

        {/* Constitution annotation */}
        <rect
          x={centerX + boxW / 2 + 16}
          y={74}
          width={110}
          height={44}
          rx={4}
          fill="#6ee7b710"
          stroke="#6ee7b730"
          strokeWidth={0.5}
        />
        <text
          x={centerX + boxW / 2 + 71}
          y={90}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="7"
          fontWeight="500"
        >
          &quot;Choose the response
        </text>
        <text
          x={centerX + boxW / 2 + 71}
          y={100}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="7"
        >
          that is less likely to
        </text>
        <text
          x={centerX + boxW / 2 + 71}
          y={110}
          textAnchor="middle"
          fill="#6ee7b7"
          fontSize="7"
        >
          encourage illegal activity&quot;
        </text>
        {/* Connector */}
        <line
          x1={centerX + boxW / 2}
          y1={96}
          x2={centerX + boxW / 2 + 16}
          y2={96}
          stroke="#6ee7b740"
          strokeWidth={0.8}
          strokeDasharray="3,3"
        />

        {/* Arrow down */}
        <line
          x1={centerX}
          y1={118}
          x2={centerX}
          y2={136}
          stroke="#475569"
          strokeWidth={1}
          markerEnd="url(#crArrow)"
        />

        {/* Step 3: AI Critique */}
        <rect
          x={centerX - boxW / 2}
          y={138}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#78350f"
          stroke="#fbbf24"
          strokeWidth={0.8}
        />
        <text
          x={centerX}
          y={156}
          textAnchor="middle"
          fill="#fbbf24"
          fontSize="9"
          fontWeight="600"
        >
          3. AI Critiques Response
        </text>
        <text
          x={centerX}
          y={170}
          textAnchor="middle"
          fill="#fbbf24"
          fontSize="8"
          opacity={0.7}
        >
          using the selected principle
        </text>

        {/* Arrow down */}
        <line
          x1={centerX}
          y1={182}
          x2={centerX}
          y2={200}
          stroke="#475569"
          strokeWidth={1}
          markerEnd="url(#crArrow)"
        />

        {/* Step 4: AI Revises */}
        <rect
          x={centerX - boxW / 2}
          y={202}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#4c1d95"
          stroke="#c4b5fd"
          strokeWidth={0.8}
        />
        <text
          x={centerX}
          y={220}
          textAnchor="middle"
          fill="#c4b5fd"
          fontSize="9"
          fontWeight="600"
        >
          4. AI Revises Response
        </text>
        <text
          x={centerX}
          y={234}
          textAnchor="middle"
          fill="#c4b5fd"
          fontSize="8"
          opacity={0.7}
        >
          addressing the critique
        </text>

        {/* Arrow down */}
        <line
          x1={centerX}
          y1={246}
          x2={centerX}
          y2={264}
          stroke="#475569"
          strokeWidth={1}
          markerEnd="url(#crArrow)"
        />

        {/* Step 5: SFT Training Data */}
        <rect
          x={centerX - boxW / 2}
          y={266}
          width={boxW}
          height={boxH}
          rx={5}
          fill="#1e1b4b"
          stroke="#818cf8"
          strokeWidth={0.8}
        />
        <text
          x={centerX}
          y={284}
          textAnchor="middle"
          fill="#818cf8"
          fontSize="9"
          fontWeight="600"
        >
          5. SFT Training Pair
        </text>
        <text
          x={centerX}
          y={298}
          textAnchor="middle"
          fill="#818cf8"
          fontSize="8"
          opacity={0.7}
        >
          (prompt, revised response)
        </text>

        {/* Data generation callout */}
        <rect
          x={16}
          y={266}
          width={100}
          height={44}
          rx={4}
          fill="#f8717110"
          stroke="#f8717130"
          strokeWidth={0.5}
        />
        <text
          x={66}
          y={282}
          textAnchor="middle"
          fill="#f87171"
          fontSize="7"
          fontWeight="500"
        >
          DATA GENERATION
        </text>
        <text
          x={66}
          y={293}
          textAnchor="middle"
          fill="#f87171"
          fontSize="7"
        >
          Not inference-time
        </text>
        <text
          x={66}
          y={303}
          textAnchor="middle"
          fill="#f87171"
          fontSize="7"
        >
          behavior
        </text>
        {/* Connector */}
        <line
          x1={116}
          y1={288}
          x2={centerX - boxW / 2}
          y2={288}
          stroke="#f8717140"
          strokeWidth={0.8}
          strokeDasharray="3,3"
        />

        <defs>
          <marker
            id="crArrow"
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

export function ConstitutionalAiLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Constitutional AI"
            description="How explicit principles replace human annotators as the source of alignment data&mdash;and why this changes the scaling equation for alignment."
            category="Alignment"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints (Section 1 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            In RLHF &amp; Alignment, you learned how human preference pairs
            become the training signal that gives language models judgment.
            That approach works&mdash;but it depends on thousands of human
            annotators who are expensive, inconsistent, and cannot scale.
            This lesson explains how constitutional AI replaces human annotators
            with AI-generated feedback guided by explicit principles, and why
            this matters for scaling alignment to the next generation of models.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why human annotation creates a bottleneck: cost, consistency, and scale',
              'Constitutional AI principles as explicit alignment criteria',
              'Critique-and-revision: AI generates improved responses as SFT data',
              'RLAIF: AI-generated preference labels for the RL training stage',
              'How CAI modifies the RLHF pipeline (what changes, what stays)',
              'Limitations: principle design is hard, principles can conflict',
              'NOT: implementing CAI in code (conceptual lesson)',
              'NOT: DPO variations or other alignment techniques (next lesson)',
              'NOT: red teaming, adversarial evaluation, or safety benchmarks',
              'NOT: specific principles used by any particular company',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Series 5 Begins">
            Welcome to Recent LLM Advances. This series builds on the
            foundation you established in Series 4. Each lesson here
            extends a concept you already know into its modern form.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap — Reconnecting RLHF Mental Models (Section 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where We Left Off"
            subtitle="Reconnecting the RLHF mental models from Series 4"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In RLHF &amp; Alignment, you saw the three-stage progression:
              pretraining gives the model <strong>knowledge</strong>, SFT gives
              it a <strong>voice</strong>, and alignment gives
              it <strong>judgment</strong>. The alignment stage works by
              collecting human preference pairs&mdash;a prompt, two
              responses, and a human judgment of which is
              better&mdash;and training a reward model to generalize those
              preferences. The reward model acts as an{' '}
              <strong>experienced editor</strong>: it does not write the
              article, but it can score which draft is better.
            </p>

            <p className="text-muted-foreground">
              But that editor has limitations. It learned from a finite
              set of human comparisons&mdash;approximately 33,000 for
              InstructGPT&mdash;and those humans do not always agree.
              The editor has <strong>blind spots</strong> that come from
              learning implicitly from examples rather than from explicit
              criteria. And producing those 33,000 comparisons took months
              of expensive human labor.
            </p>

            <p className="text-muted-foreground">
              This lesson asks: <strong>what if we could give the editor
              a written style guide?</strong> Instead of learning preferences
              implicitly from thousands of human examples, what if we could
              write down the principles that define &ldquo;better&rdquo; and
              have them applied consistently to every comparison?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Editor Analogy Continues">
            The editor does not disappear in constitutional AI. The editor
            gets a <strong>style guide</strong>&mdash;explicit principles
            that define what &ldquo;better&rdquo; means. Same editor, better
            tools.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook — The Human Bottleneck (Section 3 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Human Bottleneck"
            subtitle="Why scaling RLHF with human annotators breaks down"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the lock-picking prompt from RLHF &amp; Alignment? An
              SFT-only model helpfully explains how to pick a lock. With RLHF,
              human annotators provide the signal that such a response should be
              refused. But here is the problem:
            </p>

            <div className="rounded-lg border border-border bg-muted/30 p-5 space-y-4">
              <p className="text-sm font-medium text-foreground">
                Prompt: &ldquo;How do I pick a lock?&rdquo;
              </p>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
                  <p className="text-xs font-semibold text-amber-400 mb-2">
                    Annotator 1 says: Refuse
                  </p>
                  <p className="text-xs text-muted-foreground">
                    &ldquo;This could be used for burglary. The model
                    should not provide this information.&rdquo;
                  </p>
                </div>
                <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
                  <p className="text-xs font-semibold text-amber-400 mb-2">
                    Annotator 2 says: Answer
                  </p>
                  <p className="text-xs text-muted-foreground">
                    &ldquo;Locksmiths exist. This is legitimate knowledge.
                    The model should explain it.&rdquo;
                  </p>
                </div>
              </div>
              <p className="text-xs text-center text-muted-foreground">
                With 5 annotators, you get a 3-2 split. The preference
                label is <strong className="text-amber-400">noisy</strong>.
              </p>
            </div>

            <p className="text-muted-foreground">
              Now scale this. What about medical questions where cultural norms
              differ? Legal advice that varies by jurisdiction? Scientific
              topics where communicating uncertainty is critical? The problem is
              not just cost&mdash;it is that <strong>human judgment is not
              consistent enough</strong> for the nuance required at scale.
            </p>

            <GradientCard title="Three Problems With Human Annotation" color="amber">
              <ul className="space-y-2 text-sm">
                <li>
                  <strong>Cost:</strong> Each comparison requires a human to
                  read two full responses and decide. 33K comparisons took months
                  of labeling for InstructGPT.
                </li>
                <li>
                  <strong>Consistency:</strong> Different annotators make
                  different judgments on the same pair. Inter-annotator
                  disagreement introduces noise into the training signal.
                </li>
                <li>
                  <strong>Scale:</strong> The next generation of models needs
                  orders of magnitude more preference data. You cannot hire
                  enough humans.
                </li>
              </ul>
            </GradientCard>

            <p className="text-muted-foreground">
              The question becomes: what if you could write down
              exactly what &ldquo;good&rdquo; means and have it applied
              consistently, at scale, to every comparison?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Just a Cost Issue">
            The human bottleneck is tempting to dismiss as &ldquo;just throw
            more money at it.&rdquo; But more annotators do not resolve genuine
            disagreement&mdash;they just surface how much disagreement exists.
            For truly ambiguous cases, the label stays noisy no matter how
            many humans you ask.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain — The Constitution (Section 4 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Constitution"
            subtitle="Explicit principles as alignment criteria"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Constitutional AI starts with a simple idea: instead of asking
              humans &ldquo;which response is better?&rdquo;, write down the{' '}
              <strong>principles</strong> that define &ldquo;better&rdquo;
              and have an AI model apply those principles.
            </p>

            <p className="text-muted-foreground">
              A <strong>constitution</strong> is a set of written principles.
              Each principle defines one dimension of what makes a response
              good. Here are example principles:
            </p>

            <div className="space-y-2">
              <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
                <p className="text-sm text-muted-foreground">
                  &ldquo;Choose the response that is less likely to be used
                  for illegal activity.&rdquo;
                </p>
              </div>
              <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
                <p className="text-sm text-muted-foreground">
                  &ldquo;Choose the response that is more informative
                  without being dangerous.&rdquo;
                </p>
              </div>
              <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
                <p className="text-sm text-muted-foreground">
                  &ldquo;Choose the response that better acknowledges
                  uncertainty when the answer is unclear.&rdquo;
                </p>
              </div>
              <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
                <p className="text-sm text-muted-foreground">
                  &ldquo;Choose the response that is less likely to be
                  perceived as harmful or offensive.&rdquo;
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Notice what these are: <strong>the criteria that human
              annotators had in their heads but never wrote down.</strong>{' '}
              When an annotator picked Response B over Response A, they
              were implicitly applying principles like these. Constitutional
              AI makes those principles explicit and auditable.
            </p>

            <p className="text-muted-foreground">
              Think about what this means. The reward model already learns human
              preferences from examples. If we can describe those preferences as
              explicit principles, <strong>of course</strong> we can have an AI
              apply those principles directly. We are cutting out the
              middleman&mdash;human annotation&mdash;and making the criteria
              explicit and auditable. Constitutional AI is not a radical
              departure from RLHF. It is the natural next step once you realize
              the criteria were always there, just never written down.
            </p>

            <GradientCard title="Critical Misconception" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>
                    The constitution is NOT a set of rules the model follows
                    at inference time.
                  </strong>
                </p>
                <p>
                  The principles are used only during <em>training</em> to
                  generate preference data. At inference time, the model has
                  no access to the constitution&mdash;the alignment was baked
                  into the weights during training, just like RLHF. You could
                  delete the constitution after training and the model&rsquo;s
                  behavior would not change.
                </p>
                <p>
                  Think of it this way: the constitution is the style guide
                  used to <em>train</em> the editor, not a rulebook the
                  editor consults for every decision.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Implicit → Explicit">
            Human annotators apply implicit principles (&ldquo;this feels
            wrong&rdquo;). Constitutional AI makes those principles
            explicit (&ldquo;choose the response less likely to cause
            harm&rdquo;). Same criteria, auditable form.
          </InsightBlock>
          <TipBlock title="Auditable Alignment">
            A key benefit: you can read the constitution and understand
            exactly what criteria shaped the model&rsquo;s behavior. With
            RLHF, the criteria are locked inside the collective judgment
            of anonymous annotators.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain — Critique-and-Revision (Section 5 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Critique-and-Revision"
            subtitle="Stage 1: AI generates improved responses as SFT data"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The first stage of constitutional AI is{' '}
              <strong>critique-and-revision</strong>: the model critiques
              its own responses using constitutional principles and generates
              improved versions. Here is a concrete walkthrough:
            </p>

            <div className="space-y-3">
              <PhaseCard
                number={1}
                title="Initial Response"
                subtitle="The model generates a potentially problematic response"
                color="rose"
              >
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Prompt:</strong> &ldquo;How can I pick a lock?&rdquo;
                  </p>
                  <p>
                    <strong>Model response:</strong> &ldquo;To pick a pin
                    tumbler lock, you will need a tension wrench and a pick.
                    Insert the tension wrench into the bottom of the keyway
                    and apply slight rotational pressure. Then insert the
                    pick and feel for each pin&hellip;&rdquo;
                  </p>
                </div>
              </PhaseCard>

              <PhaseCard
                number={2}
                title="Principle Selection"
                subtitle="Choose a relevant principle from the constitution"
                color="emerald"
              >
                <p className="text-sm">
                  Selected principle: &ldquo;Choose the response that is
                  less likely to be used for illegal activity.&rdquo;
                </p>
              </PhaseCard>

              <PhaseCard
                number={3}
                title="AI Critique"
                subtitle="The model evaluates its own response against the principle"
                color="amber"
              >
                <p className="text-sm">
                  &ldquo;The response provides detailed, step-by-step
                  instructions for picking a lock, which could be used for
                  breaking and entering. While lock-picking has legitimate
                  uses, the response does not distinguish between lawful
                  and unlawful contexts and provides enough detail to
                  enable unauthorized entry.&rdquo;
                </p>
              </PhaseCard>

              <PhaseCard
                number={4}
                title="AI Revision"
                subtitle="The model rewrites its response to address the critique"
                color="blue"
              >
                <p className="text-sm">
                  &ldquo;Lock-picking is a skill used by licensed
                  locksmiths and security professionals. If you are
                  locked out of your own property, I recommend contacting
                  a licensed locksmith in your area. If you are interested
                  in lock-picking as a hobby, look into local lock sport
                  clubs and organizations like TOOOL, which teach these
                  skills in a legal, educational context.&rdquo;
                </p>
              </PhaseCard>

              <PhaseCard
                number={5}
                title="SFT Training Pair"
                subtitle="The result becomes training data"
                color="violet"
              >
                <p className="text-sm">
                  The pair (original prompt, revised response) becomes SFT
                  training data. The model is fine-tuned on thousands of
                  these improved responses.
                </p>
              </PhaseCard>
            </div>

            <CritiqueRevisionDiagram />

            <GradientCard title="Data Generation, Not Inference" color="amber">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>
                    Critique-and-revision is a data generation process, not
                    an inference-time behavior.
                  </strong>
                </p>
                <p>
                  The iterative critique-revise cycle happens{' '}
                  <em>before training</em> to create high-quality SFT data.
                  The model trained on this data produces good responses
                  directly&mdash;in a single forward pass, with no
                  iterating, no critique step, no principle lookup.
                </p>
                <p>
                  Compare: chain-of-thought happens at inference time (the
                  model reasons in real time). Critique-and-revision happens
                  during data generation (before training even starts).
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Self-Improvement">
            The model uses its own capabilities to generate better training
            data for itself. The principles steer the direction of
            improvement&mdash;they define what &ldquo;better&rdquo; means.
          </InsightBlock>
          <WarningBlock title="Not Prompt Engineering">
            This looks like prompt engineering, but the output is{' '}
            <strong>training data</strong>, not user-facing responses.
            The entire critique-and-revision process could be deleted
            after generating the data and nothing would change.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Check 1 — Predict and Verify (Section 6 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Apply the Mechanism" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A model responds to &ldquo;What medication should I take for
                my headache?&rdquo; with: &ldquo;Take 800mg of ibuprofen
                every 4 hours until the headache subsides.&rdquo;
              </p>
              <p>
                <strong>
                  Given the principle &ldquo;Choose the response that better
                  acknowledges when the model does not know something,&rdquo;
                  what would a critique say? What would a revision look like?
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Critique:</strong> &ldquo;The response provides
                    a specific dosage recommendation without acknowledging
                    that the model cannot assess the user&rsquo;s medical
                    history, other medications, or allergies. 800mg every 4
                    hours exceeds standard OTC dosing guidelines. The
                    response presents medical advice with false
                    certainty.&rdquo;
                  </p>
                  <p>
                    <strong>Revision:</strong> &ldquo;For common headaches,
                    over-the-counter options like ibuprofen or
                    acetaminophen can help, but the right choice and dosage
                    depends on your health history and other medications.
                    I would recommend checking the dosage instructions on
                    the package and consulting a pharmacist or doctor if
                    headaches are frequent or severe.&rdquo;
                  </p>
                  <p>
                    The principle steered the critique toward the
                    uncertainty problem, and the revision acknowledged
                    limitations instead of providing false confidence.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          8. Explain — RLAIF (Section 7 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="RLAIF: AI-Generated Preference Labels"
            subtitle="Stage 2: Replacing human annotators in the RL training step"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Critique-and-revision generates better SFT data. But remember
              the full RLHF pipeline: after SFT comes the RL training step,
              which requires <strong>preference labels</strong>&mdash;thousands
              of comparisons between pairs of responses, labeled as
              &ldquo;this one is better.&rdquo;
            </p>

            <p className="text-muted-foreground">
              In RLHF, those labels come from human annotators. In
              constitutional AI, the second innovation is{' '}
              <strong>RLAIF (Reinforcement Learning from AI Feedback)</strong>:
              an AI model applies constitutional principles to compare two
              responses and generate the preference label. The format is
              identical&mdash;prompt, two responses, which is
              better&mdash;but the annotator is an AI applying explicit
              principles instead of a human applying implicit judgment.
            </p>

            <p className="text-muted-foreground">
              The diagram below shows the full pipeline comparison across
              both stages&mdash;Stage 1 (SFT data generation, which you
              already saw in critique-and-revision) and Stage 2 (preference
              labels for RL training, the new piece):
            </p>

            <PipelineComparisonDiagram />

            <p className="text-muted-foreground">
              Look at what changed and what stayed the same. The reward model
              architecture is identical. The RL training step (PPO or DPO) is
              identical. The KL penalty still prevents reward hacking. The
              only difference is <strong>where the preference labels come
              from</strong>: AI applying principles instead of humans
              applying intuition.
            </p>

            <GradientCard title="Scale Comparison" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>RLHF (InstructGPT):</strong> ~33,000 human
                  comparisons. Months of labeling. Significant cost per
                  comparison.
                </p>
                <p>
                  <strong>RLAIF:</strong> Millions of comparisons generated
                  in hours. Cost of compute only. Each comparison is
                  consistent&mdash;the same principle applied the same way
                  every time.
                </p>
                <p>
                  This is not &ldquo;cheaper humans.&rdquo; This is a{' '}
                  <strong>different scaling regime entirely</strong>.
                  100x&ndash;1000x more preference data, with zero
                  inter-annotator disagreement.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              And the results? Anthropic&rsquo;s original constitutional AI
              paper (Bai et al., 2022) showed that models trained with
              AI-generated preference labels matched or exceeded human-feedback
              RLHF on harmlessness benchmarks&mdash;while being{' '}
              <em>more</em> helpful, not less. AI feedback is not an
              approximation of human feedback. In some dimensions, it is
              an improvement.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pipeline, Different Source">
            Constitutional AI does not replace the RLHF pipeline. It
            replaces the <em>data source</em> that feeds into the pipeline.
            The optimization machinery is identical.
          </InsightBlock>
          <TipBlock title="KL Penalty Still Applies">
            The RL training step still uses the KL penalty to prevent reward
            hacking. Constitutional AI changes where the preference data
            comes from, not how the model is optimized. Everything you
            learned about the KL constraint still applies.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Elaborate — When Principles Fail (Section 8 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When Principles Fail"
            subtitle="Constitutional AI shifts the challenge, it does not eliminate it"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Constitutional AI is elegant, but it is not magic. The quality
              of alignment depends entirely on the quality of the principles.
              Here are the failure modes:
            </p>

            <div className="space-y-3">
              <PhaseCard
                number={1}
                title="Vague Principles"
                subtitle="Too broad to discriminate between responses"
                color="rose"
              >
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Principle:</strong> &ldquo;Choose the response
                    that is more helpful.&rdquo;
                  </p>
                  <p>
                    <strong>Problem:</strong> Almost any response can be
                    argued to be &ldquo;helpful.&rdquo; The AI critique
                    produces vague feedback (&ldquo;this response could be
                    more helpful&rdquo;) that does not identify any specific
                    problem. The preference label is essentially random.
                  </p>
                </div>
              </PhaseCard>

              <PhaseCard
                number={2}
                title="Conflicting Principles"
                subtitle="Two principles pull in opposite directions"
                color="rose"
              >
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Principles:</strong> &ldquo;Choose the response
                    that is maximally helpful&rdquo; and &ldquo;Choose the
                    response that refuses dangerous requests.&rdquo;
                  </p>
                  <p>
                    <strong>Problem:</strong> For a prompt like &ldquo;How
                    do I remove a stripped screw?&rdquo; (legitimate) vs
                    &ldquo;How do I bypass a car ignition?&rdquo; (ambiguous),
                    the critique depends on which principle is selected.
                    The alignment quality depends on principle selection
                    logic that is itself a design challenge.
                  </p>
                </div>
              </PhaseCard>

              <PhaseCard
                number={3}
                title="Missing Principles"
                subtitle="A failure mode that no principle covers"
                color="rose"
              >
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Problem:</strong> The constitution does not
                    include a principle about cultural sensitivity. The
                    model generates responses that are technically accurate
                    but culturally offensive in certain contexts. No
                    principle flags this because the designers did not
                    anticipate this failure mode.
                  </p>
                  <p className="text-rose-400/80">
                    This is the &ldquo;editor with blind spots&rdquo;
                    problem again&mdash;but now the blind spots are in the
                    constitution, not in the annotator pool.
                  </p>
                </div>
              </PhaseCard>
            </div>

            <GradientCard title="The Challenge Shifts, Not Disappears" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  RLHF&rsquo;s challenge: <strong>find and train enough
                  human annotators</strong> whose collective judgment
                  produces good alignment.
                </p>
                <p>
                  Constitutional AI&rsquo;s challenge: <strong>design the
                  right set of principles</strong> that cover all the
                  dimensions of &ldquo;good.&rdquo;
                </p>
                <p>
                  The difficulty does not disappear. It moves from a labor
                  problem (enough annotators) to a design problem (right
                  principles). The design problem is arguably more
                  tractable&mdash;principles can be iterated, version-controlled,
                  and audited&mdash;but it requires careful thought
                  about edge cases.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              One more important point: <strong>constitutional AI does not
              replace RLHF entirely</strong>. The original paper still uses
              RL training (PPO) in its second stage&mdash;with AI-generated
              labels instead of human labels. If you removed the RL step,
              you would only have the SFT-based critique-and-revision, which
              produces weaker alignment. The RL stage is still needed for
              strong alignment; CAI changes the data source, not the
              optimization mechanism.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Principles ≠ Magic">
            Constitutional AI is only as good as its constitution. Poorly
            written, vague, or incomplete principles produce poor alignment.
            The constitution is a design artifact that requires engineering
            judgment.
          </WarningBlock>
          <InsightBlock title="Blind Spots Move">
            In RLHF, blind spots live in the annotator pool. In CAI,
            blind spots live in the constitution. The pattern is the same:
            the training data source determines the alignment quality.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Check 2 — Transfer Question (Section 9 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Design a Constitution" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A company wants to align a model specifically for medical
                advice assistance (supporting doctors, not replacing them).
              </p>
              <p>
                <strong>
                  What principles would you include in the constitution?
                  What failure modes might emerge even with good principles?
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Possible principles:</strong>
                  </p>
                  <ul className="space-y-1 ml-2">
                    <li>
                      &bull; &ldquo;Choose the response that more clearly
                      distinguishes established medical consensus from
                      uncertain or debated evidence.&rdquo;
                    </li>
                    <li>
                      &bull; &ldquo;Choose the response that recommends
                      professional consultation for conditions beyond
                      standard first aid.&rdquo;
                    </li>
                    <li>
                      &bull; &ldquo;Choose the response that avoids
                      specific dosage recommendations without knowing the
                      patient&rsquo;s history.&rdquo;
                    </li>
                    <li>
                      &bull; &ldquo;Choose the response that presents
                      differential diagnoses rather than a single confident
                      diagnosis.&rdquo;
                    </li>
                  </ul>
                  <p>
                    <strong>Failure modes:</strong> Conflicting principles
                    (be informative vs avoid specific recommendations);
                    missing principles (cultural differences in medical
                    practice); the model may become overly cautious, adding
                    disclaimers to every response rather than being
                    substantively helpful. Over-specification of principles
                    can be as harmful as under-specification.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          11. Practice — Notebook (Section 10 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Experience the critique-and-revision mechanism hands-on"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook lets you run the constitutional AI mechanism
              yourself&mdash;on a small scale, using API calls to demonstrate
              how principles steer critique, revision, and preference data
              generation.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Principle-Guided Critique (Guided)" color="blue">
                <p className="text-sm">
                  Write a principle and use an LLM to critique a response.
                  Then try a different principle on the same response and see
                  how the critique changes. The insight: principles steer the
                  feedback.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Revision and Preference Pairs (Supported)" color="blue">
                <p className="text-sm">
                  Generate a revised response from a critique and construct a
                  preference pair (revised &gt; original). This is the complete
                  critique-and-revision loop&mdash;the same data generation
                  process that CAI runs at scale.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: When Principles Fail (Supported)" color="blue">
                <p className="text-sm">
                  Write a deliberately vague principle and observe how critique
                  quality degrades. The insight: constitution quality determines
                  alignment quality.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-1-1-constitutional-ai.ipynb"
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
          <TipBlock title="Lightweight Notebook">
            These exercises use API calls to demonstrate the mechanism, not
            full CAI training. You are doing manually what constitutional AI
            does at scale&mdash;the mechanism is the same, only the scale
            differs.
          </TipBlock>
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
                headline:
                  'Constitutional AI extends RLHF with explicit principles.',
                description:
                  'Instead of relying on human annotators\' implicit judgment, write down the principles that define "better" and have AI apply them. The editor gets a style guide.',
              },
              {
                headline:
                  'Two stages: critique-and-revision + RLAIF.',
                description:
                  'Stage 1 generates improved SFT data through AI self-critique. Stage 2 generates RL preference labels with AI applying principles instead of humans comparing responses.',
              },
              {
                headline:
                  'The constitution is used during training, not inference.',
                description:
                  'Principles generate training data. At inference time, the alignment is baked into the weights. The model never consults the constitution when responding.',
              },
              {
                headline:
                  'Quality depends on principle design.',
                description:
                  'The alignment challenge shifts from "enough annotators" to "right principles." Vague, conflicting, or missing principles produce poor alignment. The blind spots move, they don\'t disappear.',
              },
              {
                headline:
                  'RLAIF scales alignment by orders of magnitude.',
                description:
                  '~33K human comparisons for InstructGPT vs millions of AI-generated comparisons for CAI. Not cheaper humans\u2014a different scaling regime entirely, with zero inter-annotator disagreement.',
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
                title:
                  'Constitutional AI: Harmlessness from AI Feedback',
                authors: 'Bai et al., 2022',
                url: 'https://arxiv.org/abs/2212.08073',
                note: 'The original constitutional AI paper. Section 2 describes the critique-and-revision mechanism; Section 3 covers RLAIF. The results tables show CAI matching or exceeding human-feedback RLHF.',
              },
              {
                title:
                  'Training language models to follow instructions with human feedback',
                authors: 'Ouyang et al., 2022',
                url: 'https://arxiv.org/abs/2203.02155',
                note: 'The InstructGPT paper that established the RLHF pipeline. Useful context for understanding what constitutional AI modifies.',
              },
              {
                title:
                  'RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback',
                authors: 'Lee et al., 2023',
                url: 'https://arxiv.org/abs/2309.00267',
                note: 'Focused study on RLAIF specifically. Shows that AI-generated preference labels produce comparable alignment quality to human labels across multiple tasks.',
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
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Review the pipeline comparison diagram, the critique-and-revision walkthrough, and the principle failure modes. Make sure you can explain why principles-based feedback scales better than human feedback, and what the tradeoffs are. Next up: DPO was introduced in Series 4 as a simpler alternative to PPO—what if even DPO is not the right formulation? We map the full landscape of alignment techniques."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
