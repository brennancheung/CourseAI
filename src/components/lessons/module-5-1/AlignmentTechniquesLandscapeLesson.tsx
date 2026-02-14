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

/**
 * The Alignment Techniques Landscape
 *
 * Second lesson in Module 5.1 (Advanced Alignment).
 * Conceptual lesson — inline SVG diagram (design space map), no heavy notebook.
 *
 * Cognitive load: BUILD — organizing breadth, not introducing depth.
 * After the STRETCH lesson on constitutional AI, this lesson maps the
 * preference optimization design space along four axes.
 *
 * Core concept at DEVELOPED:
 * - Design space axes framework (data format, reference model, online/offline, reward model)
 *
 * Concepts at INTRODUCED:
 * - IPO (Identity Preference Optimization)
 * - KTO (Kahneman-Tversky Optimization)
 * - ORPO (Odds Ratio Preference Optimization)
 * - Online vs offline preference optimization
 * - Iterative alignment / self-play
 *
 * EXPLICITLY NOT COVERED:
 * - Mathematical loss function derivations (no Bradley-Terry, no loss equations)
 * - Implementing any technique in code
 * - Benchmarking or performance comparisons
 * - Constitutional AI (previous lesson) or red teaming (next lesson)
 * - RL formalism (policy gradient, advantage estimation)
 * - Specific company choices
 *
 * Previous: Constitutional AI (Module 5.1, Lesson 1)
 * Next: Red Teaming & Adversarial Evaluation (Module 5.1, Lesson 3)
 */

// ---------------------------------------------------------------------------
// Inline SVGs: Design Space Maps
// Central visual artifacts. Sparse version (PPO + DPO only) shown first,
// complete version (all five methods) shown after variations are explained.
// ---------------------------------------------------------------------------

function SparseDesignSpaceMap() {
  const svgW = 580
  const svgH = 420

  // Axis layout — four horizontal rows
  const axisX = 90
  const axisW = 440
  const axisEndX = axisX + axisW
  const labelX = 10

  // Row positions
  const row1Y = 70 // Data format
  const row2Y = 150 // Reference model
  const row3Y = 230 // Online vs offline
  const row4Y = 310 // Reward model

  // Method colors
  const ppoColor = '#f59e0b' // amber
  const dpoColor = '#6366f1' // indigo

  // Marker radius
  const r = 8

  // Method positions (normalized 0-1 along axis, then mapped)
  const pos = (frac: number) => axisX + frac * axisW

  // Empty position markers — dashed circles showing gaps
  const ghostColor = '#334155'

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
          fill="#e2e8f0"
          fontSize="13"
          fontWeight="600"
        >
          Preference Optimization Design Space
        </text>
        <text
          x={svgW / 2}
          y={42}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="9"
          fontStyle="italic"
        >
          Two methods placed — where do the gaps lead?
        </text>

        {/* ---- Axis 1: Data Format ---- */}
        <text
          x={labelX}
          y={row1Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Data Format
        </text>
        <line
          x1={axisX}
          y1={row1Y}
          x2={axisEndX}
          y2={row1Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        <text x={axisX} y={row1Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Comparison pairs
        </text>
        <text x={pos(0.55)} y={row1Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Single responses
        </text>
        <text x={axisEndX} y={row1Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          No preference data
        </text>
        {/* PPO and DPO */}
        <circle cx={pos(0.05)} cy={row1Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.05)} y={row1Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>
        <circle cx={pos(0.15)} cy={row1Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.15)} y={row1Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        {/* Ghost markers for gaps */}
        <circle cx={pos(0.55)} cy={row1Y} r={r} fill="none" stroke={ghostColor} strokeWidth={1} strokeDasharray="3,3" />
        <text x={pos(0.55)} y={row1Y - 14} fill={ghostColor} fontSize="8" fontStyle="italic" textAnchor="middle">?</text>
        <circle cx={pos(0.92)} cy={row1Y} r={r} fill="none" stroke={ghostColor} strokeWidth={1} strokeDasharray="3,3" />
        <text x={pos(0.92)} y={row1Y - 14} fill={ghostColor} fontSize="8" fontStyle="italic" textAnchor="middle">?</text>

        {/* ---- Axis 2: Reference Model ---- */}
        <text
          x={labelX}
          y={row2Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Reference Model
        </text>
        <line
          x1={axisX}
          y1={row2Y}
          x2={axisEndX}
          y2={row2Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        <text x={axisX} y={row2Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Required
        </text>
        <text x={axisEndX} y={row2Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Not required
        </text>
        <circle cx={pos(0.1)} cy={row2Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.1)} y={row2Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>
        <circle cx={pos(0.2)} cy={row2Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.2)} y={row2Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        {/* Ghost marker */}
        <circle cx={pos(0.9)} cy={row2Y} r={r} fill="none" stroke={ghostColor} strokeWidth={1} strokeDasharray="3,3" />
        <text x={pos(0.9)} y={row2Y - 14} fill={ghostColor} fontSize="8" fontStyle="italic" textAnchor="middle">?</text>

        {/* ---- Axis 3: Online vs Offline ---- */}
        <text
          x={labelX}
          y={row3Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Online / Offline
        </text>
        <line
          x1={axisX}
          y1={row3Y}
          x2={axisEndX}
          y2={row3Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        <text x={axisX} y={row3Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Offline (static data)
        </text>
        <text x={axisEndX} y={row3Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Online (generated during training)
        </text>
        <circle cx={pos(0.1)} cy={row3Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.1)} y={row3Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        <circle cx={pos(0.9)} cy={row3Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.9)} y={row3Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>
        {/* Ghost markers in the middle */}
        <circle cx={pos(0.3)} cy={row3Y} r={r} fill="none" stroke={ghostColor} strokeWidth={1} strokeDasharray="3,3" />
        <text x={pos(0.3)} y={row3Y - 14} fill={ghostColor} fontSize="8" fontStyle="italic" textAnchor="middle">?</text>

        {/* ---- Axis 4: Reward Model ---- */}
        <text
          x={labelX}
          y={row4Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Reward Model
        </text>
        <line
          x1={axisX}
          y1={row4Y}
          x2={axisEndX}
          y2={row4Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        <text x={axisX} y={row4Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          No (implicit in loss)
        </text>
        <text x={axisEndX} y={row4Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Yes (separate model)
        </text>
        <circle cx={pos(0.1)} cy={row4Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.1)} y={row4Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        <circle cx={pos(0.9)} cy={row4Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.9)} y={row4Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>

        {/* Legend */}
        <g transform={`translate(${axisX}, ${row4Y + 40})`}>
          <circle cx={0} cy={0} r={5} fill={ppoColor} />
          <text x={10} y={4} fill={ppoColor} fontSize="8" fontWeight="500">PPO</text>
          <circle cx={55} cy={0} r={5} fill={dpoColor} />
          <text x={65} y={4} fill={dpoColor} fontSize="8" fontWeight="500">DPO</text>
          <circle cx={130} cy={0} r={5} fill="none" stroke={ghostColor} strokeWidth={1} strokeDasharray="3,3" />
          <text x={140} y={4} fill={ghostColor} fontSize="8" fontWeight="500" fontStyle="italic">Gaps to fill</text>
        </g>

        {/* Annotation */}
        <text
          x={svgW / 2}
          y={row4Y + 60}
          textAnchor="middle"
          fill="#64748b"
          fontSize="8"
          fontStyle="italic"
        >
          PPO and DPO cluster together on some axes. The dashed circles mark unexplored positions.
        </text>
      </svg>
    </div>
  )
}

function DesignSpaceMap() {
  const svgW = 580
  const svgH = 420

  // Axis layout — four horizontal rows
  const axisX = 90
  const axisW = 440
  const axisEndX = axisX + axisW
  const labelX = 10

  // Row positions
  const row1Y = 70 // Data format
  const row2Y = 150 // Reference model
  const row3Y = 230 // Online vs offline
  const row4Y = 310 // Reward model

  // Method colors
  const ppoColor = '#f59e0b' // amber
  const dpoColor = '#6366f1' // indigo
  const ipoColor = '#8b5cf6' // violet
  const ktoColor = '#06b6d4' // cyan
  const orpoColor = '#10b981' // emerald

  // Marker radius
  const r = 8

  // Method positions (normalized 0-1 along axis, then mapped)
  const pos = (frac: number) => axisX + frac * axisW

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
          fill="#e2e8f0"
          fontSize="13"
          fontWeight="600"
        >
          Preference Optimization Design Space
        </text>
        <text
          x={svgW / 2}
          y={42}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="9"
          fontStyle="italic"
        >
          Each method occupies a different position — tradeoffs, not upgrades
        </text>

        {/* ---- Axis 1: Data Format ---- */}
        <text
          x={labelX}
          y={row1Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Data Format
        </text>
        <line
          x1={axisX}
          y1={row1Y}
          x2={axisEndX}
          y2={row1Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        {/* Axis labels */}
        <text x={axisX} y={row1Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Comparison pairs
        </text>
        <text x={pos(0.55)} y={row1Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Single responses
        </text>
        <text x={axisEndX} y={row1Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          No preference data
        </text>
        {/* Methods on axis 1 */}
        <circle cx={pos(0.05)} cy={row1Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.05)} y={row1Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>
        <circle cx={pos(0.15)} cy={row1Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.15)} y={row1Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        <circle cx={pos(0.25)} cy={row1Y} r={r} fill={ipoColor} opacity={0.9} />
        <text x={pos(0.25)} y={row1Y - 14} fill={ipoColor} fontSize="8" fontWeight="600" textAnchor="middle">IPO</text>
        <circle cx={pos(0.55)} cy={row1Y} r={r} fill={ktoColor} opacity={0.9} />
        <text x={pos(0.55)} y={row1Y - 14} fill={ktoColor} fontSize="8" fontWeight="600" textAnchor="middle">KTO</text>
        <circle cx={pos(0.92)} cy={row1Y} r={r} fill={orpoColor} opacity={0.9} />
        <text x={pos(0.92)} y={row1Y - 14} fill={orpoColor} fontSize="8" fontWeight="600" textAnchor="middle">ORPO</text>

        {/* ---- Axis 2: Reference Model ---- */}
        <text
          x={labelX}
          y={row2Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Reference Model
        </text>
        <line
          x1={axisX}
          y1={row2Y}
          x2={axisEndX}
          y2={row2Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        <text x={axisX} y={row2Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Required
        </text>
        <text x={axisEndX} y={row2Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Not required
        </text>
        {/* Methods on axis 2 */}
        <circle cx={pos(0.1)} cy={row2Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.1)} y={row2Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>
        <circle cx={pos(0.2)} cy={row2Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.2)} y={row2Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        <circle cx={pos(0.3)} cy={row2Y} r={r} fill={ipoColor} opacity={0.9} />
        <text x={pos(0.3)} y={row2Y - 14} fill={ipoColor} fontSize="8" fontWeight="600" textAnchor="middle">IPO</text>
        <circle cx={pos(0.4)} cy={row2Y} r={r} fill={ktoColor} opacity={0.9} />
        <text x={pos(0.4)} y={row2Y - 14} fill={ktoColor} fontSize="8" fontWeight="600" textAnchor="middle">KTO</text>
        <circle cx={pos(0.9)} cy={row2Y} r={r} fill={orpoColor} opacity={0.9} />
        <text x={pos(0.9)} y={row2Y - 14} fill={orpoColor} fontSize="8" fontWeight="600" textAnchor="middle">ORPO</text>

        {/* ---- Axis 3: Online vs Offline ---- */}
        <text
          x={labelX}
          y={row3Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Online / Offline
        </text>
        <line
          x1={axisX}
          y1={row3Y}
          x2={axisEndX}
          y2={row3Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        <text x={axisX} y={row3Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Offline (static data)
        </text>
        <text x={axisEndX} y={row3Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Online (generated during training)
        </text>
        {/* Methods on axis 3 */}
        <circle cx={pos(0.1)} cy={row3Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.1)} y={row3Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        <circle cx={pos(0.2)} cy={row3Y} r={r} fill={ipoColor} opacity={0.9} />
        <text x={pos(0.2)} y={row3Y - 14} fill={ipoColor} fontSize="8" fontWeight="600" textAnchor="middle">IPO</text>
        <circle cx={pos(0.3)} cy={row3Y} r={r} fill={ktoColor} opacity={0.9} />
        <text x={pos(0.3)} y={row3Y - 14} fill={ktoColor} fontSize="8" fontWeight="600" textAnchor="middle">KTO</text>
        <circle cx={pos(0.35)} cy={row3Y} r={r} fill={orpoColor} opacity={0.9} />
        <text x={pos(0.35)} y={row3Y - 14} fill={orpoColor} fontSize="8" fontWeight="600" textAnchor="middle">ORPO</text>
        <circle cx={pos(0.9)} cy={row3Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.9)} y={row3Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>

        {/* ---- Axis 4: Reward Model ---- */}
        <text
          x={labelX}
          y={row4Y - 16}
          fill="#94a3b8"
          fontSize="9"
          fontWeight="600"
        >
          Reward Model
        </text>
        <line
          x1={axisX}
          y1={row4Y}
          x2={axisEndX}
          y2={row4Y}
          stroke="#334155"
          strokeWidth={1.5}
        />
        <text x={axisX} y={row4Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          No (implicit in loss)
        </text>
        <text x={axisEndX} y={row4Y + 20} fill="#64748b" fontSize="8" textAnchor="middle">
          Yes (separate model)
        </text>
        {/* Methods on axis 4 */}
        <circle cx={pos(0.1)} cy={row4Y} r={r} fill={dpoColor} opacity={0.9} />
        <text x={pos(0.1)} y={row4Y - 14} fill={dpoColor} fontSize="8" fontWeight="600" textAnchor="middle">DPO</text>
        <circle cx={pos(0.2)} cy={row4Y} r={r} fill={ipoColor} opacity={0.9} />
        <text x={pos(0.2)} y={row4Y - 14} fill={ipoColor} fontSize="8" fontWeight="600" textAnchor="middle">IPO</text>
        <circle cx={pos(0.3)} cy={row4Y} r={r} fill={ktoColor} opacity={0.9} />
        <text x={pos(0.3)} y={row4Y - 14} fill={ktoColor} fontSize="8" fontWeight="600" textAnchor="middle">KTO</text>
        <circle cx={pos(0.35)} cy={row4Y} r={r} fill={orpoColor} opacity={0.9} />
        <text x={pos(0.35)} y={row4Y - 14} fill={orpoColor} fontSize="8" fontWeight="600" textAnchor="middle">ORPO</text>
        <circle cx={pos(0.9)} cy={row4Y} r={r} fill={ppoColor} opacity={0.9} />
        <text x={pos(0.9)} y={row4Y - 14} fill={ppoColor} fontSize="8" fontWeight="600" textAnchor="middle">PPO</text>

        {/* Legend */}
        <g transform={`translate(${axisX}, ${row4Y + 40})`}>
          <circle cx={0} cy={0} r={5} fill={ppoColor} />
          <text x={10} y={4} fill={ppoColor} fontSize="8" fontWeight="500">PPO</text>
          <circle cx={55} cy={0} r={5} fill={dpoColor} />
          <text x={65} y={4} fill={dpoColor} fontSize="8" fontWeight="500">DPO</text>
          <circle cx={110} cy={0} r={5} fill={ipoColor} />
          <text x={120} y={4} fill={ipoColor} fontSize="8" fontWeight="500">IPO</text>
          <circle cx={160} cy={0} r={5} fill={ktoColor} />
          <text x={170} y={4} fill={ktoColor} fontSize="8" fontWeight="500">KTO</text>
          <circle cx={215} cy={0} r={5} fill={orpoColor} />
          <text x={225} y={4} fill={orpoColor} fontSize="8" fontWeight="500">ORPO</text>
        </g>

        {/* Annotation */}
        <text
          x={svgW / 2}
          y={row4Y + 60}
          textAnchor="middle"
          fill="#64748b"
          fontSize="8"
          fontStyle="italic"
        >
          Different positions = different tradeoffs. No method dominates all axes.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function AlignmentTechniquesLandscapeLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Alignment Techniques Landscape"
            description="PPO and DPO are not the only options&mdash;the design space of preference optimization is wider than you think, and the right choice depends on your constraints."
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
            In RLHF &amp; Alignment, you learned PPO (two models, a reward
            model, online generation) and DPO (one model, no reward model,
            directly adjust probabilities). That binary covered the
            essentials&mdash;but the field has not stopped at two options. This
            lesson maps the full design space of preference optimization along
            four concrete axes, so you can locate any alignment technique&mdash;
            current or future&mdash;by asking what tradeoffs it makes.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Mapping the preference optimization design space along four axes',
              'What IPO, KTO, and ORPO each change relative to DPO and why',
              'The online vs offline distinction and its practical implications',
              'Iterative alignment: using model outputs for the next training round',
              'Building a mental map for classifying future techniques',
              'NOT: mathematical loss function derivations',
              'NOT: implementing any technique in code (conceptual lesson)',
              'NOT: benchmarking or performance comparisons between methods',
              'NOT: constitutional AI (previous lesson) or red teaming (next lesson)',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Map, Not Ladder">
            This lesson organizes many methods without ranking them. The goal
            is a <strong>map</strong> you can navigate, not a leaderboard of
            techniques.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap — PPO and DPO as Starting Points (Section 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where We Left Off"
            subtitle="PPO and DPO as the two endpoints you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In RLHF &amp; Alignment, you saw two approaches to preference
              optimization:
            </p>

            <ComparisonRow
              left={{
                title: 'PPO',
                color: 'amber',
                items: [
                  'Two models (policy + reward model)',
                  'Online: generates responses during training',
                  'Explicit reward model scores each response',
                  'KL penalty prevents drift from SFT model',
                  'Generate → Score → Update loop',
                ],
              }}
              right={{
                title: 'DPO',
                color: 'blue',
                items: [
                  'One model (no separate reward model)',
                  'Offline: trains on pre-collected preference pairs',
                  'Reward signal implicit in the loss function',
                  'Reference model provides implicit KL constraint',
                  'Directly adjusts probabilities of preferred vs dispreferred',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Remember the quantum computing preference pair? A user asks
              &ldquo;explain quantum computing to a 10-year-old.&rdquo;
              Response A gives a specific, age-appropriate analogy; Response B
              is vague but not wrong. DPO took that pair and directly adjusted
              probabilities&mdash;no reward model needed. PPO would have scored
              each response with a reward model first. Same data, different
              mechanism.
            </p>

            <p className="text-muted-foreground">
              You left that lesson with a reasonable mental model: PPO is the
              full-infrastructure approach, DPO is the streamlined alternative
              that gets comparable results with less complexity. Both use the
              same data format (comparison pairs) and both need a reference
              model. But <strong>what if even this binary is too
              narrow?</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Extending the Analogy">
            In Constitutional AI, you saw that the alignment pipeline is more
            flexible than it first appears&mdash;you can change the data{' '}
            <em>source</em> (human vs AI). Now we change the data{' '}
            <em>format</em> and optimization <em>mechanism</em>.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook — Three Unsolved Scenarios (Section 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Problems PPO and DPO Cannot Solve"
            subtitle="Real constraints that break the PPO/DPO binary"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider three real-world alignment scenarios. For each one, ask
              yourself: which of the two methods you know (PPO, DPO) cleanly
              solves this?
            </p>

            <div className="space-y-3">
              <GradientCard title="Scenario A: Thumbs-Up/Down Data" color="cyan">
                <p className="text-sm">
                  You have a chatbot app. Users click thumbs-up or thumbs-down
                  on individual responses. You do <strong>not</strong> have
                  paired comparisons&mdash;just single responses labeled good or
                  bad. Neither PPO nor DPO can directly use this data format.
                </p>
              </GradientCard>

              <GradientCard title="Scenario B: Memory-Constrained" color="cyan">
                <p className="text-sm">
                  You want to align a 13B parameter model but cannot afford to
                  keep a frozen reference model in GPU memory alongside the
                  training model. Both PPO and DPO require a reference
                  model&mdash;that doubles your memory footprint.
                </p>
              </GradientCard>

              <GradientCard title="Scenario C: Learn From Current Mistakes" color="cyan">
                <p className="text-sm">
                  You want the model to learn from its own <em>current</em>{' '}
                  mistakes, not from a static dataset generated by an earlier
                  version. DPO trains on a fixed dataset. PPO generates online,
                  but the full PPO pipeline is too expensive for your team.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Each scenario points to a gap in the PPO/DPO binary. These gaps
              are not hypothetical&mdash;they drove the development of the
              techniques we are about to map.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Hypothetical">
            Every scenario here reflects a real constraint that alignment
            teams face. Scenario A is the most common&mdash;most user
            feedback is thumbs-up/down, not paired comparisons.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain: The Design Space Axes (Section 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Four Axes"
            subtitle="A framework for mapping any preference optimization method"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Instead of comparing methods head-to-head, we map the design
              space along four independent axes. Each axis asks a different
              question about how a method works:
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <GradientCard title="1. Data Format" color="blue">
                <p className="text-sm">
                  What does your training data look like? Comparison pairs
                  (A is better than B)? Single responses rated good/bad? Or no
                  preference data at all?
                </p>
              </GradientCard>

              <GradientCard title="2. Reference Model" color="blue">
                <p className="text-sm">
                  Do you need a frozen copy of the SFT model? The reference
                  model provides the KL constraint&mdash;removing it means you
                  need another stability mechanism.
                </p>
              </GradientCard>

              <GradientCard title="3. Online vs Offline" color="blue">
                <p className="text-sm">
                  When are responses generated? From a static dataset collected
                  before training (offline)? Or by the current model during
                  training (online)?
                </p>
              </GradientCard>

              <GradientCard title="4. Reward Model" color="blue">
                <p className="text-sm">
                  Is there a separate learned reward function that scores
                  responses? Or is the reward signal implicit in the
                  optimization loss?
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Now place PPO and DPO on this map. They are not just
              &ldquo;different methods&rdquo;&mdash;they already differ on
              multiple axes:
            </p>

            <ComparisonRow
              left={{
                title: 'PPO on the Map',
                color: 'amber',
                items: [
                  'Data: comparison pairs',
                  'Reference model: required (for KL penalty)',
                  'Online: generates during training',
                  'Reward model: yes, separate learned model',
                ],
              }}
              right={{
                title: 'DPO on the Map',
                color: 'blue',
                items: [
                  'Data: comparison pairs',
                  'Reference model: required (implicit in loss)',
                  'Offline: trains on pre-collected data',
                  'Reward model: no, implicit in loss function',
                ],
              }}
            />

            <p className="text-muted-foreground">
              PPO and DPO share the same data format (comparison pairs) and
              both require a reference model. They differ on online-vs-offline
              and on whether there is a separate reward model. Every variation
              we are about to see changes <strong>at least one</strong> of
              these axes.
            </p>

            <SparseDesignSpaceMap />
            <p className="text-muted-foreground text-sm text-center mt-2">
              Two methods placed, with visible gaps. The dashed circles mark
              positions no existing method covers yet. Each gap is a design
              pressure that led to a new technique.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Axes, Not Features">
            These axes are independent dimensions, not a feature checklist. A
            method&rsquo;s position on one axis does not determine its
            position on another. That is what makes this a <em>space</em>,
            not a list.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1: Predict (Section 5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Predict" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Look at the four axes. PPO and DPO both require comparison
                pairs and a reference model.
              </p>
              <p>
                <strong>
                  What would a method look like that uses single
                  responses instead of pairs? What would you gain? What would
                  you lose?
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>You would gain:</strong> Access to a much larger
                    pool of training data. Most real-world feedback is
                    thumbs-up/down on single responses, not paired comparisons.
                    App users rate one response at a time.
                  </p>
                  <p>
                    <strong>You would lose:</strong> The relative signal. A
                    comparison pair says &ldquo;A is better than B&rdquo;&mdash;
                    the relative ranking is informative even when absolute
                    quality is hard to judge. A single label says
                    &ldquo;good&rdquo; or &ldquo;bad&rdquo; but not
                    &ldquo;better than what?&rdquo;
                  </p>
                  <p>
                    This is exactly the tradeoff KTO makes. You predicted the
                    design pressure correctly.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain: The Variations (Section 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Variations"
            subtitle="Each method moves along one or more axes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Choosing an alignment technique is like choosing a vehicle for a
              trip. A car (DPO) works for most trips with good roads. A
              motorcycle (KTO) carries less cargo but goes places the car
              cannot. A bus (PPO) carries the most but needs infrastructure.
              There is no &ldquo;best vehicle&rdquo;&mdash;only the best one
              for your constraints.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Tradeoffs, Not Upgrades">
            Each variation that follows solves a specific constraint. None is
            universally better than DPO. Resist the instinct to rank them.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* --- IPO --- */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="IPO — Identity Preference Optimization" color="violet">
              <div className="space-y-3 text-sm">
                <p>
                  <strong>What it changes:</strong> DPO treats &ldquo;slightly
                  preferred&rdquo; and &ldquo;overwhelmingly preferred&rdquo;
                  the same way. If Response A is barely better than B, DPO
                  optimizes just as hard as if A were dramatically better. IPO{' '}
                  <strong>bounds the preference signal</strong>&mdash;it says
                  &ldquo;this response is better, but only by this much.&rdquo;
                </p>
                <p>
                  <strong>Position on the map:</strong> Same as DPO on most
                  axes (paired data, reference model, offline, no reward
                  model). Differs on: bounded vs unbounded preference signal.
                  This is a refinement of DPO, not a departure.
                </p>
                <p>
                  <strong>Connection:</strong> If DPO is the editor saying
                  &ldquo;this draft is better,&rdquo; IPO is the editor saying
                  &ldquo;this draft is <em>slightly</em> better.&rdquo; The
                  calibration prevents overfitting on weak preferences.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Bounded Preferences">
            DPO can overfit on noisy preference pairs because it pushes
            probabilities as far as possible. IPO caps the signal, trading
            some optimization strength for stability.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* --- KTO --- */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="KTO — Kahneman-Tversky Optimization" color="cyan">
              <div className="space-y-3 text-sm">
                <p>
                  <strong>What it changes:</strong> KTO does not require paired
                  comparisons at all. It works with{' '}
                  <strong>single responses labeled good or bad</strong>{' '}
                  (thumbs-up/thumbs-down). This moves along the data format
                  axis&mdash;the most practical axis for real-world alignment.
                </p>
                <p>
                  <strong>Position on the map:</strong> Moves from
                  &ldquo;comparison pairs&rdquo; to &ldquo;single
                  responses.&rdquo; Still requires a reference model. Offline.
                  No separate reward model.
                </p>
                <p>
                  <strong>The insight from prospect theory:</strong> KTO is
                  named after Kahneman and Tversky&rsquo;s finding that humans
                  are more sensitive to losses than gains. The model is
                  penalized <em>more</em> for generating a bad response than
                  it is rewarded for generating a good one. This asymmetry
                  mirrors how real users react&mdash;a single bad response
                  damages trust more than a good response builds it.
                </p>
                <p>
                  <strong>Connection:</strong> This solves Scenario A from
                  the hook. Most real user feedback IS thumbs-up/thumbs-down,
                  not paired comparisons. KTO fits the data you actually have.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Data You Actually Have">
            Paired comparisons are expensive to collect (show two responses,
            ask which is better). Single ratings are cheap (show one response,
            ask thumbs-up or thumbs-down). KTO lets you use the feedback
            format most apps already collect.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* --- ORPO --- */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="ORPO — Odds Ratio Preference Optimization" color="emerald">
              <div className="space-y-3 text-sm">
                <p>
                  <strong>What it changes:</strong> ORPO eliminates the
                  reference model entirely. It folds preference optimization
                  into the SFT training process using an odds ratio penalty.
                  Instead of a two-phase approach (SFT, then preference
                  optimization), ORPO does both in a single pass.
                </p>
                <p>
                  <strong>Position on the map:</strong> Moves from
                  &ldquo;reference model required&rdquo; to &ldquo;not
                  required.&rdquo; Still uses paired data. Offline. No
                  separate reward model.
                </p>
                <p>
                  <strong>Connection:</strong> This solves Scenario B from
                  the hook. But remember why the reference model
                  exists&mdash;it prevents catastrophic drift. The KL penalty
                  is the continuous version of &ldquo;freeze the
                  backbone.&rdquo; ORPO removes this anchor, which means it
                  needs another mechanism (the odds ratio) to maintain
                  stability. <strong>Simplification in one dimension creates
                  complexity in another.</strong>
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Removing the Anchor">
            The reference model in DPO is not just extra weight in GPU
            memory&mdash;it <em>is</em> the KL penalty mechanism. Removing
            it is not free. ORPO trades one form of complexity (memory) for
            another (sensitivity to data quality).
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* --- Design Space Diagram --- */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Design Space"
            subtitle="All five methods mapped across four axes"
          />
          <DesignSpaceMap />
          <p className="text-muted-foreground text-sm text-center mt-2">
            Each row is an independent axis. Each method occupies a different
            position. No single method dominates all four axes.
          </p>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Read the Gaps">
            The interesting part of a design space map is not where the dots
            are&mdash;it is where they are <em>not</em>. Gaps suggest
            unexplored tradeoffs or fundamental constraints.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* --- Comparison Table --- */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Method Comparison"
            subtitle="All five methods across all four axes at a glance"
          />
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 pr-3 text-muted-foreground font-semibold">Method</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-semibold">Data Format</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-semibold">Reference Model</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-semibold">Online / Offline</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-semibold">Reward Model</th>
                  <th className="text-left py-2 pl-3 text-muted-foreground font-semibold">Key Insight</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border/50">
                  <td className="py-2 pr-3 font-medium text-amber-400">PPO</td>
                  <td className="py-2 px-3">Comparison pairs</td>
                  <td className="py-2 px-3">Required (KL penalty)</td>
                  <td className="py-2 px-3">Online</td>
                  <td className="py-2 px-3">Yes, separate</td>
                  <td className="py-2 pl-3">Full infrastructure, maximum control</td>
                </tr>
                <tr className="border-b border-border/50">
                  <td className="py-2 pr-3 font-medium text-indigo-400">DPO</td>
                  <td className="py-2 px-3">Comparison pairs</td>
                  <td className="py-2 px-3">Required (implicit in loss)</td>
                  <td className="py-2 px-3">Offline</td>
                  <td className="py-2 px-3">No, implicit</td>
                  <td className="py-2 pl-3">Simpler pipeline, one model</td>
                </tr>
                <tr className="border-b border-border/50">
                  <td className="py-2 pr-3 font-medium text-violet-400">IPO</td>
                  <td className="py-2 px-3">Comparison pairs</td>
                  <td className="py-2 px-3">Required</td>
                  <td className="py-2 px-3">Offline</td>
                  <td className="py-2 px-3">No, implicit</td>
                  <td className="py-2 pl-3">Bounded preferences prevent overfitting</td>
                </tr>
                <tr className="border-b border-border/50">
                  <td className="py-2 pr-3 font-medium text-cyan-400">KTO</td>
                  <td className="py-2 px-3">Single responses</td>
                  <td className="py-2 px-3">Required</td>
                  <td className="py-2 px-3">Offline</td>
                  <td className="py-2 px-3">No, implicit</td>
                  <td className="py-2 pl-3">Works with thumbs-up/down, not pairs</td>
                </tr>
                <tr>
                  <td className="py-2 pr-3 font-medium text-emerald-400">ORPO</td>
                  <td className="py-2 px-3">Comparison pairs</td>
                  <td className="py-2 px-3">Not required</td>
                  <td className="py-2 px-3">Offline</td>
                  <td className="py-2 px-3">No, implicit</td>
                  <td className="py-2 pl-3">Folds alignment into SFT, no reference model</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Scannable Reference">
            This table compresses the design space diagram into a format you
            can quickly scan. Use it to compare methods on any single axis by
            reading down a column.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain: Online vs Offline (Section 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Online vs Offline"
            subtitle="When are responses generated, and why does it matter?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The online/offline axis deserves its own section because it is
              orthogonal to the method choice&mdash;you can run DPO either
              way, and the same is true for most variations.
            </p>

            <ComparisonRow
              left={{
                title: 'Offline',
                color: 'blue',
                items: [
                  'Train on a pre-collected static dataset of preferences',
                  'This is what DPO, IPO, KTO, ORPO do by default',
                  'Cheaper: no generation step during training',
                  'Risk: data was generated by an older model version',
                  'The model is "tested" on a distribution it did not "train" on',
                ],
              }}
              right={{
                title: 'Online',
                color: 'amber',
                items: [
                  'Generate responses with the current model during training',
                  'This is what PPO does natively',
                  'More expensive: N forward passes per training step',
                  'Advantage: trains on the distribution the model actually produces',
                  'Closes the train/test distribution gap',
                ],
              }}
            />

            <div className="space-y-4">
              <p className="text-muted-foreground">
                <strong>Online DPO</strong> combines these: apply DPO&rsquo;s
                simpler optimization but generate fresh data during training.
                You get DPO&rsquo;s single-model simplicity with online
                learning&rsquo;s distribution match. This addresses Scenario C
                from the hook.
              </p>

              <p className="text-muted-foreground">
                <strong>Iterative alignment</strong> takes this further: run
                multiple rounds of generate, label, train. Each round produces
                a better model whose outputs feed the next round. Connection to
                constitutional AI&mdash;RLAIF can generate the preference
                labels in each round, making iteration practical at scale.
              </p>
            </div>

            <GradientCard title="The Cold-Start Problem" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Online methods sound strictly better&mdash;&ldquo;train on
                  what you actually produce&rdquo; is more principled. But
                  there is a catch: <strong>early in training, the model
                  generates low-quality responses.</strong>
                </p>
                <p>
                  If you train on the model&rsquo;s own bad outputs before it
                  has improved, you can reinforce bad behavior. The model
                  generates poor responses, trains on them, and gets worse. This
                  cold-start problem is why many teams start with offline
                  training and switch to online only after the model is already
                  decent.
                </p>
                <p>
                  <strong>The practical reality:</strong> for many cases the
                  performance gap between online and offline is small when
                  the offline preference data is high quality. Most deployed
                  models use offline methods because the cost/quality tradeoff
                  favors it.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Online ≠ Always Better">
            Online training generates N responses per training step, each
            requiring a full forward pass. For a 70B model, that is
            enormous additional compute. The theoretical advantage is real,
            but the cost often is not justified.
          </WarningBlock>
          <TipBlock title="Iterative Alignment">
            Iterative alignment does not require a single long training run.
            You can run offline DPO, generate new data from the improved
            model, and run another round. Multiple discrete rounds rather
            than continuous online generation.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 2: Design Space Navigation (Section 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Place a New Method" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A new paper describes a method with these properties:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>
                  Works with single responses (thumbs-up/down), not pairs
                </li>
                <li>Does not require a reference model</li>
                <li>Trains on a static, pre-collected dataset</li>
                <li>No separate reward model</li>
              </ul>
              <p>
                <strong>
                  Where would this method sit on the design space map? Which
                  existing methods is it closest to? What tradeoffs would you
                  expect?
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Position:</strong> Data format axis: single
                    responses (like KTO). Reference model: not required (like
                    ORPO). Online/offline: offline (like DPO). Reward model:
                    no (like DPO).
                  </p>
                  <p>
                    <strong>Closest to:</strong> A hybrid of KTO (data format)
                    and ORPO (no reference model). It combines the data
                    flexibility of KTO with the memory efficiency of ORPO.
                  </p>
                  <p>
                    <strong>Expected tradeoffs:</strong> Without paired data
                    AND without a reference model, this method has fewer
                    constraints to prevent degeneration. You would expect it
                    to be more sensitive to data quality and potentially less
                    stable during training. It gains maximum flexibility at
                    the cost of maximum guardrails.
                  </p>
                  <p>
                    If you could reason about the tradeoffs before being
                    told, <strong>the map is working</strong>. You do not need
                    to know the method&rsquo;s name or loss function to
                    predict its behavior.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Elaborate: Choosing a Method (Section 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Choosing a Method"
            subtitle="The design space map as a decision tool"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Return to the three scenarios from the hook. The map gives you
              the answer:
            </p>

            <div className="space-y-3">
              <GradientCard title="Scenario A: Thumbs-Up/Down Data → KTO" color="cyan">
                <p className="text-sm">
                  Your constraint is data format. You have single-response
                  feedback, not pairs. KTO moves along the data format axis to
                  meet you where your data is.
                </p>
              </GradientCard>

              <GradientCard title="Scenario B: Memory-Constrained → ORPO" color="emerald">
                <p className="text-sm">
                  Your constraint is GPU memory. You cannot afford the
                  reference model. ORPO removes it&mdash;at the cost of
                  needing another stability mechanism.
                </p>
              </GradientCard>

              <GradientCard title="Scenario C: Learn From Current Mistakes → Online DPO" color="amber">
                <p className="text-sm">
                  Your constraint is distribution mismatch. You want to train
                  on what the model currently produces. Online DPO (or
                  iterative alignment) moves along the online/offline axis.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Now a fourth scenario:
            </p>

            <GradientCard title="Scenario D: Well-Resourced Team → DPO" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  You have a well-funded team with high-quality paired
                  comparisons from expert annotators. You have enough GPU
                  memory for the reference model. You do not need single-response
                  data because you specifically collected comparison pairs.
                </p>
                <p>
                  <strong>The answer is DPO</strong>&mdash;the
                  &ldquo;oldest&rdquo; variation in this lesson. It has the
                  most research validation, the most stable training dynamics,
                  and the simplest optimization. When your constraints do not
                  force you elsewhere on the map, DPO is the established
                  choice.
                </p>
                <p className="text-blue-400/80">
                  Newer does not mean better. It means different tradeoffs.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="The Landscape Keeps Growing" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  More alignment techniques do not mean more aligned models.
                  The landscape exists because{' '}
                  <strong>different teams face different constraints</strong>.
                  A startup with user thumbs-up/down data needs a different
                  method than a research lab with expert annotators. A team
                  training a 7B model has different memory constraints than a
                  team training a 70B model.
                </p>
                <p>
                  The field is not converging on one method. It is mapping a
                  design space. When you encounter a new alignment technique
                  next month (and you will), ask:{' '}
                  <strong>where does it sit on these four axes? What
                  constraint does it relax? What does it give up?</strong>
                </p>
                <p>
                  And remember: preference optimization is one approach to
                  alignment among several. Constitutional AI&rsquo;s
                  critique-and-revision loop (from the previous lesson) is
                  another. Red teaming and adversarial evaluation (next lesson)
                  is yet another lens entirely. This lesson mapped one region of
                  a larger landscape.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Constraints Drive Choice">
            The right method depends on your constraints, not on which
            paper is newest. Data format, memory budget, compute budget,
            and data quality are the actual decision variables.
          </InsightBlock>
          <WarningBlock title="Newer ≠ Better">
            DPO outperforms KTO when you have high-quality paired data.
            ORPO is more sensitive to data quality than DPO. Each
            &ldquo;improvement&rdquo; is actually a tradeoff along a
            different axis.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Practice — Notebook (Section 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Explore the design axes hands-on with GPT-2"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook demonstrates three of the four design axes using
              small-scale proxies. You will not train full alignment
              models&mdash;you will manipulate preference data, compute log
              probabilities, and see distribution mismatch directly.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Preference Data Format Conversion (Guided)" color="blue">
                <p className="text-sm">
                  Convert between paired comparison format and single-response
                  thumbs-up/down format. See what information is lost when you
                  drop from relative to absolute labels&mdash;this is why KTO
                  exists.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Reference Model Drift (Supported)" color="blue">
                <p className="text-sm">
                  Compute log-probability ratios between a policy model and a
                  reference model. Visualize how the KL divergence grows as the
                  policy drifts&mdash;the reference model constraint in action.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Online vs Offline Mismatch (Supported)" color="blue">
                <p className="text-sm">
                  See how a policy&rsquo;s output distribution shifts after an
                  update, making pre-collected preference data stale. This is
                  the core motivation for online methods.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-1-2-alignment-techniques-landscape.ipynb"
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
          <TipBlock title="Design Axes in Code">
            Each exercise targets a different axis from the lesson. By the end,
            three of the four axes have a concrete, code-level intuition behind
            them&mdash;not just a diagram.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Summary (Section 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'The PPO/DPO binary was a starting point, not the full picture.',
                description:
                  'The design space of preference optimization extends along multiple independent axes. PPO and DPO are two points in this space, not the only two options.',
              },
              {
                headline:
                  'Four axes map the design space: data format, reference model, online vs offline, reward model.',
                description:
                  'Each axis asks a different question about how a method works. A method\'s position on one axis does not determine its position on another.',
              },
              {
                headline:
                  'Each method makes different tradeoffs\u2014there is no universally "best" technique.',
                description:
                  'IPO bounds preference signals to prevent overfitting. KTO uses single responses instead of pairs. ORPO removes the reference model. Each solves a specific constraint at the cost of something else.',
              },
              {
                headline:
                  'The map is more durable than any specific method.',
                description:
                  'When you encounter a new alignment technique, ask: where does it sit on these axes? What constraint does it relax? What does it give up? The framework outlasts any individual method.',
              },
              {
                headline:
                  'Alignment techniques are points in a design space, not steps on a ladder.',
                description:
                  'The field is not converging on one method. It is mapping a space of tradeoffs driven by different teams facing different constraints.',
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
                title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
                authors: 'Rafailov et al., 2023',
                url: 'https://arxiv.org/abs/2305.18290',
                note: 'The DPO paper. Section 4 derives the loss function that all variations build upon. Useful as the anchor point for understanding what IPO, KTO, and ORPO each change.',
              },
              {
                title: 'A General Theoretical Paradigm to Understand Learning from Human Feedback',
                authors: 'Azar et al., 2023 (IPO)',
                url: 'https://arxiv.org/abs/2310.12036',
                note: 'Introduces IPO. Section 3 explains why DPO can overfit and how bounding the preference signal addresses it.',
              },
              {
                title: 'KTO: Model Alignment as Prospect Theoretic Optimization',
                authors: 'Ethayarajh et al., 2024',
                url: 'https://arxiv.org/abs/2402.01306',
                note: 'The KTO paper. Sections 1-2 explain the connection to prospect theory and why single-response data is sufficient.',
              },
              {
                title: 'ORPO: Monolithic Preference Optimization without Reference Model',
                authors: 'Hong et al., 2024',
                url: 'https://arxiv.org/abs/2403.07691',
                note: 'Introduces ORPO. Focus on Section 3 for how the odds ratio replaces the reference model\'s KL constraint.',
              },
              {
                title: 'Proximal Policy Optimization Algorithms',
                authors: 'Schulman et al., 2017',
                url: 'https://arxiv.org/abs/1707.06347',
                note: 'The original PPO paper. Background for understanding the full-infrastructure end of the design space.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Next Step (Section 12)
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Review the design space diagram and make sure you can place each method (PPO, DPO, IPO, KTO, ORPO) on all four axes from memory. Test yourself: if someone described a new method's constraints, could you predict where it sits? Next up: lessons 1 and 2 built alignment techniques. Now we ask—how do you find what they missed? Red teaming is the adversarial complement to everything we have built."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
