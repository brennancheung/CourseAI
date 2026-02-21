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
 * Mixture of Experts
 *
 * First lesson in Module 5.3 (Scaling Architecture).
 * STRETCH lesson — introduces conditional computation, a new paradigm.
 *
 * Cognitive load: 2-3 new concepts:
 *   1. Conditional computation (not every parameter activates per token)
 *   2. Router mechanism (linear layer + softmax + top-k)
 *   3. Load balancing / auxiliary loss (borderline third)
 *
 * Core concepts at DEVELOPED:
 * - Conditional computation (decoupling parameters from compute)
 * - Router mechanism (connection to attention's dot-product-softmax)
 *
 * Concepts at INTRODUCED:
 * - Load balancing / auxiliary loss (problem + concept of solution)
 * - Expert specialization patterns (emergent, per-token)
 * - Mixtral / DeepSeek-V3 as MoE examples
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing MoE in code (notebook demos routing on a small proxy)
 * - Training an MoE model from scratch
 * - Token dropping or capacity factors in detail
 * - Switch Transformer or other historical MoE variants in depth
 * - Communication overhead across devices (deferred to Lesson 3)
 * - Comparing specific benchmark results
 *
 * Previous: Reasoning Models (Module 5.2, Lesson 4)
 * Next: Long Context & Efficient Attention (Module 5.3, Lesson 2)
 */

// ---------------------------------------------------------------------------
// Inline SVG: MoE Block Architecture Diagram
// Side-by-side comparison of standard transformer block vs MoE block.
// Color-coded: shared (violet), router (amber), active experts (emerald),
// inactive experts (gray).
// ---------------------------------------------------------------------------

function MoEArchitectureDiagram() {
  const svgW = 620
  const svgH = 460

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const sharedColor = '#a78bfa' // violet for shared components
  const routerColor = '#f59e0b' // amber for router
  const activeColor = '#34d399' // emerald for selected experts
  const inactiveColor = '#475569' // gray for inactive experts
  const arrowColor = '#64748b'

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
          Standard Block vs MoE Block
        </text>
        <text
          x={svgW / 2}
          y={42}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          Only the FFN changes. Attention and the residual stream are identical.
        </text>

        {/* ===== LEFT SIDE: Standard Transformer Block ===== */}
        <text
          x={150}
          y={70}
          textAnchor="middle"
          fill={brightText}
          fontSize="11"
          fontWeight="600"
        >
          Standard Block
        </text>

        {/* Input */}
        <rect x={80} y={85} width={140} height={32} rx={6} fill={sharedColor} fillOpacity={0.12} stroke={sharedColor} strokeWidth={1.5} />
        <text x={150} y={105} textAnchor="middle" fill={sharedColor} fontSize="10" fontWeight="500">Input (hidden state)</text>

        {/* Arrow */}
        <line x1={150} y1={117} x2={150} y2={135} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="150,139 145,131 155,131" fill={arrowColor} />

        {/* Attention */}
        <rect x={80} y={141} width={140} height={32} rx={6} fill={sharedColor} fillOpacity={0.12} stroke={sharedColor} strokeWidth={1.5} />
        <text x={150} y={161} textAnchor="middle" fill={sharedColor} fontSize="10" fontWeight="500">Multi-Head Attention</text>

        {/* Arrow + residual label */}
        <line x1={150} y1={173} x2={150} y2={191} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="150,195 145,187 155,187" fill={arrowColor} />
        <text x={230} y={186} textAnchor="start" fill={dimText} fontSize="8">+ residual</text>

        {/* FFN (single block) */}
        <rect x={80} y={197} width={140} height={55} rx={6} fill={sharedColor} fillOpacity={0.12} stroke={sharedColor} strokeWidth={1.5} />
        <text x={150} y={218} textAnchor="middle" fill={sharedColor} fontSize="10" fontWeight="500">FFN</text>
        <text x={150} y={232} textAnchor="middle" fill={dimText} fontSize="8">(all parameters active)</text>
        <text x={150} y={244} textAnchor="middle" fill={dimText} fontSize="8">768 → 3072 → 768</text>

        {/* Arrow + residual */}
        <line x1={150} y1={252} x2={150} y2={270} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="150,274 145,266 155,266" fill={arrowColor} />
        <text x={230} y={265} textAnchor="start" fill={dimText} fontSize="8">+ residual</text>

        {/* Output */}
        <rect x={80} y={276} width={140} height={32} rx={6} fill={sharedColor} fillOpacity={0.12} stroke={sharedColor} strokeWidth={1.5} />
        <text x={150} y={296} textAnchor="middle" fill={sharedColor} fontSize="10" fontWeight="500">Output</text>

        {/* ===== RIGHT SIDE: MoE Block ===== */}
        <text
          x={460}
          y={70}
          textAnchor="middle"
          fill={brightText}
          fontSize="11"
          fontWeight="600"
        >
          MoE Block
        </text>

        {/* Input */}
        <rect x={390} y={85} width={140} height={32} rx={6} fill={sharedColor} fillOpacity={0.12} stroke={sharedColor} strokeWidth={1.5} />
        <text x={460} y={105} textAnchor="middle" fill={sharedColor} fontSize="10" fontWeight="500">Input (hidden state)</text>

        {/* Arrow */}
        <line x1={460} y1={117} x2={460} y2={135} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="460,139 455,131 465,131" fill={arrowColor} />

        {/* Attention (shared, identical) */}
        <rect x={390} y={141} width={140} height={32} rx={6} fill={sharedColor} fillOpacity={0.12} stroke={sharedColor} strokeWidth={1.5} />
        <text x={460} y={161} textAnchor="middle" fill={sharedColor} fontSize="10" fontWeight="500">Multi-Head Attention</text>

        {/* Arrow + residual */}
        <line x1={460} y1={173} x2={460} y2={191} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="460,195 455,187 465,187" fill={arrowColor} />
        <text x={540} y={186} textAnchor="start" fill={dimText} fontSize="8">+ residual</text>

        {/* Router */}
        <rect x={410} y={197} width={100} height={28} rx={6} fill={routerColor} fillOpacity={0.15} stroke={routerColor} strokeWidth={1.5} />
        <text x={460} y={215} textAnchor="middle" fill={routerColor} fontSize="10" fontWeight="500">Router</text>

        {/* Arrows from router to experts */}
        <line x1={430} y1={225} x2={375} y2={250} stroke={activeColor} strokeWidth={1.5} />
        <polygon points="373,253 378,244 383,252" fill={activeColor} />

        <line x1={445} y1={225} x2={425} y2={250} stroke={activeColor} strokeWidth={1.5} />
        <polygon points="423,253 428,244 433,252" fill={activeColor} />

        <line x1={475} y1={225} x2={480} y2={250} stroke={inactiveColor} strokeWidth={1} strokeDasharray="3,3" />

        <line x1={490} y1={225} x2={530} y2={250} stroke={inactiveColor} strokeWidth={1} strokeDasharray="3,3" />

        {/* Expert boxes */}
        {/* Expert 1 (active) */}
        <rect x={350} y={255} width={48} height={38} rx={4} fill={activeColor} fillOpacity={0.15} stroke={activeColor} strokeWidth={1.5} />
        <text x={374} y={272} textAnchor="middle" fill={activeColor} fontSize="8" fontWeight="500">E₁</text>
        <text x={374} y={286} textAnchor="middle" fill={activeColor} fontSize="7">active</text>

        {/* Expert 2 (active) */}
        <rect x={404} y={255} width={48} height={38} rx={4} fill={activeColor} fillOpacity={0.15} stroke={activeColor} strokeWidth={1.5} />
        <text x={428} y={272} textAnchor="middle" fill={activeColor} fontSize="8" fontWeight="500">E₂</text>
        <text x={428} y={286} textAnchor="middle" fill={activeColor} fontSize="7">active</text>

        {/* Expert 3 (inactive) */}
        <rect x={458} y={255} width={48} height={38} rx={4} fill={inactiveColor} fillOpacity={0.08} stroke={inactiveColor} strokeWidth={1} strokeDasharray="3,3" />
        <text x={482} y={272} textAnchor="middle" fill={inactiveColor} fontSize="8">E₃</text>
        <text x={482} y={286} textAnchor="middle" fill={inactiveColor} fontSize="7">skip</text>

        {/* Expert N (inactive) */}
        <rect x={512} y={255} width={48} height={38} rx={4} fill={inactiveColor} fillOpacity={0.08} stroke={inactiveColor} strokeWidth={1} strokeDasharray="3,3" />
        <text x={536} y={272} textAnchor="middle" fill={inactiveColor} fontSize="8">Eₙ</text>
        <text x={536} y={286} textAnchor="middle" fill={inactiveColor} fontSize="7">skip</text>

        {/* Dots between E3 and EN */}
        <text x={508} y={275} textAnchor="middle" fill={inactiveColor} fontSize="10">...</text>

        {/* Weighted sum node */}
        <line x1={374} y1={293} x2={440} y2={320} stroke={activeColor} strokeWidth={1.5} />
        <line x1={428} y1={293} x2={440} y2={320} stroke={activeColor} strokeWidth={1.5} />

        <circle cx={440} cy={325} r={14} fill={activeColor} fillOpacity={0.15} stroke={activeColor} strokeWidth={1.5} />
        <text x={440} y={329} textAnchor="middle" fill={activeColor} fontSize="9" fontWeight="600">+</text>
        <text x={470} y={329} textAnchor="start" fill={dimText} fontSize="7">weighted sum</text>

        {/* Arrow to output + residual */}
        <line x1={440} y1={339} x2={460} y2={365} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="460,369 455,361 465,361" fill={arrowColor} />
        <text x={540} y={362} textAnchor="start" fill={dimText} fontSize="8">+ residual</text>

        {/* Output */}
        <rect x={390} y={371} width={140} height={32} rx={6} fill={sharedColor} fillOpacity={0.12} stroke={sharedColor} strokeWidth={1.5} />
        <text x={460} y={391} textAnchor="middle" fill={sharedColor} fontSize="10" fontWeight="500">Output</text>

        {/* Legend */}
        <rect x={60} y={330} width={10} height={10} rx={2} fill={sharedColor} fillOpacity={0.3} stroke={sharedColor} strokeWidth={1} />
        <text x={76} y={340} fill={dimText} fontSize="8">Shared (unchanged)</text>

        <rect x={60} y={348} width={10} height={10} rx={2} fill={routerColor} fillOpacity={0.3} stroke={routerColor} strokeWidth={1} />
        <text x={76} y={358} fill={dimText} fontSize="8">Router (new)</text>

        <rect x={60} y={366} width={10} height={10} rx={2} fill={activeColor} fillOpacity={0.3} stroke={activeColor} strokeWidth={1} />
        <text x={76} y={376} fill={dimText} fontSize="8">Active experts (top-k)</text>

        <rect x={60} y={384} width={10} height={10} rx={2} fill={inactiveColor} fillOpacity={0.08} stroke={inactiveColor} strokeWidth={1} strokeDasharray="2,2" />
        <text x={76} y={394} fill={dimText} fontSize="8">Inactive experts (skipped)</text>

        {/* Divider */}
        <line x1={310} y1={65} x2={310} y2={410} stroke={arrowColor} strokeWidth={1} strokeDasharray="4,4" />
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Per-Token Routing Visualization
// Shows "The mitochondria is the powerhouse of the cell" with expert
// assignments per token.
// ---------------------------------------------------------------------------

function PerTokenRoutingDiagram() {
  const svgW = 600
  const svgH = 260

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'

  // Expert color assignments
  const expertColors: Record<string, { fill: string; label: string }> = {
    'E2': { fill: '#60a5fa', label: 'Expert 2' },
    'E5': { fill: '#f59e0b', label: 'Expert 5' },
    'E7': { fill: '#a78bfa', label: 'Expert 7' },
  }

  const tokens = [
    { text: 'The', expert: 'E2', x: 30 },
    { text: 'mitochondria', expert: 'E5', x: 80 },
    { text: 'is', expert: 'E2', x: 200 },
    { text: 'the', expert: 'E2', x: 230 },
    { text: 'powerhouse', expert: 'E7', x: 280 },
    { text: 'of', expert: 'E2', x: 400 },
    { text: 'the', expert: 'E2', x: 430 },
    { text: 'cell', expert: 'E5', x: 475 },
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
          fill={brightText}
          fontSize="12"
          fontWeight="600"
        >
          Per-Token Expert Routing
        </text>
        <text
          x={svgW / 2}
          y={36}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          Different tokens in the same sentence activate different experts
        </text>

        {/* Tokens with expert color coding */}
        {tokens.map((token, i) => {
          const color = expertColors[token.expert].fill
          const y = 75
          const textWidth = token.text.length * 8 + 16
          return (
            <g key={i}>
              <rect
                x={token.x}
                y={y}
                width={textWidth}
                height={28}
                rx={4}
                fill={color}
                fillOpacity={0.15}
                stroke={color}
                strokeWidth={1.5}
              />
              <text
                x={token.x + textWidth / 2}
                y={y + 17}
                textAnchor="middle"
                fill={brightText}
                fontSize="10"
                fontWeight="500"
              >
                {token.text}
              </text>
              {/* Expert label below */}
              <text
                x={token.x + textWidth / 2}
                y={y + 45}
                textAnchor="middle"
                fill={color}
                fontSize="8"
                fontWeight="500"
              >
                {token.expert}
              </text>
              {/* Connecting line */}
              <line
                x1={token.x + textWidth / 2}
                y1={y + 28}
                x2={token.x + textWidth / 2}
                y2={y + 36}
                stroke={color}
                strokeWidth={1}
              />
            </g>
          )
        })}

        {/* Observation boxes */}
        <rect x={30} y={150} width={240} height={44} rx={6} fill="#60a5fa" fillOpacity={0.08} stroke="#60a5fa" strokeWidth={1} strokeOpacity={0.3} />
        <text x={150} y={168} textAnchor="middle" fill="#60a5fa" fontSize="9" fontWeight="500">Function words: &quot;The,&quot; &quot;is,&quot; &quot;the,&quot; &quot;of,&quot; &quot;the&quot;</text>
        <text x={150} y={183} textAnchor="middle" fill={dimText} fontSize="8">All route to Expert 2 (syntax patterns)</text>

        <rect x={300} y={150} width={270} height={44} rx={6} fill="#f59e0b" fillOpacity={0.08} stroke="#f59e0b" strokeWidth={1} strokeOpacity={0.3} />
        <text x={435} y={168} textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="500">&quot;mitochondria,&quot; &quot;cell&quot; → Expert 5 (science)</text>
        <text x={435} y={183} textAnchor="middle" fill="#a78bfa" fontSize="9" fontWeight="500">&quot;powerhouse&quot; → Expert 7 (domain)</text>

        {/* Key takeaway */}
        <rect x={80} y={210} width={440} height={35} rx={6} fill={dimText} fillOpacity={0.06} stroke={dimText} strokeWidth={1} strokeOpacity={0.2} />
        <text x={svgW / 2} y={232} textAnchor="middle" fill={dimText} fontSize="9">
          Specialization is emergent and per-token, not designed and per-topic
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Router Collapse Diagram
// Shows the positive feedback loop that leads to expert collapse.
// ---------------------------------------------------------------------------

function RouterCollapseDiagram() {
  const svgW = 520
  const svgH = 240

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const dangerColor = '#f87171' // rose
  const bgColor = '#1e293b'
  const arrowColor = '#64748b'

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Title */}
        <text x={svgW / 2} y={20} textAnchor="middle" fill={brightText} fontSize="12" fontWeight="600">
          Router Collapse: A Positive Feedback Loop
        </text>

        {/* Step 1 */}
        <rect x={20} y={50} width={120} height={45} rx={8} fill={bgColor} stroke={dangerColor} strokeWidth={1.5} />
        <text x={80} y={70} textAnchor="middle" fill={dangerColor} fontSize="9" fontWeight="500">Expert 3 starts</text>
        <text x={80} y={84} textAnchor="middle" fill={dimText} fontSize="8">slightly better (random)</text>

        {/* Arrow 1 */}
        <line x1={140} y1={72} x2={165} y2={72} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="169,72 161,67 161,77" fill={arrowColor} />

        {/* Step 2 */}
        <rect x={171} y={50} width={120} height={45} rx={8} fill={bgColor} stroke={dangerColor} strokeWidth={1.5} />
        <text x={231} y={70} textAnchor="middle" fill={dangerColor} fontSize="9" fontWeight="500">Router sends more</text>
        <text x={231} y={84} textAnchor="middle" fill={dimText} fontSize="8">tokens to Expert 3</text>

        {/* Arrow 2 */}
        <line x1={291} y1={72} x2={316} y2={72} stroke={arrowColor} strokeWidth={1.5} />
        <polygon points="320,72 312,67 312,77" fill={arrowColor} />

        {/* Step 3 */}
        <rect x={322} y={50} width={120} height={45} rx={8} fill={bgColor} stroke={dangerColor} strokeWidth={1.5} />
        <text x={382} y={70} textAnchor="middle" fill={dangerColor} fontSize="9" fontWeight="500">Expert 3 gets more</text>
        <text x={382} y={84} textAnchor="middle" fill={dimText} fontSize="8">gradient updates</text>

        {/* Curved feedback arrow (from step 3 back to step 2) */}
        <path
          d="M 382 95 C 382 130, 231 130, 231 95"
          fill="none"
          stroke={dangerColor}
          strokeWidth={1.5}
          strokeDasharray="4,3"
        />
        <polygon points="231,95 226,103 236,103" fill={dangerColor} />
        <text x={307} y={128} textAnchor="middle" fill={dangerColor} fontSize="8" fontWeight="500">
          positive feedback loop
        </text>

        {/* Result box */}
        <rect x={60} y={155} width={400} height={55} rx={8} fill={dangerColor} fillOpacity={0.08} stroke={dangerColor} strokeWidth={1} strokeOpacity={0.3} />
        <text x={260} y={175} textAnchor="middle" fill={dangerColor} fontSize="10" fontWeight="500">
          Result: Expert 3 handles 80%+ of tokens
        </text>
        <text x={260} y={192} textAnchor="middle" fill={dimText} fontSize="9">
          Other 7 experts are undertrained and effectively dead weight.
        </text>
        <text x={260} y={205} textAnchor="middle" fill={dimText} fontSize="9">
          The model degenerates to approximately a dense model.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function MixtureOfExpertsLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Mixture of Experts"
            description="Every dense transformer activates all its parameters for every token&mdash;even when most of the FFN's stored knowledge is irrelevant. What if the model could activate only the knowledge it needs? Mixture of experts replaces the monolithic FFN with specialized sub-networks and a learned router, decoupling total parameters from per-token computation."
            category="Scaling Architecture"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            By the end of this lesson, you will be able to explain why
            conditional computation targets the FFN specifically, trace how a
            token flows through a router to selected experts, and articulate
            the parameter-compute tradeoff using Mixtral&rsquo;s concrete
            numbers.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Conditional computation: activating only a subset of parameters per token',
              'MoE architecture: replacing the FFN with router + multiple expert FFNs',
              'Router mechanism: linear layer + softmax + top-k selection',
              'Parameter-compute decoupling: total parameters vs active parameters',
              'Load balancing: why it is necessary and how auxiliary loss prevents collapse',
              'Expert specialization: emergent, per-token, not designed',
              'NOT: implementing MoE in code (notebook exercises use toy routing)',
              'NOT: communication overhead of distributing experts across GPUs (Lesson 3)',
              'NOT: training recipes, hyperparameters, or MoE training from scratch',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="New Module">
            This begins the Scaling Architecture module. Every model you have
            studied so far activates all parameters for every input. This is the
            first lesson where that assumption changes.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap: Three Facts from the Transformer Block
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where You Left Off"
            subtitle="Three facts from The Transformer Block, about to combine"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              From The Transformer Block, you established three facts about the
              FFN sub-layer:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>&ldquo;Attention reads, FFN writes.&rdquo;</strong> The
                two complementary sub-layers of every transformer block.
                Attention gathers context from other tokens; the FFN processes
                what attention found and updates the representation.
              </li>
              <li>
                <strong>The FFN has ~2/3 of the parameters.</strong> In GPT-2:
                ~57M FFN parameters vs ~28M attention parameters per block. You
                were surprised by this&mdash;you expected attention to dominate.
              </li>
              <li>
                <strong>FFN neurons as key-value memories.</strong> Research
                by Geva et al. showed that individual FFN neurons store specific
                knowledge&mdash;facts, patterns, associations. The FFN is the
                model&rsquo;s knowledge store.
              </li>
            </ol>

            <p className="text-muted-foreground">
              Now connect them: if 2/3 of the model&rsquo;s capacity is stored
              knowledge, and most stored knowledge is irrelevant to any given
              token, what happens during a forward pass?
            </p>

            <p className="text-muted-foreground">
              And from Scaling &amp; Efficiency, Chinchilla taught you
              &ldquo;scale both, not just one&rdquo;&mdash;model size and data
              should grow together. But what if you could scale total parameters
              without proportionally scaling per-token compute?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Established Mental Models">
            &ldquo;Attention reads, FFN writes&rdquo; and &ldquo;2/3 of
            parameters store knowledge&rdquo; are about to combine into the
            motivation for MoE. Each fact alone is familiar. Together, they
            reveal a problem.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: The Waste Problem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Waste in Every Forward Pass"
            subtitle="What if most of the computation is unnecessary?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              GPT-3 has 175 billion parameters. Roughly ~117 billion of those
              are in FFN layers. Every token&mdash;whether &ldquo;the&rdquo;
              or &ldquo;mitochondria&rdquo;&mdash;activates all 117 billion FFN
              parameters. The function word &ldquo;the&rdquo; fires the same
              parameters that encode French cooking, JavaScript syntax, and
              medieval history.
            </p>

            <p className="text-muted-foreground">
              Consider how much of the FFN&rsquo;s stored knowledge could be
              relevant to any given token. &ldquo;The&rdquo; does not need the
              parameters that encode French cooking or JavaScript syntax. For
              most tokens, the vast majority of FFN knowledge is irrelevant.
              Every forward pass pays the full cost of the FFN even though most
              of the knowledge stored there has nothing to do with the current
              token.
            </p>

            <GradientCard title="The Problem" color="orange">
              <p className="text-sm">
                The dense transformer activates all knowledge for every token.
                Most of that knowledge is irrelevant. If the FFN is a knowledge
                store and most knowledge does not apply to the current
                token&mdash;<strong>what if the model could activate only the
                knowledge it needs?</strong>
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a New Idea">
            MoE was proposed in 1991 by Jacobs et al. The concept predates
            transformers by decades. What changed is the scale at which it
            became practical and the specific application to FFN sub-layers
            in transformer blocks.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain Part 1: The MoE Architecture
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The MoE Architecture"
            subtitle="Replace one FFN with many, add a router"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The solution: replace the single monolithic FFN with N smaller FFN
              &ldquo;experts&rdquo; and a router that selects which ones to
              activate for each token.
            </p>

            <p className="text-muted-foreground">
              Walk through the architecture change, piece by piece:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Same attention layer</strong>&mdash;completely unchanged.
                &ldquo;Attention reads&rdquo; is untouched.
              </li>
              <li>
                <strong>Same residual stream</strong>&mdash;unchanged. The
                shared backbone that carries information between layers.
              </li>
              <li>
                <strong>The FFN sub-layer changes:</strong> instead of one
                FFN(x), there are N expert FFNs (Expert₁(x), Expert₂(x), ...,
                Expertₙ(x)). Each has the same structure&mdash;two-layer
                network, expansion, GELU&mdash;but its own learned weights.
              </li>
              <li>
                <strong>A router network</strong> takes the token&rsquo;s hidden
                state and produces a probability distribution over experts.
              </li>
              <li>
                <strong>Top-k selection:</strong> only k experts (typically k=1
                or k=2) activate for each token. The rest are skipped entirely.
              </li>
              <li>
                <strong>Weighted output:</strong> the outputs of the selected
                experts are weighted by the router probabilities and summed, then
                added to the residual stream.
              </li>
            </ul>

            <MoEArchitectureDiagram />

            {/* Misconception 1: MoE is not an ensemble */}
            <GradientCard title="Misconception: &ldquo;MoE models are just ensembles of smaller models&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  In an ensemble, each model processes the entire input
                  independently and outputs are aggregated. In MoE, experts are
                  FFN sub-networks <strong>within a single transformer
                  block</strong>. They share the same attention layers, residual
                  stream, and embeddings. Only the FFN varies per token.
                </p>
                <p className="text-muted-foreground">
                  Think of it this way: the experts cannot function
                  independently. They depend on the shared attention output as
                  their input. An ensemble has 8 complete models. An MoE block
                  has 1 attention layer feeding 8 FFN alternatives.
                </p>
              </div>
            </GradientCard>

            {/* Library analogy */}
            <GradientCard title="The Library Analogy" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A dense FFN is <strong>one librarian who has read every
                  book</strong> and answers every question. An MoE FFN is a
                  library with <strong>8 specialist librarians</strong>
                  &mdash;one for biology, one for history, one for syntax&mdash;
                  plus a <strong>front desk clerk</strong> (the router) who
                  directs your question to the 2 most relevant librarians.
                </p>
                <p className="text-muted-foreground">
                  You get answers from 2 people instead of 1, but the library
                  collectively knows far more because each librarian only needs
                  deep expertise in their specialty. The front desk clerk
                  is not an expert&mdash;they just do a quick lookup to direct
                  your question.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Only the FFN Changes">
            &ldquo;Attention reads, the right experts write.&rdquo; MoE modifies
            only the &ldquo;writes&rdquo; half of the transformer block.
            Attention is identical. The residual stream is identical. The
            modification is targeted and minimal.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain Part 2: The Router Mechanism
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Router Mechanism"
            subtitle="Simpler than you think"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The router sounds like it might be a complex component&mdash;a
              small neural network that &ldquo;understands&rdquo; which expert
              is best for each token. In reality, it is remarkably simple:
            </p>

            <CodeBlock
              code={`# The entire router is ONE linear layer + softmax + top-k
# Compare to attention: same dot-product + softmax pattern

router_logits = W_router @ hidden_state     # shape: (num_experts,)
#               ^ one matrix, no bias       # one score per expert

router_probs = softmax(router_logits)       # probability distribution
#              ^ same softmax from attention  # over experts

top_k_indices, top_k_probs = top_k(router_probs, k=2)  # select top-2

# Run only the selected experts (the rest are skipped entirely)
output = sum(
    top_k_probs[i] * Expert_i(hidden_state)
    for i in top_k_indices
)
# ^ weighted combination, same as attention's weighted sum of values`}
              language="python"
              filename="moe_router.py"
            />

            <p className="text-muted-foreground">
              This is <strong>dot-product + softmax + weighted sum</strong>.
              The same pattern you have used dozens of times for attention, but
              selecting experts instead of tokens. The rows of W_router are
              learned &ldquo;expert embeddings&rdquo;; the dot product measures
              relevance of the token to each expert. One matrix multiplication
              and a softmax&mdash;simpler than a single attention head.
            </p>

            {/* Misconception 4: Router is simple */}
            <GradientCard title="Misconception: &ldquo;The router is a complex learned component&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  The router is a single linear layer. One matrix
                  multiplication followed by softmax. It has (num_experts
                  &times; d_model) parameters&mdash;for Mixtral with 8 experts
                  and d_model=4096, that is 32,768 parameters. The attention
                  mechanism in the same block has millions.
                </p>
                <p className="text-muted-foreground">
                  The routing &ldquo;decision&rdquo; is a dot product
                  between the token&rsquo;s hidden state and learned expert
                  embeddings, selected via top-k. The same mechanism as
                  attention scores (dot product + softmax), but over expert
                  indices instead of token positions.
                </p>
              </div>
            </GradientCard>

            {/* "Of course" beat */}
            <GradientCard title="Of Course" color="violet">
              <p className="text-sm italic">
                You already knew that 2/3 of parameters are in the FFN. You
                already knew FFN neurons store specific knowledge. If most
                knowledge is irrelevant to any given token, of course you should
                only activate the relevant knowledge. Of course the model should
                be able to have more total knowledge while keeping per-token
                compute constant.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Familiar Pattern">
            The router uses dot-product + softmax + top-k. Attention uses
            dot-product + softmax + weighted sum. The mechanism is the same.
            Attention selects which tokens to attend to. The router selects
            which experts to activate.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Explain Part 3: The Parameter-Compute Decoupling
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Parameter-Compute Decoupling"
            subtitle="The central insight: bigger without being slower"
          />
          <div className="space-y-4">
            {/* Misconception 3: More params = more compute */}
            <GradientCard title="Misconception: &ldquo;More parameters always means proportionally more compute&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Your entire experience&mdash;GPT-2 (124M), GPT-3 (175B), the
                  scaling laws lesson&mdash;has established a direct relationship
                  between parameter count and compute. &ldquo;Bigger model =
                  more computation per forward pass&rdquo; has been true for
                  every model you have seen. MoE breaks this relationship.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              Here is where the numbers make it concrete. Consider Mixtral
              8&times;7B&mdash;one of the most well-known MoE architectures:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>8 experts per MoE layer, top-2 routing</li>
              <li>Each expert &asymp; 7B-class FFN parameters</li>
              <li>Shared attention layers &asymp; same as a 7B model</li>
              <li><strong>Total parameters: ~47B</strong></li>
              <li><strong>Active parameters per token: ~13B</strong> (shared attention + 2 selected experts)</li>
              <li>Per-token compute: comparable to a 13B dense model</li>
            </ul>

            <p className="text-muted-foreground">
              The three-way comparison makes the decoupling concrete:
            </p>

            {/* Three-way comparison table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-muted">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Model</th>
                    <th className="text-right py-2 px-3 text-muted-foreground font-medium">Active Params</th>
                    <th className="text-right py-2 px-3 text-muted-foreground font-medium">Total Params</th>
                    <th className="text-right py-2 px-3 text-muted-foreground font-medium">Per-Token Compute</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">13B Dense</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">13B</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">13B</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">Baseline</td>
                  </tr>
                  <tr className="border-b border-muted/50 bg-emerald-500/5">
                    <td className="py-2 px-3 font-medium text-emerald-400">Mixtral 8&times;7B</td>
                    <td className="text-right py-2 px-3 text-emerald-400">~13B</td>
                    <td className="text-right py-2 px-3 text-emerald-400">~47B</td>
                    <td className="text-right py-2 px-3 text-emerald-400">&asymp; Baseline</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-muted-foreground">47B Dense</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">47B</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">47B</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">~3.5&times; Baseline</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              Mixtral gets <strong>47B worth of stored knowledge at 13B worth
              of per-token compute</strong>. It is competitive with models
              3-4&times; its active parameter count on benchmarks. The remaining
              34B parameters exist but do not contribute to any given
              token&rsquo;s forward pass.
            </p>

            <p className="text-muted-foreground">
              Connect this to Chinchilla: &ldquo;Scale both, not just
              one.&rdquo; MoE adds a third option&mdash;<strong>scale total
              parameters without scaling per-token compute</strong>. A new
              dimension in the scaling equation.
            </p>

            <ComparisonRow
              left={{
                title: 'Dense Scaling',
                color: 'amber',
                items: [
                  'More parameters = proportionally more compute',
                  'All parameters active for every token',
                  'Scale model size → scale training + inference cost',
                  'Chinchilla: scale model and data together',
                ],
              }}
              right={{
                title: 'MoE Scaling',
                color: 'emerald',
                items: [
                  'More total parameters, same per-token compute',
                  'Only k of N experts active per token',
                  'Scale knowledge → same inference cost per token',
                  'New dimension: total params ≠ active params',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The memory tradeoff: all 47B parameters must be loaded into
              memory even though only ~13B activate per token. MoE models use
              more memory than their active parameter count suggests. You get
              more knowledge at the same compute, but you need more memory to
              store the inactive experts. This tradeoff is why MoE models are
              often distributed across multiple GPUs.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Decoupling">
            Dense models: total params = active params. MoE models: total
            params &gt;&gt; active params. Per-token compute scales with
            active parameters, not total parameters. More knowledge, same
            speed, more memory.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Check 1: Predict-and-Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Predict the Numbers" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A model has 8 experts per layer, <strong>top-1</strong> routing
                (only 1 expert per token instead of 2). A user sends the prompt:
                &ldquo;Translate this French sentence to English: Le chat est
                sur le tapis.&rdquo;
              </p>

              <p><strong>Predict before revealing:</strong></p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>How many experts activate for each token?</li>
                <li>Do all tokens in the sentence activate the same expert?</li>
                <li>What happens to the other 7 experts for each token?</li>
                <li>How does the per-token FFN compute compare to an 8&times; dense model?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Make your predictions, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
                    <li>
                      <strong>1 expert per token.</strong> Top-1 routing means
                      exactly one expert is selected for each token.
                    </li>
                    <li>
                      <strong>Different tokens likely activate different
                      experts.</strong> &ldquo;Le&rdquo; might route differently
                      than &ldquo;chat&rdquo; or &ldquo;tapis.&rdquo; The
                      router assigns experts based on each token&rsquo;s hidden
                      state, and different tokens have different hidden states.
                    </li>
                    <li>
                      <strong>The other 7 experts are skipped
                      entirely.</strong> Their parameters exist but do not
                      contribute compute or gradients for this token.
                    </li>
                    <li>
                      <strong>~1/8th the FFN compute.</strong> Only 1 of 8
                      expert FFNs runs per token. The FFN portion of the
                      forward pass uses 1/8th the FLOPs of an equivalent dense
                      model with all 8 experts combined.
                    </li>
                  </ol>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          9. Explain Part 4: Expert Specialization
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Expert Specialization"
            subtitle="Emergent patterns, not designed categories"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What do experts actually specialize in? Consider a concrete
              sentence:
            </p>

            <PerTokenRoutingDiagram />

            {/* Misconception 2: Expert specialization */}
            <GradientCard title="Misconception: &ldquo;Every expert sees every token (experts specialize by topic)&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  The word &ldquo;specialist&rdquo; evokes a topic expert&mdash;a
                  person who knows about one subject. You might picture Expert 1
                  handling &ldquo;science tokens&rdquo; and Expert 2 handling
                  &ldquo;literature tokens.&rdquo; But expert specialization is
                  <strong> emergent and per-token, not designed and
                  per-topic</strong>.
                </p>
                <p className="text-muted-foreground">
                  Within a single sentence, different tokens route to different
                  experts. &ldquo;The&rdquo; and &ldquo;is&rdquo; might share
                  an expert because they are both function words, while
                  &ldquo;mitochondria&rdquo; routes to a different expert. The
                  router learns what groupings are useful from the training
                  data. Some experts specialize in syntax, others in domain
                  vocabulary, others in positional patterns. The boundaries
                  are often surprising and do not map to human categories.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              Research on Mixtral confirms this: experts develop preferences
              but not clean topic boundaries. Expert utilization varies by
              layer&mdash;early layers show more syntactic specialization
              (function words cluster together), later layers show more
              semantic patterns (domain-specific vocabulary). No expert is
              &ldquo;the biology expert&rdquo; in any clean sense.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Per-Token, Not Per-Topic">
            The router decides per-token, not per-sequence or per-topic. In the
            same sentence, &ldquo;The&rdquo; might go to Expert 2 while
            &ldquo;mitochondria&rdquo; goes to Expert 5. Different tokens,
            same sentence, different experts.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Elaborate: Load Balancing
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Load Balancing"
            subtitle="What happens when the router plays favorites"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now the problem: what happens if the router consistently prefers
              one expert? This is not hypothetical&mdash;it is the default
              outcome without intervention.
            </p>

            <RouterCollapseDiagram />

            <p className="text-muted-foreground">
              The collapse scenario: Expert 3 starts slightly better due to
              random initialization. The router sends more tokens to Expert 3.
              Expert 3 gets more gradient updates and improves further. The
              router sends even more tokens to it. Eventually, Expert 3 handles
              80%+ of tokens. The other 7 experts are undertrained and
              effectively dead weight. The model degenerates to approximately a
              dense model&mdash;all the extra parameters provide no benefit.
            </p>

            {/* Misconception 5: Load balance is just efficiency */}
            <GradientCard title="Misconception: &ldquo;Load imbalance is just an efficiency problem&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  You might see load imbalance as a GPU utilization issue&mdash;
                  some GPUs idle while others are overloaded. That is correct
                  but misses the deeper problem. If one expert receives 80% of
                  tokens, that expert gets 80% of the gradient updates. Other
                  experts get too few examples to learn effectively.
                </p>
                <p className="text-muted-foreground">
                  The imbalance creates a <strong>positive feedback
                  loop</strong>: the popular expert gets better (more gradient
                  updates), which makes the router send even more tokens to it.
                  This is a <strong>training collapse</strong> problem, not
                  just an efficiency problem. Without correction, only 1-2
                  experts survive and the rest become dead weights.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              The solution: an <strong>auxiliary loss</strong> that penalizes
              uneven expert utilization. For each expert, the loss measures two
              things: (1) what fraction of tokens in the batch were routed to
              it, and (2) how much probability the router assigned to it on
              average. If both are high for one expert, the penalty is large.
              If both are spread evenly, the penalty is small. This
              product-based penalty gently pushes the router toward balanced
              utilization without forcing uniform routing or overriding the
              router&rsquo;s learned preferences.
            </p>

            <p className="text-muted-foreground">
              There is a pattern here you have seen before: an unconstrained
              optimization process finds a degenerate solution. In RLHF, the
              policy exploits the reward model without a KL constraint. In MoE,
              gradient descent concentrates tokens on one expert without a
              balancing constraint. Different mechanisms, same pattern:
              unconstrained optimization of a local signal leads to collapse.
              The auxiliary loss constrains the optimization, keeping the system
              in a useful regime.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Training Collapse">
            Without load balancing, MoE training reliably degenerates. The
            auxiliary loss is not optional. It is as necessary as gradient
            clipping for stable training.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Check 2: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Apply the Concepts" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A company is training a large language model. They have a fixed
                compute budget (GPU-hours) and want to maximize model quality.
                They are deciding between:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>
                  <strong>Option A:</strong> Train a 13B dense model for 1T
                  tokens (compute-optimal per Chinchilla)
                </li>
                <li>
                  <strong>Option B:</strong> Train a 47B MoE model (8 experts,
                  top-2) for 1T tokens (same per-token compute as 13B dense)
                </li>
              </ul>

              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>Which option uses approximately the same total compute?</li>
                <li>Which has more stored knowledge?</li>
                <li>Which is likely to perform better on diverse benchmarks?</li>
                <li>What is the tradeoff?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reason through it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p className="text-muted-foreground">
                    <strong>Compute:</strong> Both use approximately the same
                    per-token compute (same FLOPs per token, same number of
                    tokens), so total training compute is roughly similar.
                  </p>
                  <p className="text-muted-foreground">
                    <strong>Knowledge:</strong> The MoE model stores more
                    knowledge (47B parameters vs 13B) but uses the same
                    compute to process each token.
                  </p>
                  <p className="text-muted-foreground">
                    <strong>Performance:</strong> The MoE model is likely to
                    perform better on diverse benchmarks because it can store
                    more specialized knowledge while keeping the same per-token
                    computation budget.
                  </p>
                  <p className="text-muted-foreground">
                    <strong>Tradeoff:</strong> The MoE model uses ~3.5&times;
                    more memory (all 47B parameters must be loaded even though
                    only 13B activate per token), and distributing experts
                    across multiple GPUs introduces communication overhead.
                    More knowledge, same speed, more memory.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          12. Practice: Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Explore MoE routing and mechanics"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Four exercises that explore the concepts from this lesson with
              toy models. Each builds on the previous conceptually but can be
              done independently.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Implement a Simple Router (Guided)" color="blue">
                <p className="text-sm">
                  Build a router for 4 experts on a toy d_model=64 hidden
                  state. Apply softmax, select top-2, and observe how different
                  random input vectors route to different experts. Visualize
                  router probabilities as a bar chart for 5 different inputs.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: MoE Forward Pass (Supported)" color="blue">
                <p className="text-sm">
                  Build a complete MoE layer: 4 small expert FFNs (d_model=64,
                  d_ff=256) + router. Run a forward pass for a batch of 8
                  tokens. Compare output shapes to a single dense FFN. Count
                  active parameters per token vs total parameters.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Visualize Expert Routing on Real Text (Supported)" color="blue">
                <p className="text-sm">
                  Tokenize 3-5 sentences and visualize which expert each token
                  is routed to. Color-code tokens by expert assignment. Look
                  for patterns: do function words cluster? Do domain words
                  cluster? Sentence: &ldquo;The mitochondria is the powerhouse
                  of the cell.&rdquo;
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: Router Collapse Experiment (Independent)" color="blue">
                <p className="text-sm">
                  Train a toy MoE model with and without an auxiliary
                  load-balancing loss. Track expert utilization over training
                  steps. Plot the distribution of tokens across experts at
                  step 0, 100, 500, 1000. Observe collapse without the loss
                  and stability with it.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                  Each exercise explores a different aspect of MoE mechanics
                  with immediate feedback.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-3-1-mixture-of-experts.ipynb"
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
            Exercise 1 demystifies the router. Exercise 2 builds a complete
            MoE layer. Exercise 3 shows real routing patterns. Exercise 4
            demonstrates why load balancing matters.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'MoE replaces the monolithic FFN with multiple specialized experts and a learned router.',
                description:
                  '"Attention reads, the right experts write." The attention sub-layer is unchanged. Only the FFN sub-layer changes: N expert FFNs replace the single FFN, and a router selects which k to activate per token.',
              },
              {
                headline:
                  'The router is a single linear layer with softmax\u2014the same dot-product-softmax pattern as attention.',
                description:
                  'Router logits are dot products between the token\'s hidden state and learned expert embeddings. One matrix multiplication + softmax + top-k. Simpler than a single attention head.',
              },
              {
                headline:
                  'MoE decouples total parameters from per-token compute.',
                description:
                  'Mixtral 8\u00d77B: ~47B total parameters, ~13B active per token. More stored knowledge, same inference cost per token. The model is "bigger without being slower."',
              },
              {
                headline:
                  'Expert specialization is emergent and per-token, not designed and per-topic.',
                description:
                  'Different tokens in the same sentence activate different experts. Function words cluster together. Domain vocabulary clusters together. The boundaries are learned, not imposed.',
              },
              {
                headline:
                  'Load balancing prevents router collapse.',
                description:
                  'Without an auxiliary loss, one expert dominates and others atrophy\u2014a positive feedback loop. The auxiliary loss gently pushes toward balanced utilization, preventing training collapse.',
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
              The dense transformer activates all knowledge for every token.
              MoE activates only the relevant knowledge. More total knowledge,
              same compute per token. The library got bigger, but you still only
              talk to two librarians.
            </p>
          </GradientCard>
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
                title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer',
                authors: 'Shazeer et al., 2017',
                url: 'https://arxiv.org/abs/1701.06538',
                note: 'The foundational MoE paper for deep learning. Introduces the sparsely-gated MoE layer with top-k routing and load balancing. Section 2 covers the architecture.',
              },
              {
                title: 'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity',
                authors: 'Fedus, Zoph & Shazeer, 2022',
                url: 'https://arxiv.org/abs/2101.03961',
                note: 'Simplifies MoE to top-1 routing (one expert per token). Shows that simpler routing works at scale. Section 2 for the architecture, Section 3 for training stability.',
              },
              {
                title: 'Mixtral of Experts',
                authors: 'Jiang et al., 2024',
                url: 'https://arxiv.org/abs/2401.04088',
                note: 'The Mixtral 8x7B architecture discussed in this lesson. 8 experts, top-2 routing, ~47B total parameters. Section 2 for the architecture, Section 3 for benchmark results.',
              },
              {
                title: 'DeepSeek-V3 Technical Report',
                authors: 'DeepSeek-AI, 2024',
                url: 'https://arxiv.org/abs/2412.19437',
                note: 'A recent frontier MoE model. Uses auxiliary-loss-free load balancing and multi-head latent attention. Sections 2.1-2.2 cover the MoE architecture.',
              },
              {
                title: 'Transformer Feed-Forward Layers Are Key-Value Memories',
                authors: 'Geva et al., 2021',
                url: 'https://arxiv.org/abs/2012.14913',
                note: 'The paper showing FFN neurons store specific knowledge. Referenced in the recap section to motivate why conditional FFN activation makes sense.',
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
            title="Up next: Long Context & Efficient Attention"
            description="MoE solved the parameter-compute problem: more knowledge without more per-token cost. But attention is quadratic in sequence length. As context windows grow to 100K+ tokens, attention itself becomes the bottleneck. Next: how positional encodings enable long context and how attention can be made more efficient."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
