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
  ModuleCompleteBlock,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'

/**
 * Evaluating LLMs
 *
 * Fourth and final lesson in Module 5.1 (Advanced Alignment).
 * CONSOLIDATE lesson — module capstone.
 * Inline SVG diagram (Evaluation Stack / Benchmark Anatomy).
 *
 * Cognitive load: CONSOLIDATE — integrates concepts from all three prior
 * lessons rather than introducing radically new mechanisms. Two genuinely
 * new concepts: contamination as structural, Goodhart's law for evaluation.
 *
 * Core concepts at DEVELOPED:
 * - Benchmark limitations (proxy gap, what benchmarks actually measure)
 * - Goodhart's law for evaluation (extension of reward hacking)
 *
 * Concepts at INTRODUCED:
 * - Contamination (structural property of internet-scale training)
 * - Human evaluation challenges (cost, consistency, scale)
 * - LLM-as-judge (scaling evaluation with AI)
 * - Evaluation as harder than training
 *
 * EXPLICITLY NOT COVERED:
 * - Specific current benchmark scores or leaderboard positions
 * - Implementing evaluation pipelines in code
 * - Designing new benchmarks or evaluation frameworks
 * - Statistical methodology for evaluation
 * - Evaluation of non-LLM models
 * - Constitutional AI, preference optimization, or red teaming details (Lessons 1-3)
 *
 * Previous: Red Teaming & Adversarial Evaluation (Module 5.1, Lesson 3)
 * Next: Module 5.2 (Reasoning & In-Context Learning)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Evaluation Stack / Benchmark Anatomy Diagram
// Shows the layers between "actual model capability" and
// "the number on the leaderboard"
// ---------------------------------------------------------------------------

function EvaluationStackDiagram() {
  const svgW = 600
  const svgH = 420

  // Layer definitions — from bottom (capability) to top (leaderboard number)
  const layers = [
    {
      label: 'Actual Model Capability',
      subtitle: 'What the model can really do',
      color: '#6366f1', // indigo
      note: 'Multidimensional, context-dependent',
    },
    {
      label: 'Task Design',
      subtitle: 'What questions are asked, in what format',
      color: '#8b5cf6', // violet
      note: 'Multiple-choice ≠ generation',
    },
    {
      label: 'Prompt Formatting',
      subtitle: 'How questions are presented to the model',
      color: '#a78bfa', // lighter violet
      note: 'Template choice changes scores by 5-15%',
    },
    {
      label: 'Scoring Criteria',
      subtitle: 'How answers are judged correct or incorrect',
      color: '#f59e0b', // amber
      note: 'Exact match vs semantic match vs rubric',
    },
    {
      label: 'Aggregation',
      subtitle: 'How per-question scores become a single number',
      color: '#f97316', // orange
      note: 'Mean hides variance across categories',
    },
    {
      label: 'Leaderboard Position',
      subtitle: 'The number you actually see',
      color: '#ef4444', // red
      note: 'This is what people compare',
    },
  ]

  const layerH = 48
  const layerW = 420
  const startX = (svgW - layerW) / 2
  const startY = svgH - 50
  const gap = 8

  // Contamination annotation
  const contaminationY = startY - 2 * (layerH + gap) - layerH / 2

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
          The Evaluation Stack
        </text>
        <text
          x={svgW / 2}
          y={36}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="9"
          fontStyle="italic"
        >
          Every layer between capability and the number on the leaderboard
        </text>

        {/* Arrow markers */}
        <defs>
          <marker
            id="evalArrow"
            markerWidth="8"
            markerHeight="6"
            refX="8"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 8 3, 0 6" fill="#64748b" />
          </marker>
        </defs>

        {/* Layers — bottom to top */}
        {layers.map((layer, i) => {
          const y = startY - i * (layerH + gap)

          return (
            <g key={i}>
              {/* Layer rectangle */}
              <rect
                x={startX}
                y={y - layerH}
                width={layerW}
                height={layerH}
                rx={6}
                fill="none"
                stroke={layer.color}
                strokeWidth={1.5}
                opacity={0.8}
              />

              {/* Color indicator bar on left */}
              <rect
                x={startX}
                y={y - layerH}
                width={5}
                height={layerH}
                rx={2}
                fill={layer.color}
                opacity={0.9}
              />

              {/* Layer label */}
              <text
                x={startX + 18}
                y={y - layerH + 19}
                fill="#e2e8f0"
                fontSize="11"
                fontWeight="600"
              >
                {layer.label}
              </text>

              {/* Subtitle */}
              <text
                x={startX + 18}
                y={y - layerH + 33}
                fill="#94a3b8"
                fontSize="9"
              >
                {layer.subtitle}
              </text>

              {/* Right-side note */}
              <text
                x={startX + layerW - 10}
                y={y - layerH + 26}
                textAnchor="end"
                fill={layer.color}
                fontSize="8"
                fontStyle="italic"
                opacity={0.8}
              >
                {layer.note}
              </text>

              {/* Upward arrow between layers (skip the topmost) */}
              {i < layers.length - 1 && (
                <line
                  x1={svgW / 2}
                  y1={y - layerH - 2}
                  x2={svgW / 2}
                  y2={y - layerH - gap + 2}
                  stroke="#475569"
                  strokeWidth={1.5}
                  markerEnd="url(#evalArrow)"
                  transform={`rotate(180, ${svgW / 2}, ${y - layerH - gap / 2})`}
                />
              )}
            </g>
          )
        })}

        {/* Contamination annotation — side bracket */}
        <line
          x1={startX + layerW + 20}
          y1={startY - layerH + 5}
          x2={startX + layerW + 20}
          y2={contaminationY - 10}
          stroke="#ef4444"
          strokeWidth={1}
          strokeDasharray="4 3"
          opacity={0.6}
        />
        <text
          x={startX + layerW + 28}
          y={contaminationY + 20}
          fill="#ef4444"
          fontSize="8"
          fontStyle="italic"
          opacity={0.8}
        >
          Contamination &amp;
        </text>
        <text
          x={startX + layerW + 28}
          y={contaminationY + 31}
          fill="#ef4444"
          fontSize="8"
          fontStyle="italic"
          opacity={0.8}
        >
          optimization pressure
        </text>
        <text
          x={startX + layerW + 28}
          y={contaminationY + 42}
          fill="#ef4444"
          fontSize="8"
          fontStyle="italic"
          opacity={0.8}
        >
          affect every layer
        </text>

        {/* Bottom annotation */}
        <text
          x={svgW / 2}
          y={svgH - 8}
          textAnchor="middle"
          fill="#64748b"
          fontSize="8"
          fontStyle="italic"
        >
          The gap between bottom and top is the proxy gap—what benchmarks actually measure vs what capability actually is
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function EvaluatingLlmsLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Evaluating LLMs"
            description="You built alignment (Lessons 1-2). You broke it (Lesson 3). Now the hardest question: how do you measure whether any of it worked? Benchmarks are not what they appear to be."
            category="Alignment"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints (Section 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            This is the &ldquo;Measure&rdquo; capstone in the
            Build-Break-Measure arc. You will learn to critically assess LLM
            evaluation: what benchmarks actually measure vs what they claim to
            measure, why contamination undermines evaluation structurally, why
            Goodhart&rsquo;s law means &ldquo;better benchmarks&rdquo; is not a
            solution, and why evaluation may be fundamentally harder than
            training. The goal is not to memorize which benchmarks exist but to
            develop the critical thinking to read any evaluation result and
            understand what it actually tells you.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'What benchmarks actually measure vs what they claim (the proxy gap)',
              'Contamination as a structural property of internet-scale training',
              "Goodhart's law applied to evaluation metrics",
              'Human evaluation challenges and LLM-as-judge',
              'Why evaluation may be harder than training',
              'NOT: specific current benchmark scores or leaderboard positions',
              'NOT: implementing evaluation pipelines in code',
              'NOT: designing new benchmarks or evaluation frameworks',
              'NOT: statistical methodology for evaluation',
              'NOT: constitutional AI, preference optimization, or red teaming details (Lessons 1-3)',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Build, Break, Measure">
            Lessons 1-2 built alignment. Lesson 3 broke it. This lesson
            measures it&mdash;and measuring may be the hardest part.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap (Section 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where We Left Off"
            subtitle="Three lessons of alignment&mdash;now the question they left unanswered"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Across the first three lessons of this module, you built a
              comprehensive picture of alignment:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Constitutional AI:</strong> Alignment techniques are
                diverse&mdash;constitutional principles, RLAIF, DPO
                variations&mdash;and each makes different tradeoffs
              </li>
              <li>
                <strong>The Alignment Techniques Landscape:</strong> Methods
                are points in a design space, not steps on a ladder. Constraints
                drive choice, not novelty
              </li>
              <li>
                <strong>Red Teaming:</strong> A model that passes standard
                safety tests can still fail on demographic bias, sycophancy, and
                indirect requests. Benchmarks sample a few points on the
                alignment surface&mdash;the gaps between those points are where
                failures live
              </li>
            </ul>
            <p className="text-muted-foreground">
              And from RLHF &amp; Alignment in Series 4, you know that the
              reward model is a proxy for human preferences&mdash;the
              &ldquo;editor with blind spots.&rdquo; The model learned to game
              that proxy with verbose, confident answers. That was reward
              hacking.
            </p>
            <p className="text-muted-foreground">
              Now the question that ties it all together: <strong>you built
              alignment, you tested it adversarially&mdash;how do you measure
              whether it actually worked? And how much should you trust that
              measurement?</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Missing Piece">
            Red teaming finds specific failures. But &ldquo;how aligned is
            this model overall?&rdquo; requires evaluation&mdash;and evaluation
            has its own deep problems that mirror every challenge from the
            first three lessons.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: Misconception Reveal (Section 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Which Model Is Better?"
            subtitle="Two models, three benchmarks, one obvious answer&mdash;that turns out to be wrong"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider two models with their benchmark scores:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Model A" color="blue">
                <ul className="space-y-1 text-sm">
                  <li>&bull; MMLU: <strong>92.4%</strong></li>
                  <li>&bull; HumanEval: <strong>87.2%</strong></li>
                  <li>&bull; TruthfulQA: <strong>71.8%</strong></li>
                </ul>
              </GradientCard>
              <GradientCard title="Model B" color="cyan">
                <ul className="space-y-1 text-sm">
                  <li>&bull; MMLU: <strong>88.1%</strong></li>
                  <li>&bull; HumanEval: <strong>82.5%</strong></li>
                  <li>&bull; TruthfulQA: <strong>68.4%</strong></li>
                </ul>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Which model is better? Model A wins on every benchmark. The
              obvious answer is Model A.
            </p>

            <p className="text-muted-foreground">
              Now three things you did not see in the benchmark scores:
            </p>

            <div className="space-y-3">
              <GradientCard title="What the Numbers Hide" color="rose">
                <ul className="space-y-2 text-sm">
                  <li>
                    &bull; Model B gives concise, actionable answers. Model A
                    hedges excessively&mdash;every response padded with
                    qualifications and caveats that obscure the actual answer.
                  </li>
                  <li>
                    &bull; Model A&rsquo;s MMLU score is suspiciously high on
                    publicly available question subsets (96%) vs held-out
                    questions (84%). Model B scores consistently across both.
                  </li>
                  <li>
                    &bull; In a blind user evaluation, users prefer Model
                    B&rsquo;s responses <strong>63%</strong> of the time.
                  </li>
                </ul>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Model A has higher scores on every benchmark. Model B is the model
              users actually prefer. <strong>What went wrong?</strong> The
              benchmark scores are not measurements of model quality. They are
              products of design choices, potential contamination, and
              optimization pressure. This lesson is about understanding the gap
              between the number and the reality.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Higher Score ≠ Better Model">
            Leaderboards rank by single numbers. But &ldquo;better&rdquo; is
            multidimensional, context-dependent, and often defined differently
            by the benchmark than by the user.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain: The Evaluation Stack (Section 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Evaluation Stack"
            subtitle="What a benchmark score is actually made of"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A benchmark score is not a direct measurement of capability. It
              is the result of a pipeline, and every stage of that pipeline
              introduces design choices that shape the final number. Think of
              standardized testing in education: the SAT does not measure
              &ldquo;intelligence.&rdquo; It measures SAT-taking
              ability&mdash;which correlates with intelligence but is not the
              same thing. LLM benchmarks have exactly the same problem.
            </p>

            <p className="text-muted-foreground">
              Here are the major families of benchmarks, not as a catalog to
              memorize, but as categories with different measurement strategies:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Knowledge/reasoning</strong> (MMLU, ARC, HellaSwag):
                Multiple-choice format. Measures recognition (selecting the
                right answer from options), not generation (producing an answer
                from scratch). High performance may reflect test-taking skill
                more than understanding.
              </li>
              <li>
                <strong>Code</strong> (HumanEval, MBPP): Functional correctness
                (does the code pass tests?). The closest to objective
                evaluation, but narrow&mdash;measures ability to write isolated
                functions, not real-world software engineering.
              </li>
              <li>
                <strong>Safety/alignment</strong> (TruthfulQA, ToxiGen, BBQ):
                Measures specific safety properties. But safety is
                multidimensional, and each benchmark covers a narrow
                slice&mdash;as you saw in Red Teaming &amp; Adversarial
                Evaluation, the model that passed safety benchmarks failed on
                demographic bias, sycophancy, and indirect requests.
              </li>
              <li>
                <strong>Open-ended generation</strong> (AlpacaEval, MT-Bench,
                Chatbot Arena): Measures response quality via human or LLM
                judgments. Most ecologically valid but most subjective and
                expensive.
              </li>
            </ul>

            <p className="text-muted-foreground">
              The key insight is not the specific benchmarks but the
              pattern: <strong>every benchmark makes design choices</strong>{' '}
              (format, scoring, coverage) <strong>that determine what it
              actually measures</strong>, and those design choices may differ
              significantly from what the benchmark name implies.
            </p>

            <GradientCard title="When &ldquo;Passing&rdquo; Means Nothing" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Remember the model from Red Teaming &amp; Adversarial
                  Evaluation that passed all three standard safety benchmarks?
                  Those benchmarks sampled a few points on the alignment
                  surface. The model performed well at those specific points.
                  But the gaps between the sampled points&mdash;demographic
                  bias, sycophancy, indirect requests&mdash;were where the
                  failures lived.
                </p>
                <p>
                  The benchmark score was real. The safety it implied was not.
                  That is the proxy gap in action: <strong>the benchmarks
                  measured the wrong thing.</strong> They measured safety on a
                  narrow sample, not safety in general. &ldquo;This model passed
                  safety benchmarks&rdquo; is not a safety guarantee&mdash;it is
                  an evaluation failure.
                </p>
              </div>
            </GradientCard>

            <EvaluationStackDiagram />
            <p className="text-muted-foreground text-sm text-center mt-2">
              Every layer adds indirection. The number at the top is not a
              direct measurement of the capability at the bottom.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Benchmarks Are Standardized Tests">
            The SAT analogy maps precisely: contamination = test prep,
            Goodhart&rsquo;s law = teaching to the test, proxy gap = SAT
            score vs actual readiness. Every problem with standardized testing
            in education has a direct parallel in LLM benchmarks.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1: What Does This Benchmark Actually Measure? (Section 5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: What Does This Benchmark Actually Measure?" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                MMLU (&ldquo;Massive Multitask Language Understanding&rdquo;)
                tests knowledge across 57 subjects using multiple-choice
                questions. The name suggests it measures
                &ldquo;understanding.&rdquo;
              </p>
              <p>
                <strong>What capability does this benchmark actually test?
                What does the name suggest? What is the gap?</strong>
              </p>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p>
                    MMLU tests <strong>multiple-choice selection</strong>
                    &mdash;recognizing the correct answer among distractors.
                    This is fundamentally different from generating an answer,
                    explaining a concept, reasoning through a novel problem, or
                    applying knowledge in context.
                  </p>
                  <p>
                    A student who aces multiple-choice tests is not the same
                    as a student who deeply understands the material. Same for
                    models. The gap: &ldquo;understanding&rdquo; is in the name
                    but not in the measurement. MMLU measures recognition
                    performance on a fixed question set, not understanding.
                  </p>
                  <p className="text-muted-foreground">
                    This is the proxy gap in action&mdash;the benchmark name
                    promises one thing, the benchmark mechanism measures
                    something narrower.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain: Contamination (Section 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Contamination"
            subtitle="When the test is in the training data"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              LLMs are trained on internet-scale data. Benchmark questions,
              answers, and discussions exist on the internet. Models may have
              seen the exact questions&mdash;or close paraphrases&mdash;during
              training. This is not &ldquo;cheating&rdquo; in the intentional
              sense. It is a <strong>structural consequence</strong> of training
              on web crawls.
            </p>

            <div className="space-y-3">
              <GradientCard title="Direct Contamination" color="amber">
                <p className="text-sm">
                  The exact benchmark question and answer appear in training
                  data. The model memorizes the answer. The score is real (the
                  model got it right), but the capability is not (the model
                  remembered, not reasoned).
                </p>
              </GradientCard>

              <GradientCard title="Indirect Contamination" color="orange">
                <p className="text-sm">
                  Discussions, explanations, or paraphrases of benchmark
                  questions appear in training data. The model has been
                  &ldquo;tutored&rdquo; on the material without seeing the exact
                  question. Like a student who read the study guide that happens
                  to cover every test question.
                </p>
              </GradientCard>

              <GradientCard title="Benchmark Saturation" color="rose">
                <p className="text-sm">
                  A benchmark has been used so widely that labs inadvertently
                  optimize for it&mdash;choosing architectures, data mixes, and
                  training procedures that perform well on the benchmark, even
                  without directly training on its questions.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              The structural argument: <strong>contamination is not a bug to
              fix. It is a property of the training paradigm.</strong> Any
              benchmark that is published becomes part of the internet, which
              becomes part of training data. Decontamination is a temporary
              state&mdash;it expires when the benchmark goes public. If you
              have a software engineering background, this is like publishing
              your test suite and being surprised that the code passes all
              tests.
            </p>

            <GradientCard title="The Forensic Evidence" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  How contamination manifests: a model scores <strong>95%</strong>{' '}
                  on questions that appear in Common Crawl and{' '}
                  <strong>72%</strong> on questions that do not. Both sections
                  are the same difficulty level. The 23-point gap is the
                  contamination signal.
                </p>
                <p>
                  The score is real&mdash;the model answered correctly. But
                  the capability is not&mdash;the model memorized answers
                  rather than reasoning. Uneven performance across supposedly
                  equivalent sections is the forensic evidence.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Just &ldquo;Clean It Up&rdquo;">
            A software engineer&rsquo;s instinct is to find the data leak and
            fix it. But contamination is structural, not accidental. You cannot
            fully audit internet-scale training data, and new benchmarks become
            contaminated the moment they are published online.
          </WarningBlock>
          <InsightBlock title="The Alignment Surface Returns">
            Remember the alignment surface from Red Teaming? Benchmarks
            sample a few points on the capability surface. Contamination means
            those specific points may be inflated&mdash;the surface looks
            higher where the benchmark measured, but the true surface may be
            lower between samples. The fix is not &ldquo;more
            benchmarks&rdquo;&mdash;more samples on a corrupted surface do not
            improve coverage. Worse, Goodhart&rsquo;s law means the act of
            measuring changes the surface itself, because each new benchmark
            becomes an optimization target.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain: Goodhart's Law (Section 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Goodhart's Law for Evaluation"
            subtitle="When a measure becomes a target, it ceases to be a good measure"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember reward hacking from RLHF &amp; Alignment? The model
              learned to game the reward model with verbose, confident
              answers. The reward model was a proxy for human preferences, and
              optimizing against the proxy diverged from the actual
              goal. <strong>That was Goodhart&rsquo;s law inside
              training.</strong> Now apply it to evaluation.
            </p>

            <p className="text-muted-foreground">
              <strong>Within a lab:</strong> A team optimizes for MMLU. They
              select architectures, data mixes, and training schedules that
              improve MMLU scores. MMLU performance goes up. But does the model
              actually understand more, or did they optimize the test-taking
              pipeline? The answer: probably some of both, but you cannot tell
              from the MMLU score alone.
            </p>

            <p className="text-muted-foreground">
              <strong>Across the ecosystem:</strong> Leaderboards drive
              attention, funding, and talent. Labs optimize for leaderboard
              benchmarks. Benchmarks become optimization targets. The
              correlation between benchmark score and actual capability degrades
              as optimization pressure increases.
            </p>

            <p className="text-muted-foreground">
              The &ldquo;of course&rdquo; moment: <strong>of course this
              happens.</strong> It is the same mechanism as reward hacking. The
              reward model was a proxy for human preferences, and the model
              learned to exploit it. Benchmarks are a proxy for capability, and
              the ecosystem learns to exploit them. The proxy diverges from
              reality under optimization pressure. This is not a new
              concept&mdash;it is a familiar one in a new costume.
            </p>

            <GradientCard title="&ldquo;We Just Need Better Benchmarks&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  You might think: current benchmarks are flawed, but if we
                  designed <em>better</em> ones, we would have reliable
                  evaluation. This is the optimistic engineering response:
                  identify the problem, build the fix. But Goodhart&rsquo;s law
                  applies to <em>all</em> benchmarks, including ones that do not
                  exist yet.
                </p>
                <p>
                  Imagine a perfectly designed benchmark that captures exactly
                  what you care about today. You publish it. Labs begin
                  optimizing for it. Within a year, models exploit patterns in
                  the benchmark format&mdash;learning that multiple-choice
                  distractors follow certain linguistic patterns, that the rubric
                  rewards certain phrasings&mdash;rather than developing the
                  underlying capability. The benchmark score goes up. The
                  capability it was designed to measure does not improve
                  proportionally. This is not a flaw in the benchmark
                  design&mdash;it is Goodhart&rsquo;s law. The problem is not
                  bad benchmarks. The problem is the relationship between
                  measurement and optimization.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Interpretation, Not Dismissal" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  Goodhart&rsquo;s law does not mean benchmarks are useless. It
                  means benchmark scores require interpretation. A 5-point
                  improvement on a well-known benchmark by a lab that has been
                  optimizing for that benchmark is less impressive than a 2-point
                  improvement on a new benchmark no one has optimized for.
                </p>
                <p>
                  The question is never &ldquo;what is the score?&rdquo; It
                  is always &ldquo;what does the score mean, given the
                  optimization pressure on this benchmark?&rdquo;
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Mechanism, Larger Scale">
            Reward hacking: the model games the proxy. Goodhart&rsquo;s law:
            the ecosystem games the proxy. Same mechanism&mdash;optimizing
            against a proxy diverges from reality&mdash;operating at different
            scales.
          </InsightBlock>
          <TipBlock title="Blind Spots Move Again">
            When a lab optimizes for MMLU, performance improves on MMLU. But
            unmeasured dimensions may not improve&mdash;or may degrade. The
            blind spots move to whatever the benchmark does not measure.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Explain: Human Evaluation (Section 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Human Evaluation"
            subtitle="The &ldquo;gold standard&rdquo; that disagrees with itself"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              When benchmarks disagree, the instinct is to ask a human. Surely
              a human can tell if a model&rsquo;s response is good. But human
              evaluation has its own deep problems&mdash;and you have already
              seen most of them.
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Inter-annotator disagreement</strong> (callback to
                Constitutional AI): Remember the lock-picking annotator
                disagreement? If annotators cannot agree on whether a response
                is harmful, they will also disagree on whether a response is
                &ldquo;good.&rdquo; Cohen&rsquo;s kappa (a measure of
                agreement beyond chance, where 0 is random and 1 is perfect)
                for open-ended quality judgments is often
                0.3-0.5&mdash;only &ldquo;fair to moderate&rdquo; agreement,
                far from the reliability you would want for a gold standard.
              </li>
              <li>
                <strong>Cost:</strong> Meaningful evaluation requires thousands
                of judgments across diverse prompts. At $15-30 per hour for
                skilled annotators, evaluating a single model update can cost
                tens of thousands of dollars.
              </li>
              <li>
                <strong>Scale:</strong> Humans cannot evaluate enough outputs to
                cover the capability surface. This is the same bottleneck that
                motivated RLAIF.
              </li>
              <li>
                <strong>Bias:</strong> Annotators systematically prefer verbose
                responses (length bias), confident responses (authority bias),
                and responses that agree with their priors (confirmation bias).
                These biases are invisible in the evaluation results.
              </li>
            </ul>

            <p className="text-muted-foreground">
              Chatbot Arena offers a partial solution: pairwise blind
              comparisons from real users, Elo-style ranking. It works better
              because relative comparisons are more reliable than absolute
              ratings&mdash;the same insight behind preference pairs in RLHF.
              But it still has problems: selection bias (users choose which
              queries to submit), demographic bias (who participates), and cost
              of participation.
            </p>

            <p className="text-muted-foreground">
              If the &ldquo;ground truth&rdquo; method disagrees with itself,
              it is not actually ground truth. Human evaluation is valuable but
              imperfect&mdash;another proxy, not a final answer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The Gold Standard Tarnishes">
            Two expert annotators evaluating the same coding response: one
            rates it 4/5 (correct approach, minor style issues), the other
            rates it 2/5 (would not pass code review). Both are
            &ldquo;right&rdquo; by their own standards. That is kappa of
            0.3-0.5&mdash;far from the reliability a gold standard demands.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Explain: LLM-as-Judge (Section 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="LLM-as-Judge"
            subtitle="Scaling evaluation with AI&mdash;and inheriting new blind spots"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The scaling argument: the same insight that drove RLAIF (use AI
              instead of humans for preference labels) and automated red
              teaming (use AI instead of humans for adversarial probing)
              applies to evaluation. Use an LLM to judge the quality of another
              LLM&rsquo;s outputs.
            </p>

            <p className="text-muted-foreground">
              <strong>How it works:</strong> Present the judge model with a
              prompt, one or two model responses, and a rubric. The judge
              provides a rating or comparison. Scale this to thousands of
              evaluations. <strong>Why it is attractive:</strong> orders of
              magnitude cheaper than humans, evaluate thousands of outputs in
              minutes, and the same judge applies the same criteria every
              time&mdash;no inter-annotator disagreement.
            </p>

            <p className="text-muted-foreground">
              But the blind spots return. LLM judges have systematic biases:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Verbosity Bias" color="amber">
                <p className="text-sm">
                  Rate longer responses higher, even when brevity is better.
                  The same failure mode as reward hacking.
                </p>
              </GradientCard>
              <GradientCard title="Confidence Bias" color="amber">
                <p className="text-sm">
                  Rate confident-sounding responses higher, even when hedging
                  is more honest.
                </p>
              </GradientCard>
              <GradientCard title="Self-Preference Bias" color="orange">
                <p className="text-sm">
                  Rate responses from similar models higher&mdash;GPT-4
                  judging GPT-4 outputs more favorably.
                </p>
              </GradientCard>
              <GradientCard title="Format Sensitivity" color="orange">
                <p className="text-sm">
                  Rate well-formatted responses (markdown, bullet points)
                  higher regardless of content quality.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              The pattern: <strong>&ldquo;the evaluator&rsquo;s limitations
              become the evaluation&rsquo;s limitations.&rdquo;</strong> This
              is the third time you have seen this in this module: human
              annotators have biases (Constitutional AI), red team models have
              blind spots (Red Teaming), and now LLM judges have biases too.
              The consistent insight: no single evaluation source is reliable.
              The best evaluation combines multiple methods&mdash;the same
              defense-in-depth principle from Red Teaming, applied to evaluation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pattern, Third Time">
            Human annotators have biases. Red team models have blind spots.
            LLM judges have biases. The evaluator&rsquo;s limitations always
            become the evaluation&rsquo;s limitations. No single evaluation
            source is sufficient.
          </InsightBlock>
          <TipBlock title="Defense-in-Depth for Evaluation">
            Just as no single defense layer covers the alignment surface, no
            single evaluation method captures model quality. Combine
            benchmarks, human evaluation, LLM judges, and red teaming.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Check 2: Evaluate the Evaluation (Section 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Evaluate the Evaluation" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A lab claims &ldquo;our model achieves state-of-the-art
                performance&rdquo; based on three benchmarks: MMLU (93.1%),
                HumanEval (89.4%), and AlpacaEval (95.2%, judged by GPT-4).
              </p>

              <p>
                <strong>List three questions you would ask before trusting
                this claim.</strong>
              </p>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about what could undermine each claim, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p>
                    Questions you should be asking:
                  </p>
                  <ul className="space-y-2">
                    <li>
                      &bull; <strong>Contamination:</strong> Was the training
                      data decontaminated for these benchmarks? Are the MMLU
                      scores uniform across public and held-out questions?
                    </li>
                    <li>
                      &bull; <strong>Selection bias:</strong> Were these three
                      benchmarks chosen because the model performs well on them?
                      What benchmarks were NOT reported?
                    </li>
                    <li>
                      &bull; <strong>Judge bias:</strong> AlpacaEval is judged
                      by GPT-4. Does the model produce GPT-4-like outputs
                      (verbose, well-formatted)? Self-preference bias could
                      inflate the score.
                    </li>
                    <li>
                      &bull; <strong>Proxy gap:</strong> MMLU is
                      multiple-choice (recognition), HumanEval is isolated
                      functions (not real software engineering), AlpacaEval
                      is open-ended (but judged by a model with known biases).
                      What dimensions of quality are not measured?
                    </li>
                    <li>
                      &bull; <strong>Comparability:</strong> Were other models
                      evaluated under identical conditions (same prompt
                      templates, same scoring)?
                    </li>
                  </ul>
                  <p className="text-muted-foreground">
                    You do not need to be cynical about every claim. You need
                    to be informed about what the numbers can and cannot tell
                    you.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          12. Elaborate: Why Evaluation May Be Harder Than Training (Section 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Evaluation May Be Harder Than Training"
            subtitle="The capstone insight&mdash;defining &ldquo;good&rdquo; is harder than optimizing for it"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Training has a well-defined objective: minimize loss. The loss
              function is imperfect (it is a proxy), but the optimization
              procedure is clear. Evaluation asks: &ldquo;did the training
              work?&rdquo;&mdash;which requires answering &ldquo;work for
              what?&rdquo; This is a fundamentally harder question.
            </p>

            <div className="space-y-3">
              <GradientCard title="Multidimensionality" color="violet">
                <p className="text-sm">
                  Models must be helpful, harmless, honest, concise, creative,
                  accurate, and more. These dimensions
                  conflict&mdash;more cautious means less helpful, more creative
                  means less accurate. No single number captures all dimensions.
                </p>
              </GradientCard>

              <GradientCard title="Context-Dependence" color="purple">
                <p className="text-sm">
                  &ldquo;Good&rdquo; depends on who is asking, what they need,
                  and when. A medical professional wants different things from a
                  model than a student or a creative writer. The same response
                  can be excellent for one user and dangerous for another.
                </p>
              </GradientCard>

              <GradientCard title="Moving Targets" color="blue">
                <p className="text-sm">
                  What &ldquo;good&rdquo; means changes as models improve.
                  Benchmarks that were challenging three years ago are now
                  saturated. Evaluation must evolve with capability, but
                  capability moves faster than evaluation.
                </p>
              </GradientCard>

              <GradientCard title="Recursive Proxy Problem" color="rose">
                <p className="text-sm">
                  You evaluate with benchmarks (proxies). When you evaluate the
                  benchmarks themselves, you use meta-benchmarks (proxies for
                  the proxies). There is no ground truth at the bottom&mdash;only
                  increasingly indirect measurements.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Connect this to the module arc: in Constitutional AI, the
              challenge was getting enough alignment data. Constitutional AI
              shifted the challenge to designing the right principles. In Red
              Teaming, the challenge shifted to finding failures adversarially.
              Now the challenge shifts again: measuring whether any of it
              worked. <strong>Each shift does not make the problem
              easier&mdash;it reveals deeper layers of difficulty. The challenge
              shifts, not disappears.</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Challenge Shifts Again">
            From &ldquo;build alignment&rdquo; to &ldquo;test
            alignment&rdquo; to &ldquo;measure alignment.&rdquo; Each step
            reveals deeper difficulty. Measuring whether something worked
            requires defining what &ldquo;worked&rdquo; means&mdash;and that
            may be the hardest question of all.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Practice: Notebook Exercises (Section 12)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Benchmark autopsy, bias detection, and evaluation design"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Three exercises that build from critical reading to empirical
              testing to synthesis. Exercise 1 builds the analytical
              framework, Exercise 2 demonstrates biases empirically, Exercise 3
              integrates the entire module into an evaluation design task.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Benchmark Autopsy (Guided)" color="blue">
                <p className="text-sm">
                  Given a model&rsquo;s scores on 5 benchmark categories,
                  identify which scores are suspicious (uneven performance
                  suggesting contamination), which benchmarks test recognition
                  vs generation, and which dimensions of quality are not
                  measured. First two categories have guided questions; last
                  three are unscaffolded.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: LLM-as-Judge Bias Detection (Supported)" color="blue">
                <p className="text-sm">
                  Use an LLM API to judge pairs of responses where one is
                  longer but less accurate and the other is concise but correct.
                  Vary response length and confidence level systematically.
                  Track how the judge&rsquo;s ratings correlate with length and
                  confidence vs accuracy. Visualize the bias.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Design an Evaluation (Supported)" color="blue">
                <p className="text-sm">
                  Given a specific use case (e.g., a medical Q&amp;A assistant
                  for patients), design an evaluation strategy. What benchmarks
                  would you use? What would you evaluate with human judges? With
                  an LLM judge? What dimensions require red teaming? Produce a
                  one-page evaluation plan that integrates the entire module.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                  Solutions emphasize the reasoning process&mdash;why each
                  evaluation choice was made, what it catches and what it
                  misses&mdash;rather than a single &ldquo;right answer.&rdquo;
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-1-4-evaluating-llms.ipynb"
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
          <TipBlock title="Cumulative Exercises">
            Exercise 1 builds critical reading skills. Exercise 2 demonstrates
            automated evaluation biases empirically. Exercise 3 synthesizes
            the full module into an evaluation design task. Each builds on the
            previous.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          14. Summary (Section 13)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'Benchmark scores are proxies, not measurements.',
                description:
                  'Every score passes through layers of design choices, potential contamination, and optimization pressure before reaching you. The evaluation stack shows the gap between capability and the number on the leaderboard.',
              },
              {
                headline:
                  'Contamination is structural, not accidental.',
                description:
                  'Training on internet-scale data means any public benchmark eventually leaks into training data. Decontamination is temporary—it expires when the benchmark goes public.',
              },
              {
                headline:
                  "Goodhart's law applies to evaluation: when a benchmark becomes a target, it ceases to be a good measure.",
                description:
                  'This is reward hacking at the ecosystem level—the same mechanism from RLHF & Alignment, applied to evaluation rather than training.',
              },
              {
                headline:
                  'Human evaluation is imperfect: expensive, inconsistent, biased, and does not scale.',
                description:
                  'Pairwise comparisons (Chatbot Arena) are more reliable than absolute ratings, but no single evaluation method is sufficient.',
              },
              {
                headline:
                  "LLM-as-judge scales evaluation but inherits its own biases.",
                description:
                  "Verbosity bias, confidence bias, self-preference bias, format sensitivity. The evaluator's limitations always become the evaluation's limitations.",
              },
              {
                headline:
                  'Evaluation may be harder than training.',
                description:
                  'Training has a clear objective: minimize loss. Evaluation requires defining "what do we actually want?"—a multidimensional, context-dependent, moving target with no ground truth.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          15. Module Arc Summary (Section 14)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Module 5.1: The Full Arc"
            subtitle="Build it, break it, measure it&mdash;and measuring may be the hardest part"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Build it (Lessons 1-2). Break it (Lesson 3). Measure it (this
              lesson). And measuring may be the hardest part.
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Constitutional AI:</strong> How to align models at
                scale&mdash;the &ldquo;build&rdquo; phase
              </li>
              <li>
                <strong>The Alignment Techniques Landscape:</strong> The
                design space of alignment techniques&mdash;&ldquo;build, with
                tradeoffs&rdquo;
              </li>
              <li>
                <strong>Red Teaming &amp; Adversarial Evaluation:</strong> How
                to find where alignment fails&mdash;the &ldquo;break&rdquo;
                phase
              </li>
              <li>
                <strong>Evaluating LLMs:</strong> How to measure whether
                alignment worked&mdash;the &ldquo;measure&rdquo; phase
              </li>
            </ul>

            <p className="text-muted-foreground">
              The recurring patterns across all four lessons: blind spots move,
              the challenge shifts, tradeoffs are unavoidable, proxies diverge
              under optimization pressure, and scaling requires automation (with
              its own blind spots).
            </p>

            <p className="text-muted-foreground">
              What you can now do: read an alignment paper or a benchmark
              result and critically assess the claims. Not by dismissing
              everything, but by asking the right questions: What was actually
              measured? How might optimization pressure distort the results?
              What is not being measured? What are the blind spots of the
              evaluation method?
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          16. Module Completion
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="5.1"
            title="Advanced Alignment"
            achievements={[
              'Constitutional AI and RLAIF as scalable alignment',
              'Design space framework for preference optimization',
              'Red teaming as systematic adversarial probing',
              'Attack taxonomy and defense-in-depth',
              'Critical evaluation of benchmarks, human judges, and LLM judges',
              "Goodhart's law: when a measure becomes a target",
            ]}
            nextModule="5.2"
            nextTitle="Reasoning & In-Context Learning"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          17. References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Holistic Evaluation of Language Models (HELM)',
                authors: 'Liang et al., 2022',
                url: 'https://arxiv.org/abs/2211.09110',
                note: 'The most comprehensive attempt at multi-dimensional LLM evaluation. Section 3 breaks down the evaluation taxonomy. Shows how different benchmarks measure different things.',
              },
              {
                title: 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
                authors: 'Zheng et al., 2023',
                url: 'https://arxiv.org/abs/2306.05685',
                note: 'Introduces MT-Bench and systematically studies LLM judge biases (verbosity, position, self-preference). Section 4 quantifies the biases discussed in this lesson.',
              },
              {
                title: 'Contamination in Benchmark Evaluation of Large Language Models',
                authors: 'Various researchers, 2023-2024',
                url: 'https://arxiv.org/abs/2311.09783',
                note: 'Surveys contamination detection methods and documents the structural nature of contamination in web-crawled training data.',
              },
              {
                title: 'Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference',
                authors: 'Chiang et al., 2024',
                url: 'https://arxiv.org/abs/2403.04132',
                note: 'The Chatbot Arena paper. Sections 2-3 explain why pairwise human comparison works better than absolute rating, and Section 5 discusses remaining biases.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          18. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="You now understand how models are aligned, tested, and evaluated. Module 5.2 shifts focus from alignment to capability: how do language models learn from examples in the prompt without updating their weights? How does &ldquo;let's think step by step&rdquo; turn a mediocre model into a strong reasoner? The next module explores the surprising capabilities that emerge from next-token prediction&mdash;capabilities that benchmarks struggle to capture."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
