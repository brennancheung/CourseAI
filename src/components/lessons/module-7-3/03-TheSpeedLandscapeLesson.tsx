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
  ConceptBlock,
  GradientCard,
  ComparisonRow,
  SummaryBlock,
  ModuleCompleteBlock,
  ReferencesBlock,
  NextStepBlock,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
} from 'recharts'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-3-3-the-speed-landscape.ipynb'

/**
 * The Speed Landscape
 *
 * Lesson 3 in Module 7.3 (Fast Generation). Final lesson of Module 7.3 and
 * the speed narrative of Series 7.
 * Cognitive load: CONSOLIDATE (0 new concepts).
 *
 * This is a synthesis lesson: the student knows every acceleration approach
 * and now needs the map. Organizes all approaches (DPM-Solver++, flow matching,
 * LCM, LCM-LoRA, SDXL Turbo) into a four-dimensional decision framework
 * (speed, quality, flexibility, composability).
 *
 * Builds on: all acceleration concepts from Series 6 (6.4.2) and Module 7.2-7.3.
 *
 * Previous: Latent Consistency & Turbo (Module 7.3, Lesson 2 / BUILD)
 * Next: Module 7.4 (Next-Generation Architectures)
 */

/**
 * Schematic quality-vs-steps data for the nonlinear curve.
 * Not real FID data—illustrative shape showing the three regions:
 * flat (1000→20), gentle slope (20→4), steep drop (4→1).
 * Quality is on a 0-100 scale for visual clarity.
 */
const qualitySpeedData = [
  { steps: 1000, quality: 98, label: 'DDPM' },
  { steps: 500, quality: 98 },
  { steps: 200, quality: 97 },
  { steps: 100, quality: 97 },
  { steps: 50, quality: 96 },
  { steps: 30, quality: 95 },
  { steps: 20, quality: 94, label: 'DPM-Solver++' },
  { steps: 15, quality: 91 },
  { steps: 10, quality: 85 },
  { steps: 8, quality: 79 },
  { steps: 6, quality: 72 },
  { steps: 4, quality: 65, label: 'LCM / Flow' },
  { steps: 3, quality: 52 },
  { steps: 2, quality: 38 },
  { steps: 1, quality: 28, label: 'SDXL Turbo' },
]

function QualitySpeedChart() {
  return (
    <div className="rounded-lg border bg-[#1a1a2e] p-4 space-y-3">
      <div className="flex items-center gap-3 border-b border-white/10 pb-2">
        <span className="text-xs font-mono text-white/70">
          Quality vs Steps—The Nonlinear Tradeoff
        </span>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <LineChart
          data={qualitySpeedData}
          margin={{ top: 10, right: 20, bottom: 25, left: 10 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.07)"
          />

          {/* Three colored regions */}
          <ReferenceArea
            x1={20}
            x2={1000}
            fill="#34d399"
            fillOpacity={0.08}
            label={{
              value: 'Free',
              position: 'insideTop',
              fill: '#34d399',
              fontSize: 11,
              fontWeight: 600,
            }}
          />
          <ReferenceArea
            x1={4}
            x2={20}
            fill="#f59e0b"
            fillOpacity={0.08}
            label={{
              value: 'Cheap',
              position: 'insideTop',
              fill: '#f59e0b',
              fontSize: 11,
              fontWeight: 600,
            }}
          />
          <ReferenceArea
            x1={1}
            x2={4}
            fill="#f43f5e"
            fillOpacity={0.08}
            label={{
              value: 'Expensive',
              position: 'insideTop',
              fill: '#f43f5e',
              fontSize: 11,
              fontWeight: 600,
            }}
          />

          {/* Region boundary lines */}
          <ReferenceLine
            x={20}
            stroke="rgba(255,255,255,0.2)"
            strokeDasharray="4 4"
          />
          <ReferenceLine
            x={4}
            stroke="rgba(255,255,255,0.2)"
            strokeDasharray="4 4"
          />

          <XAxis
            dataKey="steps"
            stroke="rgba(255,255,255,0.3)"
            tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
            reversed
            scale="log"
            domain={[1, 1000]}
            type="number"
            label={{
              value: 'Steps (log scale)',
              position: 'insideBottom',
              offset: -15,
              fontSize: 10,
              fill: 'rgba(255,255,255,0.3)',
            }}
          />
          <YAxis
            stroke="rgba(255,255,255,0.3)"
            tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
            domain={[0, 100]}
            label={{
              value: 'Quality',
              angle: -90,
              position: 'insideLeft',
              offset: 5,
              fontSize: 10,
              fill: 'rgba(255,255,255,0.3)',
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e1e3a',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '6px',
              fontSize: 11,
            }}
            labelStyle={{ color: 'rgba(255,255,255,0.6)' }}
            formatter={(value: number | undefined) => [`${value ?? ''}`, 'Quality']}
            labelFormatter={(label: number) => `${label} steps`}
          />
          <Line
            type="monotone"
            dataKey="quality"
            stroke="#818cf8"
            strokeWidth={2.5}
            dot={(props: Record<string, unknown>) => {
              const { cx, cy, payload } = props as {
                cx: number
                cy: number
                payload: { label?: string }
              }
              if (!payload.label) return <circle key={`dot-${cx}`} r={0} />
              return (
                <g key={`label-${payload.label}`}>
                  <circle cx={cx} cy={cy} r={4} fill="#818cf8" />
                  <text
                    x={cx}
                    y={cy - 10}
                    textAnchor="middle"
                    fill="rgba(255,255,255,0.6)"
                    fontSize={9}
                  >
                    {payload.label}
                  </text>
                </g>
              )
            }}
          />
        </LineChart>
      </ResponsiveContainer>

      <p className="text-xs text-muted-foreground text-center">
        Schematic curve. The flat region (1000→20 steps) is free speedup.
        The steep drop (4→1 steps) is where the real tradeoff lives.
      </p>
    </div>
  )
}

/** Composability grid data. Each row is an approach; cells show whether it composes with the column approach. */
type ComposabilityEntry = {
  name: string
  level: string
  color: string
}

const COMPOSABILITY_APPROACHES: ComposabilityEntry[] = [
  { name: 'DPM-Solver++', level: '1', color: '#60a5fa' },
  { name: 'Flow Matching', level: '2', color: '#a78bfa' },
  { name: 'LCM', level: '3a', color: '#34d399' },
  { name: 'LCM-LoRA', level: '3a', color: '#34d399' },
  { name: 'SDXL Turbo', level: '3b', color: '#fb923c' },
]

/**
 * Returns whether two approaches compose.
 * null = same approach (diagonal), true = composes, false = conflicts.
 */
function getComposability(row: number, col: number): boolean | null {
  if (row === col) return null

  // Symmetric: check both directions
  const pair = [row, col].sort().join(',')

  const composable = new Set([
    '0,1', // DPM-Solver++ + Flow Matching (L1+L2): straight paths make the solver more effective
    '1,2', // Flow Matching + LCM (L2+L3a): straighter trajectories improve distillation
    '1,3', // Flow Matching + LCM-LoRA (L2+L3a): same principle, adapter form
  ])

  const conflicts = new Set([
    '0,2', // DPM-Solver++ + LCM: consistency model has no trajectory to solve
    '0,3', // DPM-Solver++ + LCM-LoRA: same conflict—the Kitchen Sink negative example
    '0,4', // DPM-Solver++ + SDXL Turbo: no trajectory to solve
    '2,3', // LCM + LCM-LoRA: both L3a, same mechanism
    '2,4', // LCM + SDXL Turbo: both L3, incompatible bypass mechanisms
    '3,4', // LCM-LoRA + SDXL Turbo: both L3, Turbo is a complete model not an adapter
    '1,4', // Flow Matching + SDXL Turbo: Turbo is a fixed model, cannot swap training objective
  ])

  if (composable.has(pair)) return true
  if (conflicts.has(pair)) return false
  return false
}

function ComposabilityGrid() {
  return (
    <div className="rounded-lg border bg-[#1a1a2e] p-4 space-y-3">
      <div className="flex items-center gap-3 border-b border-white/10 pb-2">
        <span className="text-xs font-mono text-white/70">
          Composability Matrix—Which Approaches Combine?
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="p-2 text-left text-muted-foreground font-normal" />
              {COMPOSABILITY_APPROACHES.map((approach) => (
                <th
                  key={approach.name}
                  className="p-2 text-center font-medium"
                  style={{ color: approach.color }}
                >
                  <div>{approach.name}</div>
                  <div className="text-[10px] text-muted-foreground font-normal">
                    L{approach.level}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {COMPOSABILITY_APPROACHES.map((rowApproach, rowIdx) => (
              <tr key={rowApproach.name} className="border-t border-white/5">
                <td
                  className="p-2 font-medium whitespace-nowrap"
                  style={{ color: rowApproach.color }}
                >
                  <div>{rowApproach.name}</div>
                  <div className="text-[10px] text-muted-foreground font-normal">
                    Level {rowApproach.level}
                  </div>
                </td>
                {COMPOSABILITY_APPROACHES.map((_, colIdx) => {
                  const result = getComposability(rowIdx, colIdx)
                  if (result === null) {
                    return (
                      <td
                        key={colIdx}
                        className="p-2 text-center"
                      >
                        <span className="text-muted-foreground/30">—</span>
                      </td>
                    )
                  }
                  return (
                    <td
                      key={colIdx}
                      className="p-2 text-center"
                    >
                      <span
                        className={`inline-flex items-center justify-center w-7 h-7 rounded-md text-sm font-bold ${
                          result
                            ? 'bg-emerald-500/15 text-emerald-400'
                            : 'bg-rose-500/15 text-rose-400'
                        }`}
                      >
                        {result ? '✓' : '✗'}
                      </span>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex gap-6 justify-center text-xs text-muted-foreground pt-1">
        <span className="flex items-center gap-1.5">
          <span className="inline-flex items-center justify-center w-5 h-5 rounded bg-emerald-500/15 text-emerald-400 text-[10px] font-bold">
            ✓
          </span>
          Complementary
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-flex items-center justify-center w-5 h-5 rounded bg-rose-500/15 text-rose-400 text-[10px] font-bold">
            ✗
          </span>
          Incompatible
        </span>
      </div>
    </div>
  )
}

export function TheSpeedLandscapeLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Speed Landscape"
            description="A decision framework for choosing the right acceleration approach: organize every method you have learned by speed, quality, flexibility, and composability—not by chronological order of invention."
            category="Fast Generation"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Choose the right acceleration approach for a given generation
            scenario by comparing all approaches along four dimensions: speed,
            quality, flexibility, and composability. Walk away with a decision
            framework you can apply to any diffusion pipeline.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="CONSOLIDATE Lesson">
            Zero new concepts. This lesson organizes what you already know into
            a decision framework. The cognitive work is organizational (seeing
            relationships) rather than acquisitional (learning new things).
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'A decision framework for choosing acceleration approaches',
              'A synthesis of all acceleration concepts from Series 6 and Module 7.2-7.3',
              'A composability map showing what combines and what conflicts',
              'NOT: new technical content (no new concepts, formulas, or mechanisms)',
              'NOT: a review lesson (not re-teaching anything—assumes solid knowledge)',
              'NOT: an exhaustive survey of every acceleration method',
              'NOT: a production deployment guide (no inference server optimization)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap — brief activation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Story So Far"
            subtitle="Six tools and no workshop manual"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Over the past three modules, you have learned six ways to generate
              images faster. Each was a response to a limitation of the previous
              approach:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Better ODE solvers</strong> (DPM-Solver++) reduced
                DDPM&rsquo;s 1000 steps to 20—but curved trajectories set a
                floor on step count.
              </li>
              <li>
                <strong>Flow matching</strong> straightened the
                trajectories—but you still need ODE solver steps.
              </li>
              <li>
                <strong>Consistency models</strong> bypassed the trajectory
                entirely—but 1-step output is softer than multi-step diffusion.
              </li>
              <li>
                <strong>Adversarial distillation</strong> sharpened the 1-step
                output—but locked you into a specific model.
              </li>
            </ul>
            <p className="text-muted-foreground">
              Each lesson answered one question and raised another. You now have
              six tools and no workshop manual.{' '}
              <strong>This lesson is the workshop manual.</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Right Question">
            You have learned the &ldquo;three levels of speed&rdquo; framework
            from <LessonLink slug="consistency-models">Consistency Models</LessonLink>. This lesson fills it with
            decision-relevant detail and reframes it from a progression to a
            menu.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — "The Wrong Question" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Wrong Question"
            subtitle="Why 'which is fastest?' leads you astray"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The question you are probably asking:{' '}
              <strong>&ldquo;Which approach is fastest?&rdquo;</strong> The
              answer is SDXL Turbo at 1 step. Case closed.
            </p>
            <p className="text-muted-foreground">
              Except it is not. Consider two scenarios where the fastest
              approach is the <strong>wrong</strong> answer:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="The Photographer" color="amber">
                <div className="space-y-2 text-sm">
                  <p>
                    Custom portrait model (SD 1.5 fine-tune), style LoRA for
                    film grain, ControlNet for pose. Wants fast generation for
                    client previews.
                  </p>
                  <p className="text-muted-foreground italic">
                    SDXL Turbo? No. It is a locked model—cannot use your custom
                    checkpoint, cannot load your style LoRA, different
                    architecture from SD 1.5.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="The Prototype Builder" color="cyan">
                <div className="space-y-2 text-sm">
                  <p>
                    Just installed diffusers, has a vanilla SD 1.5 model, wants
                    to iterate faster during prompt exploration.
                  </p>
                  <p className="text-muted-foreground italic">
                    SDXL Turbo? Overkill. Downloading a dedicated model for
                    quick iteration when a one-line scheduler swap gives you 20
                    steps for free.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              The right question is not &ldquo;which is fastest&rdquo; but{' '}
              <strong>
                &ldquo;which is right for what I need?&rdquo;
              </strong>{' '}
              That question requires evaluating four dimensions, not one:
            </p>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <GradientCard title="Speed" color="emerald">
                <p className="text-sm">How many steps?</p>
              </GradientCard>
              <GradientCard title="Quality" color="blue">
                <p className="text-sm">How much detail lost?</p>
              </GradientCard>
              <GradientCard title="Flexibility" color="violet">
                <p className="text-sm">Works with your model?</p>
              </GradientCard>
              <GradientCard title="Composability" color="orange">
                <p className="text-sm">Works with your adapters?</p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Menu, Not Upgrade Path">
            The three levels of speed (better solvers, straighter trajectories,
            trajectory bypass) are not version numbers where each replaces the
            last. They are menu items—pick what fits your constraints. Level 1
            might be the right answer even when Level 3 exists.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: The Complete Taxonomy */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Taxonomy"
            subtitle="Every approach on one page"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each row below references the lesson where the concept was taught.
              This is not a re-teaching—one sentence to activate the mental
              model, then evaluation across the four dimensions.
            </p>

            {/* The comparison grid, broken into individual cards for readability */}
            <GradientCard title="Level 1: DPM-Solver++" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Mechanism:</strong> Higher-order ODE solver that reads
                  trajectory curvature. Swap the scheduler, keep everything
                  else. (from <LessonLink slug="samplers-and-efficiency">Samplers &amp; Efficiency</LessonLink>)
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-muted-foreground">
                  <p><strong>Steps:</strong> 15-20</p>
                  <p><strong>What changes:</strong> Scheduler only (inference)</p>
                  <p><strong>Quality trade:</strong> None at 20 steps</p>
                  <p><strong>Flexibility:</strong> Any model, any checkpoint</p>
                  <p><strong>Composability:</strong> Full—nothing changes about the model</p>
                </div>
              </div>
            </GradientCard>

            <GradientCard title="Level 2: Flow Matching (SD3/Flux)" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Mechanism:</strong> Straight-line interpolation with
                  velocity prediction. Euler&rsquo;s method on straight paths
                  is nearly exact. (from <LessonLink slug="flow-matching">Flow Matching</LessonLink>)
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-muted-foreground">
                  <p><strong>Steps:</strong> 20-30</p>
                  <p><strong>What changes:</strong> Training objective + model weights</p>
                  <p><strong>Quality trade:</strong> None at 25 steps</p>
                  <p><strong>Flexibility:</strong> Requires flow-matching-native model</p>
                  <p><strong>Composability:</strong> Full—standard diffusion pipeline</p>
                </div>
              </div>
            </GradientCard>

            <GradientCard title="Level 3a: LCM (Consistency Distillation)" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Mechanism:</strong> Student model distilled from
                  teacher via the consistency objective. Teleport to the
                  destination. (from <LessonLink slug="consistency-models">Consistency Models</LessonLink>)
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-muted-foreground">
                  <p><strong>Steps:</strong> 2-4</p>
                  <p><strong>What changes:</strong> Student model distilled from teacher</p>
                  <p><strong>Quality trade:</strong> Small at 4 steps, noticeable at 1</p>
                  <p><strong>Flexibility:</strong> Requires distillation per base model</p>
                  <p><strong>Composability:</strong> Moderate—fixed to teacher&rsquo;s capabilities</p>
                </div>
              </div>
            </GradientCard>

            <GradientCard title="Level 3a: LCM-LoRA (Speed as a Skill)" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Mechanism:</strong> Consistency distillation captured
                  as a LoRA adapter. One adapter, many models. (from{' '}
                  <LessonLink slug="latent-consistency-and-turbo">Latent Consistency &amp; Turbo</LessonLink>)
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-muted-foreground">
                  <p><strong>Steps:</strong> 4-8</p>
                  <p><strong>What changes:</strong> LoRA adapter loaded at inference</p>
                  <p><strong>Quality trade:</strong> Moderate (LoRA approximation)</p>
                  <p><strong>Flexibility:</strong> Any compatible base model</p>
                  <p><strong>Composability:</strong> High—composable with style LoRA, ControlNet</p>
                </div>
              </div>
            </GradientCard>

            <GradientCard title="Level 3b: SDXL Turbo (Adversarial Distillation)" color="orange">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Mechanism:</strong> Hybrid diffusion + adversarial
                  loss. Two teachers: ODE consistency + discriminator realism.
                  (from <LessonLink slug="latent-consistency-and-turbo">Latent Consistency &amp; Turbo</LessonLink>)
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-muted-foreground">
                  <p><strong>Steps:</strong> 1-4</p>
                  <p><strong>What changes:</strong> Dedicated distilled model</p>
                  <p><strong>Quality trade:</strong> Sharpest at 1 step (adversarial)</p>
                  <p><strong>Flexibility:</strong> Locked to SDXL Turbo checkpoint</p>
                  <p><strong>Composability:</strong> Low—cannot swap base model, limited adapters</p>
                </div>
              </div>
            </GradientCard>

            <GradientCard title="Level 3a: Consistency Training (No Teacher)" color="amber">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Mechanism:</strong> Trained from scratch using the
                  consistency objective, no pretrained teacher. (from{' '}
                  <strong>Consistency Models</strong>)
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-muted-foreground">
                  <p><strong>Steps:</strong> 1-4</p>
                  <p><strong>What changes:</strong> Trained from scratch</p>
                  <p><strong>Quality trade:</strong> Lower than distillation (FID ~6.2 vs ~3.5)</p>
                  <p><strong>Flexibility:</strong> Requires full training</p>
                  <p><strong>Composability:</strong> Same as the trained model</p>
                </div>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Levels Are Options, Not Upgrades">
            A researcher training a new architecture uses Level 1 immediately
            and may never need Level 3. The levels have different requirements,
            different costs, and different flexibility. They are a menu, not a
            progression.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* The Quality-Speed Curve */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Quality-Speed Curve"
            subtitle="Most of the speedup is free"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              There is a common assumption that fewer steps always means worse
              quality—that the tradeoff is linear. It is not. The curve has
              three distinct regions:
            </p>

            <QualitySpeedChart />

            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Free Speedup" color="emerald">
                <div className="space-y-2 text-sm">
                  <p className="font-medium">1000 → 20 steps</p>
                  <p className="text-muted-foreground">
                    DPM-Solver++ eliminates DDPM&rsquo;s redundancy with{' '}
                    <strong>no quality loss</strong>. This is wasted computation
                    removed. A 50&times; speedup that costs nothing.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Cheap Speedup" color="amber">
                <div className="space-y-2 text-sm">
                  <p className="font-medium">20 → 4 steps</p>
                  <p className="text-muted-foreground">
                    Consistency distillation or flow matching reduces further
                    with <strong>small, measurable quality cost</strong>. Detail
                    softening at edges, some high-frequency texture loss. A
                    5&times; speedup at modest cost.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Expensive Speedup" color="rose">
                <div className="space-y-2 text-sm">
                  <p className="font-medium">4 → 1 step</p>
                  <p className="text-muted-foreground">
                    Single-step generation shows <strong>noticeable</strong>{' '}
                    softness (consistency) or occasional texture artifacts
                    (adversarial). A 4&times; speedup at the highest per-step
                    quality cost.
                  </p>
                </div>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The First 50× Is Free">
            Going from 1000 to 20 steps costs nothing. The next 5&times;
            (20→4) costs a little. The last 4&times; (4→1) costs the most
            per step gained. Most users should take the free speedup and stop
            there unless they have specific latency requirements.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check #1 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict before you read the answers"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A researcher trains a brand-new diffusion model on a novel
                  dataset. Which acceleration level can they use immediately
                  without any additional training?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Level 1: swap the scheduler to DPM-Solver++. It is an
                    inference-time change that works with any diffusion model.
                    No downloading, no LoRA loading, no distillation. One line
                    of code.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A user wants 4-step generation from their custom SD 1.5
                  fine-tune. Which approach should they consider?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    LCM-LoRA. It is composable with their custom checkpoint (no
                    retraining needed), composable with any style LoRAs they
                    use, and gives 4-step generation from a 4 MB adapter file.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  True or false: &ldquo;Flow matching (Level 2) makes Level 1
                  approaches obsolete.&rdquo;
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    False. Even flow matching models benefit from DPM-Solver++
                    over naive Euler. The straighter trajectories make the
                    solver even more effective—the levels compose. Level 1 +
                    Level 2 is better than Level 2 alone.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: The Decision Framework (interactive scenarios) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Decision Framework"
            subtitle="Three scenarios, three different answers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The comparison grid is inert until applied to specific situations.
              Here are three scenarios that reach different conclusions using the
              same four-dimensional evaluation.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Scenario 1: The Photographer (detailed walkthrough) */}
      <Row>
        <Row.Content>
          <GradientCard title="Scenario 1: The Photographer" color="violet">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Needs:</strong> Custom portrait model (SD 1.5 fine-tune),
                style LoRA for &ldquo;film grain&rdquo; look, ControlNet for
                pose, wants fast generation for client previews.
              </p>
              <p className="font-medium">Evaluate each approach:</p>
              <ul className="space-y-2 text-muted-foreground ml-2">
                <li>
                  <strong>SDXL Turbo?</strong> No—locked model, no LoRA
                  composability, different architecture from SD 1.5. Fails on
                  flexibility and composability.
                </li>
                <li>
                  <strong>Full LCM?</strong> Possible but requires distilling
                  from their specific model. High effort.
                </li>
                <li>
                  <strong>LCM-LoRA?</strong> Yes—plug into their model, compose
                  with style LoRA and ControlNet, 4-step generation. Wins on
                  flexibility and composability.
                </li>
              </ul>
              <p className="font-medium text-emerald-400">
                Answer: LCM-LoRA + style LoRA + ControlNet at 4-8 steps.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The 'Obvious' Answer Is Wrong">
            SDXL Turbo is the fastest option—but it cannot work here. The
            photographer&rsquo;s constraints (custom model, style LoRA,
            ControlNet) make flexibility and composability the deciding factors,
            not raw speed.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Scenario 2: The Prototype Builder (brief) */}
      <Row>
        <Row.Content>
          <GradientCard title="Scenario 2: The Prototype Builder" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Needs:</strong> Just installed diffusers, vanilla SD 1.5
                model, wants faster iteration during prompt exploration.
              </p>
              <ul className="space-y-2 text-muted-foreground ml-2">
                <li>
                  <strong>Any Level 3?</strong> Overkill—requires downloading
                  adapters or dedicated models.
                </li>
                <li>
                  <strong>Level 2?</strong> Requires a flow matching model they
                  do not have.
                </li>
                <li>
                  <strong>Level 1?</strong> One line change. Done.
                </li>
              </ul>
              <CodeBlock
                code={`pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
# Set num_inference_steps=20. That's it.`}
                language="python"
                filename="one_line_speedup.py"
              />
              <p className="font-medium text-emerald-400">
                Answer: DPM-Solver++ at 20 steps. Zero friction.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Simplest Solution">
            The excitement about Level 3 approaches can make Level 1 feel
            boring. But for prompt exploration and prototyping, a 50&times;
            speedup with zero setup is hard to beat.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Scenario 3: The Real-Time App (brief) */}
      <Row>
        <Row.Content>
          <GradientCard title="Scenario 3: The Real-Time App" color="orange">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Needs:</strong> Latency under 500ms, competitive
                quality, does not need custom styles or adapters.
              </p>
              <ul className="space-y-2 text-muted-foreground ml-2">
                <li>
                  <strong>Level 1 at 20 steps?</strong> Too slow for real-time.
                </li>
                <li>
                  <strong>LCM-LoRA at 4 steps?</strong> Possible but not the
                  sharpest 1-step option.
                </li>
                <li>
                  <strong>SDXL Turbo at 1 step?</strong> Yes—sharpest 1-step
                  quality, acceptable flexibility tradeoff for a fixed
                  deployment.
                </li>
              </ul>
              <p className="font-medium text-emerald-400">
                Answer: SDXL Turbo (ADD) at 1-4 steps. Trade flexibility for
                latency.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="When Speed Wins">
            This is the genuine case where the &ldquo;locked model&rdquo;
            tradeoff is worth it. A fixed deployment with latency constraints
            and no need for custom adapters is exactly what SDXL Turbo was
            designed for.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Composability — what combines and what conflicts */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Composability"
            subtitle="What combines across levels and what conflicts within them"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One of the most common misconceptions: &ldquo;these approaches are
              mutually exclusive—pick one.&rdquo; In reality, approaches{' '}
              <strong>compose across levels</strong> but{' '}
              <strong>conflict within levels</strong>.
            </p>

            <ComposabilityGrid />

            <GradientCard title="Composes Well" color="emerald">
              <ul className="space-y-2 text-sm">
                <li>
                  <strong>Level 1 + Level 2:</strong> DPM-Solver++ on a flow
                  matching model. Straight trajectories make the solver even
                  more effective. Different levels, complementary.
                </li>
                <li>
                  <strong>Level 2 + Level 3a:</strong> Flow matching teacher +
                  consistency distillation. This IS what LCM does at scale.
                  Straighter trajectories give the consistency model a better
                  training signal.
                </li>
                <li>
                  <strong>LCM-LoRA + style LoRA:</strong> Speed + style as
                  additive LoRA bypasses. Same model, different skills.
                </li>
                <li>
                  <strong>LCM-LoRA + ControlNet:</strong> Speed adapter +
                  spatial control. Target different parts of the pipeline.
                </li>
              </ul>
            </GradientCard>

            <GradientCard title="Does NOT Compose" color="rose">
              <ul className="space-y-2 text-sm">
                <li>
                  <strong>LCM-LoRA + SDXL Turbo:</strong> Both address Level 3
                  with incompatible mechanisms. SDXL Turbo is a complete model,
                  not an adapter.
                </li>
                <li>
                  <strong>DPM-Solver++ + consistency model:</strong> The
                  consistency model does not follow the trajectory. There is no
                  ODE to solve with a higher-order method. The solver has
                  nothing to solve.
                </li>
                <li>
                  <strong>Two Level 3 approaches on the same model:</strong>{' '}
                  Pick one bypass mechanism—they cannot coexist.
                </li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Across, Not Within">
            The composability rule is simple: approaches from{' '}
            <strong>different</strong> levels compose (they address different
            aspects of generation). Approaches from the{' '}
            <strong>same</strong> level conflict (they address the same aspect
            with incompatible mechanisms).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* The Negative Example: "The Kitchen Sink" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Kitchen Sink"
            subtitle="A negative example: more acceleration is not always better"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Someone loads a flow matching model, adds LCM-LoRA, sets
              DPM-Solver++ as the scheduler, and sets steps=20. What happens?
            </p>
            <GradientCard title="Why This Fails" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  The LCM-LoRA has been trained to produce quality output in 4
                  steps via the consistency objective. Running 20 steps with
                  DPM-Solver++ is solving an ODE that the consistency model was
                  trained to bypass.
                </p>
                <p>
                  The result is unpredictable and likely worse than either
                  approach used correctly. The DPM-Solver++ expects a
                  well-behaved vector field that it can step along. The
                  LCM-LoRA has modified the model to teleport in 4 steps, not
                  walk in 20. The approaches do not just fail to help—they{' '}
                  <strong>actively conflict</strong>.
                </p>
              </div>
            </GradientCard>
            <p className="text-muted-foreground">
              The correct configurations from this setup:
            </p>
            <ComparisonRow
              left={{
                title: 'Option A: Flow Matching + Solver',
                color: 'emerald',
                items: [
                  'Use the flow matching model (Level 2)',
                  'Use DPM-Solver++ as the scheduler (Level 1)',
                  'Set steps=20',
                  'Remove the LCM-LoRA',
                  'Levels 1 + 2 compose correctly',
                ],
              }}
              right={{
                title: 'Option B: LCM-LoRA',
                color: 'emerald',
                items: [
                  'Use the base model + LCM-LoRA (Level 3a)',
                  'Use the LCM scheduler (not DPM-Solver++)',
                  'Set steps=4',
                  'Level 3a—the consistency model IS the solver',
                  'DPM-Solver++ has nothing to solve here',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="More Acceleration ≠ Faster">
            Stacking acceleration approaches from the same level does not make
            them faster—it makes them conflict. The consistency model has no
            trajectory for DPM-Solver++ to step along. Know when to stop
            stacking.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Check #2 (transfer questions) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Questions"
            subtitle="Apply the framework to new situations"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Can you use a higher-order ODE solver (DPM-Solver++) to
                  improve the output quality of an LCM model at 4 steps? Why
                  or why not?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. LCM produces output via the consistency function, not by
                    ODE solving. DPM-Solver++ assumes a trajectory to step
                    along—LCM has no trajectory. There is no ODE to solve. Use
                    multi-step consistency refinement instead (teleport to x₀,
                    re-noise, teleport again).
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A team trained a diffusion model using flow matching (Level 2).
                  They now want to push to 4-step generation. What should they
                  consider?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Consistency distillation from their flow matching
                    model—Level 2 + Level 3a compose. The straight trajectories
                    from flow matching give the consistency model a better
                    training signal. Alternatively: distill an LCM-LoRA from
                    the model for universal plug-and-play across compatible
                    fine-tunes.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Speed Comparison"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes the framework concrete—compare approaches
                head-to-head, verify composability, and build your own decision
                cheat sheet.
              </p>
              <a
                href={NOTEBOOK_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <ul className="text-sm text-muted-foreground space-y-3">
                <li>
                  <strong>
                    Exercise 1 (Guided): The Sampler Swap.
                  </strong>{' '}
                  Load a vanilla SD 1.5 model. Generate the same image with DDPM
                  (50 steps), DPM-Solver++ (20 steps), and DPM-Solver++ (10
                  steps). Compare quality and time. The 50→20 step reduction is
                  nearly free.
                </li>
                <li>
                  <strong>
                    Exercise 2 (Guided): LCM-LoRA Speed Comparison.
                  </strong>{' '}
                  Load the same SD 1.5 model + LCM-LoRA. Generate at 4 steps and
                  8 steps. Compare with DPM-Solver++ at 20 steps from Exercise
                  1. Level 3a vs Level 1 head-to-head.
                </li>
                <li>
                  <strong>
                    Exercise 3 (Supported): The Composability Test.
                  </strong>{' '}
                  Combine LCM-LoRA + a style LoRA. Generate at 4 steps. Compare
                  with LCM-LoRA alone and style LoRA alone at 20 steps with
                  DPM-Solver++. Verify that speed + style compose as predicted.
                </li>
                <li>
                  <strong>
                    Exercise 4 (Independent): Build Your Decision Cheat Sheet.
                  </strong>{' '}
                  Given three generation scenarios, write which approach you would
                  choose and why using the four-dimensional framework. Then verify
                  one of your choices by implementing it and comparing with an
                  alternative.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: sampler swap (Level 1)</li>
              <li>Guided: LCM-LoRA (Level 3a vs 1)</li>
              <li>Supported: composability test</li>
              <li>Independent: decision framework</li>
            </ol>
            <p className="text-sm mt-2">
              No SDXL Turbo exercises (requires separate model download and
              significant VRAM). Its position in the framework is established
              conceptually.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Speed is a menu, not an upgrade path.',
                description:
                  'The three levels (better solvers, straighter trajectories, trajectory bypass) are options with different requirements and tradeoffs, not versions where each replaces the last.',
              },
              {
                headline: 'Most speedup is free.',
                description:
                  'Going from 1000 to 20 steps (Level 1) costs nothing. The quality-speed tradeoff only becomes real below ~10 steps.',
              },
              {
                headline: 'Four dimensions, not two.',
                description:
                  'Speed and quality are obvious. Flexibility (works with your model?) and composability (works with your adapters?) often determine the practical choice.',
              },
              {
                headline: 'Levels compose across, not within.',
                description:
                  'Flow matching (Level 2) + consistency distillation (Level 3a) is strictly better than either alone. But two Level 3 approaches on the same model conflict.',
              },
              {
                headline:
                  'The question is not "which is fastest" but "which is right for what I need."',
                description:
                  'A photographer with a custom model needs LCM-LoRA. A researcher with a new architecture needs DPM-Solver++. A real-time app needs SDXL Turbo. The framework tells you which.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 12: Module Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="7.3"
            title="Fast Generation"
            achievements={[
              'The self-consistency property and how it enables 1-step generation',
              'LCM-LoRA for practical fast generation and ADD/SDXL Turbo as an alternative',
              'A four-dimensional decision framework for choosing the right acceleration approach',
            ]}
            nextModule="7.4"
            nextTitle="Next-Generation Architectures"
          />
        </Row.Content>
      </Row>

      {/* Bridge to Module 7.4 */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have seen how to make diffusion{' '}
              <strong>faster</strong>. The next module asks: what if we also
              change the <strong>architecture</strong>? SDXL stretches the U-Net
              to its limits. Then the Diffusion Transformer (DiT) replaces it
              entirely—bringing the transformer architecture from Series 4 into
              the diffusion world. SD3 and Flux combine DiT + flow matching +
              better text encoding. Everything you have learned converges.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models',
                authors: 'Lu, Zhou, Bao, Chen, Li & Zhu, 2023',
                url: 'https://arxiv.org/abs/2211.01095',
                note: 'The DPM-Solver++ paper (Level 1). Section 3 covers the higher-order solver and guided sampling.',
              },
              {
                title: 'Flow Matching for Generative Modeling',
                authors: 'Lipman, Chen, Ben-Hamu, Nickel, & Le, 2023',
                url: 'https://arxiv.org/abs/2210.02747',
                note: 'The flow matching paper (Level 2). Section 3 introduces conditional flow matching with straight-line paths.',
              },
              {
                title: 'Consistency Models',
                authors: 'Song, Dhariwal, Chen & Sutskever, 2023',
                url: 'https://arxiv.org/abs/2303.01469',
                note: 'The consistency models paper (Level 3a). The self-consistency property and distillation procedure.',
              },
              {
                title: 'LCM-LoRA: A Universal Stable-Diffusion Acceleration Module',
                authors: 'Luo, Tan, Huang, Li & Zhao, 2023',
                url: 'https://arxiv.org/abs/2311.05556',
                note: 'The LCM-LoRA paper (Level 3a, adapter form). Demonstrates cross-model universality.',
              },
              {
                title: 'Adversarial Diffusion Distillation',
                authors: 'Sauer, Lorenz, Blattmann & Rombach, 2023',
                url: 'https://arxiv.org/abs/2311.17042',
                note: 'The SDXL Turbo paper (Level 3b). The hybrid adversarial + diffusion distillation loss.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 13: Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Module 7.3 Complete"
            description="You can now choose the right acceleration approach for any generation scenario. The next module explores the architectural frontier: SDXL, Diffusion Transformers (DiT), SD3, and Flux—where everything you have learned about transformers, flow matching, and fast generation converges."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
