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
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
/**
 * Open Weight Image Generation Models -- Landscape Survey
 *
 * Lesson 1 in Module 8.4 (Image Generation Landscape). Series 8 Special Topics.
 * Cognitive load: CONSOLIDATE (zero new concepts).
 *
 * A comprehensive survey that organizes the open weight image generation
 * landscape (2022-2025) into a navigable taxonomy. Every technology referenced
 * was taught in Series 6-7 or Module 8.3. The value is in the map: seeing
 * which innovations propagated across models, which architectural choices were
 * dead ends, and how the field moved from a single U-Net-based model to a
 * diverse ecosystem of transformer-based architectures.
 *
 * No new concepts are introduced. The cognitive work is organizational,
 * not conceptual.
 *
 * Previous: Nano Banana Pro (module 8.3, lesson 1)
 * Next: Standalone (no specific next lesson)
 */

/* ====================================================================== */
/* Inline SVG diagrams — replace Mermaid for readable sizing              */
/* ====================================================================== */

type TimelineEntry = {
  month: string
  name: string
  lab: string
  family: 'unet' | 'pixel' | 'dit' | 'mmdit' | 's3dit'
}

type YearGroup = {
  year: string
  entries: TimelineEntry[]
}

const FAMILY_COLORS: Record<TimelineEntry['family'], string> = {
  unet: '#f59e0b',
  pixel: '#f43f5e',
  dit: '#06b6d4',
  mmdit: '#8b5cf6',
  s3dit: '#22c55e',
}

const TIMELINE_DATA: YearGroup[] = [
  {
    year: '2022',
    entries: [
      { month: 'Aug', name: 'SD v1', lab: 'Stability AI / CompVis / Runway', family: 'unet' },
      { month: 'Nov', name: 'SD v2', lab: 'Stability AI', family: 'unet' },
    ],
  },
  {
    year: '2023',
    entries: [
      { month: 'Apr', name: 'DeepFloyd IF', lab: 'Stability AI / DeepFloyd', family: 'pixel' },
      { month: 'Apr', name: 'Kandinsky 2.x', lab: 'Sber AI', family: 'unet' },
      { month: 'Jul', name: 'SDXL', lab: 'Stability AI', family: 'unet' },
      { month: 'Nov', name: 'PixArt-alpha', lab: 'Huawei', family: 'dit' },
      { month: 'Dec', name: 'Stable Cascade', lab: 'Stability AI', family: 'pixel' },
    ],
  },
  {
    year: '2024',
    entries: [
      { month: 'Feb', name: 'Playground v2.5', lab: 'Playground AI', family: 'unet' },
      { month: 'Feb', name: 'SD3 paper', lab: 'Stability AI', family: 'mmdit' },
      { month: 'Apr', name: 'PixArt-sigma', lab: 'Huawei', family: 'dit' },
      { month: 'May', name: 'Hunyuan-DiT', lab: 'Tencent', family: 'dit' },
      { month: 'Jun', name: 'AuraFlow', lab: 'Fal.ai', family: 'mmdit' },
      { month: 'Jun', name: 'SD3 release', lab: 'Stability AI', family: 'mmdit' },
      { month: 'Jul', name: 'Kolors', lab: 'Kuaishou', family: 'unet' },
      { month: 'Aug', name: 'Flux.1', lab: 'Black Forest Labs', family: 'mmdit' },
      { month: 'Oct', name: 'SD3.5', lab: 'Stability AI', family: 'mmdit' },
    ],
  },
  {
    year: '2025',
    entries: [
      { month: 'Jan', name: 'Z-Image', lab: 'Freepik / Tongyi Lab', family: 's3dit' },
    ],
  },
]

function TimelineDiagram() {
  const entryH = 34
  const yearGap = 20
  const topPad = 52
  const leftCol = 60
  const dotX = 100
  const monthX = 120
  const nameX = 170
  const labX = 420

  // Pre-compute y positions
  type TimelineRow =
    | { y: number; type: 'year'; year: string }
    | { y: number; type: 'entry'; entry: TimelineEntry }
  const rows: TimelineRow[] = []
  let y = topPad
  for (const group of TIMELINE_DATA) {
    rows.push({ y, type: 'year' as const, year: group.year })
    y += entryH
    for (const entry of group.entries) {
      rows.push({ y, type: 'entry' as const, entry })
      y += entryH
    }
    y += yearGap
  }
  const svgH = y + 10

  return (
    <svg
      viewBox={`0 0 700 ${svgH}`}
      className="w-full rounded-lg border bg-card/30 p-2"
      role="img"
      aria-label="Open Weight Image Generation Timeline from 2022 to 2025"
    >
      <text x={350} y={28} textAnchor="middle" fill="#e2e8f0" fontSize={16} fontWeight={600}>
        Open Weight Image Generation Timeline
      </text>

      {/* Vertical timeline line */}
      <line x1={dotX} y1={topPad + 6} x2={dotX} y2={svgH - 20} stroke="#334155" strokeWidth={2} />

      {rows.map((row, i) => {
        if (row.type === 'year') {
          return (
            <g key={`year-${i}`}>
              <rect x={leftCol - 40} y={row.y - 12} width={80} height={26} rx={4} fill="#1e293b" stroke="#475569" strokeWidth={1} />
              <text x={leftCol} y={row.y + 5} textAnchor="middle" fill="#e2e8f0" fontSize={14} fontWeight={700}>
                {(row as { year: string }).year}
              </text>
            </g>
          )
        }

        const entry = (row as { entry: TimelineEntry }).entry
        const color = FAMILY_COLORS[entry.family]
        return (
          <g key={`entry-${i}`}>
            <circle cx={dotX} cy={row.y} r={5} fill={color} />
            <text x={monthX} y={row.y + 4} fill="#94a3b8" fontSize={12}>
              {entry.month}
            </text>
            <text x={nameX} y={row.y + 4} fill="#e2e8f0" fontSize={13} fontWeight={600}>
              {entry.name}
            </text>
            <text x={labX} y={row.y + 4} fill="#64748b" fontSize={11}>
              {entry.lab}
            </text>
          </g>
        )
      })}

      {/* Legend */}
      {[
        { label: 'U-Net', color: FAMILY_COLORS.unet },
        { label: 'Pixel-Space', color: FAMILY_COLORS.pixel },
        { label: 'DiT', color: FAMILY_COLORS.dit },
        { label: 'MMDiT', color: FAMILY_COLORS.mmdit },
        { label: 'S3-DiT', color: FAMILY_COLORS.s3dit },
      ].map((item, i) => (
        <g key={`legend-${i}`}>
          <circle cx={140 * i + 100} cy={svgH - 6} r={4} fill={item.color} />
          <text x={140 * i + 110} y={svgH - 2} fill="#94a3b8" fontSize={11}>
            {item.label}
          </text>
        </g>
      ))}
    </svg>
  )
}

type TaxonomyFamily = {
  name: string
  shortName: string
  color: string
  children: string[]
}

const TAXONOMY_FAMILIES: TaxonomyFamily[] = [
  { name: 'U-Net Family', shortName: 'U-Net', color: '#f59e0b', children: ['SD v1.x', 'SD v2.x', 'SDXL', 'Playground v2.5', 'Kolors', 'Kandinsky 2.x'] },
  { name: 'Pixel-Space / Cascaded', shortName: 'Pixel', color: '#f43f5e', children: ['DeepFloyd IF', 'Stable Cascade'] },
  { name: 'DiT Family (Cross-Attention)', shortName: 'DiT', color: '#06b6d4', children: ['PixArt-alpha', 'PixArt-sigma', 'Hunyuan-DiT'] },
  { name: 'MMDiT Family (Joint Attention)', shortName: 'MMDiT', color: '#8b5cf6', children: ['SD3 / SD3.5', 'Flux.1', 'AuraFlow'] },
  { name: 'S3-DiT (Single-Stream)', shortName: 'S3-DiT', color: '#22c55e', children: ['Z-Image'] },
]

function TaxonomyDiagram() {
  const rowH = 80
  const familyStartY = 90

  return (
    <svg
      viewBox="0 0 700 520"
      className="w-full rounded-lg border bg-card/30 p-2"
      role="img"
      aria-label="Architecture taxonomy: five backbone families for open weight image generation models"
    >
      {/* Root node */}
      <rect x={220} y={12} width={260} height={36} rx={6} fill="#1e293b" stroke="#6366f1" strokeWidth={2} />
      <text x={350} y={36} textAnchor="middle" fill="#e2e8f0" fontSize={14} fontWeight={600}>
        Open Weight Image Gen Models
      </text>

      {/* Vertical connector from root */}
      <line x1={350} y1={48} x2={350} y2={70} stroke="#475569" strokeWidth={2} />

      {/* Horizontal connector bar */}
      <line x1={40} y1={70} x2={660} y2={70} stroke="#475569" strokeWidth={2} />

      {TAXONOMY_FAMILIES.map((family, fi) => {
        const baseY = familyStartY + fi * rowH
        const dropX = 40 + fi * 155

        // Dynamic spacing based on child count
        const spacing = family.children.length > 4 ? 70 : 80

        return (
          <g key={family.shortName}>
            {/* Vertical drop from connector bar */}
            <line x1={dropX} y1={70} x2={dropX} y2={baseY} stroke="#475569" strokeWidth={1.5} />

            {/* Family box */}
            <rect x={10} y={baseY} width={220} height={32} rx={5} fill="#1e293b" stroke={family.color} strokeWidth={2} />
            <text x={120} y={baseY + 21} textAnchor="middle" fill="#e2e8f0" fontSize={12} fontWeight={600}>
              {family.name}
            </text>

            {/* Horizontal connector to children */}
            <line x1={230} y1={baseY + 16} x2={250} y2={baseY + 16} stroke={family.color} strokeWidth={1.5} />

            {/* Children as inline tags */}
            {family.children.map((child, ci) => {
              const childX = 260 + ci * spacing

              return (
                <g key={child}>
                  <rect
                    x={childX}
                    y={baseY + 4}
                    width={spacing - 6}
                    height={24}
                    rx={4}
                    fill="#0f172a"
                    stroke={family.color}
                    strokeWidth={1}
                    strokeOpacity={0.4}
                  />
                  <text
                    x={childX + (spacing - 6) / 2}
                    y={baseY + 20}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize={10}
                  >
                    {child}
                  </text>
                </g>
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}

type EvolutionStep = {
  label: string
  color: string
}

function EvolutionLine({ steps }: { steps: EvolutionStep[] }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-center gap-2">
          <div
            className="rounded-md border px-3 py-1.5 text-xs font-medium text-slate-200"
            style={{ borderColor: step.color, backgroundColor: '#1e293b' }}
          >
            {step.label}
          </div>
          {i < steps.length - 1 && (
            <span className="text-slate-500 text-sm">&rarr;</span>
          )}
        </div>
      ))}
    </div>
  )
}

export function OpenWeightImageGenLesson() {
  return (
    <LessonLayout>
      {/* ================================================================== */}
      {/* Section 1: Header                                                  */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Open Weight Image Generation Landscape"
            description="A map of every major open weight image generation model from 2022 to 2025&mdash;organized by architecture, text encoding, and training objective. You already know the building blocks. This lesson shows you the forest."
            category="Image Generation Landscape"
            duration="45-60 min"
          />
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 2: Context + Constraints                                   */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Place any open weight image generation model into the architectural
            and historical landscape by identifying its denoising backbone, text
            encoder(s), training objective, key innovation, and lineage within
            the broader evolution.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            From Series 6&ndash;7 and Module 8.3, you know U-Nets, DiTs,
            MMDiTs, S3-DiTs, VAEs, CLIP, T5, flow matching, cross-attention,
            joint attention, CFG, LoRA, consistency distillation, and DMD.
            Every technology in this lesson is vocabulary you already have.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The landscape of open weight image generation models (2022\u20132025)',
              'Architecture identification: backbone type, text encoder(s), training objective for each model',
              'Historical timeline and lineage (which lab, which researchers, what came from where)',
              'A reference comparison table you can return to',
              'A framework for placing new model announcements into this map',
              'NOT: teaching any new architecture, training objective, or technique (all assumed known)',
              'NOT: detailed architectural deep dives into models not already covered',
              'NOT: model quality benchmarks or rankings (too subjective and rapidly outdated)',
              'NOT: licensing details beyond the open-weight vs open-source distinction',
              'NOT: video generation models (brief mention only)',
              'NOT: practical deployment, inference optimization, or model selection for specific tasks',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Longer Than Typical">
            This is a 45&ndash;60 minute survey lesson&mdash;much longer than
            usual. Take breaks if needed. The goal is a reference map, not a
            sprint.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================== */}
      {/* Section 3: Hook -- The Overwhelming Landscape                      */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Overwhelming Landscape"
            subtitle="Eighteen model names. How many truly distinct architectures?"
          />
          <div className="space-y-4">
            <div className="rounded-lg border bg-card/50 p-4">
              <p className="text-sm text-muted-foreground font-mono leading-relaxed">
                SD v1.5 &middot; SD v2.1 &middot; SDXL &middot; PixArt-alpha
                &middot; PixArt-sigma &middot; DeepFloyd IF &middot; Kandinsky
                2.1 &middot; Stable Cascade &middot; Playground v2.5 &middot;
                SD3 &middot; SD3.5 &middot; Hunyuan-DiT &middot; Kolors
                &middot; AuraFlow &middot; Flux.1 Dev &middot; Flux.1 Schnell
                &middot; Z-Image Base &middot; Z-Image Turbo
              </p>
            </div>
            <p className="text-muted-foreground">
              If you tried to learn each model from scratch, this would take
              weeks. But you do not need to. Once you know the building
              blocks&mdash;and you do&mdash;the number of truly distinct
              architectural paradigms is surprisingly small.
            </p>
            <p className="text-muted-foreground">
              There are really only <strong>4&ndash;5 backbone types</strong>{' '}
              and <strong>3&ndash;4 text encoding strategies</strong>. The rest
              is combinations.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Map, Not the Trees">
            You have spent Series 6&ndash;7 studying individual
            trees&mdash;U-Nets, DiTs, MMDiTs, S3-DiTs, flow matching. You can
            trace a denoising step, explain why CFG works, describe the
            difference between SD v1.5 and SD3 at the architectural level. This
            lesson steps back to see the forest.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================== */}
      {/* Section 4: The Timeline (Historical Context)                       */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Timeline"
            subtitle="From SD v1 to Z-Image: three years of rapid evolution"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before the open weight story begins, three closed models set the
              stage. DALL-E (2021) proved autoregressive text-to-image works.
              Imagen (2022) demonstrated T5-XXL for text encoding in diffusion.
              Midjourney set the aesthetic benchmark that every open model
              chases. Then, in August 2022, Stable Diffusion v1 changed
              everything by releasing weights openly.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Researcher Lineage">
            Robin Rombach and team created Stable Diffusion at
            CompVis/Stability AI. They later left to found Black Forest Labs
            (BFL) and create Flux. SD and Flux share DNA because the{' '}
            <strong>same people</strong> made them.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <TimelineDiagram />
          <div className="mt-4 space-y-4">
            <p className="text-muted-foreground">
              Key inflection points on this timeline:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Aug 2022:</strong> SD v1 launches&mdash;open weight
                text-to-image becomes accessible on consumer GPUs
              </li>
              <li>
                <strong>Jul 2023:</strong> SDXL pushes the U-Net to its
                practical limits (&ldquo;the U-Net&rsquo;s last stand&rdquo;)
              </li>
              <li>
                <strong>Nov 2023&ndash;Feb 2024:</strong> PixArt-alpha and SD3
                independently adopt DiT/MMDiT&mdash;the transformer pivot
              </li>
              <li>
                <strong>Aug 2024:</strong> Flux.1 from the original SD
                creators&mdash;the lineage continues at BFL
              </li>
              <li>
                <strong>Jan 2025:</strong> Z-Image matches Flux with 1/5 the
                parameters via S3-DiT single-stream
              </li>
            </ul>
          </div>
        </Row.Content>
      </Row>

      {/* Parallel DiT convergence */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The parallel DiT convergence
            </p>
            <p className="text-muted-foreground">
              The shift from U-Net to DiT was not a single event. PixArt-alpha
              (Huawei, Nov 2023), SD3 (Stability AI, Feb 2024), and
              Hunyuan-DiT (Tencent, May 2024) independently adopted DiT-based
              architectures. Three labs, three continents, converging on the
              same solution because the underlying idea&rsquo;s
              merit&mdash;&ldquo;two knobs not twenty&rdquo; scaling
              recipe&mdash;drove the transition.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Single Pivot">
            The U-Net &rarr; DiT shift was a <strong>gradual
            convergence</strong> across multiple labs, not a single switch.
            The Peebles & Xie DiT paper (Dec 2022, ICCV 2023) preceded all
            of them. Multiple teams converged because transformers scale
            systematically while U-Nets do not.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================== */}
      {/* Section 5: Architecture Taxonomy                                   */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Architecture Taxonomy"
            subtitle="Five backbone families, one organizing principle"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every open weight image generation model falls into one of five
              backbone families. Think of it as a family tree: models have
              parents, siblings, and cousins&mdash;they share DNA when the
              same researchers or the same architectural ideas produced them.
              Once you identify the backbone, most of the mystery dissolves.
              The families are defined by{' '}
              <strong>how the denoising network processes image
              features</strong> and <strong>how it incorporates text
              conditioning</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Few Distinct Architectures">
            PixArt-alpha, SD3, Hunyuan-DiT, Flux, and AuraFlow sound
            completely different. But they are all DiT-family architectures
            with variations. The number of truly distinct paradigms is small.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <TaxonomyDiagram />
        </Row.Content>
      </Row>

      {/* U-Net family */}
      <Row>
        <Row.Content>
          <div className="space-y-3">
            <GradientCard title="U-Net Family" color="amber">
              <div className="space-y-2">
                <p>
                  <strong>Defining feature:</strong> Encoder-decoder U-Net with
                  skip connections for denoising. Text injected via
                  cross-attention at specific resolutions.
                </p>
                <p>
                  <strong>Mental model:</strong> &ldquo;The U-Net&rsquo;s last
                  stand&rdquo; from{' '}
                  <LessonLink slug="sdxl">SDXL</LessonLink>.
                </p>
                <ul className="space-y-1">
                  <li>&bull; <strong>SD v1.x:</strong> The original. CLIP ViT-L text encoder, 512&times;512, ~860M U-Net params.</li>
                  <li>&bull; <strong>SD v2.x:</strong> Switched to OpenCLIP ViT-H, v-prediction. Better architecture, but broke ecosystem compatibility. Community largely rejected it.</li>
                  <li>&bull; <strong>SDXL:</strong> Dual CLIP encoders, micro-conditioning, 1024&times;1024, ~3.5B U-Net. The ceiling of what U-Nets can do.</li>
                  <li>&bull; <strong>Playground v2.5:</strong> SDXL architecture, different training recipe. Proved training recipe matters as much as architecture.</li>
                  <li>&bull; <strong>Kolors:</strong> SDXL-like U-Net with ChatGLM-6B (Chinese LLM) as text encoder. LLM-as-encoder before Z-Image.</li>
                  <li>&bull; <strong>Kandinsky 2.x:</strong> U-Net with unCLIP approach (CLIP image prior). Different conditioning strategy from SD.</li>
                </ul>
              </div>
            </GradientCard>

            <GradientCard title="Pixel-Space / Cascaded" color="rose">
              <div className="space-y-2">
                <p>
                  <strong>Defining feature:</strong> Diffusion operates in
                  pixel space or uses an unusual latent compression paradigm,
                  often with cascaded upsampling stages.
                </p>
                <ul className="space-y-1">
                  <li>&bull; <strong>DeepFloyd IF:</strong> Pixel-space diffusion with T5-XXL. Cascaded 64&times;64 &rarr; 256&times;256 &rarr; 1024&times;1024. First major use of T5 for diffusion. Excellent text rendering for its era.</li>
                  <li>&bull; <strong>Stable Cascade:</strong> Three-stage cascade with extreme 42:1 latent compression (vs SD&rsquo;s 8:1). Architecturally interesting but did not gain wide adoption.</li>
                </ul>
              </div>
            </GradientCard>

            <GradientCard title="DiT Family (Cross-Attention)" color="cyan">
              <div className="space-y-2">
                <p>
                  <strong>Defining feature:</strong> Transformer backbone with
                  patchify. Text injected via cross-attention (text as K/V,
                  image as Q).
                </p>
                <p>
                  <strong>Mental model:</strong> &ldquo;Tokenize the
                  image&rdquo; from{' '}
                  <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>.
                </p>
                <ul className="space-y-1">
                  <li>&bull; <strong>PixArt-alpha:</strong> First major open weight DiT for text-to-image. T5-XXL text encoder. Competitive quality at ~10% of SD&rsquo;s training cost. ~600M params.</li>
                  <li>&bull; <strong>PixArt-sigma:</strong> Improved VAE, 4K resolution support. Demonstrated DiT scales to very high resolutions.</li>
                  <li>&bull; <strong>Hunyuan-DiT:</strong> First major bilingual (Chinese-English) open weight model. Bilingual CLIP + bilingual T5 text encoders. ~1.5B params.</li>
                </ul>
              </div>
            </GradientCard>

            <GradientCard title="MMDiT Family (Joint Attention)" color="violet">
              <div className="space-y-2">
                <p>
                  <strong>Defining feature:</strong> Text and image tokens
                  concatenated into one sequence for joint self-attention.
                  Modality-specific projections and FFNs (dual-stream).
                </p>
                <p>
                  <strong>Mental model:</strong> &ldquo;One room, one
                  conversation&rdquo; from{' '}
                  <LessonLink slug="sd3-and-flux">SD3 & Flux</LessonLink>.
                </p>
                <ul className="space-y-1">
                  <li>&bull; <strong>SD3 / SD3.5:</strong> Triple text encoders (CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL), flow matching training. SD3.5 Large: 8B params. SD3.5 Large Turbo: distilled for 4-step generation.</li>
                  <li>&bull; <strong>Flux.1:</strong> Hybrid double-stream + single-stream blocks (19 double-stream then 38 single-stream in Dev). CLIP ViT-L + T5-XXL (dropped second CLIP). Rotary embeddings. Schnell variant with guidance distillation for 1&ndash;4 steps. ~12B params.</li>
                  <li>&bull; <strong>AuraFlow:</strong> MMDiT-style with fully open training pipeline (code, data curation, training scripts). Community-driven. ~6.8B params.</li>
                </ul>
              </div>
            </GradientCard>

            <GradientCard title="S3-DiT (Single-Stream)" color="emerald">
              <div className="space-y-2">
                <p>
                  <strong>Defining feature:</strong> Fully single-stream
                  transformer. Shared projections and FFN across all token
                  types. Lightweight refiner layers for modality-specific
                  pre-processing.
                </p>
                <p>
                  <strong>Mental model:</strong> &ldquo;Translate once, then
                  speak together&rdquo; from{' '}
                  <LessonLink slug="z-image">Z-Image</LessonLink>.
                </p>
                <ul className="space-y-1">
                  <li>&bull; <strong>Z-Image:</strong> Qwen3-4B single LLM text encoder, 3D Unified RoPE, DMDR post-training. Competitive with 32B Flux at 6.15B params. Z-Image Turbo: 8 steps, sub-second.</li>
                </ul>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 6: Three Evolution Lines                                   */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Evolution Lines"
            subtitle="Architecture, text encoding, and training objective evolved somewhat independently"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A model&rsquo;s position in the landscape is defined by its
              choices on three independent axes. Each axis evolved on its own
              timeline, driven by different limitations.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Axes, One Model">
            Any model is a point in a 3D space: backbone &times; text encoder
            &times; training objective. The combinatorial space is small. New
            models pick a point in this space.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Backbone evolution */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Evolution Line 1: Denoising Backbone
            </p>
            <EvolutionLine steps={[
              { label: 'U-Net (SD v1, 2022)', color: '#f59e0b' },
              { label: 'Larger U-Net (SDXL, 2023)', color: '#f59e0b' },
              { label: 'DiT (PixArt-alpha, 2023)', color: '#06b6d4' },
              { label: 'MMDiT (SD3/Flux, 2024)', color: '#8b5cf6' },
              { label: 'S3-DiT (Z-Image, 2025)', color: '#22c55e' },
            ]} />
            <p className="text-muted-foreground">
              Each step was motivated by limitations of the previous:
              U-Net scaling is ad hoc (&ldquo;the U-Net&rsquo;s last
              stand&rdquo;). DiT brings the transformer scaling recipe
              (&ldquo;two knobs not twenty&rdquo;). MMDiT adds bidirectional
              text-image interaction (&ldquo;one room, one conversation&rdquo;).
              S3-DiT eliminates parameter duplication (&ldquo;translate once,
              then speak together&rdquo;).
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Text encoder evolution */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Evolution Line 2: Text Encoding
            </p>
            <EvolutionLine steps={[
              { label: 'CLIP ViT-L (SD v1)', color: '#f59e0b' },
              { label: 'CLIP + OpenCLIP (SDXL)', color: '#f59e0b' },
              { label: 'CLIP + OpenCLIP + T5-XXL (SD3)', color: '#8b5cf6' },
              { label: 'CLIP + T5-XXL (Flux)', color: '#8b5cf6' },
              { label: 'Single LLM (Z-Image: Qwen3-4B)', color: '#22c55e' },
            ]} />
            <p className="text-muted-foreground">
              The trajectory: from vision-language models (CLIP) to language
              models (T5) to full LLMs (Qwen3). And from one encoder to
              multiple specialized encoders&mdash;then back to one powerful
              encoder. DeepFloyd IF pioneered the T5 approach. Kolors
              pioneered LLM-as-encoder with ChatGLM. Z-Image completed the
              trajectory with Qwen3-4B.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Convergence to Simplicity">
            SD v1.5: 1 encoder. SDXL: 2 encoders. SD3: 3 encoders. Z-Image:
            back to 1. The trajectory is not &ldquo;more encoders =
            better&rdquo; but &ldquo;one powerful enough encoder replaces
            all.&rdquo;
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Training objective evolution */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Evolution Line 3: Training Objective
            </p>
            <EvolutionLine steps={[
              { label: 'DDPM epsilon-prediction (SD v1)', color: '#f59e0b' },
              { label: 'v-prediction (SD v2)', color: '#f59e0b' },
              { label: 'Flow matching (SD3, Flux, Z-Image)', color: '#8b5cf6' },
              { label: '+ Distillation (LCM, Turbo, Schnell)', color: '#06b6d4' },
              { label: '+ RL post-training (DMDR)', color: '#22c55e' },
            ]} />
            <p className="text-muted-foreground">
              The training objective evolved from predicting noise
              (epsilon-prediction) to predicting velocity (v-prediction) to
              flow matching (straight-line interpolation). Acceleration
              techniques layered on top: consistency distillation, guidance
              distillation (Flux Schnell), and DMD/DMDR (Z-Image Turbo).
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 7: Check #1 -- Taxonomy Placement                          */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Taxonomy Placement"
            subtitle="Can you identify the model from its description?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For each description below, commit to a prediction before
              clicking Reveal. Identify the <strong>backbone
              family</strong> (U-Net, DiT, MMDiT, S3-DiT, or
              pixel-space/cascaded) and the <strong>text encoder
              strategy</strong> first, then name the specific model. The
              descriptions use only information from the taxonomy
              above.
            </p>
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Description:</strong> A model that concatenates text
                  and image tokens into one sequence for joint self-attention,
                  uses CLIP + T5-XXL for text encoding, and trains with flow
                  matching.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>Flux.1</strong> (or SD3 family). Joint attention
                    = MMDiT family. CLIP + T5 is Flux&rsquo;s text encoder
                    setup (SD3 uses CLIP + OpenCLIP + T5). Flow matching is
                    common to both.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Description:</strong> A model using a U-Net backbone
                  with a single Chinese LLM (6B parameters) as its text
                  encoder, generating 1024&times;1024 images.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>Kolors</strong> (Kuaishou/KWAI). SDXL-like U-Net
                    backbone + ChatGLM-6B as text encoder. Interesting as a
                    U-Net model with LLM text encoding&mdash;combining the
                    old backbone paradigm with the new text encoding approach.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Description:</strong> A model that runs diffusion in
                  pixel space at 64&times;64 base resolution, upsamples with
                  two cascaded models, and uses T5-XXL for text encoding.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>DeepFloyd IF</strong> (Stability AI / DeepFloyd).
                    The pixel-space + cascaded + T5-XXL combination is unique
                    to IF. It pioneered T5 for diffusion text encoding, which
                    influenced SD3&rsquo;s adoption of T5.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 4" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Description:</strong> A single-stream transformer
                  with shared projections across all token types, a single
                  4B-parameter LLM as text encoder, and 3D positional
                  encoding.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>Z-Image</strong> (Freepik / Tongyi Lab). Single-stream
                    = S3-DiT. Single LLM = Qwen3-4B. 3D positional encoding
                    = 3D Unified RoPE. All the clues point to the same model.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 8: The Comparison Table (Centerpiece)                      */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Comparison Table"
            subtitle="Every major open weight model in one reference card"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This table is the centerpiece of the lesson. One row per model,
              chronological order. Each row is one sentence&mdash;no deep
              dives, just identification. Use it as a reference card you can
              return to when encountering any model. The left border color
              matches the backbone family from the taxonomy above:{' '}
              <span className="text-amber-500">amber</span> for U-Net,{' '}
              <span className="text-rose-500">rose</span> for pixel-space/cascaded,{' '}
              <span className="text-cyan-500">cyan</span> for DiT,{' '}
              <span className="text-violet-500">violet</span> for MMDiT, and{' '}
              <span className="text-emerald-500">emerald</span> for S3-DiT.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">Model</th>
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">Release</th>
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">Lab</th>
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">Backbone</th>
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">Text Encoder(s)</th>
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">Training Obj.</th>
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">Base Res.</th>
                    <th className="text-left p-2 font-semibold text-foreground whitespace-nowrap">~Params</th>
                    <th className="text-left p-2 font-semibold text-foreground">Key Innovation</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b border-border/50 border-l-2 border-l-amber-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">SD v1.x</td>
                    <td className="p-2 whitespace-nowrap">Aug 2022</td>
                    <td className="p-2 whitespace-nowrap">Stability AI / CompVis</td>
                    <td className="p-2 whitespace-nowrap">U-Net</td>
                    <td className="p-2 whitespace-nowrap">CLIP ViT-L</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">512</td>
                    <td className="p-2 whitespace-nowrap">~1.1B</td>
                    <td className="p-2">First widely available open weight text-to-image model</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-amber-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">SD v2.x</td>
                    <td className="p-2 whitespace-nowrap">Nov 2022</td>
                    <td className="p-2 whitespace-nowrap">Stability AI</td>
                    <td className="p-2 whitespace-nowrap">U-Net</td>
                    <td className="p-2 whitespace-nowrap">OpenCLIP ViT-H</td>
                    <td className="p-2 whitespace-nowrap">v-prediction</td>
                    <td className="p-2 whitespace-nowrap">512/768</td>
                    <td className="p-2 whitespace-nowrap">~1.3B</td>
                    <td className="p-2">v-prediction parameterization; cautionary tale&mdash;broke ecosystem compatibility</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-rose-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">DeepFloyd IF</td>
                    <td className="p-2 whitespace-nowrap">Apr 2023</td>
                    <td className="p-2 whitespace-nowrap">Stability AI / DeepFloyd</td>
                    <td className="p-2 whitespace-nowrap">Pixel-space (cascaded)</td>
                    <td className="p-2 whitespace-nowrap">T5-XXL</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">64</td>
                    <td className="p-2 whitespace-nowrap">~4.3B</td>
                    <td className="p-2">First major use of T5 for diffusion; excellent text rendering</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-amber-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">Kandinsky 2.x</td>
                    <td className="p-2 whitespace-nowrap">Apr 2023</td>
                    <td className="p-2 whitespace-nowrap">Sber AI</td>
                    <td className="p-2 whitespace-nowrap">U-Net (unCLIP)</td>
                    <td className="p-2 whitespace-nowrap">CLIP image prior</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">512/1024</td>
                    <td className="p-2 whitespace-nowrap">~3.3B</td>
                    <td className="p-2">unCLIP approach (DALL-E 2 style) in open weight form</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-amber-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">SDXL</td>
                    <td className="p-2 whitespace-nowrap">Jul 2023</td>
                    <td className="p-2 whitespace-nowrap">Stability AI</td>
                    <td className="p-2 whitespace-nowrap">U-Net (larger)</td>
                    <td className="p-2 whitespace-nowrap">CLIP ViT-L + OpenCLIP ViT-bigG</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">1024</td>
                    <td className="p-2 whitespace-nowrap">~3.5B</td>
                    <td className="p-2">&ldquo;The U-Net&rsquo;s last stand&rdquo;&mdash;pushed U-Net to its practical limits</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-cyan-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">PixArt-alpha</td>
                    <td className="p-2 whitespace-nowrap">Nov 2023</td>
                    <td className="p-2 whitespace-nowrap">Huawei</td>
                    <td className="p-2 whitespace-nowrap">DiT (cross-attn)</td>
                    <td className="p-2 whitespace-nowrap">T5-XXL</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">512/1024</td>
                    <td className="p-2 whitespace-nowrap">~600M</td>
                    <td className="p-2">First major open weight DiT; 10% of SD training cost via decomposed training</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-rose-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">Stable Cascade</td>
                    <td className="p-2 whitespace-nowrap">Feb 2024</td>
                    <td className="p-2 whitespace-nowrap">Stability AI</td>
                    <td className="p-2 whitespace-nowrap">Cascaded latent</td>
                    <td className="p-2 whitespace-nowrap">CLIP</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">24 (latent)</td>
                    <td className="p-2 whitespace-nowrap">~3.6B</td>
                    <td className="p-2">Extreme 42:1 latent compression; fast training but complex inference</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-amber-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">Playground v2.5</td>
                    <td className="p-2 whitespace-nowrap">Feb 2024</td>
                    <td className="p-2 whitespace-nowrap">Playground AI</td>
                    <td className="p-2 whitespace-nowrap">U-Net (SDXL arch)</td>
                    <td className="p-2 whitespace-nowrap">CLIP ViT-L + OpenCLIP ViT-bigG</td>
                    <td className="p-2 whitespace-nowrap">EDM</td>
                    <td className="p-2 whitespace-nowrap">1024</td>
                    <td className="p-2 whitespace-nowrap">~3.5B</td>
                    <td className="p-2">Proved training recipe beats architecture (same SDXL arch, better results)</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-violet-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">SD3</td>
                    <td className="p-2 whitespace-nowrap">Jun 2024</td>
                    <td className="p-2 whitespace-nowrap">Stability AI</td>
                    <td className="p-2 whitespace-nowrap">MMDiT</td>
                    <td className="p-2 whitespace-nowrap">CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL</td>
                    <td className="p-2 whitespace-nowrap">Flow matching</td>
                    <td className="p-2 whitespace-nowrap">1024</td>
                    <td className="p-2 whitespace-nowrap">2&ndash;8B</td>
                    <td className="p-2">MMDiT joint attention; triple text encoders; flow matching training</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-cyan-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">PixArt-sigma</td>
                    <td className="p-2 whitespace-nowrap">Apr 2024</td>
                    <td className="p-2 whitespace-nowrap">Huawei</td>
                    <td className="p-2 whitespace-nowrap">DiT (cross-attn)</td>
                    <td className="p-2 whitespace-nowrap">T5-XXL</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">up to 4K</td>
                    <td className="p-2 whitespace-nowrap">~600M</td>
                    <td className="p-2">4K resolution support; improved VAE; DiT scales to very high resolutions</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-cyan-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">Hunyuan-DiT</td>
                    <td className="p-2 whitespace-nowrap">May 2024</td>
                    <td className="p-2 whitespace-nowrap">Tencent</td>
                    <td className="p-2 whitespace-nowrap">DiT (cross-attn)</td>
                    <td className="p-2 whitespace-nowrap">Bilingual CLIP + bilingual T5</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">1024</td>
                    <td className="p-2 whitespace-nowrap">~1.5B</td>
                    <td className="p-2">First major bilingual (Chinese-English) open weight model</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-violet-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">AuraFlow</td>
                    <td className="p-2 whitespace-nowrap">Jun 2024</td>
                    <td className="p-2 whitespace-nowrap">Fal.ai</td>
                    <td className="p-2 whitespace-nowrap">MMDiT-style</td>
                    <td className="p-2 whitespace-nowrap">CLIP + T5</td>
                    <td className="p-2 whitespace-nowrap">Flow matching</td>
                    <td className="p-2 whitespace-nowrap">1024</td>
                    <td className="p-2 whitespace-nowrap">~6.8B</td>
                    <td className="p-2">Fully open training pipeline; community-driven</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-amber-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">Kolors</td>
                    <td className="p-2 whitespace-nowrap">Jul 2024</td>
                    <td className="p-2 whitespace-nowrap">Kuaishou/KWAI</td>
                    <td className="p-2 whitespace-nowrap">U-Net (SDXL-like)</td>
                    <td className="p-2 whitespace-nowrap">ChatGLM-6B (LLM)</td>
                    <td className="p-2 whitespace-nowrap">DDPM</td>
                    <td className="p-2 whitespace-nowrap">1024</td>
                    <td className="p-2 whitespace-nowrap">~3.5B</td>
                    <td className="p-2">LLM as text encoder (ChatGLM) before Z-Image; strong bilingual support</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-violet-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">Flux.1 Dev</td>
                    <td className="p-2 whitespace-nowrap">Aug 2024</td>
                    <td className="p-2 whitespace-nowrap">Black Forest Labs</td>
                    <td className="p-2 whitespace-nowrap">MMDiT (hybrid)</td>
                    <td className="p-2 whitespace-nowrap">CLIP ViT-L + T5-XXL</td>
                    <td className="p-2 whitespace-nowrap">Flow matching</td>
                    <td className="p-2 whitespace-nowrap">1024+</td>
                    <td className="p-2 whitespace-nowrap">~12B</td>
                    <td className="p-2">Hybrid double/single-stream blocks; RoPE; Schnell: guidance distillation</td>
                  </tr>
                  <tr className="border-b border-border/50 border-l-2 border-l-violet-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">SD3.5</td>
                    <td className="p-2 whitespace-nowrap">Oct 2024</td>
                    <td className="p-2 whitespace-nowrap">Stability AI</td>
                    <td className="p-2 whitespace-nowrap">MMDiT</td>
                    <td className="p-2 whitespace-nowrap">CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL</td>
                    <td className="p-2 whitespace-nowrap">Flow matching</td>
                    <td className="p-2 whitespace-nowrap">1024</td>
                    <td className="p-2 whitespace-nowrap">2.5&ndash;8B</td>
                    <td className="p-2">Medium/Large/Large Turbo variants; improved training over SD3</td>
                  </tr>
                  <tr className="border-l-2 border-l-emerald-500/70">
                    <td className="p-2 font-medium text-foreground whitespace-nowrap">Z-Image</td>
                    <td className="p-2 whitespace-nowrap">Jan 2025</td>
                    <td className="p-2 whitespace-nowrap">Freepik / Tongyi Lab</td>
                    <td className="p-2 whitespace-nowrap">S3-DiT</td>
                    <td className="p-2 whitespace-nowrap">Qwen3-4B (LLM)</td>
                    <td className="p-2 whitespace-nowrap">Flow matching</td>
                    <td className="p-2 whitespace-nowrap">1024+</td>
                    <td className="p-2 whitespace-nowrap">~6.15B</td>
                    <td className="p-2">Single-stream with refiner layers; single LLM encoder; DMDR post-training</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 9: Notable Innovations by Model                            */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Notable Innovations by Model"
            subtitle="The non-obvious contributions that shaped the field"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Some models contributed innovations that transcended their own
              adoption. These are the ideas that propagated across the
              landscape, often appearing in models from different labs.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <GradientCard title="SD v2: v-prediction" color="amber">
              <p>
                Instead of predicting the noise (epsilon-prediction), SD v2
                predicts the &ldquo;velocity&rdquo;&mdash;a different
                parameterization of the denoising target. v-prediction works
                better at high noise levels and was a stepping stone toward
                flow matching. But the switch to OpenCLIP ViT-H broke
                compatibility with the v1.x ecosystem (all existing LoRAs,
                ControlNets, and community models stopped working), and the
                community largely rejected it. <strong>Better architecture does
                not guarantee adoption if it breaks ecosystem
                compatibility.</strong>
              </p>
            </GradientCard>

            <GradientCard title="DeepFloyd IF: T5 for Diffusion" color="blue">
              <p>
                DeepFloyd IF proved that T5-XXL produces dramatically better
                text understanding than CLIP for diffusion conditioning.
                Excellent text rendering for its era. The pixel-space cascaded
                approach (64&times;64 base) was too expensive to scale, but
                the <strong>T5 insight was hugely influential</strong>&mdash;it
                directly led to SD3&rsquo;s adoption of T5 as the third text
                encoder.
              </p>
            </GradientCard>

            <p className="text-muted-foreground text-sm italic">
              DeepFloyd IF&rsquo;s T5 insight spread across the field&mdash;SD3,
              Flux, and eventually Z-Image all benefited from the proof that
              language models understand prompts better than CLIP. But another
              idea propagated just as widely: that you do not need massive
              compute if you structure training intelligently.
            </p>

            <GradientCard title="PixArt-alpha: Training Efficiency" color="cyan">
              <p>
                Achieved competitive quality at ~10% of SD&rsquo;s training
                cost through decomposed training: learn image distribution
                first (class-conditional ImageNet pre-training), then learn
                text-image alignment (text-conditioned fine-tuning). This
                demonstrated that you do not need massive compute if you{' '}
                <strong>structure the training curriculum
                intelligently</strong>.
              </p>
            </GradientCard>

            <GradientCard title="Playground v2.5: Training Recipe > Architecture" color="orange">
              <p>
                Used the exact same SDXL architecture but achieved reportedly
                better aesthetic quality through curated data and the EDM
                (Karras et al.) training framework. The insight:{' '}
                <strong>you can get significant quality gains without changing
                the architecture</strong>. Data curation and training procedure
                matter as much as the model itself.
              </p>
            </GradientCard>

            <GradientCard title="Stable Cascade: Extreme Latent Compression" color="purple">
              <p>
                Compressed images to a 24&times;24 latent space (42:1 spatial
                ratio vs SD&rsquo;s 8:1). Trained in 24K A100 hours (vs 200K+
                for SD). Proved that extreme compression is viable but the
                cascaded inference pipeline added complexity that limited
                adoption. An interesting architectural experiment that did not
                become the dominant paradigm.
              </p>
            </GradientCard>

            <GradientCard title="Kolors: LLM as Text Encoder (Before Z-Image)" color="sky">
              <p>
                Used ChatGLM-6B (a Chinese LLM) as text encoder with an
                SDXL-like U-Net backbone. Demonstrated that general-purpose
                LLMs can replace CLIP for conditioning&mdash;foreshadowing
                Z-Image&rsquo;s Qwen3-4B approach. Strong bilingual
                Chinese-English support.
              </p>
            </GradientCard>

            <p className="text-muted-foreground text-sm italic">
              Kolors foreshadowed the LLM-as-encoder trajectory that Z-Image
              would complete. Meanwhile, the acceleration problem&mdash;how to
              generate in fewer steps&mdash;was being solved from a different
              angle entirely.
            </p>

            <GradientCard title="Flux Schnell: Guidance Distillation" color="violet">
              <p>
                Achieved 1&ndash;4 step generation via guidance distillation
                (distilling the CFG-guided model into a model that generates
                high-quality images without needing CFG at inference). Released
                under Apache 2.0 (the most permissive license among frontier
                models). The Dev variant (~12B params) set the quality
                benchmark for open weight models through late 2024.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 10: Check #2 -- Innovation Attribution                     */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Innovation Attribution"
            subtitle="Can you trace which model pioneered which idea?"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Which model first demonstrated that T5 could work as a text
                  encoder for diffusion?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>DeepFloyd IF</strong> (April 2023). Google&rsquo;s
                    Imagen (May 2022) also used T5-XXL but was never released.
                    DeepFloyd IF was the first open weight model to use T5 and
                    prove its superiority for text understanding in diffusion.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  What was Playground v2.5&rsquo;s key insight, given that it
                  used the same architecture as SDXL?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>Training recipe matters as much as
                    architecture.</strong> Same SDXL architecture, different
                    training: aesthetic-focused dataset, EDM framework (Karras
                    et al.), modified noise schedule. Significant quality gains
                    without any architectural change.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Why did SD v2.x fail to replace SD v1.5 in the community,
                  despite being technically improved?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>It broke ecosystem compatibility.</strong> The
                    switch from CLIP ViT-L to OpenCLIP ViT-H meant that every
                    existing LoRA, ControlNet model, and community fine-tune
                    built for v1.x stopped working with v2.x. SD v1.5 has
                    vastly more community tooling, and for fine-tuning
                    workflows, ecosystem maturity can matter more than raw
                    architectural quality.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Newer Is Not Always Better">
            SD v1.5 has thousands of LoRAs, dozens of ControlNet models, and
            massive community support. SD3 has fewer despite being
            architecturally superior. For practical use, ecosystem maturity
            can matter more than architecture quality.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================== */}
      {/* Section 11: The Closed Models (Brief Context)                      */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Closed Models"
            subtitle="Historical context that shaped the open weight ecosystem"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Several closed models set benchmarks and pioneered techniques
              that the open weight models adopted. Understanding them
              explains <em>why</em> certain open models made specific choices.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Why Mention Closed Models?">
            DALL-E pioneered approaches that Kandinsky followed (unCLIP).
            Imagen pioneered T5 that DeepFloyd IF and SD3 adopted.
            Midjourney set the quality bar everyone chases. Understanding
            lineage requires acknowledging closed models.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="DALL-E Lineage (OpenAI)" color="blue">
                <ul className="space-y-1">
                  <li>&bull; <strong>DALL-E 1 (2021):</strong> Autoregressive (discrete VAE + GPT). Proved text-to-image works.</li>
                  <li>&bull; <strong>DALL-E 2 (2022):</strong> Diffusion-based (unCLIP approach). Kandinsky follows this.</li>
                  <li>&bull; <strong>DALL-E 3 (2023):</strong> Improved prompt following via synthetic captions. Integrated into ChatGPT.</li>
                </ul>
              </GradientCard>
              <GradientCard title="Imagen & Midjourney" color="violet">
                <ul className="space-y-1">
                  <li>&bull; <strong>Imagen (Google, 2022):</strong> Pixel-space cascaded diffusion with T5-XXL. Never released. Demonstrated T5&rsquo;s value&mdash;influenced DeepFloyd IF and SD3.</li>
                  <li>&bull; <strong>Midjourney:</strong> Architecture undisclosed. Dominant in aesthetic quality. Discord-based. Important benchmark but provides no architectural learning.</li>
                </ul>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Note: <strong>Flux 2</strong> (BFL, 2025) is API-only, not open
              weight. The Flux lineage moved from open (Flux.1) to closed
              (Flux 2). Its variants (Max, Pro, Flex, Klein) have not released
              weights.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Open Weight ≠ Open Source">
            &ldquo;Open weight&rdquo; means you can download and run the
            model weights. It does <strong>not</strong> mean open source.
            True open source includes training code, data, and full
            reproducibility. Compare Flux.1 Dev (weights available, but
            non-commercial license) with Flux.1 Schnell (Apache 2.0,
            genuinely permissive). Both are &ldquo;open weight&rdquo; but
            with very different terms. When evaluating a model, check
            the license&mdash;not just the download button.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================== */}
      {/* Section 12: Video Generation (Brief Mention)                       */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Video Generation"
            subtitle="Brief mention for completeness"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Wan 2.1</strong> (Alibaba/Tongyi Wanxiang, Feb 2025) is
              primarily a video generation model using a 3D VAE + DiT
              backbone. Video generation shares architectural DNA with image
              generation (DiT backbone, flow matching) but adds temporal
              modeling&mdash;it is a different domain that deserves its own
              treatment. Mentioned here for completeness, but clearly
              distinguished: this lesson covers image generation, not video.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 13: How to Read New Model Announcements                    */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="How to Read New Model Announcements"
            subtitle="The taxonomy as a tool for navigating future releases"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              When a new model drops, ask five questions. These five questions
              place any model on the map:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-3 ml-4">
              <li>
                <strong>What is the denoising backbone?</strong> U-Net / DiT /
                MMDiT / S3-DiT / something new? This immediately tells you the
                family.
              </li>
              <li>
                <strong>What text encoder(s)?</strong> CLIP / T5 / LLM /
                multi-encoder? Text encoding quality determines prompt
                understanding and compositional reasoning.
              </li>
              <li>
                <strong>What training objective?</strong> DDPM /
                v-prediction / flow matching? This determines inference step
                count and what acceleration techniques are compatible.
              </li>
              <li>
                <strong>What is the claimed innovation?</strong> Architecture /
                training recipe / data curation / post-training / scale? Most
                innovations fall into one of these categories.
              </li>
              <li>
                <strong>What is the lineage?</strong> Which lab, which prior
                models, which researchers? Lineage predicts architectural
                choices.
              </li>
            </ol>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Teach a Person to Fish">
            The comparison table is a snapshot. The five questions are the
            tool. New models will appear after this lesson. The framework
            lets you place them immediately without waiting for someone to
            update the table.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================== */}
      {/* Section 14: Check #3 -- Apply the Framework                        */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Apply the Framework"
            subtitle="Place a hypothetical model on the map"
          />
          <div className="space-y-4">
            <GradientCard title="Transfer Question" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Hypothetical announcement:</strong> &ldquo;Lab X
                  releases ModelY, a 10B parameter model using a transformer
                  backbone with cross-attention text conditioning, trained
                  with rectified flow on 2B image-text pairs, using Llama 3
                  as the text encoder.&rdquo;
                </p>
                <p>Place this model in the taxonomy using the five questions.</p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <div className="mt-2 text-muted-foreground space-y-2">
                    <ol className="list-decimal list-inside space-y-1 ml-2">
                      <li>
                        <strong>Backbone:</strong> DiT family (transformer with
                        cross-attention, not joint attention&mdash;so DiT, not
                        MMDiT).
                      </li>
                      <li>
                        <strong>Text encoder:</strong> Single LLM (Llama 3).
                        Same text encoding strategy as Z-Image/Kolors.
                      </li>
                      <li>
                        <strong>Training:</strong> Rectified flow (same family
                        as SD3/Flux/Z-Image).
                      </li>
                      <li>
                        <strong>Innovation:</strong> Combining DiT backbone
                        with LLM text encoding and flow matching&mdash;a novel
                        combination of existing approaches.
                      </li>
                      <li>
                        <strong>Closest comparison:</strong> PixArt-sigma with
                        an LLM text encoder and flow matching training.
                      </li>
                    </ol>
                    <p>
                      The taxonomy immediately reveals this model&rsquo;s
                      position: DiT family, LLM text encoding (Z-Image/Kolors
                      lineage), flow matching (SD3/Flux training lineage).
                    </p>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Real Test">
            The next time a model is announced on X or in a paper, try the
            five questions. You should be able to place it on the taxonomy
            within a minute of reading the abstract.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================== */}
      {/* Section 15: Summary                                                */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'The landscape has 4\u20135 backbone types, 3\u20134 text encoding strategies, and 3 training objectives.',
                description:
                  'U-Net, DiT (cross-attention), MMDiT (joint attention), S3-DiT (single-stream), and pixel-space/cascaded. CLIP, CLIP+OpenCLIP, CLIP+T5, and single LLM. DDPM, v-prediction, and flow matching. Every model is a combination of these.',
              },
              {
                headline: 'The evolution is driven by better scaling (U-Net \u2192 DiT), better text understanding (CLIP \u2192 LLM), and faster inference (DDPM \u2192 flow matching \u2192 distillation).',
                description:
                  'Three independent evolution lines converging on transformer backbones, LLM text encoders, and flow matching with distillation or RL post-training.',
              },
              {
                headline: 'Ecosystem matters as much as architecture.',
                description:
                  'SD v2.x was technically better than v1.5 but failed because it broke ecosystem compatibility. SD v1.5 still has more LoRAs and ControlNets than any newer model. Newer is not always better for practical use.',
              },
              {
                headline: 'Five questions place any model on the map.',
                description:
                  'Backbone type, text encoder(s), training objective, claimed innovation, and lineage. These five questions are the tool for navigating future releases.',
              },
              {
                headline: '"Convergence, not revolution."',
                description:
                  'Every model on this map is built from components you already understand. The landscape looks overwhelming until you realize the combinatorial space is small. Your deep knowledge makes it navigable.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>
              &ldquo;Convergence, not revolution.&rdquo; Every model on this
              map is built from components you already understand&mdash;U-Nets,
              DiTs, CLIP, T5, flow matching, CFG, LoRA, distillation. The
              field did not invent new physics. It combined transformers, flow
              matching, and better text encoders in different configurations.
              The taxonomy is the tool&mdash;new models slot into this
              framework.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* References                                                         */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
                authors: 'Rombach et al., 2022 (CompVis, Stability AI, Runway)',
                url: 'https://arxiv.org/abs/2112.10752',
                note: 'The original Stable Diffusion / Latent Diffusion Models paper. The foundation for the entire open weight ecosystem.',
              },
              {
                title: 'SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis',
                authors: 'Podell et al., 2023 (Stability AI)',
                url: 'https://arxiv.org/abs/2307.01952',
                note: 'The SDXL paper. Sections on micro-conditioning and dual text encoders are most relevant for the landscape.',
              },
              {
                title: 'Scaling Rectified Flow Transformers for High-Resolution Image Synthesis',
                authors: 'Esser et al., 2024 (Stability AI)',
                url: 'https://arxiv.org/abs/2403.03206',
                note: 'The SD3 paper. Introduces MMDiT and flow matching for text-to-image. The pivot from U-Net to transformer.',
              },
              {
                title: 'PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis',
                authors: 'Chen et al., 2023 (Huawei)',
                url: 'https://arxiv.org/abs/2310.00426',
                note: 'Demonstrated DiT for text-to-image with 10% of SD training cost. The decomposed training strategy is the key insight.',
              },
              {
                title: 'An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer',
                authors: 'Tongyi Lab (Alibaba), 2025',
                url: 'https://arxiv.org/abs/2511.22699',
                note: 'The Z-Image paper. S3-DiT single-stream architecture, Qwen3-4B text encoder, 3D RoPE.',
              },
              {
                title: 'Scalable Diffusion Models with Transformers',
                authors: 'Peebles & Xie, 2023 (UC Berkeley)',
                url: 'https://arxiv.org/abs/2212.09748',
                note: 'The original DiT paper. The idea that preceded PixArt-alpha, SD3, and every transformer-based diffusion model.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================== */}
      {/* Section 16: Next Step                                              */}
      {/* ================================================================== */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This landscape map is a living document&mdash;new models will
              emerge. But you now have the taxonomy to place them. The next
              time a model is announced, use the five questions. You will find
              that the model is a combination of things you already
              understand, positioned at a specific point in the backbone
              &times; text encoder &times; training objective space.
            </p>
            <p className="text-muted-foreground">
              For practical exploration: try running Flux.1 Schnell, SD3.5, and
              Z-Image Turbo side by side. Compare the results with your
              knowledge of their architectures. You will see the differences
              the taxonomy predicts&mdash;text rendering quality, compositional
              reasoning, aesthetic style&mdash;and understand <em>why</em> they
              differ.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Back to Dashboard"
            description="Review your learning journey and explore other topics."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
