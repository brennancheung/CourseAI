# Component Catalog

Quick reference for all interactive components available in lessons.

---

## Lesson Metadata

Lesson metadata lives in the curriculum tree (`src/data/curriculum/`). Each leaf node defines metadata inline:

**Type:** `src/data/curriculum/types.ts`

```typescript
// In src/data/curriculum/foundations.ts (or similar)
{
  slug: 'my-lesson',                      // URL slug — also the route
  title: 'My Lesson Title',
  description: 'One-line description of what you will learn.',
  duration: '20 min',
  category: 'Fundamentals',
  objectives: [
    'Understand concept A',
    'See why B matters',
    'Practice with interactive widget',
  ],
  skills: ['skill-tag-1', 'skill-tag-2'],
  prerequisites: ['previous-lesson-slug'],
  exercise: {
    constraints: [
      'Focus on intuition first',
      'No code yet — just concepts',
    ],
    steps: [
      'Understand concept A',
      'See why B matters',
    ],
  },
}
```

Then create `src/app/app/lesson/{slug}/page.tsx` to render the lesson component.

---

## Layout Blocks

Non-interactive but essential for lesson structure.

**Path:** `src/components/lessons/blocks.tsx`

### Main Content Blocks

Use these in the main content area (first child of `Row.Content`):

| Component | Purpose | Color | When to Use |
|-----------|---------|-------|-------------|
| `LessonHeader` | Title, description, badges | Neutral | Top of every lesson |
| `SectionHeader` | Section titles | Neutral | Start of each major section |
| `ObjectiveBlock` | Lesson goal | Primary | After header, states what they'll learn |
| `ConstraintBlock` | Scope boundaries | Amber | When limiting scope helps focus |
| `StepList` | Numbered instructions | Neutral | For procedural "do this" content |
| `SummaryBlock` | Key takeaways | Primary+Violet gradient | End of lesson or major section |
| `ReferencesBlock` | External references | Neutral card | After summary, before next step |
| `ModuleCompleteBlock` | Module celebration | Emerald gradient | End of module's final lesson |
| `NextStepBlock` | Session completion link | Primary gradient | End of lesson |

### Aside Blocks

Use these in `Row.Aside`:

| Component | Purpose | Color | When to Use |
|-----------|---------|-------|-------------|
| `TipBlock` | Helpful hints | Sky blue | Pro tips, shortcuts, recommendations |
| `WarningBlock` | Common mistakes | Rose | Pitfalls to avoid |
| `InsightBlock` | Key concepts | Violet | "Aha" moments, deeper understanding |
| `TryThisBlock` | Interactive prompts | Emerald | Hands-on experiments to try |
| `ConceptBlock` | Theory explanations | Neutral | Background theory, definitions |
| `ConceptCard` | Brief concept overview | Neutral | Multiple related concepts in a grid |
| `SidebarSection` | Groups sidebar content | Neutral | Wrap multiple blocks with a title |

### Gradient Cards (Main Content)

Visually prominent cards for categorization and comparison. Use in main content, not asides.

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `GradientCard` | Single categorized card | Feature highlights, categorized info |
| `ComparisonRow` | Two side-by-side cards | A vs B, before/after |
| `PhaseCard` | Numbered sequential card | Multi-step processes, timeline phases |

**Available Colors:** `amber` | `blue` | `cyan` | `orange` | `purple` | `emerald` | `rose` | `violet` | `sky`

**Color Semantics (suggested):**
- `amber` - Warning, caution
- `blue` - Primary, neutral info
- `cyan` - Build, anticipation
- `orange` - Important, key concept
- `purple` - Advanced, complex
- `emerald` - Success, practice
- `rose` - Error, avoid
- `violet` - Insight, theory

```tsx
// Single gradient card
<GradientCard title="Key Concept" color="blue">
  <ul className="space-y-1">
    <li>• Point one</li>
    <li>• Point two</li>
  </ul>
</GradientCard>

// Side-by-side comparison
<ComparisonRow
  left={{ title: 'Before', color: 'amber', items: ['Old way', 'Manual'] }}
  right={{ title: 'After', color: 'blue', items: ['New way', 'Automated'] }}
/>

// Sequential phases
<PhaseCard number={1} title="Step One" subtitle="Getting started" color="cyan">
  <p>First step description...</p>
</PhaseCard>
```

---

## Layout

### LessonLayout + Row
**Path:** `src/components/lessons/LessonLayout.tsx`, `src/components/layout/Row.tsx`

**CRITICAL:** Always use `Row` from `@/components/layout/Row`. Never use manual flex layouts.

```tsx
import { LessonLayout } from '@/components/lessons'
import { Row } from '@/components/layout/Row'

<LessonLayout>
  <Row>
    <Row.Content>
      <MainContent />
    </Row.Content>
    <Row.Aside>
      <TipBlock>...</TipBlock>
    </Row.Aside>
  </Row>
</LessonLayout>

// Conditional aside - Row auto-injects placeholder to maintain layout
<Row>
  <Row.Content><Card /></Row.Content>
  {isExpanded && <Row.Aside>tips</Row.Aside>}
</Row>
```

**Row dimensions:**
- Content: `flex-1` (takes remaining space)
- Aside: `lg:w-64` (always present, even if empty)
- Gap: `gap-8`

---

## Data File Patterns

### Technique/Item Lists

```typescript
// src/data/lessons/[feature].ts
export type Concept = {
  id: string
  name: string
  tagline: string
  description: string
  whyItMatters: string
  howToApply: string[]
  examples?: string[]
  pitfalls?: string[]
}
```

---

## Math Rendering (KaTeX)

Use KaTeX for LaTeX-style math formulas.

**Import:**
```tsx
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
```

**Usage:**
```tsx
// Inline math in text
<p>The loss is <InlineMath math="\hat{y} = wx + b" /> where...</p>

// Block math (centered, standalone)
<BlockMath math="L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2" />

// Styled block math
<div className="py-4 px-6 bg-muted/50 rounded-lg">
  <BlockMath math="\theta_{new} = \theta_{old} - \alpha \nabla L" />
</div>
```

**Common patterns:**
- Wrap important formulas in `bg-muted/50 rounded-lg` for emphasis
- Use `InlineMath` in list items to explain each variable
- Combine with `ConceptBlock` for formula explanations

---

## Mermaid Diagrams

Render static flowcharts, architecture diagrams, and process flows as inline SVG.

**Path:** `src/components/widgets/MermaidDiagram.tsx`

```tsx
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'

// Simple flowchart
<MermaidDiagram chart={`
  graph LR
    A[Input Layer] --> B[Hidden Layer]
    B --> C[Output Layer]
`} />

// Inside a lesson row
<Row>
  <Row.Content>
    <SectionHeader title="Network Architecture" />
    <MermaidDiagram chart={`
      graph TD
        X[Input x] --> N1[Neuron 1]
        X --> N2[Neuron 2]
        N1 --> O[Output]
        N2 --> O
    `} />
  </Row.Content>
  <Row.Aside>
    <ConceptBlock title="Architecture">Each arrow is a weighted connection.</ConceptBlock>
  </Row.Aside>
</Row>
```

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| `chart` | `string` | Mermaid diagram definition (any valid mermaid syntax) |
| `className` | `string?` | Optional CSS class for the container div |

**Supported diagram types:** flowchart, sequence, class, state, ER, gantt, pie, mindmap, timeline, and more.

**When to use vs react-konva:**
- **MermaidDiagram:** Static flowcharts, architecture diagrams, process flows, dependency graphs
- **react-konva:** Interactive visualizations where the user drags, clicks, or manipulates elements

**Agent verification:** Render to PNG with CLI before shipping:
```bash
pnpm dlx @mermaid-js/mermaid-cli -i diagram.mmd -o diagram.png -b transparent
```
Then Read the PNG to visually verify. Delete temp files after.

---

## ExercisePanel (Interactive Widget Container)

Compound component for wrapping interactive widgets with expand-to-fullscreen support.

**Path:** `src/components/widgets/ExercisePanel.tsx`

```tsx
import { ExercisePanel } from '@/components/widgets/ExercisePanel'

// Shorthand (most common)
<ExercisePanel title="Try Fitting a Line" subtitle="Drag the controls">
  <LinearFitExplorer />
</ExercisePanel>

// Compound form (if you need more control)
<ExercisePanel>
  <ExercisePanel.Header title="Explore the Loss Surface" />
  <ExercisePanel.Content>
    <LossSurfaceExplorer />
  </ExercisePanel.Content>
</ExercisePanel>
```

**Features:**
- Bordered panel with title header
- Expand button → fullscreen modal (ESC to close)
- Passes measured dimensions to child when expanded
- Children should accept optional `width` and `height` props

---

## Canvas Components

### ZoomableCanvas

Base component for Konva-based interactive visualizations.

**Path:** `src/components/canvas/ZoomableCanvas.tsx`

```tsx
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { Circle, Line, Text, Arrow, Rect } from 'react-konva'

<ZoomableCanvas width={600} height={350} backgroundColor="#1a1a2e">
  <Line points={[0, 100, 600, 100]} stroke="#666" strokeWidth={1} />
  <Circle x={300} y={175} radius={10} fill="#6366f1" />
  <Text x={10} y={10} text="Label" fontSize={12} fill="#888" />
</ZoomableCanvas>
```

### Canvas Primitives

**Path:** `src/components/canvas/primitives/`

| Primitive | Purpose |
|-----------|---------|
| `Grid` | Background grid lines |
| `Axis` | X/Y axis with arrows and labels |
| `Curve` | Function curve from points array |
| `Ball` | Draggable/animated circle |

---

## ML/DL Widgets

Pre-built interactive widgets for machine learning concepts.

**Path:** `src/components/widgets/`

| Widget | Purpose | Props |
|--------|---------|-------|
| `LinearFitExplorer` | Draggable line fit to data points | `initialSlope`, `initialIntercept`, `showResiduals`, `showMSE` |
| `LossSurfaceExplorer` | 3D loss surface with draggable point | — |
| `GradientDescentExplorer` | Animated ball rolling on loss curve | `initialPosition`, `initialLearningRate`, `showLearningRateSlider`, `showGradientArrow` |
| `LearningRateExplorer` | Compare different learning rates | `mode: 'comparison' | 'interactive'` |
| `TrainingLoopExplorer` | Complete training visualization with loss curve | `numPoints`, `initialLearningRate`, `width`, `height` |
| `MermaidDiagram` | Render Mermaid diagrams as inline SVG | `chart` (mermaid definition string) |

**Widget conventions:**
- **Always use `useContainerWidth` hook** (`src/hooks/useContainerWidth.ts`) for responsive width. Pattern: `const { containerRef, width: measuredWidth } = useContainerWidth(fallback)` then `const width = widthOverride ?? measuredWidth` and attach `ref={containerRef}` to the outer div
- Accept `width` and `height` props for ExercisePanel fullscreen override
- Use `ZoomableCanvas` as base
- Show controls below canvas (buttons, sliders)
- Display live stats in colored badges
- Include explanatory text at bottom

---

## Lesson Structure Pattern

### Page file (`src/app/app/lesson/{slug}/page.tsx`)

```tsx
import { MyLesson } from '@/components/lessons/module-X-Y'

export default function Page() {
  return <MyLesson />
}
```

### Component file (`src/components/lessons/module-X-Y/MyLesson.tsx`)

```tsx
'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import { ... } from '@/components/lessons'  // Block components
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

export function MyLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Lesson Title"
            description="One-line description."
            category="Category Name"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>...</ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock>...</TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 1, 2, 3... */}
      <Row>
        <Row.Content>
          <SectionHeader title="..." subtitle="..." />
          <div className="space-y-4">
            <p className="text-muted-foreground">...</p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock>...</InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive widget */}
      <Row>
        <Row.Content>
          <ExercisePanel title="...">
            <MyWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Try this</li>
              <li>• Notice that</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock items={[...]} />
        </Row.Content>
      </Row>

      {/* References (2-4 curated links) */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              { title: 'Paper Title', authors: 'Author et al., Year', url: '...', note: 'What to focus on' },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next step */}
      <Row>
        <Row.Content>
          <NextStepBlock href="/app/lesson/next" ... />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
```

---

## Module Completion Block

Celebration block for the end of a module.

**Path:** `src/components/lessons/blocks.tsx`

```tsx
import { ModuleCompleteBlock } from '@/components/lessons'

<Row>
  <Row.Content>
    <ModuleCompleteBlock
      module="1.1"
      title="The Learning Problem"
      achievements={[
        'ML as function approximation',
        'Generalization vs memorization',
        'Loss functions (MSE)',
      ]}
      nextModule="1.2"
      nextTitle="From Linear to Neural"
    />
  </Row.Content>
</Row>
```

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| `module` | `string` | Module number (e.g., "1.1") |
| `title` | `string` | Module title |
| `achievements` | `string[]` | List of concepts/skills learned |
| `nextModule` | `string` | Next module number |
| `nextTitle` | `string` | Next module title |

---

## Colab Notebook Links

For hands-on Python exercises, link to Colab:

```tsx
import { ExternalLink } from 'lucide-react'

<div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
  <div className="space-y-4">
    <p className="text-muted-foreground">
      Now write the code yourself in a Jupyter notebook.
    </p>
    <a
      href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/1-1-6-linear-regression.ipynb"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
    >
      <ExternalLink className="w-4 h-4" />
      Open in Google Colab
    </a>
    <p className="text-xs text-muted-foreground">
      The notebook includes exercises...
    </p>
  </div>
</div>
```

**Naming convention:** `{module}-{lesson}-{topic}.ipynb` (e.g., `1-1-6-linear-regression.ipynb`)
