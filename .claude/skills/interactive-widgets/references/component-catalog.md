# Component Catalog

Quick reference for all interactive components available in lessons.

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

**Note:** `LessonRow` is deprecated. Use `Row` instead.

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
