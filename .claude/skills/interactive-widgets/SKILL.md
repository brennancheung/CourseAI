---
description: Make lessons engaging with interactive widgets. Use when creating lessons that teach concepts which could benefit from visualization, manipulation, or exploration. Helps identify opportunities for interactivity and provides patterns for building custom React components. Trigger phrases include "make this interactive", "add a visualization", "this needs a widget", or when discussing how to better illustrate a concept.
---

# Interactive Widgets for Lessons

Transform passive reading into active exploration. This skill helps you identify where interactivity would create "aha moments" and provides patterns for building them.

## When to Consider Interactivity

Ask these questions when designing a lesson:

| Question | If Yes → Consider |
|----------|-------------------|
| Is there a **range** of values? | Slider, piano roll, spectrum display |
| Are there **comparisons**? | A/B toggle, side-by-side, before/after |
| Is there a **sequence or progression**? | Timeline, stepper, animated playback |
| Could the user **hear** it? | Audio player, Spotify link, score playback |
| Is there **data to explore**? | Expandable cards, filterable list |
| Would **manipulation** create insight? | Parameter controls with live preview |
| Is something **hard to describe in words**? | Visualization, diagram, animation |

## Ideation Prompts

Before building, answer these:

1. **What's the core concept?** (one sentence)
2. **What would be impossible to convey with just text?**
3. **What "aha moment" should the user experience?**
4. **What would they want to play with or explore?**
5. **What comparison would make the concept click?**

### Example Ideation

**Lesson:** Instrument ranges and power zones

- **Core concept:** Each instrument sounds best in a specific range
- **Hard to convey:** The *feel* of where an instrument shines vs struggles
- **Aha moment:** Seeing how little overlap exists between violin and cello power zones
- **Play with:** Select different instruments, see their ranges light up
- **Comparison:** Side-by-side family comparisons

→ **Result:** `PianoRoll` component with instrument selector and color-coded zones

---

## Component Architecture: Atoms → Molecules

Build composable components. **Atoms** are primitives that can be reused. **Molecules** combine atoms for specific use cases.

### Example: Piano

```
Piano (atom)                    → PianoRoll (molecule)
  - Renders keyboard              - Uses Piano
  - Accepts highlight map         - Adds instrument selector
  - Optional click handler        - Adds range highlighting logic
                                  - Adds legend
```

**Why this matters:** The Piano atom can be reused for:
- Chord visualizer (highlight chord notes)
- Scale visualizer (highlight scale degrees)
- WebMIDI input display (highlight played notes)
- Voicing comparison (show two voicings side-by-side)

### When to Extract an Atom

Extract a reusable atom when:
- The visualization could serve multiple purposes
- You find yourself wanting to "use part of" an existing component
- The primitive has clear inputs (data) and outputs (visual)

Keep as a molecule when:
- It's highly specific to one lesson/concept
- The state management is tightly coupled to the use case

---

## Available Components

### Score & Playback

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `ScorePlayer` | ABC notation with playback | Teaching patterns, progressions, melodies |
| `TempoProvider` + `TempoText` | Clickable tempo links | Text mentioning BPM that should be playable |
| Spotify links (`spotify:track:ID`) | Real recordings | Reference professional examples |

### Visualization

| Component | Type | Purpose | When to Use |
|-----------|------|---------|-------------|
| `Piano` | Atom | Keyboard with highlight support | Any keyboard visualization |
| `PianoRoll` | Molecule | Instrument range explorer | Teaching registers, orchestration |
| Score display | - | Static notation | Showing specific musical examples |

### MIDI Utilities (`src/lib/midi.ts`)

```typescript
import { n, midiToNote, chord, scale, CHORD_INTERVALS, SCALE_INTERVALS } from '@/lib/midi'

// Note name to MIDI
n('C4')  // 60
n('G3')  // 55

// MIDI to note name
midiToNote(60)  // 'C4'

// Build a chord
chord(n('C4'), CHORD_INTERVALS.major)  // [60, 64, 67]
chord(n('A3'), CHORD_INTERVALS.minor)  // [57, 60, 64]

// Build a scale across a range
scale(n('C4'), SCALE_INTERVALS.major, 48, 84)  // C major from C3 to C6
```

### Exploration Patterns

| Pattern | Components Used | When to Use |
|---------|-----------------|-------------|
| Expandable list | State + Card + Aside | Multiple items to explore (techniques, patterns) |
| Grouped selector | Button group + visualization | Comparing related items (instruments by family) |
| Progressive reveal | Accordion/stepper | Building understanding step-by-step |

---

## Implementation Patterns

### Pattern 1: Expandable Explorer

For lessons with multiple items to explore (techniques, instruments, patterns).

```tsx
'use client'

import { useState } from 'react'
import { LessonRow, TipBlock, WarningBlock } from '@/components/lessons'

type Item = {
  id: string
  name: string
  description: string
  details: string[]
  tips?: string[]
  warnings?: string[]
}

export function ItemExplorer({ items }: { items: Item[] }) {
  const [expandedId, setExpandedId] = useState<string | null>(null)

  return (
    <div className="space-y-4">
      {items.map((item) => {
        const isExpanded = expandedId === item.id
        return (
          <LessonRow
            key={item.id}
            aside={isExpanded ? (
              <div className="space-y-4">
                {item.tips && <TipBlock title="Tips">{item.tips.join(' ')}</TipBlock>}
                {item.warnings && <WarningBlock title="Avoid">{item.warnings.join(' ')}</WarningBlock>}
              </div>
            ) : undefined}
          >
            <ItemCard
              item={item}
              isExpanded={isExpanded}
              onToggle={() => setExpandedId(isExpanded ? null : item.id)}
            />
          </LessonRow>
        )
      })}
    </div>
  )
}
```

**Key points:**
- Only one item expanded at a time
- Aside appears alongside expanded card
- Use `LessonRow` for consistent layout (900px main + 256px aside)

### Pattern 2: Selector + Visualization

For exploring variations of a single concept (like PianoRoll).

```tsx
'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'

type Variant = {
  id: string
  name: string
  category: string
  data: unknown // Whatever the visualization needs
}

export function VariantExplorer({ variants }: { variants: Variant[] }) {
  const [selectedId, setSelectedId] = useState<string>(variants[0]?.id ?? null)
  const selected = variants.find(v => v.id === selectedId)

  // Group by category
  const categories = [...new Set(variants.map(v => v.category))]

  return (
    <div className="space-y-4">
      {/* Selector */}
      <div className="flex flex-wrap gap-2">
        {categories.map(category => (
          <div key={category} className="flex flex-wrap gap-1">
            {variants
              .filter(v => v.category === category)
              .map(variant => (
                <button
                  key={variant.id}
                  onClick={() => setSelectedId(variant.id)}
                  className={cn(
                    'px-3 py-1.5 rounded-full text-xs font-medium transition-colors',
                    selectedId === variant.id
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                  )}
                >
                  {variant.name}
                </button>
              ))}
          </div>
        ))}
      </div>

      {/* Visualization */}
      {selected && <Visualization data={selected.data} />}
    </div>
  )
}
```

### Pattern 3: Score with Context

For musical examples that should be playable.

```tsx
import { TempoProvider } from '@/components/exercises/TempoContext'
import { ScorePlayer } from '@/components/score/ScorePlayer'
import { TempoText } from '@/components/exercises/TempoText'
import { TipBlock } from '@/components/lessons'

export function MusicalExample({ abc, tempo, tip }: { abc: string; tempo: number; tip: string }) {
  return (
    <TempoProvider initialTempo={tempo}>
      <div className="space-y-4">
        <ScorePlayer abc={abc} />
        <TipBlock>
          <TempoText>{tip}</TempoText> {/* Tempos in text become clickable */}
        </TipBlock>
      </div>
    </TempoProvider>
  )
}
```

**Critical:** `TempoProvider` must wrap ALL components using `TempoText`, including asides.

---

## Creating a New Widget

### 1. Define the Data Shape

```typescript
// src/data/exercises/[feature].ts
export type WidgetItem = {
  id: string
  name: string
  // ... specific fields for this widget
}

export const items: WidgetItem[] = [
  // ... data
]
```

### 2. Create the Component

```tsx
// src/components/exercises/[WidgetName].tsx
'use client'

import { useState } from 'react'
// ... imports

interface WidgetProps {
  className?: string
  // optional: items?: WidgetItem[] // if data should be passed in
}

export function WidgetName({ className }: WidgetProps) {
  // State for interactivity
  const [selected, setSelected] = useState<string | null>(null)

  return (
    <div className={cn('space-y-4', className)}>
      {/* Interactive elements */}
    </div>
  )
}
```

### 3. Integrate into Lesson

```tsx
// src/components/exercises/[Lesson]Lesson.tsx
import { WidgetName } from './WidgetName'
import { LessonRow, TipBlock } from '@/components/lessons'

// In the lesson component:
<LessonRow
  aside={<TipBlock title="How to Use">...</TipBlock>}
>
  <section className="space-y-4">
    <h2 className="text-lg font-bold">Section Title</h2>
    <p className="text-sm text-muted-foreground">Description...</p>
    <WidgetName />
  </section>
</LessonRow>
```

---

## Color Conventions

Consistent colors help users recognize patterns across lessons:

| Family/Category | Color | Tailwind Classes |
|-----------------|-------|------------------|
| Strings | Blue | `bg-blue-500`, `text-blue-500` |
| Brass | Amber | `bg-amber-500`, `text-amber-500` |
| Woodwinds | Emerald | `bg-emerald-500`, `text-emerald-500` |
| Tips/Info | Sky | `bg-sky-500/10`, `border-sky-500/20` |
| Warnings | Rose | `bg-rose-500/10`, `border-rose-500/20` |
| Insights | Violet | `bg-violet-500/10`, `border-violet-500/20` |
| Try This | Emerald | `bg-emerald-500/10`, `border-emerald-500/20` |

---

## Widget Ideas (Not Yet Built)

These could be valuable for future lessons:

| Widget | Purpose | Potential Use |
|--------|---------|---------------|
| Frequency Spectrum | Show instrument frequency ranges | Teaching EQ, mixing, muddiness |
| Chord Voicing Visualizer | Show note distribution across octaves | Teaching voicing, spacing |
| Dynamic Curve | Visualize crescendo/diminuendo shapes | Teaching dynamics, tension |
| Arrangement Timeline | Show instrument entrances over time | Teaching arrangement, builds |
| A/B Comparator | Toggle between two audio/score versions | Teaching before/after |
| Parameter Explorer | Sliders that update a visualization | Teaching continuous concepts |

---

## Checklist Before Building

- [ ] Answered the 5 ideation questions
- [ ] Identified which existing component/pattern is closest
- [ ] Defined the data shape
- [ ] Decided: standalone component or lesson-specific?
- [ ] Chose appropriate colors for the category
- [ ] Considered mobile responsiveness (min-width, overflow-x-auto)
