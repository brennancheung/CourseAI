---
name: lesson-planning
description: Create effective, ADHD-friendly lesson plans for AI/ML learning. Use when designing new lessons or adapting curriculum.
---

# Lesson Planning

## ⚡ Context Loading (Strategic)

Load context strategically to preserve context window. **Don't load everything upfront.**

### Always Load (Small Files)
```
READ IMMEDIATELY:
1. .claude/skills/interactive-widgets/references/component-catalog.md (~200 lines)
2. src/data/curriculum/types.ts (~90 lines) — check CurriculumNode shape
```

### Load On-Demand (When Needed)

| When... | Load... |
|---------|---------|
| Planning which lesson to build | `docs/curriculum.md` — find the specific section |
| Need widget patterns | `.claude/skills/interactive-widgets/SKILL.md` |
| Need lesson structure reference | ONE recent lesson (e.g., `src/components/lessons/module-1-1/GradientDescentLesson.tsx`) |
| Building a widget | Relevant widget file from `src/components/widgets/` |

### Never Load All At Once
- ❌ Don't glob all lessons — just read ONE as a reference
- ❌ Don't load full curriculum if you already know what lesson to build
- ❌ Don't load widget skill if lesson has no interactivity

### Quick Reference (No Need to Load)
These are in this skill file already:
- Block components → see "Block Components for Content Types" section below
- Lesson structure → see "Lesson Row Layout" section below
- Text patterns → see "Text Styling Patterns" section below

---

## Implementation Patterns (Lessons Learned)

These patterns apply to building effective lessons for any subject. Follow these when creating new lesson components.

### Split Chaotic Content Into Focused Lessons

**The Problem:** A lesson that tries to cover too much becomes overwhelming and hard to learn from.

**The Solution:** Split large topics into multiple focused lessons, each with ONE core objective.

**Example:** A lesson on "Transformers" trying to cover:
- Attention mechanisms
- Self-attention vs cross-attention
- Positional encoding
- Layer normalization
- Training dynamics

Should be split into focused lessons:
1. `attention-mechanism` — Just the core attention concept with visualization
2. `self-attention` — Multi-head self-attention with code
3. `transformer-architecture` — How the pieces fit together

**Rule of thumb:** If a lesson has more than 3-4 major sections, consider splitting it.

### Multimodal Learning (Critical)

**The Problem:** Lessons that only use text and math formulas fail to engage multiple learning pathways. This leads to shallow understanding that doesn't stick.

**The Principle:** Every core concept should be presented through 2-3 different modalities:

| Modality | Examples | Engages |
|----------|----------|---------|
| **Verbal** | Text explanations, analogies | Language processing, narrative memory |
| **Symbolic** | Math formulas, code | Abstract reasoning, precision |
| **Visual** | Diagrams, animations, graphs | Spatial reasoning, pattern recognition |
| **Kinesthetic** | Sliders, draggable points, manipulation | Motor memory, active learning |

**The Check:** For each major concept in a lesson, ask:
- Is there text explaining it? (verbal)
- Is there a formula or code? (symbolic)
- Is there a diagram or visualization? (visual)
- Can the learner manipulate something? (kinesthetic)

**Example - Teaching "What is a Neuron":**
- ❌ **Weak:** Only shows formula `y = w₁x₁ + w₂x₂ + b` with text explanation
- ✅ **Strong:**
  - Text explaining weighted sum + bias (verbal)
  - Formula `y = w₁x₁ + w₂x₂ + b` (symbolic)
  - Diagram showing inputs → neuron → output (visual)
  - Sliders to adjust weights/inputs and watch output change (kinesthetic)

**Why it matters:** Different people learn through different channels. More importantly, concepts encoded through multiple modalities are more durable and transferable. When a learner can see it, manipulate it, AND read the formula, the understanding is deeper.

**Rule of thumb:** If a section only has text and math, it needs a visual or interactive element. Math alone is not enough.

### Interactive Widgets

Consider what concepts could be enhanced with interactive visualization or exploration. See the `interactive-widgets` skill for:
- Ideation prompts to identify interactivity opportunities
- Available components
- Implementation patterns (expandable explorers, selector + visualization)

**Quick check:** For each lesson section, ask:
- Would this benefit from visualization?
- Could the user explore or manipulate something?
- Is there an "aha moment" that interactivity could trigger?

### Lesson Row Layout

**CRITICAL: Always use the `Row` component from `@/components/layout/Row`.**

Never create manual flex layouts in lessons. The Row component automatically injects an empty aside placeholder when no aside is provided, which prevents content from expanding full-width.

```tsx
import { Row } from '@/components/layout/Row'

// ✅ CORRECT - Always use Row
<Row>
  <Row.Content>
    <Card ... />
  </Row.Content>
  <Row.Aside>
    <TipBlock>...</TipBlock>
  </Row.Aside>
</Row>

// ✅ CORRECT - Conditional aside (Row auto-injects empty placeholder)
<Row>
  <Row.Content>
    <Card ... />
  </Row.Content>
  {isExpanded && (
    <Row.Aside>
      <TipBlock>...</TipBlock>
    </Row.Aside>
  )}
</Row>

// ❌ WRONG - Manual flex layout (causes width issues!)
<div className="flex flex-col lg:flex-row gap-8 lg:gap-32 items-start">
  <div className="w-full lg:w-[900px] flex-shrink-0">
    <Card ... />
  </div>
</div>
```

**Why this matters:** When the aside is conditionally rendered and you use manual flex, the content expands full-width when the aside is hidden. Row prevents this by always maintaining the 2-column structure.

### Block Components for Content Types

Use lesson blocks from `@/components/lessons` for visual consistency.

**For Asides (sidebar blocks):**

| Block | Purpose | Color |
|-------|---------|-------|
| `TipBlock` | Helpful hints, "Use for" | Sky blue |
| `InsightBlock` | Key concepts, "Why it works" | Violet |
| `TryThisBlock` | Interactive prompts, variations | Emerald |
| `WarningBlock` | Common mistakes | Rose |
| `ConceptBlock` | Theory explanations | Neutral |

**For Main Content (gradient cards):**

| Block | Purpose | When to Use |
|-------|---------|-------------|
| `GradientCard` | Single categorized card | Feature highlights, categorized info |
| `ComparisonRow` | Two side-by-side cards | A vs B comparisons |
| `PhaseCard` | Numbered sequential card | Multi-step processes, timeline phases |

See `interactive-widgets` skill → `references/component-catalog.md` for full component reference.

### CodeBlock for Code Examples

Use `CodeBlock` from `@/components/common/CodeBlock` for syntax-highlighted code in lessons.

```tsx
import { CodeBlock } from '@/components/common/CodeBlock'

<CodeBlock
  code={`def forward(self, x):
    return self.linear(x)`}
  language="python"
  filename="model.py"  // Optional - shows in header
/>
```

**Features:**
- Syntax highlighting via Prism (react-syntax-highlighter)
- Faint line numbers with proper spacing
- Theme-aware (dark/light mode via next-themes)
- Copy to clipboard button (appears on hover)
- Optional filename header

**Styling notes (from ventures):**
- Line number color: `rgb(156 163 175 / 0.5)` for dark, `rgb(107 114 128 / 0.5)` for light
- Container: `text-xs` for appropriate code size
- Background: `transparent` (inherits from container)
- Uses `oneDark` / `oneLight` themes from react-syntax-highlighter

**Hydration safety:** The component handles SSR by defaulting to dark theme until mounted, avoiding hydration mismatches.

### ClipboardCopy Component

Reusable clipboard copy button from `@/components/common/ClipboardCopy`.

```tsx
import { ClipboardCopy } from '@/components/common/ClipboardCopy'

<ClipboardCopy text="content to copy" />
<ClipboardCopy text={code} className="h-7 w-7" />
<ClipboardCopy text={url} label="Copy URL" />
```

**Props:**
- `text` — Content to copy (required)
- `variant` — Button variant: `'ghost'` (default), `'default'`, `'outline'`
- `size` — Button size: `'icon'` (default), `'sm'`, `'default'`, `'lg'`
- `label` — Optional text label (shows "Copied!" on success)
- `className` — Additional classes

Uses `useTimeout` hook for the 2-second "copied" state reset.

### Math Rendering with KaTeX

For lessons with formulas, use KaTeX:

```tsx
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

// Inline math in paragraph
<p>The loss is <InlineMath math="\hat{y} = wx + b" /> where...</p>

// Block math (centered, standalone)
<BlockMath math="L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2" />

// Emphasized formula box
<div className="py-4 px-6 bg-muted/50 rounded-lg">
  <BlockMath math="\theta_{new} = \theta_{old} - \alpha \nabla L" />
</div>
```

**Pattern for explaining formulas:**
```tsx
<div className="py-4">
  <BlockMath math="L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2" />
</div>
<p className="text-muted-foreground">Let's break this down:</p>
<ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
  <li><InlineMath math="y_i - \hat{y}_i" /> — the residual for point i</li>
  <li><InlineMath math="(\cdot)^2" /> — square it (makes positive)</li>
  <li><InlineMath math="\sum" /> — sum over all points</li>
  <li><InlineMath math="\frac{1}{n}" /> — divide by count</li>
</ul>
```

### Lesson Metadata

Lesson metadata lives in the curriculum tree (`src/data/curriculum/`). Each leaf node includes exercise data inline:

```typescript
// In src/data/curriculum/foundations.ts (or similar)
{
  slug: 'my-lesson',                      // URL slug — also the route
  title: 'Lesson Title',
  description: 'One-line description.',
  duration: '20 min',
  category: 'Fundamentals',               // Module name
  objectives: [                           // Learning objectives
    'Understand concept A',
    'See why B matters',
  ],
  skills: ['skill-tag-1'],
  prerequisites: ['previous-lesson-slug'],
  exercise: {
    constraints: [                        // ADHD scope boundaries
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

Then create the lesson page at `src/app/app/lesson/{slug}/page.tsx`:

```tsx
import { MyLesson } from '@/components/lessons/module-X-Y'

export default function Page() {
  return <MyLesson />
}
```

### ExercisePanel for Widgets

Wrap interactive widgets in `ExercisePanel` for consistent styling + fullscreen:

```tsx
import { ExercisePanel } from '@/components/widgets/ExercisePanel'

<Row>
  <Row.Content>
    <ExercisePanel title="Try Fitting a Line">
      <LinearFitExplorer
        initialSlope={0.3}
        initialIntercept={-0.5}
        showResiduals={true}
      />
    </ExercisePanel>
  </Row.Content>
  <Row.Aside>
    <TryThisBlock title="Experiment">
      <ul className="space-y-2 text-sm">
        <li>• Drag the intercept point up and down</li>
        <li>• Try to minimize the MSE</li>
      </ul>
    </TryThisBlock>
  </Row.Aside>
</Row>
```

### Module Completion Block

At the end of a module's final lesson, celebrate completion with `ModuleCompleteBlock`:

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
- `module` — Module number (e.g., "1.1")
- `title` — Module title (e.g., "The Learning Problem")
- `achievements` — Array of concepts/skills learned
- `nextModule` — Next module number
- `nextTitle` — Next module title

### Google Colab Notebooks

For hands-on exercises, link to Colab notebooks stored in the `/notebooks/` directory.

**Notebook naming:** `{module}-{lesson}-{topic}.ipynb` (e.g., `1-1-6-linear-regression.ipynb`)

**Colab link format:**
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
      The notebook includes exercises: try different learning rates, add more noise...
    </p>
  </div>
</div>
```

**Best practice:** Use the in-app interactive widgets (like `TrainingLoopExplorer`) for visualization, and link to Colab for the "implement it yourself" exercises. The widget shows what happens; the notebook lets them write the code.

**Colab workflow:**
1. Create `.ipynb` file in `/notebooks/` directory
2. Commit to main branch on GitHub
3. Link opens directly in Colab from GitHub
4. User clicks "Copy to Drive" to save their own version

### Text Styling Patterns

Standard text patterns used in lessons:

```tsx
// Paragraph text
<p className="text-muted-foreground">
  Regular explanation text goes here.
</p>

// Text with emphasis
<p className="text-muted-foreground">
  We need a way to <strong>measure</strong> how good a fit is — a
  single number that tells us how wrong our predictions are.
</p>

// Bulleted list
<ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
  <li>First point</li>
  <li>Second point with <strong>emphasis</strong></li>
</ul>

// Numbered list
<ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
  <li>First step</li>
  <li>Second step</li>
</ol>

// Spaced content sections
<div className="space-y-4">
  <p className="text-muted-foreground">First paragraph...</p>
  <p className="text-muted-foreground">Second paragraph...</p>
</div>

// Side-by-side concepts
<div className="grid gap-4 md:grid-cols-2">
  <ConceptBlock title="Option A">...</ConceptBlock>
  <ConceptBlock title="Option B">...</ConceptBlock>
</div>
```

### Lesson Section Pattern

Standard section structure with header + content + aside:

```tsx
<Row>
  <Row.Content>
    <SectionHeader
      title="The Core Insight"
      subtitle="What does a machine actually 'learn'?"
    />
    <div className="space-y-4">
      <p className="text-muted-foreground">
        First paragraph of explanation...
      </p>
      <p className="text-muted-foreground">
        More precisely: <strong>key concept here</strong>.
      </p>
    </div>
  </Row.Content>
  <Row.Aside>
    <InsightBlock title="Key Point">
      The "aha moment" for this section goes in the aside.
    </InsightBlock>
  </Row.Aside>
</Row>
```

**Section with comparison:**
```tsx
<Row>
  <Row.Content>
    <SectionHeader title="Two Approaches" subtitle="..." />
    <ComparisonRow
      left={{
        title: 'Approach A',
        color: 'amber',
        items: ['Point 1', 'Point 2'],
      }}
      right={{
        title: 'Approach B',
        color: 'emerald',
        items: ['Point 1', 'Point 2'],
      }}
    />
  </Row.Content>
  <Row.Aside>
    <TipBlock>When to use which...</TipBlock>
  </Row.Aside>
</Row>
```

### Expandable Card Pattern

For lessons with multiple items to explore (concepts, techniques, models), use expandable cards **wrapped in Row**:

```tsx
import { Row } from '@/components/layout/Row'

function ConceptRow({ concept, isExpanded, onToggle }) {
  return (
    <Row>
      <Row.Content>
        <ConceptCard
          concept={concept}
          isExpanded={isExpanded}
          onToggle={onToggle}
        />
      </Row.Content>
      {isExpanded && (
        <Row.Aside>
          <TipBlock title="Key Insight">...</TipBlock>
          <WarningBlock title="Common Misconceptions">...</WarningBlock>
        </Row.Aside>
      )}
    </Row>
  )
}
```

**Key points:**
- **Always use Row** — even for expandable cards with conditional asides
- Row auto-injects empty aside when not expanded, preventing full-width expansion
- Card header is always visible with title + tagline
- Expanded content shows detailed breakdown
- Aside appears alongside expanded card with contextual tips
- Only one card expanded at a time (controlled by parent)

---

## Core Framework

### 1. Know the Learner

Before creating a lesson, understand current state:

**Learner baseline:**
- Software engineer with strong programming background
- Comfortable with Python, can read academic papers
- Has used AI tools but wants deeper understanding
- ADHD: needs low activation energy, clear scope, no decision paralysis

**Identify the learning edge:**
- What can they almost do but not quite?
- What's the next logical step from recent lessons?
- What mental model would unlock new capability?

### 2. Define the Learning Objective

Every lesson needs ONE clear outcome:

**Good objectives:**
- "Implement attention from scratch in PyTorch"
- "Understand why layer normalization matters for transformer training"
- "Build intuition for how diffusion models denoise"

**Bad objectives:**
- "Get better at deep learning" (too vague)
- "Understand transformers" (too big)
- "Learn attention and normalization and training" (too many things)

**The test:** Can you say "After this lesson, I can ___" with a specific, observable skill?

### 3. Set Scope Boundaries (What You're NOT Doing)

This is critical for ADHD. Explicitly close doors:

```
NOT optimizing for speed
NOT worrying about deployment
NOT covering all variations
Just focusing on: [the one thing]
```

Constraints liberate. Every "not" removes a decision.

---

## Lesson Structure & Flow

The order of content in a lesson matters. Motivation drives learning.

### The Pedagogical Flow

1. **Context** — What is this lesson about? Set constraints.
2. **Hook** — Motivate the learner immediately (see below)
3. **Explain** — Teach the concept now that they're interested
4. **Explore** — Let them interact and try things
5. **Elaborate** — Deeper concepts after basics are understood
6. **Practice** — Concrete steps to apply what they learned
7. **Challenge** — Final exercise to synthesize learning
8. **Reflect** — Capture what worked and what was hard

### The Hook

**The hook is critical.** It comes right after context and before explanation. Its job is to increase motivation before teaching begins.

**Types of hooks:**
| Hook Type | Example |
|-----------|---------|
| Demo | Interactive visualization showing the concept in action |
| Before/After | "Here's what GPT-2 outputs vs GPT-4 — what changed?" |
| Question | "Why can transformers understand context that RNNs can't?" |
| Challenge preview | "By the end, you'll implement this from scratch" |
| Real-world impact | "This technique enabled ChatGPT to exist" |

**Hook principles:**
- Short — 30 seconds to 2 minutes max
- Emotionally engaging — should create "I want to understand that" feeling
- Directly relevant — must connect to what you're about to teach
- No prerequisites — should work even if they don't understand the concept yet

---

## ADHD-Friendly Checklist

Before finalizing any lesson, verify:

- [ ] **Single focus** — One skill, one outcome
- [ ] **Low activation energy** — Can start immediately
- [ ] **No decisions required** — Clear path forward
- [ ] **Concrete deliverable** — "Build X" not "Study Y"
- [ ] **Scope boundaries explicit** — What you're NOT doing is stated
- [ ] **No guilt hooks** — No streaks, no "you should have..."
- [ ] **Working memory friendly** — All info in the lesson, nothing to remember

---

## Deliberate Practice Checklist

- [ ] **Specific goal** — Not vague improvement
- [ ] **At the edge** — Requires focus, not too easy, not frustrating
- [ ] **Feedback available** — Can test/verify if it's working
- [ ] **Repeatable** — Can do it again with variation
- [ ] **Builds mental model** — Develops internal representation
