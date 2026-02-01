---
name: lesson-planning
description: Create effective, ADHD-friendly lesson plans for AI/ML learning. Use when designing new lessons or adapting curriculum.
---

# Lesson Planning

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
