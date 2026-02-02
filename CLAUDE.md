# CLAUDE.md

## Project Overview

**CourseAI** is a personal learning app for understanding AI and machine learning. Single user, local only, no auth required.

### The Problem
AI is moving incredibly fast. Papers, tutorials, and frameworks pile up. The result: endless tab hoarding, surface-level understanding, no real depth.

### The Transformation
**From:** Overwhelmed by the pace of AI, reading summaries but not truly understanding.
**To:** Build intuition for how models actually work, implement key concepts, confidently explore new research.

### Course Scope
- **Deep Learning Fundamentals** — Neural networks, backprop, optimization, architectures
- **PyTorch** — Practical implementation skills
- **LLMs** — Transformers, attention, training, fine-tuning, prompting
- **Recent LLM Advances** — RLHF, constitutional AI, reasoning models, multimodal
- **Stable Diffusion** — Classical diffusion, DDPM, latent diffusion
- **Post-SD Advances** — ControlNet, SDXL, consistency models, flow matching

## User Context

- Software engineer with strong programming background
- Comfortable with Python, can read academic papers
- Has used AI tools but wants deeper understanding
- ADHD — gets passionate and dives in, but lacks consistency
- **Setup:** Python environment, access to GPUs (local or cloud)

## Design Principles

**BJ Fogg Behavioral Model** (Behavior = Motivation × Ability × Prompt)
- Motivation: Visible progress, journey feel, no streak punishment
- Ability: Chunking, constraints, eliminate decision paralysis
- Prompt: Clear next step on every app open

**Deliberate Practice (Ericsson)**
- Structured exercises at edge of current ability
- Build mental models that compress complexity
- Not passive reading — active implementation and experimentation

**ADHD-Friendly**
- Session-based, not streak-based
- Low activation energy
- No guilt for inconsistency
- Permission to pick up where you left off

## Commands

```bash
pnpm ai              # Run typecheck and lint
pnpm dev             # Run dev server (ask user to run this)
pnpm cli             # CLI tool (commands TBD)
```

## Architecture

- **Next.js 16** with App Router
- **Convex** for backend (no auth)
- **Tailwind CSS v4** with shadcn/ui components
- Single user — no users table, no auth

### Data Architecture (Hybrid)
- **Curriculum** (lessons, skills, mental models) → TypeScript files in `src/data/`
  - Claude Code can read/modify directly
  - Schema flexibility without migrations
  - Version controlled, grep-able
- **Sessions** (learning history) → Convex database
  - Accumulating user data
  - Queryable, reactive UI

### Key Routes

- `/app` — Today view (recommended lesson)
- `/app/skills` — Skill map
- `/app/journal` — Session log

### UX Components

- **Today View**: Recommended lesson with explanation, minimal decisions required
- **Skill Map**: Optional exploration of learning journey and mental models
- **Lesson View**: Concepts, code examples, exercises, clear scope boundaries
- **Session Log**: Self-assessment, what clicked, what was confusing, visible progress

### Interactive Components

See `docs/interactive-components.md` for full reference. Quick summary:

| Library | Use For |
|---------|---------|
| **Mafs** | Interactive math plots, draggable points, function visualization |
| **Visx** | Custom 2D visualizations, neural network diagrams |
| **React Three Fiber** | 3D visualizations, impressive demos |
| **Recharts** | Training curves, loss/accuracy plots, standard charts |
| **KaTeX** | Math formula rendering (LaTeX) |

### Python Notebooks (Google Colab)

Store `.ipynb` files in the repo (or separate repo). Link using:
```
https://colab.research.google.com/github/{user}/{repo}/blob/main/{path}.ipynb
```

User clicks link → notebook opens in Colab → user can run cells → saves copy to their Drive if needed.

## Claude Code Role

Claude Code runs alongside the app as a mentor with full repo access:
- Read/modify curriculum and lessons directly
- Review session logs for personalized guidance
- Have feedback conversations about progress
- No in-app AI chat — adaptation happens through Claude Code sessions

### Lesson Work: Always Run `/lesson-planning` First

**Before doing ANY work related to lessons or exercises, invoke the `/lesson-planning` skill.** This applies to:
- Planning a new lesson or exercise
- Researching what content a lesson should have
- Creating lesson components or data files
- Modifying existing lesson content
- Discussing lesson structure or pedagogy

The skill contains critical implementation patterns, ADHD-friendly design principles, and lessons learned from previous work. Skipping it leads to inconsistent lessons that don't match the established patterns.

## Open Questions (Resolve Through Use)

- What does the skill map actually contain?
- What are the right first lessons?
- How to balance theory vs implementation?
- How granular should progress tracking be?

## Conventions

- Use snake-case for documentation files
- No `switch` statements — use helper functions
- No `any` in TypeScript
- No `else if` / `else` — use early return pattern
- Use pnpm, not npm

### Lesson Layout: Always Use Row

**ALL lesson content MUST use the `Row` compound component** from `@/components/layout/Row`. Never create manual flex layouts in lessons.

```tsx
// ✅ CORRECT - Always use Row
<Row>
  <Row.Content>Main content here</Row.Content>
  <Row.Aside>Optional sidebar</Row.Aside>  {/* or omit for auto-injected empty aside */}
</Row>

// ❌ WRONG - Never do this
<div className="flex flex-col lg:flex-row gap-8">
  <div className="flex-1">content</div>
  <aside className="lg:w-64">sidebar</aside>
</div>
```

**Why:** The Row component auto-injects a fixed-width empty aside when none is provided, ensuring the 2-column layout never breaks. Manual flex layouts bypass this protection and cause content to expand full-width unpredictably.

**Even for custom/interactive components** that conditionally show asides, wrap in Row:
```tsx
<Row>
  <Row.Content><MyExpandableCard /></Row.Content>
  {isExpanded && <Row.Aside>contextual tips</Row.Aside>}
</Row>
```

**Do not use `LessonRow`** — it's deprecated. Use `Row` instead.
