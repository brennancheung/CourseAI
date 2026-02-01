# Productizing the Lesson Planning System

> Notes from exploring how to turn the Claude Code + lesson generation system into a product that non-programmers can use.

---

## The Core Challenge

**What makes the current system powerful:**
- Arbitrary React components (PianoRoll, TempoProvider, custom visualizations)
- Full filesystem (organize lessons, reference assets)
- Agentic iteration (Claude Code tries things, sees errors, fixes them)

**What makes it hard to productize:**
- Requires dev environment
- Security nightmare if you give users code execution
- The agentic loop is expensive and complex to replicate

**The goal:** Let non-programmers describe interactions in natural language and get custom interactive educational widgets, without needing to understand code or debug errors.

---

## Architecture Options Considered

### Option 1: Semantic Lesson Schema (No Code Gen)

Define a Lesson DSL that describes lessons declaratively:

```yaml
lesson:
  title: "Ostinato Fundamentals"
  flow:
    - section: context
      constraints: [strings-only, 80-bpm, c-minor]

    - section: hook
      spotify: "spotify:track:6ZFbXIJkuI1dVNWvzJzown"
      prompt: "Listen to this ostinato from Inception..."

    - section: explain
      content: "An ostinato is a repeated musical phrase..."
      aside:
        type: tip
        content: "Start simple—just 4 notes"

    - section: interactive
      widget: tempo-explorer
      config:
        range: [60, 120]
        examples: [inception, dark-knight]
```

**Pros:**
- Safe — no code execution needed
- Predictable — every lesson follows your patterns
- Cheaper — single LLM call, no agentic loop
- SKILL.md becomes the schema documentation

**Cons:**
- Limited to components you've built
- Custom interactivity requires building new widgets
- Less "magic" — users can't request arbitrary things

**Verdict:** Too limiting. The magic is Claude generating *novel* interactive components, not picking from a menu.

### Option 2: Sandboxed Code Execution

Give each user a sandboxed environment (WebContainer, Sandpack, etc.):

```
┌─────────────────────────────────────────┐
│  User's Browser                         │
│  ┌───────────────────────────────────┐  │
│  │  WebContainer (sandboxed Node.js) │  │
│  │  - File system (virtual)          │  │
│  │  - npm packages (allowlisted)     │  │
│  │  - React runtime                  │  │
│  │  - Your component library         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Claude generates code → runs in sandbox │
└─────────────────────────────────────────┘
```

**Pros:**
- Retains most of the power
- Users can create truly custom things
- Feels like magic

**Cons:**
- Complex infrastructure
- Still need guardrails (what packages? what APIs?)
- Expensive at scale

### Option 3: Hybrid — Schema Core + Widget SDK (Recommended)

**For 90% of lessons:** Use schema approach (BlockNote + custom blocks)

**For custom interactivity:** Sandboxed widget generation with constrained primitives

This is essentially "build your own Claude Artifacts" but domain-specific for education/music.

---

## How Existing Tools Work

### Claude Artifacts

```
┌─────────────────────────────────────────────────────────────┐
│  Claude.ai                                                  │
│                                                             │
│  ┌──────────────────┐    ┌────────────────────────────────┐ │
│  │  Chat Response   │    │  Artifact Preview (iframe)     │ │
│  │                  │    │  sandbox="allow-scripts"       │ │
│  │  ```artifact     │───▶│                                │ │
│  │  type: react     │    │  ┌──────────────────────────┐  │ │
│  │  code: ...       │    │  │ Pre-bundled:             │  │ │
│  │  ```             │    │  │ - React 18               │  │ │
│  │                  │    │  │ - Tailwind (CDN)         │  │ │
│  └──────────────────┘    │  │ - Recharts               │  │ │
│                          │  │ - Lucide icons           │  │ │
│                          │  │ - Transpiler (Sucrase)   │  │ │
│                          │  └──────────────────────────┘  │ │
│                          └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key details:**
- Claude outputs a special artifact block
- Frontend renders a sandboxed iframe
- iframe has pre-bundled React runtime + allowed libraries
- Code transpiled (JSX → JS) and executed inside iframe
- **Allowlisted libraries only** — can't import arbitrary packages
- Communication via postMessage

**What's allowed:**
```javascript
import React, { useState, useEffect } from 'react'
import { LineChart, Line } from 'recharts'
import { Camera, Music } from 'lucide-react'
// Tailwind classes work via CDN
```

**What's blocked:**
```javascript
import axios from 'axios'        // Not bundled
fetch('https://...')             // Network blocked
document.cookie                   // Sandboxed away
window.parent                     // Cross-origin blocked
```

### v0.dev (Vercel)

- Fine-tuned/heavily prompted model trained on shadcn/ui patterns
- Generates code following strict conventions
- Preview sandbox (WebContainer or similar)
- Key insight: model "speaks" a constrained dialect of React

### ChatGPT Canvas / Code Interpreter

**Canvas:** Similar to Artifacts — sandboxed iframe with React

**Code Interpreter:** Actual sandboxed VM (E2B or similar) running Python on server

### Gradio / Streamlit

Python DSL that generates UI. Framework generates HTML/JS from Python code. Less flexible but safer.

---

## The Common Pattern

All share this architecture:

```
1. CONSTRAINED GENERATION
   - Limited set of components/patterns
   - Model trained or prompted to use them
   - Predictable output structure

2. SANDBOXED EXECUTION
   - Iframe with sandbox attribute
   - Or containerized server-side VM
   - Pre-bundled dependencies only
   - No network/storage access

3. ERROR RECOVERY
   - Catch compile/runtime errors
   - Show graceful fallback
   - Optionally retry with error context
```

---

## Proposed Architecture: Sandboxed Widget Runtime

### The Flow

```
User: "I want students to build chords by clicking notes
       on a piano, then hear different inversions"
              ↓
        Claude API (with widget system prompt)
              ↓
        Generated React component
              ↓
   ┌─────────────────────────────────────┐
   │  1. Validate (static analysis)      │
   │  2. Transpile (Sucrase: JSX → JS)   │
   │  3. Execute (sandboxed scope)       │
   │  4. Render (isolated React root)    │
   │  5. Error boundary (catch failures) │
   └─────────────────────────────────────┘
              ↓
        Interactive widget in lesson
```

### Sandbox Security

**Blocked:**
```typescript
const BLOCKED = [
  'window', 'document', 'globalThis',
  'fetch', 'XMLHttpRequest', 'WebSocket',
  'eval', 'Function',
  'localStorage', 'sessionStorage', 'indexedDB',
  'location', 'history', 'navigator',
  'setTimeout', 'setInterval',  // Provide safe versions
  'importScripts', 'import',
]
```

**Static analysis before execution:**
```typescript
function validateCode(code: string): ValidationResult {
  const ast = parse(code)

  walk(ast, {
    MemberExpression(node) {
      if (node.object.name === 'window') {
        issues.push('Cannot access window')
      }
    },
    CallExpression(node) {
      if (node.callee.name === 'fetch') {
        issues.push('Cannot use fetch')
      }
    },
  })

  return { valid: issues.length === 0, issues }
}
```

### The Primitives Library

Claude can compose these freely but can't escape them:

**Layout:**
```tsx
<Row gap="sm|md|lg" align="start|center|end">
<Column gap="sm|md|lg">
<Card variant="default|highlighted">
<Spacer size="sm|md|lg">
```

**Music Visualization:**
```tsx
<Keyboard
  octaves={2}
  highlightedNotes={[60, 64, 67]}  // MIDI numbers
  onNoteClick={(note) => ...}
/>

<Staff
  notes={[{ pitch: 'C4', duration: 'quarter' }]}
  clef="treble|bass"
/>

<PianoRoll
  notes={[{ pitch: 60, start: 0, duration: 1 }]}
  onNoteChange={(notes) => ...}
/>
```

**Audio:**
```tsx
<PlayButton notes={midiNotes} tempo={80} />
playNotes(midiNotes, { tempo: 80, instrument: 'piano' })
stopPlayback()
```

**Theory Helpers:**
```tsx
const notes = parseChord('Cm7')           // → [60, 63, 67, 70]
const inverted = invert(notes, 1)         // → [63, 67, 70, 72]
const name = identifyChord([60, 64, 67])  // → 'C major'
transpose(notes, 5)                        // up 5 semitones
```

**Inputs:**
```tsx
<Slider min={40} max={200} value={tempo} onChange={setTempo} label="BPM" />
<Select options={['Root', '1st', '2nd']} value={inv} onChange={setInv} />
<Button onClick={...} variant="primary|secondary|ghost">
<Toggle checked={...} onChange={...} label="Show names" />
```

**Feedback:**
```tsx
<SuccessMessage>Correct!</SuccessMessage>
<HintMessage>Try adding a seventh...</HintMessage>
<ProgressBar value={3} max={5} />
```

### Making It Work for Non-Programmers

The AI needs to get it right, or iterate invisibly:

```typescript
async function generateWidget(description: string): Promise<WidgetResult> {
  let lastError: string | undefined

  for (let attempt = 0; attempt < 3; attempt++) {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      system: WIDGET_SYSTEM_PROMPT,
      messages: [
        { role: 'user', content: description },
        ...(lastError ? [{
          role: 'user',
          content: `That code had an error: ${lastError}. Please fix it.`
        }] : [])
      ],
    })

    const code = extractCode(response)

    // Validate statically
    const validation = validateCode(code)
    if (!validation.valid) {
      lastError = validation.issues.join(', ')
      continue
    }

    // Try to compile and render
    try {
      const Component = compileWidget(code)
      const testResult = await testRender(Component)
      if (testResult.error) {
        lastError = testResult.error
        continue
      }
      return { success: true, Component, code }
    } catch (error) {
      lastError = error.message
    }
  }

  // Give up — show fallback
  return {
    success: false,
    error: 'Could not generate widget',
    Component: () => <WidgetFallback description={description} />
  }
}
```

**User experience:**
```
User: "I want a tool where students match tempo markings
       to BPM ranges by dragging"

[Loading spinner for 2-3 seconds]

[Interactive widget appears]

[If failed: "I couldn't create that widget. Try describing
 it differently, or choose from these similar ones: ..."]
```

---

## System Prompt for Widget Generation

```markdown
You are generating interactive educational widgets for music education.

## Available Primitives

### React
- useState, useEffect, useMemo, useCallback, useRef
- React.Fragment, <>...</>

### Layout
- <Row gap="sm|md|lg">, <Column>, <Card>, <Spacer>

### Music
- <Keyboard octaves={2} highlightedNotes={[]} onNoteClick={} />
- <Staff notes={[]} clef="treble|bass" />
- <PianoRoll notes={[]} onNoteChange={} />
- <PlayButton notes={[]} tempo={80} />

### Functions
- playNotes(notes, { tempo, instrument })
- parseChord(name) → number[]
- identifyChord(notes) → string
- invert(notes, inversion) → number[]
- transpose(notes, semitones) → number[]

### Inputs
- <Button onClick={} variant="primary|secondary|ghost">
- <Slider min={} max={} value={} onChange={} label="" />
- <Select options={[]} value={} onChange={} />
- <Toggle checked={} onChange={} label="" />

### Feedback
- <SuccessMessage>, <HintMessage>, <ProgressBar value={} max={} />

## Rules
1. Only use primitives listed above — no imports, no DOM APIs
2. Return a single function component as default export
3. All state must be local (no external state)
4. Keep it simple — one clear interaction
5. Use Tailwind classes for additional styling
6. Make it ADHD-friendly: clear feedback, one thing at a time
```

---

## Implementation Path

1. **Build the primitives library** — Package existing components
2. **Build the sandbox runtime** — Sucrase + scoped execution + error boundary
3. **Build the generation flow** — Claude API + validation + retry loop
4. **Integrate with BlockNote** — Custom block type for widgets
5. **Add widget gallery** — Save/share successful widgets

---

## Tiered Product Structure

| Tier | Capability | Technical Approach |
|------|------------|-------------------|
| **Free** | Use pre-built templates, customize content | Schema + curated component set |
| **Pro** | Full lesson creation, all components | Schema + full component library |
| **Business** | Custom widgets, LMS features, analytics | Schema + Widget SDK + sandboxed execution |
| **Enterprise** | White-label, custom integrations | Full sandbox + API access |

---

## Tools to Investigate

- **Sandpack** (CodeSandbox) — Battle-tested sandbox runtime
- **Sucrase** — Fast JSX transpilation (lighter than Babel)
- **WebContainers** (StackBlitz) — Full Node.js in browser
- **es-module-shims** — ES module syntax support
- **Realm / SES** — More sophisticated sandboxing (TC39 proposals)

---

## Key Insight

**Why this can work:**

The magic of the current system isn't the raw code generation — it's the pedagogy system encoded in SKILL.md combined with the ability to create novel interactions.

With a good primitives library and sandboxed execution:
- Claude can generate powerful educational interactions
- Users describe what they want in plain language
- The primitives are domain-specific (music/education), so simple prompts yield rich results
- Sandboxing keeps it safe for non-programmers

The constraint (limited primitives) actually makes it more powerful for the domain, not less.
