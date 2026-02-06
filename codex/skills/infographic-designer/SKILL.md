---
name: infographic-designer
description: Use when creating infographics, data visualizations, or visual explanations of data. Applies Edward Tufte principles to ensure visual communication over mere visual appeal. Forces systematic consideration of data, encoding, and comparison before aesthetic decisions.
---

# Infographic Designer

Your objective is to create infographics that **communicate data clearly**, not just look attractive. Visual appeal is secondary to visual communication.

## Core Principle

**If the viewer cannot make comparisons at a glance, the infographic has failed.**

---

## Phase 1: Purpose & Data (Do This First)

### 1.1 Define Purpose

Before any visual work, answer these questions explicitly:

```
PURPOSE:
- What question does this answer?
- Who is the audience?
- What action should they take after viewing?
- What is the "aha moment" — the core insight?
```

### 1.2 Gather Data

- Use multiple sources to cross-validate
- Distinguish between theoretical maximums, practical ranges, and optimal zones
- Document the data sources
- Note edge cases and exceptions

**Output:** A structured data summary before any visual discussion.

---

## Phase 2: Reference Frame

### 2.1 Establish Common Axis

**Non-negotiable:** Data must be comparable on a shared reference frame.

Ask:
- What is the primary dimension for comparison?
- What reference frame is familiar to the audience?
- How should items be aligned for easy visual scanning?

**Examples:**
| Data Type | Good Reference Frame |
|-----------|---------------------|
| Musical pitch | Piano keyboard (A0-C8) |
| Time | Calendar/timeline with consistent intervals |
| Geography | Map with scale |
| Hierarchy | Org chart with levels |
| Quantity | Bar chart with zero baseline |

### 2.2 Choose Orientation

- **Horizontal:** Good for timelines, ranges, processes
- **Vertical:** Good for hierarchies, rankings, comparisons
- **Radial:** Good for cyclical data, proportions
- **Grid:** Good for matrices, two-dimensional comparisons

---

## Phase 3: Visual Encoding

### 3.1 Map Data to Visual Properties

Each data attribute should map to ONE visual property:

| Data Attribute | Visual Property Options |
|----------------|------------------------|
| Category/type | Hue (color family) |
| Quantity/extent | Length, area, position |
| Intensity/quality | Saturation, value (light/dark) |
| Certainty/range | Opacity, blur, gradient |
| Importance | Size, weight, contrast |

**Rule:** Do not double-encode. If hue shows category, don't also use it for intensity.

### 3.2 Consider Multi-Dimensional Data

If data has multiple attributes (e.g., range + category + quality zone):

```
ENCODING PLAN:
- Dimension 1 (primary): [visual property]
- Dimension 2 (secondary): [visual property]
- Dimension 3 (tertiary): [visual property]
```

### 3.3 Explore Representation Options

Don't default to the first chart type. For any data, consider:

**For ranges/intervals:**
- Bars (simple, clear)
- Dots with connecting lines (emphasizes endpoints)
- Gradients (shows intensity variation within range)
- Waveforms/envelopes (organic, appropriate for audio/music)
- Connected shapes (shows overlap patterns)

**For comparisons:**
- Side-by-side (direct comparison)
- Overlaid/superimposed (shows intersection)
- Small multiples (individual + pattern)
- Stacked (shows composition)

**Ask:** "What would make this data *feel* like what it represents?"

---

## Phase 4: Data-Ink Audit

### 4.1 Maximize Data-Ink

Every visual element should encode information. For each element, ask:
- What data does this represent?
- If I remove it, what information is lost?

### 4.2 Eliminate Chartjunk

Remove or minimize:
- Decorative elements that don't encode data
- Redundant labels
- 3D effects that distort perception
- Gradient backgrounds that don't represent data

### 4.3 Purposeful Decoration

Decoration is allowed when it:
- Reinforces the subject matter
- Aids recognition or memory
- Provides visual rhythm without obscuring data

---

## Phase 5: Structured Prompt (For AI Generation)

When generating infographics via AI, use structured prompts:

```json
{
  "format": "[aspect ratio and orientation]",
  "subject": "[what the infographic shows]",
  "layout": {
    "reference_frame": "[the common axis or grid]",
    "organization": "[how items are arranged]"
  },
  "visual_encoding": {
    "dimension_1": "[what visual property encodes what data]",
    "dimension_2": "[what visual property encodes what data]"
  },
  "style": "[aesthetic direction that reinforces content]",
  "must_include": ["[specific elements that must appear]"],
  "must_avoid": ["[common mistakes to prevent]"]
}
```

---

## Phase 6: Critique Loop

After each version, evaluate:

### Comparison Test
- [ ] Can I compare the primary dimension at a glance?
- [ ] Is the reference frame clear and consistent?
- [ ] Are similar items visually similar, different items visually different?

### Communication Test
- [ ] What's the first thing the eye sees? (Should be the core insight)
- [ ] Where does the eye get lost or confused?
- [ ] What question does this raise that it doesn't answer?

### Efficiency Test
- [ ] Can I remove any element without losing information?
- [ ] Is any information double-encoded unnecessarily?
- [ ] Is the cognitive load appropriate for the audience?

---

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| No common axis | Can't compare | Add shared reference frame |
| Pretty but empty | Visual appeal without communication | Purpose first |
| Decoration as data | Misleads viewer | Separate or remove |
| Information overload | Overwhelms | Progressive disclosure or simplify |
| Default chart type | Missed opportunity | Explore alternatives |
| Single-source data | Risk of inaccuracy | Cross-validate |
| Inconsistent encoding | Confuses | One meaning per visual property |

---

## Output Structure

When designing an infographic, provide:

1. **Purpose Statement** — What this communicates and to whom
2. **Data Summary** — The information to be visualized
3. **Encoding Plan** — How each dimension maps to visual properties
4. **Reference Frame** — The common axis enabling comparison
5. **Representation Choice** — Why this visual form (not just "it's a bar chart")
6. **Structured Prompt** — JSON or detailed specification for generation
7. **Critique Notes** — What to watch for in the output

---

## Quick Reference

**The hierarchy:**
1. Can they compare? (reference frame)
2. Can they understand? (encoding)
3. Is it efficient? (data-ink)
4. Is it appealing? (style)

**Never sacrifice 1-3 for 4.**
