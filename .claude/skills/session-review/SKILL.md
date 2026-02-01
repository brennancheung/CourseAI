---
description: Conduct post-session reviews after lessons. Captures learning, struggles, and curiosities through conversation.
---

# Session Review

Use this skill after the user completes a learning session. The goal is to have a natural conversation that captures what happened, then save useful notes for future reference.

## When to Use

- User says they just finished a lesson
- User wants to debrief on a study session
- User mentions something clicked or something was frustrating

## The Conversation Flow

### 1. Open with Context

Start by understanding what they worked on:

```
What lesson did you work on?
How long did you spend with it?
```

If you know what they've been working on recently (from previous sessions), reference that:

```
Last time you were working on attention mechanisms. Did you continue with that, or try something new?
```

### 2. Explore What Happened

Ask open-ended questions. Don't rush through a checklist — follow their energy:

**If they seem excited:**
- "What clicked for you?"
- "What made that moment feel good?"
- "Did anything surprise you?"

**If they seem frustrated:**
- "Where did you get stuck?"
- "What were you trying to understand that wasn't working?"
- "Was the lesson too hard, or was it something else?"

**If they seem neutral:**
- "What did you notice?"
- "Did anything feel different from last time?"
- "Was there a moment where you had to think hard?"

### 3. Dig Into Struggles

Struggles are valuable data. When they mention something hard, explore it:

- "What specifically was hard about it?"
- "Did you try different approaches?"
- "Do you have a theory about why it wasn't clicking?"
- "Is this something you want to work on more, or move past?"

### 4. Surface Curiosities

Find out what they want to learn next:

- "Is there anything you're curious about now?"
- "Did this session make you want to try something specific?"
- "What would make the next session feel productive?"

### 5. Close the Loop

Summarize what you heard and confirm:

```
So it sounds like:
- The basic concept is starting to make sense
- The math notation is still tricky
- You're curious about how to implement this from scratch

Does that capture it?
```

## Saving the Session

After the conversation, save a Markdown file to `src/data/sessions/`.

### File Naming

Use the format: `YYYY-MM-DD-<slug>.md`

If multiple lessons or a general session: `YYYY-MM-DD-session.md`

Examples:
- `2026-01-17-attention-mechanism.md`
- `2026-01-17-session.md`

### File Format

```markdown
# Session: [Lesson Title or "Study Session"]
Date: YYYY-MM-DD
Lesson: [lesson-slug or "multiple" or "exploratory"]
Duration: [if mentioned]

## Summary
[2-3 sentence summary of what happened]

## What Worked
- [Bullet points from conversation]

## What Was Hard
- [Bullet points from conversation]
- [Include specific details that might inform future lessons]

## Curiosities
- [What they want to learn more about]
- [Questions that came up]

## Insights
[Any patterns you noticed, connections to previous sessions, or observations
that might be useful for lesson planning]

## Next Steps
- [Specific things to try next time]
- [Lessons that might address struggles]
```

### Example

```markdown
# Session: Attention Mechanism
Date: 2026-01-17
Lesson: attention-mechanism
Duration: ~30 min

## Summary
First deep dive into the attention mechanism. Got the intuition for queries, keys,
and values but struggled with the matrix math notation.

## What Worked
- The analogy to database lookups clicked immediately
- Interactive visualization helped see what softmax does
- The "why" explanation was motivating

## What Was Hard
- Matrix multiplication notation (transposing K)
- Not clear why we divide by sqrt(d_k)
- Implementing from scratch in PyTorch

## Curiosities
- How does multi-head attention work?
- Why do we need position encoding?
- How is this different from RNN attention?

## Insights
Ready for a lesson on multi-head attention. The single-head concept is solid now,
but the matrix notation needs more practice.

## Next Steps
- Try implementing attention from scratch in PyTorch
- Watch 3Blue1Brown video on matrix multiplication
- Move to multi-head attention lesson
```

## Reading Previous Sessions

Before starting a review, check for recent sessions:

```bash
ls -la src/data/sessions/
```

Reference previous sessions in the conversation when relevant:
- "Last week you mentioned X was hard — how is that feeling now?"
- "You were curious about Y — did you explore that?"

## Updating Learner State

After saving the session, consider updating `src/data/learner-state.ts` if:
- A skill level has clearly changed
- A new struggle has emerged
- A previous struggle has been resolved
- A strong preference has been discovered

Don't update on every session — only when there's a meaningful shift.

## Tips

- **Follow their energy.** If they want to vent about frustration, let them. If they're excited, celebrate with them.
- **Don't interrogate.** This is a conversation, not a form.
- **Capture specifics.** "The math was hard" is less useful than "the matrix transpose in Q·K^T was confusing."
- **Connect sessions.** Reference what you know from before. Show that this is a journey.
- **End with direction.** They should leave knowing what to try next.
