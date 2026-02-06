# AGENTS.md

## Project Guidance (Codex)

### Core Rules
- Never run `pnpm dev` yourself; ask the user to run it.
- Prefer `pnpm` over `npm` (and `pnpm dlx` over `npx`).
- Prefer `pnpm ai` for checks.
- Use early returns; avoid `else if` / `else` where practical.
- Avoid `switch` statements; use helper dispatch functions.
- Avoid `any` in TypeScript.
- Never commit or push without explicit user permission.
- Never add AI/assistant attribution in commit messages.

### Architecture & Stack
- Next.js App Router
- Convex backend
- Tailwind CSS + shadcn/ui
- Single-user/local-first learning app

## Skills
A skill is a local instruction set in `SKILL.md`. Use these when explicitly named or when the task clearly matches.

### Available skills
- lesson-planning — ADHD-friendly AI/ML lesson planning and implementation patterns. Use for lesson design, pedagogy, and lesson structure decisions. (file: `codex/skills/lesson-planning/SKILL.md`)
- interactive-widgets — Add interactive visual learning components and identify “aha” opportunities. (file: `codex/skills/interactive-widgets/SKILL.md`)
- session-review — Conduct post-session learning debriefs and save structured session notes. (file: `codex/skills/session-review/SKILL.md`)
- css-expert — Systematic CSS/layout debugging via parent-child constraint tracing. (file: `codex/skills/css-expert/SKILL.md`)
- problem-solver-debugger — Hypothesis-first debugging and uncertainty-aware root-cause analysis. (file: `codex/skills/problem-solver-debugger/SKILL.md`)
- infographic-designer — Data-first infographic design with explicit encoding and comparison logic. (file: `codex/skills/infographic-designer/SKILL.md`)
- commit-message — Craft structured conventional commit messages from real diffs. (file: `codex/skills/commit-message/SKILL.md`)
- docsync — Audit docs against implementation and propose consolidation/update plans. (file: `codex/skills/docsync/SKILL.md`)

## Skill Trigger Rules
- If user names a skill (e.g., `$lesson-planning`) or asks for a task that clearly maps to one, use that skill.
- Read only the needed files from each skill (progressive disclosure).
- If multiple skills apply, use the minimal set and state order briefly.
- If a skill cannot be loaded, say so and proceed with best fallback.

## Notes
- This file is Codex guidance and does not replace Claude files.
- Claude files under `.claude/` and `CLAUDE.md` remain untouched.
