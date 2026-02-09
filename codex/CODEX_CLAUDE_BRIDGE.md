# Codex Claude Bridge

This repo keeps Claude assets as source-of-truth and exposes them to Codex.

## Canonical instruction files

- `AGENTS.md` is the Codex instruction entrypoint.
- `AGENT.md` is a compatibility pointer to `AGENTS.md`.
- `CLAUDE.md` remains the project context reference.

## Skill alignment

- Claude skill content under `.claude/skills/` can be mirrored to `codex/skills/`.
- If both exist, prefer the `codex/skills/` path for Codex triggers.

## Playbook reuse

- Reuse installer/playbook scripts under `.claude/playbooks/` when relevant.
- Current bridge skill:
  - `codex/skills/add-markdown-playbook/SKILL.md` -> `.claude/playbooks/add-markdown/install.sh`

## Maintenance

When a `.claude/skills/*` skill is updated, sync the matching `codex/skills/*` copy so Codex uses current guidance.
