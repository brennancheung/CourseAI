# Codex Migration Notes

This folder contains Codex-oriented copies of existing Claude guidance/skills.

## What was ported
- Repo Claude skills copied to `codex/skills/*`
- Global Claude skills copied into Codex skill folders where practical
- Claude command playbooks (`commit`, `docsync`) converted into Codex skills
- Repo `AGENTS.md` created to expose guidance and skill triggers

## Non-destructive guarantee
No Claude files were modified or removed.

## Optional next step (global install)
If you want these available across all repos, copy desired skill folders into:
`~/.codex/skills/<skill-name>/`
Then restart Codex.
