---
name: add-markdown-playbook
description: Install repository markdown rendering components using the existing Claude playbook script. Use when user asks to add markdown support (GFM, KaTeX, syntax-highlighted code blocks).
---

# Add Markdown Playbook

Reuse the existing repository playbook instead of rebuilding markdown components manually.

## Trigger

Use this skill when the user asks for:
- markdown rendering support
- GFM tables/task lists
- KaTeX math blocks
- syntax-highlighted code fences

## Workflow

1. Review the playbook docs:
   - `.claude/playbooks/add-markdown/README.md`
2. Run the installer from repo root:
   - `bash .claude/playbooks/add-markdown/install.sh`
3. Validate with project checks:
   - `pnpm ai`

## Notes

- The installer copies components from `.claude/playbooks/add-markdown/files/` into `src/components/markdown/`.
- Prefer this playbook for consistency with existing Claude Code workflows.
- Do not run `pnpm dev`; ask the user to run it if UI verification is needed.
