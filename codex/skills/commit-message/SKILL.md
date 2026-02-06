---
name: commit-message
description: Analyze current git changes and craft a clear, structured commit message (and run commit only when explicitly requested). Use when user asks to write a commit message, prepare a commit, or improve commit quality.
---

# Commit Message

Create high-quality commit messages from actual repository changes.

## Process

1. Analyze all changes
   - Run `git status`, `git diff --staged`, and if needed `git diff`
   - Review recent commits (`git log -3 --oneline`) for style/context
2. Draft commit message
   - Use conventional commit type (`feat`, `fix`, `docs`, `refactor`, etc.)
   - Keep subject imperative and concise
   - Explain what and why in body
   - Group notable changes into clear categories when useful
3. Commit only with explicit user approval
   - Never commit unless user asks for it
   - Never push unless user asks for it

## Hard Rules

- Never push automatically
- Never add AI attribution in commit messages
- Be factual; describe real changes

## Suggested Format

```text
<type>(<scope>): <subject>

<what changed and why>

<optional grouped bullets by area>
```

## Quality Checks

- Subject is imperative (`add`, `fix`, `update`)
- Scope is specific when helpful
- Body captures rationale, not just mechanics
- Message is readable in `git log --oneline` and full log view
