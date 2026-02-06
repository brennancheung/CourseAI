---
name: docsync
description: Audit and synchronize docs with code reality. Use when finding stale docs, orphaned docs, undocumented features, or consolidating scattered implementation notes into production architecture docs.
---

# Documentation Sync

Keep documentation accurate, current, and maintainable.

## Core Workflow

1. Scope the target
   - Full repo audit or a specific feature area
2. Inspect docs and code together
   - Identify orphaned docs (no matching code)
   - Identify undocumented code paths/features
   - Identify stale references (renamed paths/APIs)
3. Propose lifecycle actions
   - Keep, update, consolidate, archive
4. Produce concrete output
   - Findings list with file references
   - Recommended edits and consolidation plan

## Documentation Maturity Model

- Development docs: exploration notes, TODOs, design spikes
- Production docs: stable architecture, concepts, trade-offs, extension points

When feature stabilizes, consolidate dev artifacts into one architecture doc and preserve the decision rationale.

## Report Template

- What is accurate
- What is stale
- What is missing
- What should be consolidated
- Next highest-impact updates

## Guardrails

- Prefer factual verification from code over assumptions
- Preserve important decision history (“why”), not only current mechanics
- Avoid deleting historical context unless clearly obsolete
