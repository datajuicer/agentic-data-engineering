# Run Summarizer input guide

The Run Summarizer receives a Run-scoped immutable package for one queue-head
PM update. Read `input/manifest.json` first; it declares every accepted source
and its identity. A path not declared there is unavailable.

- `input/action.json`: fixed Run identity and the run-summary action.
- `input/task.json`: task identity and fixed data-selection boundary.
- `input/memory/MEMORY.md`, `outcomes/`, and `findings/`: the complete parent
  Run Memory. Preserve
  valid knowledge and add the queue-head update with source attribution.
- `input/subject/PLAN_UPDATE.md` and `input/subject/plan-memory/MEMORY.md`:
  the one queue-head Plan Memory update being merged. Its `outcomes/` and
  `findings/` are the complete Plan-local history.
- `input/run/PLAN_CATALOG.md` and `input/run/RANKING.md`: Harness-owned Plan
  catalog and ranking facts bound by `input/run/manifest.json`. They are context and authority
  projections, not prose to rewrite.
- `input/retry.json` and `input/prior-delivery/`: retry reason and prior Run
  Memory. Preserve unaffected content.

This is a derived Run Memory view, not state authority. Do not read raw Trial
workspaces, Engine logs, Review scratch, operator-only data, or undeclared
sources. Write only `output/MEMORY.md`.
