# Plan Summarizer input guide

The Plan Summarizer receives one Plan-scoped immutable package. Read
`input/manifest.json` first; it declares every accepted source and its
identity. A path not declared there is unavailable.

- `input/action.json`: fixed Run/Coordinator/Plan identity and the queue-head
  summary action.
- `input/task.json`: task identity and fixed data-selection boundary.
- `input/memory/MEMORY.md`, `outcomes/`, and `findings/`: the complete parent
  Plan Memory. Preserve
  still-valid parent knowledge and distinguish it from current findings.
- `input/subject/plan.md`: the immutable pre-experiment Plan commitment.
- `input/subject/trial/`: only the current Trial's objective outcome and, when
  accepted, Analyzer `analysis.md`, `findings.md`, evidence and coverage;
  failures remain explicit `failure.json` facts. Its `realization/` and
  `comparisons/` directories contain final realized behavior, frozen planning
  comparators, and Harness-owned objective relations.
- `input/retry.json` and `input/prior-delivery/`: retry reason and previous
  memory. Preserve unaffected content.

This is a derived summary view, not state authority. Do not read raw Review
scratch, Engine logs, private Trial workspaces, operator-only data, or
undeclared sources. Write only `output/MEMORY.md`.
