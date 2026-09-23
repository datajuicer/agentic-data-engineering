# Input contract

Require these UTF-8 JSON files:

- `input/manifest.json`
- `input/action.json`
- `input/task.json`

Require manifest schema version `1`, role `coordinator`, task ID
`data_selection`, run ID, subject ID, basis revision, and a declaration for
every other input file. Require `action.json` to repeat the same identity and
request one planning decision with a non-empty `target_plan_id` allocated by
ADE Runtime. Require `task.json` to define selection
objective, portfolio limits, dataset constraints, and allowed references.

For Long-CoT SFT, the declared inputs must expose stable problem and trajectory
IDs, tokenizer-based token counts, correctness or reasoning-validity evidence,
source/domain/language/difficulty metadata, duplicate-group identity,
contamination and truncation signals, and the atomic selection unit.
Realization statistics are evidence rather than constraints. If a field is
unavailable, `task.json` must say so;
absence is an uncertainty, not permission to invent it.

Read declared run, insight, ranking, prior-plan, and artifact views when
present. Treat only accepted insights and artifact references as evidence.
Require `input/subject/portfolio-target.json` to have the same Ranking revision
as `subject/ranking.json`. Use those views to reason about expected outcomes,
but do not copy comparator identities. Harness binds the relation primary
source's representative Trial and the frozen portfolio target directly. A
missing secondary score is an unavailable metric, not grounds to reject an
otherwise comparable primary result.
`subject/PLAN_CATALOG.md` separately projects every accepted Plan direction,
including active Plans without results. Use those directions only to avoid a
blindly repeated proposal; they are not learned Memory or accepted findings.
Use only Plans declared as completed with a current best artifact when choosing
a historical relation. Never infer a missing Plan or artifact reference.

The latest run state exposes the Harness-owned `budget` view. Read it before
choosing a relation and record the planning posture in `plan.md`: total and
remaining search Plans, total and remaining search Trials, the current Plan's
allocated and remaining Trial budget, and the configured per-Plan bounds. The
Coordinator may choose exploration or exploitation from these facts, but may
not change, reserve, or reinterpret the budget. A relation is not a budget
allocation. `decision.json` carries only the hypothesis reason and expected
observation; Harness owns both comparator identities, frozen metric values, and
admission checks.
Reject mismatched identities, undeclared context, stale revisions, missing
declared files, or operator-only/test data.
