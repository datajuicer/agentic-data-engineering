# Input contract

Require `input/manifest.json`, `input/action.json`, and `input/task.json` as
UTF-8 JSON. Require schema version `1`, role `coordinator`, task ID
`reward_design`, run ID, subject ID, basis revision, and declarations for all
context files.

Require `action.json` to contain the non-empty `target_plan_id` allocated by
ADE Runtime. Copy it exactly into both decision outputs. Use only Plans declared
as completed with a current best artifact when selecting historical relations.

Require `task.json` to provide the reward objective, allowed signals, numeric
constraints, supported sample/response inputs, portfolio limits, and allowed
references. Read declared run, Insight Graph, ranking, replay, prior-plan, and
artifact views when present.

Require `subject/portfolio-target.json` and `subject/ranking.json` to carry the
same frozen Ranking revision. Use them to reason about the expected observation,
but do not copy any comparator identity. Harness binds the representative Trial
of the Plan relation's primary source as the hypothesis comparator and binds the
frozen portfolio target directly. A missing secondary score remains an explicit
unavailable metric; it does not invalidate an otherwise comparable primary
score.

`subject/PLAN_CATALOG.md` is a Run-global projection, not learned Memory. Its
accepted directions include the immutable `plan.md` for every accepted Plan,
whether active or terminal. Use it to avoid blindly repeating a controlled
direction. Only completed source-eligible Plans may be used as relation
sources; active directions contain no accepted experimental finding.

For GRPO RFT, require the rollout-group size and sampling policy, score bounds,
extraction/equivalence rules, available Engine-owned evidence, Artifact
projection and final-reward facts, realized GRPO advantage and outcome-only
counterfactual when available, replay fixtures, exploit probes, and an
independent evaluation metric that is not merely the optimized reward.
Unavailable evidence must be declared instead of synthesized. Do not require a
scalar process component or a three-component weighted-sum design.

Treat declared extraction/equivalence rules, the versioned
`TrainingOutcomeAdapter`, evaluation parsers and graders as read-only fixed
controls, not proposal inputs. A Plan must be implementable solely through the
fixed reward ABI and must not depend on changing those controls. Extraction or
grading defects remain explicit measurement caveats until separately versioned
outside the reward-design Run.

Reject inconsistent identities, stale revisions, missing declarations,
operator-only/test data, or a proposal that depends on unavailable signals.
