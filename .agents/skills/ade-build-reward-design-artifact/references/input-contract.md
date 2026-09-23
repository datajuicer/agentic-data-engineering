# Input contract

Require schema version `1`, role `artifact_builder`, task ID `reward_design`,
run ID, Coordinator ID, Plan ID, Trial ID, basis revision, and declarations
for every input.

Require `input/action.json#source_artifact_ref_ids` as a non-empty unique
string list set by ADE Harness. For the first Trial it contains the frozen
Plan-relation seed artifacts. For every later Trial it contains exactly the
preceding Trial artifact. This binding is immutable.

Require `input/reference/manifest.json` to resolve those IDs without guessing.
The first ordered source is the primary editing reference and appears at
`input/reference/primary/reward.py`; any remaining ordered sources are donors
under `input/reference/donors/`. A `new_direction` first Trial uses the p000
best reward. A `revisit` first Trial uses the target Plan's best reward. A
`combine` first Trial uses its declared primary first and the other sources as
donors. A later Trial uses the immediately preceding Trial reward.

Require `input/manifest.json`, `input/action.json`, `input/task.json`,
`input/task/resolved.json`, and an accepted immutable snapshot under
`input/snapshots/current-plan/`. When frozen seed Plans are declared, require
their immutable snapshots under `input/snapshots/seed-plans/`. Treat all
snapshots as read-only evidence and never search outside declared paths.

Require the six runtime arguments `question_prompt`, `response_content`,
`extracted_answer`, Engine-fixed `outcome_score`, `response_length_tokens`, and
`max_response_length_tokens`; GRPO algorithm identity; the fixed process
evidence adapter/dimensions when enabled; finite score bounds `[0.0, 1.0]`;
exploit probes; and the independent offline evaluation metric. Do not require
three scalar components or a direct weighted-sum formula.

Treat `extracted_answer` and `outcome_score` as Engine-fixed inputs. The
Builder must not implement, replace, or repair the `TrainingOutcomeAdapter`,
answer extraction/equivalence, evaluation parser/grader, ground truth, or
checkpoint selection inside `reward.py`. Reject a Plan whose intervention
requires any of those fixed controls to change.

When Judge enrichment is enabled with Group Credit disabled, require the
trusted asynchronous `llm_judge` capability backed by `ade.rubric_jobs.v1` and
the task-owned `math_process_evidence.v1` adapter. It is the only external
capability available to `reward.py` and is injected by the trusted Engine
adapter. With Group Credit enabled, Engine owns acquisition and does not expose
this capability to `reward.py`; the Artifact receives raw evidence only in the
v3 group input.
It is not an input file, provider client, credential, or directly accessible
MCP tool.

Require `input/compliance/manifest.json` and
`input/compliance/records.jsonl`. This immutable suite uses the accepted
artifact package of the current Trial's actual direct primary source Trial. A
later Trial binds its immediately preceding accepted Trial, even when its code
bytes originated earlier; a combine Trial binds its declared primary, not a
donor. Require one latest eligible `rl_step` with exactly 32 complete groups of
the resolved `rollout_n`; do not combine positions or silently substitute
p000/another Trial. The deterministic selector covers each source-present
shape once in fixed order, allowing one group to satisfy multiple labels, then
fills by ascending `details.prompt_group_id`. It is realization input, not
scientific evidence for choosing a reward direction. Missing direct rollout
is explicitly unverified.

When declared, require `input/reflection/round.json`, prior delivery,
`source-groups.jsonl`, `group-results.jsonl`, `judge-results.jsonl`,
`judge-manifest.json`, and `realization-report.json`. Source records and
runtime/Judge binding remain frozen, and completed Judge results are reused.
Do not confuse reflection with validation retry. The reflection response must
explicitly choose `finalize` or `revise` in `design.md`; observed effect sizes
and directions are not admission targets.

The Builder never owns a Local Rubric Judge client. Harness freezes the
task-owned adapter/rubric/schema binding and submits the bounded compliance job. The
Builder may not call Review Labor, Local Judge, an MCP Judge, or a provider
directly.
