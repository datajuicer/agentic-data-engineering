# GRPO RFT reward artifact workflow

## Choose one controlled intervention

Read the current Plan snapshot and `input/reference/manifest.json` first. Edit
the declared primary `reward.py`; use donors only for the accepted idea. Keep
answer extraction, `TrainingOutcomeAdapter`, evaluation, ground truth,
checkpoint selection, trainer, and infrastructure fixed.

The Plan identifies a credit defect and falsifiable evidence hypothesis. The
Builder chooses how the available fixed evidence dimensions test that
hypothesis. With Group Credit disabled it may derive an `artifact_projection`
and change the per-row score. With Group Credit enabled the row result remains
outcome-only and group policy consumes raw dimensions directly to intervene,
tie, or abstain.
Do not invent evidence schemas, dimension names, metadata, or a second switch.

## Implement the row reward

Keep the fixed six-argument `compute_score` and `compute_fallback_score` ABI.
Return a finite bounded `ade.reward_result.v2` score and preserve the
Engine-authoritative outcome. A normal result need not contain exactly three
scalar components and need not be a direct non-negative weighted sum.

When `judge_enrichment.enabled=true` and Group Credit is disabled, await the
injected Judge exactly once on every valid path, including zero outcome, and
use its fixed structured result without overwriting raw dimensions. When Group
Credit is enabled, do not declare or reference `llm_judge`; Engine acquires the
evidence before row scoring and later supplies the raw bank to
`assign_group_credit`. When enrichment is false, do not declare or reference
the capability. Never catch broad exceptions around candidate code. Artifact/schema,
non-finite, and programming errors fail admission or the Attempt.

`compute_fallback_score` is synchronous, deterministic, calls no Judge, and
returns the authoritative outcome as `score`. Engine uses it only for the typed
single-row unavailable condition while Group Credit is disabled; it is not a
catch-all for Artifact failures.

## Implement Group Credit only when enabled

When `resolved.training.rft.group_credit.enabled=true`, implement the
synchronous pure `assign_group_credit(group_input)` v3 input entrypoint from the
output contract. It sees complete authoritative siblings, including process
availability, raw fixed dimensions, rule evidence, and authoritative response
length. Return keyed scalar training
rewards, not advantages.

If any row lacks process evidence, the policy still runs but must not declare
use of process dimensions. It may use outcome/rule/length, keep a tie, or
abstain. Keep decisions deterministic and invariant to record order and opaque
ID relabeling. For disabled or absent Group Credit, only row entrypoints are
active and the inherited dormant group function is not an intervention.

## Record and validate the experiment

Probe correct, incorrect, malformed, empty, adversarial, all-wrong, mixed,
all-correct, unavailable-evidence, reordered, and relabeled-ID cases applicable
to the active profile, including the Plan's primary exploit path. The goal is
task-aligned realized advantage and matched
downstream validation, not reward density, a high process score, or merely
creating non-flat groups.

`design.md` states the hypothesis, intervention, expected evidence, refutation
condition, fixed controls, and actual implementation. Write only `reward.py`
and `design.md`; Harness owns admission, compliance execution, receipts, and
mechanical metadata.

On reflection, compare direct-reference and candidate row reward, assigned
reward, realized advantage, sign changes, Judge availability, and declared
expectation results on the exact persisted groups. Make the smallest
artifact-correctable revision. Preserve an unavailable external evidence result
as uncertainty rather than changing bindings or inventing a fallback source.
