---
name: ade-build-reward-design-artifact
description: Build one executable GRPO RFT reward experiment from an accepted ADE Plan snapshot, using the task-owned fixed Judge evidence adapter when enabled and keeping Engine-owned evidence separate from Artifact reward logic. Use only when ADE Agent Runtime invokes the Artifact Builder with the current Plan snapshot, frozen seed Plan snapshots, and resolved task configuration.
---

# Build Reward Design Artifact

Produce one bounded reward function by editing the declared reference reward,
then perform a small development probe when the Judge tool is available.

Read `input/manifest.json`, `input/action.json`, `input/task.json`, files under
`input/snapshots/current-plan/`, any declared
`input/snapshots/seed-plans/`, `input/task/resolved.json`,
`input/reference/manifest.json`, `input/reference/primary/reward.py`, optional
declared donor rewards under `input/reference/donors/`, and the files under
`input/compliance/`. Do not inspect test data, Engine logs, undeclared code, or
another Agent workspace.

The realization suite contains exactly 32 complete prompt groups from the
current Trial's direct primary source Trial at the latest eligible published
`rl_step`. It never traces identical code back to its earliest producer and
never silently falls back to p000 or another Trial. Treat the supplied source
records, deterministic shape coverage, current runtime/Judge binding, and
direct-reference results as authoritative.

Answer extraction/equivalence, the versioned `TrainingOutcomeAdapter`,
evaluation parsers/graders, ground truth, and checkpoint selection are given
Task/Engine fixed controls. Do not reimplement, repair, replace, or bypass them
in `reward.py`; reject a Plan that requires such a change.

Scope contract: change only the accepted reward function and its declared use
of fixed evidence dimensions. Training, model, optimizer, evaluation/extraction, checkpoint, and
infrastructure controls are immutable and must not become fallback logic or
additional reward signals. The required experiment-level
`judge_enrichment.enabled` flag controls Judge use: when false, do not declare
or call Judge. When true with Group Credit disabled, the artifact executes
exactly one Judge call on each valid reward path. When true with Group Credit
enabled, Engine acquires the evidence exactly once and `reward.py` must not
declare or call `llm_judge`; it consumes the resulting raw bank only through
`assign_group_credit`.

## Execute

1. Read `references/input-guide.md` and `references/input-contract.md`, then
   validate the package identity.
2. Read `references/llm-judge-contract.md`, then follow
   `references/artifact-workflow.md`. Start from the primary reference reward
   and make only the controlled intervention accepted by the Plan. Select from
   the task-owned fixed evidence dimensions. With Group Credit disabled, decide
   their optional projection/use in per-row reward; with Group Credit enabled,
   when process evidence is used, consume the raw dimension bank directly in
   group assignment according to the accepted Plan. Do not define a new evidence schema,
   dimension name, arbitrary metadata, or a second Judge switch.
3. Write `output/design.md` using the expected outline in
   `references/output-contract.md`. Include natural observations from any
   development probe, but do not create a probe sidecar.
4. Write `output/reward.py` with the fixed six-argument entrypoints. Their
   callable kinds are part of the ABI: `compute_score` must remain
   `async def`; `compute_fallback_score`
   must be a synchronous `def` and must never call the Judge. When resolved
   training enables
   `rft.group_credit`, also implement the synchronous pure
   `assign_group_credit(group_input)` entrypoint. Its v3 input ABI is fixed and must
   preserve nested Engine evidence; do not infer alternate field names, add
   metadata, or flatten the records:

   ```python
   group_input == {
       "schema_version": "ade.group_credit_input.v3",
       "group_type": "all_correct" | "mixed" | "all_wrong",
       "records": [{
           "record_id": int,                 # opaque key only
           "is_correct": bool,                # authoritative correctness
           "outcome_score": float,            # authoritative outcome scalar
           "pre_group_reward": float,
           "evidence": {
               "outcome": {"status": "authoritative", "value": float},
               "process": {
                   "status": "available" | "unavailable" | "not_configured",
                   "adapter_id": str | None,
                   "schema_version": str | None,
                   "dimensions": dict[str, float] | None,
               },
               "rule_based": {"status": str, "value": float | None},
               "response_length": {
                   "status": "authoritative", "value": int, "maximum": int,
               },
           },
       }],
   }
   ```

   In particular, authoritative outcome is `record["outcome_score"]`, not
   `record["outcome"]`; component evidence is read from
   `record["evidence"][name]["status"]` and `...["value"]`. Return exactly:

   ```python
   {
       "schema_version": "ade.group_credit_output.v2",
       "decision": {
           "mode": "identity" | "abstained" | "shaped",
           "evidence_sources": [str, ...],
           "process_dimensions": [str, ...],
           "reason_code": str,               # lowercase
       },
       "assignments": [
           {"record_id": int, "training_reward": float}, ...
       ],
   }
   ```

   `evidence_sources` and `process_dimensions` use Engine-fixed enums, not
   free-form labels. A declared process dimension must exist and be available
   on every record. If any row has unavailable or not-configured process
   evidence, do not declare a process dimension; the complete group still
   enters the policy, which may use outcome/rule/length, keep a tie, or abstain.

   Identity and abstained assignments must equal `outcome_score` exactly;
   shaped assignments must be finite, within `[0, 1]`, and change at least
   one reward. The assignment must be pure, deterministic, complete, and
   invariant to record order and opaque-ID relabeling. Before delivery,
   execute a local synthetic probe using the exact nested input above,
   including all-wrong, mixed, all-correct, unavailable-evidence, reordered,
   and relabeled-ID cases. Jointly design the per-response evidence
   and group assignment to produce a task-aligned realized-GRPO-advantage
   intervention and matched downstream improvement; changing raw reward or
   merely producing non-zero normalized advantage is not the objective. Keep
   all entrypoints consistent with `design.md`.
   Treat `resolved.training.rft.group_credit.enabled` in
   `input/task/resolved.json` as authoritative. When false or absent, only
   `compute_score` and `compute_fallback_score` are active runtime surfaces;
   do not add, modify, or rely on group assignment as the controlled
   intervention. An inherited dormant identity entrypoint has no training
   effect in that mode.
   Compute input validity and deterministic outcome first. When
   `judge_enrichment.enabled` is true and Group Credit is disabled, call the
   injected Judge exactly once on every valid path (including
   `outcome_score == 0.0`) and consume its fixed structured dimensions without
   overwriting them. When Group Credit is enabled, do not reference
   `llm_judge`; Engine owns acquisition. Return `score == outcome_score` and
   `artifact_projection is None`;
   when process evidence is used, `assign_group_credit` consumes the raw
   dimensions directly; whether and how they affect assignment follows the
   accepted Plan. It may not receive or reconstruct a scalar process projection. When Group
   Credit is disabled, the artifact may derive a separate
   `artifact_projection` or leave `score` unchanged. When enrichment is false,
   do not reference the capability.
5. Deliver exactly `reward.py` and `design.md`. Harness independently freezes
   the rubric/schema binding, submits any bounded validation job,
   performs
   static/ABI/sandbox checks, bounded fallback replay, compliance, provenance,
   and delivery finalization.

With enrichment enabled and Group Credit disabled, `reward.py` has one
permitted external capability:

```python
process_evidence = await llm_judge(
    question_prompt,
    response_content,
)
```

With Group Credit enabled this capability is not injected into the Artifact;
Engine performs the same one-call-per-rollout acquisition before row scoring.
The trusted Engine adapter batches calls in rounds across the full RL step into
Run-local Rubric Jobs and validates the task-owned structured result. Do not call
`review_labor`, another model or judge, or any provider API.
Harness owns the bounded validation job; never put a job submission in
`reward.py`. Do not start or configure a service. Do not import provider, MCP, HTTP, or
API clients. Builder must not submit the Local Judge job itself.
`compute_fallback_score` must not use `llm_judge`. Keep all scores
finite and bounded to `[0.0, 1.0]`. Only the trusted fixed-suite runner and ADE
Controller may import and execute `reward.py`.
`assign_group_credit` must not use `llm_judge`, perform I/O, inspect environment
or time, use randomness, or modify module state. Do not return an advantage;
return keyed scalar training rewards for the fixed GRPO estimator.

When `input/reflection/` is declared, read the exact prior delivery, direct
source groups, group/Judge results, manifest, and realization report. This is a
normal next round in the same Builder Call, not retry. Reuse persisted Judge
evidence and decide whether the realized artifact faithfully implements the
accepted Plan. Add exactly one `Reflection decision: finalize` or `Reflection
decision: revise` line to `design.md`. A missing reward, advantage, sign, or
downstream effect is an observation, not by itself a reason to revise. Do not
change the frozen source Trial, position, groups, runtime binding, or submitted
Judge evidence.

## Retry

When `input/retry.json` exists, start from `input/prior-delivery/`. Modify only
the named `feedback_paths`, preserve unaffected artifacts byte-for-byte,
deliver only `reward.py` and `design.md`; Harness regenerates mechanical facts.
