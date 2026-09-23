# Semantic output contract

Write exactly `reward.py` and `design.md` under `output/`.

`reward.py` uses the fixed six-argument asynchronous `compute_score` and
synchronous `compute_fallback_score` ABI. Normal output follows the closed
`ade.reward_result.v2` contract: finite bounded `score`, Engine-authoritative
outcome evidence, and only fixed optional projection/rule fields. It does not
require `outcome/process/rule_based` scalar components, `component_weights`, or
a direct weighted sum, and it does not permit arbitrary metadata.

Both entrypoints return exactly these five fields (no omissions or additions):

```python
{
    "schema_version": "ade.reward_result.v2",
    "score": bounded_score,
    "outcome_score": authoritative_outcome_score,
    "artifact_projection": bounded_projection_or_none,
    "rule_evidence": {
        "status": "available" or "not_configured",
        "value": bounded_rule_value_or_none,
    },
}
```

`rule_evidence.status == "available"` requires a bounded numeric value;
`"not_configured"` requires `value is None`. The fallback uses
`score == outcome_score`, `artifact_projection is None`, and still returns the
same closed five-field object.

When Judge is enabled, raw `ade.process_evidence.v1` remains Engine-owned.
With Group Credit disabled, Artifact code may produce a distinct bounded
`artifact_projection`; it must not copy over, rename, or alter raw dimensions.
With Group Credit enabled, `score == outcome_score` and
`artifact_projection is None`, and neither row entrypoint references
`llm_judge`; Engine owns evidence acquisition. When Judge is disabled, no Judge
call or process declaration is present. `compute_fallback_score` returns the
authoritative outcome score only and is reserved for typed row unavailability
on the disabled Group Credit path.

When `input/task/resolved.json` enables `rft.group_credit`, `reward.py` also
defines exactly this synchronous pure entrypoint:

```python
def assign_group_credit(group_input):
    ...
```

Engine supplies `ade.group_credit_input.v3`: a complete sibling group whose
records carry opaque `record_id`, authoritative correctness/outcome,
`pre_group_reward`, and nested outcome/process/rule/response-length evidence.
The process object contains raw Engine-owned dimensions and never contains an
Artifact projection. Return `ade.group_credit_output.v2` with one
keyed bounded `training_reward` per record and a closed decision containing
`mode`, lowercase `reason_code`, fixed `evidence_sources`, and the exact fixed
`process_dimensions` actually used.

Both `evidence_sources` and `process_dimensions` are sorted, unique lists from
their fixed vocabularies. Assignments are keyed by `record_id`; do not rely on
list order as identity.

Identity and abstained assignments equal outcome; shaped output changes at
least one assignment. The function is deterministic, order/relabel invariant,
does no I/O or Judge call, and returns rewards rather than normalized
advantages. If any process row is unavailable or not configured, do not
declare process dimensions; the full policy may use outcome/rule/length,
preserve ties, or abstain. Engine validates evidence declarations and applies
the fixed GRPO estimator plus mixed-group sign guard. Admission exercises the
configured sibling count, including both one-correct and one-incorrect mixed
groups; every correct response must retain positive realized advantage and
every incorrect response must retain non-positive realized advantage.

Raw-dimension-bank sensitivity probes produce observations for Builder
reflection, not a hard admission requirement. The realization report's
`process_bank_sensitivity` observation describes fixed synthetic probes,
separately from direct-reference rollout replay. No observed change means only
that those probes did not exercise a declared process-dependent effect; it
does not establish that the policy ignores process evidence.

Interpret the observation against the accepted Plan. If the Plan requires a
process-dependent intervention, investigate whether the probes exercise its
intended conditions. For an accepted outcome-only ablation, unchanged rewards
may be intended. Explain unresolved findings and the reflection decision in
`design.md`; do not change the accepted hypothesis merely to trigger a probe.
An unresolved sensitivity observation alone does not reject an otherwise valid
artifact when the reflection limit is reached. All other ABI, evidence-integrity,
determinism, and reward-bound checks remain in force. Pre-compressing raw
dimensions into a scalar process projection remains prohibited.

Use this `design.md` outline as guidance:

1. Objective and Source Inheritance
2. Evidence Hypothesis and Projection/Group Use
3. Fixed Controls and Mutation Boundary
4. Development Probes and Observations
5. Fallbacks, Refutation Conditions, and Known Limitations

Do not write delivery, metadata, compliance, Judge batch, manifest,
provenance, digest, or self-check sidecars.

For a declared reflection round, include exactly one line in `design.md`:
`Reflection decision: finalize` or `Reflection decision: revise`. This controls
whether the same Builder Call requests another realization review; it does not
assert that an observed scientific effect is required.
