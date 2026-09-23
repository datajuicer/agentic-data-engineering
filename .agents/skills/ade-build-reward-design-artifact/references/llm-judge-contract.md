# LLM Judge contract

## Respect the acquisition owner

`input/task/resolved.json#judge_enrichment.enabled` is the only Judge switch.
When false, `reward.py` must not declare or call Judge. When true with Group
Credit disabled, every valid rollout path, including `outcome_score == 0.0`,
awaits exactly one call:

```python
process_evidence = await llm_judge(question_prompt, response_content)
```

When Group Credit is enabled, Engine performs that one acquisition per valid
rollout. `reward.py` must not declare, reference, or call `llm_judge` in either
row entrypoint; `compute_score` returns outcome-only and
`assign_group_credit` receives the resulting raw bank in its v3 input.

Pass the original inputs without credentials, endpoints, provider names, or
transport instructions. The trusted Engine adapter—not the Artifact—owns the
rubric and output schema. For math Reward Design an available result has this
closed shape:

```python
{
    "status": "available",
    "adapter_id": "math_process_evidence.v1",
    "schema_version": "ade.process_evidence.v1",
    "dimensions": {
        "derivation_soundness": 0.0,
        "conclusion_support": 0.5,
        "substantive_relevance": 0.0,
    },
}
```

The adapter validates that every required dimension is finite, bounded, and
present. With Group Credit disabled, the Builder may select these fixed
dimensions and derive a separate `artifact_projection`. With Group Credit
enabled, Engine returns the raw dimension bank directly to
`assign_group_credit`; no projection is passed into the group
input. It must not mutate raw dimensions, add evidence names or metadata, or
interpret evidence acquisition as requiring a scalar `process` reward.

## Respect availability and failure ownership

`not_configured` exists only when Judge enrichment is disabled and no call is
made. A row-level terminal Judge error is `unavailable`, never an all-zero
dimension object. Artifact code must not catch, retry, or convert that signal:

- with Group Credit disabled, Engine invokes the deterministic outcome-only
  row fallback;
- with Group Credit enabled, Engine passes that availability with the complete
  sibling group to `assign_group_credit`, which may use outcome/rule/length,
  preserve a tie, or abstain;
- systemic job/gateway failure terminalizes the physical Engine Attempt as
  retryable;
- Artifact exceptions or invalid result/schema values fail the Attempt and are
  never converted to row fallback.

The boundary is strict: Builder does not submit validation jobs, call Review
Labor, access a provider or endpoint, start a service, or import MCP/HTTP/API
clients. Harness freezes the task adapter/rubric/schema binding and submits the
bounded validation job. Training uses the same task adapter and Engine-owned
response identity.
