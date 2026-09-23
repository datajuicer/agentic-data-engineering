---
name: ade-supervise-generalization
description: Supervise one standalone ADE generalization evaluation from an Operation Prompt through Ray admission, checkpoint-by-dataset execution, recovery, monitoring, aggregation, and terminal cleanup. Use for post-hoc evaluation matrices; do not use for an ADE training/search Run or for changing benchmark, prompt, checkpoint, or scoring contracts.
---

# ADE Supervise Generalization

Own one standalone generalization evaluation as its Operator-side supervising
Agent.

Before materializing an Operation Request, contacting Ray, starting an Engine
worker, submitting a unit, monitoring, retrying, or accepting terminal output,
read [references/workflow.md](references/workflow.md) completely and follow it
as the lifecycle authority.

## Inputs and authority

The selected Operation Prompt must contain one exact
`ade.generalization_operation` YAML block. It identifies the repository,
repository-owned execution environment, deployment YAML and expected Ray master,
resource authorization, evaluation identity, execution policy, frozen contract
profiles, checkpoint list, and dataset/K list.

Copy that YAML block verbatim into the durable request path prescribed by the
workflow. Do not infer omitted values, select another deployment, replace a
checkpoint, change K, or manufacture resource authorization. Skill invocation
alone is not GPU authorization.

## Execution boundary

- This skill owns standalone evaluation only. `$ade-supervise-run` remains the
  sole skill for formal ADE Runs.
- Use `ade evaluate prepare` and `ade evaluate supervise` as the deterministic
  control plane. Do not issue separate `evaluate run/retry`, Engine worker, or
  Ray mutation commands while the supervisor owns the request.
- Keep one Agent objective and the live supervisor process handle through
  admission, execution, aggregation, cleanup, and terminal acceptance.
- Read durable `supervisor/state.json`, snapshots, `monitor.md`, and generated
  tables for observations; logs are diagnostic evidence, not success authority.
- A request with `run_mode: AUTO` is restart-safe. Reinvoke the same immutable
  request after an Agent/process interruption; never create a second evaluation
  ID for the same matrix.
- Finish only at the workflow's terminal acceptance boundary or a concrete
  blocker recorded by preflight/supervisor state.
