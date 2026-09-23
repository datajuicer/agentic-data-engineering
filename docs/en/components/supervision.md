# Operator-side supervision

[Components](README.md) | [Walkthrough](../walkthrough.md)

The supervising Agent is the glue between the operator and a durable ADE Run. It
receives an Operation Prompt selecting a config and authorized deployment, then follows
[`ade-supervise-run`](../../../.agents/skills/ade-supervise-run/SKILL.md).
Its [workflow](../../../.agents/skills/ade-supervise-run/references/workflow.md) owns the
preflight, monitoring, recovery and acceptance procedure.

## Inputs and boundaries

The prompt supplies repository/environment paths, exact experiment/deployment YAMLs,
expected Ray head, resource/service authorization, Run mode, and initial-state source.
For an existing Run it also supplies its actual ID. Templates are in
[examples/math-sft](../../../examples/README.md). Scientific values come from compiled
config, never from a second list of numbers in the prompt.

The outer Agent performs operational work. ADE's five research roles propose and
analyze strategies; Control alone commits accepted state. The Python Supervisor
manages component processes, while Run Monitor collects telemetry and tracking evidence.
These are separate responsibilities, even though both the Agent and the process manager
are sometimes called “Supervisor.”

## Execution and evidence

1. Compile and validate bindings without starting a Run; validate Seed references when present.
2. Within explicit authorization, inspect Ray capacity, shared inputs and service readiness.
3. Start one foreground supervisor, or reconcile and continue the existing Run.
4. Use the snapshot helper to append revision-bound observations to `<run-root>/monitor.md`, normally every ten minutes and on meaningful process/failure events.
5. Diagnose concrete failures, follow same-Run recovery or typed fork rules, and verify terminal artifacts and scoped cleanup before reporting completion.

Prompts are inputs, not scripts. No CLI parses the Markdown and starts a Run automatically.
The Agent uses the documented CLI and helpers. A bare Skill invocation does not authorize
GPU use, paid requests, service mutations or unrelated job cleanup.

## Standalone evaluation

Post-hoc checkpoint × dataset matrices use the separate
[`ade-supervise-generalization`](../../../.agents/skills/ade-supervise-generalization/SKILL.md)
and an `ade.generalization_operation` YAML request. They are not training/search Runs.
The compiler currently requires that operation's environment at
`<repository>/.unified-vllm-0.19.1-verl-venv`; inspect its workflow and
[evaluation guide](evaluation-analysis.md) before preparing a matrix.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [Run Skill](../../../.agents/skills/ade-supervise-run/SKILL.md) | Outer Agent entry and boundaries |
| [Snapshot helper](../../../.agents/skills/ade-supervise-run/scripts/append_monitor_snapshot.py) | One observed revision and durable monitor entry |
| [Admission helper](../../../.agents/skills/ade-supervise-run/scripts/inspect_ray_admission.py) | Read Ray groups, allocator leases and Judge actors |
| [Terminal reconciliation](../../../.agents/skills/ade-supervise-run/scripts/reconcile_terminal_run.py) | Reconcile persisted Run cleanup |
| [Preflight cleanup](../../../.agents/skills/ade-supervise-run/scripts/cleanup_terminal_transients.py) | Remove exact recorded Supervisor probe paths |
| [Python supervisor](../../../ade/harness/supervisor.py) | Process lifetime and worker policy |
| [Run Monitor](../../../ade/harness/run_monitor.py) | Telemetry and tracking reconciliation |
