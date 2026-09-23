---
name: ade-supervise-run
description: Supervise one ADE experiment Run from admission and startup through durable monitoring, recovery, and terminal acceptance. Use when an ADE Operation Prompt selects a new, resumed, forked, or already-running Run; do not use for implementing ADE code or performing Coordinator, Builder, Analyzer, or Summarizer work.
---

# ADE Supervise Run

Own one ADE Run as its operator-side supervising Agent.

Before running preflight, starting or resuming a Run, observing state, allocating
resources, or contacting an external service, read
[references/workflow.md](references/workflow.md) completely and follow it as the
sole lifecycle, monitoring, recovery, and terminal-acceptance authority.

## Inputs and authority

Take Run-specific facts from the selected Operation Prompt. It must identify the
repository, exact execution environment, exact experiment config, expected
deployment Master IP, resource authorization, Initial State Reference/Frontier
(or `NONE`/`NONE`), and Run mode. For an existing Run it must also identify the
Run ID. Apply the workflow's stop conditions when a required fact is absent or
conflicts with the compiled experiment.

Skill invocation is not resource authorization. Do not start a new Run, consume
GPUs, use an external Judge, or mutate a deployment unless the Operation Prompt
explicitly authorizes the resources resolved from its experiment and deployment.

## Execution boundary

- Treat the Operation Prompt as an input record, not a second procedure.
- Treat `ade experiment inspect` and the selected experiment as the resolved
  task, budget, deployment, Judge, tracking, and recovery authority.
- Keep one supervising Agent and one long-running objective through startup,
  monitoring, recovery, and terminal acceptance.
- Let ADE launch and coordinate its own task-role workers. Do not perform their
  scientific roles or issue standalone training, evaluation, or worker commands.
- For every scheduled or event-driven observation, use
  [scripts/append_monitor_snapshot.py](scripts/append_monitor_snapshot.py). It
  captures one `run observe`, binds the immutable revision, and appends the
  required Run-local `monitor.md` entry without streaming raw status to the
  terminal. Do not call `run observe` separately for the same snapshot.
- Finish only at the workflow's terminal acceptance boundary or a concrete
  Operator blocker.
- Before admitting work, verify Ray status from the compiled address. Before
  exiting a completed, failed, or cancelled Run, reconcile the Run cleanup
  receipt, stop its exact Local Judge gateway/vLLM launch, remove recorded
  Supervisor preflight directories, verify Ray again, and append the cleanup
  result to `monitor.md` as specified by the workflow.
