# Agent runtime and the five roles

[Components](README.md)

This page describes ADE's internal research roles. The outer [supervising Agent](supervision.md) runs a different Skill and owns operational monitoring, not scientific decisions.

## Role contracts

| Role | Main inputs | Delivery |
| --- | --- | --- |
| Coordinator | Task, accepted directions, ranking and Memory | `decision.json`, `plan.md` |
| Artifact Builder | Accepted Plan, source artifacts, fixed task inputs, realization feedback | `selection.py`, `reward.py` or `curriculum.py`, plus `design.md` |
| Analyzer | Frozen experiment evidence; then Review results | Review design followed by analysis, findings and evidence/coverage files |
| Plan Summarizer | Accepted Trial conclusions and Plan Memory basis | `MEMORY.md` |
| Run Summarizer | Accepted Plan/Trial conclusions and Run Memory basis | `MEMORY.md` |

Exact required files and semantic validation belong to each task's `role_contracts.py` and the staged Analyzer contract. A free-form answer in an Agent transcript does not substitute for a delivery.

## Call implementation

`AgentCallService` binds a scoped action to its role contract and input package. `RoleContextPackager` materializes readable task data, source references and Memory. `SkillResolver` resolves an explicit repository Skill package with `SKILL.md` and `agents/openai.yaml`. `AgentRuntime` creates an attempt, invokes the backend, maintains heartbeat and applies the delivery gate and role validators. Accepted/rejected results return to Control for admission; the backend never commits RunState.

Sessions, logical calls and physical attempts have distinct identities. Retry creates attempt evidence and feedback; Builder reflection additionally evaluates a realized proposal before final acceptance. The formal backend is wired from experiment `agent.backend`, `model`, `reasoning_effort`, retry and heartbeat settings. The Python installer does not supply backend authentication.

## Usage

Use the supervised [run commands](../operations.md) for real calls. To inspect available worker options without invoking an Agent:

```bash
.unified-vllm-0.19.1-verl-venv/bin/ade agent worker --help
```

The [scripted example](../../../examples/scripted_run.py) implements `AgentBackend.invoke(skill, attempt, feedback_paths)` with deterministic file deliveries and demonstrates the same runtime without a paid backend. For a new proposal, satisfy the input package and the selected role's file contract; do not call a role Skill against an arbitrary empty directory.

## Outputs and diagnosis

Use `active_agent_calls` and session/attempt references in RunState to locate the workspace. Read input, output, validation feedback, heartbeat and receipt for that exact attempt. Missing files are delivery errors; stale heartbeat is an execution problem; an accepted delivery still needs Control acceptance. For Analyzer evidence failures inspect the frozen package and Review coverage rather than retrying with different scientific inputs. Workspace persistence is detailed in [Memory and Workspace](memory-workspace.md).

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/agent_runtime/service.py](../../../ade/agent_runtime/service.py) | Call binding and collection |
| [ade/agent_runtime/context.py](../../../ade/agent_runtime/context.py) | RoleContextPackager |
| [ade/agent_runtime/runtime.py](../../../ade/agent_runtime/runtime.py) | AgentRuntime and AgentBackend |
| [ade/agent_runtime/backend.py](../../../ade/agent_runtime/backend.py) | Production backend invocation |
| [ade/agent_runtime/delivery.py](../../../ade/agent_runtime/delivery.py) | Delivery gate |
| [ade/agent_runtime/skills.py](../../../ade/agent_runtime/skills.py) | Explicit Skill resolution |
| [ade/tasks/role_contract.py](../../../ade/tasks/role_contract.py) | Shared role and staged Analyzer contracts |
