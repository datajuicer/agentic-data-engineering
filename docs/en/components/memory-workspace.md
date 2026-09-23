# Memory, repository and Agent workspace

[Components](README.md)

## Accepted knowledge versus working files

`FileRunRepository` stores current RunState, revision snapshots and accepted artifacts. `TrialRecordStore` admits natural files into stable, append-only slots; accepted content cannot be overwritten. `MemoryVersionStore` publishes complete immutable Plan Memory (PM) and Run Memory (RM) directories. These are Control-owned publications, not free-form files that an Agent edits in place.

`WorkspaceManager` prepares an input/output area for each Agent Call attempt and records attempts, heartbeat, validation and receipts. Context packaging chooses the accepted task/source/Memory files visible to that role. Agent output is a proposed delivery. Only successful validation and a Control transition can make it an accepted record or Memory version.

## Publication flow and locations

After accepting Trial analysis, Control requests Plan and Run summaries against explicit Memory bases. Accepted `MEMORY.md` deliveries create new versions; Trial state records basis/result and RunState updates current heads. `memory.run_head` and `memory.plan_heads` select current versions; Trial fields retain the versions used for that Trial. Read references, not directory timestamps.

Relative to a Run directory:

| Location | Meaning |
| --- | --- |
| `run.json` | Current recovery state |
| `state/revisions/rev-N/run.json` | Committed historical state |
| `coordinators/cNNN/plans/pNNN/trials/tNNN/record/` | Accepted Trial files |
| `memory/plans/cNNN/pNNN/versions/PMnnn/` | Plan Memory publication |
| `memory/run/versions/RMnnn/` | Run Memory publication |

Agent call/session paths are derived by `RunLayout` and `WorkspaceManager` from the complete role/scope identity. Engine object URIs are a separate storage boundary: retaining only the Run directory can omit referenced large evidence or checkpoints.

## Read-only usage

After the [CPU demo](../cpu-demo.md), from the repository root with its demo environment:

```python
from ade.memory.repository import FileRunRepository
repo = FileRunRepository("runs/quickstart-demo/runs")
state = repo.load("scripted-demo")
print(state.revision, state.status.value)
print(state.memory.run_head, state.memory.plan_heads)
```

This loads accepted state without advancing workflow. Follow [results](../results.md) for exact Record/Memory files. There is no separate user-facing Memory daemon to start; publication is part of the workflow.

## Diagnose

If a summary is absent, check whether the summary delivery was accepted and whether the Trial is archived. If an Agent appears to use old knowledge, inspect its input package and recorded Memory basis before looking at current heads. Missing immutable source files require restoring the referenced evidence or using the defined recovery boundary; do not rewrite accepted Memory or reconstruct a score from informal notes.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/memory/repository.py](../../../ade/memory/repository.py) | State and artifact persistence |
| [ade/memory/records.py](../../../ade/memory/records.py) | Immutable Records and Memory versions |
| [ade/memory/snapshots.py](../../../ade/memory/snapshots.py) | Snapshot materialization |
| [ade/memory/layout.py](../../../ade/memory/layout.py) | Canonical path derivation |
| [ade/agent_runtime/workspace.py](../../../ade/agent_runtime/workspace.py) | Attempt and session workspaces |
| [ade/agent_runtime/context.py](../../../ade/agent_runtime/context.py) | Role-visible context selection |
