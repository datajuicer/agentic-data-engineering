# Walkthrough: Baseline to ADE

[Quickstart](quickstart.md) | [Example prompts](../../examples/README.md) | [CLI reference](operations.md)

This example follows ADE's operating model from prepared inputs to an accepted result.
There are two sequential Runs: a baseline, then an N=1 ADE importing that baseline.
Each has its own Operation Prompt and one supervising Agent using
[`ade-supervise-run`](../../.agents/skills/ade-supervise-run/SKILL.md).

## What each layer owns

| Layer | Responsibility |
| --- | --- |
| Operator | Select the config, provide local paths and deployment, authorize resource/service use |
| Operation Prompt | Record those facts, Run mode and any initial-state source |
| Supervising Agent + Skill | Preflight, start/resume, durable monitoring, operational recovery and terminal acceptance |
| ADE process supervisor | Keep Control, Monitor and workers running according to their policies |
| ADE research roles | Propose, implement, analyze and summarize strategies within the frozen task contract |
| Control and Engine | Accept transitions and execute the configured training/evaluation |

The supervising Agent manages execution outside the research loop. It must not
write a selection strategy, choose scientific winners or change a training budget to
make a failing experiment pass. It also does not replace ADE's internal monitoring process.

## Prepare the fixed configuration

Choose the matching Baseline and ADE configurations from the [example prompts](../../examples/README.md). The baseline establishes the reference; ADE imports its accepted Bootstrap frontier and runs the configured strategy search. Coordinator, Plan and Trial counts come from the selected experiment YAML.

Before starting:

- Install the [runtime](installation.md) on the participating nodes.
- Prepare the [models, training data and benchmarks](data-preparation.md) referenced by the selected task and deployment.
- Configure the [deployment](deployment.md) and [Ray cluster](ray-cluster.md) to meet the selected runtime's resource requirements.
- Fill the [environment variables](deployment.md#environment-variables) and configure Agent authentication.

The commands below use the math SFT prompt paths as an example. Use the corresponding paths for your selected task family.

## Fill and hand off the baseline prompt

Create local copies without modifying the published templates:

```bash
mkdir -p runs/operation-prompts
# Use new filenames if local copies already exist.
cp examples/math-sft/baseline-operation.md runs/operation-prompts/baseline.md
cp examples/math-sft/ade-operation.md runs/operation-prompts/ade.md
```

Fill each input deliberately:

| Field | Value |
| --- | --- |
| `Repository` | Absolute path to this checkout, accessible to the supervising Agent |
| `Execution environment` | Absolute full-runtime directory; its ADE import must resolve to this checkout |
| `Experiment` | Keep the template's exact formal YAML path |
| `Deployment` | Your local YAML, normally `configs/deployments/local.yaml` |
| `Master IP` | Actual Ray head IP, matching the deployment address; it does not pin Judge placement |
| `Resource authorization` | Explicit authorization for the selected Run's GPU and service scope |
| `Initial State Reference / Frontier` | `NONE / NONE` for baseline; actual completed baseline ID / `bootstrap` for ADE |
| `Run mode` | `NEW`; for continuation use the existing Run identity as described below |

A resource statement, **only after the operator has assigned these resources**, can be:

```text
Resource authorization: AUTHORIZED — use the assigned nodes and GPU allocation recorded in this Operation Prompt for this Run; use the configured Agent/Review services and W&B account; perform scoped preflight (including disposable W&B probes), startup, monitoring, routine recovery and terminal cleanup. Do not restart Ray or stop unrelated work.
```

The template's `NOT_AUTHORIZED` must be replaced by the operator's real grant, not by
the supervising Agent. Model names, Plan counts and evaluation budgets remain in YAML.
The Skill reads its complete workflow before live actions; the prompt does not duplicate
that procedure. Missing paths, capacity or authorization are reported as concrete blockers.

In a persistent Agent session opened at the selected checkout, submit the completed
baseline prompt and explicitly invoke `$ade-supervise-run`. The Skill is included in
`.agents/skills/ade-supervise-run/`; if needed, give its `SKILL.md` path directly to the
Agent host. Do not start a second Agent or manual `ade run start` for the same Run.

For exact human-operated terminal commands, see [manual Agent startup](quickstart.md#3-start-the-supervising-agent-manually-codex-cli): enter the head checkout, create a tmux session, activate the full environment, and launch Codex with the completed prompt. The same section covers detaching and reconnecting.

## What the supervising Agent does

The Skill compiles the exact config, generates one Run ID from the experiment identity,
UTC timestamp and unique suffix, checks deployment and input bindings, then performs live
resource/service admission within the granted scope. It starts the foreground ADE
supervisor and keeps its process handle. These live checks can consume GPU capacity and
service calls; they are not part of static preparation.

The Agent writes an initial snapshot and normally one snapshot every ten minutes to
`<run-root>/monitor.md` using the Skill helper. It wakes for process exits or faults,
performs bounded diagnosis/recovery, and returns to normal waiting when healthy.
A long training or Judge queue wait is not by itself failure. Accepted state and receipts
control recovery; the Agent does not edit `run.json` or resubmit work to obtain a better score.

## Accept the baseline, then start ADE

The baseline Agent reports its actual Run ID and terminal evidence. Check its accepted
Base/P000 results, Trial Record, Memory and cleanup; keep the Run and its referenced
artifacts under deployment-scoped storage. The CPU demo's synthetic baseline is not eligible.

Put that actual ID into the ADE prompt's `Initial State Reference`, keep frontier
`bootstrap`, and hand the completed prompt to the Agent responsible for the ADE Run.
The Skill runs `ade run seed inspect` before resource admission. Missing or incompatible
source artifacts block ADE; it must not silently retrain the imported baseline.
ADE receives its own Run ID, monitor record and W&B group.

## Follow progress and recover

The final and intermediate reports identify `runtime_roots.control`, the Run ID and
`monitor.md`. Inspect `reports/timeline.csv`, `reports/results.csv`, Trial Records,
Memory, service logs and cleanup receipts through the [results guide](results.md).
Commands for optional human observation are in [operations](operations.md); the supervising
Agent uses its snapshot helper rather than a second independent polling loop.

If the Agent session disconnects, keep the same operation and actual Run ID. Start a
replacement supervising session with `Run mode: EXISTING` and `Run ID: <actual-id>`.
The Skill first reconciles the existing process handle and persisted state; it observes
a live supervisor instead of launching a duplicate. For a confirmed paused/suspended Run,
use `Run mode: RESUME` with that ID; it diagnoses and resumes the same persisted config.
A cancelled Run needs an evidence-backed fork, not resume; see [recovery](components/recovery.md).

## Completion

The final report includes Run ID, revision/outcome, config/source identity, accepted
artifacts, Trial closure, Memory, W&B identity, usage completeness and resource cleanup.
Completion requires the Skill's terminal acceptance, including stopped Judge resources
and `reports/cleanup/run-cleanup.json`; a terminal status or a good metric alone is insufficient.
