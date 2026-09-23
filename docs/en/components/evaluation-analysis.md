# Evaluation, analysis and result projection

[Components](README.md)

For the user workflow, see [operator test and final generalization](../generalization.md), including the four-task suite and request template.

## Evaluation purposes and implementation

Training-time online evaluation measures configured checkpoint positions. Offline evaluation measures the selected checkpoint under its configured profile. Operator evaluation is a separate Control-managed purpose. Standalone evaluation compiles a checkpoint × dataset matrix outside a search Run. These purposes can use different sample counts and cannot be compared without their resolved contracts.

`EngineEvaluationDispatcher` gives checkpoint evaluations their own command identity and receipts. The evaluation runner builds task examples, generates rollouts and calls registered scoring logic. `metrics.py` standardizes details and selects configured metrics; catalog/task modules own benchmark parsing and scoring. Keep prompt, decoding, sample count, checkpoint and dataset identity with each result. A missing value is not a zero score.

Analyzer operates after Engine evidence is frozen into an experiment package. It designs Review, then synthesizes analysis/findings with citations and coverage after Review returns. Control accepts objective comparison and analysis before summaries. Analyzer text does not replace the configured scorer. See [Review](review-judge.md) and [results](../results.md).

## Supervised standalone evaluation

Give one exact `ade.generalization_operation` YAML request to an Agent invoking
[`ade-supervise-generalization`](../../../.agents/skills/ade-supervise-generalization/SKILL.md).
Its [workflow](../../../.agents/skills/ade-supervise-generalization/references/workflow.md)
defines request fields, admission, monitoring, retry and cleanup. The request specifies
an existing deployment, repository-local full environment, expected head, explicit
resource authorization, frozen contract profiles, selected checkpoints and dataset/K list.
Use the [contract catalog](../../../configs/evaluations/generalization-contracts.yaml)
and the [request schema](../../../ade/harness/generalization_operation.py); never infer
checkpoint paths from a historical Run ID or pass an experiment YAML as this request.

The Agent saves the exact mapping under `runs/generalization-operation-requests/` and uses:

```bash
ade evaluate prepare runs/generalization-operation-requests/<evaluation-id>.yaml
ade evaluate supervise runs/generalization-operation-requests/<evaluation-id>.yaml
```

`prepare` materializes the local request and resolves units without contacting Ray or
submitting GPU work. `supervise` performs authorized live admission and owns the Engine
workers, retries, snapshots and result tables. `run_mode: AUTO` continues the same
immutable operation after interruption. The outer Agent retains the process handle and
reads durable `monitor.md` and supervisor state through terminal acceptance.

Low-level `evaluate create/run/retry` commands are component tools. Do not invoke them
or launch extra workers alongside the supervised operation.

## Outputs and diagnostics

Standalone state is `<evaluation-root>/<evaluation-id>/state.json`, with resolved configuration and per-unit attempts. `run` submits units; `status` collects receipts, so it can refresh persisted evaluation state. `retry --unit-id` prepares a new attempt for a retryable failed unit; a subsequent `run` submits it. The supervised repair retry path is explicit and budget-bound.

`ResultsReportProjector` materializes accepted Run result views under `reports/results.csv`. Generalization projection writes `results.csv`, `comparison.csv`, `family_summary.csv` and `failed_units.csv`; interpret failures/coverage alongside aggregates. Source identities and complete unit results matter more than a headline average. Checkpoint-unavailable errors require restoring actual model files; citation failures require fixing the analysis delivery against its existing package. Do not repair either by changing the evaluation contract.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/engine/evaluation_dispatcher.py](../../../ade/engine/evaluation_dispatcher.py) | Checkpoint evaluation commands |
| [ade/engine/eval/runner.py](../../../ade/engine/eval/runner.py) | Rollout evaluation |
| [ade/engine/eval/metrics.py](../../../ade/engine/eval/metrics.py) | Metric normalization and selection |
| [ade/controller/operator_evaluation.py](../../../ade/controller/operator_evaluation.py) | Operator evaluation lifecycle |
| [ade/agent_runtime/experiment_package.py](../../../ade/agent_runtime/experiment_package.py) | Frozen evidence packages |
| [ade/harness/evaluation.py](../../../ade/harness/evaluation.py) | Standalone matrix and unit lifecycle |
| [ade/harness/generalization_operation.py](../../../ade/harness/generalization_operation.py) | Supervised operation request schema |
| [ade/harness/generalization_results.py](../../../ade/harness/generalization_results.py) | Generalization tables |
| [ade/harness/results_report.py](../../../ade/harness/results_report.py) | Accepted Run result projection |
