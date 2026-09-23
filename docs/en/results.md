# Read results and evidence

[README](../../README.md) | [Operations](operations.md)

Start with persisted state, then follow its references to the accepted evidence. Service logs help diagnose execution; they do not establish which result Control accepted.

See [operator test and final generalization](generalization.md) for evaluation purposes, dataset overlap and final checkpoint selection.

## Run, Plan and Trial

| Object | Meaning | What to inspect |
| --- | --- | --- |
| Run | One configured experiment and its search budget | `status`, `completion_kind`, revision, task binding, ranking, Memory heads |
| Coordinator | Owner of a share of search Plans | Allocated/effective Plan limits and control status |
| Plan | A proposed experimental direction | Design, relation to prior Plans, best Trial, Plan Memory |
| Trial | One candidate artifact's execution and analysis | Phase, outcome, archive status, artifact, Engine/Review attempts and accepted analysis |

Identifiers are scoped: `t001` can occur under several Plans. Use the full `run/cNNN/pNNN/tNNN` subject when comparing or citing evidence. Bootstrap baseline Trials are distinct from search Trials. Retrying an Engine attempt does not create a new scientific Trial.

## Locate the files

For formal runs, `<control>` is `runtime_roots.control` from `ade experiment inspect`. Let `<run>` be `<control>/<run-id>`:

| Path relative to `<run>` | Contents |
| --- | --- |
| `run.json` | Current persisted RunState and recovery authority |
| `state/revisions/rev-<n>/run.json` | State at a committed revision |
| `reports/timeline.csv` | Materialized Trial timeline; use `ade run history` for revision transitions |
| `coordinators/cNNN/plans/pNNN/trials/tNNN/record/` | Durable accepted Trial Record |
| `memory/plans/cNNN/pNNN/versions/PMnnn/MEMORY.md` | A Plan Memory version |
| `memory/run/versions/RMnnn/MEMORY.md` | A Run Memory version |
| `services/supervisor/` | Supervised process logs |
| `reports/worker-provenance.json` | Worker provenance for diagnosis and cleanup |
| `reports/cleanup/run-cleanup.json` | Cleanup receipt when that cleanup path ran |

Use `memory.run_head` and `memory.plan_heads` in `run.json` to choose current Memory versions. A Trial also records the Memory basis and result used for its summaries. Do not choose a version by modification time. External Engine objects, checkpoints and large artifacts can live outside `<run>` in deployment-scoped storage; preserve the referenced data when retaining a Run.

## Read one Trial

A successful scripted Data Selection Trial publishes these paths beneath `record/`; real failed or partial Trials can have fewer files:

| Path | Question it answers |
| --- | --- |
| `manifest.json` | Which files were accepted, and by which transition/source? |
| `artifact/accepted.bin` | What candidate artifact was accepted? Interpret it using its task/artifact kind |
| `realization/final-realization.json` | What did the Builder proposal actually realize? |
| `comparisons/planning-comparators.json` | Which references were bound for comparison? |
| `comparisons/objective-comparison.json` | What did the objective comparison conclude? |
| `engine/outcome-package.bin` | Which packaged Engine evidence was admitted? |
| `analysis/analysis.md`, `analysis/findings.md` | What does the Analyzer conclude and cite? |
| `analysis/review-plan.json`, `analysis/review-packet.json`, `analysis/review-coverage.json` | What review was requested, returned and covered? |

A `.bin` suffix does not specify a text format: inspect the artifact kind and use the corresponding reader. Experiment packages use `decode_experiment_package` in `ade/agent_runtime/experiment_package.py`. Follow cited evidence and coverage rather than treating a readable analysis paragraph as proof that all evidence was available.

## Understand scores

`Trial.offline_score` and `offline_secondary_score` are accepted objective values. `ranking` records the metric ID, direction, evaluation profile and entries; inspect entry `level` because Trial-level and Plan-level entries may both refer to the same candidate. `null` is missing/unavailable, not zero.

Read the selected `configs/eval/*.yaml` and resolved experiment before interpreting a number. For example, `sft-formal-validation.yaml` uses offline primary `macro_pass_at_configured_k` and secondary `macro_avg_at_configured_k`, with 32 samples per input; its online profile uses 8 samples. Other tasks and purposes can differ. Do not compare profiles as though they were interchangeable or call every primary score “accuracy”. Check dataset, decoding, sample count and checkpoint selection alongside the score. Operator evaluation is a separate purpose; do not substitute it for the ranking objective.

Keep three judgments separate:

- **Execution:** Trial `outcome: succeeded` and `archive_status: archived` mean the lifecycle succeeded and was archived.
- **Scientific result:** The accepted objective comparison and evidence determine whether the hypothesis improved its specified reference. A successful Trial can reject its hypothesis.
- **Operational closure:** Run completion and its cleanup receipt describe different things. Inspect receipt status/errors and unresolved execution before declaring resources released.

Plan/Run Memory summarizes accepted learning for later Agents. It is useful for understanding conclusions, while the Trial Record and referenced evaluation evidence substantiate them.

SFT checkpoint identity uses epochs, while W&B evaluation axes use actual training steps.
The retained `checkpoint_position.json` records both. Do not interpret an epoch number as a global step. If historical
evaluation streams are marked superseded, use the current projection and accepted Engine
evidence; a tracking repair does not indicate retraining.

## Inspect the CPU example

After following [CPU demo](cpu-demo.md), run from the repository root:

```bash
.unified-vllm-0.19.1-verl-venv/bin/ade --runs-root runs/quickstart-demo/runs status scripted-demo
.unified-vllm-0.19.1-verl-venv/bin/ade --runs-root runs/quickstart-demo/runs inspect scripted-demo
.unified-vllm-0.19.1-verl-venv/bin/ade --runs-root runs/quickstart-demo/runs run history scripted-demo
cat runs/quickstart-demo/summary.json
```

The example has its own root, not a formal deployment root. `summary.json` links the exact Trial Record and Memory files. The example uses scripted scores to illustrate the record format; read the reported values from your own `summary.json`.

## When a result is incomplete

Read the Run status and latest transition, then the affected Trial's phase, `failure_kind`, attempt references, retry state and active commands. Inspect corresponding supervisor logs and accepted failure/evidence references. A missing analysis file after an Engine failure is not equivalent to an empty successful analysis. For paused or suspended work, use the [operations guide](operations.md) after resolving the underlying cause. Never repair the reported result by editing persisted state or Memory manually.

Compiler-generated training and evaluation roots are under `runs/deployments/<deployment>/engine-work/<run-id>/`; retained Trial checkpoints and evidence use `engine-work/trial_artifacts/<run-id>/...`. Use the paths in the accepted result records. The Run Monitor reconciles evaluation tracking from the same deployment root.
