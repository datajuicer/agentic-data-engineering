# Operator test and final generalization

[Input preparation](data-preparation.md) | [Results](results.md) | [Evaluation internals](components/evaluation-analysis.md)

ADE separates the validation feedback used to develop strategies from held-out evaluation. Operator test and final generalization both measure held-out performance, but have different execution scopes.

| Evaluation | When and what it evaluates | Dataset selection | How results are used |
| --- | --- | --- | --- |
| Online validation | During training, at configured checkpoint positions | Task `data.validation` | Select a checkpoint within the Trial |
| Offline validation | The selected checkpoint for each Trial | Task `data.validation` | Rank Trials/Plans and provide research feedback |
| Operator test | Automatically scheduled within the Run for Base and successful Trials, including the baseline, once their evidence/checkpoint is available | Task `data.test`, with the operator evaluation profile | Operator-facing results; excluded from the research selection objective |
| Final generalization | A separate evaluation operation after checkpoint/strategy selection | An explicit checkpoint × dataset matrix | Compare Base, baseline and selected ADE checkpoints across the complete held-out suite |

Operator test does not wait for all strategy search to finish. It evaluates each eligible Trial's selected checkpoint, rather than every training checkpoint. Its records belong to `run.json` under `operator_evaluations`; follow their `result_ref` values to the evaluation evidence.

Final generalization is **not automatically launched when a training Run finishes**. Select checkpoints using accepted in-loop validation results, freeze their identities, then submit the separate request below. Do not use operator-test or generalization scores to select a different checkpoint, rank research Plans or drive further strategy development.

The low-level evaluation runner also labels standalone units `operator_test`. That execution label does not make the two workflows identical: use the owning Run or standalone evaluation ID and request to identify the result.

## Dataset suites

These are the supplied four-task evaluation settings. `math_val` is the 500-problem MATH-S split: it is in-loop validation for RFT, and held-out evaluation for math SFT.

| Task | In-loop validation | Operator test | Final generalization: four held-out datasets |
| --- | --- | --- | --- |
| Math SFT | AIME24 | AIME25 | MATH-500, MATH-S, AIME25, HMMT |
| Code SFT | LiveCodeBench on/before 2024-03-30 | LiveCodeBench after 2024-03-30 | Later LiveCodeBench, HumanEval+, Codeforces, CodeElo |
| Reward Design | MATH-S | MATH-500 | MATH-500, AIME24, AIME25, HMMT |
| Curriculum Learning | MATH-S | MATH-500 | MATH-500, AIME24, AIME25, HMMT |

The operator-test dataset also appears in the corresponding final suite. This overlap is intentional: one is a Run-owned evaluation stream, the other is an explicitly frozen comparison matrix. The standalone operation creates its own evaluation units; it does not automatically reuse operator-test receipts.

Final evaluation generates 32 responses per problem on AIME/HMMT and 3 on the other datasets. Report Pass@K and Average@K, with an equal-weight mean over the four held-out benchmarks for each metric. SFT uses Pass@K as primary and Average@K as secondary; RFT reverses that order. Do not include the in-loop validation score in the held-out macro average or average across task families. Incomplete evaluations must remain visible as missing results.

The [benchmark catalog](../../configs/benchmarks/catalog.yaml) also contains optional math/code datasets and small check subsets. Catalog membership alone does not add a dataset to this final suite.

## Prepare the datasets

Use the [input guide](data-preparation.md) for the source downloads and MATH split. In addition to each task's validation/operator-test inputs:

```bash
# Math SFT: create math_val with the MATH split instructions, then prepare:
python scripts/build_benchmark_catalog.py math_500 aime25 hmmt
python scripts/build_benchmark_catalog.py math_val math_500 aime25 hmmt --check

# Code SFT:
python scripts/build_benchmark_catalog.py livecodebench_gt_2024_03_30 humaneval_plus codeforces codeelo
python scripts/build_benchmark_catalog.py livecodebench_gt_2024_03_30 humaneval_plus codeforces codeelo --check

# Reward Design / Curriculum Learning:
python scripts/build_benchmark_catalog.py math_500 aime24 aime25 hmmt
python scripts/build_benchmark_catalog.py math_500 aime24 aime25 hmmt --check
```

Choose only the commands for your task. Existing artifacts are checked and reused. Preparing `math_val` for math SFT uses the same split script as RFT; its generated training Parquet is not an SFT training input.

## Submit a final evaluation

1. Copy [generalization-operation.yaml](../../examples/generalization-operation.yaml) to a new file under `runs/operation-prompts/`.
2. Keep the suites and checkpoint variants you want to compare. Fill the repository environment path, local deployment, Ray head and explicit resource authorization. Use a distinct `evaluation_id` for a different matrix.
3. Fill each checkpoint's actual `subject_ref` and model/checkpoint directory from accepted Run evidence. Use `protocol_role: base` for the untrained model and `trained` for baseline/ADE checkpoints. To include several Runs, give each checkpoint a unique `id` while keeping its comparison `variant`, such as `n1` or `n3`.
4. Hand the completed request to an Agent using `$ade-supervise-generalization`. This is a separate operation from `$ade-supervise-run`.

The [contract profiles](../../configs/evaluations/generalization-contracts.yaml) bind each suite to its task, prompts and evaluation settings. The request supplies the checkpoint paths, dataset list and samples per input. Keep those settings fixed across compared checkpoints.

For local preparation and then authorized execution, the supervising workflow uses:

```bash
ade evaluate prepare runs/operation-prompts/generalization.yaml
ade evaluate supervise runs/operation-prompts/generalization.yaml
```

`prepare` resolves the request and writes local evaluation state without contacting Ray or submitting GPU work. `supervise` performs admission, execution, retries, monitoring and cleanup. Keep the same immutable request when resuming; do not replace checkpoints in an existing evaluation.

Evaluation state lives under `runs/deployments/<deployment>/evaluations/<evaluation-id>/`; result tables are written to `analysis/generalization/<evaluation-id>/`. Read `results.csv` for both metrics and unit identities, `comparison.csv` for variant comparisons, `family_summary.csv` for within-family summaries, and `failed_units.csv` for failures. A family summary is not the four-benchmark macro average. The generated `summary.md` presents the primary metric; retain both metrics from `results.csv` when reporting the full comparison.
