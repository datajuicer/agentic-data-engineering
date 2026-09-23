# Data Selection task

[Components](README.md)

## Inputs and modifiable surface

The task fixes the training source, tokenizer/model, prompt protocol, pool size and requested selection size. `fixed_pool.py` builds the candidate inventory and training records from the configured OpenThoughts JSONL; it verifies the declared source-row count and applies the native prompt/token-length contract. The Builder changes `selection.py` and explains it in `design.md`; changing the model, evaluation split or training recipe is outside this proposal surface.

The required entry point is:

```text
async def select_trajectories(candidate_inventory, select_size, judge)
```

It returns the selected trajectory IDs. IDs must belong to the provided inventory and satisfy the requested selection contract. The source contract restricts executable structure and imports; use the inventory actually materialized in the Agent input rather than inventing IDs or loading a private dataset path.

## Implementation flow

The role contract checks the delivery; `compiler.py` creates the task artifact. `SelectionRealizer` executes the canonical selector against frozen inputs, with the configured Judge capability, and publishes selection evidence and final binding. Builder reflection reads realization feedback before finalization. The Engine binder creates the SFT dataset/input references; `SFTExecutor` and LlamaFactory train/evaluate them. Control admits evidence and sends it to the staged Analyzer.

Outputs include accepted selection code, realized selected IDs, selection/training summaries, prompt-contract evidence and evaluation results. The Trial Record retains the accepted artifact and `realization/final-realization.json`; packaged Engine evidence supplies the Analyzer view.

## Configure and use

The formal task uses a 3,840-row mixed pool and selects 384 rows. Math/code variants share the pool and differ in evaluation. Prepare inputs via [data preparation](../data-preparation.md), then inspect the real baseline configuration in the full environment:

```bash
ade experiment inspect configs/experiments/openthoughts-math-sft-data-selection-baseline-formal.yaml --deployment configs/deployments/local.yaml --run-id selection-inspect
```

The command reads inputs but starts no training. Use [operations](../operations.md) to launch baseline and search. For a fully local executable example of selector delivery and realization, use [CPU demo](../cpu-demo.md); its single synthetic row is not the formal training pool.

## Diagnose

A pool-size mismatch requires correcting the prepared source/config binding. Unknown or duplicate IDs and wrong selection count require fixing the proposal. A disabled-Judge error means the proposal called a capability unavailable in its task. Check realization feedback before investigating training. Training/metric failures after artifact acceptance belong to [Engine](engine.md) and [evaluation](evaluation-analysis.md).

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/tasks/data_selection/plugin.py](../../../ade/tasks/data_selection/plugin.py) | Task registration and hooks |
| [ade/tasks/data_selection/fixed_pool.py](../../../ade/tasks/data_selection/fixed_pool.py) | Candidate inventory and prompt-length binding |
| [ade/tasks/data_selection/role_contracts.py](../../../ade/tasks/data_selection/role_contracts.py) | Role delivery contract |
| [ade/tasks/data_selection/selection_contract.py](../../../ade/tasks/data_selection/selection_contract.py) | Restricted selector source |
| [ade/tasks/data_selection/selection_runtime.py](../../../ade/tasks/data_selection/selection_runtime.py) | Selector execution and result checks |
| [ade/tasks/data_selection/realizer.py](../../../ade/tasks/data_selection/realizer.py) | Realization and publication |
| [ade/tasks/data_selection/engine_binding.py](../../../ade/tasks/data_selection/engine_binding.py) | Artifact-to-SFT binding |
