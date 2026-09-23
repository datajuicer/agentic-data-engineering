# Engine, training backends and checkpoints

[Components](README.md)

## Command boundary

Control creates `TrainSFTCommand`, `TrainRFTCommand` or `EvaluateCommand` with scope, logical command ID, attempt ID, input reference and output URI. Task binders place compiled artifacts and resolved inputs in `FileEngineObjectStore`. `FileCommandQueue` persists work; `EngineWorker.run_once` claims a command, maintains heartbeat/progress and dispatches by command kind. It returns an EngineReceipt with output references or failure classification, not a RunState mutation.

## Backend execution

`SFTExecutor` uses the LlamaFactory adapter for training and evaluation. `RFTExecutor` uses the VERL adapter for both Reward Design and Curriculum tasks; the task binder determines reward or schedule inputs. The adapters launch against the included backend sources and configured environments. Ray execution and Coordinator GPU allocation are implemented under `ade/engine/execution/`.

Training/evaluation handlers preserve attempt-local state and checkpoint/evaluation references. RFT requires results for requested checkpoint steps, performs online validation, selects under the configured policy and evaluates the selected checkpoint offline. Do not infer a checkpoint merely from a directory's newest timestamp. Publication through `TrialArtifactPublisher` makes Engine evidence available to analysis. Checkpoint staging and cleanup have separate modules; do not delete a retained checkpoint while accepted references still require it.


Backend variants are grouped under `ade/engine/backends/`: `llamafactory.py` implements
SFT and `verl.py` implements RFT. Their vendored sources live under
`third_party/llamafactory/` and `third_party/verl/`. The existing
`runtime.backends.sft` / `runtime.backends.rft` entries bind backend name, source
and environment; SFT and RFT retain their distinct command and checkpoint contracts.
Task modules supply scientific inputs, while these adapters execute the backend.

## Configuration and usage

`configs/runtime/*.yaml` owns worker resources and backend environment paths; `configs/train/*.yaml` owns recipes; `configs/eval/*.yaml` owns evaluation policy. All must resolve on the nodes that execute them. Read [installation](../installation.md) and [deployment](../deployment.md) before launching the supervised workflow.

```bash
.unified-vllm-0.19.1-verl-venv/bin/ade engine worker --help
```

This only prints the worker interface. Actually starting a standalone worker can claim queued GPU work; normally let `ade run start` assemble it. The [CPU example](../cpu-demo.md) uses a fake training backend with the real worker, queue and SFT result path. It does not test LlamaFactory, VERL or Ray execution.

## Inspect and diagnose

Trace Trial `logical_command_id` and `engine_attempt_id` to its receipt, object-store result and published artifact manifest. Large artifacts live in deployment Engine storage, not only in `run.json`. Separate queue admission/claim failures, lost heartbeat/progress, backend subprocess errors and evaluation errors. A successful training subprocess with failed offline evaluation is not a complete accepted Trial. Automatic retries preserve Trial identity; see [recovery](recovery.md).

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/core/engine.py](../../../ade/core/engine.py) | Command, receipt and attempt types |
| [ade/engine/command_queue.py](../../../ade/engine/command_queue.py) | Durable command transport |
| [ade/engine/worker.py](../../../ade/engine/worker.py) | Claim and execute |
| [ade/engine/storage/object_store.py](../../../ade/engine/storage/object_store.py) | Referenced inputs/results |
| [ade/engine/backends/llamafactory.py](../../../ade/engine/backends/llamafactory.py) | SFT adapter |
| [ade/engine/backends/verl.py](../../../ade/engine/backends/verl.py) | RFT adapter |
| [ade/tasks/reward_design/handler.py](../../../ade/tasks/reward_design/handler.py) | RFT checkpoint and evaluation lifecycle |
| [ade/engine/trial_artifacts.py](../../../ade/engine/trial_artifacts.py) | Evidence publication |
| [ade/engine/execution/ray.py](../../../ade/engine/execution/ray.py) | Ray execution |
| [ade/engine/checkpoints/cleanup.py](../../../ade/engine/checkpoints/cleanup.py) | Checkpoint cleanup |
