# Deployment configuration

Copy [`example.yaml`](example.yaml) to `local.yaml`, then configure your own host and paths. `local.yaml` is ignored by Git.

- [Deployment guide](../../docs/en/deployment.md)

The example uses one Ray cluster for training, evaluation and an eight-GPU Judge actor. Its address and filesystem paths are placeholders, not a runnable deployment. Deployment configuration supplies infrastructure; the selected experiment's runtime config supplies total cluster capacity (including Judge) and Coordinator training/evaluation allocations.

Judge defaults to `models/Qwen3.6-35B-A3B`. Ray chooses one node for its eight-GPU actor; configure only `gateway_port`, not a Judge host. All eligible nodes need the same accessible model and runtime paths. The head may advertise zero GPUs.
