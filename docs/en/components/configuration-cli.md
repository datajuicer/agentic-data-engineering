# Configuration, Harness and CLI

[Components](README.md)

## Configuration ownership

| Layer | Responsibility |
| --- | --- |
| Operation Prompt | Exact config/deployment, environment, authorization, Run mode and Seed source; passed to the supervising Agent |
| `configs/experiments/` | Root seed, task type and layer references, Agent/search/bootstrap/tracking settings |
| `configs/tasks/` | Fixed model/data, task interface and prompt protocol |
| `configs/train/`, `configs/eval/`, `configs/analysis/` | Training recipe, evaluation purpose/metric and evidence-review policy |
| `configs/runtime/` | Backend environment paths and worker/resource allocation |
| `configs/deployments/` | Machine, Ray and Local Judge service bindings |
| `.env` | Local service credentials/settings; not scientific configuration |

`ExperimentConfigCompiler.compile_file` loads the selected layers, resolves paths/protocols and produces `ResolvedExperimentConfig`, Control configuration and Engine inputs. Task hooks own vertical behavior. `ConfigCompiler` handles the lower-level Harness Run boundary; it is not a replacement for the formal experiment compiler. `Harness` creates/inspects Runs and exposes operator state requests. `processes.py` and `wiring.py` assemble runnable components.

The root seed is compiled into dependent settings. Do not add independent seeds to deployment YAML. `.env` is loaded for CLI use, with already-exported values taking precedence. YAML does not expand shell variables or `~`. Keep secrets in environment variables because resolved configuration is persisted.

## Usage and outputs

Use the full environment and prepared inputs for formal inspection:

```bash
ade experiment inspect configs/experiments/openthoughts-math-sft-data-selection-baseline-formal.yaml --deployment configs/deployments/local.yaml --run-id config-inspect
```

It prints the resolved configuration and runtime roots without starting services. Referenced tokenizer/data reads can still fail. `publication_files()` supplies the configuration files stored with a Run, while state holds the configuration reference used for resume checks.

The user-facing workflow is an [Operation Prompt handoff](../quickstart.md). Its supervising Agent uses `ade run start` / `ade run resume` as the lifecycle CLI. `ade create` creates state at the lower-level boundary; it does not launch a complete supervised experiment. `agent`, `engine`, `review` and `control` expose process-level tools for existing assembly, not independent alternatives to the formal run workflow. See [operations](../operations.md).

Global `--runs-root` precedes `status`, `inspect`, `pause`, `cancel` or `run history`. Supervised start/resume use deployment-derived roots or their dedicated overrides. Standalone evaluation uses its own root flags. Mixing these roots can make a valid Run look missing.

## Diagnose and modify

Unknown fields indicate a configuration-contract error. Missing tokenizer/data is an input-preparation error, not proof the compiler is broken. A resume configuration mismatch requires preserving the accepted scientific binding rather than editing persisted config files. For a new scientific experiment, select a new configuration/Run through the documented entry point. Machine path changes belong to deployment/runtime settings; use explicit accessible paths on participating nodes.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/harness/experiment_config.py](../../../ade/harness/experiment_config.py) | Layered experiment compiler |
| [ade/harness/config.py](../../../ade/harness/config.py) | Lower-level Run compiler |
| [ade/harness/service.py](../../../ade/harness/service.py) | Harness lifecycle API |
| [ade/harness/cli.py](../../../ade/harness/cli.py) | Actual command parser and dispatch |
| [ade/harness/processes.py](../../../ade/harness/processes.py) | Process entry points |
| [ade/harness/environment.py](../../../ade/harness/environment.py) | Environment loading |
| [ade/harness/runtime_roots.py](../../../ade/harness/runtime_roots.py) | Deployment root mapping |
| [ade/tasks/registry.py](../../../ade/tasks/registry.py) | Lazy task registry |

[Ray cluster setup](../ray-cluster.md) uses `ade cluster head`, `worker`, `status` and `stop --local`; `--dry-run` prints commands without starting services.
