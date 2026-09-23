# ADE documentation

[Project](../../README.md) | [Architecture](components/architecture.md)

## Run your first supervised experiment

1. [Installation](installation.md): one environment for development, tests and training.
2. [Inputs](data-preparation.md): model snapshots, training pools and benchmarks.
3. [Deployment](deployment.md) and [Ray setup](ray-cluster.md): explicit machine paths, capacity and services.
4. [Quickstart](quickstart.md): fill an Operation Prompt and hand it to the supervising Agent.
5. [Walkthrough](walkthrough.md): baseline acceptance, Seed import, N=1 ADE, monitoring and cleanup.
6. [Results](results.md): inspect evidence, scores, Trial Records and Memory.
7. [Final generalization](generalization.md): operator test boundaries, held-out suites and a separate evaluation request.

The [example prompts](../../examples/README.md) provide Baseline and ADE entries for each task family. Follow the same supervision workflow for each Run.

## Operate and understand ADE

- [CLI operations](operations.md), [service lifecycle](service-lifecycle.md), and [recovery](components/recovery.md).
- [Component guide](components/README.md), starting with [supervision](components/supervision.md) and [architecture](components/architecture.md).
- [Configuration matrix](../../configs/README.md) for supported task families and baseline dependencies.
- [FAQ](faq.md).

The [license](../../LICENSE) and [citation](../../CITATION.bib)
currently have explicit placeholders.
