# Optional CPU demo: one scripted Trial

[Installation](installation.md) | [Real inputs](data-preparation.md) | [Operations](operations.md) | [Results](results.md)

After [installing the unified environment](installation.md), run this from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .unified-vllm-0.19.1-verl-venv/bin/python -m examples.scripted_run --output runs/quickstart-demo
```

Use a new output directory on each invocation. The command preserves earlier results and rejects an existing target. Installation downloads Python packages; the example itself uses a locally generated tokenizer and one synthetic conversation. It needs no model download, credentials, paid Agent service, Ray cluster or GPU.

## What runs

The example starts from a synthetic accepted baseline and runs one Data Selection search Trial. It exercises the production workflow, role context packaging and delivery validation, selection compilation and realization, file queues, Engine result handling, staged analysis, Trial Record archival, and Plan/Run Memory publication.

Agent decisions, training/evaluation responses and Review responses are scripted. The baseline is constructed locally; the formal bootstrap training path is not executed. The single-row selection and fixed scores illustrate control flow, not an experimental improvement. No weights are trained. This example does not validate real Agent behavior, GPU backends, Judge deployment, or scientific reproducibility.

## Inspect the result

The command prints each committed transition and a final JSON summary. Success requires a completed Run, a succeeded and archived search Trial, a Trial Record, and published Memory. Start with:

```bash
cat runs/quickstart-demo/summary.json
cat runs/quickstart-demo/transitions.csv
```

`summary.json` contains absolute paths to the final revision's `run.json`, the Trial Record directory, and published `MEMORY.md` files. It also marks the result as `synthetic: true` and reports the scripted role-call counts. Read these files to follow how evidence becomes durable state and memory. All generated files stay beneath the requested output directory, which is ignored by Git when it is inside `runs/`.

The implementation is [examples/scripted_run.py](../../examples/scripted_run.py). The unified environment already includes all demo dependencies. GPU visibility is disabled for this command; no separate environment is needed. This is a source-checkout example, so run it from the repository root.

## Move to a real experiment

Use the same [unified environment](installation.md), prepare [models, training data and benchmarks](data-preparation.md), and configure the [public deployment example](deployment.md).
