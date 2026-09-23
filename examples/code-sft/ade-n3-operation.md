# Code SFT N=3 ADE — Operation Prompt

## Run selection

```text
Skill: $ade-supervise-run
Repository: <ABSOLUTE_CHECKOUT_PATH>
Execution environment: <ABSOLUTE_CHECKOUT_PATH>/.unified-vllm-0.19.1-verl-venv
Experiment: configs/experiments/openthoughts-code-sft-data-selection-ADE-formal-n3.yaml
Deployment: configs/deployments/local.yaml
Master IP: <RAY_HEAD_IP>
Resource authorization: NOT_AUTHORIZED — replace with the Operator's explicit resource and service authorization
Initial State Reference: <COMPLETED_BASELINE_RUN_ID>
Initial State Frontier: bootstrap
Run mode: NEW
```

## Execution

Use `$ade-supervise-run` from this repository to supervise this one Run through
preflight, startup, durable monitoring, in-scope recovery and terminal acceptance.
Follow its `SKILL.md` and `references/workflow.md`. The Operation Prompt supplies
facts; the Skill supplies the procedure. Validate the supplied baseline with Seed
inspection before allocating resources. Do not substitute another source or config,
rerun imported baseline facts, or perform ADE's scientific roles. Missing values
or authorization block live work. Report accepted results, Memory and cleanup.

The source must be a completed Code SFT baseline from the matching family.
