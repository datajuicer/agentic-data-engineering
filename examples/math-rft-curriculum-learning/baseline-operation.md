# Math RFT Curriculum Learning baseline — Operation Prompt

## Run selection

```text
Skill: $ade-supervise-run
Repository: <ABSOLUTE_CHECKOUT_PATH>
Execution environment: <ABSOLUTE_CHECKOUT_PATH>/.unified-vllm-0.19.1-verl-venv
Experiment: configs/experiments/math-math-rft-curriculum-learning-baseline-formal.yaml
Deployment: configs/deployments/local.yaml
Master IP: <RAY_HEAD_IP>
Resource authorization: NOT_AUTHORIZED — replace with the Operator's explicit resource and service authorization
Initial State Reference: NONE
Initial State Frontier: NONE
Run mode: NEW
```

## Execution

Use `$ade-supervise-run` from this repository to supervise this one Run through
preflight, startup, durable monitoring, in-scope recovery and terminal acceptance.
Follow its `SKILL.md` and `references/workflow.md`. The Operation Prompt supplies
facts; the Skill supplies the procedure. Do not substitute another config or
perform ADE's scientific roles. Missing values or authorization block live work.
Report the actual Run ID, accepted baseline evidence and cleanup result so this
completed Run can be selected explicitly by a subsequent ADE operation.

Train the task-owned random-schedule P000 independently from the configured base
model, then complete Curriculum evaluation, analysis and Memory publication.
