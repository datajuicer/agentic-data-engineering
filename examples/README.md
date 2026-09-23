# Examples

The public experiment example uses **Operation Prompt + exact config + a supervising
Agent**. Each Run has one operator-side Agent following the repository Skill.

| Family | Baseline Operation Prompt | N=1 ADE Operation Prompt | N=3 ADE Operation Prompt | Exact configs |
| --- | --- | --- | --- | --- |
| Math SFT | [Baseline](math-sft/baseline-operation.md) | [ADE](math-sft/ade-operation.md) | [ADE](math-sft/ade-n3-operation.md) | [Baseline](../configs/experiments/openthoughts-math-sft-data-selection-baseline-formal.yaml) · [N=1](../configs/experiments/openthoughts-math-sft-data-selection-ADE-formal-n1.yaml) · [N=3](../configs/experiments/openthoughts-math-sft-data-selection-ADE-formal-n3.yaml) |
| Code SFT | [Baseline](code-sft/baseline-operation.md) | [ADE](code-sft/ade-operation.md) | [ADE](code-sft/ade-n3-operation.md) | [Baseline](../configs/experiments/openthoughts-code-sft-data-selection-baseline-formal.yaml) · [N=1](../configs/experiments/openthoughts-code-sft-data-selection-ADE-formal-n1.yaml) · [N=3](../configs/experiments/openthoughts-code-sft-data-selection-ADE-formal-n3.yaml) |
| Reward Design | [Baseline](math-rft-reward-design/baseline-operation.md) | [ADE](math-rft-reward-design/ade-operation.md) | [ADE](math-rft-reward-design/ade-n3-operation.md) | [Baseline](../configs/experiments/math-math-rft-reward-design-baseline-formal.yaml) · [N=1](../configs/experiments/math-math-rft-reward-design-ADE-formal-n1.yaml) · [N=3](../configs/experiments/math-math-rft-reward-design-ADE-formal-n3.yaml) |
| Curriculum Learning | [Baseline](math-rft-curriculum-learning/baseline-operation.md) | [ADE](math-rft-curriculum-learning/ade-operation.md) | [ADE](math-rft-curriculum-learning/ade-n3-operation.md) | [Baseline](../configs/experiments/math-math-rft-curriculum-learning-baseline-formal.yaml) · [N=1](../configs/experiments/math-math-rft-curriculum-learning-ADE-formal-n1.yaml) · [N=3](../configs/experiments/math-math-rft-curriculum-learning-ADE-formal-n3.yaml) |

Follow the [walkthrough](../docs/en/walkthrough.md).
Copy the selected prompts and fill in local paths, deployment and resource authorization. Baseline uses `NONE` for both initial-state fields. After it completes, ADE uses that family's baseline Run ID and frontier `bootstrap`. Model, dataset, budget and resource settings come from the linked configurations. N is the number of Coordinators within one ADE Run. Choose the N=1 or N=3 prompt and prepare the resources specified by its configuration; both import the matching baseline. See the [configuration matrix](../configs/README.md) for the complete set.

For an optional CPU-only illustration of state and Memory, see the
[scripted demo](../docs/en/cpu-demo.md), implemented
in [scripted_run.py](scripted_run.py). It uses synthetic responses and is not a real
experiment or a source baseline for the operations above.

Start the supervising Agent using the
[Quickstart](../docs/en/quickstart.md#3-start-the-supervising-agent-manually-codex-cli).

## Final generalization

Use [generalization-operation.yaml](generalization-operation.yaml) after selecting checkpoints by in-loop validation. Keep the desired task suites and fill accepted artifact references. Follow the [guide](../docs/en/generalization.md) for dataset preparation, the distinction from operator test and execution.

<!--
## Interactive research demo

Open the [code research demo](https://ade-code-research.ruomengd.chatgpt.site) to explore a recorded 15-Plan trajectory, strategy ancestry, knowledge transfer and real training curves. The [local version](code-research-demo/README.md) also works offline without installation.
-->
