# Configuration

Select an exact `configs/experiments/*.yaml` in an Operation Prompt and hand it to the supervising Agent. The Prompt also selects the deployment; the Skill passes it explicitly as `--deployment configs/deployments/local.yaml`. See the [example prompts](../examples/README.md) and [walkthrough](../docs/en/walkthrough.md).

- [Deployment guide](../docs/en/deployment.md)
- [Public deployment example](deployments/example.yaml)

Experiment files own scientific/search settings. Deployment files own machine and service bindings. Runtime files own GPU allocation and backend environment paths. Changes to deployment addresses do not require changing scientific recipes.

## Canonical experiment matrix

Every family has exactly three entries: baseline-only, ADE N=1, and ADE N=3.
Data Selection is split into math and code domains. Reward Design and
Curriculum Learning currently use the canonical math domain.

| Task | Domain | Baseline | N=1 | N=3 |
|--- |--- |--- |--- |--- |
| Data Selection (SFT) | math | `experiments/openthoughts-math-sft-data-selection-baseline-formal.yaml` | `experiments/openthoughts-math-sft-data-selection-ADE-formal-n1.yaml` | `experiments/openthoughts-math-sft-data-selection-ADE-formal-n3.yaml` |
| Data Selection (SFT) | code | `experiments/openthoughts-code-sft-data-selection-baseline-formal.yaml` | `experiments/openthoughts-code-sft-data-selection-ADE-formal-n1.yaml` | `experiments/openthoughts-code-sft-data-selection-ADE-formal-n3.yaml` |
| Reward Design (RFT) | math | `experiments/math-math-rft-reward-design-baseline-formal.yaml` | `experiments/math-math-rft-reward-design-ADE-formal-n1.yaml` | `experiments/math-math-rft-reward-design-ADE-formal-n3.yaml` |
| Curriculum Learning (RFT) | math | `experiments/math-math-rft-curriculum-learning-baseline-formal.yaml` | `experiments/math-math-rft-curriculum-learning-ADE-formal-n1.yaml` | `experiments/math-math-rft-curriculum-learning-ADE-formal-n3.yaml` |

The 12 experiment entry points define training and search settings. Resource allocation comes from the selected runtime and deployment; use your own accepted Run IDs for Seed imports.

Training experiments use the formal configurations above. For standalone evaluation,
use `evaluations/` and the [generalization guide](../docs/en/generalization.md).
The optional [CPU demo](../docs/en/cpu-demo.md) demonstrates the workflow with
scripted responses and synthetic scores.

Generated Runs, checkpoints, logs and staging artifacts belong outside this configuration tree. Public examples use placeholders; `.env` and `configs/deployments/local.yaml` hold local settings and are ignored by Git.
